/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#define CUPHY_DEBUG 1

#include "ldpc.hpp"
#include "ldpc_load_store.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
template <uint32_t num_tdbv>
inline __device__ uint32_t funnel_shifted_bits(const uint32_t       shift,
                                               const uint32_t       tdbv_id,
                                               const uint32_t*      in,
                                               const uint32_t       remainder)
{
    bool      compense  = (tdbv_id * 32 + shift) >= (num_tdbv * 32);
    const uint32_t new_shift = shift + compense * (32 - remainder);
    uint32_t       lid       = (tdbv_id + new_shift / 32) % num_tdbv;
    uint32_t       hid       = (lid + 1) % num_tdbv;
    uint32_t remaining_shift = new_shift % 32;
    if(hid % num_tdbv == 0)
    {
        return __funnelshift_r(in[lid], ((in[hid] >> (32 - remainder))) | (in[hid + 1] << (remainder)), remaining_shift);
    }
    else
    {
        return __funnelshift_r(in[lid], in[hid], remaining_shift);
    }
}

template <uint32_t num_tdbv>
inline __device__ uint32_t funnel_shifted_bits(const uint32_t  shift,
                                               const uint32_t  tdbv_id,
                                               const uint32_t* in)
{
    uint32_t lid = (tdbv_id + shift / 32) % num_tdbv;
    uint32_t hid = (lid + 1) % num_tdbv;
    uint32_t remaining_shift = shift % 32;
    return __funnelshift_r(in[lid], in[hid], remaining_shift);
}

#if 0
// Commenting out the following specialization as recent measurements
// did not show perf. benefits.
template <>
inline __device__ uint32_t funnel_shifted_bits<12>(const uint32_t  shift,
                                                   const uint32_t  tdbv_id,
                                                   const uint32_t* in)
{
    // For the special case of num_tdbv == 12, which corresponds only to Z=384,
    // lid prior to the mod is at most 2*(num_tdbv-1). This is because tdbv_id is
    // at most num_tdbv-1 and the maximum shift value is at most 383, so
    // shift / 32 is at most num_tdbv-1. Thus, in place of the mod from the generic
    // implementation, we need at most a single subtraction to shift lid into
    // the range [0, num_tdbv).
    uint32_t lid = (tdbv_id + shift / 32);
    if (lid >= 12) {
        lid -= 12;
    }
    // Alternate implementation to (lid + 1) % 12. It is known from above that
    // lid is in the range [0, 11], so lid == 11 will wrap to hid = 0 and all
    // other values of lid will map to hid = lid + 1.
    const uint32_t hid = (lid == 11) ? 0 : lid+1;
    uint32_t remaining_shift = shift % 32;
    return __funnelshift_r(in[lid], in[hid], remaining_shift);
}
#endif

inline __device__ uint32_t circular_shifted_bits(const uint32_t  shift,
                                                 const uint32_t  tdbv_id,
                                                 const uint32_t* in,
                                                 const uint32_t  Z)
{
    uint32_t id   = tdbv_id;
    uint32_t bits = in[id];

    uint32_t l_mask = (1 << (Z - shift)) - 1;
    uint32_t h_mask = ((1 << Z) - 1) ^ l_mask;

    uint32_t lsb = (bits >> shift) & l_mask;
    uint32_t msb = bits << (Z - shift) & h_mask;
    return msb | lsb;
}

template <typename DType, uint32_t Z>
inline __device__ DType rotated_bits(const uint32_t       shift,
                                     const uint32_t       tdbv_id,
                                     const uint32_t       in_offset,
                                     const uint32_t* in)
{
    constexpr int    remainder = (Z >= 32) ? static_cast<int>(Z % 32) : -static_cast<int>(Z);
    constexpr size_t elem_size = sizeof(DType) * 8;
    constexpr int    num_tdbv = (Z / elem_size + int(Z % elem_size != 0));

    in += in_offset;

    if constexpr(remainder > 0)
        return funnel_shifted_bits<num_tdbv>(shift, tdbv_id, in, remainder);
    else if constexpr(remainder == 0)
        return funnel_shifted_bits<num_tdbv>(shift, tdbv_id, in);
    else {
        // remainder < 0, so z_circ > 0
        const uint32_t z_circ = static_cast<uint32_t>(-remainder);
        return circular_shifted_bits(shift, tdbv_id, in, z_circ);
    }
}

template <typename DType, int row_base, bool accumulated, uint32_t Z, typename Block>
inline __device__ void block_multiply_in_bit(const int       BG,
                                             const uint32_t  col_base,
                                             const uint32_t  col_end,
                                             const uint32_t* in,
                                             uint32_t*       out,
                                             Block&          block,
                                             const uint32_t  num_rows)
{
    constexpr uint32_t tile_size = (Z % 32 == 0 && Z > 256) ? 16 : 8;
    constexpr size_t   elem_size = sizeof(DType) * 8;
    constexpr uint32_t num_tdbv = (Z / elem_size + int(Z % elem_size != 0));

    cg::thread_block_tile<tile_size> tile      = cg::tiled_partition<tile_size>(block);
    const uint32_t                        lane_id   = tile.thread_rank();
    const uint32_t                        tile_id   = block.thread_rank() / tile_size;
    const uint32_t                        num_tiles = block.size() / tile_size;

    for(uint32_t row = tile_id + row_base; row < num_rows; row += num_tiles)
    {
        // Each warp is in charge of different Z-sized data
        bg_CN_row_shift_info_t CNShift(row, Z, BG);

        for(uint32_t tdbv_id = lane_id; tdbv_id < num_tdbv; tdbv_id += tile_size)
        {
            const uint32_t deg = CNShift.row_degree;
            const uint32_t deg_max_even = 2 * (deg / 2);
            uint32_t val[2] = { 0, 0 };
            // Manually unroll by a factor of 2. Higher unrolling factors
            // increase the register count enough to reduce occupancy and
            // may not help helpful for rows with smal row_degree.
            for (uint32_t cid = 0; cid < deg_max_even; cid += 2) {
                const uint32_t col0 = CNShift.column_values[cid+0];
                const uint32_t col1 = CNShift.column_values[cid+1];

                if(col0 >= col_base && col0 < col_end)
                {
                    const uint32_t shift_value = CNShift.shift_values[cid+0];
                    val[0] ^= rotated_bits<DType, Z>(shift_value,
                                                  tdbv_id,
                                                  (col0 - col_base) * num_tdbv,
                                                  in);
                }
                if(col1 >= col_base && col1 < col_end)
                {
                    const uint32_t shift_value = CNShift.shift_values[cid+1];
                    val[1] ^= rotated_bits<DType, Z>(shift_value,
                                                  tdbv_id,
                                                  (col1 - col_base) * num_tdbv,
                                                  in);
                }
            }

            val[0] ^= val[1];

            // Handle the epilogue of up to one element
            if (deg_max_even < deg)
            {
                const uint32_t col = CNShift.column_values[deg_max_even];
                if(col >= col_base && col < col_end)
                {
                    const uint32_t shift_value = CNShift.shift_values[deg_max_even];
                    val[0] ^= rotated_bits<DType, Z>(shift_value,
                                                  tdbv_id,
                                                  (col - col_base) * num_tdbv,
                                                  in);
                }
            }

            uint32_t d_idx  = tdbv_id + row * num_tdbv;
            out[d_idx] = accumulated ? out[d_idx] ^ val[0] : val[0];
        }
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// ldpc_encode_in_bit_kernel()
template <typename DType, uint32_t Z>
__global__ void ldpc_encode_in_bit_kernel(const __grid_constant__ ldpcEncodeDescr_t desc)
{
    if (blockIdx.y >= desc.num_TBs) return;

    const int                BG       = desc.BG;
    const int                Kb       = desc.Kb;
    const char               Htype    = desc.H_type;
    const bool               puncture = desc.puncture;
    const int                num_rows = desc.num_rows;

    constexpr int    remainder = (Z >= 32) ? static_cast<int>(Z % 32) : -static_cast<int>(Z);
    constexpr size_t elem_size = sizeof(DType) * 8;
    constexpr int    num_tdbv = (Z / elem_size + int(Z % elem_size != 0));

    extern __shared__ DType sbuf[];
    DType*                  info_vec = sbuf;
    DType*                  d_or_m   = sbuf + num_tdbv * Kb;

    cg::thread_block block = cg::this_thread_block();

    int K_in_word = desc.input[blockIdx.y].layout().dimensions[0];
    int C_in_word = desc.input[blockIdx.y].layout().dimensions[1];

    // Each thread block processes different code blocks
    for(int c = blockIdx.x; c < C_in_word; c += gridDim.x)
    {
        // Step 1. Load a code block segment into the on-chip SMEM
        load_from_gmem_to_smem<DType, remainder>(Z,
                                                 desc.input[blockIdx.y],
                                                 info_vec,
                                                 c,
                                                 K_in_word,
                                                 block);

        block_multiply_in_bit<DType, 0, false, Z>(BG,
                                                  0,
                                                  Kb,
                                                  info_vec,
                                                  d_or_m,
                                                  block,
                                                  num_rows);

        // Step 3. solve equations for main parity bits
        // Note that max value for Z is CUPHY_LDPC_MAX_LIFTING_SIZE=384 and so
        // with DType uint32_t, num_tdbv can be at most 12. So only part of a warp will execute
        // the if-clause below.
        if(threadIdx.x < num_tdbv)
        {
            uint32_t mask = __activemask(); // could also hardcode to ((1 << num_tdbv) - 1)

            DType d_temp[3];
            int   tdbv_id = threadIdx.x;
            for(int i = 0; i < 3; i++)
            {
                d_temp[i] = d_or_m[tdbv_id + i * num_tdbv];
            }
            d_or_m[tdbv_id] ^= d_temp[1];
            d_or_m[tdbv_id] ^= d_temp[2];
            d_or_m[tdbv_id] ^= d_or_m[tdbv_id + 3 * num_tdbv]; // m1

            __syncwarp(mask); // Needed to avoid a potential RAW (read after write) hazard that can manifest
                              // when a different thread reads the same d_or_m shared memory location as part
                              // of the rotated_bits code than the thread that modified it above.
                              // For example, for Z = 384, thread 0 is the one modifying location &d_or_m[0],
                              // but both threads 0 and 11 can read from it below (BG=1, Htype == 3).

            if(Htype == 1)
            {
                DType m_shifted                = rotated_bits<DType, Z>(1,
                                                                 tdbv_id,
                                                                 0,
                                                                 d_or_m);

                __syncwarp(mask); // No hazard possible with current implementation, as any d_or_m shared memory location
                                  // read in rotated_bits will only be modified below by the same thread.
                                  // But adding to make code future-proof.
                d_or_m[tdbv_id + 1 * num_tdbv] = d_temp[0] ^ m_shifted;                                        // m2
                d_or_m[tdbv_id + 2 * num_tdbv] = d_temp[1] ^ d_or_m[tdbv_id + 1 * num_tdbv];                   // m3
                d_or_m[tdbv_id + 3 * num_tdbv] = d_temp[2] ^ d_or_m[tdbv_id] ^ d_or_m[tdbv_id + 2 * num_tdbv]; // m4
            }
            else if(Htype == 2)
            {
                constexpr int shift = 105 % Z;
                if constexpr(shift > 0)
                {
                    DType m_shifted = rotated_bits<DType, Z>(Z - shift,
                                                                     tdbv_id,
                                                                     0,
                                                                     d_or_m);
                    __syncwarp(mask); // Needed to avoid potential WAR (write after read) hazard for lifting sizes 52, 104, 208
                                      // which can occur when the same d_or_m shared memory location read in rotated_bits by one thread
                                      // is written to below by another thread.
                                      // For example, for Z = 52, threads 0 and 1 in the warp both read from location &d_or_m[0],
                                      // which will also be modified by thread 0. The __syncwarp ensures thread 1 reads the value before the write.
                    d_or_m[tdbv_id] = m_shifted; // m1
                }
                d_or_m[tdbv_id + 1 * num_tdbv] = d_temp[0] ^ d_or_m[tdbv_id];                      // m2
                d_or_m[tdbv_id + 3 * num_tdbv] = d_or_m[tdbv_id + 3 * num_tdbv] ^ d_or_m[tdbv_id]; // m4
                d_or_m[tdbv_id + 2 * num_tdbv] = d_temp[2] ^ d_or_m[tdbv_id + 3 * num_tdbv];       // m3
            }
            else if(Htype == 3)
            {
                DType m_shifted                = rotated_bits<DType, Z>(1,
                                                                 tdbv_id,
                                                                 0,
                                                                 d_or_m);

                __syncwarp(mask); // No hazard is possible with current implementation, but adding to make code future-proof.
                                  // Currently any d_or_m shared memory location read in rotated_bits will only be modified below by the same thread.
                d_or_m[tdbv_id + 1 * num_tdbv] = d_temp[0] ^ m_shifted;                                        // m2
                d_or_m[tdbv_id + 2 * num_tdbv] = d_temp[1] ^ d_or_m[tdbv_id] ^ d_or_m[tdbv_id + 1 * num_tdbv]; // m3
                d_or_m[tdbv_id + 3 * num_tdbv] = d_temp[2] ^ d_or_m[tdbv_id + 2 * num_tdbv];                   // m4
            }
            else
            {
                DType m_shifted                = rotated_bits<DType, Z>(Z - 1,
                                                                 tdbv_id,
                                                                 0,
                                                                 d_or_m);
                __syncwarp(mask); // Needed to avoid a potential WAR (write after read) hazard
                                  // which can occur when the same d_or_m shared memory location read in rotated_bits by one thread
                                  // is written to below by another thread.
                                  // For example, for Z = 352, threads 2 and 3 in the warp both read from location &d_or_m[2],
                                  // which will also be modified by thread 2. The __syncwarp ensures thread 3 reads the value before the write.
                d_or_m[tdbv_id]                = m_shifted;                                        // m1
                d_or_m[tdbv_id + 1 * num_tdbv] = d_temp[0] ^ d_or_m[tdbv_id];                      // m2
                d_or_m[tdbv_id + 2 * num_tdbv] = d_temp[1] ^ d_or_m[tdbv_id + 1 * num_tdbv];       // m3
                d_or_m[tdbv_id + 3 * num_tdbv] = d_or_m[tdbv_id + 3 * num_tdbv] ^ d_or_m[tdbv_id]; // m4
            }
        }

        __syncthreads();

        // Step 4. block multiply to get other parity bits
        block_multiply_in_bit<DType, 4, true, Z>(BG,
                                                 Kb,
                                                 Kb + 4,
                                                 d_or_m,
                                                 d_or_m,
                                                 block,
                                                 num_rows);

        int punctured_nodes = puncture ? CUPHY_LDPC_NUM_PUNCTURED_NODES : 0;
        int N_in_word       = div_round_up(((Kb + num_rows - punctured_nodes) * Z), 32u); // LDPC's per CB size in uint32_t elements

        // Step 5. Store the resulting code block segment from the on-chip SMEM to GMEM
        store_from_smem_to_gmem<DType, remainder>(Z,
                                                  num_tdbv,
                                                  desc.output[blockIdx.y],
                                                  sbuf,
                                                  c,
                                                  N_in_word,
                                                  puncture,
                                                  block);
    }
}

////////////////////////////////////////////////////////////////////////
// get_H_type()
char get_H_type(int BG, int iLS)
{
    char H_type[2][8] =
        {
            {3, 3, 3, 3, 3, 3, 2, 3},
            {4, 4, 4, 1, 4, 4, 4, 1}};
    return H_type[BG - 1][iLS];
}

////////////////////////////////////////////////////////////////////////
// get_encode_cta_size()
int get_encode_cta_size(int Z, int num_rows)
{
    // Parity nodes / rows are divided among tiles where tiles are groups of
    // either 8 or 16 threads.
    const int tile_size = (Z % 32 == 0 && Z > 256) ? 16 : 8;
    const int min_num_threads = num_rows * tile_size;
    if (min_num_threads <= 64) {
        return 64;
    } else if (min_num_threads <= 128) {
        return 128;
    } else {
        return 256;
    }
}

} // namespace ldpc

cuphyStatus_t CUPHYWINAPI cuphySetupLDPCEncode(cuphyLDPCEncodeLaunchConfig_t ldpcEncodeLaunchConfig,
                                               cuphyTensorDescriptor_t       inDesc,
                                               void*                         inAddr,
                                               cuphyTensorDescriptor_t       outDesc,
                                               void*                         outAddr,
                                               int                           BG,
                                               int                           Z,
                                               uint8_t                       puncture,
                                               int                           max_parity_nodes,
                                               int                           max_rv,
                                               uint8_t                       batching,
                                               int                           batched_TBs,
                                               void**                        inBatchedAddr,
                                               void**                        outBatchedAddr,
                                               void*                         h_workspace,
                                               void*                         d_workspace,
                                               void*                         cpu_desc,
                                               void*                         gpu_desc,
                                               uint8_t                       enable_desc_async_copy,
                                               cudaStream_t                  strm)
{
    if((max_rv < 0) || (max_rv > 3))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(batching && ((batched_TBs < 1) || (batched_TBs > PDSCH_MAX_UES_PER_CELL_GROUP)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(batching && ((inBatchedAddr == nullptr) || (outBatchedAddr == nullptr)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if ((h_workspace == nullptr) || (d_workspace == nullptr))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    tensor_pair in_pair(static_cast<const tensor_desc&>(*inDesc), (batching == 0) ? inAddr : inBatchedAddr[0]);
    tensor_pair out_pair(static_cast<const tensor_desc&>(*outDesc), (batching == 0) ? outAddr: outBatchedAddr[0]);

    const tensor_desc& in_desc  = in_pair.first.get();
    const tensor_desc& out_desc = out_pair.first.get();

    if(out_desc.layout().rank() > 2 || in_desc.layout().rank() > 2)
    {
        return CUPHY_STATUS_UNSUPPORTED_RANK;
    }

    if(in_desc.type() != CUPHY_BIT || out_desc.type() != CUPHY_BIT)
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }

    if(BG < 1 || BG > 2)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    int iLS = set_from_Z(Z);

    if(iLS == -1)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    char H_type = ldpc::get_H_type(BG, iLS);

    /* If the optimize_nrows flag is set, the LDPC encoder will compute a subset of parity nodes (max_parity_nodes).
       This optimization is only valid for redundancy version 0.
       Setting max_parity_nodes to 0, when calling the encode function, will compute all parity nodes.
    */
    int  orig_num_rows  = (BG == 1) ? CUPHY_LDPC_MAX_BG1_PARITY_NODES : CUPHY_LDPC_MAX_BG2_PARITY_NODES;
    bool optimize_nrows = ((max_rv == 0) && (max_parity_nodes != 0) && (max_parity_nodes < orig_num_rows));

    const int num_rows      = (optimize_nrows) ? max_parity_nodes : orig_num_rows;
    /* smem_num_rows should take into consideration the punctured nodes too, if puncturing is enabled,
       to avoid illegal shared memory read in the store_from_smem_to_gmem() function in ldpc_load_store.cuh.
       Furthermore, the d_or_m shared mem array in the LDPC encoder has a minimum size of 4 * num_tdbv where
       the num_tdbv multipler is (Z / 32 + int(Z % 32 != 0). smem_num_rows covers this memory segment,
       so enforce, with a check, that it has a minimum value of 4 (CUPHY_LDPC_MIN_PARITY_NODES). It may otherwise have a value less than 4 in cases where
       max_parity_nodes is less than four without puncturing or less than two with puncturing.
       The check also protects against the user passing a negative max_parity_nodes value.
    */
    const int smem_num_rows = num_rows + (puncture ? CUPHY_LDPC_NUM_PUNCTURED_NODES : 0);
    if (smem_num_rows < CUPHY_LDPC_MIN_PARITY_NODES)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }


    tensor_layout_any inWordLayout = word_layout_from_bit_layout(in_desc.layout());
#if 0
    LDPC_output_t     input(in_pair.second,
                        LDPC_output_t::layout_t(inWordLayout.dimensions.begin(),
                                                inWordLayout.strides.begin() + 1));
#endif
    tensor_layout_any  outWordLayout = word_layout_from_bit_layout(out_desc.layout());
#if 0
    LDPC_output_t      output(out_pair.second,
                         LDPC_output_t::layout_t(outWordLayout.dimensions.begin(),
                                                 outWordLayout.strides.begin() + 1));
#endif
    ldpcEncodeDescr_t& desc = *(static_cast<ldpcEncodeDescr_t*>(cpu_desc));
    desc.BG                 = BG;
    // FIXME: should KB be renamed to something else? The way it is used in the code is not spec compliant
    desc.Kb                 = BG == 1 ? CUPHY_LDPC_BG1_INFO_NODES : CUPHY_LDPC_MAX_BG2_INFO_NODES;
    desc.Z                  = Z;
    desc.puncture           = puncture;
    desc.H_type             = H_type;
    desc.num_rows           = num_rows;
    desc.num_TBs            = (batching == 0) ? 1 : batched_TBs;
    LDPC_output_t* h_input  = (LDPC_output_t*)h_workspace;
    LDPC_output_t* h_output = (LDPC_output_t*)h_workspace + desc.num_TBs;
    if (batching == 0) {
        h_input[0]          = LDPC_output_t(inAddr,
                                            LDPC_output_t::layout_t(inWordLayout.dimensions.begin(),
                                                                    inWordLayout.strides.begin() + 1));
        h_output[0]         = LDPC_output_t(outAddr,
                                            LDPC_output_t::layout_t(outWordLayout.dimensions.begin(),
                                                                    outWordLayout.strides.begin() + 1));
    } else {
        for (int cnt = 0; cnt < batched_TBs; cnt++)
        {
            h_input[cnt]   = LDPC_output_t(inBatchedAddr[cnt],
                                           LDPC_output_t::layout_t(inWordLayout.dimensions.begin(),
                                                                   inWordLayout.strides.begin() + 1));
            h_output[cnt]  = LDPC_output_t(outBatchedAddr[cnt],
                                           LDPC_output_t::layout_t(outWordLayout.dimensions.begin(),
                                                                   outWordLayout.strides.begin() + 1));
        }
    }
    desc.input  = (LDPC_output_t*)d_workspace;
    desc.output = (LDPC_output_t*)d_workspace +  desc.num_TBs;

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if(enable_desc_async_copy)
    {
        // Copy part of the workspace at a time
        CUDA_CHECK(cudaMemcpyAsync(d_workspace, h_workspace, 2 * desc.num_TBs * sizeof(LDPC_output_t), cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(ldpcEncodeDescr_t), cudaMemcpyHostToDevice, strm));
    }

    ldpcEncodeLaunchConfig->m_desc = static_cast<ldpcEncodeDescr_t*>(gpu_desc);

    int  C_in_word = h_input[0].layout().dimensions[1];
    dim3 blocks(C_in_word, desc.num_TBs, 1);
    dim3 block_size(ldpc::get_encode_cta_size(Z, num_rows), 1, 1);

    // We use a __grid_constant__ kernel parameter, so we pass the CPU descriptor and
    // its fields will be copied to constant memory.
    ldpcEncodeLaunchConfig->m_kernelArgs[0]                 = cpu_desc;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.extra        = nullptr;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.kernelParams = &(ldpcEncodeLaunchConfig->m_kernelArgs[0]);

    ldpcEncodeLaunchConfig->m_kernelNodeParams.blockDimX = block_size.x;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.blockDimY = block_size.y;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.blockDimZ = block_size.z;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.gridDimX  = blocks.x;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.gridDimY  = blocks.y;
    ldpcEncodeLaunchConfig->m_kernelNodeParams.gridDimZ  = blocks.z;
    //printf("grid {%d, %d, %d}\n", blocks.x, blocks.y, blocks.z);

    cudaFunction_t ldpc_device_function = nullptr;

    if (Z >= 32)
    {
        constexpr size_t elem_size = sizeof(uint32_t) * 8;
        ldpcEncodeLaunchConfig->m_kernelNodeParams.sharedMemBytes = ((Z / 32 + int(Z % 32 != 0)) * (desc.Kb + smem_num_rows) * sizeof(int)) +
            (Z / elem_size + int(Z % elem_size != 0));
    }
    else
    {
        // The original shared memory size was:
        //   (desc.Kb + smem_num_rows + 1) * sizeof(int)
        // but there were cases (e.g. with Z=7) where more words than that would be read from shared
        // memory. These additional reads were to extract bits needed to fill out the last bits of the
        // last 32 bit word, but these bits are not ultimately used if they are not valid bits. To avoid
        // out-of-bounds reads, we compute the number of input words that will be read to generate the
        // expected number of output bits. We then set the shared memory allocation to the maximum of
        // this value and the original shared memory allocation value. It would likely be cleaner to
        // terminate the loop that generates the output words once the expected number of bits have been
        // read, but that would involve a bigger change to small_store_from_smem_to_gmem that will
        // require validation. We currently do not zero-initialize all of shared memory, so some of the
        // final reads may be from unitialized memory. These bits are ultimately unused, but if necessary
        // we can pass the shared memory allocation to the kernel and zero-initialize at least the epilouge
        // portion at the beginning of the kernel.
        const int output_bits = 32 * (((desc.Kb + num_rows - puncture) * Z + 31) / 32);
        const int input_words = (puncture ? CUPHY_LDPC_NUM_PUNCTURED_NODES : 0) + ((output_bits + Z - 1) / Z);
        ldpcEncodeLaunchConfig->m_kernelNodeParams.sharedMemBytes = std::max(
            input_words * sizeof(int), (desc.Kb + smem_num_rows + 1) * sizeof(int));
    }
    #define LDPC_Z_CASE(_Z) \
        case _Z: \
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&ldpc_device_function, reinterpret_cast<void*>(ldpc::ldpc_encode_in_bit_kernel<uint32_t, _Z>))); }\
            break;
    switch (Z)
    {
        LDPC_Z_CASE(2)
        LDPC_Z_CASE(3)
        LDPC_Z_CASE(4)
        LDPC_Z_CASE(5)
        LDPC_Z_CASE(6)
        LDPC_Z_CASE(7)
        LDPC_Z_CASE(8)
        LDPC_Z_CASE(9)
        LDPC_Z_CASE(10)
        LDPC_Z_CASE(11)
        LDPC_Z_CASE(12)
        LDPC_Z_CASE(13)
        LDPC_Z_CASE(14)
        LDPC_Z_CASE(15)
        LDPC_Z_CASE(16)
        LDPC_Z_CASE(18)
        LDPC_Z_CASE(20)
        LDPC_Z_CASE(22)
        LDPC_Z_CASE(24)
        LDPC_Z_CASE(26)
        LDPC_Z_CASE(28)
        LDPC_Z_CASE(30)
        LDPC_Z_CASE(32)
        LDPC_Z_CASE(36)
        LDPC_Z_CASE(40)
        LDPC_Z_CASE(44)
        LDPC_Z_CASE(48)
        LDPC_Z_CASE(52)
        LDPC_Z_CASE(56)
        LDPC_Z_CASE(60)
        LDPC_Z_CASE(64)
        LDPC_Z_CASE(72)
        LDPC_Z_CASE(80)
        LDPC_Z_CASE(88)
        LDPC_Z_CASE(96)
        LDPC_Z_CASE(104)
        LDPC_Z_CASE(112)
        LDPC_Z_CASE(120)
        LDPC_Z_CASE(128)
        LDPC_Z_CASE(144)
        LDPC_Z_CASE(160)
        LDPC_Z_CASE(176)
        LDPC_Z_CASE(192)
        LDPC_Z_CASE(208)
        LDPC_Z_CASE(224)
        LDPC_Z_CASE(240)
        LDPC_Z_CASE(256)
        LDPC_Z_CASE(288)
        LDPC_Z_CASE(320)
        LDPC_Z_CASE(352)
        LDPC_Z_CASE(384)
        default:
            return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    #undef LDPC_Z_CASE

    ldpcEncodeLaunchConfig->m_kernelNodeParams.func = ldpc_device_function;
    return CUPHY_STATUS_SUCCESS;
}
