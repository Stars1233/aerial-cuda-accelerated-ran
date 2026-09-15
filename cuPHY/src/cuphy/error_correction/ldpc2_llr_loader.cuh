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

#if !defined(LDPC2_LLR_LOADER_CUH_INCLUDED_)
#define LDPC2_LLR_LOADER_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// warp_find_block_tb_token()
// Uses a warp-based scan operation to generate a tb_token that
// indicates which codeword the current CTA should process.
template <int CW_PER_CTA>
__device__
tb_token warp_find_block_tb_token(const cuphyLDPCDecodeDesc_t& decode_desc,
                                  unsigned int                 decode_index,
                                  tb_token*                    pToken)
{
    static_assert(CUPHY_LDPC_DECODE_DESC_MAX_TB <= 32,
                  "CUPHY_LDPC_DECODE_DESC_MAX_TB must be less than warp size");
    //------------------------------------------------------------------
    // Calculate a prefix sum for the aggregate number of codewords
    // across different transport blocks. Each thread in the warp reads
    // a value from a different transport block.
    const int LANEID           = threadIdx.x & 0x1F;
    int       entry_num_cw     = (LANEID < decode_desc.num_tbs) ? decode_desc.llr_input[LANEID].num_codewords
                                                      : 0;
    // Determine how many blocks would be required to process
    // 'entry_num_cw' codewords.
    int       entry_num_blocks = (entry_num_cw + CW_PER_CTA - 1) / CW_PER_CTA;
    int       blocks_sum       = warp_exclusive_scan<int>(entry_num_blocks);
    //------------------------------------------------------------------
    // Compare the per-thread codeword sum to the decode index. The
    // ballot will have a set bit for the first thread with a sum
    // greater than the current decode index (which is the same for all
    // threads in the warp).
    unsigned int sum_gt = __ballot_sync(0xFFFFFFFF, blocks_sum > decode_index);
    //------------------------------------------------------------------
    // __ffs() returns the 1-based index of the first set bit. We subtract
    // '1' from the __ffs() return value to convert to a zero-based
    // index, and another `1` to locate the index of the thread BEFORE
    // the thread with a set bit.
    unsigned int tb      = sum_gt ? (__ffs(sum_gt) - 2) : 31;
    unsigned int offset  = (decode_index - blocks_sum) * CW_PER_CTA;
    bool         partial = ((offset + CW_PER_CTA) > entry_num_cw);
    //------------------------------------------------------------------
    // Encode the entry and the offset from the start of the entry into
    // the tb_token.
    tb_token tok = to_token<CW_PER_CTA>(tb, offset, partial);
    if(tb == threadIdx.x)
    {
        // Store the token into shared memory for the thread with the
        // correct transport block.
        *pToken = tok;
        //printf("block %u: gt = 0x%0X, tb = %u, offset = %u, tok = 0x%0X, partial = %s\n",
        //       decode_index, sum_gt, tb, offset, tok, partial ? "true" : "false");
    }
    return tok;
}

////////////////////////////////////////////////////////////////////////
// input_codeword_addr
// Provides the input LLR address for a given index for the tensor-
// based LDPC decoder interface. It is expected that the index will be
// blockIdx.x for 1 codeword at a time, blockIdx.x * 2 for 2
// 2 codewords at a time, blockIdx.x * 4 for 4 codewords at a time.
template <typename T>
struct input_codeword_addr
{
    static __device__ const void* get(const LDPC_kernel_params& params, int idx)
    {
        return static_cast<const void*>(params.input_llr + (idx * sizeof(T) * params.input_llr_stride_elements));
    }
};

////////////////////////////////////////////////////////////////////////
// loader_token
// Class wrapper to disambiguate an additional argument passed to
// loader_params constructors.
struct loader_token
{
    tb_token token;
    __device__
    loader_token(tb_token tok) : token(tok) {}
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_loader_params<>
// Structure containing parameters used by the LLR loader functions in
// the LDPC decoder. The LLR loader loads data from global memory into
// shared memory.
template <typename T>
struct ldpc_dec_loader_params
{
    char*       dst_smem;            // destination shared memory address
    const void* src_gmem;            // source global memory address for this input CTA's codeword
    int         num_cw_elements;     // Z * num_var_nodes, needed for variable loaders
    float       clamp_value;         // clamp value
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // Constructor using the LDPC_kernel_params structure for
    // initialization.
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                     smem_dst,
                           const LDPC_kernel_params& params,
                           int                       decodeIndex) :
        dst_smem(smem_dst),
        src_gmem(input_codeword_addr<T>::get(params, decodeIndex * codewords_per_CTA<T>::value)),
        num_cw_elements(get_num_LLRs(params)),
        clamp_value(params.clamp_value)
    {
    }
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // Constructor using the cuphyLDPCDecodeDesc_t structure for
    // initialization.
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                        smem_dst,
                           const cuphyLDPCDecodeDesc_t& decodeDesc,
                           int                          decodeIndex) :
        dst_smem(smem_dst),
        src_gmem(nullptr),
        num_cw_elements(get_num_LLRs(decodeDesc)),
        clamp_value(decodeDesc.config.clamp_value)
    {
        int cwIndex = decodeIndex;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.llr_input[i].num_codewords)
                {
                    const char* c_addr = static_cast<const char*>(decodeDesc.llr_input[i].addr)          +
                                         (cwIndex * sizeof(T) * decodeDesc.llr_input[i].stride_elements);
                    src_gmem = static_cast<const void*>(c_addr);
                    break;
                }
                cwIndex -= decodeDesc.llr_input[i].num_codewords;
            }
        }
    }
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                        smem_dst,
                           const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const loader_token&          ltok) :
        dst_smem(smem_dst),
        src_gmem(nullptr),
        num_cw_elements(get_num_LLRs(decodeDesc)),
        clamp_value(decodeDesc.config.clamp_value)
    {
        int       tb          = tb_from_token(ltok.token);
        int       offset      = offset_from_token(ltok.token);
        int       stride_elem = decodeDesc.llr_input[tb].stride_elements;
        const int CW_STRIDE   = sizeof(__half) * stride_elem;
        const char* c_addr    = static_cast<const char*>(decodeDesc.llr_input[tb].addr) +
                                (offset * CW_STRIDE);
        src_gmem              = static_cast<const void*>(c_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_loader_params<> specialization for __half2
template <>
struct ldpc_dec_loader_params<__half2>
{
    char*       dst_smem;            // destination shared memory address
    const void* src_gmem;            // source global memory address for this CTA's input codeword pair
    int         max_cta_cw_index;    // 0 for 1 codeword, 1 for 2 codewords (0 when number of inputs is odd)
    int         src_stride_elements; // needed for 2x codewords, where loader reads two consecutive inputs
    int         num_cw_elements;     // Z * num_var_nodes, needed for variable loaders
    float       clamp_value;         // clamp value
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // Constructor using the LDPC_kernel_params structure for
    // initialization.
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                     smem_dst,
                           const LDPC_kernel_params& params,
                           int                       decodeIndex) :
        dst_smem(smem_dst),
        // Source type is __half, but we are processing 2 at a time (__half2)
        src_gmem(input_codeword_addr<__half>::get(params, decodeIndex * codewords_per_CTA<__half2>::value)),
        max_cta_cw_index((decodeIndex*2 + 1) >= params.num_codewords ? 0 : 1),
        src_stride_elements(params.input_llr_stride_elements),
        num_cw_elements(get_num_LLRs(params)),
        clamp_value(params.clamp_value)
    {
    }
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                        smem_dst,
                           const cuphyLDPCDecodeDesc_t& decodeDesc,
                           int                          decodeIndex) :
        dst_smem(smem_dst),
        src_gmem(nullptr),
        num_cw_elements(get_num_LLRs(decodeDesc)),
        clamp_value(decodeDesc.config.clamp_value)
    {
        int blkIndex = decodeIndex;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                // How many CTAs will be "claimed" by this TB?
                int iBlocksClaimed = (decodeDesc.llr_input[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    src_stride_elements = decodeDesc.llr_input[i].stride_elements;
                    const char* c_addr  = static_cast<const char*>(decodeDesc.llr_input[i].addr) +
                                          (blkIndex * 2 * sizeof(__half) * src_stride_elements);
                    src_gmem            = static_cast<const void*>(c_addr);
                    // Last block claimed by this TB may have only 1 codeword...
                    max_cta_cw_index    = ((blkIndex*2 + 1) == decodeDesc.llr_input[i].num_codewords) ? 0 : 1;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    //------------------------------------------------------------------
    // ldpc_dec_loader_params()
    // decodeIndex is an index into the set of codewords to be decoded.
    // For launches with a CTA per codeword, decodeIndex is simply
    // blockIdx.x.
    __device__
    ldpc_dec_loader_params(char*                        smem_dst,
                           const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const loader_token&          ltok) :
        dst_smem(smem_dst),
        src_gmem(nullptr),
        num_cw_elements(get_num_LLRs(decodeDesc)),
        clamp_value(decodeDesc.config.clamp_value)
    {
        int       tb         = tb_from_token(ltok.token);
        int       offset     = offset_from_token(ltok.token);
        bool      is_partial = is_partial_from_token(ltok.token);

        src_stride_elements = decodeDesc.llr_input[tb].stride_elements;
        const int CW_STRIDE = sizeof(__half) * src_stride_elements;
        const char* c_addr  = static_cast<const char*>(decodeDesc.llr_input[tb].addr) +
                              (offset * CW_STRIDE);
        src_gmem            = static_cast<const void*>(c_addr);
        // Last block claimed by this TB may have only 1 codeword...
        max_cta_cw_index    = is_partial ? 0 : 1;
    }
};

////////////////////////////////////////////////////////////////////////
// llr_loader_fixed
// Structure to load data from global to shared memory using a fixed (at
// compile time) number of variable nodes
// T:      LLR Type
// Z:      Lifting factor (LDPC)
// VNODES: Number of variable/bit nodes to load
//
// Example use:
//
// typedef llr_loader_fixed<T, Z, V>        llr_loader_t;
//
// __shared__ char smem[llr_loader_t::LLR_BUFFER_SIZE];
// or
// extern __shared__ char smem[];
//
// llr_loader_t loader;
// ldpc_dec_loader_params params;
// loader.load_sync(params);
//
//template <typename T, int Z, int VNODES>
//struct llr_loader_fixed
//{
//    // clang-format off
//    typedef typename ldpc_traits<T>::llr_ldg_t  llr_ldg_t;  // The type for LLR loads from global memory LLR.
//    typedef typename ldpc_traits<T>::llr_sts_t  llr_sts_t;  // The type for LLR stores to shared memory.
//    typedef typename ldpc_traits<T>::app_buf_t  app_buf_t;  // The type to store APP in shared memory.
//    typedef typename ldpc_traits<T>::app_elem_t app_elem_t; // Scalar APP data type, also source input data type
//    enum { THREADS_PER_CTA = Z };                                                               // Number of threads per CTA.
//    enum { LLR_ELEMENTS = VNODES * Z };                                                         // Number of LLR elements.
//    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                                  // Number of bytes loaded by each thread per LDG -- we use LDG.128.
//    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(T) };            // Number of elements loaded by each thread per LDG.
//    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };        // Number of bytes loaded by the CTA per LDG.
//    enum { LLR_ELEM_PER_CTA_PER_LDG = LLR_ELEM_PER_THREAD_PER_LDG * THREADS_PER_CTA };          // Number of elements loaded by the CTA per LDG.
//    enum { LLR_LDGS = (LLR_ELEMENTS + LLR_ELEM_PER_CTA_PER_LDG-1) / LLR_ELEM_PER_CTA_PER_LDG }; // Number of LDGs needed to load the LLR array.
//    enum { LLR_REMAINING_ELEM = LLR_ELEMENTS - (LLR_LDGS-1) * LLR_ELEM_PER_CTA_PER_LDG };       // Number of elements for the last load.
//    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                                  // Number of bytes loaded by each thread per STS.
//    enum { LLR_ELEM_PER_THREAD_PER_STS = LLR_BYTES_PER_THREAD_PER_STS / sizeof(T) };            // Number of elements loaded by each thread per STS.
//    enum { LLR_BYTES_PER_CTA_PER_STS = LLR_BYTES_PER_THREAD_PER_STS * THREADS_PER_CTA };        // Number of bytes loaded by the CTA per STS.
//    enum { LLR_ELEM_PER_CTA_PER_STS = LLR_ELEM_PER_THREAD_PER_STS * THREADS_PER_CTA };          // Number of elements loaded by the CTA per STS.
//    enum { LLR_BUFFER_SIZE = LLR_ELEMENTS * sizeof(app_buf_t) };
//    // clang-format on
//    //------------------------------------------------------------------
//    __device__
//    void load_sync(const ldpc_dec_loader_params<T>& params)
//    {
//        const char* gmem_c = static_cast<const char*>(params.src_gmem);
//
//        // The offset in global memory for LLR elements.
//        //int llr_gmem_offset = blockIdx.x*VNODES*Z + threadIdx.x*LLR_ELEM_PER_THREAD_PER_LDG;
//        int llr_gmem_offset = threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG;
//
//        // Issue the loads to read LLR elements from global memory. Stage data in registers.
//        #pragma unroll
//        for(int ii = 0; ii < LLR_LDGS - 1; ++ii)
//        {
//            const int imm    = ii * LLR_BYTES_PER_CTA_PER_LDG;
//            int       offset = llr_gmem_offset * sizeof(app_elem_t) + imm;
//            llr_[ii]         = *reinterpret_cast<const llr_ldg_t*>(&gmem_c[offset]);
//        }
//
//        // Deal with the last (possibly) incomplete LDG.
//        if(threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG < LLR_REMAINING_ELEM)
//        {
//            const int imm     = (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_LDG;
//            int       offset  = llr_gmem_offset * sizeof(app_elem_t) + imm;
//            llr_[LLR_LDGS - 1] = *reinterpret_cast<const llr_ldg_t*>(&gmem_c[offset]);
//        }
//
//        // The offset in shared memory for LLR elements.
//        int llr_smem_offset = threadIdx.x * LLR_ELEM_PER_THREAD_PER_STS;
//
//        // Copy the LLR elements to shared memory.
//        #pragma unroll
//        for(int ii = 0; ii < LLR_LDGS - 1; ++ii)
//        {
//            const int imm                                                                    = ii * LLR_BYTES_PER_CTA_PER_STS;
//            reinterpret_cast<llr_sts_t*>(&params.dst_smem[llr_smem_offset * sizeof(T) + imm])[0] = llr_[ii];
//        }
//
//        // Deal with the last (possibly) incomplete LDG.
//        if((threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG) < LLR_REMAINING_ELEM)
//        {
//            const int imm                                                                    = (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_STS;
//            reinterpret_cast<llr_sts_t*>(&params.dst_smem[llr_smem_offset * sizeof(T) + imm])[0] = llr_[LLR_LDGS - 1];
//        }
//
//        // Make sure the data is in shared memory.
//        __syncthreads();
//    }
//    //------------------------------------------------------------------
//    // Data
//    llr_sts_t llr_[LLR_LDGS]; // Register storage (GLOBAL --> REG --> SHMEM)
//};

////////////////////////////////////////////////////////////////////////
// llr_loader_variable
// Structure to load data from global to shared memory, when the number
// of variable nodes is not known until runtime.
// T:          LLR Type
// Z:          Lifting factor (LDPC)
// MAX_VNODES: Maximum number of variable/bit nodes that could be loaded
//             (Fewer may be loaded at runtime, but registers are
//             allocated for the maximum.)
//
// Example use:
//
// typedef llr_loader_variable<T, Z, MAX_V> llr_loader_t;
// extern __shared__ char smem[];
// llr_loader_t loader;
// ldpc_dec_loader_params params;
// loader.load_sync(params);
//
// If we assume that the number of threads in the CTA is Z, then the
// number of LLR values to load will be between 26 and 68 values per
// thread for BG1, and 14 and 52 values per thread for BG2. (These
// values are approximate in that the number of threads in the CTA
// will likely be rounded up to the next multiple of 32.) Since we
// may be performing vectorized loads, the number of read instructions
// may be a factor of 2, 4, or 8 smaller (depending on the APP type
// and the ldg type).
// This means that for fp32, a maximum of 68 (32-bit) registers could
// be used if the implementation fully unrolls, and for fp16 the
// maximum is 34 (32-bit) registers. (Those values are for BG1 - the
// values for BG2 are 52 and 26 respectively.)
//template <typename T, int Z, int MAX_VNODES>
//struct llr_loader_variable
//{
//    // clang-format off
//    typedef typename ldpc_traits<T>::llr_ldg_t  llr_ldg_t;  // The type to load LLR.
//    typedef typename ldpc_traits<T>::llr_sts_t  llr_sts_t;  // The type to store LLR to shared memory.
//    typedef typename ldpc_traits<T>::app_buf_t  app_buf_t;  // The type to store APP in shared memory.
//    typedef typename ldpc_traits<T>::app_elem_t app_elem_t; // Scalar APP data type, also source input data type
//    enum { THREADS_PER_CTA = Z };                                                                   // Number of threads per CTA.
//    enum { LLR_ELEMENTS = MAX_VNODES * Z };                                                         // Maximum number of LLR elements.
//    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                                      // Number of bytes loaded by each thread per LDG -- we use LDG.128.
//    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(T) };                // Number of elements loaded by each thread per LDG.
//    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };            // Number of bytes loaded by the CTA per LDG.
//    enum { LLR_ELEM_PER_CTA_PER_LDG = LLR_ELEM_PER_THREAD_PER_LDG * THREADS_PER_CTA };              // Number of elements loaded by the CTA per LDG.
//    enum { LLR_MAX_LDGS = (LLR_ELEMENTS + LLR_ELEM_PER_CTA_PER_LDG-1) / LLR_ELEM_PER_CTA_PER_LDG }; // Number of LDGs needed to load the LLR array.
//    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                                      // Number of bytes loaded by each thread per STS.
//    enum { LLR_ELEM_PER_THREAD_PER_STS = LLR_BYTES_PER_THREAD_PER_STS / sizeof(T) };                // Number of elements loaded by each thread per STS.
//    enum { LLR_BYTES_PER_CTA_PER_STS = LLR_BYTES_PER_THREAD_PER_STS * THREADS_PER_CTA };            // Number of bytes loaded by the CTA per STS.
//    enum { LLR_ELEM_PER_CTA_PER_STS = LLR_ELEM_PER_THREAD_PER_STS * THREADS_PER_CTA };              // Number of elements loaded by the CTA per STS.
//    enum { LLR_BUFFER_SIZE = LLR_ELEMENTS * sizeof(app_buf_t) };
//    // clang-format on
//    //------------------------------------------------------------------
//    __device__
//    void load_sync(const ldpc_dec_loader_params<T>& params)
//    {
//        const char* gmem_c = static_cast<const char*>(params.src_gmem);
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Calculate the number of "active" warp-wide LDGs, based on the
//        // number of nodes.
//        const int LLR_ACTIVE_LDG     = (params.num_cw_elements + LLR_ELEM_PER_CTA_PER_LDG - 1) / LLR_ELEM_PER_CTA_PER_LDG;
//        const int LLR_REMAINING_ELEM = params.num_cw_elements - ((LLR_ACTIVE_LDG-1) * LLR_ELEM_PER_CTA_PER_LDG); // Number of elements for the last load.
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // The starting offset in global memory for this thread
//        int llr_gmem_offset = threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG;
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Issue the loads to read LLR elements from global memory. Stage data in registers.
//        #pragma unroll
//        for(int ii = 0; ii < (LLR_MAX_LDGS - 1); ++ii)
//        {
//            const int imm    = ii * LLR_BYTES_PER_CTA_PER_LDG;
//            int       offset = llr_gmem_offset * sizeof(app_elem_t) + imm;
//            if(ii < (LLR_ACTIVE_LDG - 1))
//            {
//                llr_[ii] = *reinterpret_cast<const llr_ldg_t*>(&gmem_c[offset]);
//            }
//        }
//
//        // Deal with the last (possibly) incomplete LDG.
//        if((threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG) < LLR_REMAINING_ELEM)
//        {
//            const int imm            = (LLR_ACTIVE_LDG - 1) * LLR_BYTES_PER_CTA_PER_LDG;
//            int       offset         = llr_gmem_offset * sizeof(app_elem_t) + imm;
//            llr_[LLR_ACTIVE_LDG - 1] = *reinterpret_cast<const llr_ldg_t*>(&gmem_c[offset]);
//        }
//
//        // The offset in shared memory for LLR elements.
//        int llr_smem_offset = threadIdx.x * LLR_ELEM_PER_THREAD_PER_STS;
//
//        // Copy the LLR elements to shared memory.
//        for(int ii = 0; ii < (LLR_ACTIVE_LDG - 1); ++ii)
//        {
//            const int imm                                                                 = ii * LLR_BYTES_PER_CTA_PER_STS;
//            reinterpret_cast<llr_sts_t*>(&params.dst_smem[llr_smem_offset * sizeof(T) + imm])[0] = llr_[ii];
//        }
//
//        // Deal with the last (possibly) incomplete LDG.
//        if((threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG) < LLR_REMAINING_ELEM)
//        {
//            const int imm                                                                 = (LLR_ACTIVE_LDG - 1) * LLR_BYTES_PER_CTA_PER_STS;
//            reinterpret_cast<llr_sts_t*>(&params.dst_smem[llr_smem_offset * sizeof(T) + imm])[0] = llr_[LLR_ACTIVE_LDG - 1];
//        }
//
//        // Make sure the data is in shared memory.
//        __syncthreads();
//    }
//    //------------------------------------------------------------------
//    // Data
//    llr_sts_t llr_[LLR_MAX_LDGS]; // Register storage (GLOBAL --> REG --> SHMEM)
//};

////////////////////////////////////////////////////////////////////////
// llr_loader_variable_batch_fixed_cta
// Structure to load data from global to shared memory, when amount of
// data is not known until runtime, but the CTA size is known.
// Example use:
//
// typedef llr_loader_variable_batch_fixed_cta llr_loader_t;
// extern __shared__ char smem[];
// llr_loader_t loader;
// ldpc_dec_loader_params params;
// loader.load_sync(params);
//
// If we assume that the number of threads in the CTA is Z, then the
// number of LLR values to load will be between 26 and 68 values per
// thread for BG1, and 14 and 52 values per thread for BG2. (These
// values are approximate in that the number of threads in the CTA
// will likely be rounded up to the next multiple of 32.) Since we
// may be performing vectorized loads, the number of read instructions
// may be a factor of 2, 4, or 8 smaller (depending on the APP type
// and the ldg type).
// This means that for fp32, a maximum of 68 (32-bit) registers could
// be used if the implementation fully unrolls, and for fp16 the
// maximum is 34 (32-bit) registers. (Those values are for BG1 - the
// values for BG2 are 52 and 26 respectively.)
//template <typename T, int THREADS_PER_CTA, int BATCH_SIZE>
//struct llr_loader_variable_batch_fixed_cta
//{
//    typedef typename ldpc_traits<T>::llr_ldg_t   llr_ldg_t;  // The type to load LLR.
//    typedef typename ldpc_traits<T>::llr_sts_t   llr_sts_t;  // The type to store LLR to shared memory.
//    typedef typename ldpc_traits<T>::app_elem_t  app_elem_t; // The underlying APP element type
//    typedef typename ldpc_traits<T>::app_buf_t   app_buf_t;  // The APP type in the shared memory buffer
//
//    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                              // Number of bytes loaded by each thread per LDG -- we use LDG.128.
//    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(T) };        // Number of elements loaded by each thread per LDG.
//    enum { LLR_ELEM_PER_CTA_PER_LDG = LLR_ELEM_PER_THREAD_PER_LDG * THREADS_PER_CTA };      // Number of elements loaded by the CTA per LDG.
//    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };    // Number of bytes loaded by the CTA per LDG.
//    enum { LLR_ELEM_PER_CTA_PER_BATCH = LLR_ELEM_PER_CTA_PER_LDG * BATCH_SIZE };            // Number of elements in each CTA-wide "batch"
//    enum { LLR_BYTES_PER_CTA_PER_BATCH = LLR_ELEM_PER_CTA_PER_BATCH * sizeof(app_elem_t) }; // Number of elements in each CTA-wide "batch"
//    //------------------------------------------------------------------
//    // load_sync()
//    __device__
//    void load_sync(const ldpc_dec_loader_params<T>& params)
//    {
//        const char* gmem_c           = static_cast<const char*>(params.src_gmem);
//        const int   BATCH_COUNT      = (params.num_cw_elements + LLR_ELEM_PER_CTA_PER_BATCH - 1) / LLR_ELEM_PER_CTA_PER_BATCH;
//        int         batchLoadOffset  = threadIdx.x * LLR_BYTES_PER_THREAD_PER_LDG;
//        const int   LOAD_OFFSET_END  = params.num_cw_elements * sizeof(app_elem_t);
//        for(int iBatch = 0; iBatch < BATCH_COUNT; ++iBatch)
//        {
//            #pragma unroll
//            for(int ii = 0; ii < BATCH_SIZE; ++ii)
//            {
//                int offset = batchLoadOffset + (ii * LLR_BYTES_PER_CTA_PER_LDG);
//                if(offset < LOAD_OFFSET_END)
//                {
//                    llr_[ii]   = *reinterpret_cast<const llr_ldg_t*>(gmem_c + offset);
//                    *reinterpret_cast<llr_ldg_t*>(params.dst_smem + offset) = llr_[ii];
//                }
//            }
//            batchLoadOffset += LLR_BYTES_PER_CTA_PER_BATCH;
//        }
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Make sure the data is in shared memory.
//        __syncthreads();
//    }
//    //------------------------------------------------------------------
//    // Data
//    llr_ldg_t llr_[BATCH_SIZE]; // Register storage (GLOBAL --> REG --> SHMEM)
//};

////////////////////////////////////////////////////////////////////////
// llr_op_none
// LLR operator to pass through the source value unmodified
template <typename T, typename TStore>
struct llr_op_none
{
    __device__
    static TStore apply(const TStore& src, [[maybe_unused]] float clamp_value = 32.f) { return src; }
};

////////////////////////////////////////////////////////////////////////
// llr_op_clamp
// LLR operator to clamp inputs to a maximum value
template <typename T, typename TStore> struct llr_op_clamp;

template <> struct llr_op_clamp<__half, uint4>
{
    __device__
    static uint4 apply(const uint4& src, float clamp_value)
    {
        union u
        {
            uint4   ui4;
            __half2 f16x2[4];
        };
        u uSrc, uDst;
        uSrc.ui4 = src;
        __half2 clampValue = __float2half2_rn(clamp_value);

        uDst.f16x2[0] = clamp_signed(uSrc.f16x2[0], clampValue);
        uDst.f16x2[1] = clamp_signed(uSrc.f16x2[1], clampValue);
        uDst.f16x2[2] = clamp_signed(uSrc.f16x2[2], clampValue);
        uDst.f16x2[3] = clamp_signed(uSrc.f16x2[3], clampValue);

        //if(0 == threadIdx.x)
        //{
        //    printf("in: %.3f %.3f %.3f %.3f, out: %.3f %.3f %.3f %.3f\n",
        //    __high2float(uSrc.f16x2[1]), __low2float(uSrc.f16x2[1]), __high2float(uSrc.f16x2[0]), __low2float(uSrc.f16x2[0]),
        //    __high2float(uDst.f16x2[1]), __low2float(uDst.f16x2[1]), __high2float(uDst.f16x2[0]), __low2float(uDst.f16x2[0]));
        //}

        return uDst.ui4;
    }
};

template <> struct llr_op_clamp<__half, uint2>
{
    __device__
    static uint2 apply(const uint2& src, float clamp_value)
    {
        union u
        {
            uint2   ui2;
            __half2 f16x2[2];
        };
        u uSrc, uDst;
        uSrc.ui2 = src;

        __half2 clampValue = __float2half2_rn(clamp_value);

        uDst.f16x2[0] = clamp_signed(uSrc.f16x2[0], clampValue);
        uDst.f16x2[1] = clamp_signed(uSrc.f16x2[1], clampValue);

        //if(0 == threadIdx.x)
        //{
        //    printf("in: %.3f %.3f %.3f %.3f, out: %.3f %.3f %.3f %.3f\n",
        //    __high2float(uSrc.f16x2[1]), __low2float(uSrc.f16x2[1]), __high2float(uSrc.f16x2[0]), __low2float(uSrc.f16x2[0]),
        //    __high2float(uDst.f16x2[1]), __low2float(uDst.f16x2[1]), __high2float(uDst.f16x2[0]), __low2float(uDst.f16x2[0]));
        //}

        return uDst.ui2;
    }
};

template <> struct llr_op_clamp<__nv_fp8_e4m3, uint4>
{
    __device__
    static uint2 apply(const uint4& src, float clamp_value)
    {
        __half2 clampValue = __float2half2_rn(clamp_value);

        word_t in_h[4];
        in_h[0].u32 = src.x;
        in_h[1].u32 = src.y;
        in_h[2].u32 = src.z;
        in_h[3].u32 = src.w;
        word_t clamped_h[4];
        clamped_h[0].f16x2 = clamp_signed(in_h[0].f16x2, clampValue);
        clamped_h[1].f16x2 = clamp_signed(in_h[1].f16x2, clampValue);
        clamped_h[2].f16x2 = clamp_signed(in_h[2].f16x2, clampValue);
        clamped_h[3].f16x2 = clamp_signed(in_h[3].f16x2, clampValue);

        word_t cvt[2];
        half2_to_fp8x4_array<__nv_fp8_e4m3, 8>(cvt, clamped_h);

        //if(0 == threadIdx.x)
        //{
        //    word_t out_h[4];
        //    fp8x4_to_half2_array<__nv_fp8_e4m3, 4, 8>(out_h, cvt);
        //    printf("in: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f, out: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
        //    __high2float(in_h[3].f16x2), __low2float(in_h[3].f16x2), __high2float(in_h[2].f16x2), __low2float(in_h[2].f16x2),
        //    __high2float(in_h[1].f16x2), __low2float(in_h[1].f16x2), __high2float(in_h[0].f16x2), __low2float(in_h[0].f16x2),
        //    __high2float(out_h[3].f16x2), __low2float(out_h[3].f16x2), __high2float(out_h[2].f16x2), __low2float(out_h[2].f16x2),
        //    __high2float(out_h[1].f16x2), __low2float(out_h[1].f16x2), __high2float(out_h[0].f16x2), __low2float(out_h[0].f16x2));

        //}
        uint2 res;
        res.x = cvt[0].u32;
        res.y = cvt[1].u32;
        return res;
    }
};

////////////////////////////////////////////////////////////////////////
// llr_loader_variable_batch
// Structure to load data from global to shared memory, when amount of
// data is not known until runtime.
// Example use:
//
// typedef llr_loader_variable_batch<T, 4, llr_op_none> llr_loader_t;
// extern __shared__ char smem[];
// ldpc_dec_loader_params params;
// llr_loader_t::load_sync(params);
//
// If we assume that the number of threads in the CTA is Z, then the
// number of LLR values to load will be between 26 and 68 values per
// thread for BG1, and 14 and 52 values per thread for BG2. (These
// values are approximate in that the number of threads in the CTA
// will likely be rounded up to the next multiple of 32.) Since we
// may be performing vectorized loads, the number of read instructions
// may be a factor of 2, 4, or 8 smaller (depending on the APP type
// and the ldg type).
// This means that for fp32, a maximum of 68 (32-bit) registers could
// be used if the implementation fully unrolls, and for fp16 the
// maximum is 34 (32-bit) registers. (Those values are for BG1 - the
// values for BG2 are 52 and 26 respectively.)
template <typename T, int BATCH_SIZE, template<typename, typename> class TLLROperator>
struct llr_loader_variable_batch
{
public:
    typedef typename ldpc_traits<T>::llr_ldg_t   llr_ldg_t;  // The type to load LLR.
    typedef typename ldpc_traits<T>::llr_sts_t   llr_sts_t;  // The type to store LLR to shared memory.
    typedef typename ldpc_traits<T>::app_elem_t  app_elem_t; // The underlying APP element type
    typedef typename ldpc_traits<T>::app_buf_t   app_buf_t;  // The APP type for the shared memory buffer

    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                                // Number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(app_elem_t) }; // Number of elements loaded by each thread per LDG.
private:
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(const ldpc_dec_loader_params<T>& params)
    {
        llr_ldg_t llr_[BATCH_SIZE]; // Register storage (GLOBAL --> REG --> SHMEM)

        const char* gmem_c                      = static_cast<const char*>(params.src_gmem);
        const int   LLR_BYTES_PER_CTA_PER_LDG   = LLR_BYTES_PER_THREAD_PER_LDG * blockDim.x;
        const int   LLR_BYTES_PER_CTA_PER_BATCH = LLR_BYTES_PER_CTA_PER_LDG * BATCH_SIZE;
        int         batchLoadOffset             = threadIdx.x * LLR_BYTES_PER_THREAD_PER_LDG;
        const int   LOAD_OFFSET_END             = params.num_cw_elements * sizeof(app_elem_t);
        while(batchLoadOffset < LOAD_OFFSET_END)
        {
            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                int offset = batchLoadOffset + (ii * LLR_BYTES_PER_CTA_PER_LDG);
                if(offset < LOAD_OFFSET_END)
                {
                    llr_[ii]   = *reinterpret_cast<const llr_ldg_t*>(gmem_c + offset);
                    // Perform an operation on the LLR value (clamp, "None", ...)
                    llr_sts_t storeValue = TLLROperator<app_elem_t, llr_sts_t>::apply(llr_[ii], params.clamp_value);
                    *reinterpret_cast<llr_ldg_t*>(params.dst_smem + offset) = storeValue;
                }
            }
            batchLoadOffset += LLR_BYTES_PER_CTA_PER_BATCH;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make sure the data is in shared memory.
        __syncthreads();
    }
public:
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                     smem,
                          const LDPC_kernel_params& kernelParams,
                          int                       decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, kernelParams, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_token()
    // Load LLR data and return a token that locates the CTA codeword,
    // and can be used to locate decoder output.
    __device__
    static tb_token load_sync_token(char*                        smem,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    int                          decodeIndex,
                                    tb_token*                    pToken)
    {
        // Threads in the first warp (only) will determine which codeword
        // to process.
        if(0 == (threadIdx.x / 32))
        {
            warp_find_block_tb_token<1>(decodeDesc, decodeIndex, pToken);
        }
        // Synchronize so that all threads in the CTA can access the
        // token (in shared memory) stored by a thread in the first warp.
        __syncthreads();
        tb_token tok = *pToken;
        ldpc_dec_loader_params<T> params(smem,
                                         decodeDesc,
                                         loader_token(tok));
        load_sync(params);
        return tok;
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const LDPC_kernel_params&    kernelParams,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(kernelParams) * sizeof(T),
                                                      sizeof(ldpc_traits<T>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<T> params(smem + (i * LLR_STRIDE_BYTES),
                                             kernelParams,
                                             cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                        smem,
                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                          int                          decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, decodeDesc, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(T),
                                                      sizeof(ldpc_traits<T>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<T> params(smem + (i * LLR_STRIDE_BYTES),
                                             decodeDesc,
                                             cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
};


union u_loader_64
{
    float2  f;
    uint2   u;
    __half2 h[2];
};

__device__ inline
void llr_loader_convert(u_loader_64& dst, const float4& f4)
{
    float2 f2_0 = {f4.x, f4.y};
    float2 f2_1 = {f4.z, f4.w};
    dst.h[0] = __float22half2_rn(f2_0);
    dst.h[1] = __float22half2_rn(f2_1);
}

////////////////////////////////////////////////////////////////////////
// llr_loader_variable_batch_convert
// Structure to load data from global to shared memory, when amount of
// data is not known until runtime. Conversion to a different APP type
// is performed (e.g. converting from float to fp16).
// Example use:
//
// typedef llr_loader_variable_batch_convert llr_loader_t;
// extern __shared__ char smem[];
// ldpc_dec_loader_params params;
// llr_loader_t::load_sync(params);
//
template <typename T, int BATCH_SIZE, template <typename, typename> class TLLROperator>
struct llr_loader_variable_batch_convert
{
    typedef typename ldpc_traits<T>::llr_ldg_t  llr_ldg_t;  // The type to load LLR.
    typedef typename ldpc_traits<T>::llr_sts_t  llr_sts_t;  // The type to store LLR to shared memory.
    typedef typename ldpc_traits<T>::llr_src_t  llr_src_t;  // LLR source data type
    typedef typename ldpc_traits<T>::app_elem_t app_elem_t; // The underlying APP element type
    typedef typename ldpc_traits<T>::app_buf_t  app_buf_t;  // The APP type for the shared memory buffer

    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                               // Number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                               // Number of bytes stored by each thread per STS
    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(llr_src_t) }; // Number of elements loaded by each thread per LDG.
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(const ldpc_dec_loader_params<T>& params)
    {
        llr_ldg_t llr_[BATCH_SIZE]; // Register storage (GLOBAL --> REG --> SHMEM)

        const char* gmem_c                          = static_cast<const char*>(params.src_gmem);
        const int   LLR_BYTES_PER_CTA_PER_LDG       = LLR_BYTES_PER_THREAD_PER_LDG * blockDim.x;
        const int   LLR_BYTES_PER_CTA_PER_STS       = LLR_BYTES_PER_THREAD_PER_STS * blockDim.x;
        const int   LLR_LDG_BYTES_PER_CTA_PER_BATCH = LLR_BYTES_PER_CTA_PER_LDG * BATCH_SIZE;
        const int   LLR_STS_BYTES_PER_CTA_PER_BATCH = LLR_BYTES_PER_CTA_PER_STS * BATCH_SIZE;
        int         batchLoadOffset                 = threadIdx.x * LLR_BYTES_PER_THREAD_PER_LDG;
        int         batchStoreOffset                = threadIdx.x * LLR_BYTES_PER_THREAD_PER_STS;
        const int   LOAD_OFFSET_END                 = params.num_cw_elements * sizeof(llr_src_t);
        while(batchLoadOffset < LOAD_OFFSET_END)
        {
            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                int loadOffset = batchLoadOffset + (ii * LLR_BYTES_PER_CTA_PER_LDG);
                if(loadOffset < LOAD_OFFSET_END)
                {
                    u_loader_64 cvt;
                    int storeOffset = batchStoreOffset + (ii * LLR_BYTES_PER_CTA_PER_STS);
                    llr_[ii]        = *reinterpret_cast<const llr_ldg_t*>(gmem_c + loadOffset);
                    llr_loader_convert(cvt, llr_[ii]);
                    // Perform an operation on the LLR value (clamp, "None", ...)
                    llr_sts_t storeValue = TLLROperator<app_elem_t, llr_sts_t>::apply(cvt.u, params.clamp_value);
                    *reinterpret_cast<llr_sts_t*>(params.dst_smem + storeOffset) = storeValue;
                }
            }
            batchLoadOffset  += LLR_LDG_BYTES_PER_CTA_PER_BATCH;
            batchStoreOffset += LLR_STS_BYTES_PER_CTA_PER_BATCH;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make sure the data is in shared memory.
        __syncthreads();
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                     smem,
                          const LDPC_kernel_params& kernelParams,
                          int                       decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, kernelParams, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const LDPC_kernel_params&    kernelParams,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(kernelParams) * sizeof(T),
                                                      sizeof(ldpc_traits<T>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<T> params(smem + (i * LLR_STRIDE_BYTES),
                                             kernelParams,
                                             cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                        smem,
                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                          int                          decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, decodeDesc, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(T),
                                                      sizeof(ldpc_traits<T>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<T> params(smem + (i * LLR_STRIDE_BYTES),
                                             decodeDesc,
                                             cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// llr_loader_variable_batch_pre_convert
// Structure to load data from global to shared memory, when amount of
// data is not known until runtime. Conversion to a different APP type
// is performed (e.g. converting from float to fp16). The provided
// operator is applied to loaded data and is responsible for any clamp
// or other modifications as well as conversion to the destination type.
// Note that the destination type is determined from the ldpc_traits
// for the TDst template parameter, and is typically a type appropriate
// for vectorized loads (e.g. uint3 or uint4).
// Example use:
//
// typedef llr_loader_variable_batch_pre_convert llr_loader_t;
// extern __shared__ char smem[];
// ldpc_dec_loader_params params;
// llr_loader_t::load_sync(params);
//
// TDst:         type of LLR data to store in shared memory
// TSrc:         type of LLR data loaded from global memory
// BATCH_SIZE:   number of unrolled batched global loads
// TLLROperator: operator to apply to loaded value (after conversion)
template <typename TDst,
          typename TSrc,
          int BATCH_SIZE,
          template <typename, typename> class TLLROperator>
struct llr_loader_variable_batch_pre_convert
{
    using llr_ldg_t  = typename ldpc_traits<TSrc>::llr_ldg_t;  // The type to load LLR.
    using llr_sts_t  = typename ldpc_traits<TDst>::llr_sts_t;  // The type to store LLR to shared memory.
    using llr_src_t  = TSrc;                                   // LLR source data type
    using llr_dst_t  = TDst;
    using app_elem_t = TSrc;                                   // The source APP element type
    using app_buf_t  = TDst;                                   // The APP type for the shared memory buffer

    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                               // Number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                               // Number of bytes stored by each thread per STS
    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(llr_src_t) }; // Number of elements loaded by each thread per LDG.
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(const ldpc_dec_loader_params<llr_src_t>& params)
    {
        llr_ldg_t llr_[BATCH_SIZE]; // Register storage (GLOBAL --> REG --> SHMEM)

        const char* gmem_c                          = static_cast<const char*>(params.src_gmem);
        const int   LLR_BYTES_PER_CTA_PER_LDG       = LLR_BYTES_PER_THREAD_PER_LDG * blockDim.x;
        const int   LLR_BYTES_PER_CTA_PER_STS       = LLR_BYTES_PER_THREAD_PER_STS * blockDim.x;
        const int   LLR_LDG_BYTES_PER_CTA_PER_BATCH = LLR_BYTES_PER_CTA_PER_LDG * BATCH_SIZE;
        const int   LLR_STS_BYTES_PER_CTA_PER_BATCH = LLR_BYTES_PER_CTA_PER_STS * BATCH_SIZE;
        int         batchLoadOffset                 = threadIdx.x * LLR_BYTES_PER_THREAD_PER_LDG;
        int         batchStoreOffset                = threadIdx.x * LLR_BYTES_PER_THREAD_PER_STS;
        const int   LOAD_OFFSET_END                 = params.num_cw_elements * sizeof(llr_src_t);
        while(batchLoadOffset < LOAD_OFFSET_END)
        {
            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                int loadOffset = batchLoadOffset + (ii * LLR_BYTES_PER_CTA_PER_LDG);
                if(loadOffset < LOAD_OFFSET_END)
                {
                    int storeOffset = batchStoreOffset + (ii * LLR_BYTES_PER_CTA_PER_STS);
                    llr_[ii]        = *reinterpret_cast<const llr_ldg_t*>(gmem_c + loadOffset);
                    // Perform an operation on the LLR value (clamp, "None", ...) BEFORE conversion
                    llr_sts_t storeValue = TLLROperator<llr_dst_t, llr_ldg_t>::apply(llr_[ii], params.clamp_value);
                    *reinterpret_cast<llr_sts_t*>(params.dst_smem + storeOffset) = storeValue;
                }
            }
            batchLoadOffset  += LLR_LDG_BYTES_PER_CTA_PER_BATCH;
            batchStoreOffset += LLR_STS_BYTES_PER_CTA_PER_BATCH;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make sure the data is in shared memory.
        __syncthreads();
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                     smem,
                          const LDPC_kernel_params& kernelParams,
                          int                       decodeIndex)
    {
        ldpc_dec_loader_params<llr_src_t> params(smem, kernelParams, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const LDPC_kernel_params&    kernelParams,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_DST_STRIDE_BYTES = round_up_to_next(get_num_LLRs(kernelParams) * sizeof(llr_dst_t),
                                                          sizeof(ldpc_traits<llr_dst_t>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<llr_src_t> params(smem + (i * LLR_DST_STRIDE_BYTES),
                                                     kernelParams,
                                                     cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                        smem,
                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                          int                          decodeIndex)
    {
        ldpc_dec_loader_params<llr_src_t> params(smem, decodeDesc, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_multi()
    // Multiple codewords per CTA (small Z)
    // All threads cooperate to load data for each codeword, one at a
    // time.
    __device__
    static void load_sync_multi(char*                        smem,
                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const multi_codeword_config& cwConfig)
    {
        const int LLR_DST_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(llr_dst_t),
                                                          sizeof(ldpc_traits<llr_dst_t>::llr_sts_t));
        for(int i = 0; i < cwConfig.cta_codeword_count; ++i)
        {
            ldpc_dec_loader_params<llr_src_t> params(smem + (i * LLR_DST_STRIDE_BYTES),
                                                     decodeDesc,
                                                     cwConfig.cta_start_index + i);
            load_sync(params);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// llr_loader_variable_batch (__half2 specialization)
// Structure to load data from global to shared memory using a variable
// (unknown until runtime) number of nodes.
// T:      LLR Type
//
// Example use:
//
// typedef llr_loader_variable_batch<T, 4, llr_op_clamp> llr_loader_t;
//
// __shared__ char smem[llr_loader_t::LLR_BUFFER_SIZE];
// or
// extern __shared__ char smem[];
//
// ldpc_dec_loader_params params;
// llr_loader_t::load_sync(params);
//
template <int BATCH_SIZE, template<typename, typename> class TLLROperator>
struct llr_loader_variable_batch<__half2, BATCH_SIZE, TLLROperator>
{
    typedef __half2 T;
    typedef typename ldpc_traits<T>::llr_ldg_t  llr_ldg_t;  // The type for LLR loads from global memory LLR.
    typedef typename ldpc_traits<T>::llr_sts_t  llr_sts_t;  // The type for LLR stores to shared memory.
    typedef typename ldpc_traits<T>::app_buf_t  app_buf_t;  // The type to store APP in shared memory.
    typedef typename ldpc_traits<T>::app_elem_t app_elem_t; // Scalar APP data type, also source input data type

    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                                // Number of bytes loaded by each thread per LDG
    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                                // Number of bytes stored by each thread per STS
    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(app_elem_t) }; // Number of elements loaded by each thread per LDG.

    //------------------------------------------------------------------
    __device__
    static void load_sync(const ldpc_dec_loader_params<__half2>& params)
    {
        const char* gmem_c                          = static_cast<const char*>(params.src_gmem);
        const int   LLR_BYTES_PER_CTA_PER_LDG       = LLR_BYTES_PER_THREAD_PER_LDG * blockDim.x;
        const int   LLR_BYTES_PER_CTA_PER_BATCH_LDG = LLR_BYTES_PER_CTA_PER_LDG * BATCH_SIZE;
        const int   LLR_BYTES_PER_CTA_PER_STS       = LLR_BYTES_PER_THREAD_PER_STS * blockDim.x;
        const int   LLR_BYTES_PER_CTA_PER_BATCH_STS = LLR_BYTES_PER_CTA_PER_STS * BATCH_SIZE;
        int         batchLoadOffset                 = threadIdx.x * LLR_BYTES_PER_THREAD_PER_LDG;
        int         batchStoreOffset                = threadIdx.x * LLR_BYTES_PER_THREAD_PER_STS;
        const int   LOAD_OFFSET_END                 = params.num_cw_elements * sizeof(app_elem_t);
        const int   STORE_OFFSET_END                = params.num_cw_elements * sizeof(app_buf_t);
        const char* inputLLR[2];

        inputLLR[0] = gmem_c + (0 * params.src_stride_elements * sizeof(app_elem_t));
        inputLLR[1] = gmem_c + (1 * params.src_stride_elements * sizeof(app_elem_t));
        while(batchLoadOffset < LOAD_OFFSET_END)
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // We use two sets of registers for input from global memory
            // because we are loading from two separate codewords.
            llr_ldg_t llr_ldg[2][BATCH_SIZE]; // Register storage (GLOBAL --> REG)
            llr_sts_t llr_sts[BATCH_SIZE];
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Load from global memory into registers: llr_ldg[][]
            #pragma unroll
            for(int iCW = 0; iCW < 2; ++iCW)
            {
                if(iCW <= params.max_cta_cw_index) // Avoid reads past the end of input for odd number of codewords
                {
                    #pragma unroll
                    for(int ii = 0; ii < BATCH_SIZE; ++ii)
                    {
                        int offset = batchLoadOffset + (ii * LLR_BYTES_PER_CTA_PER_LDG);
                        if(offset < LOAD_OFFSET_END)
                        {
                            llr_ldg[iCW][ii]   = *reinterpret_cast<const llr_ldg_t*>(inputLLR[iCW] + offset);
                        }
                    }
                }
            }
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Interleave values from the two input codewords and copy to
            // shared memory:
            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                llr_sts[ii] = interleave_llr(llr_ldg[0][ii], llr_ldg[1][ii]);
                // Perform an operation on the LLR value (clamp, "None", ...)
                llr_sts_t storeValue = TLLROperator<app_elem_t, llr_sts_t>::apply(llr_sts[ii], params.clamp_value);
                int offset = batchStoreOffset + (ii * LLR_BYTES_PER_CTA_PER_STS);
                if(offset < STORE_OFFSET_END)
                {
                    *reinterpret_cast<llr_sts_t*>(params.dst_smem + offset) = storeValue;
                }
            }
            batchLoadOffset  += LLR_BYTES_PER_CTA_PER_BATCH_LDG;
            batchStoreOffset += LLR_BYTES_PER_CTA_PER_BATCH_STS;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make sure the data is in shared memory.
        __syncthreads();
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                     smem,
                          const LDPC_kernel_params& kernelParams,
                          int                       decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, kernelParams, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync()
    __device__
    static void load_sync(char*                        smem,
                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                          int                          decodeIndex)
    {
        ldpc_dec_loader_params<T> params(smem, decodeDesc, decodeIndex);
        load_sync(params);
    }
    //------------------------------------------------------------------
    // load_sync_print()
    //__device__
    //static void load_sync_print(char*                        smem,
    //                            const cuphyLDPCDecodeDesc_t& decodeDesc,
    //                            int                          decodeIndex)
    //{
    //    ldpc_dec_loader_params<T> params(smem, decodeDesc, decodeIndex);
    //    if(0 == threadIdx.x)
    //    {
    //        printf("[block = %u]: src = %p, max_index = %i, stride = %i, num_cw_elem = %i\n",
    //               decodeIndex,
    //               params.src_gmem,
    //               params.max_cta_cw_index,
    //               params.src_stride_elements,
    //               params.num_cw_elements);
    //    }
    //    load_sync(params);
    //}
    //------------------------------------------------------------------
    // token_print()
    // Load LLR data and return a token that locates the CTA codeword,
    // and can be used to locate decoder output.
    //__device__
    //static void token_print(char*                        smem,
    //                        const cuphyLDPCDecodeDesc_t& decodeDesc,
    //                        int                          decodeIndex)
    //{
    //    tb_token                  tok = warp_find_block_tb_token<2>(decodeDesc, decodeIndex);
    //    ldpc_dec_loader_params<T> params(smem, decodeDesc, loader_token(tok));
    //    if(0 == threadIdx.x)
    //    {
    //        printf("[block = %u]: src = %p, max_index = %i, stride = %i, num_cw_elem = %i\n",
    //               decodeIndex,
    //               params.src_gmem,
    //               params.max_cta_cw_index,
    //               params.src_stride_elements,
    //               params.num_cw_elements);
    //    }
    //}
    //------------------------------------------------------------------
    // load_sync_token()
    // Load LLR data and return a token that locates the CTA codeword,
    // and can be used to locate decoder output.
    __device__
    static tb_token load_sync_token(char*                        smem,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    int                          decodeIndex,
                                    tb_token*                    pToken)
    {
        // Threads in the first warp (only) will determine which codeword
        // to process.
        if(0 == (threadIdx.x / 32))
        {
            warp_find_block_tb_token<2>(decodeDesc, decodeIndex, pToken);
        }
        // Synchronize so that all threads in the CTA can access the
        // token (in shared memory) stored by a thread in the first warp.
        __syncthreads();
        tb_token tok = *pToken;
        ldpc_dec_loader_params<T> params(smem,
                                         decodeDesc,
                                         loader_token(tok));
        load_sync(params);
        return tok;
    }
};

////////////////////////////////////////////////////////////////////////
// llr_loader_fixed (__half2 specialization)
// Structure to load data from global to shared memory using a fixed (at
// compile time) number of variable nodes
// T:      LLR Type
// Z:      Lifting factor (LDPC)
// VNODES: Number of variable/bit nodes to load
//
// Example use:
//
// typedef llr_loader_fixed<T, Z, V>              llr_loader_t;
//
// __shared__ char smem[llr_loader_t::LLR_BUFFER_SIZE];
// or
// extern __shared__ char smem[];
//
// llr_loader_t loader;
// ldpc_dec_loader_params params;
// loader.load_sync(params);
//
//template <int Z, int VNODES>
//struct llr_loader_fixed<__half2, Z, VNODES>
//{
//    // clang-format off
//    typedef typename ldpc_traits<__half2>::llr_ldg_t   llr_ldg_t;  // The type to load LLR.
//    typedef typename ldpc_traits<__half2>::llr_sts_t   llr_sts_t;  // The type to store LLR to shared memory.
//    typedef typename ldpc_traits<__half2>::app_buf_t   app_buf_t;  // The type to store APP in shared memory.
//    typedef typename ldpc_traits<__half2>::app_elem_t  app_elem_t; // The underlying APP element type
//    enum { THREADS_PER_CTA = Z };                                                                // Number of threads per CTA.
//    enum { LLR_ELEMENTS = VNODES * Z };                                                          // Number of LLR elements.
//    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(llr_ldg_t) };                                   // Number of bytes loaded by each thread per LDG
//    enum { LLR_ELEM_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(app_elem_t) };    // Number of elements loaded by each thread per LDG.
//    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };         // Number of bytes loaded by the CTA per LDG.
//    enum { LLR_ELEM_PER_CTA_PER_LDG = LLR_ELEM_PER_THREAD_PER_LDG * THREADS_PER_CTA };           // Number of elements loaded by the CTA per LDG.
//    enum { LLR_LDGS = (LLR_ELEMENTS + LLR_ELEM_PER_CTA_PER_LDG-1) / LLR_ELEM_PER_CTA_PER_LDG };  // Number of LDGs needed to load the LLR array.
//    enum { LLR_REMAINING_ELEM = LLR_ELEMENTS - (LLR_LDGS-1) * LLR_ELEM_PER_CTA_PER_LDG };        // Number of elements for the last load.
//
//    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(llr_sts_t) };                                   // Number of bytes loaded by each thread per STS.
//    enum { LLR_ELEM_PER_THREAD_PER_STS = LLR_BYTES_PER_THREAD_PER_STS / sizeof(app_elem_t) };    // Number of elements loaded by each thread per STS.
//    enum { LLR_BYTES_PER_CTA_PER_STS = LLR_BYTES_PER_THREAD_PER_STS * THREADS_PER_CTA };         // Number of bytes loaded by the CTA per STS.
//    enum { LLR_ELEM_PER_CTA_PER_STS = LLR_ELEM_PER_THREAD_PER_STS * THREADS_PER_CTA };           // Number of elements loaded by the CTA per STS.
//    enum { LLR_BUFFER_SIZE = LLR_ELEMENTS * sizeof(app_buf_t) };
//    // clang-format on
//    //------------------------------------------------------------------
//    __device__
//    void load_sync(const ldpc_dec_loader_params<__half2>& params)
//    {
//        const char* gmem_c = static_cast<const char*>(params.src_gmem);
//
//        //if(0 == threadIdx.x) printf("smem = %p, LLR_BUFFER_SIZE = %u\n", smem, LLR_BUFFER_SIZE);
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Load from global memory to registers. We use two sets of
//        // registers because we are loading from two separate codewords.
//        llr_ldg_t llr_ldg[2][LLR_LDGS]; // Register storage (GLOBAL --> REG)
//
//        #pragma unroll
//        for(int iCW = 0; iCW < 2; ++iCW)
//        {
//            if(iCW <= params.max_cta_cw_index) // Avoid reads past the end of input for odd number of codewords
//            {
//
//                int llr_g_offset_bytes = sizeof(app_elem_t) * ((iCW * params.src_stride_elements) +
//                                                               (threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG));
//
//                #pragma unroll
//                for(int jLDG = 0; jLDG < LLR_LDGS-1; ++jLDG)
//                {
//                    llr_ldg[iCW][jLDG] = *reinterpret_cast<const llr_ldg_t*>(gmem_c + llr_g_offset_bytes + (jLDG * LLR_BYTES_PER_CTA_PER_LDG));
//                }
//                // Deal with the last (possibly) incomplete LDG.
//                if((threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG) < LLR_REMAINING_ELEM)
//                {
//                    llr_ldg[iCW][LLR_LDGS-1] = *reinterpret_cast<const llr_ldg_t*>(gmem_c + llr_g_offset_bytes + ((LLR_LDGS-1)*LLR_BYTES_PER_CTA_PER_LDG));
//                }
//            }
//        }
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Interleave data from adjacent codewords into the high and low
//        // parts of fp16x2 values.
//        llr_sts_t llr_sts[LLR_LDGS];
//
//        #pragma unroll
//        for(int iLDG = 0; iLDG < LLR_LDGS; ++iLDG)
//        {
//            llr_sts[iLDG] = interleave_llr(llr_ldg[0][iLDG], llr_ldg[1][iLDG]);
//        }
//
//        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//        // Store register data to shared memory
//        // The offset in shared memory for LLR elements.
//        int llr_s_offset_bytes = threadIdx.x * LLR_BYTES_PER_THREAD_PER_STS;
//
//        // Copy the LLR elements to shared memory.
//        #pragma unroll
//        for(int iSTS = 0; iSTS < (LLR_LDGS-1); ++iSTS)
//        {
//            const int imm                                                  = iSTS * LLR_BYTES_PER_CTA_PER_STS;
//            *reinterpret_cast<llr_sts_t*>(params.dst_smem + llr_s_offset_bytes + imm) = llr_sts[iSTS];
//        }
//
//        // Deal with the last (possibly) incomplete LDG.
//        if((threadIdx.x * LLR_ELEM_PER_THREAD_PER_LDG) < LLR_REMAINING_ELEM)
//        {
//            const int imm                                                  = (LLR_LDGS-1) * LLR_BYTES_PER_CTA_PER_STS;
//            *reinterpret_cast<llr_sts_t*>(params.dst_smem + llr_s_offset_bytes + imm) = llr_sts[LLR_LDGS - 1];
//        }
//        // Synchronize shared memory contents
//        __syncthreads();
//    }
//    //------------------------------------------------------------------
//    // Data
//
//};

} // namespace ldpc2

#endif // !defined(LDPC2_LLR_LOADER_CUH_INCLUDED_)
