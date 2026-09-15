/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if !defined(LDPC2_ALGO103_CUH_INCLUDED_)
#define LDPC2_ALGO103_CUH_INCLUDED_

// Internal helpers for the p=4 algo103 layered decoder.

#include <algorithm>
#include <assert.h>

#include "ldpc2_box_plus.cuh"
#include "ldpc2_bg_desc.hpp"
#include "ldpc2_crc_dispatch.cuh"
#include "ldpc2_dec_output.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_llr_loader.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    static constexpr int ALGO103_Z              = 384;
    static constexpr int ALGO103_BG             = 1;
    static constexpr int ALGO103_INFO_NODES     = 22;
    static constexpr int ALGO103_MIN_PARITY     = 4;
    static constexpr int ALGO103_MAX_PARITY     = 7;
    static constexpr int ALGO103_PARITY        = 4;
    static constexpr int ALGO103_MAX_ROWS       = ALGO103_MAX_PARITY;
    static constexpr int ALGO103_MAX_ROW_DEGREE = 19;

    static constexpr int ALGO103_REF_BG1_INFO_NODES = 22;
    static constexpr int ALGO103_REF_BG2_INFO_NODES = 10;
    static constexpr int ALGO103_REF_BG1_MAX_ROWS   = 46;
    static constexpr int ALGO103_REF_BG2_MAX_ROWS   = 42;
    static constexpr int ALGO103_REF_BG1_MAX_DEG    = 19;
    static constexpr int ALGO103_REF_BG2_MAX_DEG    = 10;
    static constexpr int ALGO103_REF_MAX_ROWS       = ALGO103_REF_BG1_MAX_ROWS;
    static constexpr int ALGO103_REF_MAX_DEG        = ALGO103_REF_BG1_MAX_DEG;

    static constexpr int MAX_THREADS_PER_CTA   = ALGO103_Z;
    static constexpr int MIN_CTA_PER_SM_ALGO103_REF = 1;
    static constexpr int MIN_CTA_PER_SM_ALGO103 = 1;
    static constexpr int MIN_CTA_PER_SM_ALGO103_SCALAR = 2;

    typedef llr_loader_variable_batch<__half, 4, llr_op_clamp>  llr_loader_ref_t;
    typedef llr_loader_ref_t::app_buf_t                        app_buf_ref_t;
    typedef llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
    typedef llr_loader_t::app_buf_t                            app_buf_t;

    //------------------------------------------------------------------
    // algo103_llr_loader_narrow
    //
    // The accepted configuration has exactly (22 + 4) * 384 fp16 channel
    // LLRs per codeword.  The leading two BG1 systematic columns are always
    // punctured by the NR transport-block mapping, so their channel LLRs are
    // exact zero.  The collapsed iteration-0 schedule writes every V0 slot in
    // row 1 and every V1 slot in row 2 before either column is read.  Therefore
    // their global loads, clamps, interleaves and shared stores are dead.
    //
    // Starting at V2 leaves exactly 24 columns.  Move four adjacent fp16
    // values per thread with LDG.64: six full, branch-free CTA batches cover
    // those 24 columns.  Issuing all twelve independent requests from both
    // codeword streams before the first interleave exposes more memory-level
    // parallelism to the sole resident CTA than the three LDG.128 batches.
    //
    // The destination byte offset is exactly twice the source byte offset:
    // each scalar fp16 becomes one fp16x2 word after interleaving.  Thus V2
    // through V25 retain the generic loader's shared layout and fp16 clamp;
    // V0/V1 remain intentionally unstaged until their iteration-0 writes.
    struct algo103_llr_loader_narrow
    {
        static constexpr int PUNCTURED_ELEMS = CUPHY_LDPC_NUM_PUNCTURED_NODES * ALGO103_Z;
        static constexpr int PUNCTURED_BYTES = PUNCTURED_ELEMS * static_cast<int>(sizeof(__half));
        static constexpr int BATCH_SIZE      = 6;
        static_assert(CUPHY_LDPC_NUM_PUNCTURED_NODES == 2,
                      "algo103 staging elision requires the two NR punctured columns");

        __device__ __forceinline__
        static void load_sync(const ldpc_dec_loader_params<__half2>& params)
        {
            constexpr int LDG_BYTES        = static_cast<int>(sizeof(uint2));
            constexpr int CTA_LDG_BYTES    = ALGO103_Z * LDG_BYTES;
            constexpr int STS_BYTES        = static_cast<int>(sizeof(uint4));
            constexpr int CTA_STORE_BYTES  = ALGO103_Z * STS_BYTES;
            static_assert(PUNCTURED_BYTES + BATCH_SIZE * CTA_LDG_BYTES ==
                              (ALGO103_INFO_NODES + ALGO103_PARITY) * ALGO103_Z *
                                  static_cast<int>(sizeof(__half)),
                          "V2..V25 must be exactly six full CTA load batches");

            const char* input0 = static_cast<const char*>(params.src_gmem);
            const char* input1 = input0 +
                                 (params.src_stride_elements * static_cast<int>(sizeof(__half)));
            const int thread_load_offset = PUNCTURED_BYTES + threadIdx.x * LDG_BYTES;

            uint2 llr0[BATCH_SIZE];
            uint2 llr1[BATCH_SIZE];

            // Launch every useful request before consuming any result.
            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                const int offset = thread_load_offset + (ii * CTA_LDG_BYTES);
                llr0[ii] = *reinterpret_cast<const uint2*>(input0 + offset);
            }

            if(params.max_cta_cw_index > 0)
            {
                // Peel the first request of stream 1 ahead of its loop while
                // preserving the established stream-0 request group.
                llr1[0] = *reinterpret_cast<const uint2*>(input1 + thread_load_offset);

                #pragma unroll
                for(int ii = 1; ii < BATCH_SIZE; ++ii)
                {
                    const int offset = thread_load_offset + (ii * CTA_LDG_BYTES);
                    llr1[ii] = *reinterpret_cast<const uint2*>(input1 + offset);
                }
            }

            #pragma unroll
            for(int ii = 0; ii < BATCH_SIZE; ++ii)
            {
                const uint2 second = (params.max_cta_cw_index > 0)
                                         ? llr1[ii]
                                         : make_uint2(0u, 0u);
                uint4 store = interleave_llr(llr0[ii], second);
                store = llr_op_clamp<__half, uint4>::apply(store, params.clamp_value);

                const int store_offset = (2 * PUNCTURED_BYTES) +
                                         (threadIdx.x * STS_BYTES) +
                                         (ii * CTA_STORE_BYTES);
                *reinterpret_cast<uint4*>(params.dst_smem + store_offset) = store;
            }
            __syncthreads();
        }
    };
    //------------------------------------------------------------------
    __device__ __forceinline__
    int wrap_z_shift(int z, int shift)
    {
        const int u = z + shift;
        return (u >= ALGO103_Z) ? (u - ALGO103_Z) : u;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t box_plus_identity()
    {
        word_t out;
        out.u32 = 0x7BFF7BFFu; // max finite fp16 in both halves
        return out;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t box_pair(word_t a, word_t b)
    {
        return box_plus_op::box_plus(a, b);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half2 half2_from_raw(__half2_raw h)
    {
        return static_cast<__half2>(h);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t h2_add(word_t a, word_t b)
    {
        word_t out;
        out.f16x2 = __hadd2(half2_from_raw(a.f16x2), half2_from_raw(b.f16x2));
        return out;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t h2_sub(word_t a, word_t b)
    {
        word_t out;
        out.f16x2 = __hsub2(half2_from_raw(a.f16x2), half2_from_raw(b.f16x2));
        return out;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t h2_mul_norm(word_t a, __half2 norm)
    {
        word_t out;
        out.f16x2 = __hmul2(half2_from_raw(a.f16x2), norm);
        return out;
    }

    //------------------------------------------------------------------
    template <int V>
    __device__ __forceinline__
    word_t load_channel_app_base(const ldpc_dec_loader_params<__half2>& params, int z)
    {
        const int idx = (V * ALGO103_Z) + z;
        const __half* input0 = static_cast<const __half*>(params.src_gmem);
        const __half* input1 = input0 + params.src_stride_elements;

        const __half low  = input0[idx];
        const __half high = (params.max_cta_cw_index > 0) ? input1[idx] : __float2half(0.0f);
        word_t out;
        out.f16x2 = clamp_signed(__halves2half2(low, high), __float2half2_rn(params.clamp_value));
        return out;
    }

    //------------------------------------------------------------------
    // find_block_tb_token_allwarp()
    // Barrier-free transport-block token search (component (a) of the
    // fixed-overhead target).  The host has already interpreted the complete
    // arbitrary-length TB list once into per-TB CTA begin/end boundaries.  A
    // warp-uniform lower-bound search therefore replaces the lane-distributed
    // five-stage prefix scan.  It is O(log(num_tbs)), uses the same path for all
    // descriptor counts, and produces the same token as
    // warp_find_block_tb_token<CW_PER_CTA>().
    template <int CW_PER_CTA>
    __device__ __forceinline__
    tb_token find_block_tb_token_allwarp(const cuphyLDPCDecodeDesc_t& decode_desc,
                                         unsigned int                 decode_index)
    {
        static_assert(CUPHY_LDPC_DECODE_DESC_MAX_TB <= 32,
                      "CUPHY_LDPC_DECODE_DESC_MAX_TB must be <= warp size");
        static_assert(CW_PER_CTA == 2,
                      "prepared descriptor index stores two-codeword CTA ranges");
        // Search the starts of TBs 1..N-1 for the first interval beginning
        // after this CTA.  TB0 begins at zero by definition, so excluding that
        // sentinel avoids one redundant comparison for every descriptor list.
        unsigned int lo = 1;
        unsigned int hi = decode_desc.num_tbs;
        while(lo < hi)
        {
            const unsigned int mid = lo + ((hi - lo) >> 1);
            const unsigned int candidate_begin =
                decode_desc.llr_output[mid].stride_elements;
            if(decode_index < candidate_begin)
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }
        const unsigned int tb = lo - 1;
        const unsigned int block_begin = decode_desc.llr_output[tb].stride_elements;
        const unsigned int offset = (decode_index - block_begin) * CW_PER_CTA;
        const unsigned int entry_num_cw = decode_desc.llr_input[tb].num_codewords;
        const bool partial = ((offset + CW_PER_CTA) > entry_num_cw);
        return to_token<CW_PER_CTA>(tb, offset, partial);
    }

    //------------------------------------------------------------------
    // load_channel_app_token_nobar()
    // Barrier-free token search + the puncture-aware narrow channel staging.
    // Its trailing CTA barrier publishes V2..V25 before the collapsed first
    // iteration consumes them; V0/V1 are deliberately written later by that
    // iteration without ever reading their unstaged shared slots.
    __device__ __forceinline__
    tb_token load_channel_app_token_nobar(char*                        dst_smem,
                                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                                          int                          decodeIndex)
    {
        tb_token tok = find_block_tb_token_allwarp<2>(decodeDesc, decodeIndex);
        ldpc_dec_loader_params<__half2> params(dst_smem, decodeDesc, loader_token(tok));
        algo103_llr_loader_narrow::load_sync(params);
        return tok;
    }

    //------------------------------------------------------------------
    // load_channel_app_token_nobar_generic()
    // As load_channel_app_token_nobar(), but stages the channel LLRs with
    // the generic p-aware loader. The narrow staging above is sized to
    // algo103's fixed p=4 column extent (V2..V25 only); a runtime-p caller
    // staging through it would leave the tail parity columns
    // unwritten, so they read stale shared memory once CTAs are reused
    // across waves.
    __device__ __forceinline__
    tb_token load_channel_app_token_nobar_generic(char*                        dst_smem,
                                                  const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                  int                          decodeIndex)
    {
        tb_token tok = find_block_tb_token_allwarp<2>(decodeDesc, decodeIndex);
        ldpc_dec_loader_params<__half2> params(dst_smem, decodeDesc, loader_token(tok));
        llr_loader_t::load_sync(params);
        return tok;
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__ __forceinline__
    void clear_fwd_storage(word_t (&)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE])
    {
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int NUM_ROWS>
    __device__ __forceinline__
    word_t fwd_get(const word_t (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE])
    {
        return fwd[ROW][POS];
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int NUM_ROWS>
    __device__ __forceinline__
    void fwd_set(word_t (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE], word_t v)
    {
        fwd[ROW][POS] = v;
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int num_var_nodes(int num_parity_nodes)
    {
        return ALGO103_INFO_NODES + num_parity_nodes;
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int app_words(int num_parity_nodes)
    {
        return num_var_nodes(num_parity_nodes) * ALGO103_Z;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t* app_smem(char* smem)
    {
        return reinterpret_cast<word_t*>(smem);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t* local_c2v_smem(char* smem, int num_parity_nodes)
    {
        return app_smem(smem) + app_words(num_parity_nodes);
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int get_app_local_shmem(int num_parity_nodes)
    {
        // The active layered path keeps C2V in c2v_store_t registers.  The
        // legacy local-C2V slab is never addressed by either split kernel.
        return static_cast<int>(app_words(num_parity_nodes) * sizeof(word_t));
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int get_shmem_required(int num_parity_nodes)
    {
        int shmem_size = get_app_local_shmem(num_parity_nodes);
#if LDPC_DECODE_USE_TB_SCAN
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }

    //------------------------------------------------------------------
    int get_shmem_launch_required(int num_parity_nodes)
    {
        return static_cast<int>(ldpc2::shmem_size_with_experimental_et_context(
            static_cast<uint32_t>(get_shmem_required(num_parity_nodes))));
    }

    CUDA_BOTH
    int ref_info_nodes(int bg)
    {
        return (bg == 1) ? ALGO103_REF_BG1_INFO_NODES : ALGO103_REF_BG2_INFO_NODES;
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int ref_max_parity_nodes(int bg)
    {
        return (bg == 1) ? ALGO103_REF_BG1_MAX_ROWS : ALGO103_REF_BG2_MAX_ROWS;
    }

    //------------------------------------------------------------------
    CUDA_BOTH
    int ref_num_var_nodes(int bg, int num_parity_nodes)
    {
        return ref_info_nodes(bg) + num_parity_nodes;
    }
    CUDA_BOTH
    int get_algo103_scalar_app_local_shmem()
    {
        return static_cast<int>((app_words(ALGO103_PARITY) +
                                 (ALGO103_PARITY * ALGO103_Z)) *
                                sizeof(app_buf_ref_t));
    }

    //------------------------------------------------------------------
    int get_algo103_scalar_shmem_launch_required()
    {
        return static_cast<int>(ldpc2::shmem_size_with_experimental_et_context(
            static_cast<uint32_t>(get_algo103_scalar_app_local_shmem())));
    }
    __device__ __forceinline__
    __half* app_smem_ref(char* smem)
    {
        return reinterpret_cast<__half*>(smem);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half* local_c2v_smem_ref(char* smem, int bg, int num_parity_nodes, int Z)
    {
        return app_smem_ref(smem) + (ref_num_var_nodes(bg, num_parity_nodes) * Z);
    }
    __device__ __forceinline__
    __half half_box_plus_identity()
    {
        return __float2half(65504.0f);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    word_t half_to_dup_word(__half h)
    {
        word_t w;
        w.f16x2 = __halves2half2(h, h);
        return w;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half half_box_pair(__half a, __half b)
    {
        return __low2half(box_plus_op::box_plus(half_to_dup_word(a), half_to_dup_word(b)).f16x2);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half half_norm_from_config(const cuphyLDPCDecodeConfigDesc_t& config)
    {
        return __low2half(half2_from_raw(config.norm.f16x2));
    }
#ifdef CUPHY_EXPERIMENTAL_LDPC_ET
    //------------------------------------------------------------------
    __device__ __forceinline__
    int algo103_valid_cws(tb_token tok)
    {
        return is_partial_from_token(tok) ? 1 : 2;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    int algo103_tb_cw_index(tb_token tok, int cw_in_cta)
    {
        return offset_from_token(tok) + cw_in_cta;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    ldpc_et_context_t* algo103_et_ctx(char* smem, int num_parity_nodes)
    {
        const uint32_t base_size = static_cast<uint32_t>(get_shmem_required(num_parity_nodes));
        return ldpc2::get_et_ctx_ptr(smem, base_size);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    cuphyLDPCCrcType_t algo103_crc_type(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                         tb_token                     tok,
                                         int                          cw_in_cta)
    {
        const int tb = tb_from_token(tok);
        const int cw = algo103_tb_cw_index(tok, cw_in_cta);
        if(decodeDesc.llr_input[tb].crc_type != nullptr)
        {
            return static_cast<cuphyLDPCCrcType_t>(decodeDesc.llr_input[tb].crc_type[cw]);
        }
        return CUPHY_LDPC_CRC_NONE;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    uint32_t* algo103_crc_addr(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                tb_token                     tok,
                                int                          cw_in_cta)
    {
        const int tb = tb_from_token(tok);
        const int cw = algo103_tb_cw_index(tok, cw_in_cta);
        return (decodeDesc.tb_output[tb].crc != nullptr) ? (decodeDesc.tb_output[tb].crc + cw) : nullptr;
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    int32_t* algo103_iter_addr(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                tb_token                     tok,
                                int                          cw_in_cta)
    {
        const int tb = tb_from_token(tok);
        const int cw = algo103_tb_cw_index(tok, cw_in_cta);
        return (decodeDesc.iter_output[tb].addr != nullptr) ? (decodeDesc.iter_output[tb].addr + cw) : nullptr;
    }

    //------------------------------------------------------------------
    template <int CW_IN_CTA>
    __device__ __forceinline__
    __half algo103_extract_half(__half2 h)
    {
        if constexpr (CW_IN_CTA == 0)
        {
            return __low2half(h);
        }
        else
        {
            return __high2half(h);
        }
    }

    //------------------------------------------------------------------
    template <cuphyLDPCCrcType_t CRCVariant, int BG, int CW_IN_CTA>
    __device__ __forceinline__
    uint32_t compute_crc_x2_lane(const __half2* apps, ldpc_et_context_t* et_ctx)
    {
        if constexpr ((CRCVariant == CUPHY_LDPC_CRC_16) && (BG == 1))
        {
            return 0xBAD; // CRC-16 is not supported for BG1
        }

        constexpr int NUM_CRC_WORDS = (BG == 1) ? CUPHY_LDPC_BG1_INFO_NODES : CUPHY_LDPC_MAX_BG2_INFO_NODES;
        constexpr uint32_t WARP_REDUCE_MASK = (1u << NUM_CRC_WORDS) - 1;
        constexpr int WARP_SIZE = 32;
        const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
        const int swizzle5lsbs_threadIdx = swizzle_5lsbs(threadIdx.x);

        uint32_t selected_word = 0;
        uint32_t selected_word_idx = 0;
        uint32_t lut_value = 1;

        #pragma unroll
        for(int lane_selection_idx = 0; lane_selection_idx < NUM_CRC_WORDS; ++lane_selection_idx)
        {
            const int app_idx = lane_selection_idx * blockDim.x + swizzle5lsbs_threadIdx;
            const __half app = algo103_extract_half<CW_IN_CTA>(apps[app_idx]);
            const uint32_t word = __ballot_sync(0xffffffff, llr_hard_decision(app));

            if(lane_idx == lane_selection_idx)
            {
                selected_word = word;
                selected_word_idx = app_idx / WARP_SIZE;
                if constexpr (CRCVariant == CUPHY_LDPC_CRC_16)
                {
                    lut_value = G_CRC_16_P_LUT[selected_word_idx];
                }
                else if constexpr (CRCVariant == CUPHY_LDPC_CRC_24A)
                {
                    lut_value = G_CRC_24_A_P_LUT[selected_word_idx];
                }
                else
                {
                    lut_value = G_CRC_24_B_P_LUT[selected_word_idx];
                }
            }
        }

        uint32_t partial_crc = 0;
        int word_idx = 0;
        if(lane_idx < NUM_CRC_WORDS)
        {
            word_idx = threadIdx.x;
            selected_word = __byte_perm(selected_word, 0, 0x0123);

            if constexpr (CRCVariant == CUPHY_LDPC_CRC_16)
            {
                partial_crc = mulModCRCPolyLUT<16>(selected_word,
                                                   lut_value,
                                                   G_CRC_16_256_LUT,
                                                   G_CRC_16);
            }
            else if constexpr (CRCVariant == CUPHY_LDPC_CRC_24A)
            {
                partial_crc = mulModCRCPolyLUT<24>(selected_word,
                                                   lut_value,
                                                   G_CRC_24_A_256_LUT,
                                                   G_CRC_24_A);
            }
            else
            {
                partial_crc = mulModCRCPolyLUT<24>(selected_word,
                                                   lut_value,
                                                   G_CRC_24_B_256_LUT,
                                                   G_CRC_24_B);
            }
            partial_crc = __reduce_xor_sync(WARP_REDUCE_MASK, partial_crc);
        }

        if(lane_idx == 0)
        {
            et_ctx->partial_crcs[word_idx / WARP_SIZE] = partial_crc;
        }
        __syncthreads();

        uint32_t crc = 0;
        constexpr int NUM_WARPS = ALGO103_Z / WARP_SIZE;
        constexpr uint32_t FINAL_WARP_REDUCE_MASK = (1u << NUM_WARPS) - 1;
        if(threadIdx.x < NUM_WARPS)
        {
            crc = et_ctx->partial_crcs[threadIdx.x];
            crc = __reduce_xor_sync(FINAL_WARP_REDUCE_MASK, crc);
        }
        if(threadIdx.x == 0)
        {
            et_ctx->partial_crcs[0] = crc;
        }
        __syncthreads();

        return et_ctx->partial_crcs[0];
    }

    //------------------------------------------------------------------
    template <int CW_IN_CTA>
    __device__ __forceinline__
    uint32_t should_terminate_early_crc_x2(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                           tb_token                     tok,
                                           const app_buf_t*             apps,
                                           ldpc_et_context_t*           et_ctx)
    {
        switch(algo103_crc_type(decodeDesc, tok, CW_IN_CTA))
        {
            case CUPHY_LDPC_CRC_16:
                return compute_crc_x2_lane<CUPHY_LDPC_CRC_16, ALGO103_BG, CW_IN_CTA>(apps, et_ctx);
            case CUPHY_LDPC_CRC_24A:
                return compute_crc_x2_lane<CUPHY_LDPC_CRC_24A, ALGO103_BG, CW_IN_CTA>(apps, et_ctx);
            case CUPHY_LDPC_CRC_24B:
                return compute_crc_x2_lane<CUPHY_LDPC_CRC_24B, ALGO103_BG, CW_IN_CTA>(apps, et_ctx);
            case CUPHY_LDPC_CRC_NONE:
            default:
                return 0xBAD;
        }
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    bool algo103_should_stop_early(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    tb_token                     tok,
                                    const app_buf_t*             apps,
                                    ldpc_et_context_t*           et_ctx,
                                    uint32_t (&crc)[2])
    {
        crc[0] = should_terminate_early_crc_x2<0>(decodeDesc, tok, apps, et_ctx);
        if(algo103_valid_cws(tok) > 1)
        {
            crc[1] = should_terminate_early_crc_x2<1>(decodeDesc, tok, apps, et_ctx);
        }
        else
        {
            crc[1] = 0;
        }
        return (crc[0] == 0) && (crc[1] == 0);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    void algo103_write_crc_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                   tb_token                     tok,
                                   const uint32_t (&crc)[2])
    {
        if(threadIdx.x == 0)
        {
            uint32_t* addr0 = algo103_crc_addr(decodeDesc, tok, 0);
            if(addr0 != nullptr)
            {
                *addr0 = crc[0];
            }
            if(algo103_valid_cws(tok) > 1)
            {
                uint32_t* addr1 = algo103_crc_addr(decodeDesc, tok, 1);
                if(addr1 != nullptr)
                {
                    *addr1 = crc[1];
                }
            }
        }
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    void algo103_write_iter_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    tb_token                     tok,
                                    int32_t                      iter)
    {
        if(threadIdx.x == 0)
        {
            int32_t* addr0 = algo103_iter_addr(decodeDesc, tok, 0);
            if(addr0 != nullptr)
            {
                *addr0 = iter;
            }
            if(algo103_valid_cws(tok) > 1)
            {
                int32_t* addr1 = algo103_iter_addr(decodeDesc, tok, 1);
                if(addr1 != nullptr)
                {
                    *addr1 = iter;
                }
            }
        }
    }
#endif

#if LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    __device__
    tb_token* get_token_addr(char* smem, int num_parity_nodes)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_local_shmem(num_parity_nodes));
    }
#endif
    template <int ROW, int POS, int V, int SHIFT, class TFwd>
    __device__ __forceinline__
    word_t init_fwd_slot(word_t* app,
                         int     z,
                         TFwd&   fwd,
                         word_t acc)
    {
        fwd_set<ROW, POS>(fwd, acc);
        const int u = wrap_z_shift(z, SHIFT);
        return box_pair(acc, app[(V * ALGO103_Z) + u]);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__
    void init_fwd_from_app_p4(char* smem, TFwd& fwd)
    {
        word_t* app = app_smem(smem);
        const int z = threadIdx.x;
        word_t acc;

        clear_fwd_storage(fwd);

        acc = box_plus_identity();
        acc = init_fwd_slot<0, 18,  2,  50>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 17,  3, 369>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 16,  5, 181>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 15,  6, 216>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 14,  9, 317>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 13, 10, 288>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 12, 11, 109>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 11, 12,  17>(app, z, fwd, acc);
        acc = init_fwd_slot<0, 10, 13, 357>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  9, 15, 215>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  8, 16, 106>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  7, 18, 242>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  6, 19, 180>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  5, 20, 330>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  4, 21, 346>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  3, 22,   1>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  2, 23,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  1,  1,  19>(app, z, fwd, acc);
        acc = init_fwd_slot<0,  0,  0, 307>(app, z, fwd, acc);

        acc = box_plus_identity();
        acc = init_fwd_slot<1, 18,  2,  76>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 17,  3,  73>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 16,  4, 288>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 15,  5, 144>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 14,  7, 331>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 13,  8, 331>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 12,  9, 178>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 11, 11, 295>(app, z, fwd, acc);
        acc = init_fwd_slot<1, 10, 12, 342>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  9, 14, 217>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  8, 15,  99>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  7, 16, 354>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  6, 17, 114>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  5, 19, 331>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  4, 21, 112>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  3, 22,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  2, 23,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  1, 24,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<1,  0,  0,  76>(app, z, fwd, acc);

        acc = box_plus_identity();
        acc = init_fwd_slot<2, 18,  2, 328>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 17,  4, 332>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 16,  5, 256>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 15,  6, 161>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 14,  7, 267>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 13,  8, 160>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 12,  9,  63>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 11, 10, 129>(app, z, fwd, acc);
        acc = init_fwd_slot<2, 10, 13, 200>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  9, 14,  88>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  8, 15,  53>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  7, 17, 131>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  6, 18, 240>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  5, 19, 205>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  4, 20,  13>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  3, 24,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  2, 25,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  1,  1, 250>(app, z, fwd, acc);
        acc = init_fwd_slot<2,  0,  0, 205>(app, z, fwd, acc);

        acc = box_plus_identity();
        acc = init_fwd_slot<3, 18,  3,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 17,  4, 275>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 16,  6, 199>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 15,  7, 153>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 14,  8,  56>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 13, 10, 132>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 12, 11, 305>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 11, 12, 231>(app, z, fwd, acc);
        acc = init_fwd_slot<3, 10, 13, 341>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  9, 14, 212>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  8, 16, 304>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  7, 17, 300>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  6, 18, 271>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  5, 20,  39>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  4, 21, 357>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  3, 22,   1>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  2, 25,   0>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  1,  1,  87>(app, z, fwd, acc);
        acc = init_fwd_slot<3,  0,  0, 276>(app, z, fwd, acc);
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, class TFwd>
    __device__ __forceinline__
    word_t rebuild_fwd_slot(TFwd& fwd, word_t acc)
    {
        const word_t staged_v2c = fwd_get<ROW, POS>(fwd);
        fwd_set<ROW, POS>(fwd, acc);
        return box_pair(acc, staged_v2c);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__
    void rebuild_fwd_from_staged_v2c_p4(TFwd& fwd)
    {
        word_t acc;

        acc = box_plus_identity();
        acc = rebuild_fwd_slot<0, 18>(fwd, acc);
        acc = rebuild_fwd_slot<0, 17>(fwd, acc);
        acc = rebuild_fwd_slot<0, 16>(fwd, acc);
        acc = rebuild_fwd_slot<0, 15>(fwd, acc);
        acc = rebuild_fwd_slot<0, 14>(fwd, acc);
        acc = rebuild_fwd_slot<0, 13>(fwd, acc);
        acc = rebuild_fwd_slot<0, 12>(fwd, acc);
        acc = rebuild_fwd_slot<0, 11>(fwd, acc);
        acc = rebuild_fwd_slot<0, 10>(fwd, acc);
        acc = rebuild_fwd_slot<0,  9>(fwd, acc);
        acc = rebuild_fwd_slot<0,  8>(fwd, acc);
        acc = rebuild_fwd_slot<0,  7>(fwd, acc);
        acc = rebuild_fwd_slot<0,  6>(fwd, acc);
        acc = rebuild_fwd_slot<0,  5>(fwd, acc);
        acc = rebuild_fwd_slot<0,  4>(fwd, acc);
        acc = rebuild_fwd_slot<0,  3>(fwd, acc);
        acc = rebuild_fwd_slot<0,  2>(fwd, acc);
        acc = rebuild_fwd_slot<0,  1>(fwd, acc);
        acc = rebuild_fwd_slot<0,  0>(fwd, acc);

        acc = box_plus_identity();
        acc = rebuild_fwd_slot<1, 18>(fwd, acc);
        acc = rebuild_fwd_slot<1, 17>(fwd, acc);
        acc = rebuild_fwd_slot<1, 16>(fwd, acc);
        acc = rebuild_fwd_slot<1, 15>(fwd, acc);
        acc = rebuild_fwd_slot<1, 14>(fwd, acc);
        acc = rebuild_fwd_slot<1, 13>(fwd, acc);
        acc = rebuild_fwd_slot<1, 12>(fwd, acc);
        acc = rebuild_fwd_slot<1, 11>(fwd, acc);
        acc = rebuild_fwd_slot<1, 10>(fwd, acc);
        acc = rebuild_fwd_slot<1,  9>(fwd, acc);
        acc = rebuild_fwd_slot<1,  8>(fwd, acc);
        acc = rebuild_fwd_slot<1,  7>(fwd, acc);
        acc = rebuild_fwd_slot<1,  6>(fwd, acc);
        acc = rebuild_fwd_slot<1,  5>(fwd, acc);
        acc = rebuild_fwd_slot<1,  4>(fwd, acc);
        acc = rebuild_fwd_slot<1,  3>(fwd, acc);
        acc = rebuild_fwd_slot<1,  2>(fwd, acc);
        acc = rebuild_fwd_slot<1,  1>(fwd, acc);
        acc = rebuild_fwd_slot<1,  0>(fwd, acc);

        acc = box_plus_identity();
        acc = rebuild_fwd_slot<2, 18>(fwd, acc);
        acc = rebuild_fwd_slot<2, 17>(fwd, acc);
        acc = rebuild_fwd_slot<2, 16>(fwd, acc);
        acc = rebuild_fwd_slot<2, 15>(fwd, acc);
        acc = rebuild_fwd_slot<2, 14>(fwd, acc);
        acc = rebuild_fwd_slot<2, 13>(fwd, acc);
        acc = rebuild_fwd_slot<2, 12>(fwd, acc);
        acc = rebuild_fwd_slot<2, 11>(fwd, acc);
        acc = rebuild_fwd_slot<2, 10>(fwd, acc);
        acc = rebuild_fwd_slot<2,  9>(fwd, acc);
        acc = rebuild_fwd_slot<2,  8>(fwd, acc);
        acc = rebuild_fwd_slot<2,  7>(fwd, acc);
        acc = rebuild_fwd_slot<2,  6>(fwd, acc);
        acc = rebuild_fwd_slot<2,  5>(fwd, acc);
        acc = rebuild_fwd_slot<2,  4>(fwd, acc);
        acc = rebuild_fwd_slot<2,  3>(fwd, acc);
        acc = rebuild_fwd_slot<2,  2>(fwd, acc);
        acc = rebuild_fwd_slot<2,  1>(fwd, acc);
        acc = rebuild_fwd_slot<2,  0>(fwd, acc);

        acc = box_plus_identity();
        acc = rebuild_fwd_slot<3, 18>(fwd, acc);
        acc = rebuild_fwd_slot<3, 17>(fwd, acc);
        acc = rebuild_fwd_slot<3, 16>(fwd, acc);
        acc = rebuild_fwd_slot<3, 15>(fwd, acc);
        acc = rebuild_fwd_slot<3, 14>(fwd, acc);
        acc = rebuild_fwd_slot<3, 13>(fwd, acc);
        acc = rebuild_fwd_slot<3, 12>(fwd, acc);
        acc = rebuild_fwd_slot<3, 11>(fwd, acc);
        acc = rebuild_fwd_slot<3, 10>(fwd, acc);
        acc = rebuild_fwd_slot<3,  9>(fwd, acc);
        acc = rebuild_fwd_slot<3,  8>(fwd, acc);
        acc = rebuild_fwd_slot<3,  7>(fwd, acc);
        acc = rebuild_fwd_slot<3,  6>(fwd, acc);
        acc = rebuild_fwd_slot<3,  5>(fwd, acc);
        acc = rebuild_fwd_slot<3,  4>(fwd, acc);
        acc = rebuild_fwd_slot<3,  3>(fwd, acc);
        acc = rebuild_fwd_slot<3,  2>(fwd, acc);
        acc = rebuild_fwd_slot<3,  1>(fwd, acc);
        acc = rebuild_fwd_slot<3,  0>(fwd, acc);
    }

    //------------------------------------------------------------------
    template <int ROW, int LAST_POS, class TFwd>
    __device__ __forceinline__
    void rebuild_fwd_prefix_row(TFwd& fwd, word_t tail_suffix)
    {
        word_t acc = tail_suffix;
        if constexpr (LAST_POS >= 18) { acc = rebuild_fwd_slot<ROW, 18>(fwd, acc); }
        if constexpr (LAST_POS >= 17) { acc = rebuild_fwd_slot<ROW, 17>(fwd, acc); }
        if constexpr (LAST_POS >= 16) { acc = rebuild_fwd_slot<ROW, 16>(fwd, acc); }
        if constexpr (LAST_POS >= 15) { acc = rebuild_fwd_slot<ROW, 15>(fwd, acc); }
        if constexpr (LAST_POS >= 14) { acc = rebuild_fwd_slot<ROW, 14>(fwd, acc); }
        if constexpr (LAST_POS >= 13) { acc = rebuild_fwd_slot<ROW, 13>(fwd, acc); }
        if constexpr (LAST_POS >= 12) { acc = rebuild_fwd_slot<ROW, 12>(fwd, acc); }
        if constexpr (LAST_POS >= 11) { acc = rebuild_fwd_slot<ROW, 11>(fwd, acc); }
        if constexpr (LAST_POS >= 10) { acc = rebuild_fwd_slot<ROW, 10>(fwd, acc); }
        if constexpr (LAST_POS >=  9) { acc = rebuild_fwd_slot<ROW,  9>(fwd, acc); }
        if constexpr (LAST_POS >=  8) { acc = rebuild_fwd_slot<ROW,  8>(fwd, acc); }
        if constexpr (LAST_POS >=  7) { acc = rebuild_fwd_slot<ROW,  7>(fwd, acc); }
        if constexpr (LAST_POS >=  6) { acc = rebuild_fwd_slot<ROW,  6>(fwd, acc); }
        if constexpr (LAST_POS >=  5) { acc = rebuild_fwd_slot<ROW,  5>(fwd, acc); }
        if constexpr (LAST_POS >=  4) { acc = rebuild_fwd_slot<ROW,  4>(fwd, acc); }
        if constexpr (LAST_POS >=  3) { acc = rebuild_fwd_slot<ROW,  3>(fwd, acc); }
        if constexpr (LAST_POS >=  2) { acc = rebuild_fwd_slot<ROW,  2>(fwd, acc); }
        if constexpr (LAST_POS >=  1) { acc = rebuild_fwd_slot<ROW,  1>(fwd, acc); }
        if constexpr (LAST_POS >=  0) { acc = rebuild_fwd_slot<ROW,  0>(fwd, acc); }
    }

    //------------------------------------------------------------------
    template <int LAST0, int LAST1, int LAST2, int LAST3, class TFwd>
    __device__ __forceinline__
    void rebuild_fwd_visited_prefix_p4(TFwd&  fwd,
                                       word_t tail0,
                                       word_t tail1,
                                       word_t tail2,
                                       word_t tail3)
    {
        rebuild_fwd_prefix_row<0, LAST0>(fwd, tail0);
        rebuild_fwd_prefix_row<1, LAST1>(fwd, tail1);
        rebuild_fwd_prefix_row<2, LAST2>(fwd, tail2);
        rebuild_fwd_prefix_row<3, LAST3>(fwd, tail3);
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int SHIFT, class TFwd, class TBwd>
    __device__ __forceinline__
    word_t compute_local_c2v_scaled(word_t* local_c2v,
                                    int     z,
                                    __half2 norm,
                                    TFwd&   fwd,
                                    TBwd&   bwd)
    {
        word_t raw_c2v    = box_pair(fwd_get<ROW, POS>(fwd), bwd[ROW]);
        word_t c2v_scaled = h2_mul_norm(raw_c2v, norm);
        const int u       = wrap_z_shift(z, SHIFT);
        local_c2v[(ROW * ALGO103_Z) + u] = c2v_scaled;
        return c2v_scaled;
    }

    //------------------------------------------------------------------
    template <int ROW>
    __device__ __forceinline__
    void add_local_fixed(word_t& app_v, word_t* local_c2v, int z)
    {
        if constexpr (ROW >= 0)
        {
            app_v = h2_add(app_v, local_c2v[(ROW * ALGO103_Z) + z]);
        }
    }

    //------------------------------------------------------------------
    template <int ROW>
    __device__ __forceinline__
    void add_local_guarded(word_t& app_v, word_t* local_c2v, int z, int num_rows)
    {
        if constexpr (ROW >= 0)
        {
            if(num_rows > ROW)
            {
                app_v = h2_add(app_v, local_c2v[(ROW * ALGO103_Z) + z]);
            }
        }
    }

    //------------------------------------------------------------------
    template <int V, int ROW, int POS, int SHIFT, class TFwd, class TBwd>
    __device__ __forceinline__
    void publish_v2c(word_t c2v_scaled,
                     word_t* app,
                     int     z,
                     TFwd&   fwd,
                     TBwd&   bwd)
    {
        if constexpr (ROW >= 0)
        {
            const int u = wrap_z_shift(z, SHIFT);
            word_t app_check_domain = app[(V * ALGO103_Z) + u];
            word_t v2c = h2_sub(app_check_domain, c2v_scaled);
            fwd_set<ROW, POS>(fwd, v2c);
            bwd[ROW] = box_pair(bwd[ROW], v2c);
        }
    }

    //------------------------------------------------------------------
    template <int V, int DEG,
              int R0, int P0, int S0,
              int R1 = -1, int P1 = 0, int S1 = 0,
              int R2 = -1, int P2 = 0, int S2 = 0,
              int R3 = -1, int P3 = 0, int S3 = 0,
              int R4 = -1, int P4 = 0, int S4 = 0,
              int R5 = -1, int P5 = 0, int S5 = 0,
              int R6 = -1, int P6 = 0, int S6 = 0,
              class TFwd>
    __device__ __forceinline__
    void process_clm_fixed(word_t* app,
                           word_t* local_c2v,
                           int     z,
                           __half2 norm,
                           TFwd&   fwd,
                           word_t (&bwd)[ALGO103_PARITY],
                           const ldpc_dec_loader_params<__half2>& llr_params)
    {
        word_t lc0 = compute_local_c2v_scaled<R0, P0, S0>(local_c2v, z, norm, fwd, bwd);
        word_t lc1;
        word_t lc2;
        word_t lc3;
        word_t lc4;
        word_t lc5;
        word_t lc6;
        if constexpr (DEG >= 2) { lc1 = compute_local_c2v_scaled<R1, P1, S1>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 3) { lc2 = compute_local_c2v_scaled<R2, P2, S2>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 4) { lc3 = compute_local_c2v_scaled<R3, P3, S3>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 5) { lc4 = compute_local_c2v_scaled<R4, P4, S4>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 6) { lc5 = compute_local_c2v_scaled<R5, P5, S5>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 7) { lc6 = compute_local_c2v_scaled<R6, P6, S6>(local_c2v, z, norm, fwd, bwd); }
        __syncthreads();

        word_t app_v = load_channel_app_base<V>(llr_params, z);
        add_local_fixed<R0>(app_v, local_c2v, z);
        if constexpr (DEG >= 2) { add_local_fixed<R1>(app_v, local_c2v, z); }
        if constexpr (DEG >= 3) { add_local_fixed<R2>(app_v, local_c2v, z); }
        if constexpr (DEG >= 4) { add_local_fixed<R3>(app_v, local_c2v, z); }
        if constexpr (DEG >= 5) { add_local_fixed<R4>(app_v, local_c2v, z); }
        if constexpr (DEG >= 6) { add_local_fixed<R5>(app_v, local_c2v, z); }
        if constexpr (DEG >= 7) { add_local_fixed<R6>(app_v, local_c2v, z); }
        app[(V * ALGO103_Z) + z] = app_v;
        __syncthreads();

        publish_v2c<V, R0, P0, S0>(lc0, app, z, fwd, bwd);
        if constexpr (DEG >= 2) { publish_v2c<V, R1, P1, S1>(lc1, app, z, fwd, bwd); }
        if constexpr (DEG >= 3) { publish_v2c<V, R2, P2, S2>(lc2, app, z, fwd, bwd); }
        if constexpr (DEG >= 4) { publish_v2c<V, R3, P3, S3>(lc3, app, z, fwd, bwd); }
        if constexpr (DEG >= 5) { publish_v2c<V, R4, P4, S4>(lc4, app, z, fwd, bwd); }
        if constexpr (DEG >= 6) { publish_v2c<V, R5, P5, S5>(lc5, app, z, fwd, bwd); }
        if constexpr (DEG >= 7) { publish_v2c<V, R6, P6, S6>(lc6, app, z, fwd, bwd); }
    }

    //------------------------------------------------------------------
    template <int V, int DEG,
              int R0, int P0, int S0,
              int R1 = -1, int P1 = 0, int S1 = 0,
              int R2 = -1, int P2 = 0, int S2 = 0,
              int R3 = -1, int P3 = 0, int S3 = 0,
              int R4 = -1, int P4 = 0, int S4 = 0,
              int R5 = -1, int P5 = 0, int S5 = 0,
              int R6 = -1, int P6 = 0, int S6 = 0>
    __device__ __forceinline__
    void process_clm_guarded(word_t* app,
                             word_t* local_c2v,
                             int     num_rows,
                             int     num_vars,
                             int     z,
                             __half2 norm,
                             word_t (&fwd)[ALGO103_MAX_ROWS]
                                           [ALGO103_MAX_ROW_DEGREE],
                             word_t (&bwd)[ALGO103_MAX_ROWS])
    {
        if(V >= num_vars)
        {
            return;
        }

        word_t lc0;
        word_t lc1;
        word_t lc2;
        word_t lc3;
        word_t lc4;
        word_t lc5;
        word_t lc6;

        if constexpr (R0 >= 0) { if(num_rows > R0) { lc0 = compute_local_c2v_scaled<R0, P0, S0>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 2) { if(num_rows > R1) { lc1 = compute_local_c2v_scaled<R1, P1, S1>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 3) { if(num_rows > R2) { lc2 = compute_local_c2v_scaled<R2, P2, S2>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 4) { if(num_rows > R3) { lc3 = compute_local_c2v_scaled<R3, P3, S3>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 5) { if(num_rows > R4) { lc4 = compute_local_c2v_scaled<R4, P4, S4>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 6) { if(num_rows > R5) { lc5 = compute_local_c2v_scaled<R5, P5, S5>(local_c2v, z, norm, fwd, bwd); } }
        if constexpr (DEG >= 7) { if(num_rows > R6) { lc6 = compute_local_c2v_scaled<R6, P6, S6>(local_c2v, z, norm, fwd, bwd); } }
        __syncthreads();

        word_t app_v = app[(V * ALGO103_Z) + z];
        add_local_guarded<R0>(app_v, local_c2v, z, num_rows);
        if constexpr (DEG >= 2) { add_local_guarded<R1>(app_v, local_c2v, z, num_rows); }
        if constexpr (DEG >= 3) { add_local_guarded<R2>(app_v, local_c2v, z, num_rows); }
        if constexpr (DEG >= 4) { add_local_guarded<R3>(app_v, local_c2v, z, num_rows); }
        if constexpr (DEG >= 5) { add_local_guarded<R4>(app_v, local_c2v, z, num_rows); }
        if constexpr (DEG >= 6) { add_local_guarded<R5>(app_v, local_c2v, z, num_rows); }
        if constexpr (DEG >= 7) { add_local_guarded<R6>(app_v, local_c2v, z, num_rows); }
        app[(V * ALGO103_Z) + z] = app_v;
        __syncthreads();

        if constexpr (R0 >= 0) { if(num_rows > R0) { publish_v2c<V, R0, P0, S0>(lc0, app, z, fwd, bwd); } }
        if constexpr (DEG >= 2) { if(num_rows > R1) { publish_v2c<V, R1, P1, S1>(lc1, app, z, fwd, bwd); } }
        if constexpr (DEG >= 3) { if(num_rows > R2) { publish_v2c<V, R2, P2, S2>(lc2, app, z, fwd, bwd); } }
        if constexpr (DEG >= 4) { if(num_rows > R3) { publish_v2c<V, R3, P3, S3>(lc3, app, z, fwd, bwd); } }
        if constexpr (DEG >= 5) { if(num_rows > R4) { publish_v2c<V, R4, P4, S4>(lc4, app, z, fwd, bwd); } }
        if constexpr (DEG >= 6) { if(num_rows > R5) { publish_v2c<V, R5, P5, S5>(lc5, app, z, fwd, bwd); } }
        if constexpr (DEG >= 7) { if(num_rows > R6) { publish_v2c<V, R6, P6, S6>(lc6, app, z, fwd, bwd); } }
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    void init_bwd_p4(word_t (&bwd)[ALGO103_PARITY])
    {
        bwd[0] = box_plus_identity();
        bwd[1] = box_plus_identity();
        bwd[2] = box_plus_identity();
        bwd[3] = box_plus_identity();
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__ __forceinline__
    void process_algo103_first2(word_t* app,
                               word_t* local_c2v,
                               int     z,
                               __half2 norm,
                               TFwd&   fwd,
                               word_t (&bwd)[ALGO103_PARITY],
                               const ldpc_dec_loader_params<__half2>& llr_params)
    {
        process_clm_fixed< 0, 4, 0,  0, 307, 1,  0,  76, 2,  0, 205, 3,  0, 276>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 1, 3, 0,  1,  19, 2,  1, 250, 3,  1,  87>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__ __forceinline__
    void process_algo103_first8(word_t* app,
                               word_t* local_c2v,
                               int     z,
                               __half2 norm,
                               TFwd&   fwd,
                               word_t (&bwd)[ALGO103_PARITY],
                               const ldpc_dec_loader_params<__half2>& llr_params)
    {
        process_algo103_first2(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<25, 2, 2,  2,   0, 3,  2,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<24, 2, 1,  1,   0, 2,  3,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<23, 2, 0,  2,   0, 1,  2,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<22, 3, 0,  3,   1, 1,  3,   0, 3,  3,   1>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<21, 3, 0,  4, 346, 1,  4, 112, 3,  4, 357>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<20, 3, 0,  5, 330, 2,  4,  13, 3,  5,  39>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__ __forceinline__
    void process_algo103_first14(word_t* app,
                                word_t* local_c2v,
                                int     z,
                                __half2 norm,
                                TFwd&   fwd,
                                word_t (&bwd)[ALGO103_PARITY],
                                const ldpc_dec_loader_params<__half2>& llr_params)
    {
        process_algo103_first8(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<19, 3, 0,  6, 180, 1,  5, 331, 2,  5, 205>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<18, 3, 0,  7, 242, 2,  6, 240, 3,  6, 271>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<17, 3, 1,  6, 114, 2,  7, 131, 3,  7, 300>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<16, 3, 0,  8, 106, 1,  7, 354, 3,  8, 304>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<15, 3, 0,  9, 215, 1,  8,  99, 2,  8,  53>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<14, 3, 1,  9, 217, 2,  9,  88, 3,  9, 212>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__ __forceinline__
    void process_algo103_full(word_t* app,
                             word_t* local_c2v,
                             int     z,
                             __half2 norm,
                             TFwd&   fwd,
                             word_t (&bwd)[ALGO103_PARITY],
                             const ldpc_dec_loader_params<__half2>& llr_params)
    {
        process_algo103_first14(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<13, 3, 0, 10, 357, 2, 10, 200, 3, 10, 341>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<12, 3, 0, 11,  17, 1, 10, 342, 3, 11, 231>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<11, 3, 0, 12, 109, 1, 11, 295, 3, 12, 305>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed<10, 3, 0, 13, 288, 2, 11, 129, 3, 13, 132>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 9, 3, 0, 14, 317, 1, 12, 178, 2, 12,  63>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 8, 3, 1, 13, 331, 2, 13, 160, 3, 14,  56>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 7, 3, 1, 14, 331, 2, 14, 267, 3, 15, 153>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 6, 3, 0, 15, 216, 2, 15, 161, 3, 16, 199>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 5, 3, 0, 16, 181, 1, 15, 144, 2, 16, 256>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 4, 3, 1, 16, 288, 2, 17, 332, 3, 17, 275>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 3, 3, 0, 17, 369, 1, 17,  73, 3, 18,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed< 2, 3, 0, 18,  50, 1, 18,  76, 2, 18, 328>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__
    void do_algo103_full_iteration_p4(char* smem,
                                     __half2 norm,
                                     TFwd&   fwd,
                                     const ldpc_dec_loader_params<__half2>& llr_params)
    {
        word_t* app       = app_smem(smem);
        word_t* local_c2v = local_c2v_smem(smem, ALGO103_PARITY);
        const int z       = threadIdx.x;

        word_t bwd[ALGO103_PARITY];
        init_bwd_p4(bwd);
        process_algo103_full(app, local_c2v, z, norm, fwd, bwd, llr_params);

        rebuild_fwd_from_staged_v2c_p4(fwd);
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__
    void do_algo103_iteration_p4_scheduled(char*   smem,
                                          __half2 norm,
                                          TFwd&   fwd,
                                          int32_t sub_iter,
                                          const ldpc_dec_loader_params<__half2>& llr_params)
    {
        word_t* app       = app_smem(smem);
        word_t* local_c2v = local_c2v_smem(smem, ALGO103_PARITY);
        const int z       = threadIdx.x;

        word_t bwd[ALGO103_PARITY];
        init_bwd_p4(bwd);

        switch(sub_iter & 3)
        {
        case 0:
        {
            const word_t tail0 = fwd_get<0, 1>(fwd);
            const word_t tail1 = fwd_get<1, 0>(fwd);
            const word_t tail2 = fwd_get<2, 1>(fwd);
            const word_t tail3 = fwd_get<3, 1>(fwd);
            process_algo103_first2(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4<1, 0, 1, 1>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        case 1:
        {
            const word_t tail0 = fwd_get<0, 5>(fwd);
            const word_t tail1 = fwd_get<1, 4>(fwd);
            const word_t tail2 = fwd_get<2, 4>(fwd);
            const word_t tail3 = fwd_get<3, 5>(fwd);
            process_algo103_first8(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4<5, 4, 4, 5>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        case 2:
        {
            const word_t tail0 = fwd_get<0, 9>(fwd);
            const word_t tail1 = fwd_get<1, 9>(fwd);
            const word_t tail2 = fwd_get<2, 9>(fwd);
            const word_t tail3 = fwd_get<3, 9>(fwd);
            process_algo103_first14(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4<9, 9, 9, 9>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        default:
            process_algo103_full(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_from_staged_v2c_p4(fwd);
            break;
        }
    }

    //------------------------------------------------------------------
    template <class TFwd>
    __device__
    void do_algo103_iteration_p4(char*   smem,
                                __half2 norm,
                                TFwd&   fwd,
                                const ldpc_dec_loader_params<__half2>& llr_params)
    {
        do_algo103_iteration_p4_scheduled(smem, norm, fwd, 0, llr_params);
        do_algo103_iteration_p4_scheduled(smem, norm, fwd, 1, llr_params);
        do_algo103_iteration_p4_scheduled(smem, norm, fwd, 2, llr_params);
        do_algo103_iteration_p4_scheduled(smem, norm, fwd, 3, llr_params);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half half_add(__half a, __half b)
    {
        return __hadd(a, b);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half half_sub(__half a, __half b)
    {
        return __hsub(a, b);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half half_mul_norm(__half a, __half norm)
    {
        return __hmul(a, norm);
    }

    //------------------------------------------------------------------
    template <int V>
    __device__ __forceinline__
    __half load_channel_app_base_scalar(const ldpc_dec_loader_params<__half>& params, int z)
    {
        const int idx = (V * ALGO103_Z) + z;
        const __half* input = static_cast<const __half*>(params.src_gmem);
        const __half v = input[idx];
        return __low2half(clamp_signed(__halves2half2(v, v), __float2half2_rn(params.clamp_value)));
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int NUM_ROWS>
    __device__ __forceinline__
    __half fwd_get_scalar(const __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE])
    {
        return fwd[ROW][POS];
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int NUM_ROWS>
    __device__ __forceinline__
    void fwd_set_scalar(__half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE], __half v)
    {
        fwd[ROW][POS] = v;
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int V, int SHIFT, int NUM_ROWS>
    __device__ __forceinline__
    __half init_fwd_slot_scalar(__half* app,
                                int     z,
                                __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                __half acc)
    {
        fwd_set_scalar<ROW, POS>(fwd, acc);
        const int u = wrap_z_shift(z, SHIFT);
        return half_box_pair(acc, app[(V * ALGO103_Z) + u]);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__
    void init_fwd_from_app_p4_scalar(char* smem,
                                     __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE])
    {
        __half* app = app_smem_ref(smem);
        const int z = threadIdx.x;
        __half acc;

        acc = half_box_plus_identity();
        acc = init_fwd_slot_scalar<0, 18,  2,  50>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 17,  3, 369>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 16,  5, 181>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 15,  6, 216>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 14,  9, 317>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 13, 10, 288>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 12, 11, 109>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 11, 12,  17>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0, 10, 13, 357>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  9, 15, 215>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  8, 16, 106>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  7, 18, 242>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  6, 19, 180>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  5, 20, 330>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  4, 21, 346>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  3, 22,   1>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  2, 23,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  1,  1,  19>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<0,  0,  0, 307>(app, z, fwd, acc);

        acc = half_box_plus_identity();
        acc = init_fwd_slot_scalar<1, 18,  2,  76>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 17,  3,  73>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 16,  4, 288>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 15,  5, 144>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 14,  7, 331>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 13,  8, 331>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 12,  9, 178>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 11, 11, 295>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1, 10, 12, 342>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  9, 14, 217>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  8, 15,  99>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  7, 16, 354>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  6, 17, 114>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  5, 19, 331>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  4, 21, 112>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  3, 22,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  2, 23,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  1, 24,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<1,  0,  0,  76>(app, z, fwd, acc);

        acc = half_box_plus_identity();
        acc = init_fwd_slot_scalar<2, 18,  2, 328>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 17,  4, 332>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 16,  5, 256>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 15,  6, 161>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 14,  7, 267>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 13,  8, 160>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 12,  9,  63>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 11, 10, 129>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2, 10, 13, 200>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  9, 14,  88>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  8, 15,  53>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  7, 17, 131>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  6, 18, 240>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  5, 19, 205>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  4, 20,  13>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  3, 24,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  2, 25,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  1,  1, 250>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<2,  0,  0, 205>(app, z, fwd, acc);

        acc = half_box_plus_identity();
        acc = init_fwd_slot_scalar<3, 18,  3,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 17,  4, 275>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 16,  6, 199>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 15,  7, 153>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 14,  8,  56>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 13, 10, 132>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 12, 11, 305>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 11, 12, 231>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3, 10, 13, 341>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  9, 14, 212>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  8, 16, 304>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  7, 17, 300>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  6, 18, 271>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  5, 20,  39>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  4, 21, 357>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  3, 22,   1>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  2, 25,   0>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  1,  1,  87>(app, z, fwd, acc);
        acc = init_fwd_slot_scalar<3,  0,  0, 276>(app, z, fwd, acc);
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int NUM_ROWS>
    __device__ __forceinline__
    __half rebuild_fwd_slot_scalar(__half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                   __half acc)
    {
        const __half staged_v2c = fwd_get_scalar<ROW, POS>(fwd);
        fwd_set_scalar<ROW, POS>(fwd, acc);
        return half_box_pair(acc, staged_v2c);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__
    void rebuild_fwd_from_staged_v2c_p4_scalar(__half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE])
    {
        __half acc;

        acc = half_box_plus_identity();
        acc = rebuild_fwd_slot_scalar<0, 18>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 17>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 16>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 15>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 14>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 13>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 12>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 11>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0, 10>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  9>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  8>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  7>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  6>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  5>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  4>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  3>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  2>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  1>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<0,  0>(fwd, acc);

        acc = half_box_plus_identity();
        acc = rebuild_fwd_slot_scalar<1, 18>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 17>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 16>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 15>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 14>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 13>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 12>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 11>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1, 10>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  9>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  8>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  7>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  6>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  5>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  4>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  3>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  2>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  1>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<1,  0>(fwd, acc);

        acc = half_box_plus_identity();
        acc = rebuild_fwd_slot_scalar<2, 18>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 17>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 16>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 15>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 14>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 13>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 12>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 11>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2, 10>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  9>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  8>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  7>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  6>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  5>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  4>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  3>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  2>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  1>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<2,  0>(fwd, acc);

        acc = half_box_plus_identity();
        acc = rebuild_fwd_slot_scalar<3, 18>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 17>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 16>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 15>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 14>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 13>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 12>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 11>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3, 10>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  9>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  8>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  7>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  6>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  5>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  4>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  3>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  2>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  1>(fwd, acc);
        acc = rebuild_fwd_slot_scalar<3,  0>(fwd, acc);
    }

    //------------------------------------------------------------------
    template <int ROW, int LAST_POS, int NUM_ROWS>
    __device__ __forceinline__
    void rebuild_fwd_prefix_row_scalar(__half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                       __half tail_suffix)
    {
        __half acc = tail_suffix;
        if constexpr (LAST_POS >= 18) { acc = rebuild_fwd_slot_scalar<ROW, 18>(fwd, acc); }
        if constexpr (LAST_POS >= 17) { acc = rebuild_fwd_slot_scalar<ROW, 17>(fwd, acc); }
        if constexpr (LAST_POS >= 16) { acc = rebuild_fwd_slot_scalar<ROW, 16>(fwd, acc); }
        if constexpr (LAST_POS >= 15) { acc = rebuild_fwd_slot_scalar<ROW, 15>(fwd, acc); }
        if constexpr (LAST_POS >= 14) { acc = rebuild_fwd_slot_scalar<ROW, 14>(fwd, acc); }
        if constexpr (LAST_POS >= 13) { acc = rebuild_fwd_slot_scalar<ROW, 13>(fwd, acc); }
        if constexpr (LAST_POS >= 12) { acc = rebuild_fwd_slot_scalar<ROW, 12>(fwd, acc); }
        if constexpr (LAST_POS >= 11) { acc = rebuild_fwd_slot_scalar<ROW, 11>(fwd, acc); }
        if constexpr (LAST_POS >= 10) { acc = rebuild_fwd_slot_scalar<ROW, 10>(fwd, acc); }
        if constexpr (LAST_POS >=  9) { acc = rebuild_fwd_slot_scalar<ROW,  9>(fwd, acc); }
        if constexpr (LAST_POS >=  8) { acc = rebuild_fwd_slot_scalar<ROW,  8>(fwd, acc); }
        if constexpr (LAST_POS >=  7) { acc = rebuild_fwd_slot_scalar<ROW,  7>(fwd, acc); }
        if constexpr (LAST_POS >=  6) { acc = rebuild_fwd_slot_scalar<ROW,  6>(fwd, acc); }
        if constexpr (LAST_POS >=  5) { acc = rebuild_fwd_slot_scalar<ROW,  5>(fwd, acc); }
        if constexpr (LAST_POS >=  4) { acc = rebuild_fwd_slot_scalar<ROW,  4>(fwd, acc); }
        if constexpr (LAST_POS >=  3) { acc = rebuild_fwd_slot_scalar<ROW,  3>(fwd, acc); }
        if constexpr (LAST_POS >=  2) { acc = rebuild_fwd_slot_scalar<ROW,  2>(fwd, acc); }
        if constexpr (LAST_POS >=  1) { acc = rebuild_fwd_slot_scalar<ROW,  1>(fwd, acc); }
        if constexpr (LAST_POS >=  0) { acc = rebuild_fwd_slot_scalar<ROW,  0>(fwd, acc); }
    }

    //------------------------------------------------------------------
    template <int LAST0, int LAST1, int LAST2, int LAST3, int NUM_ROWS>
    __device__ __forceinline__
    void rebuild_fwd_visited_prefix_p4_scalar(__half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                              __half tail0,
                                              __half tail1,
                                              __half tail2,
                                              __half tail3)
    {
        rebuild_fwd_prefix_row_scalar<0, LAST0>(fwd, tail0);
        rebuild_fwd_prefix_row_scalar<1, LAST1>(fwd, tail1);
        rebuild_fwd_prefix_row_scalar<2, LAST2>(fwd, tail2);
        rebuild_fwd_prefix_row_scalar<3, LAST3>(fwd, tail3);
    }

    //------------------------------------------------------------------
    template <int ROW, int POS, int SHIFT, int NUM_ROWS>
    __device__ __forceinline__
    __half compute_local_c2v_scaled_scalar(__half* local_c2v,
                                           int     z,
                                           __half  norm,
                                           __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                           __half (&bwd)[ALGO103_PARITY])
    {
        __half raw_c2v    = half_box_pair(fwd_get_scalar<ROW, POS>(fwd), bwd[ROW]);
        __half c2v_scaled = half_mul_norm(raw_c2v, norm);
        const int u       = wrap_z_shift(z, SHIFT);
        local_c2v[(ROW * ALGO103_Z) + u] = c2v_scaled;
        return c2v_scaled;
    }

    //------------------------------------------------------------------
    template <int ROW>
    __device__ __forceinline__
    void add_local_fixed_scalar(__half& app_v, __half* local_c2v, int z)
    {
        if constexpr (ROW >= 0)
        {
            app_v = half_add(app_v, local_c2v[(ROW * ALGO103_Z) + z]);
        }
    }

    //------------------------------------------------------------------
    template <int V, int ROW, int POS, int SHIFT, int NUM_ROWS>
    __device__ __forceinline__
    void publish_v2c_scalar(__half c2v_scaled,
                            __half* app,
                            int     z,
                            __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                            __half (&bwd)[ALGO103_PARITY])
    {
        if constexpr (ROW >= 0)
        {
            const int u = wrap_z_shift(z, SHIFT);
            __half app_check_domain = app[(V * ALGO103_Z) + u];
            __half v2c = half_sub(app_check_domain, c2v_scaled);
            fwd_set_scalar<ROW, POS>(fwd, v2c);
            bwd[ROW] = half_box_pair(bwd[ROW], v2c);
        }
    }

    //------------------------------------------------------------------
    template <int V, int DEG,
              int R0, int P0, int S0,
              int R1 = -1, int P1 = 0, int S1 = 0,
              int R2 = -1, int P2 = 0, int S2 = 0,
              int R3 = -1, int P3 = 0, int S3 = 0,
              int R4 = -1, int P4 = 0, int S4 = 0,
              int R5 = -1, int P5 = 0, int S5 = 0,
              int R6 = -1, int P6 = 0, int S6 = 0,
              int NUM_ROWS>
    __device__ __forceinline__
    void process_clm_fixed_scalar(__half* app,
                                  __half* local_c2v,
                                  int     z,
                                  __half  norm,
                                  __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                  __half (&bwd)[ALGO103_PARITY],
                                  const ldpc_dec_loader_params<__half>& llr_params)
    {
        __half lc0 = compute_local_c2v_scaled_scalar<R0, P0, S0>(local_c2v, z, norm, fwd, bwd);
        __half lc1;
        __half lc2;
        __half lc3;
        __half lc4;
        __half lc5;
        __half lc6;
        if constexpr (DEG >= 2) { lc1 = compute_local_c2v_scaled_scalar<R1, P1, S1>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 3) { lc2 = compute_local_c2v_scaled_scalar<R2, P2, S2>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 4) { lc3 = compute_local_c2v_scaled_scalar<R3, P3, S3>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 5) { lc4 = compute_local_c2v_scaled_scalar<R4, P4, S4>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 6) { lc5 = compute_local_c2v_scaled_scalar<R5, P5, S5>(local_c2v, z, norm, fwd, bwd); }
        if constexpr (DEG >= 7) { lc6 = compute_local_c2v_scaled_scalar<R6, P6, S6>(local_c2v, z, norm, fwd, bwd); }
        __syncthreads();

        __half app_v = load_channel_app_base_scalar<V>(llr_params, z);
        add_local_fixed_scalar<R0>(app_v, local_c2v, z);
        if constexpr (DEG >= 2) { add_local_fixed_scalar<R1>(app_v, local_c2v, z); }
        if constexpr (DEG >= 3) { add_local_fixed_scalar<R2>(app_v, local_c2v, z); }
        if constexpr (DEG >= 4) { add_local_fixed_scalar<R3>(app_v, local_c2v, z); }
        if constexpr (DEG >= 5) { add_local_fixed_scalar<R4>(app_v, local_c2v, z); }
        if constexpr (DEG >= 6) { add_local_fixed_scalar<R5>(app_v, local_c2v, z); }
        if constexpr (DEG >= 7) { add_local_fixed_scalar<R6>(app_v, local_c2v, z); }
        app[(V * ALGO103_Z) + z] = app_v;
        __syncthreads();

        publish_v2c_scalar<V, R0, P0, S0>(lc0, app, z, fwd, bwd);
        if constexpr (DEG >= 2) { publish_v2c_scalar<V, R1, P1, S1>(lc1, app, z, fwd, bwd); }
        if constexpr (DEG >= 3) { publish_v2c_scalar<V, R2, P2, S2>(lc2, app, z, fwd, bwd); }
        if constexpr (DEG >= 4) { publish_v2c_scalar<V, R3, P3, S3>(lc3, app, z, fwd, bwd); }
        if constexpr (DEG >= 5) { publish_v2c_scalar<V, R4, P4, S4>(lc4, app, z, fwd, bwd); }
        if constexpr (DEG >= 6) { publish_v2c_scalar<V, R5, P5, S5>(lc5, app, z, fwd, bwd); }
        if constexpr (DEG >= 7) { publish_v2c_scalar<V, R6, P6, S6>(lc6, app, z, fwd, bwd); }
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    void init_bwd_p4_scalar(__half (&bwd)[ALGO103_PARITY])
    {
        bwd[0] = half_box_plus_identity();
        bwd[1] = half_box_plus_identity();
        bwd[2] = half_box_plus_identity();
        bwd[3] = half_box_plus_identity();
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__ __forceinline__
    void process_algo103_scalar_first2(__half* app,
                               __half* local_c2v,
                               int     z,
                               __half  norm,
                               __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                               __half (&bwd)[ALGO103_PARITY],
                               const ldpc_dec_loader_params<__half>& llr_params)
    {
        process_clm_fixed_scalar< 0, 4, 0,  0, 307, 1,  0,  76, 2,  0, 205, 3,  0, 276>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 1, 3, 0,  1,  19, 2,  1, 250, 3,  1,  87>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__ __forceinline__
    void process_algo103_scalar_first8(__half* app,
                               __half* local_c2v,
                               int     z,
                               __half  norm,
                               __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                               __half (&bwd)[ALGO103_PARITY],
                               const ldpc_dec_loader_params<__half>& llr_params)
    {
        process_algo103_scalar_first2(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<25, 2, 2,  2,   0, 3,  2,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<24, 2, 1,  1,   0, 2,  3,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<23, 2, 0,  2,   0, 1,  2,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<22, 3, 0,  3,   1, 1,  3,   0, 3,  3,   1>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<21, 3, 0,  4, 346, 1,  4, 112, 3,  4, 357>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<20, 3, 0,  5, 330, 2,  4,  13, 3,  5,  39>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__ __forceinline__
    void process_algo103_scalar_first14(__half* app,
                                __half* local_c2v,
                                int     z,
                                __half  norm,
                                __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                __half (&bwd)[ALGO103_PARITY],
                                const ldpc_dec_loader_params<__half>& llr_params)
    {
        process_algo103_scalar_first8(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<19, 3, 0,  6, 180, 1,  5, 331, 2,  5, 205>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<18, 3, 0,  7, 242, 2,  6, 240, 3,  6, 271>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<17, 3, 1,  6, 114, 2,  7, 131, 3,  7, 300>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<16, 3, 0,  8, 106, 1,  7, 354, 3,  8, 304>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<15, 3, 0,  9, 215, 1,  8,  99, 2,  8,  53>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<14, 3, 1,  9, 217, 2,  9,  88, 3,  9, 212>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__ __forceinline__
    void process_algo103_scalar_full(__half* app,
                             __half* local_c2v,
                             int     z,
                             __half  norm,
                             __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                             __half (&bwd)[ALGO103_PARITY],
                             const ldpc_dec_loader_params<__half>& llr_params)
    {
        process_algo103_scalar_first14(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<13, 3, 0, 10, 357, 2, 10, 200, 3, 10, 341>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<12, 3, 0, 11,  17, 1, 10, 342, 3, 11, 231>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<11, 3, 0, 12, 109, 1, 11, 295, 3, 12, 305>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar<10, 3, 0, 13, 288, 2, 11, 129, 3, 13, 132>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 9, 3, 0, 14, 317, 1, 12, 178, 2, 12,  63>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 8, 3, 1, 13, 331, 2, 13, 160, 3, 14,  56>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 7, 3, 1, 14, 331, 2, 14, 267, 3, 15, 153>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 6, 3, 0, 15, 216, 2, 15, 161, 3, 16, 199>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 5, 3, 0, 16, 181, 1, 15, 144, 2, 16, 256>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 4, 3, 1, 16, 288, 2, 17, 332, 3, 17, 275>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 3, 3, 0, 17, 369, 1, 17,  73, 3, 18,   0>(app, local_c2v, z, norm, fwd, bwd, llr_params);
        process_clm_fixed_scalar< 2, 3, 0, 18,  50, 1, 18,  76, 2, 18, 328>(app, local_c2v, z, norm, fwd, bwd, llr_params);
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__
    void do_algo103_scalar_iteration_p4_scheduled(char* smem,
                                          __half norm,
                                          __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                          int32_t sub_iter,
                                          const ldpc_dec_loader_params<__half>& llr_params)
    {
        __half* app       = app_smem_ref(smem);
        __half* local_c2v = local_c2v_smem_ref(smem, ALGO103_BG, ALGO103_PARITY, ALGO103_Z);
        const int z       = threadIdx.x;

        __half bwd[ALGO103_PARITY];
        init_bwd_p4_scalar(bwd);

        switch(sub_iter & 3)
        {
        case 0:
        {
            const __half tail0 = fwd_get_scalar<0, 1>(fwd);
            const __half tail1 = fwd_get_scalar<1, 0>(fwd);
            const __half tail2 = fwd_get_scalar<2, 1>(fwd);
            const __half tail3 = fwd_get_scalar<3, 1>(fwd);
            process_algo103_scalar_first2(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4_scalar<1, 0, 1, 1>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        case 1:
        {
            const __half tail0 = fwd_get_scalar<0, 5>(fwd);
            const __half tail1 = fwd_get_scalar<1, 4>(fwd);
            const __half tail2 = fwd_get_scalar<2, 4>(fwd);
            const __half tail3 = fwd_get_scalar<3, 5>(fwd);
            process_algo103_scalar_first8(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4_scalar<5, 4, 4, 5>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        case 2:
        {
            const __half tail0 = fwd_get_scalar<0, 9>(fwd);
            const __half tail1 = fwd_get_scalar<1, 9>(fwd);
            const __half tail2 = fwd_get_scalar<2, 9>(fwd);
            const __half tail3 = fwd_get_scalar<3, 9>(fwd);
            process_algo103_scalar_first14(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_visited_prefix_p4_scalar<9, 9, 9, 9>(fwd, tail0, tail1, tail2, tail3);
            break;
        }
        default:
            process_algo103_scalar_full(app, local_c2v, z, norm, fwd, bwd, llr_params);
            rebuild_fwd_from_staged_v2c_p4_scalar(fwd);
            break;
        }
    }

    //------------------------------------------------------------------
    template <int NUM_ROWS>
    __device__
    void do_algo103_scalar_iteration_p4(char* smem,
                                __half norm,
                                __half (&fwd)[NUM_ROWS][ALGO103_MAX_ROW_DEGREE],
                                const ldpc_dec_loader_params<__half>& llr_params)
    {
        do_algo103_scalar_iteration_p4_scheduled(smem, norm, fwd, 0, llr_params);
        do_algo103_scalar_iteration_p4_scheduled(smem, norm, fwd, 1, llr_params);
        do_algo103_scalar_iteration_p4_scheduled(smem, norm, fwd, 2, llr_params);
        do_algo103_scalar_iteration_p4_scheduled(smem, norm, fwd, 3, llr_params);
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    __half2 norm_from_config(const cuphyLDPCDecodeConfigDesc_t& config)
    {
        return half2_from_raw(config.norm.f16x2);
    }

    //------------------------------------------------------------------
    // Per-iteration normalization (alpha) scaler for the layered min-sum check
    // node.  A single constant alpha (== config.norm, the library's 0.79
    // optimum) is a sharp single-point optimum for this rate/lifting size, but
    // BP does not want the same attenuation on every pass -- so alpha is
    // scheduled per BP iteration (see ldpc2_algo103.cu).  This returns
    // base_norm * k as an fp16x2; the per-iteration value is precomputed once
    // per iteration so the per-edge scale (h2_mul_norm) is unchanged.  k is a
    // compile-time constant, so this folds to a single HMUL2 against the runtime
    // base_norm -- no per-edge cost.  (Column-degree scaling -- a distinct alpha
    // for the punctured V0,V1 columns -- was swept in both directions on the
    // slower base and found to have NO headroom: 0.79 is a sharp optimum even
    // restricted to those columns, so only the iteration axis is scheduled.)
    __device__ __forceinline__
    __half2 scale_norm(__half2 base_norm, float k)
    {
        return __hmul2(base_norm, __float2half2_rn(k));
    }

    //==================================================================
    // Layered (row-sequential) half2 min-sum for BG1 / Z=384 / mb=4.
    //
    // Same normalized min-sum check-node math as the row-layered
    // p=4 decoder (per-row forward-backward leave-one-out, sign-min + norm),
    // but the 4 parity rows are visited sequentially.  This replaces the
    // vertical schedule's ~50 per-column barriers with 4 per-iteration
    // barriers and keeps the channel LLRs resident in shared memory
    // (loaded once into 'app') instead of re-reading them from global
    // memory on every column.
    //
    // 'app' (shared) holds the running posterior  = channel + sum_r c2v[r].
    // 'c2v_reg[R][POS]' (per-thread registers) holds the last check->var
    // message that this thread's check-copy z emitted on edge POS of row R
    // (exactly the message store the vertical kernel keeps in 'fwd').  For
    // an edge (V, SHIFT) of row R the var->check message is
    //     v2c = app[V][u] - c2v_reg[R][POS],   u = (z + SHIFT) mod Z,
    // and the posterior is refreshed in place as  app[V][u] = v2c + c2v_new.
    // Each app[V][.] / c2v_reg[.] location is touched by exactly one thread
    // within a row (distinct columns; the shift is a permutation), so no
    // intra-row barrier is needed -- only a barrier between rows so the
    // posterior updates of row R become visible to row R+1.
    //==================================================================
    typedef word_t c2v_store_t[ALGO103_PARITY][ALGO103_MAX_ROW_DEGREE];

    //------------------------------------------------------------------
    // Lane-private posterior registers for the identity-shift (SHIFT==0)
    // degree-2 parity columns 23 / 24 / 25 of BG1 / Z=384 / mb=4.
    //
    // Every edge incident on these columns has circulant shift 0, so thread z
    // exclusively owns app[V*Z + z] for V in {23,24,25} across every row/sweep
    // (no other thread reads or writes those slots).  Holding their posteriors
    // in registers removes the entire shared round-trip for these columns --
    // 12 LDS + 12 STS per sweep -- replacing it with pure in-lane register
    // data flow.  Bit-identical: the value passed through the register is the
    // same the shared scatter/gather would have moved.  Loaded from the
    // (already clamped) channel posterior once before the sweep loop and
    // written back once afterwards for the output stage.
    //------------------------------------------------------------------
    //
    // p22 additionally holds the degree-3 parity column V22 *across the
    // iteration boundary only*.  V22's edges are row0(shift1), row1(shift0),
    // row3(shift1).  Within an iteration its posterior must still transit shared
    // (row1's shift-0 access is owned by a different lane than rows 0/3), but the
    // last touch of an iteration (row3, shift1) and the first touch of the next
    // (row0, shift1) are the SAME lane on the SAME slot (z+1), with no other row
    // touching V22 in between -> the row3->row0 store+reload is a pure in-lane
    // hand-off.  Carrying it in 'p22' removes one STS (row3) + one LDS (row0)
    // per BP iteration, bit-identically -- directly on the iteration-boundary
    // handoff this target attacks.  (No info column qualifies: every one changes
    // circulant shift between consecutive rows, forcing a cross-lane round-trip.)
    struct inlane_post_p4_t
    {
        word_t p23;
        word_t p24;
        word_t p25;
        word_t p22; // V22 posterior at slot (z+1)%Z, carried row3 -> next row0
    };

    __device__ __forceinline__
    void load_inlane_post_p4(const word_t* app, int z, inlane_post_p4_t& post)
    {
        post.p23 = app[(23 * ALGO103_Z) + z];
        post.p24 = app[(24 * ALGO103_Z) + z];
        post.p25 = app[(25 * ALGO103_Z) + z];
        // Iteration-0 row 3 defines p22 before its first register consumer.
        // can_decode_config pins the punctured-iteration-0 contract (n >= 1),
        // which is what guarantees row 3 has run before p22 is first read.
    }

    __device__ __forceinline__
    void store_inlane_post_p4(word_t* app, int z, const inlane_post_p4_t& post)
    {
        app[(23 * ALGO103_Z) + z] = post.p23;
        app[(24 * ALGO103_Z) + z] = post.p24;
        app[(25 * ALGO103_Z) + z] = post.p25;
        app[(22 * ALGO103_Z) + wrap_z_shift(z, 1)] = post.p22;
    }

    __device__ __forceinline__
    void init_c2v_zero_reg_p4(c2v_store_t& c2v)
    {
        word_t zero;
        zero.u32 = 0u;
#pragma unroll
        for(int r = 0; r < ALGO103_PARITY; ++r)
        {
#pragma unroll
            for(int p = 0; p < ALGO103_MAX_ROW_DEGREE; ++p)
            {
                c2v[r][p] = zero;
            }
        }
    }

    //==================================================================
    // FUSED pipelined + address-simplified + parity-residency edge ops.
    //
    // Pass A (forward, POS 0..18): LOAD the posterior, form v2c = post -
    // c2v[R][POS], record the prefix box-plus pre[POS], advance the prefix
    // accumulator.  Issuing all 19 (mutually independent) shared LOADs in this
    // pass lets the MIO/LSU pipe overlap the FMA-pipe box-plus recurrence.
    //
    // Pass B (backward, POS 18..0): form the leave-one-out check message
    // cn = norm * box_plus(pre[POS], suffix), refresh the posterior
    // post' = v2c + cn and STORE it, advance the suffix accumulator.  Each
    // store overlaps the neighbouring positions' box-plus + norm arithmetic.
    //
    // box_plus is associative & commutative (min |.| with XOR of signs), so the
    // regrouped prefix/suffix scans are bit-identical to the sequential draft;
    // c2v[R][POS] keeps the same (V,SHIFT) association.
    //
    // Address generation: the circulant gather/scatter avoids a
    // per-edge (z+SHIFT) mod Z and address LEA -- two per-thread base pointers
    // (app+z, app+z-Z) hoisted across the row are selected by a compile-time
    // thresholded predicate, and the column+shift fold into the load immediate:
    // base[V*Z + SHIFT] == app[V*Z + (z+SHIFT)%Z].
    //
    // Parity residency: the identity-shift degree-2 columns
    // 23/24/25 (compile-time V) read/write a per-lane register instead of shared
    // memory; the `if constexpr` branch folds away at compile time.
    //==================================================================
    // PREFETCHED edge: the posterior for this (V,SHIFT) was loaded from shared
    // memory BEFORE the preceding inter-row barrier (see pf_load_p4 /
    // run_layered_rows_0123_pf) because column V is provably NOT written by the
    // row that precedes this one -- so its value is stable across that barrier
    // and reading it early is bit-identical to the in-row load.  Passing it in a
    // register removes one barrier-gated LDS from the post-barrier burst and
    // hands ptxas a value it can fold into the DEFER_BLOCKING drain window.
    template <int R, int POS, int V, int SHIFT, bool FIRST = false, bool PREFETCHED = false>
    __device__ __forceinline__
    void pipe_fwd_edge(const word_t*           app,
                       int                     z,
                       const inlane_post_p4_t& post,
                       c2v_store_t&            c2v,
                       word_t (&m)  [ALGO103_MAX_ROW_DEGREE],
                       word_t (&pre)[ALGO103_MAX_ROW_DEGREE],
                       word_t&                 accP,
                       word_t                  pf = word_t{})
    {
        word_t p;
        if constexpr      (PREFETCHED) { p = pf; }
        else if constexpr (V == 23) { p = post.p23; }
        else if constexpr (V == 24) { p = post.p24; }
        else if constexpr (V == 25) { p = post.p25; }
        // V22 (shift 1) at the FIRST row of an iteration is the in-lane hand-off
        // from the previous iteration's row3: read it from the carry register
        // instead of re-loading the slot the same lane just wrote.
        else if constexpr (R == 0 && V == 22) { p = post.p22; }
        else
        {
            const word_t* base = (SHIFT != 0 && z >= (ALGO103_Z - SHIFT))
                                     ? (app + z - ALGO103_Z)
                                     : (app + z);
            p = base[(V * ALGO103_Z) + SHIFT];
        }
        const word_t mm = h2_sub(p, c2v[R][POS]);
        m[POS]   = mm;
        // box_plus(identity, x) == x exactly (identity = +max finite fp16), so
        // the first edge's prefix is the identity and its accumulator is just m
        // -- skip the redundant leading box_pair, shortening the serial chain.
        if constexpr (FIRST)
        {
            pre[POS] = box_plus_identity();
            accP     = mm;
        }
        else
        {
            pre[POS] = accP;             // prefix box-plus of positions < POS
            accP     = box_pair(accP, mm);
        }
    }

    template <int R, int POS, int V, int SHIFT, bool FIRST = false, bool LAST = false>
    __device__ __forceinline__
    void pipe_bwd_edge(word_t*           app,
                       int               z,
                       __half2           norm,
                       inlane_post_p4_t& post,
                       c2v_store_t&      c2v,
                       const word_t (&m)  [ALGO103_MAX_ROW_DEGREE],
                       const word_t (&pre)[ALGO103_MAX_ROW_DEGREE],
                       word_t&           accS)
    {
        const word_t mm = m[POS];
        // box_plus(identity, x) == x exactly (identity = +max finite fp16):
        //  - FIRST (highest POS): suffix accumulator is identity, so the
        //    leave-one-out collapses to pre[POS] and the suffix becomes m.
        //  - LAST  (POS 0):       prefix pre[0] is identity, so the
        //    leave-one-out collapses to the suffix accumulator accS (and the
        //    final accS update is dead -- no later edge consumes it).
        // Both skip a redundant box_pair, shortening the serial suffix chain.
        word_t cn;
        if constexpr (FIRST)
        {
            cn   = h2_mul_norm(pre[POS], norm);                 // leave-one-out
            accS = mm;
        }
        else if constexpr (LAST)
        {
            cn   = h2_mul_norm(accS, norm);                     // leave-one-out
        }
        else
        {
            cn   = h2_mul_norm(box_pair(pre[POS], accS), norm); // leave-one-out
            accS = box_pair(accS, mm);                          // suffix > POS
        }
        const word_t np = h2_add(mm, cn);                       // post' = v2c + c2v_new
        if constexpr      (V == 23) { post.p23 = np; }
        else if constexpr (V == 24) { post.p24 = np; }
        else if constexpr (V == 25) { post.p25 = np; }
        // V22 (shift 1) at the LAST row of an iteration: keep the freshly
        // computed posterior in the carry register for the next iteration's
        // row0 instead of scattering it to shared and gathering it straight
        // back.  Nothing between this write and that read touches the slot, so
        // the shared store is pure overhead -- skip it.  The dump kernel and the
        // post-loop flush materialize p22 back to shared (store_inlane_post_p4),
        // so every downstream reader (APP dump, soft-output) still sees it.
        else if constexpr (R == 3 && V == 22) { post.p22 = np; }
        else
        {
            word_t* base = (SHIFT != 0 && z >= (ALGO103_Z - SHIFT))
                               ? (app + z - ALGO103_Z)
                               : (app + z);
            base[(V * ALGO103_Z) + SHIFT] = np;
        }
        c2v[R][POS] = cn;
    }

    //------------------------------------------------------------------
    __device__
    void process_layer_row0_p4(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        word_t accP = box_plus_identity();
        pipe_fwd_edge<0,  0,  0, 307, true>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  1,  1,  19>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  2, 23,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  3, 22,   1>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  4, 21, 346>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  5, 20, 330>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  6, 19, 180>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  7, 18, 242>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  8, 16, 106>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  9, 15, 215>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 10, 13, 357>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 11, 12,  17>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 12, 11, 109>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 13, 10, 288>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 14,  9, 317>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 15,  6, 216>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 16,  5, 181>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 17,  3, 369>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 18,  2,  50>(app, z, post, c2v, m, pre, accP);
        word_t accS = box_plus_identity();
        pipe_bwd_edge<0, 18,  2,  50, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 17,  3, 369>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 16,  5, 181>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 15,  6, 216>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 14,  9, 317>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 13, 10, 288>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 12, 11, 109>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 11, 12,  17>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 10, 13, 357>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  9, 15, 215>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  8, 16, 106>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  7, 18, 242>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  6, 19, 180>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  5, 20, 330>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  4, 21, 346>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  3, 22,   1>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  2, 23,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  1,  1,  19>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  0,  0, 307, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //------------------------------------------------------------------
    __device__
    void process_layer_row1_p4(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        word_t accP = box_plus_identity();
        pipe_fwd_edge<1,  0,  0,  76, true>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  1, 24,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  2, 23,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  3, 22,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  4, 21, 112>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  5, 19, 331>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  6, 17, 114>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  7, 16, 354>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  8, 15,  99>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  9, 14, 217>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 10, 12, 342>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 11, 11, 295>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 12,  9, 178>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 13,  8, 331>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 14,  7, 331>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 15,  5, 144>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 16,  4, 288>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 17,  3,  73>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 18,  2,  76>(app, z, post, c2v, m, pre, accP);
        word_t accS = box_plus_identity();
        pipe_bwd_edge<1, 18,  2,  76, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 17,  3,  73>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 16,  4, 288>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 15,  5, 144>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 14,  7, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 13,  8, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 12,  9, 178>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 11, 11, 295>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 10, 12, 342>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  9, 14, 217>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  8, 15,  99>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  7, 16, 354>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  6, 17, 114>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  5, 19, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  4, 21, 112>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  3, 22,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  2, 23,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  1, 24,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  0,  0,  76, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //------------------------------------------------------------------
    __device__
    void process_layer_row2_p4(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        word_t accP = box_plus_identity();
        pipe_fwd_edge<2,  0,  0, 205, true>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  1,  1, 250>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  2, 25,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  3, 24,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  4, 20,  13>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  5, 19, 205>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  6, 18, 240>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  7, 17, 131>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  8, 15,  53>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  9, 14,  88>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 10, 13, 200>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 11, 10, 129>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 12,  9,  63>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 13,  8, 160>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 14,  7, 267>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 15,  6, 161>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 16,  5, 256>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 17,  4, 332>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 18,  2, 328>(app, z, post, c2v, m, pre, accP);
        word_t accS = box_plus_identity();
        pipe_bwd_edge<2, 18,  2, 328, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 17,  4, 332>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 16,  5, 256>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 15,  6, 161>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 14,  7, 267>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 13,  8, 160>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 12,  9,  63>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 11, 10, 129>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 10, 13, 200>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  9, 14,  88>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  8, 15,  53>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  7, 17, 131>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  6, 18, 240>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  5, 19, 205>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  4, 20,  13>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  3, 24,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  2, 25,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  1,  1, 250>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  0,  0, 205, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //------------------------------------------------------------------
    __device__
    void process_layer_row3_p4(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        word_t accP = box_plus_identity();
        pipe_fwd_edge<3,  0,  0, 276, true>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  1,  1,  87>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  2, 25,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  3, 22,   1>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  4, 21, 357>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  5, 20,  39>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  6, 18, 271>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  7, 17, 300>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  8, 16, 304>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  9, 14, 212>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 10, 13, 341>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 11, 12, 231>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 12, 11, 305>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 13, 10, 132>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 14,  8,  56>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 15,  7, 153>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 16,  6, 199>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 17,  4, 275>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 18,  3,   0>(app, z, post, c2v, m, pre, accP);
        word_t accS = box_plus_identity();
        pipe_bwd_edge<3, 18,  3,   0, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 17,  4, 275>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 16,  6, 199>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 15,  7, 153>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 14,  8,  56>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 13, 10, 132>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 12, 11, 305>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 11, 12, 231>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 10, 13, 341>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  9, 14, 212>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  8, 16, 304>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  7, 17, 300>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  6, 18, 271>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  5, 20,  39>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  4, 21, 357>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  3, 22,   1>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  2, 25,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  1,  1,  87>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  0,  0, 276, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //==================================================================
    // CROSS-BARRIER PREFETCH of barrier-INDEPENDENT columns (bit-exact).
    //
    // In the layered visit order 0->1->2->3, every inter-row __syncthreads()
    // exists only so the NEXT row's gather observes the CURRENT row's in-place
    // posterior scatter.  But a column V that the current row does NOT write is
    // not produced by that barrier at all -- its value was published by an
    // EARLIER barrier and is provably stable while the current row runs (no
    // thread touches app[V*Z+.] during the row).  So that column's posterior can
    // be loaded BEFORE the barrier and carried in a register into the next row,
    // bit-identically replacing the post-barrier LDS.
    //
    // Profiling showed the per-row barrier costs ~7.4% of warp
    // samples and is emitted as BAR.SYNC.DEFER_BLOCKING: the warp arrives, then
    // the *first dependent LDS* of the next row stalls until the barrier drains.
    // Hoisting the independent loads ahead of the barrier (a) removes them from
    // the post-barrier burst (this kernel is shared-memory/MIO-bound, so the
    // pre-barrier window where the MIO pipe is otherwise idle absorbs them) and
    // (b) hands ptxas register-resident values it can fold into the deferred
    // barrier drain instead of gating them behind the fence.
    //
    // Column->POS map of the independent set (degree-2 identity columns 23/24/25
    // are already lane-resident registers and never use shared, so they are not
    // listed): derived purely from the BG1/Z384/mb4 parity structure.
    //   row1 (cols row0 doesn't write): V17@P6, V14@P9, V8@P13, V7@P14, V4@P16
    //   row2 (cols row1 doesn't write): V1@P1, V20@P4, V18@P6, V13@P10, V10@P11, V6@P15
    //   row3 (cols row2 doesn't write): V22@P3, V21@P4, V16@P8, V12@P11, V11@P12, V3@P18
    //==================================================================
    template <int V, int SHIFT>
    __device__ __forceinline__
    word_t pf_load_p4(const word_t* app, int z)
    {
        const word_t* base = (SHIFT != 0 && z >= (ALGO103_Z - SHIFT))
                                 ? (app + z - ALGO103_Z)
                                 : (app + z);
        return base[(V * ALGO103_Z) + SHIFT];
    }

    struct pf_row1_t { word_t e6, e9, e13, e14, e16; };
    struct pf_row2_t { word_t e1, e4, e6, e10, e11, e15; };
    struct pf_row3_t { word_t e3, e4, e8, e11, e12, e18; };
    // row0 of the NEXT iteration: cols row3 (the prior iteration's last row-update)
    // does NOT write -> stable across the iteration-boundary barrier, so loadable
    // ahead of it.  V0/V1/V21/V20/V18/V16/V13/V12/V11/V10/V6/V3 ARE written by
    // row3 (dependent); V23 is lane-resident (post.p23) and V22 is the in-lane
    // carry (post.p22) -- neither uses shared, so neither is listed here.
    //   row0 indep cols: V19@180(P6) V15@215(P9) V9@317(P14) V5@181(P16) V2@50(P18)
    struct pf_row0_t { word_t e2, e5, e9, e15, e19; };

    __device__ __forceinline__
    pf_row1_t pf_load_row1(const word_t* app, int z)
    {
        pf_row1_t pf;
        pf.e6  = pf_load_p4<17, 114>(app, z);
        pf.e9  = pf_load_p4<14, 217>(app, z);
        pf.e13 = pf_load_p4< 8, 331>(app, z);
        pf.e14 = pf_load_p4< 7, 331>(app, z);
        pf.e16 = pf_load_p4< 4, 288>(app, z);
        return pf;
    }

    __device__ __forceinline__
    pf_row2_t pf_load_row2(const word_t* app, int z)
    {
        pf_row2_t pf;
        pf.e1  = pf_load_p4< 1, 250>(app, z);
        pf.e4  = pf_load_p4<20,  13>(app, z);
        pf.e6  = pf_load_p4<18, 240>(app, z);
        pf.e10 = pf_load_p4<13, 200>(app, z);
        pf.e11 = pf_load_p4<10, 129>(app, z);
        pf.e15 = pf_load_p4< 6, 161>(app, z);
        return pf;
    }

    __device__ __forceinline__
    pf_row3_t pf_load_row3(const word_t* app, int z)
    {
        pf_row3_t pf;
        pf.e3  = pf_load_p4<22,   1>(app, z);
        pf.e4  = pf_load_p4<21, 357>(app, z);
        pf.e8  = pf_load_p4<16, 304>(app, z);
        pf.e11 = pf_load_p4<12, 231>(app, z);
        pf.e12 = pf_load_p4<11, 305>(app, z);
        pf.e18 = pf_load_p4< 3,   0>(app, z);
        return pf;
    }

    __device__ __forceinline__
    pf_row0_t pf_load_row0(const word_t* app, int z)
    {
        pf_row0_t pf;
        pf.e19 = pf_load_p4<19, 180>(app, z);
        pf.e15 = pf_load_p4<15, 215>(app, z);
        pf.e9  = pf_load_p4< 9, 317>(app, z);
        pf.e5  = pf_load_p4< 5, 181>(app, z);
        pf.e2  = pf_load_p4< 2,  50>(app, z);
        return pf;
    }

    //------------------------------------------------------------------
    // process_layer_row{1,2,3}_p4_pf: identical numerics to the non-pf row
    // (same edges, same FIRST/LAST, same backward sweep) -- the only change is
    // that the independent-column forward edges consume the prefetched register
    // instead of issuing a barrier-gated LDS.  Bit-identical to the originals.
    __device__
    void process_layer_row1_p4_pf(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post, const pf_row1_t& pf)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        // Reordered visit order (box_plus is exactly commutative, so the
        // leave-one-out outputs are bit-identical): barrier-INDEPENDENT columns
        // first (5 prefetched shared + 2 lane-resident registers), then the
        // barrier-dependent shared columns.  The independent prefix box-plus
        // runs entirely on register operands, so the chain advances during the
        // preceding barrier's DEFER_BLOCKING drain instead of stalling on the
        // first dependent LDS.  m[]/pre[] stay indexed by the original POS, so
        // c2v[R][POS] association (and thus the cross-iteration trajectory) is
        // unchanged.
        word_t accP = box_plus_identity();
        pipe_fwd_edge<1,  6, 17, 114, true,  true>(app, z, post, c2v, m, pre, accP, pf.e6);
        pipe_fwd_edge<1,  9, 14, 217, false, true>(app, z, post, c2v, m, pre, accP, pf.e9);
        pipe_fwd_edge<1, 13,  8, 331, false, true>(app, z, post, c2v, m, pre, accP, pf.e13);
        pipe_fwd_edge<1, 14,  7, 331, false, true>(app, z, post, c2v, m, pre, accP, pf.e14);
        pipe_fwd_edge<1, 16,  4, 288, false, true>(app, z, post, c2v, m, pre, accP, pf.e16);
        pipe_fwd_edge<1,  2, 23,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  1, 24,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  0,  0,  76>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  3, 22,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  4, 21, 112>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  5, 19, 331>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  7, 16, 354>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1,  8, 15,  99>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 10, 12, 342>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 11, 11, 295>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 12,  9, 178>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 15,  5, 144>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 17,  3,  73>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<1, 18,  2,  76>(app, z, post, c2v, m, pre, accP);
        // Backward sweep visits the reverse of the forward order.
        word_t accS = box_plus_identity();
        pipe_bwd_edge<1, 18,  2,  76, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 17,  3,  73>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 15,  5, 144>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 12,  9, 178>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 11, 11, 295>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 10, 12, 342>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  8, 15,  99>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  7, 16, 354>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  5, 19, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  4, 21, 112>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  3, 22,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  0,  0,  76>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  1, 24,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  2, 23,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 16,  4, 288>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 14,  7, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1, 13,  8, 331>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  9, 14, 217>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<1,  6, 17, 114, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    __device__
    void process_layer_row2_p4_pf(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post, const pf_row2_t& pf)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        // Independent-first reorder (see process_layer_row1_p4_pf): 6 prefetched
        // shared + 2 lane-resident register columns, then dependent shared.
        word_t accP = box_plus_identity();
        pipe_fwd_edge<2,  1,  1, 250, true,  true>(app, z, post, c2v, m, pre, accP, pf.e1);
        pipe_fwd_edge<2,  4, 20,  13, false, true>(app, z, post, c2v, m, pre, accP, pf.e4);
        pipe_fwd_edge<2,  6, 18, 240, false, true>(app, z, post, c2v, m, pre, accP, pf.e6);
        pipe_fwd_edge<2, 10, 13, 200, false, true>(app, z, post, c2v, m, pre, accP, pf.e10);
        pipe_fwd_edge<2, 11, 10, 129, false, true>(app, z, post, c2v, m, pre, accP, pf.e11);
        pipe_fwd_edge<2, 15,  6, 161, false, true>(app, z, post, c2v, m, pre, accP, pf.e15);
        pipe_fwd_edge<2,  2, 25,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  3, 24,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  0,  0, 205>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  5, 19, 205>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  7, 17, 131>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  8, 15,  53>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2,  9, 14,  88>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 12,  9,  63>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 13,  8, 160>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 14,  7, 267>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 16,  5, 256>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 17,  4, 332>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<2, 18,  2, 328>(app, z, post, c2v, m, pre, accP);
        // Backward sweep visits the reverse of the forward order.
        word_t accS = box_plus_identity();
        pipe_bwd_edge<2, 18,  2, 328, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 17,  4, 332>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 16,  5, 256>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 14,  7, 267>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 13,  8, 160>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 12,  9,  63>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  9, 14,  88>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  8, 15,  53>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  7, 17, 131>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  5, 19, 205>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  0,  0, 205>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  3, 24,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  2, 25,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 15,  6, 161>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 11, 10, 129>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2, 10, 13, 200>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  6, 18, 240>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  4, 20,  13>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<2,  1,  1, 250, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    __device__
    void process_layer_row3_p4_pf(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post, const pf_row3_t& pf)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        // Independent-first reorder (see process_layer_row1_p4_pf): 6 prefetched
        // shared + 1 lane-resident register column, then dependent shared.
        word_t accP = box_plus_identity();
        pipe_fwd_edge<3,  3, 22,   1, true,  true>(app, z, post, c2v, m, pre, accP, pf.e3);
        pipe_fwd_edge<3,  4, 21, 357, false, true>(app, z, post, c2v, m, pre, accP, pf.e4);
        pipe_fwd_edge<3,  8, 16, 304, false, true>(app, z, post, c2v, m, pre, accP, pf.e8);
        pipe_fwd_edge<3, 11, 12, 231, false, true>(app, z, post, c2v, m, pre, accP, pf.e11);
        pipe_fwd_edge<3, 12, 11, 305, false, true>(app, z, post, c2v, m, pre, accP, pf.e12);
        pipe_fwd_edge<3, 18,  3,   0, false, true>(app, z, post, c2v, m, pre, accP, pf.e18);
        pipe_fwd_edge<3,  2, 25,   0>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  0,  0, 276>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  1,  1,  87>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  5, 20,  39>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  6, 18, 271>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  7, 17, 300>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3,  9, 14, 212>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 10, 13, 341>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 13, 10, 132>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 14,  8,  56>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 15,  7, 153>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 16,  6, 199>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<3, 17,  4, 275>(app, z, post, c2v, m, pre, accP);
        // Backward sweep visits the reverse of the forward order.
        word_t accS = box_plus_identity();
        pipe_bwd_edge<3, 17,  4, 275, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 16,  6, 199>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 15,  7, 153>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 14,  8,  56>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 13, 10, 132>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 10, 13, 341>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  9, 14, 212>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  7, 17, 300>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  6, 18, 271>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  5, 20,  39>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  1,  1,  87>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  0,  0, 276>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  2, 25,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 18,  3,   0>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 12, 11, 305>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3, 11, 12, 231>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  8, 16, 304>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  4, 21, 357>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<3,  3, 22,   1, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //------------------------------------------------------------------
    // CROSS-ITERATION-BOUNDARY prefetched row0.  Identical numerics to
    // process_layer_row0_p4 -- the only change is that the 5 barrier-independent
    // forward edges (cols row3 of the PRIOR iteration does not write) consume
    // prefetched registers loaded ahead of the iteration-boundary barrier, and
    // the visit order is reordered independent-first so the prefix box-plus chain
    // advances on register/prefetched operands during that barrier's
    // DEFER_BLOCKING drain instead of stalling on row0's first dependent LDS.
    // V23 (post.p23) and V22 (post.p22, the in-lane carry from row3) are also
    // barrier-independent and visited in the leading group.  box_plus is exactly
    // assoc/commutative and m[]/pre[] keep the original POS index, so c2v[0][POS]
    // association and the per-iteration APP trajectory are bit-identical.
    __device__
    void process_layer_row0_p4_pf(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post, const pf_row0_t& pf)
    {
        word_t m[ALGO103_MAX_ROW_DEGREE];
        word_t pre[ALGO103_MAX_ROW_DEGREE];
        word_t accP = box_plus_identity();
        // Leading group: 5 prefetched shared + 2 lane-resident register columns.
        pipe_fwd_edge<0,  6, 19, 180, true,  true>(app, z, post, c2v, m, pre, accP, pf.e19);
        pipe_fwd_edge<0,  9, 15, 215, false, true>(app, z, post, c2v, m, pre, accP, pf.e15);
        pipe_fwd_edge<0, 14,  9, 317, false, true>(app, z, post, c2v, m, pre, accP, pf.e9);
        pipe_fwd_edge<0, 16,  5, 181, false, true>(app, z, post, c2v, m, pre, accP, pf.e5);
        pipe_fwd_edge<0, 18,  2,  50, false, true>(app, z, post, c2v, m, pre, accP, pf.e2);
        pipe_fwd_edge<0,  2, 23,   0>(app, z, post, c2v, m, pre, accP); // post.p23
        pipe_fwd_edge<0,  3, 22,   1>(app, z, post, c2v, m, pre, accP); // post.p22 (R==0&&V==22)
        // Trailing group: barrier-dependent shared columns (written by row3).
        pipe_fwd_edge<0,  0,  0, 307>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  1,  1,  19>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  4, 21, 346>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  5, 20, 330>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  7, 18, 242>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0,  8, 16, 106>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 10, 13, 357>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 11, 12,  17>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 12, 11, 109>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 13, 10, 288>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 15,  6, 216>(app, z, post, c2v, m, pre, accP);
        pipe_fwd_edge<0, 17,  3, 369>(app, z, post, c2v, m, pre, accP);
        // Backward sweep visits the reverse of the forward order.
        word_t accS = box_plus_identity();
        pipe_bwd_edge<0, 17,  3, 369, true>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 15,  6, 216>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 13, 10, 288>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 12, 11, 109>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 11, 12,  17>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 10, 13, 357>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  8, 16, 106>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  7, 18, 242>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  5, 20, 330>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  4, 21, 346>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  1,  1,  19>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  0,  0, 307>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  3, 22,   1>(app, z, norm, post, c2v, m, pre, accS); // V22 -> shared (R==0)
        pipe_bwd_edge<0,  2, 23,   0>(app, z, norm, post, c2v, m, pre, accS); // V23 -> post.p23
        pipe_bwd_edge<0, 18,  2,  50>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 16,  5, 181>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0, 14,  9, 317>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  9, 15, 215>(app, z, norm, post, c2v, m, pre, accS);
        pipe_bwd_edge<0,  6, 19, 180, false, true>(app, z, norm, post, c2v, m, pre, accS);
    }

    //------------------------------------------------------------------
    // Prefetch-pipelined variant of run_layered_rows<0,1,2,3> for the algo103
    // main loop.  Each row's barrier-independent posterior columns are loaded
    // ahead of the preceding barrier and consumed from registers, so the
    // post-barrier LDS burst shrinks and the loads overlap the barrier drain.
    // Numerically identical to run_layered_rows<0,1,2,3>.
    __device__ __forceinline__
    void run_layered_rows_0123_pf(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        process_layer_row0_p4(app, z, norm, c2v, post);
        const pf_row1_t pf1 = pf_load_row1(app, z);   // cols row0 doesn't write
        __syncthreads();
        process_layer_row1_p4_pf(app, z, norm, c2v, post, pf1);
        const pf_row2_t pf2 = pf_load_row2(app, z);   // cols row1 doesn't write
        __syncthreads();
        process_layer_row2_p4_pf(app, z, norm, c2v, post, pf2);
        const pf_row3_t pf3 = pf_load_row3(app, z);   // cols row2 doesn't write
        __syncthreads();
        process_layer_row3_p4_pf(app, z, norm, c2v, post, pf3);
        __syncthreads();
    }

    //------------------------------------------------------------------
    // SOFTWARE-PIPELINED iteration: same numerics as run_layered_rows_0123_pf,
    // but row0's barrier-independent columns for the NEXT iteration are loaded
    // BEFORE this iteration's closing (boundary) barrier and returned, so those
    // gathers overlap the boundary barrier's DEFER_BLOCKING drain instead of
    // stalling the next iteration's row0.  The returned pf_row0_t is consumed by
    // the next iteration's row0 (process_layer_row0_p4_pf).
    //
    // _seed: row0 has no incoming prefetch (used for the first steady-state
    //        iteration, whose predecessor is the punctured iter-0 pass).
    // _carry: row0 consumes the prefetch produced by the previous iteration.
    __device__ __forceinline__
    pf_row0_t run_iter_pf_seed(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        // Same deeper-pipeline materialization as run_iter_pf_carry: hoist each
        // row's barrier-independent prefetch to immediately after the barrier
        // that publishes its source columns (bit-identical; see run_iter_pf_carry).
        const pf_row1_t pf1 = pf_load_row1(app, z);   // cols row0 won't write
        process_layer_row0_p4(app, z, norm, c2v, post);
        __syncthreads();                              // row0 published
        const pf_row2_t pf2 = pf_load_row2(app, z);   // cols row1 won't write
        process_layer_row1_p4_pf(app, z, norm, c2v, post, pf1);
        __syncthreads();                              // row1 published
        const pf_row3_t pf3 = pf_load_row3(app, z);   // cols row2 won't write
        process_layer_row2_p4_pf(app, z, norm, c2v, post, pf2);
        __syncthreads();                              // row2 published
        const pf_row0_t pf0 = pf_load_row0(app, z);   // next row0's indep cols
        process_layer_row3_p4_pf(app, z, norm, c2v, post, pf3);
        __syncthreads();                              // iteration-boundary barrier
        return pf0;
    }

    __device__ __forceinline__
    pf_row0_t run_iter_pf_carry(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post, const pf_row0_t& pf0_in)
    {
        // DEEPER software pipeline (assigned target: where prefetched values are
        // materialized).  Each row's barrier-independent prefetch is hoisted a
        // FULL ROW earlier than the previous generation -- to immediately
        // after the barrier that publishes its source columns, rather than
        // right before the next barrier.  Legality is unchanged: pfN reads
        // only columns the intervening row does NOT write, so they are stable
        // across that row and loading them before it is bit-identical.  The
        // win: each prefetch's 5-6
        // LDS now overlap an entire row of box-plus/store work (and the barrier
        // drain), and the longer live ranges raise the register high-water
        // toward the 1-CTA/SM spill cliff -- exactly the scheduling-freedom lever
        // the target calls for.  Numerically identical; shared by dump + timed.
        const pf_row1_t pf1 = pf_load_row1(app, z);   // cols row0 won't write
        process_layer_row0_p4_pf(app, z, norm, c2v, post, pf0_in);
        __syncthreads();                              // row0 published
        const pf_row2_t pf2 = pf_load_row2(app, z);   // cols row1 won't write
        process_layer_row1_p4_pf(app, z, norm, c2v, post, pf1);
        __syncthreads();                              // row1 published
        const pf_row3_t pf3 = pf_load_row3(app, z);   // cols row2 won't write
        process_layer_row2_p4_pf(app, z, norm, c2v, post, pf2);
        __syncthreads();                              // row2 published
        const pf_row0_t pf0 = pf_load_row0(app, z);   // next row0's indep cols
        process_layer_row3_p4_pf(app, z, norm, c2v, post, pf3);
        __syncthreads();                              // iteration-boundary barrier
        return pf0;
    }

    //==================================================================
    // FIRST-ITERATION PUNCTURE ELISION (bit-exact, structural).
    //
    // The NR BG1 TB decode interface always punctures the first
    // CUPHY_LDPC_NUM_PUNCTURED_NODES (== 2) systematic columns, so at decode
    // entry the two high-degree info columns -- V0 (POS0 of every parity row)
    // and V1 (POS1 of rows {0,2,3}) -- carry EXACTLY zero channel LLR, and the
    // register c2v store is zeroed (init_c2v_zero_reg_p4). box_plus is the
    // sign-min rule min(|a|,|b|)*xorsign; a min over a set still containing a
    // zero edge is EXACTLY 0, so any leave-one-out for a row that still holds a
    // punctured-zero edge is a provably-trivial output: posterior += 0, c2v
    // stays 0. Tracing the layered visit order 0,1,2,3 at iteration 0:
    //   row0: V0 AND V1 both zero -> EVERY edge's leave-one-out sees a remaining
    //         zero -> all 19 outputs are 0 -> the whole row (and its barrier) is
    //         a no-op and is SKIPPED. row1's gather reads the post-load `app`,
    //         already published by the loader's barrier.
    //   row1: only V0 (POS0) still zero -> only V0's output is non-trivial:
    //         norm*box_plus(the other 18 edges). Every other edge's output is 0
    //         and its posterior is unchanged -> collapse to that single output.
    //   row2: row1 made V0 non-zero, V1 (POS1) still zero -> only V1's output is
    //         non-trivial = norm*box_plus(the other 18 edges, including the V0
    //         edge now carrying row1's update). Collapse likewise.
    //   row3: V0 and V1 are now both non-zero -> an ordinary full row update.
    // box_plus is exactly associative/commutative (integer min + sign xor, no
    // rounding); the only rounded ops are the single norm-multiply and posterior
    // add, both applied once per surviving output exactly as the full row applies
    // them -> BIT-IDENTICAL to run_layered_rows<0,1,2,3> at iteration 0, keyed
    // strictly to the BG1 puncture structure, never to data. Removes ~1.8 of
    // iteration-0's 4 row-updates and one block barrier.
    //==================================================================
    template <int V, int SHIFT>
    __device__ __forceinline__
    word_t iter0_load_post(const word_t* app, int z, const inlane_post_p4_t& post)
    {
        if constexpr      (V == 23) { return post.p23; }
        else if constexpr (V == 24) { return post.p24; }
        else if constexpr (V == 25) { return post.p25; }
        else
        {
            const word_t* base = (SHIFT != 0 && z >= (ALGO103_Z - SHIFT))
                                     ? (app + z - ALGO103_Z)
                                     : (app + z);
            return base[(V * ALGO103_Z) + SHIFT];
        }
    }

    template <int V, int SHIFT>
    __device__ __forceinline__
    void iter0_box_acc(const word_t* app, int z, const inlane_post_p4_t& post, word_t& acc)
    {
        // c2v[R][POS]==0 at iter0, so v2c == loaded posterior; accumulate it
        // into the running box_plus.
        acc = box_pair(acc, iter0_load_post<V, SHIFT>(app, z, post));
    }

    __device__ __forceinline__
    void iter0_box_acc_pf(word_t p, word_t& acc)
    {
        acc = box_pair(acc, p);
    }

    template <int V, int SHIFT>
    __device__ __forceinline__
    void iter0_store_post(word_t* app, int z, word_t np)
    {
        word_t* base = (SHIFT != 0 && z >= (ALGO103_Z - SHIFT))
                           ? (app + z - ALGO103_Z)
                           : (app + z);
        base[(V * ALGO103_Z) + SHIFT] = np;
    }

    //------------------------------------------------------------------
    // Iteration-0 row 1: emit ONLY V0's (POS0, V0 SHIFT 76) check output.
    __device__
    void process_layer_row1_p4_iter0(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t acc = box_plus_identity();
        iter0_box_acc<24,   0>(app, z, post, acc); // POS1
        iter0_box_acc<23,   0>(app, z, post, acc); // POS2
        iter0_box_acc<22,   0>(app, z, post, acc); // POS3
        iter0_box_acc<21, 112>(app, z, post, acc); // POS4
        iter0_box_acc<19, 331>(app, z, post, acc); // POS5
        iter0_box_acc<17, 114>(app, z, post, acc); // POS6
        iter0_box_acc<16, 354>(app, z, post, acc); // POS7
        iter0_box_acc<15,  99>(app, z, post, acc); // POS8
        iter0_box_acc<14, 217>(app, z, post, acc); // POS9
        iter0_box_acc<12, 342>(app, z, post, acc); // POS10
        iter0_box_acc<11, 295>(app, z, post, acc); // POS11
        iter0_box_acc< 9, 178>(app, z, post, acc); // POS12
        iter0_box_acc< 8, 331>(app, z, post, acc); // POS13
        iter0_box_acc< 7, 331>(app, z, post, acc); // POS14
        iter0_box_acc< 5, 144>(app, z, post, acc); // POS15
        iter0_box_acc< 4, 288>(app, z, post, acc); // POS16
        iter0_box_acc< 3,  73>(app, z, post, acc); // POS17
        iter0_box_acc< 2,  76>(app, z, post, acc); // POS18
        const word_t cn = h2_mul_norm(acc, norm);  // leave-one-out for V0
        iter0_store_post<0, 76>(app, z, cn);        // post' = v2c(0) + cn = cn
        c2v[1][0] = cn;
    }

    //------------------------------------------------------------------
    // Iteration-0 row 2: emit ONLY V1's (POS1, V1 SHIFT 250) check output.
    __device__
    void process_layer_row2_p4_iter0_pf(word_t* app,
                                       int z,
                                       __half2 norm,
                                       c2v_store_t& c2v,
                                       inlane_post_p4_t& post,
                                       const pf_row2_t& pf)
    {
        word_t acc = box_plus_identity();
        iter0_box_acc< 0, 205>(app, z, post, acc); // POS0  (V0, now non-zero from row1)
        iter0_box_acc<25,   0>(app, z, post, acc); // POS2
        iter0_box_acc<24,   0>(app, z, post, acc); // POS3
        iter0_box_acc_pf(pf.e4,  acc);              // POS4  V20
        iter0_box_acc<19, 205>(app, z, post, acc); // POS5
        iter0_box_acc_pf(pf.e6,  acc);              // POS6  V18
        iter0_box_acc<17, 131>(app, z, post, acc); // POS7
        iter0_box_acc<15,  53>(app, z, post, acc); // POS8
        iter0_box_acc<14,  88>(app, z, post, acc); // POS9
        iter0_box_acc_pf(pf.e10, acc);              // POS10 V13
        iter0_box_acc_pf(pf.e11, acc);              // POS11 V10
        iter0_box_acc< 9,  63>(app, z, post, acc); // POS12
        iter0_box_acc< 8, 160>(app, z, post, acc); // POS13
        iter0_box_acc< 7, 267>(app, z, post, acc); // POS14
        iter0_box_acc_pf(pf.e15, acc);              // POS15 V6
        iter0_box_acc< 5, 256>(app, z, post, acc); // POS16
        iter0_box_acc< 4, 332>(app, z, post, acc); // POS17
        iter0_box_acc< 2, 328>(app, z, post, acc); // POS18
        const word_t cn = h2_mul_norm(acc, norm);  // leave-one-out for V1
        iter0_store_post<1, 250>(app, z, cn);       // post' = v2c(1) + cn = cn
        c2v[2][1] = cn;
    }

    //------------------------------------------------------------------
    // Full first BP iteration with the punctured-column trivial work elided.
    // Bit-identical to run_layered_rows<0,1,2,3> at iteration 0.
    __device__ __forceinline__
    void run_layered_rows_iter0_punctured(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        // row0 is a provable no-op at entry -> skipped (barrier included).
        // Row1 only writes V0 and row2 only writes V1; all other inputs of
        // their successors are stable and can be gathered across the two
        // internal opening-pass barriers.  Do not carry anything across the
        // iteration boundary into the separately materialized steady seed.
        const pf_row2_t pf2 = pf_load_row2(app, z);
        process_layer_row1_p4_iter0(app, z, norm, c2v, post);
        __syncthreads();
        const pf_row3_t pf3 = pf_load_row3(app, z);
        process_layer_row2_p4_iter0_pf(app, z, norm, c2v, post, pf2);
        __syncthreads();
        process_layer_row3_p4_pf(app, z, norm, c2v, post, pf3);
        __syncthreads();
    }

    //==================================================================
    // Non-uniform (degree/puncture-aware) layered schedule dispatch.
    //
    // Every row-update is one block barrier + a forward/backward shared sweep,
    // so the total row-update count is the dominant shared-memory-bound cost.
    // Convergence at the R~0.92 cliff is gated by the two punctured high-degree
    // info columns V0 (deg 4, rows {0,1,2,3}) and V1 (deg 3, rows {0,2,3}); the
    // .cu's SCHED_ITERn packs spend the budget on rows {0,2,3} carrying both.
    //==================================================================
    template <int R>
    __device__ __forceinline__
    void process_layer_row_dispatch(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        static_assert(R >= 0 && R < ALGO103_PARITY, "row index out of range");
        if constexpr      (R == 0) { process_layer_row0_p4(app, z, norm, c2v, post); }
        else if constexpr (R == 1) { process_layer_row1_p4(app, z, norm, c2v, post); }
        else if constexpr (R == 2) { process_layer_row2_p4(app, z, norm, c2v, post); }
        else                       { process_layer_row3_p4(app, z, norm, c2v, post); }
    }

    // Run a compile-time sequence of parity rows, each followed by a single
    // block barrier so the next row's gather sees the previous row's in-place
    // posterior scatter.  Fully unrolled via a C++17 fold.
    template <int... Rows>
    __device__ __forceinline__
    void run_layered_rows(word_t* app, int z, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        ( (process_layer_row_dispatch<Rows>(app, z, norm, c2v, post), __syncthreads()), ... );
    }

    //------------------------------------------------------------------
    __device__
    void do_algo103_layered_iteration_p4(char* smem, __half2 norm, c2v_store_t& c2v, inlane_post_p4_t& post)
    {
        word_t* app = app_smem(smem);
        const int z = threadIdx.x;

        process_layer_row0_p4(app, z, norm, c2v, post);
        __syncthreads();
        process_layer_row1_p4(app, z, norm, c2v, post);
        __syncthreads();
        process_layer_row2_p4(app, z, norm, c2v, post);
        __syncthreads();
        process_layer_row3_p4(app, z, norm, c2v, post);
        __syncthreads();
    }

} // namespace

#endif // !defined(LDPC2_ALGO103_CUH_INCLUDED_)
