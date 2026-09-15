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

#if !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)
#define LDPC2_DEC_OUTPUT_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// Full-warp ballot membermask (all 32 lanes). Held in CONSTANT memory on
// purpose: the literal 0xFFFFFFFF passed to __ballot_sync() is re-emitted
// by ptxas as a fresh `MOV R, 0xffffffff` before EVERY ballot of the fully
// unrolled hard-decision output loop. A constant-memory value is opaque to
// the front-end -- it cannot be folded back into a per-ballot immediate --
// so it is loaded ONCE and held resident across the unrolled loop. The value
// is identical (0xFFFFFFFF), so the decode output is bit-for-bit unchanged.
// [[maybe_unused]] + static keeps each translation unit self-contained
// (per-TU copy, no cross-TU __constant__ linkage, which this build -- no
// -rdc -- does not support).
[[maybe_unused]] static __constant__ uint32_t c_ldpc_full_warp_mask = 0xFFFFFFFFu;

////////////////////////////////////////////////////////////////////////
// output_codeword_addr
// Provides the output address for a codeword when using the tensor-
// based LDPC decoder interface.
struct output_codeword_addr
{
    static __device__ uint32_t* get(const LDPC_kernel_params& params, int idx)
    {
        return reinterpret_cast<uint32_t*>(params.out + (idx * sizeof(uint32_t) * params.output_stride_words));
    }
};

////////////////////////////////////////////////////////////////////////
// output_LLR_addr
// Provides the output address for codeword LLR values when using the
// tensor-based LDPC decoder interface.
// Each thread will write 32 bits. Since only fp16 is supported for
// soft outputs, that means that each thread will write 2 fp16 values.
// The stride must be a multiple of 2 (elements) for uint32_t storage.
template <typename T>
struct output_LLR_addr
{
    static __device__ uint32_t* get(const LDPC_kernel_params& params, int idx)
    {
        T* tOut = static_cast<T*>(params.soft_out);
        return reinterpret_cast<uint32_t*>(tOut + (idx * params.soft_out_stride_elements));
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr()
// Provides the output address for a codeword when using the transport
// block-based LDPC decoder interface.
template <typename T> struct decode_desc_output_addr
{
    __device__
    static uint32_t* get(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
    {
        uint32_t* addr = nullptr;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.tb_output[i].num_codewords)
                {
                    addr = decodeDesc.tb_output[i].addr + (cwIndex * decodeDesc.tb_output[i].stride_words);
                    break;
                }
                cwIndex -= decodeDesc.tb_output[i].num_codewords;
            }
        }
        return addr;
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_soft_output_addr()
// Provides the output address for a codeword when using the transport
// block-based LDPC decoder interface.
// T is the data type for soft output values.
template <class T>
struct decode_desc_soft_output_addr
{
    __device__
    static uint32_t* get(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
    {
        uint32_t* addr = nullptr;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.llr_output[i].num_codewords)
                {
                    T* ph = static_cast<T*>(decodeDesc.llr_output[i].addr);
                    addr = reinterpret_cast<uint32_t*>(ph + (cwIndex * decodeDesc.llr_output[i].stride_elements));
                    break;
                }
                cwIndex -= decodeDesc.llr_output[i].num_codewords;
            }
        }
        return addr;
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr
// Specialization of decode_desc_output_addr for __half2
//template <> struct decode_desc_output_addr<__half2>
//{
//    __device__
//    static uint32_t* get()
//    {
//        return nullptr;
//    }
//};

////////////////////////////////////////////////////////////////////////
// num_cta_output_codewords()
// Returns the number of output codewords for a CTA.
// For 1x codeword at a time kernels, the return value is always 1.
// For 2x codeword at a time kernels, the return value will be 2 in all
// cases, except for the last CTA when the number of output codewords is
// odd. (In that case the return value will be 1.)
// This function is useful as written when the number of CTAs is equal
// to the number of codewords, but should not be used for looping.
template <typename T> __device__
int num_cta_output_codewords(int total_num_cw)
{
    if(1 == codewords_per_CTA<T>::value)
    {
        return 1;
    }
    else
    {
        const int CW_PER_CTA = codewords_per_CTA<T>::value;
        return min(CW_PER_CTA, total_num_cw - (blockIdx.x * CW_PER_CTA));
        // IDX * CW_PER_CTA + 1 < NUM_CW
        // IDX * CW_PER_CTA < (NUM_CW-1)
        // 1 < NUM_CW - (IDX * CW_PER_CTA)
        //if((blockIdx.x * codewords_per_CTA<T>::value + 1) < total_num_cw)
        //{
        //    return 2;
        //}
        //else
        //{
        //    return 1;
        //}
    }
}


////////////////////////////////////////////////////////////////////////
// output_token
// Class wrapper to disambiguate an additional argument passed to
// output_params constructors.
struct output_token
{
    tb_token token;
    __device__ explicit
    output_token(tb_token tok) : token(tok) {}
};

template <typename T> struct ldpc_dec_output_params;

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
template <typename T> struct ldpc_dec_output_params
{
    uint32_t* dst_gmem;         // output address for this CTA
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params, int cwIndex) :
        dst_gmem(output_codeword_addr::get(params, cwIndex)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(uint32_t* dst,
                           int       nbits,
                           int       nwords) :
        dst_gmem(dst),
        num_cw_bits(nbits),
        out_words_per_cw(nwords)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex) :
        dst_gmem(decode_desc_output_addr<T>::get(decodeDesc, cwIndex)),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
        int  tb     = tb_from_token(out_tok.token);
        int  offset = offset_from_token(out_tok.token);
        int  stride = decodeDesc.tb_output[tb].stride_words;
        dst_gmem    = decodeDesc.tb_output[tb].addr + (offset * stride);
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
// specialization for __half2 (2 codewords at a time)
template <>
struct ldpc_dec_output_params<__half2>
{
    uint32_t* dst_gmem;
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    int       out_stride_words; // needed for 2x codewords, where output stores two consecutive outputs
    int       num_out_cw;       // number of output codewords (by this CTA!), needed for 2x codeword kernels only
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params) :
        dst_gmem(output_codeword_addr::get(params, blockIdx.x * codewords_per_CTA<__half2>::value)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(params.output_stride_words),
        num_out_cw(num_cta_output_codewords<__half2>(params.num_codewords))
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int blkIndex = blockIdx.x;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                int iBlocksClaimed = (decodeDesc.tb_output[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    out_stride_words = decodeDesc.tb_output[i].stride_words;
                    dst_gmem         = decodeDesc.tb_output[i].addr + (blkIndex * 2 * out_stride_words);
                    // Last block may have only 1 codeword...
                    num_out_cw       = ((blkIndex*2 + 1) == decodeDesc.tb_output[i].num_codewords) ? 1 : 2;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int  tb          = tb_from_token(out_tok.token);
        int  offset      = offset_from_token(out_tok.token);
        bool is_partial  = is_partial_from_token(out_tok.token);
        out_stride_words = decodeDesc.tb_output[tb].stride_words;
        dst_gmem         = decodeDesc.tb_output[tb].addr + (offset * out_stride_words);
        num_out_cw       = is_partial ? 1 : 2;
    }
};



////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_params
// This struct converts from the two different parameter structures
// (LDPC_kernel_params for the legacy tensor interface and cuphyLDPCDecodeDesc_t
// for the transport block interface) to a common structure, allowing
// a single function to implement the looping logic.
// This structure assumes that threads will write values packed into a
// uint32_t. Therefore, as an example, when the soft output type is fp16,
// each thread will pack two fp16 values into a uint32_t and write to
// output memory.
template <typename T>
struct ldpc_dec_soft_output_params
{
    uint32_t* dst_gmem;      // output address for this CTA
    int       num_cw_values;
    __device__
    ldpc_dec_soft_output_params(const LDPC_kernel_params& params, int cwIndex) :
        dst_gmem(output_LLR_addr<T>::get(params, cwIndex)),
        num_cw_values(get_num_output_bits(params))
    {
    }
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex) :
        dst_gmem(decode_desc_soft_output_addr<T>::get(decodeDesc, cwIndex)),
        num_cw_values(get_num_output_bits(decodeDesc))
    {
    }
    // Constructor to set up the soft output parameters using the transport
    // block token, used by some kernels to store information on the specific
    // codeword targeted by a CTA.
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc))
    {
        int  tb     = tb_from_token(out_tok.token);
        int  offset = offset_from_token(out_tok.token);
        int  stride = decodeDesc.llr_output[tb].stride_elements;
        T* h        = static_cast<T*>(decodeDesc.llr_output[tb].addr);
        dst_gmem    = reinterpret_cast<uint32_t*>(h + (offset * stride));
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_params
// Specialization of ldpc_dec_soft_output_params for kernels that decode
// two codewords in a single CTA.
template <>
struct ldpc_dec_soft_output_params<__half2>
{
    uint32_t* dst_gmem;         // output address for this CTA
    int       num_cw_values;
    int       out_stride_elems; // needed for 2x codewords, where output stores two consecutive outputs
    int       num_out_cw;       // number of output codewords (by this CTA!), needed for 2x codeword kernels only

    // We use __half for output_LLR_addr, even though the kernel uses __half2,
    // because the outputs are separated.
    __device__
    ldpc_dec_soft_output_params(const LDPC_kernel_params& params) :
        dst_gmem(output_LLR_addr<__half>::get(params, blockIdx.x * codewords_per_CTA<__half2>::value)),
        num_cw_values(get_num_output_bits(params)),
        out_stride_elems(params.soft_out_stride_elements),
        num_out_cw(num_cta_output_codewords<__half2>(params.num_codewords))
    {
    }
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc)),
        out_stride_elems(0),
        num_out_cw(0)
    {
        int blkIndex = blockIdx.x;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                int iBlocksClaimed = (decodeDesc.tb_output[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    out_stride_elems = decodeDesc.llr_output[i].stride_elements;
                    __half* h = static_cast<__half*>(decodeDesc.llr_output[i].addr);
                    h += (blkIndex * 2 * out_stride_elems);
                    dst_gmem         = reinterpret_cast<uint32_t*>(h);
                    // Last block may have only 1 codeword...
                    num_out_cw       = ((blkIndex*2 + 1) == decodeDesc.llr_output[i].num_codewords) ? 1 : 2;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    // Constructor to set up the soft output parameters using the transport
    // block token, used by some kernels to store information on the specific
    // codeword targeted by a CTA.
    __device__
    ldpc_dec_soft_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                const output_token&          out_tok) :
        dst_gmem(nullptr),
        num_cw_values(get_num_output_bits(decodeDesc)),
        out_stride_elems(0),
        num_out_cw(0)
    {
        int  tb          = tb_from_token(out_tok.token);
        int  offset      = offset_from_token(out_tok.token);
        bool is_partial  = is_partial_from_token(out_tok.token);
        out_stride_elems = decodeDesc.llr_output[tb].stride_elements;
        __half* h = static_cast<__half*>(decodeDesc.llr_output[tb].addr);
        dst_gmem         = reinterpret_cast<uint32_t*>(h + (offset * out_stride_elems));
        num_out_cw       = is_partial ? 1 : 2;
    }
};


////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if(idx + ii * THREADS_PER_WARP < NODES * Z)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half*                    app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        __half app = __float2half(0.0f);
//        if((idx + ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_impl()
//template <typename T>
//static inline __device__ void ldpc_dec_output_variable_impl(const ldpc_dec_output_params<T>& params,
//                                                            const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if((idx + ii * THREADS_PER_WARP) < params.num_cw_bits)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_impl()
template <typename T>
inline __device__ void ldpc_dec_output_variable_impl(const ldpc_dec_output_params<T>& params,
                                                     const T*                         app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const T APP     = (APP_IDX < params.num_cw_bits) ?
                              app_smem[APP_IDX]              :
                              default_llr_value<T>::value();
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop_impl()
template <typename T>
inline __device__ void ldpc_dec_output_variable_loop_impl(const ldpc_dec_output_params<T>& params,
                                                          const T*                         app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const T APP     = (APP_IDX < params.num_cw_bits) ?
                              app_smem[APP_IDX]              :
                              default_llr_value<T>::value();
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
        // Advance
        output_idx += (NUM_ACTIVE_WARPS * THREADS_PER_WARP);
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
template <typename T>
inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
                                                const T*                         app_smem)
{
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
template <typename T>
inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<T>& params,
                                                     const T*                         app_smem)
{
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                const __nv_fp8_e5m2*      app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e5m2> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                const __nv_fp8_e5m2*         app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e5m2> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                const __nv_fp8_e4m3*      app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e4m3> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                const __nv_fp8_e4m3*         app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e4m3> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
inline __device__ void ldpc_dec_output_variable_multi(const LDPC_kernel_params&    kernelParams,
                                                      const __half*                app_smem,
                                                      const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES   = round_up_to_next(get_num_LLRs(kernelParams),
                                                     static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    const int OUTPUT_STRIDE_BYTES = sizeof(uint32_t) * kernelParams.output_stride_words;
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        char* dst = kernelParams.out + ((mconfig.cta_start_index + i) * OUTPUT_STRIDE_BYTES);
        ldpc_dec_output_params<__half> params(reinterpret_cast<uint32_t*>(dst),               // output address
                                              get_num_output_bits(kernelParams),              // output bits
                                              (get_num_output_bits(kernelParams) + 31) / 32); // num output words
        ldpc_dec_output_variable_impl(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
inline __device__ void ldpc_dec_output_variable_multi(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                      const __half*                app_smem,
                                                      const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES = round_up_to_next(get_num_LLRs(decodeDesc),
                                                   static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_output_params<__half> params(decodeDesc,
                                              mconfig.cta_start_index + i);
        ldpc_dec_output_variable_impl(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __half APP type using the LDPC_kernel_params struct
inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                     const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __half APP type using the cuphyLDPCDecodeDesc_t struct
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __nv_fp8_e5m2 APP type using the LDPC_kernel_params struct
inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                     const __nv_fp8_e5m2*      app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e5m2> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __nv_fp8_e5m2 APP type using the cuphyLDPCDecodeDesc_t struct
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     const __nv_fp8_e5m2*         app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e5m2> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __nv_fp8_e4m3 APP type using the LDPC_kernel_params struct
inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                     const __nv_fp8_e4m3*      app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e4m3> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __nv_fp8_e4m3 APP type using the cuphyLDPCDecodeDesc_t struct
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     const __nv_fp8_e4m3*         app_smem)
{
    ldpc_dec_output_params<__nv_fp8_e4m3> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                tb_token                     token,
                                                const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     tb_token                     token,
                                                     const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_impl()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
inline __device__ void ldpc_dec_output_variable_impl(const ldpc_dec_output_params<__half2>& params,
                                                     const __half2*                         app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop_impl()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
inline __device__ void ldpc_dec_output_variable_loop_impl(const ldpc_dec_output_params<__half2>& params,
                                                          const __half2*                         app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
        // Advance
        output_idx += (NUM_ACTIVE_WARPS * THREADS_PER_WARP);
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<__half2>& params,
                                                const __half2*                         app_smem)
{
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<__half2>& params,
                                                     const __half2*                         app_smem)
{
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                tb_token                     token,
                                                const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __half2 APP type (2 codewords per CTA) using the
// LDPC_kernel_params struct
inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                     const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload for __half2 APP type (2 codewords per CTA) using the
// cuphyLDPCDecodeDesc_t struct
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     tb_token                     token,
                                                     const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_output_variable_loop_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_x2_all_warps()
// Streamlined post-iteration hard-decision pack for the BG1 band decoders: a
// single fully-unrolled, guard-free ballot transpose spread across ALL
// blockDim.x/32 warps, replacing the runtime-bounded, per-element-guarded
// do-while of ldpc_dec_output_variable_loop above. It is the output tail --
// serialized and dependency-latency-bound, run once after the final BP barrier.
//
// PROVABLY BIT-EXACT and OOB-free for this config: warp w packs the contiguous
// word block [w*WORDS_PER_WARP, (w+1)*WORDS_PER_WARP); with
// WORDS_PER_WARP*NUM_FULL_WARPS*32 == num_cw_bits EVERY read index is
// < num_cw_bits, so the unconditional load returns exactly what the guarded
// version's in-range branch returns, making the packed output bit-identical.
// The store keeps only the WORDS_PER_WARP lanes that actually hold a word.
//
// WORDS_PER_WARP is a template (compile-time) parameter so the pass loop fully
// unrolls. Caller must guarantee
// out_words_per_cw == WORDS_PER_WARP * (blockDim.x/32) and that this product
// equals num_cw_bits/32 (true for BG1/Z384/mb4: 22 * 12 == 264 == 8448/32).
template <int WORDS_PER_WARP>
static inline __device__ void ldpc_dec_output_x2_all_warps(const ldpc_dec_output_params<__half2>& params,
                                                           const __half2*                         app_smem)
{
    enum { THREADS_PER_WARP = 32 };
    const int WARP_IDX  = threadIdx.x / THREADS_PER_WARP;
    const int LANE      = threadIdx.x % THREADS_PER_WARP;
    // First codeword bit this warp's block covers, plus this lane's offset.
    const int start_idx = WARP_IDX * (WORDS_PER_WARP * THREADS_PER_WARP) + LANE;

    uint32_t output[2] = {0, 0};
    #pragma unroll
    for(int ii = 0; ii < WORDS_PER_WARP; ++ii)
    {
        const int APP_IDX = start_idx + (ii * THREADS_PER_WARP);
        // Unconditional, always-in-range load (see header): index < num_cw_bits.
        __half2 fp16x2(app_smem[APP_IDX]);
        unsigned int vote0 = __ballot_sync(c_ldpc_full_warp_mask, (llr_hard_decision(fp16x2.x)));
        unsigned int vote1 = __ballot_sync(c_ldpc_full_warp_mask, (llr_hard_decision(fp16x2.y)));
        if(LANE == ii)
        {
            output[0] = vote0;
            output[1] = vote1;
        }
    }
    // Only the WORDS_PER_WARP lanes that hold a packed word write; lanes
    // >= WORDS_PER_WARP held no ii and must not alias the next warp's block.
    const int output_idx = WARP_IDX * WORDS_PER_WARP + LANE;
    // BG1/Z384 has exactly 22 words per warp across 12 warps, so the lane
    // predicate already proves output_idx is in range for algo103.
    if(LANE < WORDS_PER_WARP)
    {
        params.dst_gmem[output_idx] = output[0];
        if(2 == params.num_out_cw)
        {
            params.dst_gmem[output_idx + params.out_stride_words] = output[1];
        }
    }
}

template <int WORDS_PER_WARP>
static inline __device__ void ldpc_dec_output_x2_all_warps(const LDPC_kernel_params& kernelParams,
                                                           const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_x2_all_warps<WORDS_PER_WARP>(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Fixed" output functions use parameters (NODES, Z) that are known at
// compile time.
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half2*                   app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    const int WARP_IDX = threadIdx.x / THREADS_PER_WARP;
//    const int LANE     = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output[2] = {0, 0};
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = (WARP_IDX * 32 * THREADS_PER_WARP) + LANE;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        word_t app;
//        app.u32 = 0;
//        if(idx + (ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app.f16x2 = app_smem[idx + (ii * THREADS_PER_WARP)];
//        }
//        word_t app_sign_mask = fp16x2_sign_mask(app);
//
//        unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
//        unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
//        if(LANE == ii)
//        {
//            output[0] = vote0;
//            output[1] = vote1;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x]                             = output[0];
//        // Avoid writes past the end of the output when the number of
//        // codewords is odd.
//        if(2 == params.num_out_cw)
//        {
//            params.dst_gmem[threadIdx.x + params.out_stride_words] = output[1];
//        }
//    }
//}


////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_impl()
inline __device__ void ldpc_dec_soft_output_impl(const ldpc_dec_soft_output_params<__half>& params,
                                                 const __half*                             app_smem)
{
    // Each thread will write a pair of fp16 values as a uint32_t to
    // the global memory address.
    // We are currently writing an LLR value for information bits, and it does
    // not appear that this can be an odd number. (There are 22 info nodes for
    // BG1, and the only odd number of info nodes for BG2 is 9, but this only
    // occurs for Z values that are even (60 and 64).) However, we will keep
    // the logic here in case we change the number of values written in the
    // future to allow odd numbers of values.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 2);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 2))
    {
        word_t w;
        w.u32 = 0;
        if((app_idx + 1) == NUM_VALUES)
        {
            // Odd number of values: load a single value
            w.f16x2.x = app_smem[app_idx];
        }
        else
        {
            // Load a pair of values
            w.u32 = *reinterpret_cast<const uint32_t*>(app_smem + app_idx);
        }
        // Adjust the index to account for 32-bit writes and store to global memory
        params.dst_gmem[app_idx / 2] = w.u32;
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
inline __device__ void ldpc_dec_soft_output(const ldpc_dec_soft_output_params<__half>& params,
                                            const __half*                             app_smem)
{
    ldpc_dec_soft_output_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_impl()
template <typename TFP8>
inline __device__ void ldpc_dec_soft_output_impl_fp8(const ldpc_dec_soft_output_params<TFP8>& params,
                                                     const TFP8*                              app_smem)
{
    // Each thread will write a 4 fp8 values as a uint32_t to
    // the global memory address.
    // We are currently writing an LLR value for information bits, and
    // this function assumed that the total number is a multiple of 4.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 4);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 4))
    {
        word_t w;
        // Load 4 values
        w.u32 = *reinterpret_cast<const uint32_t*>(app_smem + app_idx);
        // Adjust the index to account for 32-bit writes and store to global memory
        params.dst_gmem[app_idx / 4] = w.u32;
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
// Adaptor function for writing soft outputs with the legacy tensor
// interface to the LDPC decoder. Constructs an instance of the
// ldpc_dec_soft_output_params structure that is common to both the
// legacy and transport block interfaces, and forwards to the function
// with the actual write logic.
inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                            const __half*             app_smem)
{
    ldpc_dec_soft_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output_impl(params, app_smem);
}

inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                            const __nv_fp8_e4m3*      app_smem)
{
    ldpc_dec_soft_output_params<__nv_fp8_e4m3> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8(params, app_smem);
}

inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                            const __nv_fp8_e5m2*      app_smem)
{
    ldpc_dec_soft_output_params<__nv_fp8_e5m2> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8(params, app_smem);
}


////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output()
// Adaptor function for writing soft outputs with the transport block
// interface to the LDPC decoder. Constructs an instance of the
// ldpc_dec_soft_output_params structure that is common to both the
// legacy and transport block interfaces, and forwards to the function
// with the actual write logic.
inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            tb_token                     token,
                                            const __half*                app_smem)
{
    ldpc_dec_soft_output_params<__half> params(decodeDesc, output_token(token));
    ldpc_dec_soft_output_impl(params, app_smem);
}
inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            const __half*                app_smem)
{
    ldpc_dec_soft_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output_impl(params, app_smem);
}
inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            const __nv_fp8_e4m3*         app_smem)
{
    ldpc_dec_soft_output_params<__nv_fp8_e4m3> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8(params, app_smem);
}
inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            const __nv_fp8_e5m2*         app_smem)
{
    ldpc_dec_soft_output_params<__nv_fp8_e5m2> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8(params, app_smem);
}


inline __device__ void ldpc_dec_soft_output_impl(const ldpc_dec_soft_output_params<__half2>& params,
                                                 const __half2*                             app_smem)
{
    // Each thread will write a pair of fp16 values as a uint32_t to
    // the global memory address for EACH of TWO CODEWORDS.
    // We are currently writing an LLR value for information bits, and it does
    // not appear that this can be an odd number. (There are 22 info nodes for
    // BG1, and the only odd number of info nodes for BG2 is 9, but this only
    // occurs for Z values that are even (60 and 64).) However, we will keep
    // the logic here in case we change the number of values written in the
    // future to allow odd numbers of values.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 2);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 2))
    {
        word_t src0, src1, dst0, dst1;
        src0.u32 = src1.u32 = 0;
        src0.f16x2 = app_smem[app_idx];
        if((app_idx + 1) < NUM_VALUES)
        {
            // Load a second pair of values
            src1.f16x2 = app_smem[app_idx + 1];
        }
        dst0.f16x2 = __lows2half2(src0.f16x2, src1.f16x2);
        dst1.f16x2 = __highs2half2(src0.f16x2, src1.f16x2);
        // Shuffle from interleaved values to per-codeword values
        // Adjust the index to account for 32-bit stores to global memory (instead of
        // 16-bit half precision values).
        int u32_idx = app_idx / 2;
        params.dst_gmem[u32_idx] = dst0.u32;
        // For 2 codeword per CTA kernels, we may not have a second codeword
        if(2 == params.num_out_cw)
        {
            u32_idx = (app_idx + params.out_stride_elems) / 2;
            params.dst_gmem[u32_idx] = dst1.u32;
        }
    }
}

inline __device__ void ldpc_dec_soft_output(const ldpc_dec_soft_output_params<__half2>& params,
                                            const __half2*                             app_smem)
{
    ldpc_dec_soft_output_impl(params, app_smem);
}

inline __device__ void ldpc_dec_soft_output(const LDPC_kernel_params& kernelParams,
                                            const __half2*            app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(kernelParams);
    ldpc_dec_soft_output_impl(params, app_smem);
}

inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            tb_token                     token,
                                            const __half2*               app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(decodeDesc, output_token(token));
    ldpc_dec_soft_output_impl(params, app_smem);
}
inline __device__ void ldpc_dec_soft_output(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                            const __half2*               app_smem)
{
    ldpc_dec_soft_output_params<__half2> params(decodeDesc);
    ldpc_dec_soft_output_impl(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_multi()
inline __device__ void ldpc_dec_soft_output_multi(const LDPC_kernel_params&    kernelParams,
                                                  const __half*                app_smem,
                                                  const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES   = round_up_to_next(get_num_LLRs(kernelParams),
                                                     static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_soft_output_params<__half> params(kernelParams, mconfig.cta_start_index + i);
        ldpc_dec_soft_output_impl(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_multi()
inline __device__ void ldpc_dec_soft_output_multi(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                  const __half*                app_smem,
                                                  const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES = round_up_to_next(get_num_LLRs(decodeDesc),
                                                   static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_output_params<__half> params(decodeDesc,
                                              mconfig.cta_start_index + i);
        ldpc_dec_output_variable_impl(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

template <typename TFP8, int TCount>
struct fp8_vector_type_t;

template <>
struct fp8_vector_type_t<__nv_fp8_e4m3, 2>
{
    using type = __nv_fp8x2_e4m3;
};
template <>
struct fp8_vector_type_t<__nv_fp8_e5m2, 2>
{
    using type = __nv_fp8x2_e5m2;
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_impl_fp8_to_fp16()
template <typename TFP8>
inline __device__ void ldpc_dec_soft_output_impl_fp8_to_fp16(const ldpc_dec_soft_output_params<__half>& params,
                                                             const TFP8*                                app_smem)
{
    using fp8x2_t = typename fp8_vector_type_t<TFP8, 2>::type;

    // Each thread will write 2 fp16 values as a uint32_t to
    // the global memory address.
    // We are currently writing an LLR value for information bits, and
    // this function assumed that the total number is a multiple of 2.
    const int NUM_VALUES = params.num_cw_values;
    for(int app_idx = (threadIdx.x * 2);
        app_idx < NUM_VALUES;
        app_idx += (blockDim.x * 2))
    {
        word_t w;
        // Load 2 fp8 values
        fp8x2_t src = *reinterpret_cast<const fp8x2_t*>(app_smem + app_idx);
        w.f16x2 = static_cast<__half2>(src);
        // Adjust the index to account for 32-bit writes and store to global memory
        params.dst_gmem[app_idx / 2] = w.u32;
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_soft_output_convert()
// Write soft outputs using a data type different than the type used
// to store APP values inside the kernel.
template <typename TDst, typename TSrc>
__device__
void ldpc_dec_soft_output_convert(const LDPC_kernel_params& kernelParams,
                                  const TSrc*               app_smem);

template <>
__device__ inline
void ldpc_dec_soft_output_convert<__half, __nv_fp8_e4m3>(const LDPC_kernel_params& kernelParams,
                                                         const __nv_fp8_e4m3*      app_smem)
{
    // Output params use the output variable type (__half)
    ldpc_dec_soft_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8_to_fp16<__nv_fp8_e4m3>(params, app_smem);
}

template <>
__device__ inline
void ldpc_dec_soft_output_convert<__half, __nv_fp8_e5m2>(const LDPC_kernel_params& kernelParams,
                                                         const __nv_fp8_e5m2*      app_smem)
{
    // Output params use the output variable type (__half)
    ldpc_dec_soft_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8_to_fp16<__nv_fp8_e5m2>(params, app_smem);
}

template <typename TDst, typename TSrc>
__device__
void ldpc_dec_soft_output_convert(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                  const TSrc*                  app_smem);

template <>
inline __device__ void ldpc_dec_soft_output_convert<__half, __nv_fp8_e4m3>(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                                           const __nv_fp8_e4m3*         app_smem)
{
    // Output params use the output variable type (__half)
    ldpc_dec_soft_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8_to_fp16<__nv_fp8_e4m3>(params, app_smem);
}
template <>
inline __device__ void ldpc_dec_soft_output_convert<__half, __nv_fp8_e5m2>(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                                           const __nv_fp8_e5m2*         app_smem)
{
    // Output params use the output variable type (__half)
    ldpc_dec_soft_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_soft_output_impl_fp8_to_fp16<__nv_fp8_e5m2>(params, app_smem);
}


// extract_APP()
// Some kernels decode 2 CBs in one CTA and keep APP values interleaved in
// __half2. Extract one codeword's scalar APP value for diagnostics.
__device__ inline __half extract_APP(const __half2& v, int cb_sel)
{
    return cb_sel ? v.y : v.x;
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_interm_results
// Device-side wrapper around the optional intermediate-result descriptor for
// one codeword. NULL result buffers mean skip without side effects.
struct ldpc_dec_interm_results
{
    const cuphyTransportBlockIntermResults_t* results_p;
    int                                       cw_idx_in_TB;

    __device__
    ldpc_dec_interm_results(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
        : results_p(nullptr), cw_idx_in_TB(0)
    {
        if(nullptr == decodeDesc.interm_results) { return; }
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.interm_results[i].num_codewords)
                {
                    results_p    = &decodeDesc.interm_results[i];
                    cw_idx_in_TB = cwIndex;
                    break;
                }
                cwIndex -= decodeDesc.interm_results[i].num_codewords;
            }
        }
    }

    template <typename T>
    __device__ T* get_app_base(int itr) const
    {
        // Precondition: results_p != nullptr -- write_app / write_interleaved_app
        // guard on results_p (and app_addr) before calling this.
        return static_cast<T*>(results_p->app_addr) +
               cw_idx_in_TB * results_p->app_stride_elements_cw +
               itr * results_p->app_stride_elements_itr;
    }

    template <typename T>
    __device__ void write_app(const T* app_smem, int itr, int num_values) const
    {
        if(results_p == nullptr || results_p->app_addr == nullptr) { return; }
        T* dst = get_app_base<T>(itr);
        for(int idx = threadIdx.x; idx < num_values; idx += blockDim.x)
        {
            dst[idx] = app_smem[idx];
        }
    }

    template <typename TPair>
    __device__ void write_interleaved_app(const TPair* app_smem,
                                          int          itr,
                                          int          num_values,
                                          int          cb_sel) const
    {
        if(results_p == nullptr || results_p->app_addr == nullptr) { return; }
        __half* dst = get_app_base<__half>(itr);
        for(int idx = threadIdx.x; idx < num_values; idx += blockDim.x)
        {
            dst[idx] = extract_APP(app_smem[idx], cb_sel);
        }
    }
};

template <bool enable, typename AppBufT>
__device__ __forceinline__ void ldpc_dump_app(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                              const AppBufT*        app_smem,
                                              int                   itr,
                                              int                   num_values)
{
    if constexpr(enable)
    {
        if(0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_DUMP_INTERM))
        {
            ldpc_dec_interm_results(decodeDesc, blockIdx.x).write_app(app_smem, itr, num_values);
            __syncthreads();
        }
    }
}

template <bool enable, typename TPair>
__device__ __forceinline__ void ldpc_dump_app_x2(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                 uint32_t              tok,
                                                 const TPair*          app_smem,
                                                 int                   itr,
                                                 int                   num_values)
{
    if constexpr(enable)
    {
        if(0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_DUMP_INTERM))
        {
            // offset_from_token is TB-local; ldpc_dec_interm_results indexes
            // decodeDesc-wide, so add the codewords of the preceding TBs.
            const int tb  = tb_from_token(tok);
            int       cw0 = offset_from_token(tok);
            if(nullptr != decodeDesc.interm_results)
            {
                for(int i = 0; i < tb && i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
                {
                    cw0 += decodeDesc.interm_results[i].num_codewords;
                }
            }
            ldpc_dec_interm_results(decodeDesc, cw0).write_interleaved_app(app_smem, itr, num_values, 0);
            if(!is_partial_from_token(tok))
            {
                ldpc_dec_interm_results(decodeDesc, cw0 + 1).write_interleaved_app(app_smem, itr, num_values, 1);
            }
            __syncthreads();
        }
    }
}

} // namespace ldpc2

#endif // !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)
