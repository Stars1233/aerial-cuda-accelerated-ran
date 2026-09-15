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

#if !defined(UCI_ON_PUSCH_COMMON_CUH_INCLUDED_)
#define UCI_ON_PUSCH_COMMON_CUH_INCLUDED_

#include "cuphy_internal.h"
#include <cuda_fp16.h>

namespace uci_on_pusch_common {

static constexpr int MAX_BITS_PER_RE     = 32;
static constexpr int N_SC_PER_PRB        = 12;

/**
 * Unified resource element grid structure for UCI on PUSCH operations
 *
 * @param nRes Number of resource elements
 * @param ReStride Resource element stride
 * @param rmBufferOffset Rate matching buffer offset
 * @param gridOffset Grid offset value
 */
struct reGrid_t {
    uint16_t nRes           = 0;
    uint16_t ReStride       = 0;
    uint32_t rmBufferOffset = 0;
    uint8_t  gridOffset     = 0;
};

/**
 * Assign resource element grid parameters
 *
 * @param[in,out] reGrid Resource element grid structure to update
 * @param[in] nUnassignedResInSymbol Number of unassigned REs in symbol
 * @param[in] nUnassignedBitsInSymbol Number of unassigned bits in symbol
 * @param[in,out] nAssignedRmBits Number of assigned rate matching bits
 * @param[in] G Total number of coded bits
 * @param[in] nBitsPerRe Number of bits per resource element
 */
__host__ __device__ __forceinline__
void assignReGrid(reGrid_t& reGrid,
                  uint16_t  nUnassignedResInSymbol,
                  uint32_t  nUnassignedBitsInSymbol,
                  uint32_t& nAssignedRmBits,
                  uint32_t  G,
                  uint8_t   nBitsPerRe)
{
    reGrid.rmBufferOffset = nAssignedRmBits;

    const uint32_t need = G - nAssignedRmBits;

    // Early exit - not expected to happen
    if (need == 0u || nUnassignedResInSymbol == 0u || nUnassignedBitsInSymbol == 0u) {
        reGrid.nRes      = 0;
        reGrid.ReStride  = 1;
        reGrid.gridOffset = 0;
        return;
    }

    if (need > nUnassignedBitsInSymbol) {
        // Take everything densely
        reGrid.nRes     = nUnassignedResInSymbol;
        reGrid.ReStride = 1;
    } else {
        // Take just enough; stride is safe since 'need' > 0 here
        reGrid.nRes     = div_round_up(need, static_cast<uint32_t>(nBitsPerRe));
        reGrid.ReStride = nUnassignedBitsInSymbol / need;
    }

    nAssignedRmBits += (uint32_t)reGrid.nRes * (uint32_t)nBitsPerRe;
    reGrid.gridOffset = 0;
}


/**
 * Update number of unassigned resource elements and bits
 *
 * @param[in] reGrid Resource element grid structure
 * @param[in,out] nUnassignedResInSymbol Number of unassigned REs in symbol
 * @param[in,out] nUnassignedBitsInSymbol Number of unassigned bits in symbol
 * @param[in] nBitsPerRe Number of bits per resource element
 */
__host__ __device__ __forceinline__ void updateNumUnassigned(reGrid_t& reGrid,
                                                             uint16_t& nUnassignedResInSymbol,
                                                             uint32_t& nUnassignedBitsInSymbol,
                                                             uint8_t   nBitsPerRe)
{
    nUnassignedResInSymbol  = nUnassignedResInSymbol  - reGrid.nRes;
    nUnassignedBitsInSymbol = nUnassignedBitsInSymbol - reGrid.nRes * nBitsPerRe;
}

/**
 * Check if resource element is assigned to the rate matching buffer
 *
 * @param[in] reGrid Resource element grid structure
 * @param[in] virtualReIdx Virtual resource element index
 * @param[out] assignFlag Flag indicating if RE is assigned
 * @param[out] cumltNumAssignedRes Cumulative number of assigned resource elements
 */
__device__ __forceinline__ void checkIfReAssigned(reGrid_t& reGrid,
                                                  uint16_t  virtualReIdx,
                                                  bool&     assignFlag,
                                                  uint16_t& cumltNumAssignedRes)
{
    uint16_t r = virtualReIdx % reGrid.ReStride;
    uint16_t d = virtualReIdx / reGrid.ReStride;

    // Check if RE is assigned to the RM buffer
    if(r != reGrid.gridOffset)
    {
        assignFlag          = false;
        cumltNumAssignedRes = ((d + 1) >= reGrid.nRes) ? reGrid.nRes : (d + 1);
    }
    else if(d >= reGrid.nRes)
    {
        assignFlag          = false;
        cumltNumAssignedRes = reGrid.nRes;
    }
    else
    {
        assignFlag          = true;
        cumltNumAssignedRes = d;
    }
}


/**
 * @param[in] reDescSeq Descrambling sequence for Resource Element (RE)
 * @param[in] spx1Flag Flag for scrambling adjustment (0 or 1)
 * @param[out] dst Output: Rate Matching buffer
 * @param[in] nBitsPerRe Number of bits per RE
 * @param[in] src Input: RE LLRs
 */
__device__ __forceinline__
void descramAndStoreLLRs(uint32_t             reDescSeq,  // Descrambling sequence for Resource Element (RE)
                         uint8_t              spx1Flag,   // Flag for scrambling adjustment (0 or 1)
                         __half* __restrict__ dst,        // Output: Rate Matching buffer
                         uint8_t              nBitsPerRe, // Number of bits per RE
                         const __half* __restrict__ src)  // Input: RE LLRs
{
    // Helper to compute shifted-bit (odd indices shift by spx1_flag)
    auto scramble_bit = [&](int bitIdx) {
        const int shift = (bitIdx & 1) ? (bitIdx - spx1Flag) : bitIdx;
        return (reDescSeq >> shift) & 1u;
    };

    // Store RE LLRs into RM buffer. Try widest aligned path first
    // Check alignment of input and output pointers for vectorized operations
    const uintptr_t alignment_check = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);

    // 128-bit: process 8 halves per iteration
    if ( ((nBitsPerRe & 7) == 0) && ((alignment_check & 15) == 0) )
    {
        auto* __restrict__ d = reinterpret_cast<uint4*>(dst);
        auto const* __restrict__ s = reinterpret_cast<const uint4*>(src);

        #pragma unroll
        for (int i = 0; i < nBitsPerRe / 8; ++i)
        {
            // Load 8 halves (16 bytes) as 4x32b words
            uint4 v = s[i];

            // Build a 128-bit sign-mask for these 8 halves (0x8000 per half if flip)
            uint32_t m0 = (scramble_bit(8*i+0) ? 0x00008000u : 0u) |
                          (scramble_bit(8*i+1) ? 0x80000000u : 0u);
            uint32_t m1 = (scramble_bit(8*i+2) ? 0x00008000u : 0u) |
                          (scramble_bit(8*i+3) ? 0x80000000u : 0u);
            uint32_t m2 = (scramble_bit(8*i+4) ? 0x00008000u : 0u) |
                          (scramble_bit(8*i+5) ? 0x80000000u : 0u);
            uint32_t m3 = (scramble_bit(8*i+6) ? 0x00008000u : 0u) |
                          (scramble_bit(8*i+7) ? 0x80000000u : 0u);

            // XOR sign bits and store
            d[i] = make_uint4(v.x ^ m0, v.y ^ m1, v.z ^ m2, v.w ^ m3);
        }
        return;
    }

    // 64-bit: 4 halves per iteration
    if ( ((nBitsPerRe & 3) == 0) && ((alignment_check & 7) == 0) )
    {
        auto* __restrict__ d = reinterpret_cast<uint2*>(dst);
        auto const* __restrict__ s = reinterpret_cast<const uint2*>(src);

        #pragma unroll
        for (int i = 0; i < nBitsPerRe / 4; ++i)
        {
            uint2 v = s[i];
            uint32_t m0 = (scramble_bit(4*i+0) ? 0x00008000u : 0u) |
                          (scramble_bit(4*i+1) ? 0x80000000u : 0u);
            uint32_t m1 = (scramble_bit(4*i+2) ? 0x00008000u : 0u) |
                          (scramble_bit(4*i+3) ? 0x80000000u : 0u);
            d[i] = make_uint2(v.x ^ m0, v.y ^ m1);
        }
        return;
    }

    // 32-bit: 2 halves per iteration (very fast and simple; good fallback)
    if ( ((nBitsPerRe & 1) == 0) && ((alignment_check & 3) == 0) )
    {
        auto* __restrict__ d = reinterpret_cast<uint32_t*>(dst);
        auto const* __restrict__ s = reinterpret_cast<const uint32_t*>(src);

        #pragma unroll
        for (int i = 0; i < nBitsPerRe / 2; ++i)
        {
            uint32_t v = s[i];
            // lane0 at bits[15:0], lane1 at bits[31:16]
            uint32_t m = (scramble_bit(2*i+0) ? 0x00008000u : 0u) |
                         (scramble_bit(2*i+1) ? 0x80000000u : 0u);
            d[i] = v ^ m;
        }
        return;
    }

    // 16-bit scalar tail / odd counts (shouldn’t trigger for 5G Qm, but to be safe)
    #pragma unroll
    for (int i = 0; i < nBitsPerRe; ++i)
    {
        // flip sign by XORing sign bit
        uint16_t h = reinterpret_cast<const uint16_t*>(src)[i];
        if (scramble_bit(i)) h ^= 0x8000u;
        reinterpret_cast<uint16_t*>(dst)[i] = h;
    }
}

__device__ __forceinline__
uint16_t descramMaskHalf(uint32_t reDescSeq, uint8_t spx1Flag, int bitIdx)
{
    const int shift = (bitIdx & 1) ? (bitIdx - spx1Flag) : bitIdx;
    return ((reDescSeq >> shift) & 1u) ? 0x8000u : 0u;
}

__device__ __forceinline__
uint32_t descramMaskPair(uint32_t reDescSeq, uint8_t spx1Flag, int bitIdx)
{
    return static_cast<uint32_t>(descramMaskHalf(reDescSeq, spx1Flag, bitIdx)) |
           (static_cast<uint32_t>(descramMaskHalf(reDescSeq, spx1Flag, bitIdx + 1)) << 16);
}

__device__ __forceinline__
void descramAndStoreZeroLLRs16(uint32_t             reDescSeq,
                               uint8_t              spx1Flag,
                               __half* __restrict__ dst)
{
    uint32_t m0 = descramMaskPair(reDescSeq, spx1Flag, 0);
    uint32_t m1 = descramMaskPair(reDescSeq, spx1Flag, 2);
    uint32_t m2 = descramMaskPair(reDescSeq, spx1Flag, 4);
    uint32_t m3 = descramMaskPair(reDescSeq, spx1Flag, 6);
    uint32_t m4 = descramMaskPair(reDescSeq, spx1Flag, 8);
    uint32_t m5 = descramMaskPair(reDescSeq, spx1Flag, 10);
    uint32_t m6 = descramMaskPair(reDescSeq, spx1Flag, 12);
    uint32_t m7 = descramMaskPair(reDescSeq, spx1Flag, 14);

    if((reinterpret_cast<uintptr_t>(dst) & 15) == 0)
    {
        auto* __restrict__ out = reinterpret_cast<uint4*>(dst);
        out[0] = make_uint4(m0, m1, m2, m3);
        out[1] = make_uint4(m4, m5, m6, m7);
    }
    else
    {
        auto* __restrict__ out = reinterpret_cast<uint16_t*>(dst);
        out[0]  = static_cast<uint16_t>(m0);
        out[1]  = static_cast<uint16_t>(m0 >> 16);
        out[2]  = static_cast<uint16_t>(m1);
        out[3]  = static_cast<uint16_t>(m1 >> 16);
        out[4]  = static_cast<uint16_t>(m2);
        out[5]  = static_cast<uint16_t>(m2 >> 16);
        out[6]  = static_cast<uint16_t>(m3);
        out[7]  = static_cast<uint16_t>(m3 >> 16);
        out[8]  = static_cast<uint16_t>(m4);
        out[9]  = static_cast<uint16_t>(m4 >> 16);
        out[10] = static_cast<uint16_t>(m5);
        out[11] = static_cast<uint16_t>(m5 >> 16);
        out[12] = static_cast<uint16_t>(m6);
        out[13] = static_cast<uint16_t>(m6 >> 16);
        out[14] = static_cast<uint16_t>(m7);
        out[15] = static_cast<uint16_t>(m7 >> 16);
    }
}

__device__ __forceinline__
void descramAndStoreZeroLLRs(uint32_t             reDescSeq,
                             uint8_t              spx1Flag,
                             __half* __restrict__ dst,
                             uint8_t              nBitsPerRe)
{
    if(nBitsPerRe == 16)
    {
        descramAndStoreZeroLLRs16(reDescSeq, spx1Flag, dst);
        return;
    }

    const uintptr_t alignment = reinterpret_cast<uintptr_t>(dst);

    if(((nBitsPerRe & 7) == 0) && ((alignment & 15) == 0))
    {
        auto* __restrict__ out = reinterpret_cast<uint4*>(dst);
        #pragma unroll
        for(int i = 0; i < nBitsPerRe / 8; ++i)
        {
            const int bitIdx = 8 * i;
            out[i] = make_uint4(descramMaskPair(reDescSeq, spx1Flag, bitIdx + 0),
                                descramMaskPair(reDescSeq, spx1Flag, bitIdx + 2),
                                descramMaskPair(reDescSeq, spx1Flag, bitIdx + 4),
                                descramMaskPair(reDescSeq, spx1Flag, bitIdx + 6));
        }
        return;
    }

    if(((nBitsPerRe & 3) == 0) && ((alignment & 7) == 0))
    {
        auto* __restrict__ out = reinterpret_cast<uint2*>(dst);
        #pragma unroll
        for(int i = 0; i < nBitsPerRe / 4; ++i)
        {
            const int bitIdx = 4 * i;
            out[i] = make_uint2(descramMaskPair(reDescSeq, spx1Flag, bitIdx + 0),
                                descramMaskPair(reDescSeq, spx1Flag, bitIdx + 2));
        }
        return;
    }

    if(((nBitsPerRe & 1) == 0) && ((alignment & 3) == 0))
    {
        auto* __restrict__ out = reinterpret_cast<uint32_t*>(dst);
        #pragma unroll
        for(int i = 0; i < nBitsPerRe / 2; ++i)
        {
            out[i] = descramMaskPair(reDescSeq, spx1Flag, 2 * i);
        }
        return;
    }

    auto* __restrict__ out = reinterpret_cast<uint16_t*>(dst);
    #pragma unroll
    for(int i = 0; i < nBitsPerRe; ++i)
    {
        out[i] = descramMaskHalf(reDescSeq, spx1Flag, i);
    }
}

__device__ __forceinline__
void descramAndStoreLLRsFromGlobal(uint32_t                     reDescSeq,
                                   uint8_t                      spx1Flag,
                                   __half* __restrict__         dst,
                                   const __half* __restrict__   gBase,
                                   int                          s1,
                                   int                          idx0,
                                   uint8_t                      nBitsPerQam,
                                   uint8_t                      nLayers,
                                   const uint32_t* __restrict__ pLayerMap)
{
    #pragma unroll
    for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
    {
        const __half* __restrict__ src = gBase + (pLayerMap[layerIdx] * s1 + idx0);
        __half* __restrict__       out = dst + layerIdx * nBitsPerQam;
        int                        bitIdx = layerIdx * nBitsPerQam;
        int                        q = nBitsPerQam;

        if(q >= 8 &&
           (((reinterpret_cast<uintptr_t>(src) |
              reinterpret_cast<uintptr_t>(out)) & 15) == 0))
        {
            uint4 v = *reinterpret_cast<const uint4*>(src);
            *reinterpret_cast<uint4*>(out) =
                make_uint4(v.x ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 0),
                           v.y ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 2),
                           v.z ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 4),
                           v.w ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 6));
            src += 8;
            out += 8;
            bitIdx += 8;
            q -= 8;
        }

        if(q >= 4 &&
           (((reinterpret_cast<uintptr_t>(src) |
              reinterpret_cast<uintptr_t>(out)) & 7) == 0))
        {
            uint2 v = *reinterpret_cast<const uint2*>(src);
            *reinterpret_cast<uint2*>(out) =
                make_uint2(v.x ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 0),
                           v.y ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx + 2));
            src += 4;
            out += 4;
            bitIdx += 4;
            q -= 4;
        }

        if(q >= 2 &&
           (((reinterpret_cast<uintptr_t>(src) |
              reinterpret_cast<uintptr_t>(out)) & 3) == 0))
        {
            uint32_t v = *reinterpret_cast<const uint32_t*>(src);
            *reinterpret_cast<uint32_t*>(out) = v ^ descramMaskPair(reDescSeq, spx1Flag, bitIdx);
            src += 2;
            out += 2;
            bitIdx += 2;
            q -= 2;
        }

        auto const* __restrict__ in16 = reinterpret_cast<const uint16_t*>(src);
        auto* __restrict__       out16 = reinterpret_cast<uint16_t*>(out);
        #pragma unroll
        for(int i = 0; i < q; ++i)
        {
            out16[i] = in16[i] ^ descramMaskHalf(reDescSeq, spx1Flag, bitIdx + i);
        }
    }
}

/**
 * Vectorized copying of LLRs from global memory to shared memory.
 *
 * When `Async == false` (the default) this emits the standard
 * synchronous LD + STS sequence — each 16/8/4 B aligned chunk copied
 * through registers, the 2 B tail as a scalar store. When
 * `Async == true` the same 16/8/4 B chunks are issued as
 * `cp.async.ca.shared.global` and a `cp.async.commit_group` is emitted
 * at the end; the caller must follow up with `cpLLRsVecWait<true>()`
 * before reading the destination `reLLRs`. The 2 B tail is always
 * synchronous (cp.async has a 4 B minimum).
 *
 * The two paths are kept in one function so the alignment/size decision
 * tree lives in a single place. `if constexpr` makes the unused branch
 * disappear at template instantiation, so Async=false instantiations
 * don't carry any cp.async code and Async=true instantiations don't
 * carry the register-copy path.
 *
 * @param[in] gBase Base pointer to global memory containing source LLRs
 * @param[in] s1 Stride for layer indexing in the source data layout
 * @param[in] idx0 Base index offset into the source data
 * @param[out] reLLRs Destination buffer (typically shared memory) for copied LLRs
 * @param[in] nBitsPerQam Number of bits per QAM modulation symbol
 * @param[in] nLayers Number of layers to process and copy
 * @param[in] pLayerMap Layer mapping array specifying which layers to copy
 * @tparam Async If true, issue cp.async copies instead of sync LD+STS
 *               (caller is responsible for the subsequent cp.async.wait).
 */
template <bool Async = false>
__device__ __forceinline__
void copyLLRsVec(const __half* __restrict__   gBase,
                 int                          s1,
                 int                          idx0,
                 __half* __restrict__         reLLRs,
                 uint8_t                      nBitsPerQam,
                 uint8_t                      nLayers,
                 const uint32_t* __restrict__ pLayerMap)
{
    #ifdef __CUDA_ARCH__
        // We need to guard the use of __cvta_generic_to_shared with __CUDA_ARCH__
        // because it is a non-template-dependent name, so the host-mode parse of this
        // TU (via cuphy.cpp #include) tries to resolve it at template-definition time.
        // Use a macro rather than a lambda because the size in the inline-asm needs to
        // be a literal in the string.
        #define ASYNC_CP_HELPER(dest, src, size) \
            do { \
                asm volatile("cp.async.ca.shared.global [%0], [%1], " #size ";\n" \
                            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dest))), \
                                "l"(src)); \
            } while(0)
    #else
        #define ASYNC_CP_HELPER(dest, src, size) do {} while(0)
    #endif

    #pragma unroll
    for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
    {
        const __half* __restrict__ g = gBase + (pLayerMap[layerIdx] * s1 + idx0);
        __half* __restrict__       s = reLLRs + layerIdx * nBitsPerQam;
        int                        q = nBitsPerQam;

        if(q >= 8 &&
           (((uintptr_t)g & 0xF) == 0) &&
           (((uintptr_t)s & 0xF) == 0))
        {
            if constexpr (Async)
            {
                ASYNC_CP_HELPER(s, g, 16);
            }
            else
            {
                *reinterpret_cast<uint4*>(s) = *reinterpret_cast<const uint4*>(g);
            }
            g += 8;
            s += 8;
            q -= 8;
        }
        // #pragma unroll 1 needed here to avoid excessive register pressure on sm_120
        #pragma unroll 1
        while(q >= 4 &&
              (((uintptr_t)g & 0x7) == 0) &&
              (((uintptr_t)s & 0x7) == 0))
        {
            if constexpr (Async)
            {
                ASYNC_CP_HELPER(s, g, 8);
            }
            else
            {
                *reinterpret_cast<unsigned long long*>(s) =
                    *reinterpret_cast<const unsigned long long*>(g);
            }
            g += 4;
            s += 4;
            q -= 4;
        }
        while(q >= 2 &&
              (((uintptr_t)g & 0x3) == 0) &&
              (((uintptr_t)s & 0x3) == 0))
        {
            if constexpr (Async)
            {
                ASYNC_CP_HELPER(s, g, 4);
            }
            else
            {
                *reinterpret_cast<unsigned*>(s) =
                    *reinterpret_cast<const unsigned*>(g);
            }
            g += 2;
            s += 2;
            q -= 2;
        }
        // 2 B tail: always synchronous; cp.async has a 4 B minimum.
        if(q) { s[0] = g[0]; }
    }
    if constexpr (Async)
    {
        // No __CUDA_ARCH__ guard needed here: unlike the cp.async.ca.shared.global
        // path (wrapped by ASYNC_CP_HELPER above), this asm has no CUDA-intrinsic
        // calls that would trigger first-phase name lookup in a host-mode parse.
        // The asm string is opaque to the C++ frontend, and copyLLRsVec<true> is
        // only ever instantiated from .cu TUs.
        asm volatile("cp.async.commit_group;\n");
    }
    #undef ASYNC_CP_HELPER
}

/**
 * Wait for outstanding cp.async groups issued by `copyLLRsVec<true>()`
 * to drain. Per-thread sync; safe to call without `__syncthreads()`
 * because each thread only reads its own reLLRs slice.
 *
 * When `Async == false` (default), this is a no-op — useful so callers
 * can write `cpLLRsVecWait<Async>()` unconditionally to match a
 * `copyLLRsVec<Async>()` call.
 */
template <bool Async = false>
__device__ __forceinline__
void cpLLRsVecWait()
{
    if constexpr (Async)
    {
        asm volatile("cp.async.wait_group 0;\n");
    }
}

} // namespace uci_on_pusch_common

#endif // !defined(UCI_ON_PUSCH_COMMON_CUH_INCLUDED_)
