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

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "rate_matching.hpp"
#include "descrambling.cuh"
#include "crc.hpp"
#include "descrambling.hpp"
#include "derate_matching_modulo.hpp"

// max possible value for NUM_LLRS_PROCESSED_PER_THRD is 32, as this should not exceed warp size
// using NUM_LLRS_PROCESSED_PER_THRD = 32 will decrease the number of CTAs, but it increases the number of sequential iterations
// using smaller NUM_LLRS_PROCESSED_PER_THRD would increase number of CTAs, as a result increases number of instructions that are per CTA,
// but it reduces the number of iterations in the main for loop in de_rate_matching_global2
#define NUM_LLRS_PROCESSED_PER_THRD 32


using namespace cuphy_i;
using namespace descrambling;
using namespace crc;



// Flip sign bits of two FP16 LLRs packed in a __half2 using scrambling bits.
//   seqWord0/1  : 32-bit Gold seq words for LLR0 and LLR1.
//   bit_index0/1: bit positions inside each word.
//   llr_pair    : {low, high} halves are the two LLRs.
// Builds a mask with bits in FP16 sign positions (15, 31) and XORs in place
__device__ inline __half2 rate_match_xor_sign_pair(uint32_t seqWord0, uint32_t seqWord1, int bit_index0, int bit_index1, __half2 llr_pair)
{
    const uint32_t s0   = (seqWord0 >> bit_index0) & 1u;
    const uint32_t s1   = (seqWord1 >> bit_index1) & 1u;
    const uint32_t mask = (s0 << 15) | (s1 << 31); // low-half sign @bit15, high-half @bit31

    uint32_t bits;
    memcpy(&bits, &llr_pair, sizeof(bits));
    bits ^= mask;
    memcpy(&llr_pair, &bits, sizeof(bits));

    return llr_pair;
}

__device__ __forceinline__
float2 rate_match_xor_sign_pair(uint32_t seqWord0, uint32_t seqWord1, int bit_index0, int bit_index1, float2 llr_pair)
{
    // Extract the two sign bits to apply
    const uint32_t s0 = (seqWord0 >> bit_index0) & 1u;  // for llr_pair.x
    const uint32_t s1 = (seqWord1 >> bit_index1) & 1u;  // for llr_pair.y

    // Build per-lane sign masks (float sign is bit 31)
    const uint32_t mask0 = s0 << 31;
    const uint32_t mask1 = s1 << 31;

    // Flip sign bits by XORing the bit patterns
    uint32_t bx = __float_as_uint(llr_pair.x) ^ mask0;
    uint32_t by = __float_as_uint(llr_pair.y) ^ mask1;

    llr_pair.x = __uint_as_float(bx);
    llr_pair.y = __uint_as_float(by);
    return llr_pair;
}

__device__ inline uint32_t compute_llr_index(int tid, uint32_t j, uint32_t k,
                                             uint32_t codeBlockQAMStartIndex, uint32_t adjustedCodeBlockQAMStartIndex,
                                             uint32_t Nl, uint32_t nBBULayers, uint32_t* layer_map_array, uint8_t uciOnPuschFlag)
{
    uint32_t llr_idx;
    if(uciOnPuschFlag)
    {
        llr_idx = codeBlockQAMStartIndex + tid;
    }
    else
    {
        if (Nl < nBBULayers)
        {
            // jl can be interpreted as reIdx as there are Nl qams for this user in a singe resource element
            uint32_t jl = j / Nl;
            // LLR buffer has dimension QAM_STRIDE x nBBULayers x nRe
            // First dimension: bitIdxInQam = k
            // Second dimension: layerIdxWithinUeGrp = layer_map_array[(j - jl * Nl)]
            // Third dimension:  reIdx = jl
            // The location of the llr would be: llrIdx = bitIdxInQam + (QAM_STRIDE * layerIdxWithinUeGrp + reIdx * nBBULayers * QAM_STRIDE)
            llr_idx = (adjustedCodeBlockQAMStartIndex * nBBULayers / Nl) + (k + (jl * nBBULayers + layer_map_array[(j - jl * Nl)]) * QAM_STRIDE);
        }
        else
        {
            // when nBBULayers==Nl, the general index logic above to retrieve llr can be simplified as below
            llr_idx = adjustedCodeBlockQAMStartIndex + k + j * QAM_STRIDE;
        }
    }
    return llr_idx;
}

template <typename T_OUT, bool ndi>
__device__ __forceinline__
void processOneLLR(  uint32_t           jIdx,          // = tid /  Qm
                     uint32_t           kIdx,          // = tid %  Qm
                     T_OUT              llr,           // preloaded
                     int                EoverQm,
                     /* --- rate-matching / combining ------------------------- */
                     uint32_t           Kd,
                     uint32_t           F,
                     uint32_t           k0,
                     uint32_t           Ncb,
                     int                potentialRaceIfPositive,
                     /* --- lower/upper bounds -------------------------------- */
                     T_OUT              LLR_CLAMP_MIN,
                     T_OUT              LLR_CLAMP_MAX,
                     /* --- destination --------------------------------------- */
                     T_OUT*    __restrict__ out)
{
    if (jIdx >= static_cast<uint32_t>(EoverQm)) return;   // out of range

    // de-rate-match index -------------------------------------------
    const uint32_t inIdx  = kIdx * EoverQm + jIdx;
    const uint32_t outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
    //outIdx = inIdx % (Ncb - F); // this will not generate the correct out index for all scenarios

    const bool useAtomics = (potentialRaceIfPositive > 0) && (outIdx < potentialRaceIfPositive);

    // Write / Combine -----------------------------------------------
    // ndi true: no LLR combining (new data)
    // ndi false: LLR combining (retransmission)
    if constexpr (ndi)
    {
        if(!useAtomics)
        {
            out[outIdx] = llr;
        }
        else
        {
            atomicAdd(out + outIdx, llr);
        }
    }
    else
    {
        if(!useAtomics)
        {
            llr += out[outIdx];
            // clamp the llr
            if constexpr (std::is_same<T_OUT, __half>::value)
            {
               llr = __hmax(__hmin(llr, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
            } else
            {
                llr = max(min(LLR_CLAMP_MAX, llr), LLR_CLAMP_MIN);
            }
            // write the updated LLR.  No need for atomic, different threads work on different outIdx.
            out[outIdx] = llr;
        }
        else
        {
            atomicAdd(out + outIdx, llr);
        }
    }
}

// Vectorized zero for a linear T_OUT-range [S, S+L) in out[].
// each CTA writes its own contiguous slice; uses 8-byte stores.
// ToDo: to be further optimized for devices with cc 10 and beyond
template <typename T_OUT>
__device__ __forceinline__
void zeroRangeVec(T_OUT* __restrict__ out,
                  uint32_t            start,
                  uint32_t            end,
                  uint32_t            tid,
                  uint32_t            stride)
{
    if(start >= end) return;

    // ----- Head: scalar store until we hit 16B alignment -----
    T_OUT*   base  = out + start;
    uint32_t total = end - start;

    // elements to reach 16B alignment from 'base'
    const uintptr_t addr           = reinterpret_cast<uintptr_t>(base);
    const uint32_t  bytes_to_align = (16u - (addr & 15u)) & 15u;
    const uint32_t  head           = min(total, bytes_to_align / static_cast<uint32_t>(sizeof(T_OUT)));

    // handle unaligned indices [0, head)
    for(uint32_t i = tid; i < head; i += stride) { base[i] = static_cast<T_OUT>(0.0f); }

    // Advance to aligned boundary
    base += head;
    total -= head;
    if(!total) return;

    // ----- 16B vector stores (8 halves or 4 floats per chunk) -----
    //static_assert(16 % sizeof(T_OUT) == 0, "T_OUT size must divide 16");
    constexpr uint32_t ELEMS_PER_VEC = 16u / sizeof(T_OUT);

    uint4*   ptrVec   = reinterpret_cast<uint4*>(base);
    uint32_t vecCount = total / ELEMS_PER_VEC;  // number of 16-byte stores

    // Each global lane writes vector chunks: j = tid; j < vecCount; j += stride
    for(uint32_t j = tid; j < vecCount; j += stride)
    {
        //ptrVec[j] = {0u, 0u, 0u, 0u}; // 128-bit zero
        // slightly lower L1 pollution using .cs (evict-first) policy
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(ptrVec + j), "r"(0), "r"(0), "r"(0), "r"(0));
    }

    // ----- Tail: scalar store of leftovers (remaining elements < ELEMS_PER_VEC) -----
    T_OUT*   tailPtr = reinterpret_cast<T_OUT*>(ptrVec + vecCount);
    uint32_t tail    = total % ELEMS_PER_VEC;

    // handle remaining unaligned indices [0, tail)
    for(uint32_t i = tid; i < tail; i += stride) { tailPtr[i] = static_cast<T_OUT>(0.0f); }
}

// Loop-invariant compute_llr_index parameters stored in shared memory
// to reduce register pressure inside the main loop.
struct SmemCliParams {
    uint32_t codeBlockQAMStartIndex;
    uint32_t adjustedCodeBlockQAMStartIndex;
    uint32_t Nl;
    uint32_t nBBULayers;
    uint32_t uciOnPuschFlag;  // stored as uint32_t for alignment
};

template <typename T_IN, typename T_OUT, int Qm, bool ndi>
__device__ __forceinline__ void deRateMatchingKernelInner(puschRxRateMatchDescr_t* pRmDesc)
{
    using T_OUT_PAIR = typename std::conditional<std::is_same<T_OUT, __half>::value, __half2, float2>::type;

    constexpr int WARP_SIZE = 32;
    const T_OUT LLR_CLAMP_MAX = static_cast<T_OUT>(LLR_MAX_ABS_VALUE);
    const T_OUT LLR_CLAMP_MIN = static_cast<T_OUT>(-LLR_MAX_ABS_VALUE);
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    const uint32_t nFracCbs = gridDim.x;
    const uint32_t fracCbIdx = blockIdx.x;
    const uint32_t cbIdx = blockIdx.y;
    const uint32_t tbIdx = blockIdx.z;
    const uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Output tensor
    // @todo: rmDesc.out which holds an array of pointers to HARQ buffers (in GPU memory) lives in host pinned memory,
    // check performance impact of accessing this memory
    T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // code block index
    uint32_t r = cbIdx + tbPrms.firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrms.Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    // Adjust for codeblock offset
    out += cbStartOffset;


    // Cache uciOnPuschFlag to avoid repeated loads in hot loop
    const uint8_t uciOnPuschFlag = tbPrms.uciOnPuschFlag;
    // Enable/Disable descrambling (combined with uciOnPuschFlag check)
    const uint8_t descramblingOn = rmDesc.descramblingOn && !uciOnPuschFlag;
    // Input LLR tensor
    const T_IN* llr_vec_in = static_cast<const T_IN*>(rmDesc.llr_vec_in[tbIdx]);

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrms.Ncb;
    // number of code blocks in transport block
    uint32_t C = tbPrms.num_CBs;

    // lifting factor
    uint32_t Zc = tbPrms.Zc;
    // Number of UE layers (Number of layers occupied by transport block tbIdx)
    uint32_t Nl = tbPrms.Nl;

    // Total number of layers from all UEs (Number of BBU antenna layers, KERNEL LEVEL parameter)
    uint32_t nBBULayers = tbPrms.nBBULayers;

    // layer mapping (dynamic shared memory)
    extern __shared__ uint32_t layer_map_array[];

    // LLR transpose buffer (dynamic shared memory, after layer_map_array)
    T_OUT* smem_llr = reinterpret_cast<T_OUT*>(layer_map_array + MAX_N_LAYERS_PUSCH);

    // Shared-memory copies of loop-invariant compute_llr_index parameters
    // (populated once before the main loop to free 5 registers from the hot path).
    SmemCliParams* smem_cli = reinterpret_cast<SmemCliParams*>(smem_llr + 2 * DERM_BLK_DIM);

    /************/

    if(r < C)
    {   // Only executes code if thread is allocated a valid  codeblock (some threads will be idle)

        if (nBBULayers > Nl)
        {
            for (int i = threadIdx.x; i < Nl; i += blockDim.x)
            {
                layer_map_array[i] = tbPrms.layer_map_array[i];
            }
            __syncthreads();
        }

        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // index at which the first LLR of code block r starts within transport block tbIdx
        uint32_t codeBlockQAMStartIndex;
        // Size (number of LLRs) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRs to be read for current transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = uciOnPuschFlag ? tbPrms.G / TBLLRsPerNBBULayers : tbPrms.encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrms.encodedSize);

        if(r <= rr)
        {
            E                      = El;
            codeBlockQAMStartIndex = r * El;
        }
        else
        {
            E                      = Eh;
            codeBlockQAMStartIndex = (rr + 1) * El + (r - rr - 1) * Eh;
        }

        uint32_t EoverQm = E / Qm;

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        uint32_t k0 = tbPrms.k0;

        /************/

        // First code block LLR index for current CTA within transport block, used for descrambling sequence generation
        const uint32_t ctaLLROffset = fracCbIdx * blockDim.x * NUM_LLRS_PROCESSED_PER_THRD;
        const uint32_t cbLLRStartIndex = codeBlockQAMStartIndex + ctaLLROffset;

        //====================================================================================================================================================
        // each thread block will process (blockDim.x * NUM_LLRS_PROCESSED_PER_THRD) LLRs, i.e. we'll have a for loop that each thread in the CTA will iterate
        // maximum NUM_LLRS_PROCESSED_PER_THRD times.
        // at iteration 0, each warp reads "mySeq" from thread 0 of that warp. "mySeq" has 32 bits (WORD_SIZE), each bit associates with one thread in the warp (WARP_SIZE).
        // hence this implementation relies on the fact that WORD_SIZE == WARP_SIZE
        // at iteration i, each warp reads the corresponding word from the golden sequence, and that is word index NUM_WARPS_PER_TB * i + WARP_IDX  where
        // NUM_WARPS_PER_TB = blockDim.x / WARP_SIZE, and WARP_IDX = threadIdx.x / WARP_SIZE
        // since in this for loop, we retrieve golden sequence word of iteration i in each warp from thread i in that warp, we'd use the following indexing logic
        // (and this explains why NUM_LLRS_PROCESSED_PER_THRD as the maximum number of iterations can't be bigger than warp size)
        uint32_t mySeq = 0;
        if (descramblingOn)
        {
            const uint32_t NUM_WARPS_PER_TB   = blockDim.x / WARP_SIZE;
            const uint32_t WARP_IDX           = threadIdx.x / WARP_SIZE;
            const uint32_t THREAD_IDX_IN_WARP = threadIdx.x % WARP_SIZE;
            const uint32_t index              = THREAD_IDX_IN_WARP * NUM_WARPS_PER_TB + WARP_IDX;

            // each thread in a warp computes a word of the descrambling sequence that will be used at step "warpLane"
            mySeq = gold32n(tbPrms.cinit, cbLLRStartIndex + index * WORD_SIZE);
        }

        //====================================================================================================================================================

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        uint32_t adjustedCodeBlockQAMStartIndex = (codeBlockQAMStartIndex / Qm) * QAM_STRIDE;
        // Number of filler bits
        uint32_t F              = tbPrms.F;
        uint32_t nPuncturedBits = 2 * Zc;

        // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
        out += nPuncturedBits;

        // Number of systematic bits
        uint32_t K = tbPrms.K;
        // Number of systematic bits in output code block excluding punctured bits
        //uint32_t K_hat = K - nPuncturedBits;
        // Number of payload bits in output code block
        uint32_t Kd = K - nPuncturedBits - F;

        // nZpBitsPerCb is the total number of LLRs i.e. (2 * Zc) + E + F rounded up to a multiple of Zc (rounding needed by LDPC decoder)
        // (2 * Zc) - punctured LLRs
        // E        - rate matched LLRs
        // F        - Filler LLRs
        // uint32_t nZpBitsPerCb = tbPrmsArray[tbIdx].nZpBitsPerCb;

        //number of LLRs belonging to other transport blocks to be skipped before getting to LLRS belonging to current transport block again
        /////////uint32_t cbStep = nBBULayers / Nl;
        // Deinterleave and fill output vector except filler bits

        // detect possibility of more than one thread accessing the same outIdx
        int potentialRaceIfPositive = (E + 2 * F + k0) - Ncb; // if potentialRaceIfPositive > 0, more than one thread can write on the same outIdx
        // Note that before running rate matching, de_rate_matching_reset_buffer should run to reset first potentialRaceIfPositive LLRs in HARQ buffer
        // Resetting those LLRs in this function can potentially lead to intra-CTA race. i.e. while some CTA is writing on the HARQ buffer, another CTA may reset the same element

        // Make sure CTA is not reading beyond input code block size E
        int maxIndex            = round_up_to_next(E, WORD_SIZE); // round up to be multiple of 32
        int maxIndexThisThrdBlk = (fracCbIdx + 1) * blockDim.x * NUM_LLRS_PROCESSED_PER_THRD;
        maxIndex                = (maxIndexThisThrdBlk < maxIndex) ? maxIndexThisThrdBlk : maxIndex;

        // Process two LLRs per iteration
        constexpr int LLR_PER_ITER = 2;

        // --------------------------------------------------------------------------
        // Initial indices & bookkeeping
        uint32_t warpLane         = 0;            // 0,2,4,…,30 (increments by 2)

        // bitOffset used in descrambling is loop-invariant: DERM_BLK_DIM % 32 == 0 and stride (blockDim.x * LLR_PER_ITER) % 32 == 0,
        // so we compute it only once (threadIdx.x & 31), and note both LLR0 and LLR1 share the same bitOffset since blockDim.x % 32 == 0.
        const uint32_t bitOffset  = threadIdx.x & 31;

        // current pair (tid0 and tid1)
        // jIdx/kIdx are scoped to reduce register pressure; recomputed inside loop when needed.
        int   tid0 = ctaLLROffset + threadIdx.x;
        T_OUT llr0 = static_cast<T_OUT>(0.0f);
        {
            uint32_t jIdx0 = static_cast<uint32_t>(tid0) / Qm;
            uint32_t kIdx0 = static_cast<uint32_t>(tid0) - jIdx0 * Qm;
            if (jIdx0 < EoverQm) {
                uint32_t idx0 = compute_llr_index(tid0, jIdx0, kIdx0,
                                                  codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                                  Nl, nBBULayers, layer_map_array, uciOnPuschFlag);
                llr0 = llr_vec_in[idx0];
            }
        }

        T_OUT llr1 = static_cast<T_OUT>(0.0f);
        {
            int tid1_init = tid0 + static_cast<int>(blockDim.x);
            if (tid1_init < maxIndex) {
                uint32_t jIdx1 = static_cast<uint32_t>(tid1_init) / Qm;
                uint32_t kIdx1 = static_cast<uint32_t>(tid1_init) - jIdx1 * Qm;
                if (jIdx1 < EoverQm) {
                    uint32_t idx1 = compute_llr_index(tid1_init, jIdx1, kIdx1,
                                                      codeBlockQAMStartIndex, adjustedCodeBlockQAMStartIndex,
                                                      Nl, nBBULayers, layer_map_array, uciOnPuschFlag);
                    llr1 = llr_vec_in[idx1];
                }
            }
        }

        //------------------------------------------------------------------------------
        // Store compute_llr_index params in shared memory to reduce register pressure
        if (threadIdx.x == 0) {
            smem_cli->codeBlockQAMStartIndex         = codeBlockQAMStartIndex;
            smem_cli->adjustedCodeBlockQAMStartIndex = adjustedCodeBlockQAMStartIndex;
            smem_cli->Nl                             = Nl;
            smem_cli->nBBULayers                     = nBBULayers;
            smem_cli->uciOnPuschFlag                 = static_cast<uint32_t>(uciOnPuschFlag);
        }
        __syncthreads();

        // Main loop
        //==================================================================================================
        // The following loop processes 2 LLRs per iteration (tid0 and tid0+blockDim.x).
        // For Qm>1, full tiles use a shared-memory transpose to coalesce global stores to some extent;
        // the tail (partial tile) falls back to scalar writes not using shared memory (no CTA barriers, safe when
        // not all threads participate). Qm==1 always uses scalar writes, as it is already coalesced without
        // using shared memory transpose.
        const int tileSpan = static_cast<int>(blockDim.x) * LLR_PER_ITER;

        for (/* tid0 already set */;
             tid0 < maxIndex;
             warpLane += LLR_PER_ITER)
        {
            // ------------------- Prefetch next pair into registers -------------------
            int n_tid0 = tid0 + tileSpan;
            T_OUT n_llr0 = static_cast<T_OUT>(0.0f);
            if (n_tid0 < maxIndex) {
                uint32_t n_jIdx0 = static_cast<uint32_t>(n_tid0) / Qm;
                uint32_t n_kIdx0 = static_cast<uint32_t>(n_tid0) - n_jIdx0 * Qm;
                if (n_jIdx0 < EoverQm) {
                    uint32_t n_idx0 = compute_llr_index(n_tid0, n_jIdx0, n_kIdx0,
                                                        smem_cli->codeBlockQAMStartIndex, smem_cli->adjustedCodeBlockQAMStartIndex,
                                                        smem_cli->Nl, smem_cli->nBBULayers, layer_map_array,
                                                        static_cast<uint8_t>(smem_cli->uciOnPuschFlag));
                    n_llr0 = llr_vec_in[n_idx0];
                }
            }

            int n_tid1 = n_tid0 + static_cast<int>(blockDim.x);
            T_OUT n_llr1 = static_cast<T_OUT>(0.0f);
            if (n_tid1 < maxIndex) {
                uint32_t n_jIdx1 = static_cast<uint32_t>(n_tid1) / Qm;
                uint32_t n_kIdx1 = static_cast<uint32_t>(n_tid1) - n_jIdx1 * Qm;
                if (n_jIdx1 < EoverQm) {
                    uint32_t n_idx1 = compute_llr_index(n_tid1, n_jIdx1, n_kIdx1,
                                                        smem_cli->codeBlockQAMStartIndex, smem_cli->adjustedCodeBlockQAMStartIndex,
                                                        smem_cli->Nl, smem_cli->nBBULayers, layer_map_array,
                                                        static_cast<uint8_t>(smem_cli->uciOnPuschFlag));
                    n_llr1 = llr_vec_in[n_idx1];
                }
            }

            // ------------------- Pack, descramble, clamp ----------------------------
            T_OUT_PAIR llr_pair;
            if constexpr (std::is_same<T_OUT, __half>::value) {
                llr_pair = __halves2half2(llr0, llr1);
            } else {
                llr_pair = {llr0, llr1};
            }

            if (descramblingOn)
            {
                uint32_t seqWord0 = __shfl_sync(0xFFFFFFFF, mySeq, warpLane);
                uint32_t seqWord1 = __shfl_sync(0xFFFFFFFF, mySeq, warpLane + 1);
                llr_pair = rate_match_xor_sign_pair(seqWord0, seqWord1, bitOffset, bitOffset, llr_pair);
            }

            if constexpr (std::is_same<T_OUT, __half>::value)
            {
                const __half2 HMIN2 = __halves2half2(LLR_CLAMP_MIN, LLR_CLAMP_MIN);
                const __half2 HMAX2 = __halves2half2(LLR_CLAMP_MAX, LLR_CLAMP_MAX);
                llr_pair = __hmax2(__hmin2(llr_pair, HMAX2), HMIN2);
            } else
            {
                llr_pair.x = max(min(llr_pair.x, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
                llr_pair.y = max(min(llr_pair.y, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
            }

            // ------------------- Write LLRs to global memory ------------------------
            // tileBase is block-uniform (tid0 - threadIdx.x is the same for all threads).
            // fullTile checks that all 2·blockDim LLR slots in this tile are within maxIndex,
            // guaranteeing every thread in the block is active → __syncthreads() is safe.
            // For Qm==1, (Qm > 1) is compile-time false and the shared memory transpose path is eliminated.
            const int  tileBase = tid0 - static_cast<int>(threadIdx.x);
            const bool fullTile = (tileBase + tileSpan) <= maxIndex;

            if (fullTile && (Qm > 1))
            {
                // Shared-memory transpose: remap thread→LLR so consecutive
                // threadIdx.x values target the same kIdx group, producing
                // contiguous outIdx for coalesced global stores.
                smem_llr[threadIdx.x]                = llr_pair.x;
                smem_llr[threadIdx.x + DERM_BLK_DIM] = llr_pair.y;

                __syncthreads(); // fill complete — all threads can now read any slot

                {
                    constexpr int TPG = DERM_BLK_DIM / Qm; // threads per kIdx group
                    static_assert(DERM_BLK_DIM % Qm == 0, "DERM_BLK_DIM must be divisible by Qm");

                    // Original thread layout (kIdx = threadIdx.x % Qm):
                    //   Within a 32-lane warp and Qm=8, lanes {0,8,16,24} share kIdx=0
                    //   but are 8 apart, hence 8 scattered mini-bursts of 4 elements each.
                    // After transpose (kIdx = threadIdx.x / TPG, TPG = DERM_BLK_DIM/Qm):
                    //   threads 0..TPG-1 all have kIdx=0 with consecutive jIdx
                    //   contiguous outIdx, hence single coalesced burst per kIdx group.
                    const int remap_kIdx   = threadIdx.x / TPG;
                    const int remap_jIdx   = threadIdx.x % TPG;
                    const int remap_llrIdx = remap_jIdx * Qm + remap_kIdx;

                    const uint32_t jIdxBase = static_cast<uint32_t>(tileBase) / Qm;

                    // First half: smem[0..DERM_BLK_DIM-1]
                    processOneLLR<T_OUT, ndi>(jIdxBase + static_cast<uint32_t>(remap_jIdx),
                                              static_cast<uint32_t>(remap_kIdx),
                                              smem_llr[remap_llrIdx], EoverQm,
                                              Kd, F, k0, Ncb, potentialRaceIfPositive,
                                              LLR_CLAMP_MIN, LLR_CLAMP_MAX, out);

                    // Second half: smem[DERM_BLK_DIM..2*DERM_BLK_DIM-1]
                    constexpr uint32_t HALF_J_OFFSET = DERM_BLK_DIM / Qm;
                    processOneLLR<T_OUT, ndi>(jIdxBase + HALF_J_OFFSET + static_cast<uint32_t>(remap_jIdx),
                                              static_cast<uint32_t>(remap_kIdx),
                                              smem_llr[DERM_BLK_DIM + remap_llrIdx], EoverQm,
                                              Kd, F, k0, Ncb, potentialRaceIfPositive,
                                              LLR_CLAMP_MIN, LLR_CLAMP_MAX, out);
                }

                __syncthreads(); // flush complete — safe to overwrite smem on next iter
            }
            else
            {
                // Scalar path: Qm==1 (always), or Qm>1 tail tile (partial — no barriers).
                const uint32_t jIdx0 = static_cast<uint32_t>(tid0) / Qm;
                const uint32_t kIdx0 = static_cast<uint32_t>(tid0) - jIdx0 * Qm;
                processOneLLR<T_OUT, ndi>(jIdx0, kIdx0, llr_pair.x, EoverQm,
                                          Kd, F, k0, Ncb, potentialRaceIfPositive,
                                          LLR_CLAMP_MIN, LLR_CLAMP_MAX, out);
                const int tid1 = tid0 + static_cast<int>(blockDim.x);
                if (tid1 < maxIndex) {
                    const uint32_t jIdx1 = static_cast<uint32_t>(tid1) / Qm;
                    const uint32_t kIdx1 = static_cast<uint32_t>(tid1) - jIdx1 * Qm;
                    processOneLLR<T_OUT, ndi>(jIdx1, kIdx1, llr_pair.y, EoverQm,
                                              Kd, F, k0, Ncb, potentialRaceIfPositive,
                                              LLR_CLAMP_MIN, LLR_CLAMP_MAX, out);
                }
            }

            // ------------------- Rotate prefetched state for next iter ---------------
            tid0 = n_tid0; llr0 = n_llr0; llr1 = n_llr1;
        }

        if constexpr (ndi)
        {
            // Use all thread blocks associated with this CB
            uint32_t stride    = nFracCbs * blockDim.x;
            uint32_t globalTid = (fracCbIdx * blockDim.x) + threadIdx.x;

            // Output buffer is of length Ncb_padded

            // 1. Circular buffer initialization (Ncb long circular buffer section of output buffer)
            // NOTE: EITHER 1a, 2a, 2b below needed OR at setup set RM output to zero
            // 1a. Simply write zeros into rest of Ncb long circular buffer (including filler gap)
            if (E + F < Ncb)
            {
                const uint32_t len   = Ncb - (E + F);
                const uint32_t start = (E + F + k0) % Ncb;


                // Helper to exclude the filler gap [Kd, Kd+F) from a single non-wrapping range [a,b)
                auto zeroSpanExcludingFillerGap = [&](uint32_t a, uint32_t b) {
                    // No overlap
                    if (b <= Kd || a >= Kd + F) {
                        zeroRangeVec(out, a, b, globalTid, stride);
                        return;
                    }
                    // Overlap on the left side: zero [a, Kd)
                    if (a < Kd) zeroRangeVec(out, a, Kd, globalTid, stride);
                    // Overlap on the right side: zero [Kd+F, b)
                    if (b > Kd + F) zeroRangeVec(out, Kd + F, b, globalTid, stride);
                    // If [a,b) entirely inside [Kd, Kd+F) : no action needed
                };

                // Our "rest" interval is a physical circular interval of length `len` starting at `start`
                // Convert it to up to two non-wrapping pieces, then subtract the filler once on each piece.
                if (start + len <= Ncb) {
                    // No wrap: [start, start+len)
                    zeroSpanExcludingFillerGap(start, start + len);
                } else {
                    // Wraps: [start, Ncb) ∪ [0, (start+len - Ncb))
                    zeroSpanExcludingFillerGap(start, Ncb);
                    zeroSpanExcludingFillerGap(0, (start + len) - Ncb);
                }
            }

            // 1b. Write filler bits to circular buffer
            for(uint32_t n = Kd + globalTid; n < Kd + F; n += stride)
            {
                // Note: Location of Filler bits is fixed to tail end of systematic bit section of
                // circular buffer and is independent of k0
                uint32_t circBufIdx              = n;
                out[circBufIdx] = LLR_CLAMP_MAX;
            }

            // 2. Initialization of the rest of output buffer: section of length (Ncb_padded - Ncb)
            // 2a. Write zeros into punctured bits (first 2*Zc bits of Ncb_padded output buffer), also out address includes nPuncturedBits offset
            zeroRangeVec(out - nPuncturedBits, 0, nPuncturedBits, globalTid, stride);

            //FixMe the following block results in mismatches for PUSCH derate match standalone unit test, additionally the for loop needs to include tid; disable for now
            // 2b. Write zeros into byte padding section of Ncb_padded output buffer
//            for(uint32_t n = Ncb; n < Ncb_padded; n += stride)
//            {
//                out[n] = 0;
//            }
        }
    }
}

template <typename T_IN, typename T_OUT>
__global__ void __launch_bounds__(DERM_BLK_DIM, 16) de_rate_matching_global2(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    uint32_t tbIdx = blockIdx.z;
    uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // QAM modulation index
    uint32_t Qm = tbPrms.Qm;

#ifndef NDEBUG
    const bool isSupportedQm = (Qm == 1u) || (Qm == 2u) || (Qm == 4u) || (Qm == 6u) || (Qm == 8u);
    if (!isSupportedQm)
    {
        // Fail fast for unexpected modulation orders; avoid silent no-op dispatch.
        if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))
        {
            assert(false && "Unsupported Qm in de_rate_matching_global2");
        }
        return;
    }
#endif

    // new data indicator
    uint32_t ndi = tbPrms.ndi;

    if (ndi != 0)   // ndi true: no LLR combining (new transmission)
    {
        if(Qm == 1)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 1, true>(pRmDesc);
        }
        else if(Qm == 2)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 2, true>(pRmDesc);
        }
        else if(Qm == 4)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 4, true>(pRmDesc);
        }
        else if(Qm == 6)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 6, true>(pRmDesc);
        }
        else if(Qm == 8)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 8, true>(pRmDesc);
        }
    }
    else // ndi false: LLR combining (retransmission)
    {
        if(Qm == 1)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 1, false>(pRmDesc);
        }
        else if(Qm == 2)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 2, false>(pRmDesc);
        }
        else if(Qm == 4)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 4, false>(pRmDesc);
        }
        else if(Qm == 6)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 6, false>(pRmDesc);
        }
        else if(Qm == 8)
        {
            deRateMatchingKernelInner<T_IN, T_OUT, 8, false>(pRmDesc);
        }
    }

}

// this kernel partially resets HARQ buffer
template <typename T_OUT>
__global__ void __launch_bounds__(DERM_BLK_DIM, 10) de_rate_matching_reset_buffer(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    const uint32_t nFracCbs = gridDim.x;
    const uint32_t fracCbIdx = blockIdx.x;
    const uint32_t cbIdx = blockIdx.y;
    const uint32_t tbIdx = blockIdx.z;
    const uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];


    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];

    // new data indicator
    uint32_t ndi = tbPrms.ndi;
    if (!ndi) return; //early exit, as there is no need to reset any part of the HARQ buffer

    // QAM modulation index
    const uint32_t Qm = tbPrms.Qm;
    // code block index
    uint32_t r = cbIdx + tbPrms.firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrms.Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrms.Ncb;
    // number of code blocks in transport block
    uint32_t C = tbPrms.num_CBs;

    // lifting factor
    uint32_t Zc = tbPrms.Zc;
    // Number of UE layers (Number of layers occupied by transport block tbIdx)
    uint32_t Nl = tbPrms.Nl;

    /************/

    if(r < C)
    {
        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // Size (number of LLRs) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRs to be read for current transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = tbPrms.uciOnPuschFlag ? tbPrms.G / TBLLRsPerNBBULayers : tbPrms.encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrms.encodedSize);

        E = (r <= rr) ? El : Eh;

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        uint32_t k0 = tbPrms.k0;

        //====================================================================================================================================================

        // Number of filler bits
        uint32_t F              = tbPrms.F;

        // detect possibility of more than one thread accessing the same outIdx
        int potentialRaceIfPositive = (E + 2 * F + k0) - Ncb; // if potentialRaceIfPositive > 0, more than one thread can write on the same outIdx
        if (potentialRaceIfPositive <= 0) return;

        // Output tensor
        T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);
        // Adjust for codeblock offset
        out += cbStartOffset;
        // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
        uint32_t nPuncturedBits = 2 * Zc;
        out += nPuncturedBits;

        if (potentialRaceIfPositive > 0 && ndi)
        {
            int maxOutIdx = Ncb;
            potentialRaceIfPositive = min(potentialRaceIfPositive, maxOutIdx);
            // Use all thread blocks associated with this CB
            uint32_t stride    = nFracCbs * blockDim.x;
            uint32_t globalTid = (fracCbIdx * blockDim.x) + threadIdx.x;
            zeroRangeVec(out, 0, potentialRaceIfPositive, globalTid, stride);
        }
    }
}

// this kernel partially clamps HARQ buffer //ToDo: is clamping absolutely needed?
template <typename T_OUT>
__global__ void __launch_bounds__(DERM_BLK_DIM, 10) de_rate_matching_clamp_buffer(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    const uint32_t nFracCbs = gridDim.x;
    const uint32_t fracCbIdx = blockIdx.x;
    const uint32_t cbIdx = blockIdx.y;
    const uint32_t tbIdx = blockIdx.z;
    const uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Array of transport block parameters structs
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    // QAM modulation index
    const uint32_t Qm = tbPrms.Qm;
    // code block index
    uint32_t r = cbIdx + tbPrms.firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrms.Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrms.Ncb;
    // number of code blocks in transport block
    uint32_t C = tbPrms.num_CBs;

    // lifting factor
    uint32_t Zc = tbPrms.Zc;
    // Number of UE layers (Number of layers occupied by transport block tbIdx)
    uint32_t Nl = tbPrms.Nl;

    /************/

    if(r < C)
    {
        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // Size (number of LLRs) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRs to be read for current transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = tbPrms.uciOnPuschFlag ? tbPrms.G / TBLLRsPerNBBULayers : tbPrms.encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrms.encodedSize);

        E = (r <= rr) ? El : Eh;

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        uint32_t k0 = tbPrms.k0;

        //====================================================================================================================================================

        // Number of filler bits
        uint32_t F              = tbPrms.F;

        // detect possibility of more than one thread accessing the same outIdx
        int potentialRaceIfPositive = (E + 2 * F + k0) - Ncb; // if potentialRaceIfPositive > 0, more than one thread can write on the same outIdx
        if (potentialRaceIfPositive <= 0) return;

        // Output tensor
        T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);
        // Adjust for codeblock offset
        out += cbStartOffset;
        // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
        uint32_t nPuncturedBits = 2 * Zc;
        out += nPuncturedBits;

        const T_OUT LLR_CLAMP_MAX = static_cast<T_OUT>(LLR_MAX_ABS_VALUE);
        const T_OUT LLR_CLAMP_MIN = static_cast<T_OUT>(-LLR_MAX_ABS_VALUE);

        // clamp LLRs that fall within [0 ... potentialRaceIfPositive) index
        if (potentialRaceIfPositive > 0)
        {
            int maxOutIdx = Ncb;
            potentialRaceIfPositive = min(potentialRaceIfPositive, maxOutIdx);
            // Use all thread blocks associated with this CB
            int stride    = nFracCbs * blockDim.x;
            int globalTid = (fracCbIdx * blockDim.x) + threadIdx.x;
            // Strided clamp over [0, potentialRaceIfPositive) //ToDo vectorize
            for (int n = globalTid; n < potentialRaceIfPositive; n += stride)
            {
                T_OUT llr = out[n];
                if constexpr (std::is_same<T_OUT, __half>::value) {
                    llr = __hmax(__hmin(llr, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
                } else {
                    llr = max(min(llr, LLR_CLAMP_MAX), LLR_CLAMP_MIN);
                }
                out[n] = llr;
            }
        }
    }
}

void puschRxRateMatch::setup(uint16_t                          nSchUes,                     // number of users with sch data
                             uint16_t*                         pSchUserIdxsCpu,             // indices of users with SCH data
                             const PerTbParams*                pTbPrmsCpu,                  // starting address of transport block parameters (CPU)
                             const PerTbParams*                pTbPrmsGpu,                  // starting address of transport block parameters (GPU)
                             cuphyTensorPrm_t*                 pTPrmRmIn,                   // starting address of input LLR tensor parameters
                             cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                             void**                            ppRmOut,                     // array of rm outputs (GPU)
                             void*                             pCpuDesc,                    // pointer to descriptor in cpu
                             void*                             pGpuDesc,                    // pointer to descriptor in gpu
                             uint8_t                           enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
                             cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
                             cudaStream_t                      strm)                        // stream to perform copy
{
    // setup CPU descriptor
    puschRxRateMatchDescr_t& desc    = *(static_cast<puschRxRateMatchDescr_t*>(pCpuDesc));
    uint16_t                 nUciUes = 0;

    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx      = pSchUserIdxsCpu[i];
        desc.schUserIdxs[i] = ueIdx;
#ifndef NDEBUG
        {
            const uint32_t Qm = pTbPrmsCpu[ueIdx].Qm;
            assert(((Qm == 1u) || (Qm == 2u) || (Qm == 4u) || (Qm == 6u) || (Qm == 8u)) &&
                   "Unsupported Qm in puschRxRateMatch::setup");
        }
#endif
        if(pTbPrmsCpu[ueIdx].uciOnPuschFlag)
        {
            desc.llr_vec_in[i] = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
            nUciUes++;
        }
        else
        {
            uint32_t ueGrpIdx  = pTbPrmsCpu[ueIdx].userGroupIndex;
            if(pTbPrmsCpu[ueIdx].nDmrsCdmGrpsNoData==1)
            {
                desc.llr_vec_in[i] = pTPrmCdm1RmIn[ueGrpIdx].pAddr;
            }
            else
            {
                desc.llr_vec_in[i] = pTPrmRmIn[ueGrpIdx].pAddr;
            }
        }
    }
    desc.out            = ppRmOut;
    desc.tbPrmsArray    = pTbPrmsGpu;
    desc.descramblingOn = m_descramblingOn;

    // optional CPU->GPU copy
    if(enableCpuToGpuDescrAsyncCpy)
    {
        // added Unchecked return value
        CUDA_CHECK(cudaMemcpyAsync(pGpuDesc, pCpuDesc, sizeof(puschRxRateMatchDescr_t), cudaMemcpyHostToDevice, strm));
    }

    // Setup Launch Geometry
    uint32_t EMax = 0; // max number of encoded bits per CB
    uint32_t CMax = 0; // max number of CBs per TB
    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx = pSchUserIdxsCpu[i];
        CMax           = CMax < pTbPrmsCpu[ueIdx].num_CBs ? pTbPrmsCpu[ueIdx].num_CBs : CMax;
        uint32_t Eh    = pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * ceilf(float(pTbPrmsCpu[ueIdx].encodedSize) / float(pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * pTbPrmsCpu[ueIdx].num_CBs));
        EMax           = EMax < Eh ? Eh : EMax;
    }

    // using larger block size could result in load imbalance in some cases and lower occupancy
    dim3 gridDim(div_round_up(EMax, DERM_BLK_DIM * NUM_LLRS_PROCESSED_PER_THRD), CMax, nSchUes);
    dim3 blockDim(DERM_BLK_DIM, 1, 1);
    // printf("gridDim(%d %d %d) blockDim(%d %d %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    pLaunchCfg->desc                                  = pGpuDesc;
    pLaunchCfg->kernelArgs[0]                         = &(pLaunchCfg->desc);
    
    // Configure main de_rate_matching_global2 kernel
    pLaunchCfg->kernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->kernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->kernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->kernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->kernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->kernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->kernelNodeParamsDriver.func           = m_kernelFunc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    // Dynamic shared memory: layer_map_array + smem_llr + SmemCliParams
    constexpr uint32_t sharedMemBytes = MAX_N_LAYERS_PUSCH * sizeof(uint32_t)
                                      + 2 * DERM_BLK_DIM * sizeof(float)   // smem_llr upper bound
                                      + sizeof(SmemCliParams);              // compute_llr_index params
    pLaunchCfg->kernelNodeParamsDriver.sharedMemBytes = sharedMemBytes;
    pLaunchCfg->kernelNodeParamsDriver.extra          = nullptr;
    
    // Configure reset buffer kernel (same grid/block dimensions)
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->resetKernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->resetKernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->resetKernelNodeParamsDriver.func           = m_resetBufferKernelFunc;
    pLaunchCfg->resetKernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    pLaunchCfg->resetKernelNodeParamsDriver.sharedMemBytes = 0;
    pLaunchCfg->resetKernelNodeParamsDriver.extra          = nullptr;

    // Configure clamp buffer kernel (same grid/block dimensions)
    pLaunchCfg->clampKernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->clampKernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->clampKernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->clampKernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->clampKernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->clampKernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->clampKernelNodeParamsDriver.func           = m_clampBufferKernelFunc;
    pLaunchCfg->clampKernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    pLaunchCfg->clampKernelNodeParamsDriver.sharedMemBytes = 0;
    pLaunchCfg->clampKernelNodeParamsDriver.extra          = nullptr;
}

void puschRxRateMatch::init(int rmFPconfig,     // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
                            int descramblingOn) // enable/disable descrambling
{
    // Save configurations
    m_descramblingOn = descramblingOn;
    m_rmFPconfig = rmFPconfig;

    // Select Main Kernel and Reset Buffer Kernel
    switch(rmFPconfig)
    {
    case 0:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_clampBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_clamp_buffer<float>)));}
        break;

    case 1:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<float>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_clampBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_clamp_buffer<float>)));}
        break;

    case 2:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, __half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<__half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_clampBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_clamp_buffer<__half>)));}
        break;

    case 3:
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, __half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_resetBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_reset_buffer<__half>)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_clampBufferKernelFunc, reinterpret_cast<void*>(de_rate_matching_clamp_buffer<__half>)));}
        break;
    default:
        break;
    }
}

void puschRxRateMatch::getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes)
{
    descrSizeBytes  = sizeof(puschRxRateMatchDescr_t);
    descrAlignBytes = alignof(puschRxRateMatchDescr_t);
}
