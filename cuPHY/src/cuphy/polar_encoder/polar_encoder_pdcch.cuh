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

/*
 * Device-side polar encode + rate-match building blocks for the PDCCH/PBCH path.
 * Shared between polar_encoder.cu (standalone encodeRateMatch* kernels) and
 * embed_pdcch_tf_signal.cu (fused PDCCH TX kernel), so both compile the same
 * single definition of the encode/rate-match logic.
 */

#if !defined(CUPHY_POLAR_ENCODER_PDCCH_CUH_INCLUDED_)
#define CUPHY_POLAR_ENCODER_PDCCH_CUH_INCLUDED_

#include "cuphy.h"
#include "polar_encoder.hpp"
#include "polar_encoder.cuh"

namespace polar_encoder
{

inline constexpr uint32_t N_BITS_PER_WORD       = 32;
inline constexpr uint32_t N_BITS_PER_BYTE       = 8;
inline constexpr uint32_t N_BYTES_PER_WORD      = 4;

// Table for encoded bit sub-block interleaving
// NB: header-defined __constant__: each including TU gets its own 32-byte copy.
static __device__ __constant__ uint8_t POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[] =
{
   0,  1,  2,  4,  3,  5,  6,  7,  8, 16,  9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31
};

// rateMatch scratch (after encode, reuses start of shared memory)
//      32 bytes because in puncturing/shortening branch (nTxBits < nCodedBits):
//      src_bit_idx = nCodedBits - E_mod_32, and pack32 reads up to byte nCodedBits + 31 - E_mod_32.
inline constexpr uint32_t N_RM_SCRATCH_BYTES = N_MAX_CODED_BITS + 32;

// Round n up to the nearest power of 2 at runtime (host + device).
__host__ __device__ __forceinline__ uint32_t roundUpToPow2U32(uint32_t n)
{
#ifdef __CUDA_ARCH__
    // __clz return type changed in CUDA 13.2+ from signed to unsigned for ARM64 systems, thus the explicit cast
    return (n > 1) ? (1U << (32 - (int)__clz(n - 1))) : 1;
#else
    uint32_t p = 1;
    while(p < n) { p <<= 1; }
    return p;
#endif
}

// 38.212 section 5.3.1 polar coded-bits sizing for PDCCH: N from (K, aggregation level).
// Single shared implementation for encodeRateMatchMultipleDCIsKernel and fusedPdcchTxKernel;
// host-callable so the sizing logic is unit-testable without a GPU.
__host__ __device__ __forceinline__ uint32_t pdcchPolarNumCodedBits(uint32_t nInfoBits, uint32_t aggrLevel)
{
    const uint32_t nTxBits               = 2 * 9 * 6 * aggrLevel;
    const uint32_t roundUpToPow2_nTxBits = 2 * 64 * aggrLevel; // Reminder: possible aggregation level values {1, 2, 4, 8, 16}

    uint32_t        nMin1CodedBits         = roundUpToPow2_nTxBits / 2;
    constexpr float INFO_TX_BITS_RATIO_THD = (9.0f / 16.0f);
    if((nTxBits > (9 * nMin1CodedBits) / 8) ||
       ((static_cast<float>(nInfoBits) / static_cast<float>(nTxBits)) >= INFO_TX_BITS_RATIO_THD))
    {
        nMin1CodedBits *= 2;
    }

    // Min number of coded bits possible given the number of info bits and min code rate
    const uint32_t nMin2CodedBits = roundUpToPow2U32(nInfoBits * MIN_CODE_RATE_INV);
    uint32_t       nCodedBits     = (nMin1CodedBits < nMin2CodedBits) ? nMin1CodedBits : nMin2CodedBits;
    if(nCodedBits < N_MIN_CODED_BITS) nCodedBits = N_MIN_CODED_BITS;
    if(nCodedBits > N_MAX_CODED_BITS) nCodedBits = N_MAX_CODED_BITS;
    return nCodedBits;
}

//--------------------------------------------------------------------------------------------------------
// Helpers for Bit extraction
__device__ __forceinline__ uint32_t getBitPosIdxInWord(uint32_t bitPos)
{
    return bitPos % N_BITS_PER_WORD;
}

__device__ __forceinline__ uint32_t getBitPosIdxInByte(uint32_t bitPos)
{
    return bitPos % N_BITS_PER_BYTE;
}

__device__ __forceinline__ uint32_t getBitPosWordIdx(uint32_t bitPos)
{
    return bitPos / N_BITS_PER_WORD;
}

__device__ __forceinline__ uint32_t getBitPosByteIdx(uint32_t bitPos)
{
    return bitPos / N_BITS_PER_BYTE;
}

__device__ __forceinline__ bool getBit(uint32_t const* pWords, uint32_t bitPos)
{
    return ((pWords[getBitPosWordIdx(bitPos)] >> getBitPosIdxInWord(bitPos)) & 0x1);
}

__device__ __forceinline__ bool getBit(uint8_t const* pBytes, uint32_t bitPos)
{
    return ((pBytes[getBitPosByteIdx(bitPos)] >> getBitPosIdxInByte(bitPos)) & 0x1);
}

__device__ __forceinline__ uint8_t getBitValue(uint32_t const* pWords, uint32_t bitPos)
{
    return static_cast<uint8_t>((pWords[getBitPosWordIdx(bitPos)] >> getBitPosIdxInWord(bitPos)) & 0x1);
}

// Pack bit0 of byte array p[offset + (0..7)] into one uint8:
//   bit0 of 1st byte to bit0 in the uint8, ..., bit0 of 8th byte to bit7 in the uint8.
__device__ __forceinline__ uint8_t pack8(uint8_t const* p, uint32_t offset)
{
    uint8_t v{};
    for(uint32_t b = 0; b < 8; b++)
    {
        v |= ((p[offset + b] & 1) << b);
    }
    return v;
}

// Pack bit0 of byte array p[offset + (0..31)] into one uint32:
//   bit0 of 1st byte to bit0 in the uint32, ..., bit0 of 32nd byte to bit31 in the uint32.
__device__ __forceinline__ uint32_t pack32(uint8_t const* p, uint32_t offset)
{
    uint32_t v = 0;
    for(uint32_t b = 0; b < 32; b++)
    {
        v |= (uint32_t)(p[offset + b] & 1) << b;
    }
    return v;
}

#ifdef DEBUG
template <typename T>
static __device__ void print_1d(const T* x, uint32_t len)
{
    printf("[");
    for(int i = 0; i < len; i++)
    {
        printf("%d,", x[i]);
    }
    printf("];\n");
}

// bit0 of x[i] is i-th bit.
static __device__ void pack_and_print_1d(const uint8_t* x, uint32_t len)
{
    printf("[");
    for(uint32_t i = 0; i < (len + 7) / 8; i++)
    {
        uint8_t tmp{};
        for(int j = 0; j < 8; j++)
        {
            tmp |= ((x[i * 8 + j] & 1) << j);
        }
        printf("%d,", tmp);
    }
    printf("];\n");
}
#endif // DEBUG

// Compute d = u*G, where
//      u is Nx1 byte array (bit0 of each byte is valid), in shared memory;
//      d is Nx1 bit-array packed into uint32_t array, in device memory.
//      G is NxN matrix.
// Ref: 38.212 5.3.1.2
static __device__ void u2d(thread_block const& thisThrdBlk, uint32_t N, uint8_t* const u, uint8_t* pCodedBits)
{
    uint32_t tid    = thisThrdBlk.thread_rank();
    uint32_t num_th = thisThrdBlk.size();
    for(uint32_t size = 2; size <= N; size <<= 1)
    {
        for(uint32_t b = tid; b < N / 2; b += num_th)
        {
            uint32_t stride  = size / 2;
            uint32_t idx     = 2 * b - (b & (stride - 1));
            uint32_t bitPos1 = idx + 0;
            uint32_t bitPos2 = idx + stride;
            u[bitPos1] ^= u[bitPos2];
        }
        __syncthreads();
    }
    // pack x into pCodedWords: 1 thread packs 8 bits.
    for(uint32_t byte_idx = tid; byte_idx < N / 8; byte_idx += num_th)
    {
        uint8_t b{};
        for(uint32_t i = 0; i < 8; i++)
        {
            b |= (u[byte_idx * 8 + i] << i);
        }
        pCodedBits[byte_idx] = b;
    }
    __syncthreads();
}

// Inputs:  pInfoBits, in device memory, holding "c", the msg word, K x 1 bit-array.
// Intermediate: pSmem, in shared memory, holding "u", N x 1 byte array.
//               Each of K bits in "c" is placed in bit 0 of K bytes according to cidx2uidx table; other bytes are 0s.
// Outputs: pCodedBits, holding "d", the codeword, N x 1 bit-array (device or shared memory).
// Note: Polar in {PDCCH, PBCH} use {I_IL=1, I_BIL=0}; PUSCH/PUCCH UCI use {I_IL=0, I_BIL=1}.
//       The LUT "cidx2uidx" is only for {PDCCH, PBCH}.
static __device__ void encode_pdcch_pbch_LUT(uint32_t K, uint32_t N, uint32_t E, const uint8_t* pInfoBits, const uint16_t* cidx2uidx, uint32_t* pSmem, uint8_t* pCodedBits)
{
#ifdef DEBUG
    constexpr int blk_target = 0;
#endif

    thread_block const& this_thblk = this_thread_block();
    uint32_t            tid        = this_thblk.thread_rank();
    uint32_t            numTh      = this_thblk.size();
    uint32_t*           u_i32      = pSmem;
    uint8_t*            u          = reinterpret_cast<uint8_t*>(pSmem);

#ifdef DEBUG
    __syncthreads();
    if(tid == 0 && blockIdx.x == blk_target && blockIdx.y == 0)
    {
        printf("K=%d; N=%d; E=%d;\n", K, N, E);
        printf("cidx2uidx = ");
        print_1d(cidx2uidx, K);
        printf("c = ");
        print_1d(pInfoBits, (K + 7) / 8);
    }
    __syncthreads();
#endif

    // Clear u before filling it because u is Nx1, c is Kx1, N > K.
    // Each thread clears one uint32, corresponding to 4 entries in Nx1. Note "u" is 32-bit aligned (see above).
    // In PBCH: N = 512; In PDCCH: N in {64, 128, 256, 512}.
    for(uint32_t i = tid; i < N / sizeof(uint32_t); i += numTh)
    {
        u_i32[i] = 0;
    }
    __syncthreads();

    // Fill u with c
    for(uint32_t cidx = tid; cidx < K; cidx += numTh)
    {
        uint16_t uidx         = cidx2uidx[cidx];
        uint32_t src_byte_idx = cidx / 8;
        uint32_t src_bit_idx  = cidx % 8;
        u[uidx]               = ((pInfoBits[src_byte_idx] >> src_bit_idx) & 0x1);
    }
    __syncthreads();

#ifdef DEBUG
    __syncthreads();
    if(tid == 0 && blockIdx.x == blk_target && blockIdx.y == 0)
    {
        printf("u = ");
        pack_and_print_1d(u, N);
    }
    __syncthreads();
#endif

    u2d(this_thblk, N, u, pCodedBits);

#ifdef DEBUG
    __syncthreads();
    if(tid == 0 && blockIdx.x == blk_target && blockIdx.y == 0)
    {
        printf("d = ");
        print_1d(pCodedBits, (N + 7) / 8);
    }
    __syncthreads();
#endif
}

//--------------------------------------------------------------------------------------------------------
// Polar rate-matching for PBCH and PDCCH.
// PBCH (N_max=512) and PDCCH (N_max<=512) have the same Rate matching setup:
//      C=1 (I_seg=0), I_IL=1, I_BIL=0.
//      E_PBCH = 864; E_PDCCH = {108, 216, 432, 864, 1728} for AL={1,2,4,8,16}, respectively.
// Note: pTxBits needs to be word alinged and padded to a multiple of word length
//      E%32 = [12, 24, 8, 16, 0]; E/32 = [3, 7, 2, 5, 0]
//
// Input:        "pInCodedBits", N coded bits (device or shared memory);
// Intermediate: "pSmem", N + 32 bytes. First N bytes to hold N coded bits at bit 0; 32 extra bytes is pack32(); in shared memory;
// Output:       "pTxBits", E ratematched bits (device or shared memory).
static __device__ void rateMatch(uint32_t nInfoBits, uint32_t nCodedBits, uint32_t nTxBits, uint8_t const* pInCodedBits, uint32_t* pSmem, uint8_t* pTxBits)
{
    thread_block const& thisThrdBlk  = this_thread_block();
    uint32_t            thrdIdxInBlk = thisThrdBlk.thread_rank();
    uint32_t            nThrdsInBlk  = thisThrdBlk.size();

    uint8_t* const pUnpackedIntlv = reinterpret_cast<uint8_t*>(pSmem); // bit0 of each byte is valid

    //--------------------------------------------------------------------------------------------------------
    // Sub-block interleaving
    // Reference: 3GPP TS 38.212, section 5.4.1.1, Sub-block interleaving
    // The coded bits are divided into 32 sub-blocks. The sub-block size is [1,2,4,8,16] bits respectively for
    // nCodedBits values [32,64,128,256,512]. These sub-blocks are interleaved
    // We know sub_blk_size is a power of 2.
    uint8_t sub_blk_size      = nCodedBits / N_MIN_CODED_BITS;
    uint8_t log2_sub_blk_size = __ffs(sub_blk_size) - 1; // __ffs(1) retuns 1

    uint32_t const* pInCodedWords = reinterpret_cast<uint32_t const*>(pInCodedBits);
    // One thread writes its output bit at bit0 of its byte location. Threads work independently.
    for(uint32_t b = thrdIdxInBlk; b < nCodedBits; b += nThrdsInBlk)
    {
        uint32_t interleaverTblIdx = (b >> log2_sub_blk_size);
        uint32_t interleavedBitPos = (POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[interleaverTblIdx] << log2_sub_blk_size) +
                                     (b & (sub_blk_size - 1));
        uint32_t srcBitPos        = interleavedBitPos;
        uint32_t dstBitPos        = b;
        pUnpackedIntlv[dstBitPos] = getBitValue(pInCodedWords, srcBitPos);
    }
    __syncthreads();

#ifdef DEBUG
    __syncthreads();
    if(thrdIdxInBlk == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("pUnpackedIntlv = [");
        for(int i = 0; i < nCodedBits; i++)
        {
            printf("0x%02x, ", pUnpackedIntlv[i]);
        }
        printf("];\n");
        printf("nCodedBits=%d\n", nCodedBits);
    }
    __syncthreads();
#endif

    // Note that the following also assumes that nCodedBits is a multiple of 32
    static_assert((N_MIN_CODED_BITS == N_BITS_PER_WORD), "Number of coded bits assumed to be >= 32");

    //--------------------------------------------------------------------------------------------------------
    // Bit selection
    // Reference: 3GPP TS 38.212, Section 5.4.1.2, Bit selection
    // GPU uses little-endian byte order.
    uint8_t* pTxBytes = reinterpret_cast<uint8_t*>(pTxBits);
    uint32_t nTxWords = (nTxBits / N_BITS_PER_WORD);
    uint32_t nTxBytes = nTxWords * N_BYTES_PER_WORD; // For padding convenience
    uint32_t E_mod_32 = (nTxBits % N_BITS_PER_WORD);

    // Repetition: e[k] = y[mod(k,N)]
    // Puncturing: e[k] = y[k + N - E];
    // Shortening: e[k] = y[k]
    if(nTxBits >= nCodedBits)
    { // Repetition
        // Each thread reads bit0 from 8 bytes, pack them into one byte. "nCodedBits" is power of 2.
        for(uint32_t dst_byte_idx = thrdIdxInBlk; dst_byte_idx < nTxBytes; dst_byte_idx += nThrdsInBlk)
        {
            uint32_t src_bit_idx   = ((dst_byte_idx * N_BITS_PER_BYTE) & (nCodedBits - 1));
            pTxBytes[dst_byte_idx] = pack8(pUnpackedIntlv, src_bit_idx);
        }
        if((E_mod_32 > 0) && (0 == thrdIdxInBlk))
        {
            // Note E%32==0 at AL=16: No risk of reading out of bounds
            uint32_t  dst_byte_idx = nTxBytes;
            uint32_t  src_bit_idx  = ((dst_byte_idx * N_BITS_PER_BYTE) & (nCodedBits - 1));
            uint32_t  last_word    = pack32(pUnpackedIntlv, src_bit_idx);
            uint32_t  bit_mask     = (1U << E_mod_32) - 1;
            uint32_t* p_last_word  = reinterpret_cast<uint32_t*>(&pTxBytes[dst_byte_idx]);
            *p_last_word           = (last_word & bit_mask);
        }
        __syncthreads();
    }
    else
    { // Puncturing or Shortening
        bool     isPuncturing   = (nInfoBits * 16 <= nTxBits * 7);
        uint32_t srcStartBitPos = isPuncturing ? (nCodedBits - nTxBits) : 0;
        for(uint32_t dst_byte_idx = thrdIdxInBlk; dst_byte_idx < nTxBytes; dst_byte_idx += nThrdsInBlk)
        {
            uint32_t src_bit_idx   = (dst_byte_idx * N_BITS_PER_BYTE + srcStartBitPos);
            pTxBytes[dst_byte_idx] = pack8(pUnpackedIntlv, src_bit_idx);
        }
        if((E_mod_32 > 0) && (0 == thrdIdxInBlk))
        {
            // Note E%32==0 at AL=16: No risk of reading out of bounds
            uint32_t  dst_byte_idx = nTxBytes;
            uint32_t  src_bit_idx  = (dst_byte_idx * N_BITS_PER_BYTE + srcStartBitPos);
            uint32_t  last_word    = pack32(pUnpackedIntlv, src_bit_idx);
            uint32_t  bit_mask     = (1U << E_mod_32) - 1;
            uint32_t* p_last_word  = reinterpret_cast<uint32_t*>(&pTxBytes[dst_byte_idx]);
            *p_last_word           = (last_word & bit_mask);
        }
        __syncthreads();

#ifdef DEBUG
        __syncthreads();
        if(thrdIdxInBlk == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            uint32_t* pTxWords = reinterpret_cast<uint32_t*>(pTxBytes);
            printf("pTxWords = [");
            for(int i = 0; i < nTxWords + 1; i++)
            {
                printf("0x%08x, ", pTxWords[i]);
            }
            printf("];\n");
            printf("nTxBits=%d nCodedBits%d sub_blk_size=%d\n", nTxBits, nCodedBits, sub_blk_size);
        }
        __syncthreads();
#endif
    }
}

//--------------------------------------------------------------------------------------------------------
// Warp-synchronous polar encode + rate match for PDCCH (fused-kernel path; one warp per DCI).
// Bit-exact replacement for encode_pdcch_pbch_LUT + rateMatch, executed by a SINGLE warp with
// no block-wide barriers (only __syncwarp), so it is safe inside a warp-specialized region.
//
// Layout: lane l owns coded-bit word l = bits [32l, 32l+31], LSB-first — the same packing that
// pack8/pack32 produce. N <= 512 coded bits => at most 16 data-carrying lanes.
// The 38.212 5.3.1.2 butterfly (ascending stage order, matching u2d) runs as:
//   - strides 1..16: 5 intra-word mask/shift/XOR stages (no communication);
//   - strides 32..256: up to 4 inter-lane __shfl_xor_sync stages.
// Sub-block interleaving (5.4.1.1) exploits that sub-blocks are N/32 CONTIGUOUS bits which never
// straddle a word boundary; bit selection (5.4.1.2) is word copies (repetition) or a funnel shift
// (puncturing/shortening).
//
// K            - # info bits (payload + CRC)
// N            - # coded bits (power of two, 32..512)
// E            - # rate-matched tx bits (<= CUPHY_PDCCH_MAX_TX_BITS_PER_DCI)
// pInfoBits    - input info bits (bit-reversed payload + CRC, byte packed LSB-first)
// cidx2uidx    - info-bit -> codeword-position LUT for this (K, aggregation level)
// sWarpScratch - shared memory scratch, >= N/32 (<= 16) uint32 words, private to this warp
// pTxWords     - output, ceil(E/32) uint32 words (word aligned; shared or global memory)
static __device__ void encodeRateMatchPdcchWarp(uint32_t K, uint32_t N, uint32_t E,
                                                const uint8_t* __restrict__ pInfoBits,
                                                const uint16_t* __restrict__ cidx2uidx,
                                                uint32_t* sWarpScratch,
                                                uint32_t* pTxWords)
{
    constexpr uint32_t WARP_SZ = 32;
    const uint32_t     lane    = threadIdx.x & (WARP_SZ - 1);
    const uint32_t     nWords  = N / N_BITS_PER_WORD; // power of two, 1..16

    //--------------------------------------------------------------------------------------------------------
    // Encode: scatter the K info bits into the codeword "u" (bit-packed) via shared staging.
    if(lane < nWords) sWarpScratch[lane] = 0;
    __syncwarp();
    for(uint32_t cidx = lane; cidx < K; cidx += WARP_SZ)
    {
        if((pInfoBits[cidx >> 3] >> (cidx & 7)) & 0x1)
        {
            const uint16_t uidx = cidx2uidx[cidx];
            atomicOr(&sWarpScratch[uidx >> 5], 1u << (uidx & 31)); // OR is commutative: deterministic
        }
    }
    __syncwarp();
    uint32_t w = (lane < nWords) ? sWarpScratch[lane] : 0;

    // Butterfly d = u*G, ascending stage order as in u2d.
    // Strides 1..16 stay inside the word: bit p (with (p & stride) == 0) ^= bit (p + stride).
    // N >= 32 always, so all five stages apply.
    w ^= (w >> 1) & 0x55555555u;
    w ^= (w >> 2) & 0x33333333u;
    w ^= (w >> 4) & 0x0F0F0F0Fu;
    w ^= (w >> 8) & 0x00FF00FFu;
    w ^= (w >> 16) & 0x0000FFFFu;
    // Strides 32..N/2 pair word l with word l + d (lane distance d = stride/32):
    // lanes with (lane & d) == 0 accumulate their partner's word.
    for(uint32_t d = 1; (d * 2u * N_BITS_PER_WORD) <= N; d <<= 1)
    {
        const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, w, d);
        if((lane & d) == 0) w ^= other;
    }

    //--------------------------------------------------------------------------------------------------------
    // Sub-block interleaving: stage coded words, then each lane gathers its output word as
    // 32/sbs chunks of sbs = N/32 contiguous bits (chunk k of word m comes from sub-block
    // POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[m * chunks + k]).
    if(lane < nWords) sWarpScratch[lane] = w;
    __syncwarp();
    const uint32_t sbs    = nWords;      // sub-block size in bits (= N / 32), 1..16
    const uint32_t chunks = WARP_SZ / sbs;
    uint32_t       wi     = 0;
    if(lane < nWords)
    {
        const uint32_t chunkMask = (1u << sbs) - 1u; // sbs <= 16
        for(uint32_t k = 0; k < chunks; k++)
        {
            const uint32_t j      = POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[lane * chunks + k];
            const uint32_t srcBit = j * sbs;
            const uint32_t chunk  = (sWarpScratch[srcBit >> 5] >> (srcBit & 31)) & chunkMask;
            wi |= chunk << (k * sbs);
        }
    }
    __syncwarp(); // all gathers done before the staging buffer is overwritten
    if(lane < nWords) sWarpScratch[lane] = wi;
    __syncwarp();

    //--------------------------------------------------------------------------------------------------------
    // Bit selection. Output is ceil(E/32) words; the last word is masked to E%32 bits like
    // the pack32 tail in rateMatch.
    const uint32_t nTxWords  = E / N_BITS_PER_WORD;
    const uint32_t E_mod_32  = E % N_BITS_PER_WORD;
    const uint32_t nOutWords = nTxWords + ((E_mod_32 != 0) ? 1u : 0u);
    if(E >= N)
    { // Repetition: e[k] = y[k mod N]; N is a multiple of 32, so word m repeats word (m mod nWords)
        for(uint32_t m = lane; m < nOutWords; m += WARP_SZ)
        {
            uint32_t word = sWarpScratch[m & (nWords - 1u)];
            if((E_mod_32 != 0) && (m == nOutWords - 1u)) word &= ((1u << E_mod_32) - 1u);
            pTxWords[m] = word;
        }
    }
    else
    { // Puncturing (e[k] = y[k + N - E]) or shortening (e[k] = y[k])
        const bool     isPuncturing = (K * 16u <= E * 7u);
        const uint32_t srcStart     = isPuncturing ? (N - E) : 0u;
        const uint32_t shift        = srcStart & 31u;
        for(uint32_t m = lane; m < nOutWords; m += WARP_SZ)
        {
            const uint32_t w0 = ((m * N_BITS_PER_WORD) + srcStart) >> 5;
            const uint32_t lo = sWarpScratch[w0];
            const uint32_t hi = ((w0 + 1u) < nWords) ? sWarpScratch[w0 + 1u] : 0u;
            uint32_t word     = (shift == 0u) ? lo : __funnelshift_r(lo, hi, shift);
            if((E_mod_32 != 0) && (m == nOutWords - 1u)) word &= ((1u << E_mod_32) - 1u);
            pTxWords[m] = word;
        }
    }
}

/**
 * @brief Compute the offset of a given (log2AL, K) in the PDCCH polar Cidx=>Uidx lookup table (AL for aggregation level).
 *        The table is stored as (log2_AL, K) order, where log2_AL is 1st storage dim;
 *        K is the number of info bits (payload + CRC), in 36..164.
 *        There are 56 invalid (log2_AL, K) combinations, creating 7644 invalid entries and wasting 12% of 126 KiB storgae.
 *        We keep these invalid entries to simplify indexing logic here; invalid entries are set as zeros.
 * @return The offset of a given (log2AL, K) in the PDCCH polar Cidx=>Uidx lookup table.
 */
[[nodiscard]] __host__ __device__ __forceinline__ int pdcch_polar_cidx2uidx_lut_elem_offset(int K, int log2_AL)
{
    if(K < CUPHY_PDCCH_POLAR_K_MIN || K > CUPHY_PDCCH_POLAR_K_MAX || log2_AL < 0 || log2_AL >= CUPHY_PDCCH_POLAR_NUM_AGGR_LEVELS)
    {
        return -1;
    }
    const int span       = K - CUPHY_PDCCH_POLAR_K_MIN;
    const int num_prefix = (CUPHY_PDCCH_POLAR_K_MIN + (K - 1)) * span / 2;
    return num_prefix * CUPHY_PDCCH_POLAR_NUM_AGGR_LEVELS + log2_AL * K;
}

} // namespace polar_encoder

#endif // !defined(CUPHY_POLAR_ENCODER_PDCCH_CUH_INCLUDED_)
