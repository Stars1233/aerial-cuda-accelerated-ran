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

#include "polar_decoder.hpp"
#include "../cuphy_internal.h"
#include "polar_cw_tree_layout.hpp"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <functional>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

#define HFLT_MAX 65504.0

using namespace cooperative_groups;
using namespace cuphy_i;

//#define ENABLE_DEBUG
//#define DEBUG_PRINT

namespace polar_decoder
{

static constexpr int N_MAX_CODED_BITS  = CUPHY_POLAR_DECODER_MAX_BITS;   // biggest polar code word length (1 << 10)
static constexpr int WORD_LENGTH       = sizeof(uint32_t) * 8;           // word length used for storing coded bits
static constexpr int N_MAX_WORDS       = N_MAX_CODED_BITS / WORD_LENGTH; // number of words required to store all bits
static constexpr int BCO               = 3;                              // Bank conflict offset to minimize bank conflicts in list polar decoder
static constexpr int CUDA_WARP_SIZE             = 32;
static constexpr int POLAR_DECODER_BLOCK_SIZE   = CUDA_WARP_SIZE;        // threads per codeword (one warp)
// Number of codewords decoded per block by the single/SC decoder kernel for
// large batches. Each codeword is handled by one warp with its own dynamic
// shared-memory slice; packing several warps per block lifts the
// resident-block occupancy limit, which otherwise caps large launches at
// half the warp slots. Small (latency-critical) batches keep one warp per
// block; they cannot saturate the SMs anyway and single-warp blocks preserve
// the shortest per-codeword critical path.
static constexpr int POLAR_SC_WARPS_PER_BLOCK   = 4;
static constexpr int POLAR_SC_MULTIWARP_MIN_CWS = 128;

// The decoder currently launches one warp per codeword. Preserve block-wide
// synchronization semantics if a future launch configuration uses more threads.
template<uint32_t BLOCK_SIZE>
__device__ __forceinline__ void sync_barrier()
{
    if constexpr(BLOCK_SIZE == CUDA_WARP_SIZE)
    {
        __syncwarp();
    }
    else
    {
        __syncthreads();
    }
}

// clang-format off
// depth array stored in constant mem
static __device__ __constant__  uint8_t POLAR_DEPTH[N_MAX_CODED_BITS] =
       {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1,
        0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
        2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2,
        0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
        1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
        0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3,
        0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0,
        1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
        0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
        2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
        0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0,
        1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
        0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
        3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1,
        0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
        2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2,
        0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 9, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
        1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1,
        0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
        0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
        1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
        0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0,
        2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
        0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
        1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
        0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0,
        3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1,
        0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
        2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2,
        0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
        1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
        0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3,
        0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 10};

// CRC8 LUT based on polynomial = 388 (b110000100)
static __device__ uint8_t CRC8_LUT[256] = {
    0,   	132,	140,	8,  	156,	24,	    16,	    148,
    188,	56,	    48,	    180,	32,	    164,	172,	40,
    252,	120,	112,	244,	96,	    228,	236,	104,
    64,	    196,	204,	72,	    220,	88,	    80,	    212,
    124,	248,	240,	116,	224,	100,	108,	232,
    192,	68,	    76,	    200,	92,	    216,	208,	84,
    128,	4,	    12,	    136,	28,	    152,	144,	20,
    60,	    184,	176,	52,	    160,	36,	    44, 	168,
    248,	124,	116,	240,	100,	224,	232,	108,
    68,	    192,	200,	76, 	216,	92,	    84, 	208,
    4,	    128,	136,	12,	    152,	28,	    20, 	144,
    184,	60,	    52,	    176,	36,	    160,	168,	44,
    132,	0,	    8,  	140,	24,	    156,	148,	16,
    56,	    188,	180,	48, 	164,	32,	    40, 	172,
    120,	252,	244,	112,	228,	96, 	104,	236,
    196,	64,	    72, 	204,	88,	    220,	212,	80,
    116,	240,	248,	124,	232,	108,	100,	224,
    200,	76,	    68,	    192,	84,	    208,	216,	92,
    136,	12,	    4,	    128,	20,	    144,	152,	28,
    52,	    176,	184,	60,	    168,	44,	    36, 	160,
    8,	    140,	132,	0,  	148,	16, 	24, 	156,
    180,	48,	    56,	    188,	40,	    172,	164,	32,
    244,	112,	120,	252,	104,	236,	228,	96,
    72, 	204,	196,	64,	    212,	80,	    88, 	220,
    140,	8,	    0,	    132,	16, 	148,	156,	24,
    48,	    180,	188,	56, 	172,	40, 	32,	    164,
    112,	244,	252,	120,	236,	104,	96, 	228,
    204,	72,	    64, 	196,	80,	    212,	220,	88,
    240,	116,	124,	248,	108,	232,	224,	100,
    76, 	200,	192,	68,	    208,	84,	    92, 	216,
    12,	    136,	128,	4,	    144,	20,	    28, 	152,
    176,	52, 	60, 	184,	44,	    168,	160,	36
};

// CRC16 LUT based on polynomial = 50208 (b11000100 00100000)
static __device__ uint16_t CRC16_LUT[256] = {
    0,	    50208,	19552,	34880,	39104,	23776,	54432,	4224,
    62880,	12672,	47552,	32224,	28000,	43328,	8448,	58656,
    12128,	60224,	25344,	42784,	47008,	29568,	64448,	16352,
    56000,	7904,	38560,	21120,	16896,	34336,	3680,	51776,
    24256,	39648,	4768,	54912,	50688,	544,	35424,	20032,
    43872,	28480,	59136,	8992,	13216,	63360,	32704,	48096,
    29088,	46464,	15808,	63968,	59744,	11584,	42240,	24864,
    33792,	16416,	51296,	3136,	7360,	55520,	20640,	38016,
    48512,	31136,	61920,	13760,	9536,	57696,	26912,	44288,
    18464,	35840,	1088,	49248,	53472,	5312,	40064,	22688,
    37600,	22208,	56960,	6816,	2592,	52736,	17984,	33376,
    26432,	41824,	11040,	61184,	65408,	15264,	46048,	30656,
    58176,	10080,	44832,	27392,	31616,	49056,	14304,	62400,
    5856,	53952,	23168,	40608,	36384,	18944,	49728,	1632,
    52256,	2048,	32832,	17504,	21728,	37056,	6272,	56480,
    14720,	64928,	30176,	45504,	41280,	25952,	60704,	10496,
    48928,	31488,	62272,	14176,	10208,	58304,	27520,	44960,
    19072,	36512,	1760,	49856,	53824,	5728,	40480,	23040,
    36928,	21600,	56352,	6144,	2176,	52384,	17632,	32960,
    26080,	41408,	10624,	60832,	64800,	14592,	45376,	30048,
    57824,	9664,	44416,	27040,	31008,	48384,	13632,	61792,
    5184,	53344,	22560,	39936,	35968,	18592,	49376,	1216,
    52864,	2720,	33504,	18112,	22080,	37472,	6688,	56832,
    15136,	65280,	30528,	45920,	41952,	26560,	61312,	11168,
    672,	50816,	20160,	35552,	39520,	24128,	54784,	4640,
    63232,	13088,	47968,	32576,	28608,	44000,	9120,	59264,
    11712,	59872,	24992,	42368,	46336,	28960,	63840,	15680,
    55392,	7232,	37888,	20512,	16544,	33920,	3264,	51424,
    23648,	38976,	4096,	54304,	50336,	128,	35008,	19680,
    43456,	28128,	58784,	8576,	12544,	62752,	32096,	47424,
    29440,	46880,	16224,	64320,	60352,	12256,	42912,	25472,
    34464,	17024,	51904,	3808,	7776,	55872,	20992,	38432
};
// clang-format on

template <typename T>
__device__ __forceinline__ T find_msb(T in)
{
    T out = 0;
    while(in > 1)
    {
        in = in >> 1;
        out++;
    }
    return out;
}

// if a and b are same sign, return +1, otherwise return -1
__device__ __forceinline__ __half signof(__half x, __half y)
{
    const __half h0 = __float2half(0.f);
    const bool same_sign = (__hgt(x, h0) == __hgt(y, h0));
    return same_sign ? __float2half(1.f) : __float2half(-1.f);
}

// vectorized singof
__device__ __forceinline__ __half2 signof2(__half2 a, __half2 b)
{
    union Half2U {
        __half2 h2;
        uint32_t u32;
    };

    Half2U ua{a};
    Half2U ub{b};
    Half2U ur;

    // XOR the sign bits of a and b, then mask out everything except the sign bits
    // Using XOR, if a and b have the same sign, corresponding sign bit is 0, otherwise it is 1
    uint32_t sign_mask = (ua.u32 ^ ub.u32) & 0x80008000u;

    // 0x3C00 = +1.0 in half, so 0x3C003C00 = {+1, +1} in half2
    // XORing with sign_mask flips the sign bit per lane if signs differ
    ur.u32 = 0x3C003C00u ^ sign_mask;

    return ur.h2;
}

// ==== register-resident low sub-tree (sub-trees of size <= 32) ================================
// The SC decoder spends most of its node visits near the leaves, on small
// sub-trees. Rather than run F/G/hard-decision on those through the LLR array
// in (L1-cached) global memory -- a load/store plus a __syncwarp per stage --
// the bottom 64 LLR values of the tree are held in registers and worked on with
// warp shuffles, which are register-to-register and implicitly synchronized.
//
// Layout: `lowTreeLLR` is one __half2 per lane; lane l owns tree elements
// (2l, 2l+1), so the 32 lanes cover the 64 low elements. A node of size sz
// reads its two inputs from elements [2sz,3sz) ("a") and [3sz,4sz) ("b") and
// writes its output to [sz,2sz), matching the array layout the vector F/G use.
//
// The SC main loop dispatches by sub-tree size:
//   sz  > 32 : F_func / G_func         -- the original array (memory) path
//   sz == 32 : F_lowTreeFromMem / ...  -- boundary: read the array inputs once
//              and populate the register tree (elements [32,64))
//   sz  < 32 : F_lowTree / G_lowTree   -- pure register/shuffle, no memory, no sync
// The arithmetic is identical to the vector path, so decoding is bit-exact.
// (The fast-SSC REP/SPC handlers reuse these for their small-node tail only;
// they do their larger-node work in the memory array -- see rep_decide. That
// is code reuse, not an algorithmic dependency.)

// box-plus (min-sum) on a half2 pair; same arithmetic as the F_func vector path
__device__ __forceinline__ __half2 F_pair(__half2 a2, __half2 b2)
{
    __half2 minAbs2 = __hmin2(__habs2(a2), __habs2(b2));
    return __hmul2(signof2(a2, b2), minAbs2);
}

// F (min-sum) for a sub-tree of size sz < 32, held entirely in the register
// tree: inputs and output are fetched/stored by warp shuffle, no memory.
__device__ __forceinline__ void F_lowTree(__half2& lowTreeLLR, int32_t sz, uint32_t lane)
{
    if(sz == 1)
    {
        const __half2 in = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, 1);
        if(lane == 0)
        {
            const __half aAbs   = __habs(__low2half(in));
            const __half bAbs   = __habs(__high2half(in));
            const __half minAbs = __hlt(aAbs, bAbs) ? aAbs : bAbs;
            lowTreeLLR = __halves2half2(__low2half(lowTreeLLR), __hmul(signof(__low2half(in), __high2half(in)), minAbs));
        }
        return;
    }
    const __half2 a2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, lane + (sz >> 1));
    const __half2 b2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, lane + sz);
    if((lane >= static_cast<uint32_t>(sz >> 1)) && (lane < static_cast<uint32_t>(sz)))
    {
        lowTreeLLR = F_pair(a2, b2);
    }
}

// F at the sz == 32 boundary: this is where the descent enters the register
// tree. Read the two input rows from the LLR array in memory and populate the
// upper register elements [32,64); smaller stages then stay in registers.
__device__ __forceinline__ void F_lowTreeFromMem(__half2& lowTreeLLR, const __half* __restrict__ cwTreeLLR, uint32_t lane)
{
    if(lane >= 16)
    {
        const __half2* a2 = reinterpret_cast<const __half2*>(&cwTreeLLR[64]);
        const __half2* b2 = reinterpret_cast<const __half2*>(&cwTreeLLR[96]);
        lowTreeLLR = F_pair(a2[lane - 16], b2[lane - 16]);
    }
}

// G for a sub-tree of size sz < 32, held entirely in the register tree; `est`
// is the sibling's boolean codeword estimate (same sign arithmetic as G_func).
__device__ __forceinline__ void G_lowTree(__half2& lowTreeLLR, const bool* __restrict__ est, int32_t sz, uint32_t lane)
{
    if(sz == 1)
    {
        const __half2 in = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, 1);
        if(lane == 0)
        {
            const __half u = est[0] ? __float2half(-1.f) : __float2half(1.f);
            lowTreeLLR = __halves2half2(__low2half(lowTreeLLR), __hfma(u, __low2half(in), __high2half(in)));
        }
        return;
    }
    const __half2 a2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, lane + (sz >> 1));
    const __half2 b2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, lane + sz);
    if((lane >= static_cast<uint32_t>(sz >> 1)) && (lane < static_cast<uint32_t>(sz)))
    {
        const uint32_t bits = reinterpret_cast<const uint16_t*>(est)[lane - (sz >> 1)];
        union {
            __half2 h2;
            uint32_t u32;
        } u;
        u.u32  = 0x3C003C00u ^ ((bits & 0x1u) << 15) ^ ((bits & 0x100u) << 23);
        lowTreeLLR = __hfma2(u.h2, a2, b2);
    }
}

// G at the sz == 32 boundary: read the input rows from the LLR array in memory
// and populate the upper register elements [32,64) (see F_lowTreeFromMem).
__device__ __forceinline__ void G_lowTreeFromMem(__half2& lowTreeLLR, const __half* __restrict__ cwTreeLLR, const bool* __restrict__ est, uint32_t lane)
{
    if(lane >= 16)
    {
        const __half2* a2 = reinterpret_cast<const __half2*>(&cwTreeLLR[64]);
        const __half2* b2 = reinterpret_cast<const __half2*>(&cwTreeLLR[96]);
        const uint32_t bits = reinterpret_cast<const uint16_t*>(est)[lane - 16];
        union {
            __half2 h2;
            uint32_t u32;
        } u;
        u.u32  = 0x3C003C00u ^ ((bits & 0x1u) << 15) ^ ((bits & 0x100u) << 23);
        lowTreeLLR = __hfma2(u.h2, a2[lane - 16], b2[lane - 16]);
    }
}

// Hard-decide a node of size sz <= 32 straight from the register tree (sign of
// each LLR), writing the boolean codeword estimate cs[0..sz).
__device__ __forceinline__ void hardDecision_lowTree(bool* __restrict__ cs, __half2 lowTreeLLR, int32_t sz, uint32_t lane)
{
    const int32_t e  = sz + static_cast<int32_t>(lane); // tree element holding bit `lane`
    const __half2 v2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, e >> 1);
    if(lane < static_cast<uint32_t>(sz))
    {
        const __half v = (e & 1) ? __high2half(v2) : __low2half(v2);
        cs[lane] = __hgt(v, 0) ? 0 : 1;
    }
}

// REP node (fast-SSC): all leaves frozen except the last. Under min-sum every
// G step of the descent sees an all-zero estimate, so the information bit's
// LLR is the strided pairwise half-precision sum of the node's input LLRs.
// The reduction below reproduces the exact combining order (and rounding) of
// the descent: out[i] = in[i] + in[i + len/2], level by level. Returns the
// hard decision of the single information bit.
__device__ __forceinline__ bool rep_decide(__half* __restrict__ cwTreeLLR, __half2& lowTreeLLR, int32_t sz, const uint32_t lane)
{
    int32_t len = sz;
    while(len > 64)
    {
        const __half2* a2 = reinterpret_cast<const __half2*>(&cwTreeLLR[len]);
        const __half2* b2 = reinterpret_cast<const __half2*>(&cwTreeLLR[len + (len >> 1)]);
        __half2*       o2 = reinterpret_cast<__half2*>(&cwTreeLLR[len >> 1]);
        for(int32_t i = lane; i < (len >> 2); i += warpSize)
        {
            o2[i] = __hadd2(a2[i], b2[i]);
        }
        sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
        len >>= 1;
    }
    if(len == 64)
    {
        // bridge into the register region: out elements [32,64)
        if(lane >= 16)
        {
            const __half2* a2 = reinterpret_cast<const __half2*>(&cwTreeLLR[64]);
            const __half2* b2 = reinterpret_cast<const __half2*>(&cwTreeLLR[96]);
            lowTreeLLR = __hadd2(a2[lane - 16], b2[lane - 16]);
        }
        len = 32;
    }
    // register phase: gather element (len + lane) and reduce with the same
    // strided order down to one value on lane 0
    const int32_t e  = len + static_cast<int32_t>(lane);
    const __half2 v2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, e >> 1);
    __half        v  = (e & 1) ? __high2half(v2) : __low2half(v2);
    for(int32_t half = len >> 1; half >= 1; half >>= 1)
    {
        const __half o = __shfl_sync(0xFFFFFFFFu, v, lane + half);
        if(lane < static_cast<uint32_t>(half))
        {
            v = __hadd(v, o);
        }
    }
    const __half total = __shfl_sync(0xFFFFFFFFu, v, 0);
    return __hgt(total, __float2half(0.f)) ? 0 : 1;
}

// SPC node (fast-SSC): all leaves info except the first (a frozen parity
// bit). Min-sum equivalent of the descent: keep the hard decisions and, if
// their parity is odd, flip the least-reliable position (first occurrence on
// ties). cs already holds the hard decisions.
__device__ __forceinline__ void spc_fix(bool* __restrict__ cs, const __half* __restrict__ cwTreeLLR, __half2 lowTreeLLR, int32_t sz, const uint32_t lane)
{
    float    m   = HFLT_MAX;
    int32_t  mi  = 0;
    uint32_t par = 0;
    if(sz <= 32)
    {
        const int32_t e  = sz + static_cast<int32_t>(lane);
        const __half2 v2 = __shfl_sync(0xFFFFFFFFu, lowTreeLLR, e >> 1);
        if(lane < static_cast<uint32_t>(sz))
        {
            const __half v = (e & 1) ? __high2half(v2) : __low2half(v2);
            m  = fabsf(__half2float(v));
            mi = static_cast<int32_t>(lane);
        }
        const uint32_t bits = __ballot_sync(0xFFFFFFFFu, (lane < static_cast<uint32_t>(sz)) && cs[lane]);
        par = __popc(bits) & 1u;
    }
    else
    {
        uint32_t lpar = 0;
        for(int32_t i = lane; i < sz; i += warpSize)
        {
            const float a = fabsf(__half2float(cwTreeLLR[sz + i]));
            if(a < m)
            {
                m  = a;
                mi = i;
            }
            lpar ^= cs[i] ? 1u : 0u;
        }
        par = __popc(__ballot_sync(0xFFFFFFFFu, (lpar & 1u) != 0)) & 1u;
    }
    // lexicographic (value, index) minimum keeps the first occurrence on ties
    for(uint32_t off = 16; off > 0; off >>= 1)
    {
        const float   om = __shfl_down_sync(0xFFFFFFFFu, m, off);
        const int32_t oi = __shfl_down_sync(0xFFFFFFFFu, mi, off);
        if((om < m) || ((om == m) && (oi < mi)))
        {
            m  = om;
            mi = oi;
        }
    }
    mi = __shfl_sync(0xFFFFFFFFu, mi, 0);
    if((par != 0) && (lane == 0))
    {
        cs[mi] = !cs[mi];
    }
    __syncwarp();
}

// to return type of polar tree node
// type 0 => both child nodes are type 0 (for stage 0, it means frozen bits), no need to traverse the tree
// type 1 => both child nodes are type 1 (for stage 0, it means info bits), no need to traverse the tree
// type 3 => two child nodes are mix of type 0 and 1, traverse the tree as usual
__device__ __forceinline__ uint8_t get_type(int32_t stage, int32_t n, int32_t sub_idx, const uint8_t* __restrict__ treeTypes)
{
    const int32_t node_idx = (1 << (n - stage)) + sub_idx;
    return __ldg(&treeTypes[node_idx]);//treeTypes[node_idx];
}

    // similar to get_type, but reading 8-values of uint8_t at a time
__device__ __forceinline__ uint8x8 get_type8(int32_t stage, int32_t n, int32_t sub_idx, const uint8x8* __restrict__ treeTypes)
{
    const int32_t node_idx = ((1 << (n - stage)) + sub_idx) / 8;
    uint8x8 type8;
    type8.u64 = __ldg(&treeTypes[node_idx].u64);//treeTypes[node_idx].u64;
    return type8;
}

// Apply the lowest min(stage, 5) XOR-butterfly stages inside one 32-bit word.
__device__ __forceinline__ uint32_t xor_butterfly_word(uint32_t wrd, int32_t stage)
{
    if(stage >= 5) { wrd ^= (wrd & 0xFFFF0000u) >> 16; }
    if(stage >= 4) { wrd ^= (wrd & 0xFF00FF00u) >> 8; }
    if(stage >= 3) { wrd ^= (wrd & 0xF0F0F0F0u) >> 4; }
    if(stage >= 2) { wrd ^= (wrd & 0xCCCCCCCCu) >> 2; }
    if(stage >= 1) { wrd ^= (wrd & 0xAAAAAAAAu) >> 1; }
    return wrd;
}

// XOR butterfly over the (1 << stage) codeword bits held in cs, computed in
// registers: bit word k of the result ends up in lane k's return value
// (lanes >= ceil(sz/32) hold undefined words). Word-level stages use warp
// shuffles; the five lowest stages use in-word bit arithmetic.
__device__ __forceinline__ uint32_t xor_butterfly_regs(const bool* __restrict__ cs, const int32_t stage, const int32_t sz, const uint32_t lane)
{
#ifdef _DEBUG
    assert(sz == (1 << stage));
#endif
    const int32_t nWrds  = div_round_up(sz, static_cast<int32_t>(WORD_LENGTH));
    uint32_t      myWord = 0;
    for(int32_t k = 0; k < nWrds; k++)
    {
        const int32_t  i    = k * WORD_LENGTH + lane;
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, (i < sz) && cs[i]);
        if(lane == static_cast<uint32_t>(k))
        {
            myWord = word;
        }
    }

    // butterfly stages with bit jumps >= 32 combine whole words across lanes
    for(int32_t j = stage; j > 5; j--)
    {
        const uint32_t jump  = 1u << (j - 6);
        const uint32_t mask  = (jump << 1) - 1;
        const uint32_t other = __shfl_sync(0xFFFFFFFFu, myWord, (lane + jump) & (CUDA_WARP_SIZE - 1));
        if((lane < static_cast<uint32_t>(nWrds)) && ((lane & mask) < jump))
        {
            myWord ^= other;
        }
    }

    // remaining stages stay inside each 32-bit word
    return xor_butterfly_word(myWord, stage < 5 ? stage : 5);
}

// Store the codeword estimate cs[0,sz) into the packed estimate buffer at bit
// offset sz using warp ballots.
__device__ __forceinline__ void store_est_bits(uint32_t* __restrict__ estW, const bool* __restrict__ cs, int32_t sz, const uint32_t lane)
{
    if(sz >= static_cast<int32_t>(WORD_LENGTH))
    {
        const int32_t wordsPerHalf = sz / WORD_LENGTH;
        for(int32_t w = 0; w < wordsPerHalf; w++)
        {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, cs[w * WORD_LENGTH + lane]);
            if(lane == 0)
            {
                estW[wordsPerHalf + w] = word;
            }
        }
    }
    else
    {
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, (lane < static_cast<uint32_t>(sz)) && cs[lane]);
        if(lane == 0)
        {
            const uint32_t maskSz = (1u << sz) - 1;
            estW[0] = (estW[0] & ~(maskSz << sz)) | (word << sz);
        }
    }
}

// F function: implementation of box-plus (min-sum)
// combine 2 arrays of length sz/2 and output an array of length sz/2
__device__ __forceinline__ void F_func(__half* __restrict__ llrOut, const __half* __restrict__ llrIn, int32_t sz, const uint32_t lane)
{
    const __half* a = llrIn;      // first half input
    const __half* b = &llrIn[sz]; // second half input
    if(sz == 1)
    {
        __half minAbs = __hlt(__habs(a[0]), __habs(b[0])) ? __habs(a[0]) : __habs(b[0]);
        llrOut[0]      = __hmul(signof(a[0], b[0]), minAbs);
    }
    else
    {
        // reinterpret as half2 to vectorize processing
        const __half2 *a2 = reinterpret_cast<const __half2*>(a);
        const __half2 *b2 = reinterpret_cast<const __half2*>(b);
        __half2 *o2 = reinterpret_cast<      __half2*>(llrOut);

        for (int32_t i = lane; i < (sz >> 1); i += warpSize)
        {
            __half2 a2i = a2[i];
            __half2 b2i = b2[i];

#if __CUDA_ARCH__ >= 800
            __half2 minAbs2 = __hmin2(__habs2(a2i), __habs2(b2i));
#else
            __half2 aAbs = __habs2(a2i);
            __half2 bAbs = __habs2(b2i);
            __half2 minAbs2;
            minAbs2.x = __hlt(aAbs.x, bAbs.x) ? aAbs.x : bAbs.x;
            minAbs2.y = __hlt(aAbs.y, bAbs.y) ? aAbs.y : bAbs.y;
#endif

            __half2 sign2   = signof2(a2i, b2i);

            o2[i] = __hmul2(sign2, minAbs2);
        }
    }
}

// G function: implementation of repetition likelihood
// combine 2 arrays of length sz/2 based on bit array Est0 and output an array of length sz/2
// examine different variants impact on performance
__device__ __forceinline__ void G_func(__half* __restrict__ llrOut, const __half* __restrict__ llrIn, const bool* __restrict__ est, int32_t sz, const uint32_t lane)
{
    const __half* a = llrIn;      // first half input
    const __half* b = &llrIn[sz]; // second half input

    if(sz == 1)
    {
        const __half u = est[0] ? __float2half(-1.f) : __float2half(1.f);
        llrOut[0] = __hfma(u, a[0], b[0]);
        return;
    }

    const auto* a2 = reinterpret_cast<const __half2*>(a);
    const auto* b2 = reinterpret_cast<const __half2*>(b);
    auto* out2     = reinterpret_cast<__half2*>(llrOut);
    const auto* est2 = reinterpret_cast<const uint16_t*>(est);

    for(int32_t i = lane; i < (sz >> 1); i += warpSize)
    {
        const uint32_t bits = est2[i];
        union {
            __half2 h2;
            uint32_t u32;
        } u;
        u.u32 = 0x3C003C00u ^ ((bits & 0x1u) << 15) ^ ((bits & 0x100u) << 23);
        out2[i] = __hfma2(u.h2, a2[i], b2[i]);
    }
}

// H function: implementation of combining codewords in polar code
// unlike F and G, H input/output arrays of hard decisions (bits)
// combine 2 arrays of length sz/2 and output an array of length sz/2
// est_in0: stage estimate read from the packed estimate buffer at bit offset sz
// est_in1: input from second child node (boolean array), size sz/2
__device__ __forceinline__ void H_func(bool* __restrict__ estOut, const uint32_t* __restrict__ estW, const bool* __restrict__ estIn1, int32_t sz, const uint32_t lane)
{
    bool* out0 = estOut;      // output first m values
    bool* out1 = &estOut[sz]; // output second m values
    if(sz == 1)
    {
        out0[0] = (((estW[0] >> 1) & 1u) != 0) != estIn1[0];
        out1[0] = estIn1[0];
    }
    else
    {
        for(int32_t i = lane; i < sz; i += warpSize)
        {
            const int32_t estBit = sz + i;
            const bool in0 = ((estW[estBit / WORD_LENGTH] >> (estBit % WORD_LENGTH)) & 1u) != 0;
            out0[i] = in0 != estIn1[i]; // boolean xor, similar to (in0 + in1) % 2
            out1[i] = estIn1[i];
        }
    }
}

// bit-manipulation functions ====================================================================

// `xorBit()` is a legacy/utility device helper that is not called by any shipped kernel path in
// the polar decoder. No test can cover it through the public API. Exclude from coverage.
/*VCAST_DONT_INSTRUMENT_START*/
__device__ __forceinline__ void xorBit(uint32_t* __restrict__ Z, int32_t n1, uint32_t g, int32_t n2)
{
    int32_t  wIdx = n1 / WORD_LENGTH;
    int32_t  bIdx = n1 % WORD_LENGTH;
    uint32_t w32  = Z[wIdx];
    // move the corresponding bits to LSB
    w32 = 1 & (w32 >> bIdx);
    g   = 1 & (g >> n2);
    // xor and update Z[wIdx]
    if(g == w32)
    {
        // reset bit
        Z[wIdx] = Z[wIdx] & ~(1 << bIdx);
    }
    else
    {
        // set bit
        Z[wIdx] = Z[wIdx] | (1 << bIdx);
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

__device__ __forceinline__ uint8_t isBitSet(const uint32_t* __restrict__ Z, int32_t idx)
{
    int32_t  wIdx  = idx / WORD_LENGTH;
    int32_t  bIdx  = idx % WORD_LENGTH;
    uint32_t w32   = Z[wIdx];
    uint8_t  isSet = static_cast<uint8_t>((w32 >> bIdx) & uint32_t(1));//(w32 & (1 << bIdx)) == 0 ? 0 : 1;
    return isSet;
}

__device__ __forceinline__ uint8_t isBitSet(const uint32_t w32, uint32_t idx)
{
    return (static_cast<uint8_t>((w32 >> idx) & uint32_t(1)));
}

__device__ __forceinline__ uint64_t isBitSet8(const uint32_t w32, uint32_t idx)
{
    bool8 res;

    //for(int i = 0; i < 8; i++) { res.b8[i] = (w32 >> (i + idx)) & uint32_t(1); }
    const uint32_t bits = w32 >> idx;
    res.b8[0] = (bits & 0x01) == 0x01;
    res.b8[1] = (bits & 0x02) == 0x02;
    res.b8[2] = (bits & 0x04) == 0x04;
    res.b8[3] = (bits & 0x08) == 0x08;
    res.b8[4] = (bits & 0x10) == 0x10;
    res.b8[5] = (bits & 0x20) == 0x20;
    res.b8[6] = (bits & 0x40) == 0x40;
    res.b8[7] = (bits & 0x80) == 0x80;

    return res.u64;
}

// This is a `__device__` template overload. The polar decoder uses the pointer overload
// `setResetSingleBit(uint32_t* wrdArray, ...)` in all executed paths; this template overload is
// not instantiated/executed in the instrumented kernels, so statement/branch coverage will show
// it uncovered. Exclude from coverage unless a real call site is added.
/*VCAST_DONT_INSTRUMENT_START*/
template<typename T>
__device__ __forceinline__ void setResetSingleBit(T& wrd, uint32_t idx, uint8_t val) {
#ifdef _DEBUG
    assert(sizeof(T) > (idx >> 3));
#endif
    if (val == 0) {
        wrd &= ~(1 << idx);
    } else {
        wrd |= (1 << idx);
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

// return a single 32-bit word starting from bitIdx in wrdArray
__device__ __forceinline__ uint32_t getWord(const uint32_t* __restrict__ wrdArray, const uint32_t arrSize, const uint32_t bitIdx) {
    uint32_t wIdx = bitIdx / WORD_LENGTH;
    uint32_t bIdx = bitIdx % WORD_LENGTH;

    uint32_t wrd = wrdArray[wIdx];
    uint32_t wrd_nxt;
    if(bIdx > 0)
    {
        wrd_nxt = (wIdx + 1) < arrSize ? wrdArray[wIdx + 1] : 0;
        wrd     = __funnelshift_rc(wrd, wrd_nxt, bIdx);
    }
    return wrd;
}

// sets bits bitIdx in wrdArray to val
// (note there is no out of bound check)
__device__ __forceinline__ void setResetSingleBit(uint32_t* __restrict__ wrdArray, uint32_t bitIdx, uint8_t val) {
    uint32_t wIdx = bitIdx / WORD_LENGTH;
    uint32_t bIdx = bitIdx % WORD_LENGTH;

    if (val == 0) {
        atomicAnd(wrdArray + wIdx, ~(1 << bIdx));   //wrdArray[wIdx] &= ~(1 << bIdx);
    } else {
        atomicOr(wrdArray + wIdx, (1 << bIdx));     //wrdArray[wIdx] |= (1 << bIdx);
    }
}

// reset sz bits in wrdArray starting from bitIdx
__device__ __forceinline__ void resetBits(uint32_t* __restrict__ wrdArray, uint32_t bitIdx, uint32_t sz)
{
    uint32_t  wIdx  = bitIdx / WORD_LENGTH;
    uint32_t  bIdx  = bitIdx % WORD_LENGTH;

    if (sz == 1) {
        // we can reset a single bit faster than using the general approach below
        atomicAnd(wrdArray + wIdx, ~(1 << bIdx)); //wrdArray[wIdx] &= ~(1 << bIdx);
        return;
    }

    constexpr uint32_t ones = 0xffffffff;
    constexpr uint32_t zeros = 0x00000000;
    int mask;
    // depending on sz and idx, we may have to reset (head-word + mid-words + tail-words)
    // where mid-words or tail-words may or may not exist

    // update the head-word
    uint32_t num_head_bits = WORD_LENGTH - bIdx;
    if (num_head_bits <= sz) {
        mask = __funnelshift_rc(ones, zeros, num_head_bits);
    } else {
        mask = __funnelshift_rc(ones, zeros, sz);
        mask = __funnelshift_rc(mask, ones, (num_head_bits - sz));
    }
    atomicAnd(wrdArray + wIdx, mask);   //wrdArray[wIdx] &= mask;

    // if num_head_bits >= sz, we only need to update the head-word
    // otherwise, we need to update mid-words or tail-word
    if (num_head_bits < sz) {
        uint32_t remaining_bits = sz - num_head_bits;
        uint32_t num_mid_words = remaining_bits / WORD_LENGTH;
        for (uint32_t i = 0; i < num_mid_words; i++) {
            wrdArray[wIdx + 1 + i] = 0;
        }
        // check if partially resetting tail-word is needed
        uint32_t num_tail_bits = remaining_bits % WORD_LENGTH;
        if (num_tail_bits > 0) {
            mask = __funnelshift_lc(zeros, ones, num_tail_bits);
            wrdArray[wIdx + num_mid_words + 1] &= mask;
        }
    }
}

// Bits are copied to Dst array, starting from bitIdxDst
// note: start index for Src is always 0, for general case,
// we need to implement an approach similar to that used in xorBits function
__device__ __forceinline__ void copyBits(const uint32_t* __restrict__ Src, uint32_t sz, uint32_t* __restrict__ Dst, uint32_t bitIdxDst)
{
    uint32_t wrd     = Src[0];

    uint32_t wIdx_d  = bitIdxDst / WORD_LENGTH;
    uint32_t bIdx_d  = bitIdxDst % WORD_LENGTH;

    if (sz == 1) {
        // we can copy (the first) single bit faster than using the general approach below
        uint32_t bit = wrd & uint32_t(1);
        if (bit == 0) {
            Dst[wIdx_d] &= ~(1 << bIdx_d);
        } else {
            Dst[wIdx_d] |= (1 << bIdx_d);
        }
        return;
    }

    constexpr uint32_t ones = 0xffffffff;
    constexpr uint32_t zeros = 0x00000000;

    int maskZeros, maskOnes;
    // depending on sz and idx, we may have to copy (head-word + mid-words + tail-words)
    // where mid-words or tail-words may or may not exist

    // update the head-word
    uint32_t numHeadBits = WORD_LENGTH - bIdx_d;
    if (numHeadBits <= sz) {
        maskZeros  = __funnelshift_rc(ones, wrd, numHeadBits);
        maskOnes   = __funnelshift_rc(zeros, wrd, numHeadBits);
    } else {
        maskZeros  = __funnelshift_rc(ones, wrd, sz);
        maskZeros  = __funnelshift_rc(maskZeros, ones, (numHeadBits - sz));
        maskOnes   = __funnelshift_rc(zeros, wrd, sz);
        maskOnes   = __funnelshift_rc(maskOnes, zeros, (numHeadBits - sz));
    }
    Dst[wIdx_d] &= maskZeros; // copy zeros
    Dst[wIdx_d] |= maskOnes; // copy ones

    // if num_head_bits >= sz, we only need to update the head-word
    // otherwise, we need to update mid-words or tail-word
    if (numHeadBits < sz) {
        uint32_t remainingBits = sz - numHeadBits;
        uint32_t numMidWords   = remainingBits / WORD_LENGTH;
        uint32_t wrd_nxt;
        for (uint32_t i = 0; i < numMidWords; i++) {
            wrd_nxt             = Src[i + 1];
            wrd                 = __funnelshift_lc(wrd, wrd_nxt, bIdx_d);
            Dst[wIdx_d + 1 + i] = wrd;
            wrd                 = wrd_nxt;
        }

        // check if partially resetting tail-word is needed
        uint32_t numTailBits = remainingBits % WORD_LENGTH;

        if (numTailBits > 0) {
            // The remaining source bits start at bit numHeadBits + 32*numMidWords.
            // They may straddle two source words when neither sz nor the
            // destination offset is word aligned, so read them with getWord()
            // rather than indexing a single word. (The former funnelshift-based
            // tail took the top bits of the last source word, which is only
            // correct when sz is a multiple of the word length.)
            uint32_t srcBitIdx = numHeadBits + numMidWords * WORD_LENGTH;
            uint32_t srcWords  = div_round_up(sz, static_cast<uint32_t>(WORD_LENGTH));
            uint32_t tailMask  = (1u << numTailBits) - 1;
            uint32_t tailBits  = getWord(Src, srcWords, srcBitIdx) & tailMask;
            Dst[wIdx_d + numMidWords + 1] &= (~tailMask | tailBits); // copy zeros
            Dst[wIdx_d + numMidWords + 1] |= tailBits;               // copy ones
        }
    }
}

// Start indices for sources can be different, they can also be different from destination start index
// sz bits from two sources are XORed and stored in Dst
__device__ __forceinline__ void xorBits(const uint32_t* __restrict__ Src0, uint32_t srcBitIdx0, uint32_t Src0size,
                                        const uint32_t* __restrict__ Src1, uint32_t srcBitIdx1, uint32_t Src1size,
                                        uint32_t* __restrict__ Dst, uint32_t bitIdxDst, uint32_t sz)
{
    uint32_t wIdx_s0  = srcBitIdx0 / WORD_LENGTH;
    uint32_t bIdx_s0  = srcBitIdx0 % WORD_LENGTH;
    uint32_t wIdx_s1  = srcBitIdx1 / WORD_LENGTH;
    uint32_t bIdx_s1  = srcBitIdx1 % WORD_LENGTH;

    uint32_t wrd0, wrd0_0, wrd0_1;
    wrd0 = wrd0_0 = Src0[wIdx_s0];

    if (bIdx_s0 > 0) {
        wrd0_1 = (wIdx_s0 + 1) < Src0size ? Src0[wIdx_s0 + 1] : 0;
        wrd0   = __funnelshift_rc(wrd0_0, wrd0_1, bIdx_s0);
        wrd0_0 = wrd0_1;
    }

    uint32_t wrd1, wrd1_0, wrd1_1;
    wrd1 = wrd1_0 = Src1[wIdx_s1];
    // passes srcBitIdx1 = 0, bIdx_s1 is always 0
    /*VCAST_DONT_INSTRUMENT_START*/
    if (bIdx_s1 > 0) {
        wrd1_1 = (wIdx_s1 + 1) < Src1size ? Src1[wIdx_s1 + 1] : 0;
        wrd1   = __funnelshift_rc(wrd1_0, wrd1_1, bIdx_s1);
        wrd1_0 = wrd1_1;
    }
    /*VCAST_DONT_INSTRUMENT_END*/

    uint32_t wIdx_d  = bitIdxDst / WORD_LENGTH;
    uint32_t bIdx_d  = bitIdxDst % WORD_LENGTH;

    if (sz == 1) {
        // we can xor a single bit faster than using the general approach below
        uint32_t bit0 = (wrd0 >> bIdx_s0) & uint32_t(1);
        uint32_t bit1 = (wrd1 >> bIdx_s1) & uint32_t(1);
        if (bit0 == bit1) {
            Dst[wIdx_d] &= ~(1 << bIdx_d);
        } else {
            Dst[wIdx_d] |= (1 << bIdx_d);
        }
        return;
    }

    constexpr uint32_t ones = 0xffffffff;
    constexpr uint32_t zeros = 0x00000000;

    int maskZeros, maskOnes;
    // depending on sz and idx, we may have to copy (head-word + mid-words + tail-words)
    // where mid-words or tail-words may or may not exist
    uint32_t wrd = wrd0 ^ wrd1;

    // update the head-word
    uint32_t numHeadBits = WORD_LENGTH - bIdx_d;
    if (numHeadBits <= sz) {
        maskZeros = __funnelshift_rc(ones, wrd, numHeadBits); // = __funnelshift_lc(ones, wrd, bIdx_d);
        maskOnes  = __funnelshift_rc(zeros, wrd, numHeadBits);
    } else {
        maskZeros = __funnelshift_rc(ones, wrd, sz);
        maskZeros = __funnelshift_rc(maskZeros, ones, (numHeadBits - sz));
        maskOnes  = __funnelshift_rc(zeros, wrd, sz);
        maskOnes  = __funnelshift_rc(maskOnes, zeros, (numHeadBits - sz));
    }
    Dst[wIdx_d] &= maskZeros; // copy zeros
    Dst[wIdx_d] |= maskOnes; // copy ones

    // if num_head_bits >= sz, we only need to update the head-word
    // otherwise, we need to update mid-words or tail-word
    if (numHeadBits < sz) {
        uint32_t remainingBits = sz - numHeadBits;
        uint32_t numMidWords   = remainingBits / WORD_LENGTH;
        uint32_t wrd_nxt;
        for (uint32_t i = 0; i < numMidWords; i++) {
            // extract wrd_next, but first need to extract wrd0 and wrd1

            /*VCAST_DONT_INSTRUMENT_START*/
            if (bIdx_s0 == 0) {
                // `wrd0` already contains the first extracted 32-bit chunk (starting at `srcBitIdx0`).
                // For mid-word i, load the next source word (+1) and then advance by i.
                wrd0 = Src0[wIdx_s0 + 1 + i];
            } else {
                wrd0_1 = (wIdx_s0 + 2 + i) < Src0size ? Src0[wIdx_s0 + 2 + i] : 0;
                wrd0 = __funnelshift_rc(wrd0_0, wrd0_1, bIdx_s0);
                wrd0_0 = wrd0_1;
            }

            if (bIdx_s1 == 0) {
                wrd1 = Src1[wIdx_s1 + 1 + i];
            } else {
                // Unaligned: build the aligned 32-bit chunk at bit offset `bIdx_s1` from the 2-word window
                // (`wrd1_0` = previous upper word, `wrd1_1` = newly loaded lower word). `+2` because the
                // initial extraction already consumed `Src1[wIdx_s1]`/`Src1[wIdx_s1 + 1]`.]
                // Extract the next aligned 32-bit chunk across (`wrd1_0`, `wrd1_1`).
                // Advance the sliding 32-bit window for the next iteration.
                wrd1_1 = (wIdx_s1 + 2 + i) < Src1size ? Src1[wIdx_s1 + 2 + i] : 0;
                wrd1 = __funnelshift_rc(wrd1_0, wrd1_1, bIdx_s1);
                wrd1_0 = wrd1_1;
            }
            /*VCAST_DONT_INSTRUMENT_END*/

            wrd_nxt = wrd0 ^ wrd1;

            // now concatenate the current word with the next word
            wrd = __funnelshift_lc(wrd, wrd_nxt, bIdx_d);
            Dst[wIdx_d + 1 + i] = wrd;
            wrd = wrd_nxt;
        }

        // check if partially resetting tail-word is needed
        uint32_t numTailBits = remainingBits % WORD_LENGTH;
        if (numTailBits > 0) {
            wrd >>= numHeadBits;
            Dst[wIdx_d + numMidWords + 1] = wrd; // copy remaining bits
        }
    }
}

//======================================================================================================================
// CRC functions

#ifdef ENABLE_DEBUG
__device__ __forceinline__ uint32_t ComputeCRC8Basic(const uint8_t* __restrict__ bytes, int nBytes, uint32_t poly)
{
    uint32_t crc = 0; // start with 0 so the first byte can be 'xor'ed

    for (int b = 0; b < nBytes; b++)
    {
        uint32_t revByte = (__brev(static_cast<uint32_t>(bytes[b]))) >> 24;
        crc ^= revByte; // xor the next input byte

        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x80) != 0)
            {
                crc = (crc << 1) ^ poly;
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
}

__device__ __forceinline__ uint32_t ComputeCRC16Basic(const uint8_t* __restrict__ bytes, int nBytes, uint32_t poly)
{
    uint32_t crc = 0; // start with 0 so first byte can be 'xored' in

    for (int b = 0; b < nBytes; b++)
    {
        uint32_t revByte = (__brev(static_cast<uint32_t>(bytes[b]))) >> 24;
        crc ^= (revByte << 8); // CRC is 16-bit

        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x8000) != 0)
            {
                crc = (crc << 1) ^ poly;
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
}

__device__ __forceinline__ void printCRC8table(uint32_t poly)
{
    for (int b = 0; b < 256; b++)
    {
        uint8_t crc = b;
        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x80) != 0)
            {
                crc = (crc << 1) ^ poly;
            }
            else
            {
                crc <<= 1;
            }
        }
        printf("%d\n", crc);
    }
}

__device__ __forceinline__ void printCRC16table(uint32_t poly)
{
    for (int b = 0; b < 256; b++)
    {
        uint16_t crc = b << 8;
        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x8000) != 0)
            {
                crc = (crc << 1) ^ poly;
            }
            else
            {
                crc <<= 1;
            }
        }
        printf("%d\n", crc);
    }
}
#endif

__device__ __forceinline__ uint8_t ComputeCRC8LUT(const uint8_t* __restrict__ bytes, int nBytes)
{
    uint8_t crc = 0; // start with 0 so the first byte can be 'xor'ed

    for (int b = 0; b < nBytes; b++)
    {
        uint8_t revByte = (__brev(static_cast<uint32_t>(bytes[b]))) >> 24;
        uint8_t pos     = crc ^ revByte; // xor the next input byte
        crc = CRC8_LUT[pos];
    }
    return crc;
}

__device__ __forceinline__ uint16_t ComputeCRC16LUT(const uint8_t* __restrict__ bytes, int nBytes)
{
    uint16_t crc = 0; // start with 0 so first byte can be 'xored' in

    for (int b = 0; b < nBytes; b++)
    {
        uint16_t revByte = (__brev(static_cast<uint32_t>(bytes[b]))) >> 24;
        uint16_t pos     = (crc >> 8) ^ revByte; // equal to ((crc ^ (revByte << 8)) >> 8)
        // shift out the MSB used for division per lookup table and xor with the remainder
        crc = (crc << 8) ^ CRC16_LUT[pos];
    }
    return crc;
}

// verify CRC
__device__ __forceinline__ bool validate_crc(const uint8_t nCrcBits, const uint32_t* __restrict__ X, const uint16_t nPayloadBits, uint32_t* __restrict__ sharedBuf)
{
    // copy input X to shared memory
    int nWords = div_round_up(static_cast<int>(nPayloadBits + nCrcBits), WORD_LENGTH);
#ifdef _DEBUG
    assert(nWords < 33);
#endif
    for(int i = 0; i < nWords; i++)
    {
        sharedBuf[i] = X[i];
    }

    // compute CRC, if no errors detected it should be 0
    uint32_t crc    = 1;
    int      nBytes = nWords * 4;
    if(nCrcBits == 11)
    {
        crc = ComputeCRC16LUT(reinterpret_cast<uint8_t*>(sharedBuf), nBytes);
    }
    else if(nCrcBits == 6)
    {
        crc = ComputeCRC8LUT(reinterpret_cast<uint8_t*>(sharedBuf), nBytes);
    }
    return (crc != 0);
}

// append received CRC bits to info bits, then verify CRC correctness
// sharedBuf: per-codeword scratch of N_MAX_WORDS words
__device__ __forceinline__ uint8_t append_and_validate_crc(const uint32_t receivedCrc, const uint8_t nCrcBits, const uint32_t* __restrict__ X, const uint16_t nPayloadBits, uint32_t* __restrict__ sharedBuf)
{
    int nWords        = div_round_up(static_cast<int>(nPayloadBits + nCrcBits), WORD_LENGTH);
#ifdef _DEBUG
    assert(nWords < 33);
#endif
    // X stores only decoded payload bits. The received CRC is tracked separately
    // in receivedCrc and appended below, so do not read the CRC word from X.
    // The caller streams X/cbEst out of an accumulator that starts at zero and
    // only ever ORs payload bits in, so the bits above nPayloadBits in the final
    // payload word are already clean.
    int nPayloadWords = div_round_up(static_cast<int>(nPayloadBits), WORD_LENGTH);
    for(int i = 0; i < nPayloadWords; i++)
    {
        sharedBuf[i] = X[i];
    }

    for(int i = nPayloadWords; i < nWords; i++)
    {
        sharedBuf[i] = 0;
    }

    //append crc bits
    uint32_t Zero = 0;
    xorBits(&receivedCrc, 0, 1, &Zero, 0, 1, sharedBuf, nPayloadBits, nCrcBits);

    // compute CRC, if no errors detected it should be 0
    uint32_t crc    = 1;
    int      nBytes = nWords * 4;
    if(nCrcBits == 11)
    {
        crc = ComputeCRC16LUT(reinterpret_cast<uint8_t*>(sharedBuf), nBytes);
    }
    else if(nCrcBits == 6)
    {
        crc = ComputeCRC8LUT(reinterpret_cast<uint8_t*>(sharedBuf), nBytes);
    }
    uint8_t crcErr = crc==0? 0 : 1;
    return crcErr;
}

__device__ __forceinline__ void updateCRCstatus(polarDecoderDynDescr_t* pDynDescr, uint8_t crcErrFlag, const uint32_t BLOCK_IDX)
{
    {
        if(crcErrFlag)
        {
            *(pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCrcStatus) = CUPHY_FAPI_CRC_FAILURE;
            if((pDynDescr->pCwPrmsGpu[BLOCK_IDX].en_CrcStatus&CUPHY_PUCCH_DET_EN) == CUPHY_PUCCH_DET_EN)
            {
                *(pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCrcStatus1) = CUPHY_FAPI_CRC_FAILURE;
            }
        }
        else
        {
            *(pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCrcStatus) = CUPHY_FAPI_CRC_PASS;
            if((pDynDescr->pCwPrmsGpu[BLOCK_IDX].en_CrcStatus&CUPHY_PUCCH_DET_EN) == CUPHY_PUCCH_DET_EN)
            {
                *(pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCrcStatus1) = CUPHY_FAPI_CRC_PASS;
            }
        }
    }
}

//======================================================================================================================

__device__ __forceinline__ void updateUciSegEstForTwoCbsInUciSeg(polarDecoderDynDescr_t* __restrict__ pDynDescr, uint16_t A_cw, uint32_t* __restrict__ cbEst, const uint32_t BLOCK_IDX)
{
    uint8_t   cbIdxWithinUciSeg = pDynDescr->pCwPrmsGpu[BLOCK_IDX].cbIdxWithinUciSeg;
    uint8_t   zeroInsertFlag    = pDynDescr->pCwPrmsGpu[BLOCK_IDX].zeroInsertFlag;
    uint32_t* pUciSegEst        = pDynDescr->pCwPrmsGpu[BLOCK_IDX].pUciSegEst;

    uint16_t nBitsUciSeg        = 2 * A_cw - zeroInsertFlag;
    uint16_t nWordsUciSeg       = (nBitsUciSeg + 31) / 32;
    uint16_t nWordsPerDecodedCb = (A_cw + 31) / 32;

    uint16_t nBitsCb0      = A_cw - zeroInsertFlag;
    uint16_t nWordsCb0     = (nBitsCb0 + 31) / 32;
    uint16_t nWordsOnlyCb1 = nWordsUciSeg - nWordsCb0;

    uint16_t nCb0BitsInLastCb0Word = nBitsCb0 % 32;
    if(nCb0BitsInLastCb0Word == 0)
    {
        nCb0BitsInLastCb0Word = 32;
    }
    uint16_t nCb1BitsInLastCb0Word = 32 - nCb0BitsInLastCb0Word;

    if(cbIdxWithinUciSeg == 0)
    {
        for(int wordIdx = 0; wordIdx < (nWordsPerDecodedCb - 1); wordIdx++)
        {
            uint32_t currentWord = cbEst[wordIdx];
            if(zeroInsertFlag)
            {
                uint32_t nextWord = cbEst[wordIdx + 1];
                currentWord       = (currentWord >> 1) | (nextWord << 31);
            }
            pUciSegEst[wordIdx] = currentWord;
        }

        if(nWordsCb0 == nWordsPerDecodedCb)
        {
            if(zeroInsertFlag == 1)
            {
                uint32_t clearWord   = 0xffffffff << nCb0BitsInLastCb0Word;
                atomicAnd(pUciSegEst + nWordsCb0 - 1, clearWord);

                uint32_t currentWord = cbEst[nWordsPerDecodedCb - 1] >> 1;
                atomicOr(pUciSegEst + nWordsCb0 - 1, currentWord);
            }else
            {
                uint32_t clearWord   = 0xffffffff << nCb0BitsInLastCb0Word;
                atomicAnd(pUciSegEst + nWordsCb0 - 1, clearWord);

                uint32_t currentWord = cbEst[nWordsPerDecodedCb - 1];
                atomicOr(pUciSegEst + nWordsCb0 - 1, currentWord);
            }
        }
    }else{
        if(nCb0BitsInLastCb0Word > 0)
        {
            uint32_t clearWord   = 0xffffffff >> nCb1BitsInLastCb0Word;
            atomicAnd(pUciSegEst + nWordsCb0 - 1, clearWord);

            uint32_t currentWord = cbEst[0] <<  nCb0BitsInLastCb0Word;
            atomicOr(pUciSegEst + nWordsCb0 - 1, currentWord);
        }

        for(int wordIdx = 0; wordIdx < (nWordsPerDecodedCb - 1); ++wordIdx)
        {
            uint32_t currentWord = cbEst[wordIdx];
            uint32_t nextWord    = cbEst[wordIdx + 1];

            currentWord                     = (currentWord >> nCb1BitsInLastCb0Word) | (nextWord << nCb0BitsInLastCb0Word);
            pUciSegEst[nWordsCb0 + wordIdx] = currentWord;
        }

        if(nWordsOnlyCb1 == nWordsPerDecodedCb)
        {
            uint32_t currentWord         = cbEst[nWordsPerDecodedCb - 1];
            currentWord                  = currentWord >> nCb1BitsInLastCb0Word;
            pUciSegEst[nWordsUciSeg - 1] = currentWord;
        }
    }
}

//======================================================================================================================

// Polar Decoder: Successive Cancellation with Compressed storage and Pruned Tree
// Decodes codeword cwIdx with one warp; shSlice is the warp's private dynamic
// shared-memory slice (see scDecoderSliceBytes()).
__device__ __forceinline__ void
singlePolarDecoder(polarDecoderDynDescr_t* pDynDescr, const uint32_t cwIdx, const uint32_t lane, bool* __restrict__ shSlice)
{
    const uint32_t BLOCK_IDX = cwIdx;

    uint16_t N_cw     = pDynDescr->pCwPrmsGpu[BLOCK_IDX].N_cw;
    uint8_t  nCrcBits = pDynDescr->pCwPrmsGpu[BLOCK_IDX].nCrcBits;
    uint16_t A_cw     = pDynDescr->pCwPrmsGpu[BLOCK_IDX].A_cw;
    uint16_t n_cw     = find_msb(N_cw);

    __half*   cwTreeLLR  = pDynDescr->cwTreeLLRsAddrs[BLOCK_IDX];
    uint32_t* cbEst      = pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCbEst;
    uint8_t&  crcErrFlag = pDynDescr->pPolCrcErrorFlags[BLOCK_IDX];
    uint8_t*  treeTypes  = pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCwTreeTypes;
    // Precomputed fast-SSC operation list (see polar_cw_tree_layout.hpp): one
    // byte per visited node in bit order; bits[3:0] = node stage,
    // bits[6:4] = node type (0 frozen, 1 info, 2 parity leaf, 4 REP, 5 SPC).
    const uint8_t* opList = &treeTypes[cuphy::polar::PolarCwTreeLayout::fssOpListOffset(1u << n_cw)];
    // crcEst is to store CRC part of the decoded message
    uint32_t crcEst = 0;
    // Decoded payload bits form a sequential stream; lane 0 accumulates them
    // in a register word and writes each cbEst word exactly once when full
    // (plus a final partial flush), so cbEst needs no zero pass and no
    // read-modify-write.
    uint32_t outAcc  = 0;
    int32_t  outFill = 0;
    int32_t  outWrd  = 0;
    // register-resident low LLR tree: lane l holds elements (2l, 2l+1)
    __half2  lowTreeLLR  = __float2half2_rn(0.f);
    // For small trees the input LLRs themselves fall inside the register
    // region (elements [treeSz, 2*treeSz) with treeSz <= 32): load them now.
    {
        const int32_t treeSz = 1 << n_cw;
        if(treeSz <= 32 && (lane >= static_cast<uint32_t>(treeSz / 2)) && (lane < static_cast<uint32_t>(treeSz)))
        {
            lowTreeLLR = reinterpret_cast<const __half2*>(cwTreeLLR)[lane];
        }
    }

#ifdef ENABLE_DEBUG
    if(lane == 0)
    {
        printf("LLRs ------------------------------------------------------------------\n");
        for(int i = 0; i < 2 * N_cw - 1; i++)
        {
            printf("%9.1f ", static_cast<float>(cwTreeLLR[i]));
            if((i + 1) % 10 == 0)
            {
                printf("\n");
            }
        }
        printf("\n\n");
    }
#endif

    // Initialize ==========================================================================
    // Length of the current sub-array in cwTreeLLR. The tree is laid out so
    // that a node of size sz occupies [sz, 2*sz), which is why sz doubles as
    // the node's start index below.
    int32_t sz;
    int32_t stage     = n_cw; // stage variable
    uint8_t type      = 10;   // decoding type used to simplify in pruned tree
    int32_t bitIdx    = -1;   // keeps track of the decoded bit index in the successive algorithm
    int32_t msgBitIdx = 0;    // keeps track of index of non-frozen decoded bits
    int32_t crcBitIdx = 0;    // keeps track of index of CRC decoded bits

    // first visited pruned node comes from the precomputed opList
    int32_t opIdx   = 0;
    uint8_t opEntry = __ldg(&opList[opIdx++]);

    // propagate input LLRs to the first visited node's stage.
    // Dispatch by sub-tree size: sz > 32 stays on the memory array (F_func);
    // sz == 32 crosses into the register tree; sz < 32 is pure register/shuffle
    // (see the "register-resident low sub-tree" note above the F_lowTree family).
    while(stage > (opEntry & 0xF))
    {
        stage--;
        sz = 1 << stage;
        if(sz > 32)
        {
            auto* in  = &cwTreeLLR[2 * sz];
            auto* out = &cwTreeLLR[sz];
            F_func(out, in, sz, lane);
            sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
#ifdef ENABLE_DEBUG
            // in/out only exist on this branch; for sz <= 32 the data is
            // register-resident (lowTreeLLR), so there is nothing to print here.
            if (lane == 0) {
                printf("F function ----------------- stage %d, size %d ------------------\n", stage, sz);
                printf("in1: ");
                for (int i = 0; i < sz; i++) {
                    printf("%6.1f ", __half2float(in[i]));
                }
                printf("\nin2: ");
                for (int i = sz; i < 2 * sz; i++) {
                    printf("%6.1f ", __half2float(in[i]));
                }
                printf("\nout: ");
                for (int i = 0; i < sz; i++) {
                    printf("%6.1f ", __half2float(out[i]));
                }
                printf("\n");
            }
#endif
        }
        else if(sz == 32)
        {
            F_lowTreeFromMem(lowTreeLLR, cwTreeLLR, lane);
        }
        else
        {
            F_lowTree(lowTreeLLR, sz, lane);
        }
    }

    // Main loop  ==========================================================================
    bool*     cs_buffer_a = shSlice;                // temporary buffer for codeword at a given stage
    bool*     cs_buffer_b = &shSlice[N_cw / 2];     // temporary buffer for codeword at a given stage
    // Per-stage codeword estimates, packed as bit words with stage sz stored at
    // bit offset sz; the byte offset is rounded up for word alignment.
    const uint32_t estByteOffset = (2u * (N_cw / 2u) + 3u) & ~3u;
    uint32_t* estW        = reinterpret_cast<uint32_t*>(&shSlice[estByteOffset]);
    // per-codeword CRC scratch behind the estimates
    uint32_t* crcBuf      = &estW[max(N_cw / WORD_LENGTH, 1)];

    while(bitIdx < (N_cw - 1))
    {
        bool* cs = cs_buffer_a;
        type     = opEntry >> 4;
        sz = 1 << stage;

        // type: 0 frozen, 1 info, 2 parity leaf, 4 REP, 5 SPC
#ifdef _DEBUG
        assert(type < 6 && type != 3);
#endif
        if(type == 0)
        {
            // set cs array for this stage to 0
            for(int32_t i = lane; i < sz; i += warpSize)
            {
                cs[i] = 0;
            }
        }
        else if(type == 4)
        {
            // REP node: decide the single information bit, codeword = repeat
            const bool b = rep_decide(cwTreeLLR, lowTreeLLR, sz, lane);
            for(int32_t i = lane; i < sz; i += warpSize)
            {
                cs[i] = b;
            }
        }
        else //if type == 1/5 for any stage or type==2 for leaf nodes
        {
            // make hard decisions based on the LLR values
            if(sz <= 32)
            {
                hardDecision_lowTree(cs, lowTreeLLR, sz, lane);
            }
            else
            {
                for(int32_t i = lane; i < sz; i += warpSize)
                {
                    bool c = __hgt(cwTreeLLR[sz + i], 0) ? 0 : 1;
                    cs[i] = c;
                }
            }
        }
        sync_barrier<POLAR_DECODER_BLOCK_SIZE>();

        if(type == 5)
        {
            // SPC node: enforce even parity on the hard decisions
            spc_fix(cs, cwTreeLLR, lowTreeLLR, sz, lane);
        }

#ifdef ENABLE_DEBUG
        if(lane == 0)
        {
            printf("\n====================================================\n");
            printf("Bit index %d, type %d, sz %d\n", bitIdx, type, sz);
            printf("LLRs -------------------------------------------------\n");
            for(int i = 0; i < 2 * N_cw - 1; i++)
            {
                printf("%9.1f ", static_cast<float>(cwTreeLLR[i]));
                if((i + 1) % 10 == 0)
                {
                    printf("\n");
                }
            }
            printf("\n");
        }
#endif

        // store message bits: an info node contributes sz of them, a REP node
        // one, an SPC node sz-1 (its u[0] is the frozen parity bit)
        if(type == 1 || type == 4 || type == 5)
        {
            uint32_t myWord;
            int32_t  bitOff;
            int32_t  mlen;
            if(type == 4)
            {
                myWord = cs[0] ? 1u : 0u; // the single information bit
                bitOff = 0;
                mlen   = 1;
            }
            else
            {
                // Recover the decoded bits with a register-resident butterfly
                // (word k of the result in lane k)
                myWord = xor_butterfly_regs(cs, stage, sz, lane);
                bitOff = (type == 5) ? 1 : 0;
                mlen   = sz - bitOff;
            }
            const int32_t  nWrds = div_round_up(sz, static_cast<int32_t>(WORD_LENGTH));
            const uint16_t K_cw  = A_cw + nCrcBits;

            if(msgBitIdx < A_cw)
            {
                const int32_t write_sz = (mlen + msgBitIdx) < A_cw ? mlen : A_cw - msgBitIdx;
                for(int32_t i = 0; i < write_sz; i += WORD_LENGTH)
                {
                    const int32_t  chunk = (write_sz - i) < static_cast<int32_t>(WORD_LENGTH) ? (write_sz - i) : static_cast<int32_t>(WORD_LENGTH);
                    const int32_t  s0    = bitOff + i;
                    const uint32_t wA    = __shfl_sync(0xFFFFFFFFu, myWord, s0 / WORD_LENGTH);
                    const uint32_t wB    = __shfl_sync(0xFFFFFFFFu, myWord, (s0 / WORD_LENGTH + 1) < nWrds ? (s0 / WORD_LENGTH + 1) : s0 / WORD_LENGTH);
                    uint32_t word        = (s0 % WORD_LENGTH) ? __funnelshift_rc(wA, wB, s0 % WORD_LENGTH) : wA;
                    if(chunk < static_cast<int32_t>(WORD_LENGTH))
                    {
                        word &= (1u << chunk) - 1;
                    }
                    if(lane == 0)
                    {
                        outAcc |= word << outFill;
                        const int32_t newFill = outFill + chunk;
                        if(newFill >= static_cast<int32_t>(WORD_LENGTH))
                        {
                            cbEst[outWrd++] = outAcc;
                            outAcc  = (outFill == 0) ? 0u : (word >> (WORD_LENGTH - outFill));
                            outFill = newFill - WORD_LENGTH;
                        }
                        else
                        {
                            outFill = newFill;
                        }
                    }
                }
                // If this node crosses the payload boundary, pack its remaining CRC bits.
                if(mlen > write_sz)
                {
                    const int32_t  s0 = bitOff + write_sz;
                    const uint32_t wA = __shfl_sync(0xFFFFFFFFu, myWord, s0 / WORD_LENGTH);
                    const uint32_t wB = __shfl_sync(0xFFFFFFFFu, myWord, (s0 / WORD_LENGTH + 1) < nWrds ? (s0 / WORD_LENGTH + 1) : s0 / WORD_LENGTH);
                    if(lane == 0)
                    {
                        const uint32_t bits = (s0 % WORD_LENGTH) ? __funnelshift_rc(wA, wB, s0 % WORD_LENGTH) : wA;
                        crcEst |= bits & ((1u << (mlen - write_sz)) - 1);
                    }
                    crcBitIdx = mlen - write_sz;
                }
            }
            else if(msgBitIdx < K_cw)
            {
                const int32_t  write_sz = (mlen + msgBitIdx) < K_cw ? mlen : K_cw - msgBitIdx;
                const uint32_t wA       = __shfl_sync(0xFFFFFFFFu, myWord, bitOff / WORD_LENGTH);
                const uint32_t wB       = __shfl_sync(0xFFFFFFFFu, myWord, (bitOff / WORD_LENGTH + 1) < nWrds ? (bitOff / WORD_LENGTH + 1) : bitOff / WORD_LENGTH);
                if(lane == 0)
                {
                    const uint32_t bits = (bitOff % WORD_LENGTH) ? __funnelshift_rc(wA, wB, bitOff % WORD_LENGTH) : wA;
                    crcEst |= (bits & ((1u << write_sz) - 1)) << crcBitIdx;
                }
                crcBitIdx += write_sz;
            }
            msgBitIdx += mlen; // update message idx for every lane
        }

        // update bit index:
        bitIdx += sz;
        if(bitIdx == (N_cw - 1)) break;

        // prefetch the next visited node's opList entry
        opEntry = __ldg(&opList[opIdx++]);

        // use H function to combine codeword estimates all the way up to stage_idx
        int32_t stage_idx   = POLAR_DEPTH[bitIdx];
        int32_t temp_H_cntr = 0;
        while(stage < stage_idx)
        {
            sz = 1 << stage;
            // in0 is read from the packed estimate buffer at bit offset sz.
            // in1 and cs keep switching. Initially, in1 is read from cs_buffer_b
            auto* in_1 = temp_H_cntr % 2 ? cs_buffer_b : cs_buffer_a;
            cs         = temp_H_cntr % 2 ? cs_buffer_a : cs_buffer_b;
            H_func(cs, estW, in_1, sz, lane);
            sync_barrier<POLAR_DECODER_BLOCK_SIZE>();

            temp_H_cntr++;
            stage++;
        }

        // use G function with new codeword for stage_idx to update LLRs of the sibling branch
        sz = 1 << stage;
        if(sz > 32)
        {
            G_func(&cwTreeLLR[sz], &cwTreeLLR[2 * sz], cs, sz, lane);
            sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
        }
        else if(sz == 32)
        {
            G_lowTreeFromMem(lowTreeLLR, cwTreeLLR, cs, lane);
        }
        else
        {
            G_lowTree(lowTreeLLR, cs, sz, lane);
        }

        // use F function to propagate updated LLRs down to the next visited node's stage
        while(stage > (opEntry & 0xF))
        {
            stage--;
            sz = 1 << stage;
            if(sz > 32)
            {
                auto* in  = &cwTreeLLR[2 * sz];
                auto* out = &cwTreeLLR[sz];
                F_func(out, in, sz, lane);
                sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
            }
            else if(sz == 32)
            {
                F_lowTreeFromMem(lowTreeLLR, cwTreeLLR, lane);
            }
            else
            {
                F_lowTree(lowTreeLLR, sz, lane);
            }
        }

        // store stage_idx codeword estimate
        sz = 1 << stage_idx;
        store_est_bits(estW, cs, sz, lane);
        sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
    }
    //======================================================================================
    // flush the partial tail word of the payload stream
    if((lane == 0) && (outFill > 0))
    {
        cbEst[outWrd] = outAcc;
    }

    // now compute CRC from info bits and compare with crcEst
    // ToDo currently only look at CRC of first CB. Need to use uint32_t CRC along with atomic operations. Requires API and cuPHY controller changes
    if((lane == 0) && (pDynDescr->pCwPrmsGpu[BLOCK_IDX].cbIdxWithinUciSeg == 0))
    {
        crcErrFlag = append_and_validate_crc(crcEst, nCrcBits, cbEst, A_cw, crcBuf);
    }

    //======================================================================================
    // If parentUciSeg composed of two codeblocks place cbEst carefully into uciSegEst

    if((lane == 0) && (pDynDescr->pCwPrmsGpu[BLOCK_IDX].nCbsInUciSeg == 2) )
    {
       updateUciSegEstForTwoCbsInUciSeg(pDynDescr, A_cw, cbEst, BLOCK_IDX);
    }

#ifdef ENABLE_DEBUG
    if((0 == cwIdx) && (0 == lane))
    {
        printf("\n polar codeword %d has the following parameters: \n N_cw = %d,\n nCrcBits = %d,\n A_cw = %d \n",
               BLOCK_IDX,
               N_cw,
               nCrcBits,
               A_cw);
    }
#endif
}

__global__ void
polarDecoderKernel(polarDecoderDynDescr_t* pDynDescr)
{
    // One warp per codeword; the host packs one warp per block for small
    // batches and POLAR_SC_WARPS_PER_BLOCK warps per block for large batches.
    // The decoder is warp-autonomous (no block-wide barriers), so warps whose
    // codeword index is out of range or flagged may simply return.
    const uint32_t warpId = threadIdx.x / CUDA_WARP_SIZE;
    const uint32_t lane   = threadIdx.x % CUDA_WARP_SIZE;
    const uint32_t cwIdx  = blockIdx.x * (blockDim.x / CUDA_WARP_SIZE) + warpId;
    if(cwIdx >= pDynDescr->nPolCws)
    {
        return;
    }

    uint8_t exitFlag = pDynDescr->pCwPrmsGpu[cwIdx].exitFlag;
    if(exitFlag == 1)
    {
        return;
    }

    __shared__ extern bool sh_buff[];
    bool* shSlice = &sh_buff[warpId * pDynDescr->scSharedSliceBytes];

    singlePolarDecoder(pDynDescr, cwIdx, lane, shSlice);
    // Update Detection (CRC) Status
    // ToDo currently only look at CRC of first CB. Need to use uint32_t CRC along with atomic operations. Requires API and cuPHY controller changes
    if((lane == 0) && (pDynDescr->pCwPrmsGpu[cwIdx].cbIdxWithinUciSeg == 0))
    {
        updateCRCstatus(pDynDescr, pDynDescr->pPolCrcErrorFlags[cwIdx], cwIdx);
    }
}

//**********************************************************************************************//
//                                                                                              //
//                      List Polar Decoder & Utility Functions                                  //
//                                                                                              //
//**********************************************************************************************//

// merge sort functions ==========================================================================

//ToDo add epsilon for float comparison?
template<uint SORT_DIR>
__forceinline__ __device__ uint binarySearchInclusive(__half val, const __half* __restrict__ data, uint L, uint stride) {
    // loop starts with stride = 1
    /*VCAST_DONT_INSTRUMENT_START*/
    if (L == 0) {
        return 0;
    }
    /*VCAST_DONT_INSTRUMENT_END*/

    uint pos = 0;

    for (; stride > 0; stride >>= 1) {
        uint newPos = umin(pos + stride, L);

        if ((SORT_DIR && __hle(data[newPos - 1], val)) || (!SORT_DIR && __hge(data[newPos - 1], val))) {
            pos = newPos;
        }
    }

    return pos;
}

template<uint SORT_DIR>
__forceinline__ __device__ uint binarySearchExclusive(__half val, const __half* __restrict__ data, uint L, uint stride) {
    // loop starts with stride = 1
    /*VCAST_DONT_INSTRUMENT_START*/
    if (L == 0) {
        return 0;
    }
    /*VCAST_DONT_INSTRUMENT_END*/

    uint pos = 0;

    for (; stride > 0; stride >>= 1) {
        uint newPos = umin(pos + stride, L);

        if ((SORT_DIR && __hlt(data[newPos - 1], val)) || (!SORT_DIR && __hgt(data[newPos - 1], val))) {
            pos = newPos;
        }
    }

    return pos;
}

// block-level merge sort (binary search-based)
template<uint32_t LIST_SZ, uint32_t ITEM_PER_THRD, uint32_t SORT_DIR>
__forceinline__ __device__ void mergeSortShared(__half* __restrict__ key, uint32_t* __restrict__ val, const thread_group& grp, __half* __restrict__ s_key) {
    constexpr uint32_t arrayLength = LIST_SZ * ITEM_PER_THRD;
    //__shared__ __half s_key[arrayLength];
    __shared__ uint32_t  s_val[arrayLength];

    // number of threads per group should be equal to LIST_SZ * ITEM_PER_THRD / 2
#ifdef _DEBUG
    assert(grp.size() == arrayLength/2);
#endif


    int tid = grp.thread_rank();
    if (tid < LIST_SZ)
    {
#pragma unroll (ITEM_PER_THRD)
        for (int i = 0; i < ITEM_PER_THRD; i++) {
            s_key[ITEM_PER_THRD * tid + i] = key[i];
            s_val[ITEM_PER_THRD * tid + i] = val[i];
        }
    }
    grp.sync();

    for (uint stride = 1; stride < arrayLength; stride <<= 1) {
        uint    lPos    = grp.thread_rank() & (stride - 1);
        __half* baseKey = s_key + 2 * (grp.thread_rank() - lPos);
        uint*   baseVal = s_val + 2 * (grp.thread_rank() - lPos);

        grp.sync();
        __half keyA = baseKey[lPos + 0];
        uint   valA = baseVal[lPos + 0];
        __half keyB = baseKey[lPos + stride];
        uint   valB = baseVal[lPos + stride];
        uint   posA = binarySearchExclusive<SORT_DIR>(keyA, baseKey + stride, stride, stride) + lPos;
        uint   posB = binarySearchInclusive<SORT_DIR>(keyB, baseKey + 0, stride, stride) + lPos;

        grp.sync();

        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }
    grp.sync();

    if (tid < LIST_SZ)
    {
#pragma unroll (ITEM_PER_THRD)
        for (int i = 0; i < ITEM_PER_THRD; i++) {
            key[i] = s_key[ITEM_PER_THRD * grp.thread_rank() + i];
            val[i] = s_val[ITEM_PER_THRD * grp.thread_rank() + i];
        }
    }
}

// =================================================================================================================================

// For each path function decodes R0 codeword and updates path metric
template<uint32_t TILE_SIZE>
__device__ __forceinline__ void
R0_decoder(int16_t* __restrict__ pathPrime, __half* __restrict__ pathMetric, uint32_t* __restrict__ csBitWords, const int stage, const int num_path,
           const __half* __restrict__ cwLLR, const int N, const thread_block_tile<TILE_SIZE>& tile)
{
    int tile_rank = tile.meta_group_rank();
    if(tile_rank < num_path)
    {
        pathPrime[tile_rank] = tile_rank;
        int sz = 1 << stage;
        auto LLRs = &cwLLR[2 * N * tile_rank + sz];
        resetBits(csBitWords, tile_rank * (N + BCO * WORD_LENGTH), sz);   // + BCO * WORD_LENGTH offset is to reduce bank conflicts when accessed by different tiles
            for(int i = tile.thread_rank(); i < sz; i += tile.size())
            {
                // penalize path metric if 0 bit "unexpected"
                if(__hlt(LLRs[i], 0))
                {
                    atomicAdd(&pathMetric[tile_rank], LLRs[i]);
            }
        }
    }
}

template<int32_t LIST_SZ = 8>
__device__ __forceinline__ void
R1_S0_decoder(int16_t* __restrict__ pathPrime, __half* __restrict__ pathMetric, uint32_t* __restrict__ csBitWords,
              int &num_path, const __half* __restrict__ cwLLR, const int N, const thread_group& grp) {
    int      tid               = grp.thread_rank();
    __half   childrenPm[2]    = {-HFLT_MAX, -HFLT_MAX};
    uint32_t childrenIds[2];

    if (tid < num_path) {
        __half LLR               = cwLLR[(2 * N) * tid + 1];
        bool   expected_estimate = __hgt(LLR, 0) ? 0 : 1; // make hard decision based on LLR
        if (expected_estimate == 0) {
            childrenPm[0] = pathMetric[tid];
            childrenPm[1] = pathMetric[tid] - __habs(LLR);
        } else {
            childrenPm[0] = pathMetric[tid] - __habs(LLR);
            childrenPm[1] = pathMetric[tid];
        }
    }

    // shared memory used in transpose operation and merge-sort function
    __shared__ __half temp[2 * LIST_SZ];
    // first we need to rearrange registers as following
    // [t0,0 t0,1]          [t0,0 t1,0]
    // [t1,0 t1,1]   -->    [t0,1 t1,1]
    //--------------------------------------------------------------
    if(tid < num_path)
    {
        temp[tid]            = childrenPm[0];
        temp[tid + num_path] = childrenPm[1];
    } else if (tid < LIST_SZ){
        temp[2 * tid]     = childrenPm[0];
        temp[2 * tid + 1] = childrenPm[1];
    }
    grp.sync();

    // if number of paths is less than list size, keep both children and update number of the paths
    if (num_path < LIST_SZ) {
        if (tid < num_path) {
            setResetSingleBit(csBitWords, tid * (N + BCO * WORD_LENGTH), 0);  // + BCO * WORD_LENGTH offset is to reduce bank conflicts when accessed by different tiles
            setResetSingleBit(csBitWords, (tid + num_path) * (N + BCO * WORD_LENGTH), 1);
            pathPrime[tid] = tid;
            pathPrime[tid + num_path] = tid;
            pathMetric[2 * tid] = temp[2 * tid];
            pathMetric[2 * tid + 1] = temp[2 * tid + 1];
        }
        num_path *= 2;
        grp.sync();
    } else {
        // keep the best LIST_SZ children: sort and select top L
        // before sorting, we need to rearrange registers as following
        // [t0,0 t0,1]          [t0,0 t1,0]
        // [t1,0 t1,1]   -->    [t0,1 t1,1]
        //--------------------------------------------------------------
        childrenIds[0] = 2 * tid;
        childrenIds[1] = 2 * tid + 1;

        if (tid < LIST_SZ) {
            childrenPm[0] = temp[2 * tid];
            childrenPm[1] = temp[2 * tid + 1];
        }
        grp.sync();

        auto tile = tiled_partition<LIST_SZ>(this_thread_block());
        if (tile.meta_group_rank() == 0) {
            mergeSortShared<LIST_SZ, 2, 0>(childrenPm, childrenIds, tile, temp);
        }

        if (LIST_SZ >= 2) {
            if (tid < LIST_SZ / 2) {
                pathPrime[2 * tid] = childrenIds[0] % LIST_SZ;
                pathPrime[2 * tid + 1] = childrenIds[1] % LIST_SZ;
                setResetSingleBit(csBitWords, (2 * tid) * (N + BCO * WORD_LENGTH), childrenIds[0] / LIST_SZ);
                setResetSingleBit(csBitWords, (2 * tid + 1) * (N + BCO * WORD_LENGTH), childrenIds[1] / LIST_SZ);
                pathMetric[2 * tid] = childrenPm[0];
                pathMetric[2 * tid + 1] = childrenPm[1];
            }
        } else {
            // When nPolLists == 1, cuPHY selects the non-list kernel (polarDecoderKernel)
            /*VCAST_DONT_INSTRUMENT_START*/
            if (tid == 0) {
                pathPrime[0] = childrenIds[0] % LIST_SZ;
                setResetSingleBit(csBitWords, 0, childrenIds[0] / LIST_SZ);
                pathMetric[0] = childrenPm[0];
            }
            /*VCAST_DONT_INSTRUMENT_END*/
        }
    }
}

// Function decodes R1 codeword:
// 1) Computes ML solution by slicing LLRs
// 2) Compute four "near ML" children by flipping the two least reliable
// 3) compute path metric for each child
// 4) If nChildren > L, the L best are kept
template<int32_t LIST_SZ = 8>
__device__ __forceinline__ void
R1_decoder(int16_t* __restrict__ pathPrime, __half* __restrict__ pathMetric, uint32_t* __restrict__ csBitWords, bool* __restrict__ c0_tmp,
           int & numPath, const int stage, const __half* __restrict__ cwLLR, const int N, const thread_group& grp)
{
    int      tid           = grp.thread_rank();
    int      sz            = 1 << stage;
    __half   childrenPm[4] = {-HFLT_MAX, -HFLT_MAX, -HFLT_MAX, -HFLT_MAX};
    uint32_t childrenIds[4];

    const auto& p = tid;
    if (p < numPath) {
        auto LLRs = &cwLLR[(2 * N) * tid + sz];
        // total assumed length of c0_tmp array = 4 * N/2 * L
        bool *c0_0 = &c0_tmp[sz * p];
        bool *c0_1 = &c0_0[LIST_SZ * sz];
        bool *c0_2 = &c0_1[LIST_SZ * sz];
        bool *c0_3 = &c0_2[LIST_SZ * sz];
        // find the two least reliable bits
        __half lrb_LLR0 = HFLT_MAX;
        __half lrb_LLR1 = HFLT_MAX;
        int    lrb_idx0 = 0;
        int    lrb_idx1 = 0;

        for(int i = 0; i < sz; i++)
        {
            auto LLR_abs = __habs(LLRs[i]);
            if(__hlt(LLR_abs, lrb_LLR0))
            {
                lrb_LLR1 = lrb_LLR0;
                lrb_idx1 = lrb_idx0;
                lrb_LLR0 = LLR_abs;
                lrb_idx0 = i;
            }
            else if(__hlt(LLR_abs, lrb_LLR1))
            {
                lrb_LLR1 = LLR_abs;
                lrb_idx1 = i;
            }
            bool est = __hgt(LLRs[i], 0) ? 0 : 1;
            c0_0[i]  = est;
            c0_1[i]  = est;
            c0_2[i]  = est;
            c0_3[i]  = est;
        }

        // compute the bit flips of the two least reliable bits
        bool f0 = __hgt(LLRs[lrb_idx0], 0) ? 1 : 0;
        bool f1 = __hgt(LLRs[lrb_idx1], 0) ? 1 : 0;

        // first candidate is ML solution
        childrenPm[0] = pathMetric[p];
        /*c0_0 has been already updated in the for loop above*/

        // second candidate flips 1st lrb of ML solution
        childrenPm[1] = pathMetric[p] - lrb_LLR0;
        c0_1[lrb_idx0] = f0;

        // third candidate flips 2nd lrb of ML solution
        childrenPm[2] = pathMetric[p] - lrb_LLR1;
        c0_2[lrb_idx1] = f1;

        // fourth candidate flips 1st and 2nd lrbs of ML solution
        childrenPm[3] = pathMetric[p] - lrb_LLR0 - lrb_LLR1;
        c0_3[lrb_idx0] = f0;
        c0_3[lrb_idx1] = f1;
    }

    // shared memory used in transpose operation and merge-sort function
    __shared__ __half temp[4 * LIST_SZ];
    // first we need to rearrange registers as following
    // [t0,0 t0,1 t0,2 t0,3]          [t0,0 t1,0 t2,0 t3,0]
    // [t1,0 t1,1 t1,2 t1,3]   -->    [t0,1 t1,1 t2,1 t3,1]
    // [t2,0 t2,1 t2,2 t2,3]          [t0,2 t1,2 t2,2 t3,2]
    // [t3,0 t3,1 t3,2 t3,3]          [t0,3 t1,3 t2,3 t3,3]
    //--------------------------------------------------------------
    if(tid < numPath)
    {
        temp[tid]               = childrenPm[0];
        temp[tid + numPath]     = childrenPm[1];
        temp[tid + 2 * numPath] = childrenPm[2];
        temp[tid + 3 * numPath] = childrenPm[3];
    } else if (tid < LIST_SZ){
        temp[4 * tid]     = childrenPm[0];
        temp[4 * tid + 1] = childrenPm[1];
        temp[4 * tid + 2] = childrenPm[2];
        temp[4 * tid + 3] = childrenPm[3];
    }
    grp.sync();

    // if number of paths is less than list size, keep all 4 children and update number of the paths
    if (4 * numPath <= LIST_SZ) {
        if (p < numPath) {
            pathPrime[p] = p;
            pathPrime[p + numPath] = p;
            pathPrime[p + 2 * numPath] = p;
            pathPrime[p + 3 * numPath] = p;
            // store the transposed data
            pathMetric[4 * p]     = temp[4 * tid];     //children_pm[0];
            pathMetric[4 * p + 1] = temp[4 * tid + 1]; //children_pm[1];
            pathMetric[4 * p + 2] = temp[4 * tid + 2]; //children_pm[2];
            pathMetric[4 * p + 3] = temp[4 * tid + 3]; //children_pm[3];

            for (int i = 0; i < sz; i++) {
                bool* tmp = &c0_tmp[sz * p];
                setResetSingleBit(csBitWords, p * (N + BCO * WORD_LENGTH) + i, tmp[i]); // + BCO * WORD_LENGTH offset is to reduce bank conflicts when accessed by different tiles
                tmp = &tmp[LIST_SZ * sz];
                setResetSingleBit(csBitWords, (2 + p) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                tmp = &tmp[LIST_SZ * sz];
                setResetSingleBit(csBitWords, (4 + p) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                tmp = &tmp[LIST_SZ * sz];
                setResetSingleBit(csBitWords, (6 + p) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
            }
        }
        numPath *= 4;
        grp.sync();
    } else {
        // keep the best LIST_SZ children: sort and select top LIST_SZ
        // before sorting, we need to rearrange registers as following
        // [t0,0 t0,1 t0,2 t0,3]          [t0,0 t1,0 t2,0 t3,0]
        // [t1,0 t1,1 t1,2 t1,3]   -->    [t0,1 t1,1 t2,1 t3,1]
        // [t2,0 t2,1 t2,2 t2,3]          [t0,2 t1,2 t2,2 t3,2]
        // [t3,0 t3,1 t3,2 t3,3]          [t0,3 t1,3 t2,3 t3,3]
        //--------------------------------------------------------------
        childrenIds[0] = 4 * tid;
        childrenIds[1] = 4 * tid + 1;
        childrenIds[2] = 4 * tid + 2;
        childrenIds[3] = 4 * tid + 3;

        if (tid < LIST_SZ) {
            childrenPm[0] = temp[4 * tid];
            childrenPm[1] = temp[4 * tid + 1];
            childrenPm[2] = temp[4 * tid + 2];
            childrenPm[3] = temp[4 * tid + 3];
        }
        grp.sync();

        auto tile = tiled_partition<LIST_SZ * 2>(this_thread_block());
        if (tile.meta_group_rank() == 0) {
            mergeSortShared<LIST_SZ, 4, 0>(childrenPm, childrenIds, tile, temp);
        }

        if (LIST_SZ >= 4) {
            if (tid < LIST_SZ / 4) {
                pathPrime[4 * tid]     = childrenIds[0] % LIST_SZ;
                pathPrime[4 * tid + 1] = childrenIds[1] % LIST_SZ;
                pathPrime[4 * tid + 2] = childrenIds[2] % LIST_SZ;
                pathPrime[4 * tid + 3] = childrenIds[3] % LIST_SZ;

                pathMetric[4 * tid]     = childrenPm[0];
                pathMetric[4 * tid + 1] = childrenPm[1];
                pathMetric[4 * tid + 2] = childrenPm[2];
                pathMetric[4 * tid + 3] = childrenPm[3];

                int cxa = childrenIds[0] / numPath;
                int cxb = childrenIds[1] / numPath;
                int cxc = childrenIds[2] / numPath;
                int cxd = childrenIds[3] / numPath;
                int cya = childrenIds[0] & (numPath - 1); //% numPath;
                int cyb = childrenIds[1] & (numPath - 1); //% numPath;
                int cyc = childrenIds[2] & (numPath - 1); //% numPath;
                int cyd = childrenIds[3] & (numPath - 1); //% numPath;

                for (int i = 0; i < sz; i++)
                    {
                        bool* tmp = &c0_tmp[(cxa * LIST_SZ + cya) * sz];
                        setResetSingleBit(csBitWords, 4 * p * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                        tmp = &c0_tmp[(cxb * LIST_SZ + cyb) * sz];
                        setResetSingleBit(csBitWords, (4 * p + 1) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                        tmp = &c0_tmp[(cxc * LIST_SZ + cyc) * sz];
                        setResetSingleBit(csBitWords, (4 * p + 2) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                        tmp = &c0_tmp[(cxd * LIST_SZ + cyd) * sz];
                        setResetSingleBit(csBitWords, (4 * p + 3) * (N + BCO * WORD_LENGTH) + i, tmp[i]);
                }
            }
        } else if (LIST_SZ == 2) {
            if(tid == 0)
            {
                pathPrime[0] = childrenIds[0] % LIST_SZ;
                pathPrime[1] = childrenIds[1] % LIST_SZ;

                pathMetric[0] = childrenPm[0];
                pathMetric[1] = childrenPm[1];

                for(int i = 0; i < sz; i++)
                {
                    bool* tmp = &c0_tmp[sz * childrenIds[0]];
                    setResetSingleBit(csBitWords, i, tmp[i]);
                    tmp = &c0_tmp[sz * childrenIds[1]];
                    setResetSingleBit(csBitWords, (N + BCO * WORD_LENGTH) + i, tmp[i]);
                }
            }
        // When nPolLists == 1, cuPHY selects the non-list kernel (polarDecoderKernel)
        /*VCAST_DONT_INSTRUMENT_START*/
        } else if (LIST_SZ == 1) {
            if(tid == 0)
            {
                pathPrime[0]  = childrenIds[0] % LIST_SZ;
                pathMetric[0] = childrenPm[0];
                for(int i = 0; i < sz; i++)
                {
                    bool* tmp = &c0_tmp[sz * childrenIds[0]];
                    setResetSingleBit(csBitWords, i, tmp[i]);
                }
            }
        }
        /*VCAST_DONT_INSTRUMENT_END*/

        numPath = LIST_SZ;
    }
}

// XOR Butterfly algorithm =============================================================================================

// This is a special case of xor butterfly where sz = 32
__device__ __forceinline__ void xor32_butterfly(bool8* __restrict__ outBits, int outBitsIdx0, uint32_t& sh_wrd)
{
    uint32_t wrd = sh_wrd;
    // stage = 5
    uint32_t upperHalfShifted   = (wrd & 0xFFFF0000) >> 16;
    wrd                         = wrd ^ upperHalfShifted;
    // stage = 4
    upperHalfShifted   = (wrd & 0xFF00FF00) >> 8;
    wrd                = wrd ^ upperHalfShifted;
    // stage = 3
    upperHalfShifted   = (wrd & 0xF0F0F0F0) >> 4;
    wrd                = wrd ^ upperHalfShifted;
    // stage = 2
    upperHalfShifted   = (wrd & 0xCCCCCCCC) >> 2;
    wrd                = wrd ^ upperHalfShifted;
    // stage = 1
    upperHalfShifted   = (wrd & 0xAAAAAAAA) >> 1;
    wrd                = wrd ^ upperHalfShifted;

    // store decoded message back to msg
    for(int i = 0; i < 32; i += 8)
    {
        outBits[(i + outBitsIdx0) / 8].u64 = isBitSet8(wrd, i);
    }

    sh_wrd = wrd;
}

// This is a special case of xor butterfly where sz < 32
__device__ __forceinline__ void xor2to16_butterfly(bool* __restrict__ outBits, int outBitsIdx0, uint32_t& sh_wrd, const int sz, const thread_group& grp)
{
    uint32_t upperHalfShifted;

    uint32_t wrd = sh_wrd;

    if(sz == 16)
    {
        // stage = 4
        upperHalfShifted   = (wrd & 0xFF00FF00) >> 8;
        wrd                = wrd ^ upperHalfShifted;
    }
    if(sz >= 8)
    {
        // stage = 3
        upperHalfShifted   = (wrd & 0xF0F0F0F0) >> 4;
        wrd                = wrd ^ upperHalfShifted;
    }
    if(sz >= 4)
    {
        // stage = 2
        upperHalfShifted   = (wrd & 0xCCCCCCCC) >> 2;
        wrd                = wrd ^ upperHalfShifted;
    }
    // stage = 1
    upperHalfShifted   = (wrd & 0xAAAAAAAA) >> 1;
    wrd                = wrd ^ upperHalfShifted;

    // store decoded message to gmem
    for(int i = 0; i < sz; i++)
    {
        outBits[i + outBitsIdx0] = isBitSet(wrd, i);
    }

    sh_wrd = wrd;
}

// General xor-butterfly
__device__ __forceinline__ void xor_butterfly3(bool* __restrict__ outBits, const uint32_t* __restrict__ inWords, uint32_t inWordsSize, int bitIdx, uint32_t* __restrict__ shTempWordBuf, const int stage, const int sz, const thread_group& grp)
{
    int nWrds = div_round_up(sz, WORD_LENGTH);

    // copy words to shared mem
    for (int i = grp.thread_rank(); i < nWrds; i += grp.size()) {
        shTempWordBuf[i] = getWord(inWords, inWordsSize, bitIdx + i * WORD_LENGTH);
    }
    grp.sync();

    if (stage < 5) {
        for (int i = grp.thread_rank(); i < nWrds; i += grp.size()) {
            xor2to16_butterfly(outBits, i * WORD_LENGTH, shTempWordBuf[i], sz, grp);
        }
    } else {
        // run the following update of shTempWordBuf single threaded to avoid RAW hazard
        if (grp.thread_rank()==0)
        {
            for (int j = stage; j > 5; j--) {
                int jump_sz = 1 << (j - 6);
                int mask = (jump_sz << 1) - 1;
                for (int i = 0; i < nWrds; i++) {
                    //if ((i % (2 * jump_sz)) < jump_sz) {
                    if ((i & mask) < jump_sz) {
                        shTempWordBuf[i] = shTempWordBuf[i] ^ shTempWordBuf[i + jump_sz];
                    }
                }
            }
        }
        grp.sync();

        bool8* outBits8 = reinterpret_cast<bool8 *>(outBits);
        // once reached to stage 5 (where size of bits is 32), use xor32_butterfly instead
        for (int i = grp.thread_rank(); i < nWrds; i += grp.size()) {
            xor32_butterfly(outBits8, i * WORD_LENGTH, shTempWordBuf[i]);
        }
        grp.sync();
    }
}

//======================================================================================================================

// get the primary path indices for a given stage and path
template<int32_t LIST_SZ = 8>
__device__ __forceinline__ void
get_p_prime(int16_t* __restrict__ pathPrime, const int16_t* __restrict__ llPointers, const int numStages, const int stage, const thread_group& grp)
{
    __shared__ int16_t temp[LIST_SZ];
    int tid = grp.thread_rank();
    if (tid < LIST_SZ) {
        temp[tid] = pathPrime[tid];
    }
    grp.sync();

    if (tid < LIST_SZ) {
        pathPrime[tid] = llPointers[stage + numStages * temp[tid]];
    }
    grp.sync();
}


// ======================== LIST POLAR DECODER =================================================

// F function: implementation of box-plus (min-sum)
// combine 2 arrays of length sz/2 and output an array of length sz/2

__device__ __forceinline__ void F_func1(__half* __restrict__ llrOut, __half* __restrict__ llrIn, int sz, const thread_group& grp)
{
    __half*a = llrIn;           // first half input
    __half*b = &llrIn[sz];      // second half input
    for (int i = grp.thread_rank(); i < sz; i += grp.size()) {
#if __CUDA_ARCH__ >= 800
        __half minAbs = __hmin(__habs(a[i]), __habs(b[i]));
#else
        __half minAbs = __hlt(__habs(a[i]), __habs(b[i])) ? __habs(a[i]) : __habs(b[i]);
#endif
        llrOut[i]    = __hmul(signof(a[i], b[i]), minAbs);
    }
    grp.sync();
}

__device__ __forceinline__ void F_func4(half2x2_u64* __restrict__ llrOut, half2x2_u64* __restrict__ llrIn, int sz, const thread_group& grp)
{
    half2x2_u64* a = llrIn;           // first half input
    half2x2_u64* b = &llrIn[sz];      // second half input
    half2x2_u64 ai, bi, ci, minAbs;

    for (int i = grp.thread_rank(); i < sz; i += grp.size()) {

        // read 64-bit
        ai.u64 = a[i].u64;
        bi.u64 = b[i].u64;

#if __CUDA_ARCH__ >= 800
        minAbs.hf2.x = __hmin2(__habs2(ai.hf2.x), __habs2(bi.hf2.x));
        minAbs.hf2.y = __hmin2(__habs2(ai.hf2.y), __habs2(bi.hf2.y));
#else
        half2x2_u64 aAbs, bAbs;
        aAbs.hf2.x = __habs2(ai.hf2.x);
        aAbs.hf2.y = __habs2(ai.hf2.y);
        bAbs.hf2.x = __habs2(bi.hf2.x);
        bAbs.hf2.y = __habs2(bi.hf2.y);

        minAbs.hf2.x.x = __hlt(aAbs.hf2.x.x, bAbs.hf2.x.x) ? aAbs.hf2.x.x : bAbs.hf2.x.x;
        minAbs.hf2.x.y = __hlt(aAbs.hf2.x.y, bAbs.hf2.x.y) ? aAbs.hf2.x.y : bAbs.hf2.x.y;
        minAbs.hf2.y.x = __hlt(aAbs.hf2.y.x, bAbs.hf2.y.x) ? aAbs.hf2.y.x : bAbs.hf2.y.x;
        minAbs.hf2.y.y = __hlt(aAbs.hf2.y.y, bAbs.hf2.y.y) ? aAbs.hf2.y.y : bAbs.hf2.y.y;
#endif
        ci.hf2.x = __hmul2(signof2(ai.hf2.x, bi.hf2.x), minAbs.hf2.x);
        ci.hf2.y = __hmul2(signof2(ai.hf2.y, bi.hf2.y), minAbs.hf2.y);

        // store 64-bit
        llrOut[i].u64 = ci.u64;
    }
    grp.sync();
}

__device__ __forceinline__ void F_func(__half* __restrict__ llrOut, __half* __restrict__ llrIn, int sz, const thread_group& grp)
{
    if(sz < 4) {
        F_func1(llrOut, llrIn, sz, grp);
    } else {
        half2x2_u64* llrOut4 = reinterpret_cast<half2x2_u64*>(llrOut);
        half2x2_u64* llrIn4  = reinterpret_cast<half2x2_u64*>(llrIn);
        F_func4(llrOut4, llrIn4, sz / 4, grp);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------
// G function: implementation of repetition likelihood
// combine 2 arrays of length sz/2 based on bit array Est0 and output an array of length sz/2
// examine different variants impact on performance

__device__ __forceinline__ void G_func1(__half* __restrict__ llrOut, __half* __restrict__ llrIn, uint32_t est, int sz, const thread_group& grp) {
    __half *a = llrIn;            // first half input
    __half *b = &llrIn[sz];       // second half input

    for (int i = grp.thread_rank(); i < sz; i += grp.size()) {
        const __half u = isBitSet(est, i) ? __float2half(-1.f) : __float2half(1.f); // u is +1 or -1
        llrOut[i] = __hfma(u, a[i], b[i]);
    }
    grp.sync();
}

__device__ __forceinline__ void G_func4(half2x2_u64* __restrict__ llrOut, half2x2_u64* __restrict__ llrIn, const uint32_t* __restrict__ estBits, int sz, const thread_group& grp) {
    half2x2_u64 *a = llrIn;          // first half input
    half2x2_u64 *b = &llrIn[sz];     // second half input
    half2x2_u64 ai, bi, ci, ui;

    for (int i = grp.thread_rank(); i < sz; i += grp.size())
    {
        int32_t  wIdx  = (4 * i) / WORD_LENGTH;
        int32_t  bIdx  = (4 * i) % WORD_LENGTH;
        int32_t est   = estBits[wIdx];

        ui.hf2.x.x = static_cast<__half>(1 - 2 * ((est >> bIdx++) & int32_t(1)));
        ui.hf2.x.y = static_cast<__half>(1 - 2 * ((est >> bIdx++) & int32_t(1)));
        ui.hf2.y.x = static_cast<__half>(1 - 2 * ((est >> bIdx++) & int32_t(1)));
        ui.hf2.y.y = static_cast<__half>(1 - 2 * ((est >> bIdx) & int32_t(1)));

        // read 64-bit
        ai.u64 = a[i].u64;
        bi.u64 = b[i].u64;

        ci.hf2.x = __hfma2(ui.hf2.x, ai.hf2.x, bi.hf2.x);
        ci.hf2.y = __hfma2(ui.hf2.y, ai.hf2.y, bi.hf2.y);

        // store 64-bit
        llrOut[i].u64 = ci.u64;
    }

    grp.sync();
}

__device__ __forceinline__ void G_func(__half* __restrict__ llrOut, __half* __restrict__ llrIn, const uint32_t* __restrict__ estBits, int sz, const thread_group& grp)
{
    if(sz < 4) {
        G_func1(llrOut, llrIn, estBits[0], sz, grp);
    } else {
        half2x2_u64* llrOut4 = reinterpret_cast<half2x2_u64*>(llrOut);
        half2x2_u64* llrIn4  = reinterpret_cast<half2x2_u64*>(llrIn);
        G_func4(llrOut4, llrIn4, estBits, sz / 4, grp);
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------

// H function: implementation of combining codewords in polar code
// unlike F and G, H input/output arrays of hard decisions (bits)
// combine 2 arrays of length sz/2 and output an array of length sz/2
// examine bitwise storage instead of boolean
// est_in0 : input from first child node, size sz/2
// est_in1 : input from second child node, size sz/2
__device__ __forceinline__ void H_func(uint32_t* __restrict__      bitsOut,
                                         const uint32_t* __restrict__ bitsIn0,
                                         const int                    in0IdxOffset,
                                         uint32_t                     in0ArraySz,
                                         const uint32_t* __restrict__ bitsIn1,
                                         uint32_t                     in1ArraySz,
                                         int                          sz,
                                         const thread_group&          grp)
{
    // NOTE:  use in0IdxOffset for index of bitsIn0,
    // iniIdxOffset is always 0, hence not passed as input arg
    if (sz == 1) {
        auto out1 = isBitSet(bitsIn1, 0);
        auto out0 = isBitSet(bitsIn0, in0IdxOffset) != out1 ? 1 : 0;
        setResetSingleBit(bitsOut, 0, out0);
        setResetSingleBit(bitsOut, sz, out1);
    } else if (sz == 16) {
        if (grp.thread_rank() == 0) {
            constexpr uint32_t mask = 0xFFFFu;
            const uint32_t in1 = bitsIn1[0] & mask;
            bitsOut[0] = ((getWord(bitsIn0, in0ArraySz, in0IdxOffset) ^ in1) & mask) | (in1 << 16);
        }
    } else if (sz >= WORD_LENGTH) {
        const int32_t wordsPerHalf = sz / WORD_LENGTH;
        for (int32_t i = grp.thread_rank(); i < 2 * wordsPerHalf; i += grp.size()) {
            if (i < wordsPerHalf) {
                bitsOut[i] = getWord(bitsIn0, in0ArraySz, in0IdxOffset + i * WORD_LENGTH) ^ bitsIn1[i];
            } else {
                bitsOut[i] = bitsIn1[i - wordsPerHalf];
            }
        }
    } else {
        if (grp.thread_rank() == 0) {
            xorBits(bitsIn0, in0IdxOffset, in0ArraySz,
                    bitsIn1, 0, in1ArraySz,
                    bitsOut, 0, sz);
            copyBits(bitsIn1, sz, bitsOut, sz);
        }
    }
    grp.sync();
}


// Polar Decoder SCCL-PT: Successive Cancellation with Compressed storage and List Decoder, with Pruned Tree
template<uint32_t TILE_SZ, uint32_t LIST_SZ = 8>
__device__ __forceinline__ void
listPolarDecoder(polarDecoderDynDescr_t* pDynDescr)
{
    const uint32_t BLOCK_IDX = blockIdx.x;

    uint16_t N_cw     = pDynDescr->pCwPrmsGpu[BLOCK_IDX].N_cw;
    uint8_t  nCrcBits = pDynDescr->pCwPrmsGpu[BLOCK_IDX].nCrcBits;
    uint16_t A_cw     = pDynDescr->pCwPrmsGpu[BLOCK_IDX].A_cw;
    uint16_t n_cw     = find_msb(N_cw);
    uint32_t N_words  = N_cw / WORD_LENGTH;

    __half*   cwTreeLLR   = pDynDescr->cwTreeLLRsAddrs[BLOCK_IDX];
    uint32_t* cbEst       = pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCbEst;
    bool*     scratchBuf  = pDynDescr->listPolScratchAddrs[BLOCK_IDX];
    uint8_t*  treeTypes   = pDynDescr->pCwPrmsGpu[BLOCK_IDX].pCwTreeTypes;
    // Precomputed SC operation list stored behind the tree types (see
    // polar_cw_tree_layout.hpp): one byte per visited pruned node in bit order;
    // bits[3:0] = node stage, bits[5:4] = node type (0 frozen, 1 info, 2 parity).
    const uint8_t* opList = &treeTypes[cuphy::polar::PolarCwTreeLayout::scOpListOffset(1u << n_cw)];

    thread_block const& thisThrdBlk = this_thread_block();
    auto     tile = tiled_partition<TILE_SZ>(thisThrdBlk);
    uint32_t tid  = thisThrdBlk.thread_rank();

    // shared memory assignments
    __shared__ extern bool sh_buff[];

    int16_t* llPointers  = reinterpret_cast<int16_t*>(sh_buff);                     // for linked list pointers
    __half*  pathMetric  = reinterpret_cast<__half*>(&llPointers[n_cw * LIST_SZ]);  // for path metric
    // for bit-word arrays
    uint32_t* csBitWordsA     = reinterpret_cast<uint32_t*>(&pathMetric[LIST_SZ]);  // temporary bit-word buffer for codeword at a given stage, each word is for one warp
    uint32_t* csBitWordsB     = &csBitWordsA[LIST_SZ * (N_words + BCO)];            // temporary bit-word buffer for codeword at a given stage, each word is for one warp
    uint32_t* cwEstBitWords   = &csBitWordsB[LIST_SZ * (N_words + BCO)];            // buffer to store estimated codewords per stage

#if ENABLE_DEBUG
    if(threadIdx.x == 0)
    {
        if (BLOCK_IDX==0 && tid == 0) printf("N_cw %d, A_cw %d \n", N_cw, A_cw);
        //printf("LLRs ------------------------------------------------------------------\n");
        for(int i = 0; i < 2 * N_cw; i++)
        {
            printf("%9.1f ", static_cast<float>(cwTreeLLR[i]));
            if(i % 16 == 0)
            {
                printf("\n");
            }
        }
        printf("\n\n");
    }
#endif

    // Initialize ==========================================================================
    int     sz;              // length of sub-array in cwLLR
    int     stage    = n_cw; // stage variable
    uint8_t type     = 10;   // type used to prune decoder tree
    int     bitIdx   = -1;   // keeps track of the decoded bit index in the successive algorithm
    int     numPaths = 1;    // for list decoder
#ifdef _DEBUG
    assert (LIST_SZ <= 32);
    assert (LIST_SZ == 1 || LIST_SZ == 2 || LIST_SZ % 4 == 0);
    assert (N_cw > 31);
#endif
    if (tid < LIST_SZ) {
        pathMetric[tid] = 0;
    }
    for (int i = tid; i < n_cw * LIST_SZ; i += blockDim.x) {
        llPointers[i] = 0;
    }

    // first visited pruned node comes from the precomputed opList
    int     opIdx   = 0;
    uint8_t opEntry = __ldg(&opList[opIdx++]);

    // propagate input LLRs to the first visited node's stage
    while (stage > (opEntry & 0xF)) {
        stage--;
        sz = 1 << stage;
        auto* out = &cwTreeLLR[sz];
        auto* in  = &out[sz];
        F_func(out, in, sz, thisThrdBlk);
#if ENABLE_DEBUG
        if (threadIdx.x == 0 && stage==4) {
            printf("F function ----------------- stage %d, size %d ------------------\n", stage, sz);
            printf("in1: ");
            for (int i = 0; i < sz; i++) {
                printf("%6.1f ", __half2float(in[i]));
            }
            printf("\nin2: ");
            for (int i = sz; i < 2 * sz; i++) {
                printf("%6.1f ", __half2float(in[i]));
            }
            printf("\nout: ");
            for (int i = 0; i < sz; i++) {
                printf("%6.1f ", __half2float(out[i]));
            }
            printf("\n");
        }
#endif
    }


    // path_prime is primary path metric for a given stage
    __shared__ int16_t pathPrime[LIST_SZ];
    if (tid < LIST_SZ) {
        pathPrime[tid] = 0;
    }

    // Main loop  ==========================================================================
    while (bitIdx < (N_cw - 1)) {
        uint32_t * csBits = csBitWordsA;
        type = opEntry >> 4;

        // at this point, type is either 0 or 1
#ifdef _DEBUG
        assert(type < 10);
#endif
        if (type == 0) {
            // set cs array for this stage to 0
            R0_decoder<TILE_SZ>(pathPrime, pathMetric, csBits, stage, numPaths, cwTreeLLR, N_cw, tile);
        } else if ((type == 1 || type == 2) && stage == 0) {
            // decode leaf node
            R1_S0_decoder<LIST_SZ>(pathPrime, pathMetric, csBits, numPaths, cwTreeLLR, N_cw, thisThrdBlk);
        } else if (type == 1 && stage > 0) {
            // decode rate one condeword
            R1_decoder<LIST_SZ>(pathPrime, pathMetric, csBits, scratchBuf, numPaths, stage, cwTreeLLR, N_cw, thisThrdBlk);
        }
        thisThrdBlk.sync();

        // update bit index
        sz = 1 << stage;
        bitIdx += sz;
        if (bitIdx == (N_cw - 1)) break;

        // prefetch the next visited node's opList entry
        opEntry = __ldg(&opList[opIdx++]);

        // use H function to combine codeword estimates all the way up to stage d
        int stage_idx = POLAR_DEPTH[bitIdx];
        int csBufferSelector = 0;

        while (stage < stage_idx) {
            sz = 1 << stage;
            if (tile.meta_group_rank() < numPaths) {
                int  p    = tile.meta_group_rank();
                auto path = pathPrime[p];
                // input 0: it is always read from cwEst
                const auto* inBits0 = &cwEstBitWords[path * (N_words + BCO)];
                // input1: in1 and cs keep switching. Initially, in1 is read from cs_buffer_b and cs is stored in cs_bit_words_a
                const auto* inBits1 = csBufferSelector % 2 ? &csBitWordsB[p * (N_words + BCO)] : &csBitWordsA[p * (N_words + BCO)];
                // output: cs and in1 keep switching.
                csBits = csBufferSelector % 2 ? &csBitWordsA[p * (N_words + BCO)] : &csBitWordsB[p * (N_words + BCO)];
                //
                H_func(csBits, inBits0, sz - 1, N_words, inBits1, N_words, sz, tile);
            }

            get_p_prime<LIST_SZ>(pathPrime, llPointers, n_cw, stage, thisThrdBlk);
            stage++;
            csBufferSelector++;
        }

        // use G function with new codeword for stage_idx to update LLRs of the sibling branch
        sz = 1 << stage;

        if(tile.meta_group_rank() < numPaths)
        {
            int  p       = tile.meta_group_rank();
            auto path    = pathPrime[p];
            csBits       = csBufferSelector % 2 ? &csBitWordsB[p * (N_words + BCO)] : &csBitWordsA[p * (N_words + BCO)];
            auto llrOut  = &cwTreeLLR[sz + (2 * N_cw) * p];
            auto llrIn   = &cwTreeLLR[2 * sz + (2 * N_cw) * path];
            //
            G_func(llrOut, llrIn, csBits, sz, tile);
        }

        // use F function to propagate updated LLRs down to the next visited node's stage
        while (stage > (opEntry & 0xF)) {
            stage--;
            sz   = 1 << stage;
            if (tile.meta_group_rank() < numPaths) {
                int   p   = tile.meta_group_rank();
                auto* out = &cwTreeLLR[sz + (2 * N_cw) * p];
                auto* in  = &out[sz];
                F_func(out, in, sz, tile);
            }
        }

        // store stage_idx codeword estimates
        sz = 1 << stage_idx;
        if (tile.meta_group_rank() < numPaths) {
            int p   = tile.meta_group_rank();
            csBits  = csBufferSelector % 2 ? &csBitWordsB[p * (N_words + BCO)] : &csBitWordsA[p * (N_words + BCO)];
            if (tile.thread_rank() == 0) {
                copyBits(csBits, sz, cwEstBitWords, sz - 1 + p * (N_cw + BCO * WORD_LENGTH));
            }
        }
        thisThrdBlk.sync();

        // store linked list pointers
        if (tid < numPaths) {
            llPointers[stage_idx + n_cw * tid] = pathPrime[tid];
        }
        thisThrdBlk.sync();
    }
    //=========================================================================================

    // Finalize

    for (int i = tid; i <= N_cw * LIST_SZ; i += blockDim.x) {
        scratchBuf[i] = 0;
    }

    for (int i = tile.meta_group_rank(); i < numPaths; i+=tile.meta_group_size()) {
        auto cs_bits = &csBitWordsA[i * (N_words + BCO)];
        auto sharedBuf = &csBitWordsB[i * (N_words + BCO)];
        tile.sync();
        xor_butterfly3(&scratchBuf[N_cw - sz + i * N_cw], cs_bits, N_words, 0, sharedBuf, stage, sz, tile);
    }

    // use XOR butterfly structure to propagate codeword estimates to stage "0"
    int stageTmp = stage;
    for (stage = stageTmp; stage < n_cw; stage++) {
        sz = 1 << stage;
        for (int i = tile.meta_group_rank(); i < numPaths; i+=tile.meta_group_size()) {
            auto path      = pathPrime[i];
            auto cs_bits   = &cwEstBitWords[path * (N_words + BCO)];
            auto sharedBuf = &csBitWordsB[i * (N_words + BCO)];
            tile.sync();
            xor_butterfly3(&scratchBuf[N_cw - 2 * sz + i * N_cw], cs_bits, N_words, sz - 1, sharedBuf, stage, sz, tile);
        }
        get_p_prime<LIST_SZ>(pathPrime, llPointers, n_cw, stage, thisThrdBlk);
    }

    // Perform CRC check for each decoder in the list
    __shared__ bool crcFlags[LIST_SZ];
    uint8_t  crcErrFlag = 1;
    for (int i = tile.meta_group_rank(); i < LIST_SZ; i+=tile.meta_group_size())
    {
        int msgBitIdx = 0;    // to keep track of index of info bits
        auto* estBits   = &csBitWordsA[i * (N_words + BCO)];
        auto* sharedBuf = &csBitWordsB[i * (N_words + BCO) + i];        // + i is to avoid bank conflict

        for (int j = tile.thread_rank(); j < N_words; j += tile.size()) {
            estBits[j] = 0;
        }
        tile.sync();

        if (tile.thread_rank() == 0)
        {
            uint8x8* treeType8 = reinterpret_cast<uint8x8*>(treeTypes);
            bool8*   outBits8  = reinterpret_cast<bool8*>(scratchBuf);

            for(int j = 0; j < N_cw; j += 8)
            {
                uint8x8 type8 = get_type8(0, n_cw, j, treeType8);
                if (__popcll(type8.u64) > 0)
                {
                    bool8 out8 = outBits8[(j + N_cw * i) / 8];
#pragma unroll
                    for(int k = 0; k < 8; k++)
                    {
                        if(type8.u8[k] == 1)
                        {
                            int wIdx      = msgBitIdx / WORD_LENGTH;
                            int bIdx      = msgBitIdx % WORD_LENGTH;
                            int tmpW      = out8.b8[k] << bIdx;
                            atomicOr(estBits + wIdx, tmpW); //estBits[wIdx] = estBits[wIdx] | tmpW;
                            msgBitIdx++;
                        }
                    }
                }

            }
            // now compute CRC from info bits and compare with the last nCrcBits
            crcFlags[i] = validate_crc(nCrcBits, estBits, A_cw, sharedBuf);
        }
    }

    thisThrdBlk.sync();

    // select the decoder with correct crc, if all fail return the first one
    if(threadIdx.x == 0)
    {
        int dcdrIdx = 0;
        for(int i = 0; i < LIST_SZ; i++)
        {
            if(!crcFlags[i])
            {
                dcdrIdx    = i;
                crcErrFlag = 0;
                break;
            }
        }
        // write back crc flag
        pDynDescr->pPolCrcErrorFlags[BLOCK_IDX] = crcErrFlag;

        uint32_t* estBits = &csBitWordsA[dcdrIdx * (N_words + BCO)];
        resetBits(estBits, A_cw, nCrcBits);

        // copy output to gmem
        int numCbEstWords = div_round_up(static_cast<int>(A_cw), WORD_LENGTH);
        for(int j = 0; j < numCbEstWords; j++)
        {
            cbEst[j] = estBits[j];
        }
    }

    //======================================================================================
    // If parentUciSeg composed of two codeblocks place cbEst carefully into uciSegEst

    if((threadIdx.x == 0) && (pDynDescr->pCwPrmsGpu[BLOCK_IDX].nCbsInUciSeg == 2) )
    {
        updateUciSegEstForTwoCbsInUciSeg(pDynDescr, A_cw, cbEst, BLOCK_IDX);
    }
}

template<uint32_t TILE_SZ, uint32_t LIST_SZ = 8>
__launch_bounds__(1024,1)
static __global__ void
listPolarDecoderKernel(polarDecoderDynDescr_t* pDynDescr)
{
    // check for early exit:
    uint8_t exitFlag = pDynDescr->pCwPrmsGpu[blockIdx.x].exitFlag;
    if(exitFlag == 1)
    {
        return;
    }

    // first run simple polar decoder (list size 1); the list kernel runs one
    // 32-thread block per codeword, so the SC pass uses the whole shared buffer
    __shared__ extern bool sh_buff[];
    singlePolarDecoder(pDynDescr, blockIdx.x, threadIdx.x, sh_buff);
    sync_barrier<POLAR_DECODER_BLOCK_SIZE>();
    // then check if decoding was successful
    uint8_t crcErrFlag = pDynDescr->pPolCrcErrorFlags[blockIdx.x];
    //  if there is CRC error, then run list polar decoder
    if(crcErrFlag)
    {
        listPolarDecoder<TILE_SZ, LIST_SZ>(pDynDescr);
    }
    // Update Detection (CRC) Status
    // ToDo currently only look at CRC of first CB. Need to use uint32_t CRC along with atomic operations. Requires API and cuPHY controller changes
    if((threadIdx.x == 0) && (pDynDescr->pCwPrmsGpu[blockIdx.x].cbIdxWithinUciSeg == 0))
    {
        updateCRCstatus(pDynDescr, pDynDescr->pPolCrcErrorFlags[blockIdx.x], blockIdx.x);
    }
}

} //namespace polar_decoder
//---------------------------------------------------------------------------------------------------

// Per-codeword dynamic shared-memory slice of the single/SC decoder for
// codewords up to maxN bits: two boolean cs buffers of maxN/2, the packed
// per-stage estimate buffer (maxN bits) and a CRC scratch of N_MAX_WORDS
// words. Region offsets and the total are word-aligned so that per-warp
// slices remain aligned.
static inline int scDecoderSliceBytes(int maxN)
{
    const int estByteOffset = (2 * (maxN / 2) + 3) & ~3;
    const int estBytes      = ((std::max(maxN / 32, 1)) + polar_decoder::N_MAX_WORDS) * static_cast<int>(sizeof(uint32_t));
    return estByteOffset + estBytes;
}

template<int LIST_SZ>
void polarDecoder::kernelSelect(uint16_t                      nPolCws,
                                const cuphyPolarCwPrm_t*      pPolUciSegPrmsCpu,
                                cuphyPolarDecoderLaunchCfg_t* pLaunchCfg)
{
    // Size buffers for the largest codeword of THIS launch rather than the
    // absolute maximum: both kernels compute their shared-memory offsets from
    // the per-block N_cw, so the allocation only needs to cover the largest
    // block, and a smaller footprint raises the resident-block limit.
    int launchMaxN = 32;
    for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
    {
        launchMaxN = std::max(launchMaxN, static_cast<int>(pPolUciSegPrmsCpu[cwIdx].N_cw));
    }

    // launch geometry
    constexpr int blkSize = polar_decoder::POLAR_DECODER_BLOCK_SIZE;
    constexpr int tileSize = blkSize / LIST_SZ;
    dim3 gridDim(nPolCws);
    dim3 blockDim(blkSize);
    int scWarpsPerBlock = 1;
    if(LIST_SZ == 1)
    {
        // the SC decoder packs several one-warp codewords into each block for
        // large batches; small latency-critical batches keep one warp per block
        scWarpsPerBlock = (nPolCws >= polar_decoder::POLAR_SC_MULTIWARP_MIN_CWS) ? polar_decoder::POLAR_SC_WARPS_PER_BLOCK : 1;
        gridDim.x  = (nPolCws + scWarpsPerBlock - 1) / scWarpsPerBlock;
        blockDim.x = scWarpsPerBlock * polar_decoder::CUDA_WARP_SIZE;
    }

    // kernel
    void* kernelFunc;
    if (LIST_SZ > 1) {
        kernelFunc = reinterpret_cast<void*>(polar_decoder::listPolarDecoderKernel<tileSize, LIST_SZ>);
    } else {
        kernelFunc = reinterpret_cast<void*>(polar_decoder::polarDecoderKernel);
    }

   {MemtraceDisableScope md;cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);}

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra = nullptr;

    if (LIST_SZ > 1) {
        int launchMaxWords = launchMaxN / polar_decoder::WORD_LENGTH;
        int launchMaxDepth = 0;
        while((1 << launchMaxDepth) < launchMaxN)
        {
            launchMaxDepth++;
        }
        int dyn_shared_sz = 0;
        dyn_shared_sz += LIST_SZ * launchMaxDepth * sizeof(int16_t);                             // for linked list pointers
        dyn_shared_sz += LIST_SZ * sizeof(__half);                                               // for path metrics
        dyn_shared_sz += LIST_SZ * 2 * (launchMaxWords + polar_decoder::BCO) * sizeof(uint32_t); // for both temp buffers used to keep a copy of codeword per stage,
                                                                                                 // each word stores 32 bits, +BCO is to minimize bank conflicts
        dyn_shared_sz += LIST_SZ * (launchMaxWords + polar_decoder::BCO) * sizeof(uint32_t);     // for storing estimated code word per stage, each word
                                                                                                 // stores 32 bits, +BCO is to minimize bank conflicts

        // since in fall-back method, we first run with list size 1 and use polarDecoderKernel() instead of listPolarDecoderKernel<tileSize, 1>(),
        // let's ensure allocated dynamic shared mem is large enough for the SC pass as well
        dyn_shared_sz = std::max(dyn_shared_sz, scDecoderSliceBytes(launchMaxN));

        kernelNodeParamsDriver.sharedMemBytes = dyn_shared_sz;
    } else {
        kernelNodeParamsDriver.sharedMemBytes = scWarpsPerBlock * scDecoderSliceBytes(launchMaxN);
    }

}

void polarDecoder::setup(uint16_t                      nPolCws,                     // number of polar codewords
                         __half**                      pCwTreeLLRsAddrs,            // pointer to codeword tree LLR addresses
                         cuphyPolarCwPrm_t*            pCwPrmsGpu,                  // pointer to codeword parameters in GPU
                         cuphyPolarCwPrm_t*            pCwPrmsCpu,                  // pointer to codeword parameters in CPU
                         uint32_t**                    pPolCbEstAddrs,              // pointer to estimated codeblock addresses
                         bool**                        pListPolScratchAddrs,        // pointer to scratch buffer used in list polar decoder
                         uint8_t                       nPolLists,                   // list size for polar decoder
                         uint8_t*                      pPolCrcErrorFlags,           // pointer to buffer storing CRC error flags
                         bool                          enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                         polarDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                         void*                         pGpuDynDesc,                 // pointer to descriptor in gpu
                         cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                         cudaStream_t                  strm)                        // stream to perform copy
{
    // populate dynamic descriptor:
    pCpuDynDesc->pCwPrmsGpu        = pCwPrmsGpu;
    pCpuDynDesc->pPolCrcErrorFlags = pPolCrcErrorFlags;
    pCpuDynDesc->nPolCws           = nPolCws;
    {
        int launchMaxN = 32;
        for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
        {
            launchMaxN = std::max(launchMaxN, static_cast<int>(pCwPrmsCpu[cwIdx].N_cw));
        }
        pCpuDynDesc->scSharedSliceBytes = static_cast<uint16_t>(scDecoderSliceBytes(launchMaxN));
    }

    for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
    {
        pCpuDynDesc->cwTreeLLRsAddrs[cwIdx] = pCwTreeLLRsAddrs[cwIdx];
        pCpuDynDesc->polCbEstAddrs[cwIdx]   = pPolCbEstAddrs[cwIdx];
    }

    if (pListPolScratchAddrs != nullptr) { // using list decoder for polar codes
        for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
        {
            pCpuDynDesc->listPolScratchAddrs[cwIdx] = pListPolScratchAddrs[cwIdx];
        }
    }


    // save pointer to GPU descriptor
    polarDecoderKernelArgs_t& kernelArgs = m_kernelArgs;
    kernelArgs.pDynDescr                 = reinterpret_cast<polarDecoderDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(polarDecoderDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
    // kernelSelect reads per-codeword N_cw on the host, so it must receive the
    // CPU copy of the codeword parameters.
    if (nPolLists == 1) {
        kernelSelect<1>(nPolCws, pCwPrmsCpu, pLaunchCfg);
    } else if (nPolLists == 2) {
        kernelSelect<2>(nPolCws, pCwPrmsCpu, pLaunchCfg);
    } else if (nPolLists == 4) {
        kernelSelect<4>(nPolCws, pCwPrmsCpu, pLaunchCfg);
    } else { // list size 8
        kernelSelect<8>(nPolCws, pCwPrmsCpu, pLaunchCfg);
    }
    pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void polarDecoder::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    dynDescrSizeBytes  = sizeof(polarDecoderDynDescr_t);
    dynDescrAlignBytes = alignof(polarDecoderDynDescr_t);
}
