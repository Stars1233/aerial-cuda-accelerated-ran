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

#include "cuphy_internal.h"
//#include <cooperative_groups.h>
#include "polar_encoder.hpp"
#include "polar_encoder.cuh"
#include "cuphy_api.h"

using namespace cooperative_groups;

// #define ENABLE_DEBUG

namespace polar_encoder
{

template <typename T>
CUDA_BOTH_INLINE constexpr T round_up_to_next(T val, T increment)
{
    return ((val + (increment - 1)) / increment) * increment;
}

static constexpr uint32_t N_THRDS_PER_WARP      = 32;         // cudaDeviceProp::warpSize;
static constexpr uint32_t FULL_WARP_ACTIVE_BMSK = 0xFFFFFFFF; // bitmaks when all threads in a warp are active
static constexpr uint32_t N_THRDS_PER_TILE      = N_THRDS_PER_WARP;
static constexpr uint32_t N_BITS_PER_WORD       = 32;
static constexpr uint32_t N_BITS_PER_BYTE       = 8;
static constexpr uint32_t N_BYTES_PER_WORD      = 4;

// Sizing the thread block as needed by maximum number of coded bits
// max(N_MAX_INFO_BITS, N_MAX_CODED_BITS) = N_MAX_CODED_BITS
static constexpr uint32_t N_MAX_THRDS_PER_THRD_BLK = round_up_to_next(N_MAX_CODED_BITS, N_THRDS_PER_TILE);
static constexpr uint32_t N_MAX_THRD_TILES         = N_MAX_THRDS_PER_THRD_BLK / N_THRDS_PER_WARP;

// clang-format off
static __device__ __constant__  uint8_t POLAR_ENC_INFO_BIT_INTERLEAVER_IDX[] =  
{
    0,   2,   4,   7,   9,  14,  19,  20,  24,  25,  26,  28,  31,   34,  42,  45,  49,  50,  51,  53,  54,  
   56,  58,  59,  61,  62,  65,  66,  67,  69,  70,  71,  72,  76,   77,  81,  82,  83,  87,  88,  89,  91,  
   93,  95,  98, 101, 104, 106, 108, 110, 111, 113, 115, 118, 119,  120, 122, 123, 126, 127, 129, 132, 134,
  138, 139, 140,   1,   3,   5,   8,  10,  15,  21,  27,  29,  32,   35,  43,  46,  52,  55,  57,  60,  63,
   68,  73,  78,  84,  90,  92,  94,  96,  99, 102, 105, 107, 109,  112, 114, 116, 121, 124, 128, 130, 133,
  135, 141,   6,  11,  16,  22,  30,  33,  36,  44,  47,  64,  74,  79,  85,   97, 100, 103, 117, 125, 131, 
  136, 142,  12,  17,  23,  37,  48,  75,  80,  86, 137, 143,  13,  18,  38,  144,  39, 145,  40, 146,  41,
  147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,  162, 163
};

// Reliability sequence values < 32
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_32[] = 
{
   0,	 1,	 2,	 4,	 8,	16,	 3,	 5,	 9,	 6,	17,	10,	18,	12,	20,	24,
   7,	11,	19,	13,	14,	21,	26,	25,	22,	28,	15,	23,	27,	29,	30,	31
};

// Reliability sequence values < 64 
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_64[] = 
{
   0,	 1,	 2,  4,	 8,	16,	32,	 3,	 5,	 9,	 6,	17,	10,	18,	12,	33,
  20,	34,	24,	36,	 7,	11,	40,	19,	13,	48,	14,	21,	35,	26,	37,	25,
  22,	38,	41,	28,	42,	49,	44,	50,	15,	52,	23,	56,	27,	39,	29,	43,
  30,	45,	51,	46,	53,	54,	57,	58,	60,	31,	47,	55,	59,	61,	62,	63
};

// Reliability sequence values < 128 
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_128[] = 
{
     0,	  1,	  2,	  4,	  8,	 16,	 32,	 3,	  5,	 64,	  9,	  6,	 17,	 10,	 18,	 12,
    33,	 65,	 20,	 34,	 24,	 36,	  7,  66,	 11,	 40,	 68,	 19,	 13,	 48,	 14,	 72,
    21,	 35,	 26,	 80,	 37,	 25,	 22,  38,	 96,	 67,	 41,	 28,	 69,	 42,	 49,	 74,
    70,	 44,	 81,	 50,	 73,	 15,	 52,  23,	 76,	 82,	 56,	 27,	 97,	 39,	 84,	 29,
    43,	 98,	 88,	 30,	 71,	 45,	100,  51,	 46,	 75,	104,	 53,	 77,	 54,	 83,	 57,
   112,	 78,	 85,	 58,	 99,	 86,	 60,  89,	101,	 31,	 90,	102,	105,	 92,	 47,	106,
    55,	113,	 79,	108,	 59,	114,	 87, 116,	 61,	 91,	120,	 62,	103,	 93,	107,	 94,
   109,	115,	110,	117,	118,	121,	122,	63,	124,	 95,	111,	119,	123,	125,	126,	127
};

// Reliability sequence values < 256 
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_256[] = 
{
    0,	  1,	  2,	  4,	  8,	 16,	 32,	  3,	  5,	 64,	  9,	  6,	 17,	 10,	 18,	128,
   12,	 33,	 65,	 20,	 34,	 24,	 36,	  7,	129,	 66,	 11,	 40,	 68,	130,	 19,	 13,
   48,	 14,	 72,	 21,	132,	 35,	 26,	 80,	 37,	 25,	 22,	136,	 38,	 96,	 67,	 41,
  144,	 28,	 69,	 42,	 49,	 74,	160,	192,	 70,	 44,	131,	 81,	 50,	 73,	 15,	133,
   52,	 23,	134,	 76,	137,	 82,	 56,	 27,	 97,	 39,	 84,	138,	145,	 29,	 43,	 98,
   88,	140,	 30,	146,	 71,	161,	 45,	100,	 51,	148,	 46,	 75,	104,	162,	 53,	193,
  152,	 77,	164,	 54,	 83,	 57,	112,	135,	 78,	194,	 85,	 58,	168,	139,	 99,	 86,
   60,	 89,	196,	141,	101,	147,	176,	142,	 31,	200,	 90,	149,	102,	105,	163,	 92,
   47,	208,	150,	153,	165,	106,	 55,	113,	154,	 79,	108,	224,	166,	195,	 59,	169,
  114,	156,	 87,	197,	116,	170,	 61,	177,	 91,	198,	172,	120,	201,	 62,	143,	103,
  178,	 93,	202,	107,	180,	151,	209,	 94,	204,	155,	210,	109,	184,	115,	167,	225,
  157,	110,	117,	212,	171,	226,	216,	158,	118,	173,	121,	199,	179,	228,	174,	122,
  203,	 63,	181,	232,	124,	205,	182,	211,	185,	240,	206,	 95,	213,	186,	227,	111,
  214,	188,	217,	229,	159,	119,	218,	230,	233,	175,	123,	220,	183,	234,	125,	241,
  207,	187,	236,	126,	242,	244,	189,	215,	219,	231,	248,	190,	221,	235,	222,	237,
  243,	238,	245,	127,	191,	246,	249,	250,	252,	223,	239,	251,	247,	253,	254,	255
};

// Reliability sequence values < 512 
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_512[] = 
{
    0,	  1,	  2,	  4,	  8,	 16,	 32,	  3,	  5,	 64,	  9,	  6,	 17,	 10,	 18,	128,
   12,	 33,	 65,	 20,	256,	 34,	 24,	 36,	  7,	129,	 66,	 11,	 40,	 68,	130,	 19,
   13,	 48,	 14,	 72,	257,	 21,	132,	 35,	258,	 26,	 80,	 37,	 25,	 22,	136,	260,
  264,	 38,	 96,	 67,	 41,	144,	 28,	 69,	 42,	 49,	 74,	272,	160,	288,	192,	 70,
   44,	131,	 81,	 50,	 73,	 15,	320,	133,	 52,	 23,	134,	384,	 76,	137,	 82,	 56,
   27,	 97,	 39,	259,	 84,	138,	145,	261,	 29,	 43,	 98,	 88,	140,	 30,	146,	 71,
  262,	265,	161,	 45,	100,	 51,	148,	 46,	 75,	266,	273,	104,	162,	 53,	193,	152,
   77,	164,	268,	274,	 54,	 83,	 57,	112,	135,	 78,	289,	194,	 85,  276,   58,  168,
  139,	 99,	 86,	 60,	280,	 89,	290,	196,	141,	101,	147,	176,	142,	321,	 31,	200,
   90,	292,	322,	263,	149,	102,	105,	304,	296,	163,	 92,	 47,	267,	385,	324,	208,
  386,	150,	153,	165,	106,	 55,	328,	113,	154,	 79,	269,	108,	224,	166,	195,	270,
  275,	291,	 59,	169,	114,	277,	156,	 87,	197,	116,	170,	 61,	281,	278,	177,	293,
  388,	 91,	198,	172,	120,	201,	336,	 62,	282,	143,	103,	178,	294,	 93,  202,  323,
  392,	297,	107,	180,	151,	209,	284,	 94,	204,	298,	400,	352,	325,	155,	210,	305,
  300,	109,	184,	115,	167,	225,	326,	306,	157,	329,	110,	117,	212,	171,	330,	226,
  387,	308,	216,	416,	271,	279,	158,	337,	118,	332,	389,	173,	121,	199,	179,	228,
  338,	312,	390,	174,	393,	283,	122,	448,	353,	203,	 63,  340,  394,  181,  295,  285,
  232,	124,	205,	182,	286,	299,	354,	211,	401,	185,	396,	344,	240,	206,	 95,	327,
  402,	356,	307,	301,	417,	213,	186,	404,	227,	418,	302,	360,	111,	331,	214,	309,
  188,	449,	217,	408,	229,	159,	420,	310,	333,	119,	339,	218,	368,	230,	391,	313,
  450,	334,	233,	175,	123,	341,	220,	314,	424,	395,	355,	287,	183,	234,	125,	342,
  316,	241,	345,	452,	397,	403,	207,	432,	357,	187,	236,	126,	242,	398,	346,	456,
  358,	405,	303,	244,	189,	361,	215,	348,	419,	406,	464,	362,	409,	219,	311,	421,
  410,	231,	248,	369,	190,	364,	335,	480,	315,	221,	370,	422,	425,	451,	235,	412,
  343,	372,	317,	222,	426,	453,	237,	433,	347,	243,	454,	318,	376,	428,	238,	359,
  457,	399,	434,	349,	245,	458,	363,	127,	191,	407,	436,	465,	246,	350,	460,	249,
  411,	365,	440,	374,	423,	466,	250,	371,	481,	413,	366,	468,	429,	252,	373,	482,
  427,	414,	223,	472,	455,	377,	435,	319,	484,	430,	488,	239,	378,	459,	437,	380,
  461,	496,	351,	467,	438,	251,	462,	442,	441,	469,	247,	367,	253,	375,	444,	470,
  483,	415,	485,	473,	474,	254,	379,	431,	489,	486,	476,	439,	490,	463,	381,	497,
  492,	443,	382,	498,	445,	471,	500,	446,	475,	487,	504,	255,	477,	491,	478,	383,
  493,	499,	502,	494,	501,	447,	505,	506,	479,	508,	495,	503,	507,	509,	510,	511
};

// Table for encoded bit sub-block interleaving 
static __device__ __constant__ uint8_t POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[] = 
{
   0,  1,  2,  4,  3,  5,  6,  7,  8, 16,  9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31
};

// clang-format on

static constexpr uint32_t N_POLAR_REL_SEQ_IDXS_32  = sizeof(POLAR_REL_SEQ_IDXS_32) / sizeof(POLAR_REL_SEQ_IDXS_32[0]);
static constexpr uint32_t N_POLAR_REL_SEQ_IDXS_512 = sizeof(POLAR_REL_SEQ_IDXS_512) / sizeof(POLAR_REL_SEQ_IDXS_512[0]);
static_assert((N_MIN_CODED_BITS == N_POLAR_REL_SEQ_IDXS_32), "Smallest realiability sequence index table size doesn't match min coded bit count");
static_assert((N_MAX_CODED_BITS == N_POLAR_REL_SEQ_IDXS_512), "Largest realiability sequence index table size doesn't match max coded bit count");

// clang-format off

// Pointers to reliability sequence LUTs
static __device__ __constant__ uint16_t const* POLAR_REL_SEQ_IDXS_LUT_PTR[] =
{
   POLAR_REL_SEQ_IDXS_32,
   POLAR_REL_SEQ_IDXS_64,
   POLAR_REL_SEQ_IDXS_128,
   POLAR_REL_SEQ_IDXS_256,
   POLAR_REL_SEQ_IDXS_512
};

// Forward and backward forbidden reliability sequence indices
static __device__ __constant__ int8_t POLAR_REL_SEQ_FORBID_IDXS_FWD[][32] = 
{
  { -1,  -1,  -1,   1,  5,  -1,  -1,  -1,  -1,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,  -1,  -1,  -1,  -1,   1,  29,  -1,  -1,  -1, },
  { -1,  -1,  -1,   3,  5,  -1,  -1,  -1,  -1,   9,  17,  10,  18,  11,  19,  12,  20,  13,  21,  14,  22,  15,  23,  -1,  -1,  -1,  -1,  27,  29,  -1,  -1,  -1, },
  {  1,   1,   1,   5,  1,   1,   1,   1,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,   1,  17,   1,   1,   1,   1,   1,  29,   1,   1,   1,   1, },
  {  1,   2,   3,   5,  4,   6,   7,   8,   9,  17,  10,  18,  11,  19,  12,  20,  13,  21,  14,  22,  15,  23,  16,  24,  25,  26,  27,  29,  28,  30,  31,  32  }
};

static __device__ __constant__ int8_t POLAR_REL_SEQ_FORBID_IDXS_BWD[][32] = 
{
  { -1,  -1,  -1,  30,  28,  -1,  -1,  -1,  -1,  24,  16,  23,  15,  22,  14,  21,  13,  20,  12,  19,  11,  18,  10,  -1,  -1,  -1,  -1,   6,  4,  -1,  -1,  -1, },
  { -1,  -1,  -1,  32,  28,  -1,  -1,  -1,  -1,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  -1,  -1,  -1,  -1,  32,  4,  -1,  -1,  -1, },
  { 32,  31,  30,  28,  29,  27,  26,  25,  24,  16,  23,  15,  22,  14,  21,  13,  20,  12,  19,  11,  18,  10,  17,   9,   8,   7,   6,   4,  5,   3,   2,   1, },
  { 32,  32,  32,  28,  32,  32,  32,  32,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  32,  16,  32,  32,  32,  32,  32,   4, 32,  32,  32,  32  }
};
// clang-format on

static CUDA_BOTH_INLINE constexpr uint32_t roundUpToPow2(int32_t n, int32_t pow2)
{
    return pow2 >= n ? pow2 : roundUpToPow2(n, pow2 * 2);
}

// Round n upto nearest power of 2 compile time
static CUDA_BOTH_INLINE constexpr uint32_t roundUpToPow2(int32_t n)
{
    return roundUpToPow2(n, 1);
}
static constexpr uint32_t N_MAX_SORT_ENTRIES = roundUpToPow2(N_MAX_INFO_BITS);

// Relative sequence index buffer needs to be sized to:
// - hold the largest sequence possible, this is N_MAX_CODED_BITS
// - hold the largest padding for bitonic sort
//   Maximum number of entries to pad = largest bitonic sort size (256) - smallest number of information bits which needs the largest bitonic sort size (129)
static constexpr uint32_t REL_SEQ_IDX_BUF_PAD = N_MAX_SORT_ENTRIES - ((N_MAX_SORT_ENTRIES/2) + 1);
static constexpr uint32_t REL_SEQ_IDX_BUF_LEN = N_MAX_CODED_BITS + REL_SEQ_IDX_BUF_PAD;

// Round n upto nearest power of 2 runtime
static __device__ __forceinline__ uint32_t roundUpToPow2Gpu(uint32_t n)
{
    return (n > 1) ? (1U << (32 - __clz(n - 1))) : 1;
}

static __host__ __forceinline__ uint32_t roundUpToPow2Cpu(uint32_t n)
{
    return (n > 1) ? (1U << (32 - __builtin_clz(n - 1))) : 1;
}

//  __lanemask_lt() returns the mask of all lanes (including inactive ones) with ID less than the current
// lane
__device__ __forceinline__ uint32_t __lanemask_lt()
{
    uint32_t mask;
    // Move 32b special register, lanemask_lt, into result register (stored in output operand "mask")
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(mask));
    return mask;
}

// Compute warp level prefix sum - exclusive scan
__device__ __forceinline__ uint32_t warpLevelExclusiveScan(bool pred)
{
    uint32_t validRelSeqBmsk = __ballot_sync(FULL_WARP_ACTIVE_BMSK, pred);
    return __popc(validRelSeqBmsk & __lanemask_lt());
}

// the stream-compaction code uses exclusive scan, not inclusive scan, so these functions are never called
/*VCAST_DONT_INSTRUMENT_START*/
//  __lanemask_le() returns the mask of all lanes (including inactive ones) with ID less than or equal to the
// current lane
__device__ __forceinline__ uint32_t __lanemask_le()
{
    uint32_t mask;
    // Move 32b special register, lanemask_le, into result register (stored in output operand "mask")
    asm("mov.u32 %0, %lanemask_le;"
        : "=r"(mask));
    return mask;
}

// Compute warp level prefix sum - inclusive scan
__device__ __forceinline__ uint32_t warpLevelInclusiveScan(bool pred)
{
    uint32_t validRelSeqBmsk = __ballot_sync(FULL_WARP_ACTIVE_BMSK, pred);
    return __popc(validRelSeqBmsk & __lanemask_le());
}
/*VCAST_DONT_INSTRUMENT_END*/

// 1. Per warp use __ballot to receive a bit mask representing all the threads in the warp which returns true
//    to the predicate argument (i.e. if the information bit location corresponding to the thread is valid)
// 2. Mask all the bits corresponding to threads above the thread laneid
// 3. Use __popc to get the thread offset within the warp. Warp offset is the offset of last thread
// 4. Sync
// 5. Accumulate start offsets of each warp

// Stream compaction helper
template <uint32_t N_THRDS_PER_TILE>
__device__ void strmCompactionHelper(thread_block const& thisThrdBlk, thread_block_tile<N_THRDS_PER_TILE> const& thisThrdTile, bool pred, uint32_t nActiveThrdTiles, int32_t* pTileStartOffset, int32_t& thrdOffset)
{
    uint32_t thrdIdxInBlk  = thisThrdBlk.thread_rank();
    uint32_t thrdIdxInTile = thisThrdTile.thread_rank();
    uint32_t nTileThrds    = thisThrdTile.size();
    uint32_t tileIdx       = thrdIdxInBlk / nTileThrds;

    // Reset thread start offset accumulators (in shared memory)
    if((0 == thrdIdxInTile) && (tileIdx < nActiveThrdTiles))
    {
        pTileStartOffset[tileIdx] = 0;
    }

    static_assert((N_THRDS_PER_TILE == N_THRDS_PER_WARP), "Using warp level scan to compute threads offsets in tile");
    thrdOffset = warpLevelExclusiveScan(pred);

    // Wait for all shared memory accumulators to reset
    __syncthreads();

    // Thread offset of the last thread in current thread tile is the start offset for the next thread tile
    uint32_t lastThrdRank = nTileThrds - 1;
    for(uint32_t i = 1; i < nActiveThrdTiles; ++i)
    {
        // Accumulate offsets in shared memory: to compute start offset of a given thread tile, accumulate
        // thread offsets of the last thread in thread tiles leading upto the thread tile whose start offset is
        // being computed
        if((lastThrdRank == thrdIdxInTile) && (tileIdx < i))
        {
            // Since warpLevelScan is exclusive, the thrdOffset does not include the current thread's
            // contribution. However when computing the start offset of the next tile, the current thread
            // (which here is the last thread) contribution needs to be accountuned for
            uint32_t lastThrdOffset = pred ? 1 : 0;
            atomicAdd(&pTileStartOffset[i], thrdOffset + lastThrdOffset);
        }
    }

    // Wait for accumulations to complete
    __syncthreads();

#ifdef ENABLE_DEBUG
    if(0 == thrdIdxInBlk) printf("nActiveThrdTiles = %d\n", nActiveThrdTiles);
    if(tileIdx < nActiveThrdTiles)
    {
        printf("blockIdx.x %d ThreadOffset[%d] = %d (pred %u)\n", blockIdx.x, thrdIdxInBlk, thrdOffset, pred);
        if(0 == thrdIdxInTile)
        {
            printf("blockIdx.x %d TileStartOffset[%d] = %d\n", blockIdx.x, tileIdx, pTileStartOffset[tileIdx]);
        }
    }
#endif
}

template <uint32_t N_THRDS_PER_TILE>
__device__ void computeInterleaverIdxs(thread_block const& thisThrdBlk, thread_block_tile<N_THRDS_PER_TILE> const& thisThrdTile, uint32_t nInfoBits, uint32_t nCodedBits, int32_t* pTileStartOffsets, uint8_t* pInvInterleaverIdxs, uint32_t procModeBmsk)
{
    uint32_t thrdIdxInBlk           = thisThrdBlk.thread_rank();
    uint32_t nTileThrds             = thisThrdTile.size();
    uint32_t tileIdx                = thrdIdxInBlk / nTileThrds;
    int16_t  interleaverTblStartIdx = N_MAX_INFO_BITS - nInfoBits;

    // Flag to determine Bit Interleaving based on UL or DL Encoder implementation
    // DL Encoder (default) needs Bit Interleaving, while UL Encoder doesn't need it
    uint8_t bitInterleaverUlDlIdx =  (thrdIdxInBlk < N_MAX_INFO_BITS) ? (procModeBmsk && 0x1) ? thrdIdxInBlk : POLAR_ENC_INFO_BIT_INTERLEAVER_IDX[thrdIdxInBlk] : 0; 
    
    // predicate for stream compaction
    bool validInterleaverIdx =
        ((thrdIdxInBlk < N_MAX_INFO_BITS) && (bitInterleaverUlDlIdx >= interleaverTblStartIdx)) ?
            true :
            false;

    // Activate as many thread tiles as there are interleaver indices (i.e. max number of coded bits)
    static constexpr uint32_t N_ACTIVE_THRDS      = round_up_to_next(N_MAX_INFO_BITS, N_THRDS_PER_TILE);
    static constexpr uint32_t N_ACTIVE_THRD_TILES = N_ACTIVE_THRDS / N_THRDS_PER_TILE;    
    static_assert((N_ACTIVE_THRD_TILES <= N_MAX_THRD_TILES), "Active thread tiles needed exceed max limit");

    int32_t  thrdOffset  = 0;
    strmCompactionHelper<N_THRDS_PER_TILE>(thisThrdBlk, thisThrdTile, validInterleaverIdx, N_ACTIVE_THRD_TILES, pTileStartOffsets, thrdOffset);

    // Wait for accumulations to complete
    __syncthreads();

    // uint32_t nValidInterleaverBits = __syncthreads_count(pred);
    if(validInterleaverIdx)
    {
        int32_t interleaverIdxTblOffset        = pTileStartOffsets[tileIdx] + thrdOffset;
        int16_t invInterleaverIdx              = bitInterleaverUlDlIdx - interleaverTblStartIdx;
        pInvInterleaverIdxs[invInterleaverIdx] = interleaverIdxTblOffset;

#ifdef ENABLE_DEBUG
        printf("tileIdx %d, thrdIdxInBlk %d, TileStartOffset %d, ThrdOffset %d, POLAR_ENC_INFO_BIT_INTERLEAVER_IDX %d, InvInterleaverIdxs[%d] = %d\n", tileIdx, thrdIdxInBlk, pTileStartOffsets[tileIdx], thrdOffset, POLAR_ENC_INFO_BIT_INTERLEAVER_IDX[thrdIdxInBlk], invInterleaverIdx, pInvInterleaverIdxs[invInterleaverIdx]);
#endif
    }

    // Wait for all (valid) threads to build the inverse interleaver index table
    __syncthreads();

#ifdef ENABLE_DEBUG
    if(0 == thrdIdxInBlk) printf("nInfoBits %d, interleaverTblStartIdx %d\n", nInfoBits, interleaverTblStartIdx);
    uint32_t nInterleaverIdxs = __syncthreads_count(validInterleaverIdx);
    if(thrdIdxInBlk < nInterleaverIdxs)
    {
        printf("InvInterleaverIdxs[%d] = %d\n", thrdIdxInBlk, pInvInterleaverIdxs[thrdIdxInBlk]);
    }
#endif
}


// Number of entries to be sorted needs to be power of 2 for Bitonic sort
__device__ __forceinline__ uint32_t getNumBitonicSortEntries(uint32_t nEntries)
{
    uint32_t nSortEntries = roundUpToPow2Gpu(nEntries);
    if(nSortEntries < 2) nSortEntries = 2;

    // the bitonic sort code is never called
    /*VCAST_DONT_INSTRUMENT_START*/
    // The number of sort entires is not expected to exceed 512 (N_MAX_CODED_BITS)
    if(nSortEntries > N_MAX_CODED_BITS) nSortEntries = N_MAX_CODED_BITS;
    /*VCAST_DONT_INSTRUMENT_END*/

    return nSortEntries;
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

//--------------------------------------------------------------------------------------------------------
// Helpers for Bit set/clear/move
__device__ __forceinline__ void atomicSetBit(uint32_t* pWords, uint32_t bitPos)
{
    uint32_t bitMsk = 1U << getBitPosIdxInWord(bitPos);
    atomicOr(&pWords[getBitPosWordIdx(bitPos)], bitMsk);
}

__device__ __forceinline__ void atomicClrBit(uint32_t* pWords, uint32_t bitPos)
{
    uint32_t bitMsk = 1U << getBitPosIdxInWord(bitPos);
    atomicAnd(&pWords[getBitPosWordIdx(bitPos)], ~bitMsk);
}

__device__ __forceinline__ void atomicMoveBit(uint32_t const* pSrcWords, uint32_t srcBitPos, uint32_t* pDstWords, uint32_t dstBitPos)
{
    if(getBit(pSrcWords, srcBitPos))
    {
        atomicSetBit(pDstWords, dstBitPos);
    }
    else
    {
        atomicClrBit(pDstWords, dstBitPos);
    }
}

__device__ __inline__ void xorBfly(thread_block const& thisThrdBlk, uint32_t nCodedBits, uint32_t* pCodedWords)
{
    uint32_t thrdIdxInBlk = thisThrdBlk.thread_rank();

    bool thrdEnable = (thrdIdxInBlk < nCodedBits / 2);

    // Need to operate at word granularity since atomicOr does not support bytewise Or
    for(uint32_t size = 2; size <= nCodedBits; size <<= 1)
    {
        uint32_t stride  = size / 2;
        uint32_t idx     = 2 * thrdIdxInBlk - (thrdIdxInBlk & (stride - 1));
        uint32_t bitPos1 = idx + 0;
        uint32_t bitPos2 = idx + stride;

        uint32_t bitIdx1  = bitPos1 % 32;
        uint32_t wordIdx1 = bitPos1 / 32;

        uint32_t bitIdx2  = bitPos2 % 32;
        uint32_t wordIdx2 = bitPos2 / 32;

        if(thrdEnable)
        {
            uint8_t  resBitVal = ((pCodedWords[wordIdx1] >> bitIdx1) & 0x1) ^ ((pCodedWords[wordIdx2] >> bitIdx2) & 0x1);
            uint32_t bitMsk    = 1U << bitIdx1;
            if(resBitVal)
            {
                // atomic bit set
                atomicOr(&pCodedWords[wordIdx1], bitMsk);
            }
            else
            {
                // atomic bit clear
                atomicAnd(&pCodedWords[wordIdx1], ~bitMsk);
            }

#ifdef ENABLE_DEBUG
            printf("ThrdIdx[%03d], Stride[%02d]: bitPos1 %d, bitPos2 %d, bitMsk 0x%08x, resBitVal 0b%d, CodedWords[%d] = 0x%08x\n", thrdIdxInBlk, stride, bitPos1, bitPos2, bitMsk, resBitVal, wordIdx1, pCodedWords[wordIdx1]);
#endif
        }
        __syncthreads();

#ifdef ENABLE_DEBUG
        uint32_t nCodedWords = nCodedBits / N_BITS_PER_WORD;
        if(thrdIdxInBlk < nCodedWords)
        {
            printf("Stride[%02d]: bitPos1 %d, bitPos2 %d, CodedWords[%d] = 0x%08x\n", stride, bitPos1, bitPos2, thrdIdxInBlk, pCodedWords[thrdIdxInBlk]);
        }
#endif
    }
}

//--------------------------------------------------------------------------------------------------------
#if 0
// Sorting
enum class SortDir_t : uint8_t
{
    ASCENDING  = 0,
    DESCENDING = 1
};


template <typename T_VAL>
__device__ __forceinline__ void sortComparator(SortDir_t dir, T_VAL& valA, T_VAL& valB)
{
    if(static_cast<SortDir_t>(valA > valB) == dir)
    {
        return;
    }

    // swap(valA, valB)
    T_VAL val = valA;
    valA      = valB;
    valB      = val;
}

// Reference: CUDA sample bitonicSortShared
// sortDir      - sort direction (ascending vs descending)
// nSortEntries - number of elements to be sorted, needs to be a power of 2. Also need blockDim.x*blockDim.y >= nItems/2
// pShEntries     - Pointer to entries to be sorted inplace in shared memory
template <typename Tentry>
__device__ void bitonicSort(SortDir_t sortDir, uint32_t nSortEntries, Tentry* pShEntries)
{
    thread_block const& thisThrdBlk = this_thread_block();
    uint32_t            tIdx        = thisThrdBlk.thread_rank();

    // Each thread implements a comparator which compares 2 elements in the input sequence. Thus a max of
    // (nSortEntries/2) threads are needed in the thread block. All comparisons occur in parllel.
    // Multiple such parallel comparisons are composed to form a comparator stage and multiple comparator stages
    // form the comparator network.
    // There are log2(nSortEntries) comparator stages with stage-n perfomring n*(nSortEntries/2) parallel comparisons
    // Consequently nSortEntries is constrained to be a power of 2

    bool thrdEnable = (tIdx < nSortEntries / 2);

    // Within each comparator stage, the first parallel comparison set starts with the widest possible stride
    // for that stage (i.e. stride = size/2) with subsequent stages narrowing strides down to stride = 1

    //---------------------------------------------------------------------------
    // Push the data through the sorting network
    // for loop below implements a comparator network potentially made of several comparator stages
    for(uint32_t size = 2; size < nSortEntries; size <<= 1)
    {
        // Bitonic sort divides input sequence into 2 halves: threads sorting the Lower half sort in
        // non-increasing order and those sorting the Upper half sort in non-decreasing order

        // Based on the value of size, the sort direction is held steady for the first (size/2) elements after
        // which the direction is flipped.

        // Bitonic merge
        SortDir_t dir = static_cast<SortDir_t>(((tIdx & (size / 2)) != 0));

        //---------------------------------------------------------------------------
        // for loop below implements a single comparator stage where (nSortEntries/2) parallel comparisons are
        // performed. Stride specifies the distance between the two elements in the sequence being compared
        for(uint32_t stride = size / 2; stride > 0; stride >>= 1)
        {
            if(thrdEnable)
            {
                // Since stride is a power of 2, (stride - 1) acts as a bitmask to modulo the thread index to range [0, stride-1]
                // Effectively 2 * tIdx - (tIdx & (stride - 1)) generates indices so that the input indices skip by stride for contiguous thread indices

                // idx + stride generates indices of the second input to the comparator
                uint32_t idx       = 2 * tIdx - (tIdx & (stride - 1));
                uint32_t entry1Idx = idx + 0;
                uint32_t entry2Idx = idx + stride;

                sortComparator<Tentry>(dir,
                                       pShEntries[entry1Idx],
                                       pShEntries[entry2Idx]);
            }
            thisThrdBlk.sync();
        }
    }

    // The sort direction for the last comparator stage is the same as the sort direction
    {
        for(uint32_t stride = nSortEntries / 2; stride > 0; stride >>= 1)
        {
            if(thrdEnable)
            {
                uint32_t idx       = 2 * tIdx - (tIdx & (stride - 1));
                uint32_t entry1Idx = idx + 0;
                uint32_t entry2Idx = idx + stride;

                sortComparator<Tentry>(sortDir,
                                       pShEntries[entry1Idx],
                                       pShEntries[entry2Idx]);
            }
            thisThrdBlk.sync();
        }
    }
}
#endif

//--------------------------------------------------------------------------------------------------------
// Polar encoding
__device__ void encode(uint32_t nInfoBits, uint32_t nCodedBits, uint32_t nTxBits, uint8_t const* pInfoBits, uint32_t* pSmem, uint8_t* pCodedBits, uint32_t procModeBmsk)
{
    thread_block const& thisThrdBlk = this_thread_block();

    // 1 tile per warp
    thread_block_tile<N_THRDS_PER_TILE> const& thisThrdTile =
        tiled_partition<N_THRDS_PER_TILE>(thisThrdBlk);

    uint32_t thrdIdxInBlk  = thisThrdBlk.thread_rank();
    uint32_t thrdIdxInTile = thisThrdTile.thread_rank();
    uint32_t nThrdTiles    = thisThrdTile.size();
    uint32_t tileIdx       = thrdIdxInBlk / nThrdTiles;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory carve out
    int32_t* pTileStartOffsets    = reinterpret_cast<int32_t*>(pSmem);
    int16_t* pRelSeqIdxsPruned    = reinterpret_cast<int16_t*>(&pTileStartOffsets[N_MAX_THRD_TILES]);
    uint8_t* pInvInterleaverIdxs  = reinterpret_cast<uint8_t*>(&pRelSeqIdxsPruned[REL_SEQ_IDX_BUF_LEN]);
    uint8_t* pInterleavedInfoBits = &pInvInterleaverIdxs[N_MAX_INFO_BITS];
    uint8_t* pCodedBitIdxs        = &pInterleavedInfoBits[N_MAX_INFO_BITS];

    uint32_t        relSeqLutIdx = __ffs(nCodedBits) - __ffs(N_MIN_CODED_BITS);
    uint16_t const* pRelSeqLut   = POLAR_REL_SEQ_IDXS_LUT_PTR[relSeqLutIdx];

    //--------------------------------------------------------------------------------------------------------
    // Forbidden index interval computation

    // Initialize to invalid values
    int16_t intervalStart[3] = {N_MAX_CODED_BITS, N_MAX_CODED_BITS, N_MAX_CODED_BITS};
    int16_t intervalEnd[3]   = {-1, -1, -1};
    if(nTxBits < nCodedBits)
    {
        int32_t blkLen   = nCodedBits / N_MIN_CODED_BITS;
        int32_t nBlks    = (nCodedBits - nTxBits) / blkLen;
        int32_t nRemBits = (nCodedBits - nTxBits) - (nBlks * blkLen);

        constexpr float INFO_TX_BITS_RATIO_THD = (7.0f / 16.0f);
        if((static_cast<float>(nInfoBits) / static_cast<float>(nTxBits)) <= INFO_TX_BITS_RATIO_THD)
        {
            intervalStart[0] = blkLen * (POLAR_REL_SEQ_FORBID_IDXS_FWD[0][nBlks] - 1);
            intervalEnd[0]   = (blkLen * POLAR_REL_SEQ_FORBID_IDXS_FWD[1][nBlks]) - 1;
            intervalStart[1] = blkLen * (POLAR_REL_SEQ_FORBID_IDXS_FWD[2][nBlks] - 1);
            intervalEnd[1]   = blkLen * (POLAR_REL_SEQ_FORBID_IDXS_FWD[3][nBlks] - 1) - 1 + nRemBits;

            intervalStart[2] = 0;

            // Note: nCodedBits is a multiple of 32, thus 3*nCodedBits/4 is still an integer and a multiple of 8
            uint32_t threshold1 = 3 * nCodedBits / 4;
            if(nTxBits >= threshold1)
            {
                intervalEnd[2] = static_cast<int16_t>(ceilf(static_cast<float>(threshold1) - (static_cast<float>(nTxBits) / 2.0f))) - 1;
            }
            else
            {
                // Note: threshold1 is a multiple of 8, thus 3*threshold1/4 is still an integer
                intervalEnd[2] = static_cast<int16_t>(ceilf(static_cast<float>(3 * threshold1 / 4) - (static_cast<float>(nTxBits) / 4.0f))) - 1;
            }
        }
        else
        {
            intervalStart[0] = blkLen * (POLAR_REL_SEQ_FORBID_IDXS_BWD[0][nBlks] - 1);
            intervalEnd[0]   = (blkLen * POLAR_REL_SEQ_FORBID_IDXS_BWD[1][nBlks]) - 1;
            intervalStart[1] = (blkLen * POLAR_REL_SEQ_FORBID_IDXS_BWD[2][nBlks]) - nRemBits;
            intervalEnd[1]   = (blkLen * POLAR_REL_SEQ_FORBID_IDXS_BWD[3][nBlks]) - 1;
        }
    }

#ifdef ENABLE_DEBUG
    if(0 == thrdIdxInBlk)
    {
        printf("Forbidden index intervals: Interval0: [%d, %d], Interval1: [%d, %d], Interval2: [%d, %d]\n", intervalStart[0], intervalEnd[0], intervalStart[1], intervalEnd[1], intervalStart[2], intervalEnd[2]);
    }
#endif

    //--------------------------------------------------------------------------------------------------------
    // Prune out the forbidden indices from the reliability sequence

    // predicate value for stream compaction that follows
    bool pred = false;
    if(thrdIdxInBlk < nCodedBits)
    {
        int16_t const& lutRelSeqIdx = pRelSeqLut[thrdIdxInBlk];
        bool           isForbidden  = (((lutRelSeqIdx >= intervalStart[0]) && (lutRelSeqIdx <= intervalEnd[0])) ||
                            ((lutRelSeqIdx >= intervalStart[1]) && (lutRelSeqIdx <= intervalEnd[1])) ||
                            ((lutRelSeqIdx >= intervalStart[2]) && (lutRelSeqIdx <= intervalEnd[2]))) ?
                               true :
                               false;

        if(!isForbidden) pred = true;
    }

    uint32_t nActiveThrdTiles = div_round_up(nCodedBits, N_THRDS_PER_TILE);
    int32_t  thrdOffset       = 0;
    strmCompactionHelper<N_THRDS_PER_TILE>(thisThrdBlk, thisThrdTile, pred, nActiveThrdTiles, pTileStartOffsets, thrdOffset);

    if(pred)
    {
        int32_t prunedRelSeqIdx = pTileStartOffsets[tileIdx] + thrdOffset;
        // pRelSeqIdxsPruned[prunedRelSeqIdx] = pRelSeqIdxs[thrdIdxInBlk];

        // Store the pruned reliability sequence indices indices in reflected form (most reliable to least
        // reliable indices). This way the first nInfoBits indices are the most reliable indices. Note that
        // the indices are stored towards the end of the buffer
        // printf("threadIdx (%d %d %d) blockIdx (%d %d %d): prunedRelSeqIdx %d thrdIdxInBlk %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        pRelSeqIdxsPruned[(REL_SEQ_IDX_BUF_LEN - 1) - prunedRelSeqIdx] = pRelSeqLut[thrdIdxInBlk];
    }

    // Total number of reliability sequence indices available
    uint32_t prunedRelSeqLen = __syncthreads_count(pred);


    // Indices are stored in the last prunedRelSeqLen locations of the buffer
    int32_t prunedRelSeqStartIdx = REL_SEQ_IDX_BUF_LEN - prunedRelSeqLen;

#ifdef ENABLE_DEBUG
    if(thrdIdxInBlk < prunedRelSeqLen)
    {
        printf("RelSeqIdxsPruned[%d] = %d\n", thrdIdxInBlk, pRelSeqIdxsPruned[prunedRelSeqStartIdx + thrdIdxInBlk]);
    }
#endif

    //--------------------------------------------------------------------------------------------------------
    // Sort the reliability sequence indices

    // Bitonic sort requires the sequence to be a power of 2, pad the pruned array with -1s so that the padded
    // array length is a power of 2.
    int32_t nSortEntries  = getNumBitonicSortEntries(nInfoBits);
    int32_t nEntriesToPad = nSortEntries - nInfoBits;

    // Since reliability sequence indices are stored towards the end of the buffer, pad to the beginning of
    // the buffer
    /*if ((prunedRelSeqStartIdx - nEntriesToPad) < 0) {
        printf("threadIdx.x %d, blockIdx.x %d, blockIdx.y %d, prunedRelSeqLen %d, prunedRelSeqStartIdx %d, nEntriesToPad %d\n",
        threadIdx.x, blockIdx.x, blockIdx.y, prunedRelSeqLen, prunedRelSeqStartIdx, nEntriesToPad);
    }*/

    int16_t* pRelSeqIdxsPrunedPadded = &pRelSeqIdxsPruned[prunedRelSeqStartIdx - nEntriesToPad];

    if(thrdIdxInBlk < nEntriesToPad)
    {
        pRelSeqIdxsPrunedPadded[thrdIdxInBlk] = -1;
    }

    // initialize the coded bit array
    uint32_t nCodedBytes = nCodedBits / N_BITS_PER_BYTE;
    if(thrdIdxInBlk < nCodedBytes)
    {
        // printf("thrdIdxInBlk: %d, Thd :%d, nCodedBits: %d, sizeof(pCodedBits[0]): %d\n", thrdIdxInBlk, (nCodedBits / sizeof(pCodedBits[0])), nCodedBits, sizeof(pCodedBits[0]));
        pInterleavedInfoBits[thrdIdxInBlk] = 0;
        pCodedBits[thrdIdxInBlk]           = 0;
    }

    // Initialize coded bit indices as parity bits
    // set parity bit index value to be 1 more than max info bits
    static constexpr uint8_t PARITY_BIT_IDX = N_MAX_INFO_BITS + 1;
    if(thrdIdxInBlk < nCodedBits)
    {        
        // Initialize by marking all coded bits as parity bits
        pCodedBitIdxs[thrdIdxInBlk] = PARITY_BIT_IDX;
    }

    __syncthreads();

#ifdef ENABLE_DEBUG
    if(0 == thrdIdxInBlk) printf("prunedRelSeqLen %d nEntriesToPad %d nSortEntries %d\n", prunedRelSeqLen, nEntriesToPad, nSortEntries);

    if(thrdIdxInBlk < prunedRelSeqLen)
    {
        printf("RelSeqIdxsPruned[%d] = %d\n", thrdIdxInBlk, pRelSeqIdxsPruned[prunedRelSeqStartIdx + thrdIdxInBlk]);
    }
#endif

    // Inplace sort
    bitonicSort(SortDir_t::ASCENDING, nSortEntries, pRelSeqIdxsPrunedPadded);

    //--------------------------------------------------------------------------------------------------------
    // Interleave reliability sequence indices (indices are interleaved instead of info bits themselves)
    // skip entries padded with -1s

    // Skip padding
    int16_t* pRelSeqIdxsPrunedSorted = &pRelSeqIdxsPruned[prunedRelSeqStartIdx];

#ifdef ENABLE_DEBUG
    if(thrdIdxInBlk < nInfoBits)
    {
        printf("RelSeqIdxsPrunedSorted[%d] = %d\n", thrdIdxInBlk, pRelSeqIdxsPrunedSorted[thrdIdxInBlk]);
    }
#endif
    computeInterleaverIdxs<N_THRDS_PER_TILE>(thisThrdBlk, thisThrdTile, nInfoBits, nCodedBits, pTileStartOffsets, pInvInterleaverIdxs, procModeBmsk);

    // Save the information bit index in the reliability sequence index location to be used later in placing
    // info bits at respective reliability sequence index locations
    if(thrdIdxInBlk < nInfoBits)
    {
        // Apply bit interleaving to reliability sequence indices (convenient instead of interleaving information bits)
        //  sensitive_memory_access fixing
        int32_t interleaverIdxTblOffset  = pInvInterleaverIdxs[thrdIdxInBlk];
        int16_t interleavedRelSeqIdx = pRelSeqIdxsPrunedSorted[interleaverIdxTblOffset];
#if 0
        if ((interleavedRelSeqIdx < 0) || (interleavedRelSeqIdx >= N_MAX_CODED_BITS)) {
            printf("threadIdx.x %d, block %d, %d, interleavedRelSeqIdx invalid %d, max %d\n",
                   threadIdx.x, blockIdx.x, blockIdx.y, interleavedRelSeqIdx, N_MAX_CODED_BITS);
        }
#endif
        // Record position of information bit in the coded bit sequence
        pCodedBitIdxs[interleavedRelSeqIdx] = thrdIdxInBlk;

#ifdef ENABLE_DEBUG
        uint32_t srcBitPos = thrdIdxInBlk;
        printf("CodedBitIdxs[%03d] = %03d, Bit[%03d] = 0b%d\n", interleavedRelSeqIdx, pCodedBitIdxs[interleavedRelSeqIdx], thrdIdxInBlk, getBit(pInfoBits, srcBitPos));
#endif
    }
    __syncthreads();

    //--------------------------------------------------------------------------------------------------------
    // Form coded bit sequence placing info bits in locations specified by reliability sequence indices and
    // zeros elsewhere (i.e. at parity bit locations)
    // Note: nCodedBits is a multiple of 32
    uint32_t  nCodedWords = nCodedBits / N_BITS_PER_WORD;
    uint32_t* pCodedWords = reinterpret_cast<uint32_t*>(pCodedBits);
    if(tileIdx < nCodedWords)
    {
        // Place a 0 in coded bit sequence if predicate is false
        // Place a 1 in coded bit sequence if predicate is true

        // Per above a false predidcate is assigned to all parity bits and info bits which are 0
        bool pred = false;
        if(PARITY_BIT_IDX != pCodedBitIdxs[thrdIdxInBlk])
        {
            // Extract input information bit and place it in position given by reliability index
            // nInfoBits need not be a multiple of 32 and hence access at the smallest possible unit i.e. in
            // bytes
            uint32_t srcBitPos = pCodedBitIdxs[thrdIdxInBlk];
            pred               = getBit(pInfoBits, srcBitPos);

#ifdef ENABLE_DEBUG
            printf("InfoBit[%d] = 0b%d\n", srcBitPos, static_cast<uint8_t>(pred));
#endif
        }

        // Assemble information and parity bits into coded words
        pCodedWords[tileIdx] = __ballot_sync(FULL_WARP_ACTIVE_BMSK, pred);
    }

    // Wait for all encoded words to be computed
    __syncthreads();

#ifdef ENABLE_DEBUG
    if(thrdIdxInBlk < nCodedWords)
        printf("xorBfly input: CodedWords[%d] = 0x%08x\n", thrdIdxInBlk, pCodedWords[thrdIdxInBlk]);
#endif

    //--------------------------------------------------------------------------------------------------------
    // Butterfly xor
    xorBfly(thisThrdBlk, nCodedBits, pCodedWords);
}

//--------------------------------------------------------------------------------------------------------
// Polar rate-matching
// Note: pTxBits needs to be word alinged and padded to a multiple of word length
__device__ void rateMatch(uint32_t nInfoBits, uint32_t nCodedBits, uint32_t nTxBits, uint8_t const* pInCodedBits, uint32_t* pSmem, uint8_t* pTxBits)
{
    thread_block const& thisThrdBlk  = this_thread_block();
    uint32_t            thrdIdxInBlk = thisThrdBlk.thread_rank();
    uint32_t            nThrds       = thisThrdBlk.size();

    //--------------------------------------------------------------------------------------------------------
    // Sub-block interleaving
    // Reference: 3GPP TS 38.212, section 5.4.1.1, Sub-block interleaving
    // The coded bits are divided into 32 sub-blocks. The sub-block size is [1,2,4,8,16] bits respectively for
    // nCodedBits values [32,64,128,256,512]. These sub-blocks are interleaved
    uint8_t scaleFactor = nCodedBits / N_MIN_CODED_BITS;
    uint8_t nCodedBytes = nCodedBits / N_BITS_PER_BYTE;

    // Bit selection uses interleaved coded bits
    uint32_t const* pInCodedWords          = reinterpret_cast<uint32_t const*>(pInCodedBits);
    uint32_t*       pInterleavedCodedWords = pSmem;
    uint8_t*        pInterleavedCodedBits  = reinterpret_cast<uint8_t*>(pSmem);
    // Since nThrdsInBlk >= nCodedBits, each thread may be used to interleave one coded bit
    if(thrdIdxInBlk < nCodedBits)
    {
        uint32_t interleaverTblIdx = thrdIdxInBlk / scaleFactor;
        uint32_t interleavedBitPos = (POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[interleaverTblIdx] * scaleFactor) +
                                     (thrdIdxInBlk % scaleFactor);

        uint32_t srcBitPos = interleavedBitPos;
        uint32_t dstBitPos = thrdIdxInBlk;
        atomicMoveBit(pInCodedWords, srcBitPos, pInterleavedCodedWords, dstBitPos);
                
#ifdef ENABLE_DEBUG
        printf("polarRateMatch Wr: dstBitPos %d, dstByteIdx %d, dstWordIdx %d\n", srcBitPos, getBitPosByteIdx(dstBitPos), getBitPosWordIdx(dstBitPos));
        printf("interleavedBitPos/srcBitPos %d dstBitPos %d interleaverTblIdx %d \n", interleavedBitPos, dstBitPos, interleaverTblIdx);
#endif
    }

    __syncthreads();

    // Note that the following also assumes that nCodedBits is a multiple of 32
    static_assert((N_MIN_CODED_BITS == N_BITS_PER_WORD), "Number of coded bits assumed to be >= 32");
    uint32_t nCodedWords = nCodedBits / N_BITS_PER_WORD;

#ifdef ENABLE_DEBUG
    if(thrdIdxInBlk < nCodedWords)
        printf("CodedWords[%d]: input 0x%08x interleaved 0x%08x\n", thrdIdxInBlk, pInCodedWords[thrdIdxInBlk], pInterleavedCodedWords[thrdIdxInBlk]);
#endif

    //--------------------------------------------------------------------------------------------------------
    // Bit selection
    // Reference: 3GPP TS 38.212, Section 5.4.1.2, Bit selection

    uint32_t* pTxWords    = reinterpret_cast<uint32_t*>(pTxBits);
    uint32_t  nTxWords    = nTxBits / N_BITS_PER_WORD;
    uint32_t  nTxRemBytes = (nTxBits % N_BITS_PER_WORD) / N_BITS_PER_BYTE;
    uint32_t  nTxRemBits  = ((nTxBits % N_BITS_PER_WORD) % N_BITS_PER_BYTE);

    // Repetition
    if(nTxBits >= nCodedBits)
    {
        // Marshal the bits in word sized chunks
        if(thrdIdxInBlk < nTxWords)
        {
            pTxWords[thrdIdxInBlk] = pInterleavedCodedWords[thrdIdxInBlk % nCodedWords];

#ifdef ENABLE_DEBUG
            printf("TxWord[%03d]: 0x%08x\n", thrdIdxInBlk, pTxWords[thrdIdxInBlk]);
#endif
        }

        // Marshal rest of the bits (which fit in byte sized chunks) in byte units
        uint32_t startByteOffset = nTxWords * N_BYTES_PER_WORD;
        if(thrdIdxInBlk < nTxRemBytes)
        {
            uint32_t byteIdx = startByteOffset + thrdIdxInBlk;
            pTxBits[byteIdx] = pInterleavedCodedBits[byteIdx % nCodedBytes];

#ifdef ENABLE_DEBUG
            printf("TxBytes[%03d]: 0x%02x\n", byteIdx, pTxBits[byteIdx]);
#endif
        }

        // Marshal remaining in bits
        uint32_t byteIdx = startByteOffset + nTxRemBytes;

        // Pick thread with rank 0
        if((nTxRemBits > 0) && (0 == thrdIdxInBlk))
        {
            // nTxRemBits is expected to be < 8bits (if > 8 bits, it would already be moved by the byte
            // movement section above)
            uint8_t bmsk     = (1U << nTxRemBits) - 1;
            pTxBits[byteIdx] = pInterleavedCodedBits[byteIdx] & bmsk;

#ifdef ENABLE_DEBUG
            printf("InterleavedCodedBits[%03d]: 0x%02x bmsk %0x08 startByteOffset %d nTxRemBytes %d\n", byteIdx, pInterleavedCodedBits[byteIdx], bmsk, startByteOffset, nTxRemBytes);
            printf("TxBytes[%03d]: 0x%02x\n", byteIdx, pTxBits[byteIdx]);
#endif
        }
    }
    // nTxBits < nCodedBits
    else
    {
        // Shortening: From nCodedBits, select the first nTxBits
        uint32_t srcStartBitPos = 0;

        // Puncturing: From nCodedBits, select the last nTxBits
        constexpr float INFO_TX_BITS_RATIO_THD = (7.0f / 16.0f);
        if((static_cast<float>(nInfoBits) / static_cast<float>(nTxBits)) <= INFO_TX_BITS_RATIO_THD)
        {
            srcStartBitPos = nCodedBits - nTxBits;
        }

        // if(0 == thrdIdxInBlk) printf("polarRateMatch: srcStartBitPos %d nCodedBits %d nTxBits %d\n", srcStartBitPos, nCodedBits, nTxBits);

        // If we are here then nCodedBits > nTxBits. Since nThrdsInBlk >= nCodedBits, there must be enough
        // threads to move all info bits with one thread per bit
        if(thrdIdxInBlk < nTxBits)
        {
            uint32_t srcBitPos = srcStartBitPos + thrdIdxInBlk;
            uint32_t dstBitPos = thrdIdxInBlk;
            atomicMoveBit(pInterleavedCodedWords, srcBitPos, pTxWords, dstBitPos);            
            // printf("polarRateMatch Rd: srcBitPos %d, srcByteIdx %d, srcWordIdx %d\n", srcBitPos, getBitPosByteIdx(srcBitPos), getBitPosWordIdx(srcBitPos));
        }
    }
}

//--------------------------------------------------------------------------------------------------------
// Encode rate match kernel entry
// nInfoBits   - # of information bits to be encoded
// nCodedBits  - # of Polar encoded bits
// nTxBits     - # of rate matched bits for transmission
// pInfoBits   - Input , Pointer to storge of information bits (array of bytes with atleast uint32_t alignment)
// pCodedBits  - Output, Pointer to storge of coded bits for debub (array of bytes with atleast uint32_t alignment)
// pTxBits     - Output, Pointer to storage of transmit bits (array of bytes with atleast uint32_t alignment)
__global__ void encodeRateMatchKernel(uint32_t nInfoBits, uint32_t nCodedBits, uint32_t nTxBits, uint8_t const* pInfoBits, uint8_t* pCodedBits, uint8_t* pTxBits, uint32_t procModeBmsk)
{
    // invInterleaverIdxs - interleaver index inverse map: offsets of interleaver indices within the

    //    object              overlaid             size
    // tileStartOffsets          no       N_MAX_THRD_TILES*sizeof(int32_t)
    // relSeqIdxsPruned          no       REL_SEQ_IDX_BUF_LEN*sizeof(int16_t)
    // invInterleaverIdxs        no       N_MAX_INFO_BITS*sizeof(uint8_t)
    // interleavedInfoBits       no       N_MAX_INFO_BITS*sizeof(uint8_t)
    // codedBitIdxs              no       N_MAX_CODED_BITS*sizeof(uint8_t)     // array containing the index of info bit or parity bit to be placed in a coded bit location

    // smemRateMatch            yes       (N_MAX_CODED_BITS/8)*sizeof(uint8_t) // overlaid with invInterleaverIdxs

    constexpr uint32_t N_SMEM_ELEMS =
        N_MAX_THRD_TILES * sizeof(int32_t) + REL_SEQ_IDX_BUF_LEN * sizeof(int16_t) +
        N_MAX_INFO_BITS * sizeof(uint8_t) + N_MAX_INFO_BITS * sizeof(uint8_t) +
        N_MAX_CODED_BITS * sizeof(uint8_t);

    __shared__ __align__(sizeof(uint32_t)) uint8_t smemBlk[N_SMEM_ELEMS];
    uint32_t*                                      pSmem = reinterpret_cast<uint32_t*>(smemBlk);

    /*if ((threadIdx.x==0) && (blockIdx.x == 0) && (threadIdx.y == 0)) {
        int max_bytes = (nInfoBits/8) + (int)(nInfoBits % 8 == 1);
        for (int i = 0; i < max_bytes; i++) {
            printf("byte %d info 0x%x\n", i, pInfoBits[i]);
        }
    }*/
    /*if ((threadIdx.x==0) && (blockIdx.x == 0) && (threadIdx.y == 0))
        printf("nInfoBits %d, coded bits %d, tx bits %d\n", nInfoBits, nCodedBits, nTxBits); */
    encode(nInfoBits, nCodedBits, nTxBits, pInfoBits, pSmem, pCodedBits, procModeBmsk);
    rateMatch(nInfoBits, nCodedBits, nTxBits, pCodedBits, pSmem, pTxBits);
}

__global__ void encodeRateMatchMultipleDCIsKernel(uint8_t const* pInfoBits, uint8_t* pCodedBits, uint8_t* pTxBits, cuphyPdcchDciPrm_t* pPdcchDciParams, uint8_t* pDciTmInfo)
{
    constexpr uint32_t N_SMEM_ELEMS =
        N_MAX_THRD_TILES * sizeof(int32_t) + 2*N_MAX_CODED_BITS * sizeof(int16_t) +
        N_MAX_INFO_BITS * sizeof(uint8_t) + N_MAX_INFO_BITS * sizeof(uint8_t) +
        N_MAX_CODED_BITS * sizeof(uint8_t);
    // Added check in PDCCH setup prep. function to ensure this kernel is never launched if there is a DCI in !testing_mode with invalid payload size,
    // to avoid illegal accesses to shared memory.

    __shared__ __align__(sizeof(uint32_t)) uint8_t smemBlk[N_SMEM_ELEMS];
    uint32_t*                                      pSmem = reinterpret_cast<uint32_t*>(smemBlk);

    const int DCI_id = blockIdx.x;
    cuphyPdcchDciPrm_t& dci_params = pPdcchDciParams[DCI_id];

    const uint32_t offset = DCI_id * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC;
    const uint32_t d_x_coded_offset = DCI_id * (CUPHY_POLAR_ENC_MAX_CODED_BITS / 8);
    const uint32_t tx_offset = DCI_id * (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8); // offset in bytes

    const uint32_t nInfoBits = CUPHY_PDCCH_N_CRC_BITS + dci_params.Npayload;
    const uint32_t nTxBits = 2 * 9 * 6 * dci_params.aggr_level;
    const uint32_t roundUpToPow2_nTxBits = 2 * 64 * dci_params.aggr_level; // Reminder: possible aggregation level values {1, 2, 4, 8, 16}

    // Find if this DCI is in testing mode or not. If it is, perform D2D copy of payload,
    // skipping encoding and rate-matching, and exit.
    uint8_t testing_mode = (pDciTmInfo[DCI_id >> 3] >> (DCI_id & 0x7)) & 0x1;
    if (testing_mode != 0) //Could also check byte first and then narrow down if needed
    {
        if ((threadIdx.y == 0) && (threadIdx.x < N_MAX_TM_DCI_TX_BYTES)) {
            // Assumes byte padding which is covered since a 24-bit 0 CRC is added to the 108 bits of the PN sequence
            // in cuphyPdcchPipelinePrepare()
            pTxBits[tx_offset + threadIdx.x] = pInfoBits[offset + threadIdx.x];
        }
        return;
    }

    //--------------------------------------------------------------------------------------------------------
    // Compute # of coded bits (see section "5.3.1 Polar coding " in 3GPP TS 38.212)

    uint32_t        nMin1CodedBits         = roundUpToPow2_nTxBits / 2;
    constexpr float INFO_TX_BITS_RATIO_THD = (9.0f / 16.0f);
    if((nTxBits > (9 * nMin1CodedBits) / 8) ||
       ((static_cast<float>(nInfoBits) / static_cast<float>(nTxBits)) >= INFO_TX_BITS_RATIO_THD))
    {
        nMin1CodedBits *= 2;
    }

    // Min number of coded bits possible given the number of info bits and min code rate
    uint32_t nMin2CodedBits = roundUpToPow2Gpu(nInfoBits * MIN_CODE_RATE_INV);
    uint32_t nMinCodedBits = (nMin1CodedBits < nMin2CodedBits) ? nMin1CodedBits : nMin2CodedBits;

    uint32_t nCodedBits = nMinCodedBits;
    if(nCodedBits < N_MIN_CODED_BITS) nCodedBits = N_MIN_CODED_BITS;
    if(nCodedBits > N_MAX_CODED_BITS) nCodedBits = N_MAX_CODED_BITS;

    // DCI sent over Downlink Channel, Mask set to 0
    uint32_t procModeBmsk = 0;

    // Call device functions for DCIs that are not in TM
    encode(nInfoBits, nCodedBits, nTxBits, pInfoBits + offset, pSmem, pCodedBits + d_x_coded_offset, procModeBmsk);
    rateMatch(nInfoBits, nCodedBits, nTxBits, pCodedBits + d_x_coded_offset, pSmem, pTxBits + tx_offset);
}

void encodeRateMatch(uint32_t nInfoBits, uint32_t nTxBits, uint8_t const* pInfoBits, uint32_t* pNCodedBits, uint8_t* pCodedBits, uint8_t* pTxBits, uint32_t procModeBmsk, cudaStream_t strm)
{
    //--------------------------------------------------------------------------------------------------------
    // Compute # of coded bits (see section "5.3.1 Polar coding " in 3GPP TS 38.212)

    uint32_t        nMin1CodedBits         = roundUpToPow2Cpu(nTxBits) / 2;
    constexpr float INFO_TX_BITS_RATIO_THD = (9.0f / 16.0f);
    if((nTxBits > (9 * nMin1CodedBits) / 8) ||
       ((static_cast<float>(nInfoBits) / static_cast<float>(nTxBits)) >= INFO_TX_BITS_RATIO_THD))
    {
        nMin1CodedBits *= 2;
    }


    // Min number of coded bits possible given the number of info bits and min code rate
    uint32_t nMin2CodedBits = roundUpToPow2Cpu(nInfoBits * MIN_CODE_RATE_INV);

    uint32_t nMinCodedBits = (nMin1CodedBits < nMin2CodedBits) ? nMin1CodedBits : nMin2CodedBits;

    uint32_t nCodedBits = nMinCodedBits;
    if(nCodedBits < N_MIN_CODED_BITS) nCodedBits = N_MIN_CODED_BITS;
    if(nCodedBits > N_MAX_CODED_BITS) nCodedBits = N_MAX_CODED_BITS;

    //--------------------------------------------------------------------------------------------------------
    // Kernel launch
    dim3 gridDim(1);
    dim3 blockDim(N_THRDS_PER_TILE, N_MAX_THRD_TILES);

#ifdef ENABLE_DEBUG
    printf("PolarEncoder: nCodedBits %d nMin1CodedBits %d nMin2CodedBits %d nMinCodedBits %d\n",
           nCodedBits,
           nMin1CodedBits,
           nMin2CodedBits,
           nMinCodedBits);
    printf("PolarEncoder: nInfoBits %d nCodedBits %d nTxBits %d N_THRDS_PER_TILE %d N_MAX_THRD_TILES %d\n",
           nInfoBits,
           nCodedBits,
           nTxBits,
           N_THRDS_PER_TILE,
           N_MAX_THRD_TILES);
#endif

    *pNCodedBits = nCodedBits;
    encodeRateMatchKernel<<<gridDim, blockDim, 0, strm>>>(nInfoBits, nCodedBits, nTxBits, pInfoBits, pCodedBits, pTxBits, procModeBmsk);
}

cuphyStatus_t kernelSelectEncodeRateMatchMultiDCIs(cuphyEncoderRateMatchMultiDCILaunchCfg_t* pLaunchCfg,
                                                   uint32_t                                  num_DCIs)
{
    if (pLaunchCfg == nullptr) return CUPHY_STATUS_INVALID_ARGUMENT;

    if (pLaunchCfg->kernelNodeParamsDriver.func == nullptr) {
        // kernel (only one kernel option for now) and without any specialization, so update function only once
        // func is set to nullptr in PDCCH channel constructor
        void* kernelFunc = reinterpret_cast<void*>(encodeRateMatchMultipleDCIsKernel);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    }

    // launch geometry (can change!)
    dim3 gridDim(num_DCIs);
    dim3 blockDim(N_THRDS_PER_TILE, N_MAX_THRD_TILES);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;

    return CUPHY_STATUS_SUCCESS;
}


__global__ void encodeRateMatchMultipleSSBsKernel(uint8_t const* pInfoBits, uint8_t* pCodedBits, uint8_t* pTxBits)
{

#if 0
    constexpr uint32_t N_SMEM_ELEMS =
        N_MAX_THRD_TILES * sizeof(int32_t) + 2*N_MAX_CODED_BITS * sizeof(int16_t) +
        N_MAX_INFO_BITS * sizeof(uint8_t) + N_MAX_INFO_BITS * sizeof(uint8_t) +
        N_MAX_CODED_BITS * sizeof(uint8_t);
#else
    // 64 is is nTxBits rounded up to be divible by 32
    // nCodedBits is the same as N_MAX_CODED_BITS
    constexpr uint32_t N_SMEM_ELEMS =
        N_MAX_THRD_TILES * sizeof(int32_t) + 2*N_MAX_CODED_BITS * sizeof(int16_t) +
        64 * 2 * sizeof(uint8_t) +
        N_MAX_CODED_BITS * sizeof(uint8_t);
#endif

    __shared__ __align__(sizeof(uint32_t)) uint8_t smemBlk[N_SMEM_ELEMS];
    uint32_t*                                      pSmem = reinterpret_cast<uint32_t*>(smemBlk);

    const int SSB_id = blockIdx.x;

    // For SSB, there is always a fixed number of information bits per SSB. So the number of coded bits is known in advance too.
    const uint32_t nInfoBits  = CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS;
    const uint32_t nCodedBits = CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS; //512, same as  CUPHY_POLAR_ENC_MAX_CODED_BITS
    const uint32_t nTxBits    = CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS;

    // Compute offsets in bytes for each SSB. NB: the expectation is that each offset is divisible by 32 bits.
    // Since nInfoBits is not evenly divisible by 32, so it's rounded up accodingly
    const uint32_t info_offset    = SSB_id * 8; // offset for pInfoBits is (nInfoBits rounded to be divisible by 32) then div. 8
    const uint32_t x_coded_offset = SSB_id * (nCodedBits >> 3); // offset for pCodedBits
    const uint32_t tx_offset      = SSB_id * (nTxBits >> 3); // offset for pTxBits

    // SSB sent over Broadcast Channel, Mask set to 0
    uint32_t procModeBmsk = 0;

    // Call device functions
    encode(nInfoBits, nCodedBits, nTxBits, pInfoBits + info_offset, pSmem, pCodedBits + x_coded_offset, procModeBmsk);
    rateMatch(nInfoBits, nCodedBits, nTxBits, pCodedBits + x_coded_offset, pSmem, pTxBits + tx_offset);
}

cuphyStatus_t kernelSelectEncodeRateMatchMultiSSBs(cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pLaunchCfg,
                                                   uint16_t                                  num_SSBs)
{
    if (pLaunchCfg == nullptr) return CUPHY_STATUS_INVALID_ARGUMENT;

    if (pLaunchCfg->kernelNodeParamsDriver.func == nullptr)
    {
        // kernel (only one kernel option for now), so update function once; func set to nullptr in SSB channel constructor
        void* kernelFunc = reinterpret_cast<void*>(encodeRateMatchMultipleSSBsKernel);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    }

    // launch geometry (can change!)
    dim3 gridDim(num_SSBs);
    dim3 blockDim(N_THRDS_PER_TILE, N_MAX_THRD_TILES); // since code bits are the max. supported, the block size is good

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;

    return CUPHY_STATUS_SUCCESS;
}


} // namespace polar_encoder


