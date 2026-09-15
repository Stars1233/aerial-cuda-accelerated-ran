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

#include "comp_cwTreeTypes.hpp"
#include "../cuphy_internal.h"
#include "../polar_decoder/polar_cw_tree_layout.hpp"

#include <cooperative_groups.h>

#include <stdio.h>

#include <functional>

using namespace cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////
// following block vvvvv has many shared and repeated code with polar_encoder.cu              //
// ToDo: to be fixed!                                                                         //
////////////////////////////////////////////////////////////////////////////////////////////////
//#define ENABLE_DEBUG

namespace comp_tree
{
template <typename T>
CUDA_BOTH_INLINE constexpr T round_up_to_next(T val, T increment)
{
    return ((val + (increment - 1)) / increment) * increment;
}

static constexpr uint32_t N_THRDS_PER_WARP      = 32;         // cudaDeviceProp::warpSize;
static constexpr uint32_t FULL_WARP_ACTIVE_BMSK = 0xFFFFFFFF; // bitmask when all threads in a warp are active
static constexpr uint32_t N_THRDS_PER_TILE      = N_THRDS_PER_WARP;

// ============================================================================================

static constexpr uint32_t N_MIN_CODED_BITS = 32;   // smallest polar code word length
static constexpr uint32_t N_MAX_CODED_BITS = 1024; // biggest polar code word length
// Sizing the thread block as needed by maximum number of coded bits
// max(N_MAX_INFO_BITS, N_MAX_CODED_BITS) = N_MAX_CODED_BITS
static constexpr uint32_t N_MAX_THRDS_PER_THRD_BLK = round_up_to_next(N_MAX_CODED_BITS, N_THRDS_PER_TILE);
static constexpr uint32_t N_MAX_THRD_TILES         = N_MAX_THRDS_PER_THRD_BLK / N_THRDS_PER_WARP;
//
static constexpr uint32_t N_MAX_TREE_TYPES = 2 * N_MAX_CODED_BITS - 2;

// Round n upto nearest power of 2 compile time
static CUDA_BOTH_INLINE constexpr uint32_t roundUpToPow2(int32_t n, int32_t pow2)
{
    return pow2 >= n ? pow2 : roundUpToPow2(n, pow2 * 2);
}
static CUDA_BOTH_INLINE constexpr uint32_t roundUpToPow2(int32_t n)
{
    return roundUpToPow2(n, 1);
}
static constexpr uint32_t N_MAX_INFO_BITS    = CUPHY_POLAR_ENC_MAX_INFO_BITS;
static constexpr uint32_t N_MAX_SORT_ENTRIES = roundUpToPow2(N_MAX_INFO_BITS);
// Relative sequence index buffer needs to be sized to:
// - hold the largest sequence possible, this is N_MAX_CODED_BITS
// - hold the largest padding for bitonic sort
//   Maximum number of entries to pad = largest bitonic sort size (256) - smallest number of information bits which needs the largest bitonic sort size (129)
static constexpr uint32_t REL_SEQ_IDX_BUF_PAD = N_MAX_SORT_ENTRIES - ((N_MAX_SORT_ENTRIES / 2) + 1);
static constexpr uint32_t REL_SEQ_IDX_BUF_LEN = N_MAX_CODED_BITS + REL_SEQ_IDX_BUF_PAD;
// ============================================================================================

// clang-format off
static __device__ __constant__  uint16_t POLAR_WM_ARRAY_32[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,
   2,    4,    4,    8,    4,    8,    8,   16,
   2,    4,    4,    8,    4,    8,    8,   16,
   4,    8,    8,   16,    8,   16,   16,   32
};

static __device__ __constant__  uint16_t POLAR_WM_ARRAY_64[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,
   2,    4,    4,    8,    4,    8,    8,   16,
   2,    4,    4,    8,    4,    8,    8,   16,
   4,    8,    8,   16,    8,   16,   16,   32,
   2,    4,    4,    8,    4,    8,    8,   16,
   4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,
   8,   16,   16,   32,   16,   32,   32,   64
};

static __device__ __constant__  uint16_t POLAR_WM_ARRAY_128[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,    2,    4,    4,    8,    4,    8,    8,   16,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128
};

static __device__ __constant__  uint16_t POLAR_WM_ARRAY_256[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,    2,    4,    4,    8,    4,    8,    8,   16,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256
};

static __device__ __constant__  uint16_t POLAR_WM_ARRAY_512[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,    2,    4,    4,    8,    4,    8,    8,   16,    2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512
};

static __device__ __constant__  uint16_t POLAR_WM_ARRAY_1024[] =
{
   1,    2,    2,    4,    2,    4,    4,    8,    2,    4,    4,    8,    4,    8,    8,   16,    2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,
   2,    4,    4,    8,    4,    8,    8,   16,    4,    8,    8,   16,    8,   16,   16,   32,    4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,
   4,    8,    8,   16,    8,   16,   16,   32,    8,   16,   16,   32,   16,   32,   32,   64,    8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,
   8,   16,   16,   32,   16,   32,   32,   64,   16,   32,   32,   64,   32,   64,   64,  128,   16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,
  16,   32,   32,   64,   32,   64,   64,  128,   32,   64,   64,  128,   64,  128,  128,  256,   32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,
  32,   64,   64,  128,   64,  128,  128,  256,   64,  128,  128,  256,  128,  256,  256,  512,   64,  128,  128,  256,  128,  256,  256,  512,  128,  256,  256,  512,  256,  512,  512, 1024
};

// Pointers to weight metric arrays LUTs
static __device__ __constant__ uint16_t const* POLAR_WM_LUT_PTR[] =
{
    POLAR_WM_ARRAY_32,
    POLAR_WM_ARRAY_64,
    POLAR_WM_ARRAY_128,
    POLAR_WM_ARRAY_256,
    POLAR_WM_ARRAY_512,
    POLAR_WM_ARRAY_1024
};

//------------------------------------------------------------------------------------------------

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

// Reliability sequence values < 1024
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_1024[] =
{
    0,	1,	2,	4,	8,	16,	32,	3,	5,	64,	9,	6,	17,	10,	18,	128,	12,	33,	65,	20,	256,	34,	24,	36,	7,	129,	66,	512,
    11,	40,	68,	130,	19,	13,	48,	14,	72,	257,	21,	132,	35,	258,	26,	513,	80,	37,	25,	22,	136,	260,	264,
    38,	514,	96,	67,	41,	144,	28,	69,	42,	516,	49,	74,	272,	160,	520,	288,	528,	192,	544,	70,
    44,	131,	81,	50,	73,	15,	320,	133,	52,	23,	134,	384,	76,	137,	82,	56,	27,	97,	39,	259,	84,	138,	145,
    261,	29,	43,	98,	515,	88,	140,	30,	146,	71,	262,	265,	161,	576,	45,	100,	640,	51,	148,	46,
    75,	266,	273,	517,	104,	162,	53,	193,	152,	77,	164,	768,	268,	274,	518,	54,	83,	57,	521,
    112,	135,	78,	289,	194,	85,	276,	522,	58,	168,	139,	99,	86,	60,	280,	89,	290,	529,	524,	196,
    141,	101,	147,	176,	142,	530,	321,	31,	200,	90,	545,	292,	322,	532,	263,	149,	102,
    105,	304,	296,	163,	92,	47,	267,	385,	546,	324,	208,	386,	150,	153,	165,	106,	55,	328,
    536,	577,	548,	113,	154,	79,	269,	108,	578,	224,	166,	519,	552,	195,	270,	641,	523,
    275,	580,	291,	59,	169,	560,	114,	277,	156,	87,	197,	116,	170,	61,	531,	525,	642,	281,
    278,	526,	177,	293,	388,	91,	584,	769,	198,	172,	120,	201,	336,	62,	282,	143,	103,
    178,	294,	93,	644,	202,	592,	323,	392,	297,	770,	107,	180,	151,	209,	284,	648,	94,
    204,	298,	400,	608,	352,	325,	533,	155,	210,	305,	547,	300,	109,	184,	534,	537,
    115,	167,	225,	326,	306,	772,	157,	656,	329,	110,	117,	212,	171,	776,	330,	226,
    549,	538,	387,	308,	216,	416,	271,	279,	158,	337,	550,	672,	118,	332,	579,	540,
    389,	173,	121,	553,	199,	784,	179,	228,	338,	312,	704,	390,	174,	554,	581,	393,
    283,	122,	448,	353,	561,	203,	63,	340,	394,	527,	582,	556,	181,	295,	285,	232,	124,
    205,	182,	643,	562,	286,	585,	299,	354,	211,	401,	185,	396,	344,	586,	645,	593,
    535,	240,	206,	95,	327,	564,	800,	402,	356,	307,	301,	417,	213,	568,	832,	588,	186,
    646,	404,	227,	896,	594,	418,	302,	649,	771,	360,	539,	111,	331,	214,	309,	188,	449,
    217,	408,	609,	596,	551,	650,	229,	159,	420,	310,	541,	773,	610,	657,	333,	119,	600,
    339,	218,	368,	652,	230,	391,	313,	450,	542,	334,	233,	555,	774,	175,	123,	658,	612,
    341,	777,	220,	314,	424,	395,	673,	583,	355,	287,	183,	234,	125,	557,	660,	616,	342,
    316,	241,	778,	563,	345,	452,	397,	403,	207,	674,	558,	785,	432,	357,	187,	236,	664,
    624,	587,	780,	705,	126,	242,	565,	398,	346,	456,	358,	405,	303,	569,	244,	595,	189,
    566,	676,	361,	706,	589,	215,	786,	647,	348,	419,	406,	464,	680,	801,	362,	590,	409,
    570,	788,	597,	572,	219,	311,	708,	598,	601,	651,	421,	792,	802,	611,	602,	410,	231,
    688,	653,	248,	369,	190,	364,	654,	659,	335,	480,	315,	221,	370,	613,	422,	425,	451,
    614,	543,	235,	412,	343,	372,	775,	317,	222,	426,	453,	237,	559,	833,	804,	712,	834,
    661,	808,	779,	617,	604,	433,	720,	816,	836,	347,	897,	243,	662,	454,	318,	675,	618,
    898,	781,	376,	428,	665,	736,	567,	840,	625,	238,	359,	457,	399,	787,	591,	678,	434,
    677,	349,	245,	458,	666,	620,	363,	127,	191,	782,	407,	436,	626,	571,	465,	681,	246,
    707,	350,	599,	668,	790,	460,	249,	682,	573,	411,	803,	789,	709,	365,	440,	628,	689,
    374,	423,	466,	793,	250,	371,	481,	574,	413,	603,	366,	468,	655,	900,	805,	615,	684,
    710,	429,	794,	252,	373,	605,	848,	690,	713,	632,	482,	806,	427,	904,	414,	223,	663,
    692,	835,	619,	472,	455,	796,	809,	714,	721,	837,	716,	864,	810,	606,	912,	722,	696,
    377,	435,	817,	319,	621,	812,	484,	430,	838,	667,	488,	239,	378,	459,	622,	627,	437,
    380,	818,	461,	496,	669,	679,	724,	841,	629,	351,	467,	438,	737,	251,	462,	442,	441,
    469,	247,	683,	842,	738,	899,	670,	783,	849,	820,	728,	928,	791,	367,	901,	630,	685,
    844,	633,	711,	253,	691,	824,	902,	686,	740,	850,	375,	444,	470,	483,	415,	485,	905,
    795,	473,	634,	744,	852,	960,	865,	693,	797,	906,	715,	807,	474,	636,	694,	254,	717,
    575,	913,	798,	811,	379,	697,	431,	607,	489,	866,	723,	486,	908,	718,	813,	476,	856,
    839,	725,	698,	914,	752,	868,	819,	814,	439,	929,	490,	623,	671,	739,	916,	463,	843,
    381,	497,	930,	821,	726,	961,	872,	492,	631,	729,	700,	443,	741,	845,	920,	382,	822,
    851,	730,	498,	880,	742,	445,	471,	635,	932,	687,	903,	825,	500,	846,	745,	826,	732,
    446,	962,	936,	475,	853,	867,	637,	907,	487,	695,	746,	828,	753,	854,	857,	504,	799,
    255,	964,	909,	719,	477,	915,	638,	748,	944,	869,	491,	699,	754,	858,	478,	968,	383,
    910,	815,	976,	870,	917,	727,	493,	873,	701,	931,	756,	860,	499,	731,	823,	922,	874,
    918,	502,	933,	743,	760,	881,	494,	702,	921,	501,	876,	847,	992,	447,	733,	827,	934,
    882,	937,	963,	747,	505,	855,	924,	734,	829,	965,	938,	884,	506,	749,	945,	966,	755,
    859,	940,	830,	911,	871,	639,	888,	479,	946,	750,	969,	508,	861,	757,	970,	919,	875,
    862,	758,	948,	977,	923,	972,	761,	877,	952,	495,	703,	935,	978,	883,	762,	503,	925,
    878,	735,	993,	885,	939,	994,	980,	926,	764,	941,	967,	886,	831,	947,	507,	889,	984,
    751,	942,	996,	971,	890,	509,	949,	973,	1000,	892,	950,	863,	759,	1008,	510,	979,	953,
    763,	974,	954,	879,	981,	982,	927,	995,	765,	956,	887,	985,	997,	986,	943,	891,	998,
    766,	511,	988,	1001,	951,	1002,	893,	975,	894,	1009,	955,	1004,	1010,	957,	983,	958,	987,
    1012,	999,	1016,	767,	989,	1003,	990,	1005,	959,	1011,	1013,	895,	1006,	1014,	1017,	1018,	991,
    1020,	1007,	1015,	1019,	1021,	1022,	1023
};

// Pointers to reliability sequence LUTs
static __device__ __constant__ uint16_t const* POLAR_REL_SEQ_IDXS_LUT_PTR[] =
{
    POLAR_REL_SEQ_IDXS_32,
    POLAR_REL_SEQ_IDXS_64,
    POLAR_REL_SEQ_IDXS_128,
    POLAR_REL_SEQ_IDXS_256,
    POLAR_REL_SEQ_IDXS_512,
    POLAR_REL_SEQ_IDXS_1024
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

////////////////////////////////////////////////////////////////////////////////////////////////
// above block ^^^^ has many shared and repeated code with polar_encoder.cu                   //
// to be fixed!                                                                               //
////////////////////////////////////////////////////////////////////////////////////////////////

// Compute warp level min reduction. In the process of reduction, the order is preserved.
// This criterion is required in case of ties
__device__ __forceinline__ void warpLevelReduceMinIdx(int32_t& val, int32_t& idx)
{
    int32_t tmpVal = INT_MAX;
    int32_t tmpIdx = 0;
    tmpVal         = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, val, 1);
    tmpIdx         = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, idx, 1);
    if(threadIdx.x % 2 == 0 && tmpVal < val)
    {
        val = tmpVal;
        idx = tmpIdx;
    }
    //----------------------------------------------------
    tmpVal = INT_MAX;
    tmpVal = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, val, 2);
    tmpIdx = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, idx, 2);
    if(threadIdx.x % 4 == 0 && tmpVal < val)
    {
        val = tmpVal;
        idx = tmpIdx;
    }
    //----------------------------------------------------
    tmpVal = INT_MAX;
    tmpVal = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, val, 4);
    tmpIdx = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, idx, 4);
    if(threadIdx.x % 8 == 0 && tmpVal < val)
    {
        val = tmpVal;
        idx = tmpIdx;
    }
    //----------------------------------------------------
    tmpVal = INT_MAX;
    tmpVal = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, val, 8);
    tmpIdx = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, idx, 8);
    if(threadIdx.x % 16 == 0 && tmpVal < val)
    {
        val = tmpVal;
        idx = tmpIdx;
    }
    //----------------------------------------------------
    tmpVal = INT_MAX;
    tmpVal = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, val, 16);
    tmpIdx = __shfl_down_sync(FULL_WARP_ACTIVE_BMSK, idx, 16);
    if(threadIdx.x % 32 == 0 && tmpVal < val)
    {
        val = tmpVal;
        idx = tmpIdx;
    }
}

// this kernel reads values from valArray with indices read from idxArray within [startRange, endRange],
// then returns the index corresponding to the minimum value, in case of tie, it will return index with
// lower tid (higher reliability)
__inline__ __device__ int32_t blockReduceMinIdx(thread_block const& thisThrdBlk,
                                                const int16_t*      idxArray,
                                                const uint16_t*     valArray,
                                                const uint32_t      startRange,
                                                const uint32_t      endRange)
{
    // shared mem for 32 partial min values and corresponding indices
    static __shared__ int32_t values[N_MAX_THRD_TILES];
    static __shared__ int32_t indices[N_MAX_THRD_TILES];

    uint32_t thrdIdxInBlk = thisThrdBlk.thread_rank();
    uint32_t thrdLane     = thrdIdxInBlk % N_THRDS_PER_WARP;
    uint32_t warpIdx      = thrdIdxInBlk / N_THRDS_PER_WARP;

    int32_t val = INT32_MAX;
    int32_t idx = 0;
    if(thrdIdxInBlk >= startRange && thrdIdxInBlk <= endRange)
    {
        idx = idxArray[thrdIdxInBlk];
        val = valArray[idx];
    }

    warpLevelReduceMinIdx(val, idx); // Each warp performs partial reduction

    if(thrdLane == 0)
    {
        values[warpIdx]  = val; // Write reduced value per warp to shared memory
        indices[warpIdx] = idx; // Write corresponding index to shared memory
    }

    __syncthreads(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if(thrdIdxInBlk < thisThrdBlk.size() / N_THRDS_PER_WARP)
    {
        val = values[thrdLane];
        idx = indices[thrdLane];
    }
    else
    {
        val = INT_MAX;
        idx = 0;
    }

    if(warpIdx == 0)
    {
        warpLevelReduceMinIdx(val, idx); //Final reduce within first warp
    }

    // thread 0 has the correct idx
    return idx;
}

static __global__ void
compCwTreeTypesKernel(compCwTreeTypesDynDescr_t* pDynDescr)
{
    const uint32_t UCI_SEG_IDX = blockIdx.x;

    uint32_t nTxBits          = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].E_cw;
    uint32_t nInfoBits        = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].K_cw;
    uint32_t nCodedBits       = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].N_cw;
    uint8_t  n                = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].n_cw;
    uint8_t* pTreeTypesOutput = pDynDescr->pCwTreeTypesAddrs[UCI_SEG_IDX];
    uint8_t  exitFlag         = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].exitFlag;

    if(exitFlag == 1)
    {
       return;
    }

    uint8_t nPc    = 0; // number of parity check bits - wmFlag
    uint8_t wmFlag = 0;

    if((nInfoBits >= 18) && (nInfoBits <= 25))
    {
        wmFlag = (nTxBits - nInfoBits + 3) > 192 ? 1 : 0;
        nPc    = 3;
    }
    
    // Force number information and parity bits to enable coding (WARNING! THIS IS NOT FOLLOWING 3GPP SPEC)
    if((nInfoBits + nPc) > nTxBits)
    {
        if(nInfoBits > nTxBits)
        {
            nInfoBits = nTxBits;
            nPc = 0;  
        }else
        {
            nPc = nTxBits - nInfoBits;
        }
    }



    thread_block const& thisThrdBlk = this_thread_block();

    // 1 tile per warp
    thread_block_tile<N_THRDS_PER_TILE> const& thisThrdTile =
        tiled_partition<N_THRDS_PER_TILE>(thisThrdBlk);

    uint32_t thrdIdxInBlk = thisThrdBlk.thread_rank();
    uint32_t tileIdx      = thrdIdxInBlk / thisThrdTile.size();

    //--------------------------------------------------------------------------------------------------------
    //ToDo improve shared memory usage:
    // 1- memory dedicated for pCwTreeTypes is not used until the very end, it can be first used for other
    //    arrays such as pRelSeqIdxsPruned or pTileStartOffsets
    // 2- change to dynamic shared memory and use smaller footprint?

    // shared memory
    constexpr uint32_t N_SMEM_ELEMS =
        N_MAX_THRD_TILES * sizeof(int32_t) +
        REL_SEQ_IDX_BUF_LEN * sizeof(int16_t) +
        N_MAX_CODED_BITS * sizeof(int8_t) +
        2 * N_MAX_TREE_TYPES * sizeof(int8_t);

    __shared__ __align__(sizeof(uint32_t)) uint8_t smemBlk[N_SMEM_ELEMS];
    //
    uint32_t* pSmem             = reinterpret_cast<uint32_t*>(smemBlk);
    int32_t*  pTileStartOffsets = reinterpret_cast<int32_t*>(pSmem);
    int16_t*  pRelSeqIdxsPruned = reinterpret_cast<int16_t*>(&pTileStartOffsets[N_MAX_THRD_TILES]);
    int8_t*   pCwBitTypes       = reinterpret_cast<int8_t*>(&pRelSeqIdxsPruned[REL_SEQ_IDX_BUF_LEN]);
    int8_t*   pCwTreeTypes      = &pCwBitTypes[N_MAX_CODED_BITS];
    // node classes for the fast-SSC operation list: 0/1/2 as tree types, 4 = REP
    // (all leaves frozen except the last), 5 = SPC (all info except the
    // first), 3 = mixed (descend), 6 = contains a parity-check leaf (blocks
    // fusion). Same binary-tree indexing as pCwTreeTypes.
    int8_t*   pCwNodeClasses    = &pCwTreeTypes[N_MAX_TREE_TYPES];

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

    uint32_t        relSeqLutIdx = __ffs(nCodedBits) - __ffs(N_MIN_CODED_BITS);
    uint16_t const* pRelSeqLut   = POLAR_REL_SEQ_IDXS_LUT_PTR[relSeqLutIdx];

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

        // Store the pruned reliability sequence indices indices in reflected form (most reliable to least
        // reliable indices). This way the first nInfoBits indices are the most reliable indices. Note that
        // the indices are stored towards the end of the buffer
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
    // Set bit types
    if(thrdIdxInBlk < nCodedBits)
    {
        pCwBitTypes[thrdIdxInBlk] = 0;
    }
    __syncthreads();

    if(thrdIdxInBlk < (nInfoBits + nPc))
    {
        int16_t const& infoBitIdx = pRelSeqIdxsPruned[prunedRelSeqStartIdx + thrdIdxInBlk];
        pCwBitTypes[infoBitIdx]   = 1;
        if(thrdIdxInBlk >= (nInfoBits + wmFlag))
        {
            pCwBitTypes[infoBitIdx] = 2;
        }
    }

    __syncthreads();

    //---------------------------------------------------------------------------------------------------------
    // in the special case where wmFlag is non-zero, index of one parity bit needs to be computed as follows
    if(wmFlag > 0)
    {
        int16_t* pIdxsPruned = &pRelSeqIdxsPruned[prunedRelSeqStartIdx];
        auto     pWmLut      = POLAR_WM_LUT_PTR[relSeqLutIdx];
        int32_t  minWmIdx    = blockReduceMinIdx(thisThrdBlk, pIdxsPruned, pWmLut, 3, nInfoBits - 1);
        // thread 0 in blockReduceMinIdx return correct minWmIdx
        if(thrdIdxInBlk == 0)
        {
            pCwBitTypes[minWmIdx] = 2;
        }
        __syncthreads();
    }

#ifdef ENABLE_DEBUG
    if(thrdIdxInBlk < nCodedBits)
    {
        printf("pCwBitTypes[%d] = %d\n", thrdIdxInBlk, pCwBitTypes[thrdIdxInBlk]);
    }
#endif

    //--------------------------------------------------------------------------------------------------------
    // Compute tree types
    // first fill types for stage 0
    int32_t cwTypeStartIdx = nCodedBits - 2;
    if(thrdIdxInBlk < nCodedBits)
    {
        pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = pCwBitTypes[thrdIdxInBlk];
    }

    __syncthreads();

    // next fill the higher stages recursively, starting based on stage 0
    for(int32_t s = 1; s < n; s++)
    {
        int32_t stgSize = 1 << (n - s);
        cwTypeStartIdx  = stgSize - 2;
        if(thrdIdxInBlk < stgSize)
        {
            int32_t childStartIdx = stgSize * 2 - 2;
            int8_t  childNodeA    = pCwTreeTypes[childStartIdx + 2 * thrdIdxInBlk];
            int8_t  childNodeB    = pCwTreeTypes[childStartIdx + 2 * thrdIdxInBlk + 1];

            if(childNodeA == 0 && childNodeB == 0)
            {
                pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 0;
            }
            else if(childNodeA == 1 && childNodeB == 1)
            {
                pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 1;
            }
            else
            {
                pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 3;
            }
        }
        __syncthreads();
    }

    // copy tree types to output
    for(int i = thrdIdxInBlk; i < (2 * nCodedBits - 2); i += thisThrdBlk.size())
    {
        pTreeTypesOutput[i + 2] = pCwTreeTypes[i];
    }

    //--------------------------------------------------------------------------------------------------------
    // Emit the successive-cancellation traversal operation list consumed by the polar
    // decoder: one byte per visited pruned node, in bit order, stored in the
    // SC operation-list region (see polar_cw_tree_layout.hpp). Entry layout:
    // bits[3:0] = node stage, bits[5:4] = node type (0 frozen, 1 info, 2 parity).
    //
    // The node visited for leaf j is its highest ancestor whose subtree is
    // uniform (type != 3); type 3 is monotone towards the root, so the walk up
    // can stop at the first mixed ancestor. Only the first leaf of each visited
    // node emits an entry; stream compaction orders entries by leaf index.
    bool    opPred  = false;
    uint8_t opEntry = 0;
    if(thrdIdxInBlk < nCodedBits)
    {
        const uint32_t j        = thrdIdxInBlk;
        int8_t         nodeType = pCwTreeTypes[(nCodedBits - 2) + j];
        int32_t        nodeStage = 0;
        while(nodeStage < (n - 1))
        {
            const int32_t parentRowStart = (1 << (n - nodeStage - 1)) - 2;
            const int8_t  parentType     = pCwTreeTypes[parentRowStart + (j >> (nodeStage + 1))];
            if(parentType == 3)
            {
                break;
            }
            nodeType = parentType;
            nodeStage++;
        }
        opPred  = ((j & ((1u << nodeStage) - 1)) == 0);
        opEntry = static_cast<uint8_t>(nodeStage | (nodeType << 4));
    }

    int32_t opThrdOffset = 0;
    strmCompactionHelper<N_THRDS_PER_TILE>(thisThrdBlk, thisThrdTile, opPred, nActiveThrdTiles, pTileStartOffsets, opThrdOffset);

    if(opPred)
    {
        const int32_t opPos = pTileStartOffsets[tileIdx] + opThrdOffset;
        pTreeTypesOutput[cuphy::polar::PolarCwTreeLayout::scOpListOffset(nCodedBits) + opPos] = opEntry;
    }

    //--------------------------------------------------------------------------------------------------------
    // Fast-SSC node classes: like the tree types, but single-information REP
    // subtrees (all leaves frozen except the last) and single-parity SPC
    // subtrees (all info except the first) stay fused instead of mixed.
    // Stage 0: classes are the bit types themselves, except parity-check
    // leaves which block fusion.
    __syncthreads();
    {
        int32_t clsStartIdx = nCodedBits - 2;
        if(thrdIdxInBlk < nCodedBits)
        {
            int8_t t = pCwBitTypes[thrdIdxInBlk];
            pCwNodeClasses[clsStartIdx + thrdIdxInBlk] = (t == 2) ? 6 : t;
        }
        __syncthreads();

        for(int32_t s = 1; s < n; s++)
        {
            int32_t stgSize = 1 << (n - s);
            clsStartIdx     = stgSize - 2;
            if(thrdIdxInBlk < stgSize)
            {
                int32_t childStartIdx = stgSize * 2 - 2;
                int8_t  cL            = pCwNodeClasses[childStartIdx + 2 * thrdIdxInBlk];
                int8_t  cR            = pCwNodeClasses[childStartIdx + 2 * thrdIdxInBlk + 1];
                int8_t  cls;
                if(cL == 0 && cR == 0)                                { cls = 0; }
                else if(cL == 1 && cR == 1)                           { cls = 1; }
                else if(cL == 0 && ((s == 1 && cR == 1) || cR == 4))  { cls = 4; } // REP
                else if(cR == 1 && ((s == 2 && cL == 4) || cL == 5))  { cls = 5; } // SPC ((0,1) pair == SPC of size 2)
                else                                                  { cls = 3; }
                pCwNodeClasses[clsStartIdx + thrdIdxInBlk] = cls;
            }
            __syncthreads();
        }
    }

    // Emit the fast-SSC operation list into its own region (see
    // polar_cw_tree_layout.hpp): identical emission to the SC operation list,
    // but ascending through any non-mixed class (0/1/4/5), so REP/SPC subtrees
    // appear as single entries.
    bool    fssPred  = false;
    uint8_t fssEntry = 0;
    if(thrdIdxInBlk < nCodedBits)
    {
        const uint32_t j        = thrdIdxInBlk;
        int8_t         nodeCls  = pCwNodeClasses[(nCodedBits - 2) + j];
        if(nodeCls == 6) { nodeCls = 2; } // lone parity-check leaf decodes like a hard-decision leaf
        int32_t        nodeStage = 0;
        while(nodeStage < (n - 1))
        {
            const int32_t parentRowStart = (1 << (n - nodeStage - 1)) - 2;
            const int8_t  parentCls      = pCwNodeClasses[parentRowStart + (j >> (nodeStage + 1))];
            if(parentCls == 3 || parentCls == 6)
            {
                break;
            }
            nodeCls = parentCls;
            nodeStage++;
        }
        fssPred  = ((j & ((1u << nodeStage) - 1)) == 0);
        fssEntry = static_cast<uint8_t>(nodeStage | (nodeCls << 4));
    }

    int32_t fssThrdOffset = 0;
    strmCompactionHelper<N_THRDS_PER_TILE>(thisThrdBlk, thisThrdTile, fssPred, nActiveThrdTiles, pTileStartOffsets, fssThrdOffset);

    if(fssPred)
    {
        const int32_t fssPos = pTileStartOffsets[tileIdx] + fssThrdOffset;
        pTreeTypesOutput[cuphy::polar::PolarCwTreeLayout::fssOpListOffset(nCodedBits) + fssPos] = fssEntry;
    }

    //--------------------------------------------------------------------------------------------------------
    #ifdef ENABLE_DEBUG
        if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
        {
            printf("\n compCwTreeTypesKernel running... \n");
        }

        if((0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
        {
            printf("\n UCI segment %d has the following parameters: E = %d, K = %d, N = %d, n = %d \n", UCI_SEG_IDX, nTxBits, nInfoBits, nCodedBits, n);
        }
    #endif
}

} //namespace comp_tree

void compCwTreeTypes::kernelSelect(uint16_t                         nPolUciSegs,
                                   const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                   cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg)
{
    // determine max encoded cb size:
    uint16_t max_N_cw = 0;
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        if(pPolUciSegPrmsCpu[segIdx].N_cw > max_N_cw)
            max_N_cw = pPolUciSegPrmsCpu[segIdx].N_cw;
    }

    // launch geometry (can change!)
    dim3 gridDim(nPolUciSegs);
    dim3 blockDim(max_N_cw);

    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(comp_tree::compCwTreeTypesKernel);
   {MemtraceDisableScope md;cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);}

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void compCwTreeTypes::setup(uint16_t                         nPolUciSegs,                 // number of polar UCI segments
                            const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,           // starting adreass of polar UCI segment parameters (CPU)
                            const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,           // starting adreass of polar UCI segment parameters (GPU)
                            uint8_t**                        pCwTreeTypesAddrs,           // pointer to cwTreeTypes addresses
                            compCwTreeTypesDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                            void*                            pGpuDynDesc,                 // pointer to descriptor in gpu
                            uint8_t                          enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
                            cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
                            cudaStream_t                     strm)                                            // stream to perform copy
{
    // populate dynamic descriptor:
    pCpuDynDesc->pPolarUciSegPrms = pPolUciSegPrmsGpu;
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        pCpuDynDesc->pCwTreeTypesAddrs[segIdx] = pCwTreeTypesAddrs[segIdx];
    }

    // save pointer to GPU descriptor
    compCwTreeTypesKernelArgs_t& kernelArgs = m_kernelArgs;
    kernelArgs.pDynDescr                    = reinterpret_cast<compCwTreeTypesDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(compCwTreeTypesDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect(nPolUciSegs, pPolUciSegPrmsCpu, pLaunchCfg);
    pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void compCwTreeTypes::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    dynDescrSizeBytes  = sizeof(compCwTreeTypesDynDescr_t);
    dynDescrAlignBytes = alignof(compCwTreeTypesDynDescr_t);
}