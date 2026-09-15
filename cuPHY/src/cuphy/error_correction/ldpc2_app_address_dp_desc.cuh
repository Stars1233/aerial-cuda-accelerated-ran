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

#if !defined(LDPC2_APP_ADDRESS_DP_DESC_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_DP_DESC_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"

////////////////////////////////////////////////////////////////////////
// APP address calculation using the dot product (DP) instruction.
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)    (threadIdx + shift) < Z
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)   (threadIdx + shift) >= Z

////////////////////////////////////////////////////////////////////////
// GENERAL CONSIDERATIONS FOR 2x SIMD ADDRESS CALCULATIONS
// 1.) Overall maximum APP address (as integer value):
//     BG1: ((68 * 384) - 1) * sizeof(T)
//         half:   52,222
//         half2: 104,444
//     BG2: ((52 * 384) - 1) * sizeof(T)
//         half:   39,934
//         half2:  79,868
//     Therefore:
//         - the maximum address can be stored in 16 bits for all (BG, Z) for half
//         - the maximum address CANNOT be stored in 16 bits for half2
// 2.) Calculating an APP address for a given (col_idx, shift) pair
//
// addr_app = [(col_idx * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_idx * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
// 3.) When shift is zero:
// addr_app = [(col_idx * Z) + (threadIdx)] * sizeof(T) = (col_idx * Z * sizeof(T)) + threadIdx
// (Depending on what base graph data is stored, this can be a simple ADD instruction.)
//
// Note also that the shift for the rightmost (last) element of each parity node
// row is zero. As such, when the row degree is odd, we can avoid the conditional
// processing for the last row element. When the row degree is even, this zero
// shift value will be paired with another (non-zero) shift value.
// BG1: Total edges  = 316
//      num rows     = 46
//      num odd rows = 30
//
// 4.) Max column for the address of a 32-bit APP element (e.g. float or half2)
//     that can be stored in 16 bits:
// USHRT_MAX = 2^16 - 1 = 65535
// MAX_COL_IDX_32 = floor(USHRT_MAX / (384 * sizeof(half2)))
//                = 42
//
// The APP address (where address = index * sizeof(T)) is given by:
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx - Z) * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx) * sizeof(T)                ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx) * sizeof(T) - Z * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [((col_idx * Z  + shift) * sizeof(T)) + (threadIdx * sizeof(T))                         ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [((col_idx * Z) + shift) * sizeof(T)) + (threadIdx * sizeof(T)) + (-1) * (Z * sizeof(T))]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [((col_idx * Z + shift) * sizeof(T)) + (threadIdx * sizeof(T))                         ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [((col_idx * Z + shift) * sizeof(T)) + (threadIdx * sizeof(T)) + (-1) * (Z * sizeof(T))]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// --base offset---->|
//     <---shift---->|
//     | - - - - - - - - - - - - - - - - - - - - |
//  0  | . . . . . . x                           |
//  1  |     (6)       x                         |
//  2  |                 x                       |
//  3  |                   x                     |
//  4  |                     x                   |
//  5  |                       x                 |
//  6  |                         x               |
//  7  |                           x             |
//  8  |                             x           |
//  9  |                               x         |
// 10  |                                 x       |
// 11  |                                   x     |
// 12  |                                     x   |
// 13  |                                       x |
// 14  | x.......................................|<-- wrap index (Z - shift)
// 15  |   x                                     |
// 16  |     x                                   |
// 17  |       x                                 |
// 18  |         x                               |
// 19  |           x                             |
//     | - - - - - - - - - - - - - - - - - - - - |
//
// APPROACH:
//  REQUIRED ADDRESS GENERATION VARIABLES:
//    R0 =  -(Z*sz)                                                     (constant for all addresses and threads)
//    THREADIDX                                                         (per-thread)
//  PER-NODE STORAGE
//    COL_IDX_SHIFT_LOW  = ((col_idx[0] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
//    COL_IDX_SHIFT_HIGH = ((col_idx[1] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
//    WRAP_INDEX = Z - shift
//      
//
// ADDR_0 = threadIdx * sizeof(T) + COL_IDX_SHIFT_LOW  (muladd.lo)
// ADDR_1 = threadIdx * sizeof(T) + COL_IDX_SHIFT_HIGH (muladd.lo)
// R1 = HSET2.LT.BM(threadIdx, wrapIndex)                    (threadIdx < wrap_index) ? : 0xFFFF : 0x0000
//                                                           (0xFF interpreted as signed negative one by dp2a)
// dp2a.lo.s32.s32 ADDR_0, [ 0 | -(Z*sz) ], R1, ADDR_0;      Add 0 or Z*sz to ADDR_0
// dp2a.hi.s32.s32 ADDR_1, [ 0 | -(Z*sz) ], R1, ADDR_1;      Add 0 or Z*sz to ADDR_1

namespace
{

template <typename T>
__device__
ushort2 make_ushort2(T x, T y)
{
    ushort2 u2;
    u2.x = static_cast<short>(x);
    u2.y = static_cast<short>(y);
    return u2;
}
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_loc_address_dp_desc
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG>
struct app_loc_address_dp_desc
{
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    typedef BG_adj_desc<BG> bg_desc_t;
    //------------------------------------------------------------------
    // app_loc_address_dp_desc()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_dp_desc(const LDPC_kernel_params& params,
                            const bg_desc_t&          bgd,
                            unsigned int              t_idx) : bg_desc(bgd),
                                                               negZsz(to_word(make_short2(-params.Z * sizeof(T), 0))),
                                                               tIdx_tIdx(h0_h0(t_idx))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_dp_desc()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_dp_desc(const cuphyLDPCDecodeConfigDesc_t& config,
                            const bg_desc_t&                   bgd,
                            unsigned int                       t_idx) : bg_desc(bgd),
                                                                        negZsz(to_word(make_short2(-config.Z * sizeof(T), 0))),
                                                                        tIdx_tIdx(h0_h0(t_idx))
    {
    }
    //------------------------------------------------------------------
    // generate()
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        const int ROW_DEGREE  = row_degree<BG, CHECK_IDX>::value;
        const int PAIR_OFFSET = row_pair_index<BG, CHECK_IDX>::value;
        #pragma unroll
        for(int i = 0; i < (ROW_DEGREE + 1) / 2; ++i)
        {
            word_t WRAP_INDEX, R1;
            int32_t ADDR_0, ADDR_1;
            
            const int32_t COL_IDX_SHIFT_LOW  = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_low;
            const int32_t COL_IDX_SHIFT_HIGH = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_high;
            WRAP_INDEX.u32                   = bg_desc.nodes[PAIR_OFFSET + i].wrap_index;

            ADDR_0 = muladd_lo_s32(threadIdx.x, sizeof(T), COL_IDX_SHIFT_LOW);
            ADDR_1 = muladd_lo_s32(threadIdx.x, sizeof(T), COL_IDX_SHIFT_HIGH);
            R1 = hset2_bm_lt(tIdx_tIdx, WRAP_INDEX); // R1 = HSET2.LT.BM(threadIdx, WRAP_INDEX)  (HSET2.BM instruction)

            //if((0 == CHECK_IDX) && (0 == threadIdx.x) && (0 == i))
            //{
            //    printf("threadIdx = %u, i = (%i, %i), WRAP_INDEX = (%u, %u), COL_Z_SHIFT = (%i, %i), R1 = (%.1f, %.1f), R2 = (%u, %u)\n",
            //           threadIdx.x,
            //           i * 2 + 1,
            //           i * 2,
            //           WRAP_INDEX.u32 >> 16,
            //           WRAP_INDEX.u32 & 0x0000FFFF,
            //           COL_IDX_SHIFT_HIGH,
            //           COL_IDX_SHIFT_LOW,
            //           __high2float(R1.f16x2),
            //           __low2float(R1.f16x2),
            //           R2.u16x2.y,
            //           R2.u16x2.x);
            //}

            //   -------------------------
            //   |    high   |    low    |  RA
            //   -------------------------
            //
            //   -------------------------
            //   |  a  |  b  |  c  |  d  |  RB
            //   -------------------------
            //
            //   dp2a.hi: output = (RB.a * RA.high) + (RB.b * RA.low)
            //   dp2a.lo: output = (RB.c * RA.high) + (RB.d * RA.low)
            
            LDPC2_ASM("\t"
                      "{\n\t\t"
                      "dp2a.lo.s32.s32 %0, %1, %2, %3;\n\t"
                      "}\n"
                      : "=r"(app_addr[i*2])
                      : "r"(negZsz.i32),
                        "r"(R1.i32),
                        "r"(ADDR_0));


            if(((i * 2) + 1) < ROW_DEGREE)
            {
                LDPC2_ASM("\t"
                          "{\n\t\t"
                          "dp2a.hi.s32.s32 %0, %1, %2, %3;\n\t"
                          "}\n"
                          : "=r"(app_addr[i*2 + 1])
                          : "r"(negZsz.i32),
                            "r"(R1.i32),
                            "r"(ADDR_1));
            }
        }
    }
    //------------------------------------------------------------------
    // get_bg_desc()
    static const bg_desc_t* get_bg_desc(int Z)
    {
        return get_adj_BG_desc<T, BG>(Z);
    }
    //------------------------------------------------------------------
    // Data
    const bg_desc_t& bg_desc;
    const word_t     negZsz;
    const word_t     tIdx_tIdx;
};


} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_DP_DESC_CUH_INCLUDED_)
