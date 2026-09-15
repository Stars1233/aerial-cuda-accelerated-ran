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


#if !defined(LDPC2_APP_ADDRESS_P_DESC_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_P_DESC_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"

////////////////////////////////////////////////////////////////////////
// APP address calculation using the predicate variables.
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
// Below, COND(Z), means:
//    COND(Z) = Z * [(threadIdx < (Z-shift)) ? 1 : 0]
//
// APPROACH:
//  REQUIRED ADDRESS GENERATION VARIABLES:
//    R0 =  -(Z*sz)                                                     (constant for all addresses and threads)
//                                                                      Note that this is only stored as a negative
//                                                                      value because the descriptor structure passed
//                                                                      as a kernel argument for a different address
//                                                                      calculator (dp_desc) requires a negative value.
//                                                                      Keeping it negative here has no effect on the
//                                                                      number of instructions, and allows us to use the
//                                                                      same structure.
//    THREADIDX                                                         (per-thread)
//  PER-NODE STORAGE
//    COL_IDX_SHIFT_LOW  = ((col_idx[0] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
//    COL_IDX_SHIFT_HIGH = ((col_idx[1] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
//    WRAP_INDEX = Z - shift
//
//
// ADDR_0 = threadIdx * sizeof(T) + COL_IDX_SHIFT_LOW  (muladd.lo)
// ADDR_1 = threadIdx * sizeof(T) + COL_IDX_SHIFT_HIGH (muladd.lo)
// P0, P1 = HSETP2.LT(threadIdx, wrapIndex)                  (threadIdx < wrap_index) ? : true : false
// @P0 sub.s32 ADDR_0, ADDR_0, -(Z*sz)                       Add Z*sz only if (threadIdx < wrap_index)
// @P1 sub.s32 ADDR_1, ADDR_1, -(Z*sz)                       Add Z*sz only if (threadIdx < wrap_index)

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_loc_address_p_desc
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG>
struct app_loc_address_p_desc
{
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    typedef BG_adj_desc<BG> bg_desc_t;
    //------------------------------------------------------------------
    // app_loc_address_p_desc()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_p_desc(const LDPC_kernel_params& params,
                            const bg_desc_t&          bgd,
                            unsigned int              t_idx) : bg_desc(bgd),
                                                               negZsz(-params.Z * sizeof(T)),
                                                               tIdx_tIdx(h0_h0(t_idx))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_p_desc()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_p_desc(const cuphyLDPCDecodeConfigDesc_t& config,
                            const bg_desc_t&                   bgd,
                            unsigned int                       t_idx) : bg_desc(bgd),
                                                                        negZsz(-config.Z * sizeof(T)),
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
            word_t WRAP_INDEX;
            int32_t BASE_ADDR_0, BASE_ADDR_1, ADDR_0, ADDR_1;

            const int32_t COL_IDX_SHIFT_LOW  = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_low;
            const int32_t COL_IDX_SHIFT_HIGH = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_high;
            WRAP_INDEX.u32                   = bg_desc.nodes[PAIR_OFFSET + i].wrap_index;

            ADDR_0 = muladd_lo_s32(threadIdx.x, sizeof(T), COL_IDX_SHIFT_LOW);
            ADDR_1 = muladd_lo_s32(threadIdx.x, sizeof(T), COL_IDX_SHIFT_HIGH);
            BASE_ADDR_0 = ADDR_0;
            BASE_ADDR_1 = ADDR_1;

            LDPC2_ASM("{\n\t\t"
                      ".reg .pred tidx_lt_wrap_0, tidx_lt_wrap_1;\n\t\t"
                      "setp.ltu.f16x2 tidx_lt_wrap_0|tidx_lt_wrap_1, %2, %3;\n\t\t"  // compare x2 thread idx to wrap values
                      "@tidx_lt_wrap_0 sub.s32 %0, %0, %4;\n\t\t"                    // if tidx_0 < wrap_0, sub (-Z * sz) from addr
                      "@tidx_lt_wrap_1 sub.s32 %1, %1, %4;\n\t\t"                    // if tidx_1 < wrap_1, sub (-Z * sz) from addr
                      "}\n"
                      : "+r"(ADDR_0), "+r"(ADDR_1)
                      : "r"(tIdx_tIdx.u32), "r"(WRAP_INDEX.u32), "r"(negZsz));
            app_addr[i*2] = ADDR_0;
            if(((i * 2) + 1) < ROW_DEGREE)
            {
                app_addr[i*2 + 1] = ADDR_1;
            }

            //if((0 == CHECK_IDX) && (0 == threadIdx.x) && (0 == i))
            //{
            //    printf("threadIdx = %u, i = (%i, %i), WRAP_INDEX = (%u, %u), COL_INDEX_SHIFT = (%i, %i), BASE_ADDR = (%i, %i), APP_ADDR = (%i, %i), negZsz = %i\n",
            //           threadIdx.x,
            //           i * 2 + 1,
            //           i * 2,
            //           WRAP_INDEX.u32 >> 16,
            //           WRAP_INDEX.u32 & 0x0000FFFF,
            //           COL_IDX_SHIFT_HIGH,
            //           COL_IDX_SHIFT_LOW,
            //           BASE_ADDR_0,
            //           BASE_ADDR_1,
            //           ADDR_0,
            //           ADDR_1,
            //           negZsz);
            //}
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
    const int32_t    negZsz;
    const word_t     tIdx_tIdx;
};


} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_P_DESC_CUH_INCLUDED_)
