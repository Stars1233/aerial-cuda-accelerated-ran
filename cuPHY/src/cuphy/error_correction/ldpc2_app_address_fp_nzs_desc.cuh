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


#if !defined(LDPC2_APP_ADDRESS_FP_NZS_DESC_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_FP_NZS_DESC_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"
#include "ldpc2_app_address_nzs_util.cuh"

// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)    (threadIdx + shift) < Z
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)   (threadIdx + shift) >= Z
//
// RA = shift + threadIdx
// RB = ((threadIdx + shift)  >= Z) ? 1.0 : 0.0
// RC = RA - RB * Z
// addr_app = (RC * sizeof(T)) + (col_app * Z * sizeof(T))
//
// OR:
//
// addr_app = [(col_app * Z * sizeof(T)) + (shift + threadIdx    )*sizeof(T)]     threadIdx < (Z-shift)    (threadIdx + shift) < Z
// addr_app = [(col_app * Z * sizeof(T)) + (shift + threadIdx - Z)*sizeof(T)]     threadIdx >= (Z-shift)   (threadIdx + shift) >= Z
//
// OR:
//

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
//         - the maximum address can be stored for all (BG, Z) for half
//         - the maximum address CANNOT be stored for half2
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
// 4.) Max column for the address of a 32-bit APP element (e.g. float or half2):
// USHRT_MAX = 2^16 - 1 = 65535
// MAX_COL_IDX_32 = floor(USHRT_MAX / (384 * sizeof(half2)))
//                = 42

////////////////////////////////////////////////////////////////////////
// APPROACH 1
// CONSTANT FOR ALL ADDRESSES:
// R0 = [Z - threadIdx | Z - threadIdx] (UNIFORM, +, DENORM_FP16(half, float, half2))
// R1 = [            Z |             Z] (UNIFORM, +, DENORM_FP16(half, float, half2))
// R2 = [    threadIdx |     threadIdx] (UNIFORM, +, DENORM_FP16(half, float, half2))
// PER NODE CONSTANTS:
// shift:                    (VARIABLE, +, DENORM_FP16(half, float, half2))
// col_idx * Z * sizeof(T) : (VARIABLE, +, DENORM_FP16(half))
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx - Z) * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// RC = (R0 <= shift) ? 1.0 : 0.0                           HSET2.BF
// RD = (shift + threadIdx) = (shift + R2)                  HADD2 or IADD?  (always positive, always can be denorm float)
// RD = RD - (RC * Z) = RD - (RC * R1)                      HFMA            (always positive, always can be denorm float)
// addr_app = (RD * sizeof(T)) + (col_idx * Z * sizeof(T))  IMAD
//
////////////////////////////////////////////////////////////////////////
// APPROACH 2
// CONSTANT FOR ALL ADDRESSES:
// RA = (Z - threadIdx) * sizeof(T) (UNIFORM, +, DENORM_FP16(half, float, half2))
// TS = threadIdx * sizeof(T)       (UNIFORM, +, DENORM_FP16(half, float, half2))
// ZS = Z * sizeof(T)               (UNIFORM, +, DENORM_FP16(half, float, half2))
// STORED CONSTANTS PER NODE:
// SS = shift * sizeof(T)           (VARIABLE, +, DENORM_FP16(half, float, half2))
// CS = col_idx * Z * sizeof(T)     (VARIABLE, +, DENORM_FP16(half))
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx - Z) * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [((col_idx * Z  + shift) * sizeof(T)) + (threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [((col_idx * Z  + shift) * sizeof(T)) + (threadIdx - Z) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift

// RB = (RA <= SS) ? 1.0 : 0.0                              HSET2.BF
// RC = (SS + TS) = (shift + threadIdx)*sizeof(T)           HADD2 or IADD? (always positive, max value approx. 2*384*sizeof(T))
// RC = RC - RB * ZS                                        HFMA           (always positive, max value approx. 384 * sizeof(T), always can be denorm float)
// addr_app = RC + CS                                       IADD

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_loc_address_fp_nzs_desc
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG>
struct app_loc_address_fp_nzs_desc
{
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    typedef BG_nzs_desc<BG> bg_desc_t;
    //------------------------------------------------------------------
    // app_loc_address_fp_nzs_desc()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_fp_nzs_desc(const LDPC_kernel_params& params,
                                const bg_desc_t&          bgd,
                                unsigned int              t_idx) : bg_desc(bgd),
                                                                   Z_Z(h0_h0(params.Z)),
                                                                   Z(params.Z),
                                                                   tIdx(h0_h0(t_idx)),
                                                                   Z_tIdx(h0_h0(params.Z - t_idx))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_fp_nzs_desc()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_fp_nzs_desc(const cuphyLDPCDecodeConfigDesc_t& config,
                                const bg_desc_t&                   bgd,
                                unsigned int                       t_idx) : bg_desc(bgd),
                                                                            Z_Z(h0_h0(config.Z)),
                                                                            Z(config.Z),
                                                                            tIdx(h0_h0(t_idx)),
                                                                            Z_tIdx(h0_h0(config.Z - t_idx))
    {
    }
    //------------------------------------------------------------------
    // generate()
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        // R0 = [Z - threadIdx | Z - threadIdx] (UNIFORM, +, DENORM_FP16(half, float, half2))
        // R1 = [            Z |             Z] (UNIFORM, +, DENORM_FP16(half, float, half2))
        // R2 = [    threadIdx |     threadIdx] (UNIFORM, +, DENORM_FP16(half, float, half2))
        // PER NODE CONSTANTS:
        // shift:                    (VARIABLE, +, DENORM_FP16(half, float, half2))
        // col_idx * Z * sizeof(T) : (VARIABLE, +, DENORM_FP16(half))
        //
        // addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
        // addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx - Z) * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
        //
        // RC = (R0 <= shift) ? 1.0 : 0.0                           HSET2.BF
        // RD = (shift + threadIdx) = (shift + R2)                  HADD2 or IADD?  (always positive, always can be denorm float)
        // RD = RD - (RC * Z) = RD + ((-RC) * Z)                    HFMA            (always positive, always can be denorm float)
        // addr_app = (RD * sizeof(T)) + (col_idx * Z * sizeof(T))  IMAD
        //const int ROW_DEGREE     = row_degree<BG, CHECK_IDX>::value;
        const int NZS_ROW_DEGREE = nzs_row_degree<BG, CHECK_IDX>::value;
        const int PAIR_OFFSET    = nzs_row_pair_index<BG, CHECK_IDX>::value;
        #pragma unroll
        //for(int i = 0; i < (ROW_DEGREE + 1) / 2; ++i)
        for(int i = 0; i < (NZS_ROW_DEGREE / 2); ++i)
        {
            uint32_t col_Z_sz = bg_desc.nodes[PAIR_OFFSET + i].col_Z_sz;

            word_t shift_mod, RC, RD, addrLow, addrPair;
            shift_mod.u32 = bg_desc.nodes[PAIR_OFFSET + i].shift_mod;
            RC            = hset2_bf_le(Z_tIdx, shift_mod); // RC   = (R0 <= shift) ? 1.0 : 0.0         HSET2.BF
#if 1
            RD            = hadd2(shift_mod, tIdx);         // RD   = (shift + threadIdx)               HADD2 or IADD
#else
            RD.u32        = shift_mod.u32 + tIdx.u32;       //
#endif
            RD            = hfma2(hneg2(RC), Z_Z, RD);      // RD   = RD - (RC * Z) = RD + ((-RC) * Z)  HFMA
            addrPair.u32  = RD.u32 * sizeof(T) + col_Z_sz;  // addr = (RD * sizeof(T)) + (col_Z_sz)     IMAD
            addrLow       = set_high_zero(addrPair);
            app_addr[i*2] = addrLow.i32;
            app_addr[(i * 2) + 1] = static_cast<int>(addrPair.u32 >> 16);
        }
        // For parity check rows with an odd non-zero shift row degree,
        // the last entry is stored differently and processed as a single
        // value instead of a SIMD pair.
        single_address_generator<BG, CHECK_IDX, T>::generate(Z, bg_desc, app_addr);

        // Generate address for zero-shift base graph nodes
        zs_generator<BG, CHECK_IDX, T>::generate(Z, app_addr);
    }
    //------------------------------------------------------------------
    // get_bg_desc()
    static const bg_desc_t* get_bg_desc(int Z)
    {
        return get_BG_nzs_desc<T, BG>(Z);
    }
private:
    //------------------------------------------------------------------
    // Data
    const bg_desc_t& bg_desc;
    const word_t     Z_Z;
    const int        Z;
    const word_t     tIdx;
    const word_t     Z_tIdx;

};


} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_FP_NZS_DESC_CUH_INCLUDED_)
