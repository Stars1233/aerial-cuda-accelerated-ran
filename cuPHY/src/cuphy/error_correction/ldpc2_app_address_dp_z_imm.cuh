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

#if !defined(LDPC2_APP_ADDRESS_DP_Z_IMM_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_DP_Z_IMM_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"

////////////////////////////////////////////////////////////////////////
// Fixed-Z compile-time-immediate variant of the dp_desc APP address
// calculator (ldpc2_app_address_dp_desc.cuh). Fixed-Z helper: dispatch
// pins the lifting size (Z = Z_FIXED), so every descriptor value the
// generic dp_desc path loads at runtime -- the packed wrap-index pair
// and the two Z-biased column/shift byte offsets per edge pair -- is a
// compile-time constant here, taken from the SAME nrLDPC_templates.cuh
// metafunctions that generate the checked-in BG_adj_desc tables
// (ldpc2_bg_desc_half2.cpp):
//
//   descriptor field                 compile-time source
//   -------------------------------  ---------------------------------
//   nodes[P].wrap_index              wrap_index_pair<BG,Z,ROW,P>
//   nodes[P].col_Z_shift_low         vnode_adj_shift_offset   <BG,Z,ROW,2P>   * sizeof(T)
//   nodes[P].col_Z_shift_high        vnode_adj_shift_offset_if<BG,Z,ROW,2P+1> * sizeof(T)
//
// Every emitted APP byte address is therefore BIT-IDENTICAL to the
// generic dp_desc path's; only the operand source changes (immediate
// instead of a per-row per-iteration LDC/LDCU of the kernel-parameter
// descriptor). Per non-degenerate pair this removes all three constant
// loads while keeping the measured-best HSET2 + dp2a wrap-add form:
//
//   dp_desc (runtime desc)                dp_z_imm (this file)
//   ------------------------------------  -----------------------------------
//   LDC   wrap_index                      (immediate)
//   LDCU  col_Z_shift_low                 (immediate)
//   LDCU  col_Z_shift_high                (immediate)
//   IADD  ADDR_0 = tIdx_sz + low          IADD  ADDR_0 = tIdx_sz + IMM_low
//   HSET2 R1 = tIdx < wrap ? mask : 0     HSET2 R1 = tIdx < IMM_wrap ? mask : 0
//   IDP   addr_lo (dp2a.lo)               IDP   addr_lo (dp2a.lo)
//   IADD  ADDR_1 = tIdx_sz + high         IADD  ADDR_1 = tIdx_sz + IMM_high
//   IDP   addr_hi (dp2a.hi)               IDP   addr_hi (dp2a.hi)
//
// In addition, the compile-time shift lets EVERY zero-shift edge (not
// just the always-zero-shift final edge of each BG1 row) skip the wrap
// machinery: a zero-shift edge never wraps (threadIdx < Z = wrap
// index), so its address is exactly COL*(Z*sizeof(T)) + tIdx*sizeof(T),
// a single IADD with an immediate. This is the generalization of the
// dp_desc last-edge fast path, whose bit-identity argument applies
// unchanged: for a zero-shift edge the generic path computes
// ((COL-1)*Z + 0)*sizeof(T) + tIdx*sizeof(T) and then unconditionally
// adds Z*sizeof(T) because the no-wrap branch is always taken.
//
// The bg_desc_t kernel argument and get_bg_desc() are retained
// verbatim so kernel signatures, host launch paths, and graph-capture
// argument marshalling stay unchanged; the descriptor argument is
// simply no longer read by generate() (its loads vanish from SASS).

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_loc_address_dp_z_imm
// Manager for calculation of APP locations (in shared memory) with all
// per-edge descriptor values folded to compile-time immediates for a
// fixed lifting size Z_FIXED. Drop-in replacement for
// app_loc_address_dp_desc when dispatch guarantees Z == Z_FIXED.
template <typename T, int BG, int Z_FIXED>
struct app_loc_address_dp_z_imm
{
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    // (kept for interface compatibility; not read by generate()).
    typedef BG_adj_desc<BG> bg_desc_t;
    //------------------------------------------------------------------
    // app_loc_address_dp_z_imm()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_dp_z_imm(const LDPC_kernel_params& params,
                             const bg_desc_t&          bgd,
                             unsigned int              t_idx) :
        negZsz(to_word(make_short2(-Z_FIXED * static_cast<int>(sizeof(T)), 0))),
        tIdx_tIdx(h0_h0(t_idx)),
        tIdx_sz(static_cast<int>(t_idx) * static_cast<int>(sizeof(T)))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_dp_z_imm()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_dp_z_imm(const cuphyLDPCDecodeConfigDesc_t& config,
                             const bg_desc_t&                   bgd,
                             unsigned int                       t_idx) :
        negZsz(to_word(make_short2(-Z_FIXED * static_cast<int>(sizeof(T)), 0))),
        tIdx_tIdx(h0_h0(t_idx)),
        tIdx_sz(static_cast<int>(t_idx) * static_cast<int>(sizeof(T)))
    {
    }
    //------------------------------------------------------------------
    // gen_pair()
    // Emit the address pair (edges 2*I, 2*I+1) of row CHECK_IDX. I is a
    // template parameter so the packed wrap-index pair, the Z-biased
    // byte offsets, and the per-edge shift==0 test are all evaluated at
    // compile time.
    template <int CHECK_IDX, int I>
    __device__ __forceinline__
    void gen_pair(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        constexpr int  ROW_DEGREE = row_degree<BG, CHECK_IDX>::value;
        constexpr int  E_LO       = I * 2;
        constexpr int  E_HI       = I * 2 + 1;
        constexpr bool HI_VALID   = (E_HI < ROW_DEGREE);
        constexpr int  E_HI_SAFE  = HI_VALID ? E_HI : 0; // avoid OOB template inst.

        // Compile-time per-edge shift (mod Z) == 0 test.
        constexpr bool LO_ZERO = (vnode_shift_mod_if<BG, Z_FIXED, CHECK_IDX, E_LO>::value == 0);
        constexpr bool HI_ZERO = (!HI_VALID) ||
                                 (vnode_shift_mod_if<BG, Z_FIXED, CHECK_IDX, E_HI_SAFE>::value == 0);

        // Plain byte offsets for zero-shift edges: COL * (Z * sizeof(T)).
        constexpr int OFF0_LO = static_cast<int>(vnode_index<BG, CHECK_IDX, E_LO>::value) *
                                Z_FIXED * static_cast<int>(sizeof(T));
        constexpr int OFF0_HI = static_cast<int>(vnode_index<BG, CHECK_IDX, E_HI_SAFE>::value) *
                                Z_FIXED * static_cast<int>(sizeof(T));

        // Z-biased byte offsets for wrapping edges: exactly the values the
        // checked-in descriptor stores in col_Z_shift_low / col_Z_shift_high.
        constexpr int32_t ADJ_LO = vnode_adj_shift_offset<BG, Z_FIXED, CHECK_IDX, E_LO>::value *
                                   static_cast<int32_t>(sizeof(T));
        constexpr int32_t ADJ_HI = static_cast<int32_t>(vnode_adj_shift_offset_if<BG, Z_FIXED, CHECK_IDX, E_HI_SAFE>::value) *
                                   static_cast<int32_t>(sizeof(T));

        if constexpr(LO_ZERO && HI_ZERO)
        {
            // Neither edge wraps: pure compile-time offsets (no wrap
            // compare, no dp2a).
            app_addr[E_LO] = tIdx_sz + OFF0_LO;
            if constexpr(HI_VALID)
            {
                app_addr[E_HI] = tIdx_sz + OFF0_HI;
            }
        }
        else
        {
            // At least one edge wraps. The packed wrap-index pair is the
            // exact u16x2 value the descriptor stores; the unused lane of
            // a mixed (zero/non-zero) pair is harmless.
            word_t WRAP_INDEX;
            WRAP_INDEX.u32 = wrap_index_pair<BG, Z_FIXED, CHECK_IDX, I>::value;
            const word_t R1 = hset2_bm_lt(tIdx_tIdx, WRAP_INDEX); // (threadIdx < WRAP_INDEX) ? -1 : 0

            // The dp2a accumulates the wrap add onto the SHARED per-thread
            // base (tIdx*sizeof(T)) and the per-edge column/shift byte
            // offset is applied as the FINAL immediate add:
            //     (tIdx_sz + wrap_add) + ADJ  ==  (tIdx_sz + ADJ) + wrap_add
            // (exact int32 identity). Trailing the immediate lets ptxas
            // fold it into the consuming LDS/STS addressing-mode offset
            // ([R+UR+imm]), deleting the per-edge IADD from the schedule;
            // if it declines, the same single IADD re-materializes and the
            // sequence matches the descriptor path op-for-op.
            if constexpr(LO_ZERO)
            {
                app_addr[E_LO] = tIdx_sz + OFF0_LO;
            }
            else
            {
                int32_t WBASE_LO;
                //   dp2a.lo: output = (RB.c * RA.high) + (RB.d * RA.low)
                LDPC2_ASM("\t"
                          "{\n\t\t"
                          "dp2a.lo.s32.s32 %0, %1, %2, %3;\n\t"
                          "}\n"
                          : "=r"(WBASE_LO)
                          : "r"(negZsz.i32),
                            "r"(R1.i32),
                            "r"(tIdx_sz));
                app_addr[E_LO] = WBASE_LO + ADJ_LO;
            }

            if constexpr(HI_VALID)
            {
                if constexpr(HI_ZERO)
                {
                    app_addr[E_HI] = tIdx_sz + OFF0_HI;
                }
                else
                {
                    int32_t WBASE_HI;
                    //   dp2a.hi: output = (RB.a * RA.high) + (RB.b * RA.low)
                    LDPC2_ASM("\t"
                              "{\n\t\t"
                              "dp2a.hi.s32.s32 %0, %1, %2, %3;\n\t"
                              "}\n"
                              : "=r"(WBASE_HI)
                              : "r"(negZsz.i32),
                                "r"(R1.i32),
                                "r"(tIdx_sz));
                    app_addr[E_HI] = WBASE_HI + ADJ_HI;
                }
            }
        }
    }
    //------------------------------------------------------------------
    // gen_loop()
    template <int CHECK_IDX, int I, int NPAIRS>
    __device__ __forceinline__
    void gen_loop(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        if constexpr(I < NPAIRS)
        {
            gen_pair<CHECK_IDX, I>(app_addr);
            gen_loop<CHECK_IDX, I + 1, NPAIRS>(app_addr);
        }
    }
    //------------------------------------------------------------------
    // generate()
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        constexpr int NPAIRS = (row_degree<BG, CHECK_IDX>::value + 1) / 2;
        gen_loop<CHECK_IDX, 0, NPAIRS>(app_addr);
    }
    //------------------------------------------------------------------
    // get_bg_desc()
    static const bg_desc_t* get_bg_desc(int Z)
    {
        return get_adj_BG_desc<T, BG>(Z);
    }
    //------------------------------------------------------------------
    // Data
    const word_t negZsz;
    const word_t tIdx_tIdx;
    const int    tIdx_sz;   // threadIdx.x * sizeof(T), precomputed once (ALU-pipe base add)
};

} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_DP_Z_IMM_CUH_INCLUDED_)
