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

// BG1 / Z=256..384 (runtime; the five 38.212 liftings 256/288/320/352/384),
// p=4..11 (runtime), 2 codewords per CTA exact box-plus layered decoder.
// LDPC_ALGO_BG1_Z256UP_BP_ALLREG_X2 = 201. (The Z256to352 spelling this file
// grew up with survives only in internal helper names below; it is stale --
// the id is 201 and the zone reaches 384.)
//
// ONE kernel, no per-Z or per-p specialization: the evolved all-register
// body -- deferred normalization, register-resident APP columns,
// iteration-0 puncture collapse, final-pass dead-store elision, V24
// forwarding -- with the compile-time Z-immediate addressing replaced by the
// runtime bg_desc descriptor path (see a201_app_addr), and the row band
// extended to row 10. This TU #includes ldpc2_algo103.cuh and reuses its
// box-plus primitives, TB-token search, scalar-of-config norm, APP layout,
// and all-warps writer.
//
// WHY THE BAND ENDS AT p=11 (column-ownership expiry -- MEASURED, do not
// extend without redesign): the register-resident column and dead-store
// mechanisms rest on compile-time proofs that specific parity columns are
// touched ONLY by specific rows. Those proofs expire as deeper rows enter
// the schedule (BG1 connectivity, verified from nrLDPC_templates.cuh):
//   V23 (resident slot 0, rows 0/1 elide their shared stores): read by
//        row 11 (p>=12) and row 13 (p>=14) -- a p=12 probe build measured
//        LLR_CHECK FAIL at ~18 dB SNR (trajectory divergence: row 11 reads
//        the stale shared copy while rows 0/1 use the register mirror).
//   V22 (dead-store chain r0->r1->r3->r5->r8): read again by row 11
//        (p>=12), row 16 (p>=17), row 20 (p>=21).
//   V25 (resident slot 1, rows 2/3): read by row 15 (p>=16).
//   V24 (forwarding, rows 1/2 + row 8): clean through row 22.
// Rows 9 and 10 touch none of the protected columns (p=11 verified clean:
// llr_check PASS, waterfall equivalent to box-plus). p >= 12 at these Z
// belongs to the E5M5-family band decoder (bg1_z256up_bp_gmext_x2), whose mechanisms are
// p-aware by design.

#include <array>

#include "ldpc2_algo103.cuh"   // box-plus prims, token search, norm, output, APP
#include "ldpc2_app_address_dp_desc.cuh" // BG_adj_desc tables + get_adj_BG_desc(Z)
#include "nrLDPC_templates.cuh"
#include "ldpc2_algo201.hpp"

namespace
{
    static constexpr int A201_MIN_PARITY = 4;
    static constexpr int A201_MAX_PARITY = 11;

    // Top of the Z zone. blockDim == Z at launch, so this is both the
    // __launch_bounds__ thread count and the largest Z the kernel may see.
    //
    //   A201_Z_MAX=352 -> the original zone {256,288,320,352}
    //   A201_Z_MAX=384 -> {256,288,320,352,384}, i.e. every lifting >= 256
    //
    // The original 352 was chosen on a FALSE premise, preserved here as a
    // warning: the code claimed "register budget = 65536/352 -> 184 for every Z
    // in the zone", i.e. that folding Z=384 in would cost ~16 registers. It does
    // not. Registers are allocated per warp within a subpartition, so the
    // ceiling is ceil(warps/4), and Z=288..384 (9..12 warps) ALL land on 168.
    // Measured: this kernel compiles to REG:168 / STACK:0 either way. Extending
    // to Z=384 costs no registers at all.
#ifndef A201_Z_MAX
#define A201_Z_MAX 384
#endif
    static constexpr int A201_MAX_THREADS_PER_CTA = A201_Z_MAX;
    static constexpr int A201_MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Compile-time cumulative edge offset: edge_base<R> = sum of degrees of
    // rows 0..R-1. Used to index the flat register C2V array.
    template <int R> struct edge_base
    {
        static constexpr int value = edge_base<R - 1>::value + row_degree<ALGO103_BG, R - 1>::value;
    };
    template <> struct edge_base<0> { static constexpr int value = 0; };

    static constexpr int A201_TOTAL_EDGES = edge_base<A201_MAX_PARITY>::value; // 113

    // Rows 4..10 each end in one isolated degree-1 parity edge. That edge's
    // C2V is never read by a later gather, so the register C2V frame stores
    // only update_row_degree entries per row while the APP address schedule
    // still walks the full row_degree edge list.
    template <int R> struct c2v_base
    {
        static constexpr int value =
            c2v_base<R - 1>::value + update_row_degree<ALGO103_BG, R - 1>::value;
    };
    template <> struct c2v_base<0> { static constexpr int value = 0; };

    static constexpr int A201_TOTAL_C2V_WORDS = c2v_base<A201_MAX_PARITY>::value;
    static_assert(A201_TOTAL_C2V_WORDS == A201_TOTAL_EDGES - (A201_MAX_PARITY - A201_MIN_PARITY),
                  "compact C2V layout assumes one isolated edge in every non-core row");

    //------------------------------------------------------------------
    // Deferred normalization: C2V stores the RAW box-plus result and the
    // min-sum scale is folded into both consumers with one fused fp16x2 op
    // each -- the posterior update (norm*raw + v2c) and the next-iteration
    // V2C subtract (app - norm*raw) -- replacing {scale, add} + {subtract}
    // with two HFMA2 and rounding the scaled message once per use instead of
    // once at the store plus once per use.
    __device__ __forceinline__
    word_t h2_fma_norm(word_t a, __half2 norm, word_t b)   // a*norm + b
    {
        word_t out;
        out.f16x2 = __hfma2(half2_from_raw(a.f16x2), norm, half2_from_raw(b.f16x2));
        return out;
    }

    __device__ __forceinline__
    word_t h2_fnma_norm(word_t a, __half2 norm, word_t b)  // b - a*norm
    {
        word_t out;
        out.f16x2 = __hfma2(__hneg2(half2_from_raw(a.f16x2)), norm, half2_from_raw(b.f16x2));
        return out;
    }

    //------------------------------------------------------------------
    // Iteration-invariant APP addressing (BG1, RUNTIME Z = 256..384).
    //
    // Every edge's APP location is app[V*Z + ((z + S) mod Z)]; V is a
    // compile-time column index but S = base_shift mod Z and Z itself are
    // runtime here (each Z is a different 38.212 lifting set), so the
    // per-edge (col_Z_shift, wrap_index) values come from the checked-in
    // BG_adj_desc runtime descriptor tables (get_adj_BG_desc(Z), which cover
    // every lifting size) instead of compile-time immediates. This is the
    // dp_desc form (ldpc2_app_address_dp_desc.cuh) with one difference: the
    // shared-window byte address of app[z] (cvta of the APP buffer +
    // z*sizeof(word_t)) is folded into the additive base ONCE per kernel
    // (member tbase, replacing dp_desc's tIdx_sz), so the generated
    // addresses are ABSOLUTE shared-window addresses and every consumer of
    // the inherited body (smem_address_as / write_shared_word) is
    // unchanged. The addresses are bit-identical to dp_desc's + base.
    //
    // Per edge pair: one LDCU descriptor load, one packed HSET2 no-wrap
    // test, one dp2a per edge folding the +Z*sizeof(word_t) adjustment.
    // BG1's last edge per row is the identity/parity column with shift 0 in
    // EVERY lifting set: its address is the descriptor-free fast path
    // V*Zsz + tbase (single IMAD).
    static constexpr int A201_WSZ = static_cast<int>(sizeof(word_t));

    typedef ldpc2::BG_adj_desc<1> a201_bg_desc_t;

    template <int R, int POS>
    struct a201_shift0_zone
    {
        static constexpr bool value =
            ((vnode_shift<ALGO103_BG, set_index<256>::value, R, POS>::value % 256) == 0) &&
            ((vnode_shift<ALGO103_BG, set_index<288>::value, R, POS>::value % 288) == 0) &&
            ((vnode_shift<ALGO103_BG, set_index<320>::value, R, POS>::value % 320) == 0) &&
            ((vnode_shift<ALGO103_BG, set_index<352>::value, R, POS>::value % 352) == 0) &&
            // Z=384 is a DIFFERENT 38.212 lifting set from 352, so it needs its
            // own term -- this predicate is the real reason the kernel could not
            // simply be pointed at Z=384, not the register budget above.
            ((A201_Z_MAX < 384) ||
             ((vnode_shift<ALGO103_BG, set_index<384>::value, R, POS>::value % 384) == 0));
    };

    static_assert(a201_shift0_zone<1, 16>::value, "row 1 V22 is shift-0 for the ALGO201 Z zone");
    static_assert(a201_shift0_zone<1, 17>::value, "row 1 V23 is shift-0 for the ALGO201 Z zone");

    struct a201_app_addr
    {
        const a201_bg_desc_t& bg_desc; // runtime per-Z descriptor table
        word_t tt;      // (z, z) packed u16x2 for the HSET2 wrap compare
        word_t negZsz;  // short2(-Z*sizeof(word_t), 0) dp2a wrap-add operand
        int    tbase;   // shared-window byte address of app[z]
        int    Zsz;     // Z * sizeof(word_t) (runtime)

        __device__ __forceinline__
        a201_app_addr(const a201_bg_desc_t& bgd, word_t* app, int z, int Z) :
            bg_desc(bgd),
            tt(h0_h0(z)),
            negZsz(to_word(make_short2(-Z * A201_WSZ, 0))),
            tbase(static_cast<int>(__cvta_generic_to_shared(app)) + (z * A201_WSZ)),
            Zsz(Z * A201_WSZ)
        {
        }

        // generate(): all edge addresses of row R, computed once per row visit
        // and shared by the forward (gather) and backward (scatter) sweeps.
        template <int R>
        __device__ __forceinline__
        void generate(int (&addr)[row_degree<ALGO103_BG, R>::value]) const
        {
            constexpr int D           = row_degree<ALGO103_BG, R>::value;
            constexpr int PAIR_OFFSET = row_pair_index<ALGO103_BG, R>::value;
            constexpr int LAST_EDGE   = D - 1;
            #pragma unroll
            for(int i = 0; i < (D + 1) / 2; ++i)
            {
                const int LO = i * 2;
                const int HI = i * 2 + 1;

                if constexpr(R == 1)
                {
                    if(LO == 16)
                    {
                        addr[16] = vnode_index<ALGO103_BG, 1, 16>::value * Zsz + tbase;
                        addr[17] = vnode_index<ALGO103_BG, 1, 17>::value * Zsz + tbase;
                        continue;
                    }
                }

                // Shift-0 last-edge fast path: the last variable node of every
                // BG1 parity row is the row's identity/parity column with
                // circulant shift 0 in ALL lifting sets -- no descriptor load,
                // no wrap machinery.
                if(LO == LAST_EDGE)
                {
                    addr[LO] = vnode_index<ALGO103_BG, R, LAST_EDGE>::value * Zsz + tbase;
                    continue;
                }

                const int32_t CZS_LO = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_low;
                word_t        WRAP;
                WRAP.u32 = bg_desc.nodes[PAIR_OFFSET + i].wrap_index;

                const word_t  nw     = hset2_bm_lt(tt, WRAP); // (z < Z-S) ? 0xFFFF : 0 per lane
                int32_t       WBASE_0;

                LDPC2_ASM("dp2a.lo.s32.s32 %0, %1, %2, %3;"
                          : "=r"(WBASE_0)
                          : "r"(negZsz.i32), "r"(nw.i32), "r"(tbase));
                addr[LO] = WBASE_0 + CZS_LO;

                if(HI < D)
                {
                    if constexpr(R == 2)
                    {
                        if(HI == 17)
                        {
                            addr[17] = 24 * Zsz + tbase;
                            continue;
                        }
                    }
                    if(HI == LAST_EDGE)
                    {
                        addr[HI] = vnode_index<ALGO103_BG, R, LAST_EDGE>::value * Zsz + tbase;
                        continue;
                    }
                    const int32_t CZS_HI = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_high;
                    int32_t       WBASE_1;
                    LDPC2_ASM("dp2a.hi.s32.s32 %0, %1, %2, %3;"
                              : "=r"(WBASE_1)
                              : "r"(negZsz.i32), "r"(nw.i32), "r"(tbase));
                    addr[HI] = WBASE_1 + CZS_HI;
                }
            }
        }
    };

    //------------------------------------------------------------------
    // Register-resident same-thread APP state (compile-time BG1 geometry).
    //
    // Within rows 0..8 of BG1/Z=384, a few APP columns are touched ONLY by
    // rows that use the SAME cyclic shift, so one thread owns the value across
    // every visit and the shared-memory round trip carries no cross-thread
    // information:
    //
    //   slot 0: V=23, rows 0 (POS 18) and 1 (POS 17), both shift 0
    //   slot 1: V=25, rows 2 (POS 18) and 3 (POS 18), both shift 0
    //   slot 2..6: V=26..30, the degree-1 extension parity columns of rows
    //              4..8 (each touched by exactly one row, shift 0)
    //
    // For the degree-2 columns (slots 0/1) the register mirrors the APP value
    // most recently stored: the forward gather reads the register instead of
    // re-loading the word this thread itself stored (bit-exact store-to-load
    // forwarding). For the degree-1 extension columns the register holds the
    // V2C message, which is ITERATION-INVARIANT: app = v2c + c2v after every
    // update and only this edge touches the column, so the gather's
    // (app - c2v) always reproduces the channel LLR captured in iteration 0.
    // Keeping v2c directly removes the per-iteration reload AND the subtract,
    // and the edge's previous-C2V register is never read again (freed).
    //
    // In the TIMED kernel the resident columns' APP scatter to shared memory
    // is elided (see a201_backward: the value is carried in the register, and
    // the hard-output writer reads only the 22 info columns, which are all
    // shared-resident). Consequently the kernel's
    // shared APP image is STALE for these parity columns -- any future reader
    // of parity APP from shared (e.g. a soft-output path) must first remove
    // this elision.
    template <bool DUMP>
    struct a201_resident
    {
        word_t slot[DUMP ? 10 : 3];
    };

    // Slot index of the resident column for edge (R, POS); -1 when the edge
    // goes through shared memory. Degree-1 extension slots are 2..6; slot 7 is
    // the V24 forwarding-only register (see below).
    template <int R, int POS> struct a201_res_slot { static constexpr int value = -1; };
    template <> struct a201_res_slot<0, 18> { static constexpr int value = 0; }; // V=23
    template <> struct a201_res_slot<1, 17> { static constexpr int value = 0; }; // V=23
    template <> struct a201_res_slot<2, 18> { static constexpr int value = 1; }; // V=25
    template <> struct a201_res_slot<3, 18> { static constexpr int value = 1; }; // V=25
    template <> struct a201_res_slot<1, 18> { static constexpr int value = 2; }; // V=24 (forwarding)
    template <> struct a201_res_slot<2, 17> { static constexpr int value = 2; }; // V=24 (forwarding)
    template <> struct a201_res_slot<4, 2>  { static constexpr int value = 3; }; // V=26
    template <> struct a201_res_slot<5, 7>  { static constexpr int value = 4; }; // V=27
    template <> struct a201_res_slot<6, 8>  { static constexpr int value = 5; }; // V=28
    template <> struct a201_res_slot<7, 6>  { static constexpr int value = 6; }; // V=29
    template <> struct a201_res_slot<8, 9>  { static constexpr int value = 7; }; // V=30
    template <> struct a201_res_slot<9, 8>  { static constexpr int value = 8; }; // V=31
    template <> struct a201_res_slot<10, 6> { static constexpr int value = 9; }; // V=32

    // V24 (slot 2) is only PARTIALLY resident. Its band touches are rows
    // 1 (POS18, shift 0), 2 (POS17, shift 0), and 8 (POS8, shift 170 -- a
    // CROSS-THREAD rotation, executed when p == 9). The register therefore
    // forwards the value only along the same-shift consecutive pair
    // row1 -> row2: row 1's gather always RE-LOADS from shared (the previous
    // writer was row 2's published store or row 8's rotated store), and row
    // 2's update is always STORED (published for row 8 and for the next
    // iteration's row 1). Only row 1's timed store and row 2's load are
    // elided; the value row 2 consumes is the exact word row 1 stored
    // (bit-exact store-to-load forwarding), so p < 9 and p == 9 share one
    // branch-free body.
    template <int R, int POS> struct a201_res_fwd_load { static constexpr bool value = false; };
    template <> struct a201_res_fwd_load<1, 18> { static constexpr bool value = true; };
    template <int R, int POS> struct a201_res_publish { static constexpr bool value = false; };
    template <> struct a201_res_publish<2, 17> { static constexpr bool value = true; };

    static_assert(vnode_index<ALGO103_BG, 1, 18>::value == 24, "V24 forwarding site row1");
    static_assert(vnode_index<ALGO103_BG, 2, 17>::value == 24, "V24 forwarding site row2");
    static_assert(vnode_index<ALGO103_BG, 1, 17>::value == 23, "V23 shift-0 direct site row1");
    static_assert(vnode_index<ALGO103_BG, 9, 8>::value  == 31, "row9 identity extension V31");
    static_assert(vnode_index<ALGO103_BG, 10, 6>::value == 32, "row10 identity extension V32");

    //------------------------------------------------------------------
    // Final-iteration dead parity-column posterior stores.
    //
    // The Z384 hard-output writer reads ONLY the 22 info columns, so on the
    // LAST iteration of a hard-output-only decode a store to a parity column
    // (V >= 22) is dead unless a LATER row of the SAME (final) iteration
    // re-reads that column. The register-resident columns (V=23,25,26..30)
    // are already never stored in the timed kernel; the remaining shared
    // parity columns are V=22 (rows 0,1,3,5,8) and V=24 (rows 1,2,8). Per
    // BG1 rows 0..8 connectivity:
    //   V=22: row0 -> read by row1; row1 -> row3; row3 -> row5 (iff p>=6);
    //         row5 -> row8 (iff p==9); row8 -> nothing.
    //   V=24: row1 -> read by row2; row2 -> row8 (iff p==9); row8 -> nothing.
    // A site's final-iteration store is therefore dead iff p <= p_max below
    // (p_max==0: always consumed; p_max==9: dead whenever the row executes).
    // The store is RUNTIME-PREDICATED (setp + @p st.shared in one asm block,
    // no branch, no loop peeling). The stored value is computed identically
    // either way; only the provably unconsumed shared write is suppressed.
    template <int R, int POS> struct a201_last_dead { static constexpr int p_max = 0; };
    template <> struct a201_last_dead<3, 17> { static constexpr int p_max = 5; }; // V=22
    template <> struct a201_last_dead<5, 6>  { static constexpr int p_max = 8; }; // V=22
    template <> struct a201_last_dead<8, 7>  { static constexpr int p_max = 11; }; // V=22 (row 11 = first later reader, outside band)
    template <> struct a201_last_dead<2, 17> { static constexpr int p_max = 8; }; // V=24
    template <> struct a201_last_dead<8, 8>  { static constexpr int p_max = 11; }; // V=24 (no reader through row 22)

    static_assert(vnode_index<ALGO103_BG, 3, 17>::value == 22, "dead-store site row3/V22");
    static_assert(vnode_index<ALGO103_BG, 5, 6>::value  == 22, "dead-store site row5/V22");
    static_assert(vnode_index<ALGO103_BG, 8, 7>::value  == 22, "dead-store site row8/V22");
    static_assert(vnode_index<ALGO103_BG, 2, 17>::value == 24, "dead-store site row2/V24");
    static_assert(vnode_index<ALGO103_BG, 8, 8>::value  == 24, "dead-store site row8/V24");

    __device__ __forceinline__
    void a201_write_shared_word_if(word_t w, int offset, int en)
    {
        LDPC2_ASM_VOLATILE("{\n\t"
                           ".reg .pred p_st;\n\t"
                           "setp.ne.s32 p_st, %2, 0;\n\t"
                           "@p_st st.shared.b32 [%0], %1;\n\t"
                           "}"
                           :: "r"(offset), "r"(w.u32), "r"(en));
    }

    template <int R, bool FIRST, bool DUMP, int POS, int D>
    __device__ __forceinline__
    void a201_forward(const int (&addr)[D],
                      __half2 norm,
                      word_t* c2v,
                      a201_resident<DUMP>& res,
                      word_t (&v2c)[D],
                      word_t (&pre)[D],
                      word_t& acc)
    {
        constexpr int  B    = c2v_base<R>::value;
        constexpr int  RS   = a201_res_slot<R, POS>::value;
        constexpr bool DEG1 = (RS >= 3 && RS <= 9); // degree-1 extension slots only (slot 2 = V24 forwarding)
        // First row to touch a resident column loads the loader-staged value.
        constexpr bool RES_INIT = DEG1 || (R == 0 && POS == 18) || (R == 2 && POS == 18);

        word_t m;
        if constexpr(DEG1)
        {
            // Degree-1 extension column: v2c is iteration-invariant.
            if constexpr(!DUMP)
            {
                // Timed kernels never publish DEG1 parity APP stores, so the
                // shared slot remains the loader-staged channel value. Reload
                // it rather than preserving long-lived DEG1 registers that
                // otherwise spill across the guarded row tail.
                m = smem_address_as<word_t>(addr[POS]);
            }
            else if constexpr(FIRST)
            {
                m = smem_address_as<word_t>(addr[POS]); // channel LLR
                res.slot[RS] = m;
            }
            else
            {
                m = res.slot[RS];
            }
        }
        else
        {
            word_t p;
            if constexpr(RS < 0 || a201_res_fwd_load<R, POS>::value || (FIRST && RES_INIT))
            {
                p = smem_address_as<word_t>(addr[POS]);
            }
            else
            {
                p = res.slot[RS]; // value this thread stored on the last visit
            }
            // C2V holds the RAW box result; the min-sum scale is fused here.
            m = FIRST ? p : h2_fnma_norm(c2v[B + POS], norm, p);
        }
        if constexpr(POS > 0)
        {
            v2c[POS] = m;
        }
        pre[POS] = acc;
        if constexpr((POS == 0) && (R >= 4))
        {
            // Optional low-degree rows seed the prefix directly from the first
            // edge: box_plus(identity, m) == m, and pre[0] stays the identity
            // for the backward leave-one-out of edge 0.
            acc = m;
        }
        else
        {
            acc = box_pair(acc, m);
        }

        if constexpr((POS + 1) < D)
        {
            a201_forward<R, FIRST, DUMP, POS + 1>(addr, norm, c2v, res, v2c, pre, acc);
        }
    }

    template <int R, int POS, bool DUMP, int D>
    __device__ __forceinline__
    void a201_backward(const int (&addr)[D],
                       __half2 norm,
                       word_t* c2v,
                       a201_resident<DUMP>& res,
                       const word_t (&v2c)[D],
                       const word_t (&pre)[D],
                       word_t& suf,
                       int p,
                       int elide)
    {
        constexpr int  B    = c2v_base<R>::value;
        constexpr int  RS   = a201_res_slot<R, POS>::value;
        constexpr bool DEG1 = (RS >= 3 && RS <= 9); // degree-1 extension slots only (slot 2 = V24 forwarding)

        word_t cn;
        word_t msg;
        if constexpr(POS == 0)
        {
            // pre[0] is the box-plus identity, so the first edge's
            // leave-one-out is just the accumulated suffix.  Also, for every
            // row in this zone D > 1 and pre[1] is exactly v2c[0]; reusing it
            // avoids keeping v2c[0] live across the whole row body.
            cn  = suf;
            msg = pre[1];
        }
        else
        {
            cn  = box_pair(pre[POS], suf); // RAW leave-one-out
            msg = v2c[POS];
        }
        const word_t up = h2_fma_norm(cn, norm, msg); // norm*cn + v2c, one rounding
        // Redundant scatter elision: a register-resident column (RS>=0) is
        // single-thread-owned -- its updated APP is carried to the next visit
        // through res.slot (degree-2) or the iteration-invariant v2c register
        // (degree-1), and the Z384 hard-output writer reads ONLY the 22 info
        // columns. So this shared store is DEAD in the timed decode: the value
        // (up) is computed identically, but writing it back to shared is pure
        // redundant scatter-half APP work. Non-resident
        // columns (RS<0) are cross-thread and always stored, and the V24
        // forwarding slot's row-2 site always publishes -- except the
        // final-iteration dead parity sites (a201_last_dead), whose store is
        // runtime-predicated off when 'elide' says nothing can consume it
        // (row2/V24's published word is consumed only by p==9's row 8 on the
        // final pass, so its a201_last_dead p_max==8 entry applies here too).
        if constexpr(RS < 0 || a201_res_publish<R, POS>::value || DUMP)
        {
            constexpr int DEAD_P_MAX = a201_last_dead<R, POS>::p_max;
            if constexpr(DUMP || (DEAD_P_MAX == 0))
            {
                write_shared_word(up, addr[POS]);
            }
            else if constexpr(DEAD_P_MAX >= A201_MAX_PARITY)
            {
                a201_write_shared_word_if(up, addr[POS], elide ^ 1);
            }
            else
            {
                a201_write_shared_word_if(up, addr[POS],
                                          (elide & static_cast<int>(p <= DEAD_P_MAX)) ^ 1);
            }
        }
        if constexpr(RS >= 0 && !DEG1 && !a201_res_publish<R, POS>::value)
        {
            res.slot[RS] = up; // mirror the (logical) APP value for next visit
        }
        if constexpr(!DEG1)
        {
            c2v[B + POS] = cn; // degree-1 resident edges never re-read their C2V
        }
        if constexpr(POS > 0)
        {
            suf = box_pair(suf, v2c[POS]);
            a201_backward<R, POS - 1, DUMP>(addr, norm, c2v, res, v2c, pre, suf, p, elide);
        }
    }

    //------------------------------------------------------------------
    // a201_process_row(): one exact box-plus layered update of BG1 parity row R
    // for this thread's check-node instance z. Forward pass gathers the
    // posterior, subtracts the previous C2V (skipped when FIRST), and builds the
    // prefix box-plus; backward pass forms the leave-one-out message
    // norm * box(prefix, suffix), refreshes the posterior, and stores the new
    // C2V (raw; the norm scale is deferred to its consumers, see h2_fma_norm).
    // Prefix/suffix grouping matches algo103's pipe_fwd/bwd.
    // POS is a template parameter in both traversals because the connectivity
    // tables expose each row edge through compile-time specializations.
    // The row's APP byte addresses are generated once (iteration-invariant
    // geometry, see a201_app_addr) and shared by both traversals.
    template <int R, bool FIRST, bool DUMP>
    __device__ __forceinline__
    void a201_process_row(const a201_app_addr& ag, __half2 norm, word_t* c2v, a201_resident<DUMP>& res,
                          int p, int elide)
    {
        constexpr int D = row_degree<ALGO103_BG, R>::value;

        int addr[D];
        ag.generate<R>(addr);

        word_t v2c[D];
        word_t pre[D];
        word_t acc = box_plus_identity();
        a201_forward<R, FIRST, DUMP, 0>(addr, norm, c2v, res, v2c, pre, acc);

        word_t suf = box_plus_identity();
        a201_backward<R, D - 1, DUMP>(addr, norm, c2v, res, v2c, pre, suf, p, elide);
    }

    //------------------------------------------------------------------
    // do_rows(): process rows R..(p-1), one barrier per row (layered). Rows are
    // instantiated up to A201_MAX_PARITY-1; the runtime (R+1 < p) guard skips
    // the tail. p >= A201_MIN_PARITY (== 4) is guaranteed by can_decode, so
    // rows whose successor is still below that floor can chain without paying
    // an always-true runtime row-existence branch.
    template <int R, bool FIRST, bool DUMP>
    __device__ __forceinline__
    void a201_do_rows(const a201_app_addr& ag, __half2 norm, word_t* c2v, a201_resident<DUMP>& res, int p,
                      int elide)
    {
        a201_process_row<R, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        __syncthreads();
        if constexpr((R + 1) < A201_MIN_PARITY)
        {
            a201_do_rows<R + 1, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        }
        else if constexpr((R + 1) < A201_MAX_PARITY)
        {
            if((R + 1) < p)
            {
                a201_do_rows<R + 1, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
            }
        }
    }

    // Rows 0..3 are active for every supported runtime p (p >= 4). Keep the
    // high-degree core block separate from the runtime-p tail so the compiler
    // does not have to carry tail-control state through the mandatory rows.
    template <bool FIRST, bool DUMP>
    __device__ __forceinline__
    void a201_do_core_rows(const a201_app_addr& ag, __half2 norm, word_t* c2v, a201_resident<DUMP>& res,
                           int p, int elide)
    {
        a201_process_row<0, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        __syncthreads();
        a201_process_row<1, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        __syncthreads();
        a201_process_row<2, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        __syncthreads();
        a201_process_row<3, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        __syncthreads();
        if(4 < p)
        {
            a201_do_rows<4, FIRST, DUMP>(ag, norm, c2v, res, p, elide);
        }
    }

    //------------------------------------------------------------------
    // Iteration-0 puncture collapse (bit-exact, keyed to the BG1 puncture
    // structure -- never to data or SNR).
    //
    // NR BG1 TB decode punctures the first two systematic columns, so at decode
    // entry V0 and V1 carry EXACTLY zero channel LLR.
    //
    // INPUT CONTRACT: this collapse is valid only when the input buffer really
    // carries zeros in columns 0,1. The production PUSCH derate matcher always
    // does, for every HARQ RV and after LLR combining -- the 38.212 Ncb
    // circular buffer excludes cols 0,1, so no (re)transmission can populate
    // them. It would break only if a-priori LLR injection into punctured
    // columns were ever introduced. BLER tests with cuphy_ex_ldpc MUST pass -P
    // (without it the harness generates non-standard nonzero LLRs there and
    // the decode is simply wrong, not merely unfair).
    //
    // box_plus is the sign-min rule, so a leave-one-out that contains a zero
    // input is EXACTLY 0:
    //
    //   row 0 holds BOTH V0 (POS 0) and V1 (POS 1): every leave-one-out of
    //   row 0 (including V0's and V1's own, which each still contain the other
    //   zero) is 0. All of row 0's C2V outputs are 0 and its posterior scatter
    //   is app[V] += 0 -- row 0 and the __syncthreads() that would publish it
    //   are a provable iteration-0 no-op. Only its side effects are kept: its
    //   C2V edges are zeroed (read by iteration 1's row-0 subtract) and the
    //   resident col-23 slot is seeded with the staged channel LLR (row 0's
    //   col-23 output is 0, so its posterior mirror equals the channel value).
    //
    //   row 1 holds V0 (POS 0) but not V1: after the row-0 no-op V0 is still
    //   zero, so every leave-one-out EXCEPT V0's own contains that zero and is
    //   exactly 0 (posteriors unchanged, C2V 0). Only V0's output -- box_plus
    //   over the row's OTHER edges -- is non-trivial. The whole row collapses
    //   to one accumulation and one output, bit-identical to the full row.
    //
    //   row 2 holds V0 (POS 0, updated by row 1) and V1 (POS 1, still zero):
    //   the same argument collapses it to V1's single output. Its resident
    //   col-25 edge (POS 18, first touch) seeds the slot with the staged
    //   channel LLR (its own output is 0, so posterior = channel + 0).
    //
    // Rows 3..p-1 then run in full (V0/V1 now both non-zero). The collapsed
    // rows have no timed/dump fork: the punctured column's store always
    // happens (cross-thread), and every elided store would have rewritten the
    // value already resident in the shared APP image (posterior += exact 0),
    // so the per-iteration dump snapshot is unchanged.
    template <bool DUMP, int R, int SKIP, int POS, int D>
    __device__ __forceinline__
    void a201_iter0_box_skip(const int (&addr)[D], a201_resident<DUMP>& res, word_t& acc)
    {
        if constexpr(POS != SKIP)
        {
            constexpr int RS = a201_res_slot<R, POS>::value;
            word_t p;
            if constexpr(RS < 0)
            {
                p = smem_address_as<word_t>(addr[POS]);
            }
            else if constexpr((R == 1 && POS == 18) || (R == 2 && POS == 18))
            {
                // col 24 (row 1) / col 25 (row 2) first touch. Row 1's V24
                // seed is the forwarding register's iteration-0 value: its
                // own C2V output here is exactly 0, so the forwarded word
                // equals the staged channel value row 2 would have loaded.
                p = smem_address_as<word_t>(addr[POS]);
                res.slot[RS] = p;                       // own output is 0 -> mirror = channel
            }
            else
            {
                p = res.slot[RS];                       // col 23 / col 24, seeded earlier
            }
            acc = box_pair(acc, p);
        }
        if constexpr((POS + 1) < D)
        {
            a201_iter0_box_skip<DUMP, R, SKIP, POS + 1>(addr, res, acc);
        }
    }

    // Process row R at iteration 0 where exactly the punctured column at POS
    // SKIP has a non-trivial output. Bit-identical to the full row.
    template <bool DUMP, int R, int SKIP>
    __device__ __forceinline__
    void a201_process_row_iter0_punctured(const a201_app_addr& ag,
                                          __half2 norm,
                                          word_t* c2v,
                                          a201_resident<DUMP>& res)
    {
        constexpr int B = c2v_base<R>::value;
        constexpr int D = row_degree<ALGO103_BG, R>::value;
        static_assert(a201_res_slot<R, SKIP>::value < 0, "punctured column must be shared-resident");

        int addr[D];
        ag.generate<R>(addr);

        word_t acc = box_plus_identity();
        a201_iter0_box_skip<DUMP, R, SKIP, 0>(addr, res, acc);
        const word_t out = h2_mul_norm(acc, norm);    // scaled leave-one-out for the punctured col
        write_shared_word(out, addr[SKIP]);           // its posterior was 0 -> 0 + out = out

        word_t c2v_zero;
        c2v_zero.u32 = 0u;
        #pragma unroll
        for(int e = 0; e < D; ++e) { c2v[B + e] = c2v_zero; }
        c2v[B + SKIP] = acc;                          // C2V stores the RAW box result
    }

    //------------------------------------------------------------------
    // Shared memory: APP buffer only (C2V and the TB token are register
    // resident), at RUNTIME Z. The all-warp token search returns the token in
    // registers, so ALGO201 does not reserve a shared token slot.
    CUDA_BOTH
    int a201_shmem_required(int num_parity_nodes, int Z)
    {
        return (ALGO103_INFO_NODES + num_parity_nodes) * Z *
               static_cast<int>(sizeof(word_t));
    }
    int a201_shmem_launch_required(int num_parity_nodes, int Z)
    {
        return static_cast<int>(ldpc2::shmem_size_with_experimental_et_context(
            static_cast<uint32_t>(a201_shmem_required(num_parity_nodes, Z))));
    }

    struct a201_loader_params
    {
        const char* src_gmem;
        int         max_cta_cw_index;
        int         src_stride_elements;
        float       clamp_value;

        __device__ __forceinline__
        a201_loader_params(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok) :
            src_gmem(nullptr),
            max_cta_cw_index(0),
            src_stride_elements(0),
            clamp_value(decodeDesc.config.clamp_value)
        {
            const int  tb         = tb_from_token(tok);
            const int  offset     = offset_from_token(tok);
            const bool is_partial = is_partial_from_token(tok);

            src_stride_elements = decodeDesc.llr_input[tb].stride_elements;
            const int cw_stride = static_cast<int>(sizeof(__half)) * src_stride_elements;
            src_gmem = static_cast<const char*>(decodeDesc.llr_input[tb].addr) + (offset * cw_stride);
            max_cta_cw_index = is_partial ? 0 : 1;
        }
    };

    template <int CW_PER_CTA>
    __device__ __forceinline__
    tb_token a201_find_block_tb_token_allwarp(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                              unsigned int                 decodeIndex)
    {
        if(decodeDesc.num_tbs == 1)
        {
            const int          num_cw  = decodeDesc.llr_input[0].num_codewords;
            const unsigned int offset  = decodeIndex * CW_PER_CTA;
            const bool         partial = ((offset + CW_PER_CTA) > static_cast<unsigned int>(num_cw));
            return to_token<CW_PER_CTA>(0, offset, partial);
        }
        return find_block_tb_token_allwarp<CW_PER_CTA>(decodeDesc, decodeIndex);
    }

    //------------------------------------------------------------------
    // ALGO201 channel setup: directly zero BG1's two punctured systematic APP
    // columns, then vector-load columns 2..(21+p). The iteration-0 collapse
    // already relies on those two columns being exact zero for TB decode.
    __device__ __forceinline__
    tb_token a201_load_channel_app_token_nobar(char*                        dst_smem,
                                               const cuphyLDPCDecodeDesc_t& decodeDesc,
                                               int                          decodeIndex)
    {
        static_assert(CUPHY_LDPC_NUM_PUNCTURED_NODES == 2,
                      "ALGO201 punctured-column loader assumes BG1 punctures two columns");

        tb_token tok = a201_find_block_tb_token_allwarp<2>(decodeDesc, decodeIndex);

        const int Z = decodeDesc.config.Z;
        word_t* const app = app_smem(dst_smem);
        uint2 zero2;
        zero2.x = 0u;
        zero2.y = 0u;
        *reinterpret_cast<uint2*>(app + (threadIdx.x << 1)) = zero2;

        const a201_loader_params params(decodeDesc, tok);
        typedef uint2 ldg_t;
        typedef uint4 sts_t;

        constexpr int PUNCT_COLS    = CUPHY_LDPC_NUM_PUNCTURED_NODES;
        constexpr int MIN_LOAD_COLS = ALGO103_INFO_NODES + A201_MIN_PARITY - PUNCT_COLS;
        constexpr int MIN_CHUNKS    = MIN_LOAD_COLS / 4;
        static_assert((MIN_LOAD_COLS % 4) == 0, "ALGO201 minimum-p loader expects whole chunks");

        const int   z         = static_cast<int>(threadIdx.x);
        const int   ld_stride = Z * static_cast<int>(sizeof(ldg_t));
        const int   st_stride = Z * static_cast<int>(sizeof(sts_t));
        const int   ld_skip   = PUNCT_COLS * Z * static_cast<int>(sizeof(__half));
        const int   st_skip   = PUNCT_COLS * Z * static_cast<int>(sizeof(word_t));
        const char* in0       = params.src_gmem + ld_skip;
        const int   stride1   = params.max_cta_cw_index * params.src_stride_elements;
        const char* in1       = in0 + (stride1 * static_cast<int>(sizeof(__half)));
        char*       out       = dst_smem + st_skip;
        const int   ld_off    = z * static_cast<int>(sizeof(ldg_t));
        const int   st_off    = z * static_cast<int>(sizeof(sts_t));
        const int   Z_quads   = Z >> 2;

        #pragma unroll
        for(int ii = 0; ii < MIN_CHUNKS; ++ii)
        {
            const ldg_t r0 = *reinterpret_cast<const ldg_t*>(in0 + ld_off + (ii * ld_stride));
            const ldg_t r1 = *reinterpret_cast<const ldg_t*>(in1 + ld_off + (ii * ld_stride));
            const sts_t sv = llr_op_clamp<__half, sts_t>::apply(interleave_llr(r0, r1),
                                                                 params.clamp_value);
            *reinterpret_cast<sts_t*>(out + st_off + (ii * st_stride)) = sv;
        }

        const int rem_cols = decodeDesc.config.num_parity_nodes - A201_MIN_PARITY;
        if(rem_cols > 0)
        {
            const int cols0  = (rem_cols < 4) ? rem_cols : 4;
            const int active = cols0 * Z_quads;
            if(z < active)
            {
                ldg_t t0 = *reinterpret_cast<const ldg_t*>(in0 + ld_off + (MIN_CHUNKS * ld_stride));
                ldg_t t1 = *reinterpret_cast<const ldg_t*>(in1 + ld_off + (MIN_CHUNKS * ld_stride));
                const sts_t sv = llr_op_clamp<__half, sts_t>::apply(interleave_llr(t0, t1),
                                                                     params.clamp_value);
                *reinterpret_cast<sts_t*>(out + st_off + (MIN_CHUNKS * st_stride)) = sv;
            }
        }
        if(rem_cols > 4)
        {
            const int cols1  = rem_cols - 4;
            const int active = cols1 * Z_quads;
            if(z < active)
            {
                constexpr int CHUNK = MIN_CHUNKS + 1;
                ldg_t t0 = *reinterpret_cast<const ldg_t*>(in0 + ld_off + (CHUNK * ld_stride));
                ldg_t t1 = *reinterpret_cast<const ldg_t*>(in1 + (ld_off + (CHUNK * ld_stride)));
                const sts_t sv = llr_op_clamp<__half, sts_t>::apply(interleave_llr(t0, t1),
                                                                     params.clamp_value);
                *reinterpret_cast<sts_t*>(out + st_off + (CHUNK * st_stride)) = sv;
            }
        }

        __syncthreads();
        return tok;
    }

    template <int WORDS_PER_WARP>
    __device__ __forceinline__
    void a201_hard_output_x2_all_warps(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                       tb_token                     tok,
                                       const app_buf_t*             app)
    {
        enum { THREADS_PER_WARP = 32 };
        const int WARP_IDX  = threadIdx.x / THREADS_PER_WARP;
        const int LANE      = threadIdx.x & (THREADS_PER_WARP - 1);
        const int start_idx = WARP_IDX * (WORDS_PER_WARP * THREADS_PER_WARP) + LANE;

        uint32_t output[2] = {0, 0};
        #pragma unroll
        for(int ii = 0; ii < WORDS_PER_WARP; ++ii)
        {
            const int APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            __half2 fp16x2(app[APP_IDX]);
            const unsigned int vote0 = __ballot_sync(c_ldpc_full_warp_mask, llr_hard_decision(fp16x2.x));
            const unsigned int vote1 = __ballot_sync(c_ldpc_full_warp_mask, llr_hard_decision(fp16x2.y));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }

        if(LANE < WORDS_PER_WARP)
        {
            const int tb               = tb_from_token(tok);
            const int offset           = offset_from_token(tok);
            const int out_stride_words = decodeDesc.tb_output[tb].stride_words;
            uint32_t* dst              = decodeDesc.tb_output[tb].addr + (offset * out_stride_words);
            const int output_idx       = WARP_IDX * WORDS_PER_WARP + LANE;

            dst[output_idx] = output[0];
            if(!is_partial_from_token(tok))
            {
                dst[output_idx + out_stride_words] = output[1];
            }
        }
    }

    //------------------------------------------------------------------
    __device__ __forceinline__
    void a201_tb_body(cuphyLDPCDecodeDesc_t decodeDesc, const a201_bg_desc_t& bgdesc)
    {
        extern __shared__ char smem[];

        const tb_token tok = a201_load_channel_app_token_nobar(smem, decodeDesc, blockIdx.x);

        const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
        const __half2 norm = norm_from_config(config);
        const int     p    = config.num_parity_nodes;
        const int     Z    = config.Z;

        word_t* const app = app_smem(smem);
        const int       z = threadIdx.x;

        const a201_app_addr ag(bgdesc, app, z, Z);

        word_t c2v[A201_TOTAL_C2V_WORDS];
        a201_resident<false> res;

        // Final-iteration dead-store predicate base: only a hard-output-only
        // decode (no soft outputs) leaves the dead parity-column posterior
        // stores of the LAST iteration unconsumed.
        const int hard_only =
            ((0 == (config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ? 1 : 0);

        int32_t iter = 0;
        if(iter < config.max_iterations)
        {
            // Iteration 0: row 0 is a provable no-op (both punctured columns
            // still zero -- see a201_iter0_box_skip) and is elided together
            // with its barrier; only its side effects are reproduced.
            word_t c2v_zero;
            c2v_zero.u32 = 0u;
            #pragma unroll
            for(int e = 0; e < row_degree<ALGO103_BG, 0>::value; ++e)
            {
                c2v[e] = c2v_zero;                    // row 0 edges: all-zero C2V
            }
            // col 23 first touch (= the row-0 no-op's posterior mirror)
            res.slot[a201_res_slot<0, 18>::value] =
                smem_address_as<word_t>(ag.tbase + (23 * ag.Zsz));

            // Rows 1 and 2 collapse to a single output each (V0's, then V1's);
            // their barriers remain (V0/V1 feed later rows).
            const int elide0 = hard_only & static_cast<int>(1 == config.max_iterations);
            a201_process_row_iter0_punctured<false, 1, 0>(ag, norm, c2v, res);
            __syncthreads();
            a201_process_row_iter0_punctured<false, 2, 1>(ag, norm, c2v, res);
            __syncthreads();
            a201_do_rows<3, true, false>(ag, norm, c2v, res, p, elide0);   // rows 3..p-1: no C2V subtract
            ++iter;
        }
        for(; iter < config.max_iterations; ++iter)
        {
            const int elide = hard_only & static_cast<int>((iter + 1) == config.max_iterations);
            a201_do_core_rows<false, false>(ag, norm, c2v, res, p, elide);
        }

        a201_hard_output_x2_all_warps<22>(decodeDesc, tok, reinterpret_cast<const app_buf_t*>(smem));
    }

} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_z256up_bp_allreg_x2_tb()
extern "C"
__global__ __launch_bounds__(A201_MAX_THREADS_PER_CTA, A201_MIN_CTA_PER_SM)
void ldpc2_BG1_z256up_bp_allreg_x2_tb(cuphyLDPCDecodeDesc_t decodeDesc, a201_bg_desc_t bgdesc)
{
    a201_tb_body(decodeDesc, bgdesc);
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::bg1_z256up_bp_allreg_x2()
bg1_z256up_bp_allreg_x2::bg1_z256up_bp_allreg_x2(ldpc::decoder& dec)
{
    const int MAX_SHMEM_A201 = a201_shmem_launch_required(A201_MAX_PARITY, A201_Z_MAX);
    const int MAX_SHMEM         = dec.max_shmem_per_block_optin();

    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 1> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_z256up_bp_allreg_x2_tb, std::min(MAX_SHMEM_A201, MAX_SHMEM))
    };
    if(MAX_SHMEM_A201 > (48 * 1024))
    {
        for(func_attr_t f_a : func_attrs)
        {
            cudaError_t e = cudaFuncSetAttribute(f_a.first,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 f_a.second);
            if(cudaSuccess != e)
            {
                throw cuphy_i::cuda_exception(e);
            }
        }
    }
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_z256up_bp_allreg_x2_tb);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::decode()
cuphyStatus_t bg1_z256up_bp_allreg_x2::decode(ldpc::decoder&,
                                             LDPC_output_t&,
                                             const_tensor_pair&,
                                             const cuphy_optional<tensor_pair>&,
                                             const cuphyLDPCDecodeConfigDesc_t&,
                                             cudaStream_t)
{
    DEBUG_PRINTF("ldpc2::bg1_z256up_bp_allreg_x2::decode() tensor interface is unsupported\n");
    return CUPHY_STATUS_NOT_SUPPORTED;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::decode_tb()
cuphyStatus_t bg1_z256up_bp_allreg_x2::decode_tb(ldpc::decoder&               dec,
                                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::bg1_z256up_bp_allreg_x2::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    // No assert on the soft-output flag here: can_decode_config() below
    // REFUSES CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS outright (this kernel has
    // no soft-LLR writer), so an explicit request for it must return
    // NOT_SUPPORTED and let the caller fall through to a decoder that has
    // one. Asserting first turned that clean refusal into a debug-build
    // abort, and inside the accepted branch the condition is vacuous.

    if(decodeDesc.config.llr_type == CUPHY_R_16F && can_decode_config(dec, decodeDesc.config))
    {
        // The shared (algo103.cuh) token search used on the multi-TB path reads
        // the pair-CTA boundary index that prepare_ldpc_tb_pair_index() derives
        // into the descriptor's llr_output[] scalar fields.
        cuphyLDPCDecodeDesc_t indexedDecodeDesc = decodeDesc;
        prepare_ldpc_tb_pair_index(indexedDecodeDesc);
        dim3 blkDim(decodeDesc.config.Z);
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        const uint32_t SHMEM_SIZE = a201_shmem_launch_required(decodeDesc.config.num_parity_nodes,
                                                                   decodeDesc.config.Z);

        const a201_bg_desc_t* bgdesc = ldpc2::get_adj_BG_desc<__half2, 1>(decodeDesc.config.Z);
        if(!bgdesc) return CUPHY_STATUS_INTERNAL_ERROR;


        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_z256up_bp_allreg_x2_tb, blkDim, SHMEM_SIZE);
        ldpc2_BG1_z256up_bp_allreg_x2_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(indexedDecodeDesc, *bgdesc);
        s = CUPHY_STATUS_SUCCESS;
    }

    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::get_workspace_size()
std::pair<bool, size_t> bg1_z256up_bp_allreg_x2::get_workspace_size(const ldpc::decoder&,
                                                                   const cuphyLDPCDecodeConfigDesc_t&,
                                                                   int)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::can_decode_config()
bool bg1_z256up_bp_allreg_x2::can_decode_config(const ldpc::decoder&               dec,
                                               const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Z zone: the five valid 38.212 liftings in 256..A201_Z_MAX (384 by
    // default). Every one is a multiple of 32 -- in this range that is the
    // same set, which is what lets the %32 test below stand in for a table
    // lookup -- so the all-warps output writer and full-warp token machinery
    // hold. See EveryAcceptedLiftingHasABaseGraphDescriptor, which pins that
    // coincidence; it does NOT survive widening the zone below 256.
    //
    // Hard decisions only, and that is enforced rather than assumed: this
    // kernel has no soft-LLR writer, and prepare_ldpc_tb_pair_index()
    // repurposes llr_output[].stride_elements / .num_codewords as the
    // pair-CTA boundary index the token search reads. Those are the very
    // fields ldpc2_dec_output.cuh would use to address soft outputs
    // (ph + cwIndex * stride_elements), so accepting the flag would both
    // drop the requested output and corrupt the descriptor it needs.
    // Refusing lets choose_algo() fall through to a decoder that has one.
    if((cfg.llr_type != CUPHY_R_16F)                     ||
       (cfg.BG != ALGO103_BG)                            ||
       (cfg.Kb != 22)                                    ||
       (0 != (cfg.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
       (cfg.Z < 256) || (cfg.Z > A201_Z_MAX) || (cfg.Z % 32) ||
       (cfg.num_parity_nodes < A201_MIN_PARITY)          ||
       (cfg.num_parity_nodes > A201_MAX_PARITY))
    {
        return false;
    }
    const uint32_t SHMEM_SIZE = a201_shmem_launch_required(cfg.num_parity_nodes, cfg.Z);
    return (static_cast<int>(SHMEM_SIZE) * A201_MIN_CTA_PER_SM <= dec.max_shmem_per_block_optin());
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2::get_launch_config()
cuphyStatus_t bg1_z256up_bp_allreg_x2::get_launch_config(const ldpc::decoder&           dec,
                                                        cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const cuphyLDPCDecodeConfigDesc_t& config = launchConfig.decode_desc.config;
    if(!can_decode_config(dec, config))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    prepare_ldpc_tb_pair_index(launchConfig.decode_desc);

#if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = config.Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX =
        ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra        = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        a201_shmem_launch_required(config.num_parity_nodes, config.Z);

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const void* kernel_func = (const void*)ldpc2_BG1_z256up_bp_allreg_x2_tb;
    cudaError_t e = cudaGetFuncBySymbol(&deviceFunction, kernel_func);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
#endif

    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    const a201_bg_desc_t* bgdesc = ldpc2::get_adj_BG_desc<__half2, 1>(config.Z);
    launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));

    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
