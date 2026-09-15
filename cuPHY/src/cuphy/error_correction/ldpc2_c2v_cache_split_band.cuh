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

#if !defined(LDPC2_C2V_CACHE_SPLIT_BAND_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_SPLIT_BAND_CUH_INCLUDED_

// Split C2V message cache for the BG1 band decoders.
// Everything in this header is used exclusively by ldpc2_algo202.cu; no
// other decoder algorithm includes it.

#include "ldpc2.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_c2v_cache_split.cuh" // noncore_reg_chain / has_resize
#include <type_traits>

namespace ldpc2
{

#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
////////////////////////////////////////////////////////////////////////
// cC2V_storage_x2_tail_packed
// 2-word pre-norm packed compressed C2V storage for the ALGO202 deep
// tail rows 19/20 (see the c2v_storage_prenorm_packed trait machinery
// in ldpc2_c2v_x2.cuh): the whole record round trip is ONE LDS.64 +
// ONE STS.64 per row per iteration. The 4-arg constructor receives the
// finalize()-captured PRE-NORM signed E5M5 magnitudes (low 5 bits per
// half zero by construction) plus the argmin index pair and per-edge
// signs word, and folds index+signs into the free low bits; the
// norm-aware from-storage constructor reconstructs the normalized
// values with finalize()'s exact __hmul2/__hsub2 sequence, so the APP
// trajectory stays bit-identical. Requires row degree <= 7 (rows
// 19/20: degree 6) and argmin index <= 7.
struct cC2V_storage_x2_tail_packed
{
    word_t a; // pre-norm signed min0 | idx(3b @ 2..0) | signs e0..e1 (@ 4..3)
    word_t b; // pre-norm signed min1 | signs e2..e6 (@ 4..0)
    //------------------------------------------------------------------
    typedef signs_pair_low_degree signs_pair_t;
    // Trait tag: selects the pre-norm capture / pack / norm-aware
    // reconstruction paths in cC2V_row_context (ldpc2_c2v_x2.cuh).
    typedef void prenorm_packed_tag;
    //------------------------------------------------------------------
    cC2V_storage_x2_tail_packed() = default;
    //------------------------------------------------------------------
    // Pack from pre-norm signed E5M5 magnitudes + argmin index pair +
    // signs word (sign of edge e at bit 6+e / 22+e). Lossless for row
    // degree <= 7: index fits 3 bits, signs e0..e1 -> a bits 4..3,
    // signs e2..e6 -> b bits 4..0.
    __device__
    cC2V_storage_x2_tail_packed(word_t              pn_m0,
                                word_t              pn_m1,
                                word_t              m0_index,
                                const signs_pair_t& s_pair)
    {
        a.u32 = pn_m0.u32 | (m0_index.u32 & 0x00070007u) |
                ((s_pair.signs_0_9 >> 3) & 0x00180018u);
        b.u32 = pn_m1.u32 | ((s_pair.signs_0_9 >> 8) & 0x001F001Fu);
    }
};
#endif // LDPC2_A202_TAIL_PACKED2

////////////////////////////////////////////////////////////////////////
// bg1_noncore_urd()
// update_row_degree<1, row> for the BG1 non-core rows as a constexpr
// FUNCTION of a runtime row index, usable in both the host-side shared
// memory sizing and device-side address math. Rows outside 4..13
// contribute 0 (rows 0..13 are the only ones instantiated; core rows 0..3
// never use the non-core shared region).
CUDA_BOTH_INLINE constexpr int bg1_noncore_urd(int row)
{
    return (row ==  4) ? 2 : (row ==  5) ? 7 : (row ==  6) ? 8 :
           (row ==  7) ? 6 : (row ==  8) ? 9 : (row ==  9) ? 8 :
           (row == 10) ? 6 : (row == 11) ? 7 : (row == 12) ? 6 :
           (row == 13) ? 5 : 0;
}
// Keep the runtime table in lock-step with the canonical 38.212 tables.
static_assert(bg1_noncore_urd( 4) == update_row_degree<1,  4>::value, "URD table mismatch (row 4)");
static_assert(bg1_noncore_urd( 5) == update_row_degree<1,  5>::value, "URD table mismatch (row 5)");
static_assert(bg1_noncore_urd( 6) == update_row_degree<1,  6>::value, "URD table mismatch (row 6)");
static_assert(bg1_noncore_urd( 7) == update_row_degree<1,  7>::value, "URD table mismatch (row 7)");
static_assert(bg1_noncore_urd( 8) == update_row_degree<1,  8>::value, "URD table mismatch (row 8)");
static_assert(bg1_noncore_urd( 9) == update_row_degree<1,  9>::value, "URD table mismatch (row 9)");
static_assert(bg1_noncore_urd(10) == update_row_degree<1, 10>::value, "URD table mismatch (row 10)");
static_assert(bg1_noncore_urd(11) == update_row_degree<1, 11>::value, "URD table mismatch (row 11)");
static_assert(bg1_noncore_urd(12) == update_row_degree<1, 12>::value, "URD table mismatch (row 12)");
static_assert(bg1_noncore_urd(13) == update_row_degree<1, 13>::value, "URD table mismatch (row 13)");

////////////////////////////////////////////////////////////////////////
// bg1_noncore_shm_words()
// Number of per-thread C2V words the shared non-core region holds for
// rows [first_shm_row, num_parity_nodes). Used with a runtime p for
// shared-memory sizing (0 when p <= first_shm_row) and with compile-time
// row indices for per-row word offsets.
CUDA_BOTH_INLINE constexpr int bg1_noncore_shm_words(int first_shm_row, int num_parity_nodes)
{
    int words = 0;
    for(int r = first_shm_row; r < num_parity_nodes; ++r)
    {
        words += bg1_noncore_urd(r);
    }
    return words;
}

////////////////////////////////////////////////////////////////////////
// bg1_tail_urd()
// update_row_degree<1, row> for the ALGO202 shared-tail rows (BG1 rows
// 14..20) as a constexpr function, usable in host shared-memory sizing
// and device layout math. 0 outside 14..20.
CUDA_BOTH_INLINE constexpr int bg1_tail_urd(int row)
{
    return (row == 14) ? 6 : (row == 15) ? 6 :
           ((row >= 16) && (row <= 20)) ? 5 : 0;
}
// Keep the runtime table in lock-step with the canonical 38.212 tables.
static_assert(bg1_tail_urd(14) == update_row_degree<1, 14>::value, "tail URD mismatch (row 14)");
static_assert(bg1_tail_urd(15) == update_row_degree<1, 15>::value, "tail URD mismatch (row 15)");
static_assert(bg1_tail_urd(16) == update_row_degree<1, 16>::value, "tail URD mismatch (row 16)");
static_assert(bg1_tail_urd(17) == update_row_degree<1, 17>::value, "tail URD mismatch (row 17)");
static_assert(bg1_tail_urd(18) == update_row_degree<1, 18>::value, "tail URD mismatch (row 18)");
static_assert(bg1_tail_urd(19) == update_row_degree<1, 19>::value, "tail URD mismatch (row 19)");
static_assert(bg1_tail_urd(20) == update_row_degree<1, 20>::value, "tail URD mismatch (row 20)");

////////////////////////////////////////////////////////////////////////
// bg1_a202_tail_words_row() / bg1_a202_tail_eoff()
// ALGO202 tail-row storage form (per ROW, identical at every p -- the
// kernel stays a single shared runtime-p body): rows 14..18 hold their
// C2V EXPANDED (URD per-edge message words); rows 19..20 keep the
// 3-word COMPRESSED min-sum record. The two deepest rows stay
// compressed so the region can also carry the 6-word extension-column
// SURVIVAL strip (below) inside the same 39-word budget -- the
// measured re-adjudication outcome: giving rows 19/20 the expanded
// form but sourcing the deep rows' loop-invariant extension values
// from global/L2 instead costs ~6 always-live registers of stream-
// pointer state, which sits exactly on this kernel's register-plan
// cliff and measured strictly worse at every p.
// bg1_a202_tail_eoff(row) = the row's fixed word offset (prefix sum);
// compile-time immediates independent of the runtime p.
CUDA_BOTH_INLINE constexpr int bg1_a202_tail_words_row(int row)
{
    return (row <= 18) ? bg1_tail_urd(row) : ((row <= 20) ? 3 : 0);
}
CUDA_BOTH_INLINE constexpr int bg1_a202_tail_eoff(int row)
{
    int off = 0;
    for(int r = 14; r < row; ++r) { off += bg1_a202_tail_words_row(r); }
    return off;
}
// Extension-column SURVIVAL strip: the loop-invariant extension APP of
// the runtime-conditional rows 15..20 (staged once by the entry
// loader, never rewritten) lives in six PERMANENT word planes behind
// the last C2V slot -- word plane row+18, i.e. planes 33..38 -- so a
// deep row's warm-iteration re-read stays one LDS at a compile-time
// offset. Total region: 33 C2V words + 6 survival words = 39 planes.
CUDA_BOTH_INLINE constexpr int bg1_a202_tail_total_words()
{
    return bg1_a202_tail_eoff(21) + 6;
}
static_assert(bg1_a202_tail_eoff(21) == 33, "tail C2V words (rows 14..18 expanded + 19/20 compressed)");

////////////////////////////////////////////////////////////////////////
// bg1_a202_ext_stage_plane()
// Extension staging map. The entry stager writes the (loop-invariant)
// extension-column value of non-core row r into word plane:
//   * r+13 for the ALWAYS-EXECUTED rows 4..14 -- planes 17..27, which
//     belong to tail rows 17..19's C2V slots. Those slots are DEAD
//     until their owning row's first-iteration C2V store, and every
//     staged value is consumed (and register-captured) strictly before
//     its hosting slot goes live:
//       planes 17..21 (row 17's slot) host rows 4..8   -- read by phase 8  < 17
//       planes 22..26 (row 18's slot) host rows 9..13  -- read by phase 13 < 18
//       plane  27     (row 19's slot) hosts row 14     -- read at phase 14 < 19
//   * r+18 for the RUNTIME-CONDITIONAL rows 15..20 -- the PERMANENT
//     survival planes 33..38 behind the last C2V slot, re-read (one
//     LDS, compile-time offset) by the row on every iteration; no C2V
//     store ever touches them.
CUDA_BOTH_INLINE constexpr int bg1_a202_ext_stage_plane(int row)
{
    return (row <= 14) ? (row + 13) : (row + 18);
}
static_assert(bg1_a202_ext_stage_plane(4)  == bg1_a202_tail_eoff(17), "staging must start at row 17's slot");
static_assert(bg1_a202_ext_stage_plane(8)  <  bg1_a202_tail_eoff(18), "rows 4..8 must fit row 17's slot");
static_assert(bg1_a202_ext_stage_plane(13) <  bg1_a202_tail_eoff(19), "rows 9..13 must fit row 18's slot");
static_assert(bg1_a202_ext_stage_plane(14) <  bg1_a202_tail_eoff(20), "row 14 must fit row 19's slot");
static_assert(bg1_a202_ext_stage_plane(15) == bg1_a202_tail_eoff(21), "survival strip starts behind the last C2V slot");
static_assert(bg1_a202_ext_stage_plane(20) <  bg1_a202_tail_total_words(), "survival strip in bounds");

////////////////////////////////////////////////////////////////////////
// bg1_row_contains
// Does BG1 parity row ROW touch variable-node column COL? Evaluated
// entirely from the checked-in vnode_index<> descriptor tables
// (nrLDPC_templates.cuh), so any cross-barrier independence proof built
// on it is tied to the canonical 38.212 BG1 structure, not to a
// hand-entered copy.
template <int ROW, int COL, int I, int DEG>
struct bg1_row_contains_impl
{
    static constexpr bool value = (vnode_index<1, ROW, I>::value == COL) ||
                                  bg1_row_contains_impl<ROW, COL, I + 1, DEG>::value;
};
template <int ROW, int COL, int DEG>
struct bg1_row_contains_impl<ROW, COL, DEG, DEG>
{
    static constexpr bool value = false;
};
template <int ROW, int COL>
struct bg1_row_contains : bg1_row_contains_impl<ROW, COL, 0, row_degree<1, ROW>::value> {};

////////////////////////////////////////////////////////////////////////
// bg1_preload_mask
// Cross-barrier APP preload eligibility for the in-order layered
// schedule: bit i is set iff row CUR's i-th APP column may be LOADED
// during row PREV's processing phase, i.e. BEFORE the __syncthreads()
// that separates the two rows.
//
// Independence proof (per column c = vnode_index<1, CUR, i>):
//   * Between the barrier that precedes row PREV and the barrier that
//     follows it, the only APP writes in flight anywhere in the CTA are
//     row PREV's write-backs to its own columns (the schedule is
//     strictly in-order, one row per barrier interval).
//   * If PREV does not touch c, no thread writes c in that interval, so
//     a load of c is race-free and returns exactly the value it would
//     return after the barrier: the last writer of c was some earlier
//     row, whose stores were made CTA-visible by the barrier that
//     preceded row PREV. Bit-exact by construction.
//
// Extension columns (c >= 26, the degree-1 parity extensions) are
// excluded unconditionally:
//   * on iteration 0 the entry stager's tail rounds
//     (llr_stager_z384::after_row) are still STOREing columns >= 26
//     inside row 3/7/11's phases, so a cross-barrier load could race
//     with the staging store;
//   * on every later iteration the extension-column register cache
//     serves them without any shared-memory load, so there is nothing
//     to hoist.
template <int PREV, int CUR, int I, int DEG>
struct bg1_preload_mask_impl
{
    static constexpr int  COL      = vnode_index<1, CUR, I>::value;
    static constexpr bool ELIGIBLE = !bg1_row_contains<PREV, COL>::value && (COL < 26);
    static constexpr uint32_t value = (ELIGIBLE ? (1u << I) : 0u) |
                                      bg1_preload_mask_impl<PREV, CUR, I + 1, DEG>::value;
};
template <int PREV, int CUR, int DEG>
struct bg1_preload_mask_impl<PREV, CUR, DEG, DEG>
{
    static constexpr uint32_t value = 0u;
};
template <int PREV, int CUR>
struct bg1_preload_mask : bg1_preload_mask_impl<PREV, CUR, 0, row_degree<1, CUR>::value> {};

////////////////////////////////////////////////////////////////////////
// bg1_preload_mask2
// Two-row-ahead variant: bit i is set iff row CUR's i-th APP column is
// touched by NEITHER row PREV2 NOR row PREV1 (and is not an extension
// column), so a load issued during row PREV2's phase survives both
// intervening barriers unchanged (no thread writes the column in either
// phase). Same independence argument as bg1_preload_mask, applied to
// two consecutive barrier intervals.
template <int PREV2, int PREV1, int CUR, int I, int DEG>
struct bg1_preload_mask2_impl
{
    static constexpr int  COL      = vnode_index<1, CUR, I>::value;
    static constexpr bool ELIGIBLE = !bg1_row_contains<PREV2, COL>::value &&
                                     !bg1_row_contains<PREV1, COL>::value && (COL < 26);
    static constexpr uint32_t value = (ELIGIBLE ? (1u << I) : 0u) |
                                      bg1_preload_mask2_impl<PREV2, PREV1, CUR, I + 1, DEG>::value;
};
template <int PREV2, int PREV1, int CUR, int DEG>
struct bg1_preload_mask2_impl<PREV2, PREV1, CUR, DEG, DEG>
{
    static constexpr uint32_t value = 0u;
};
template <int PREV2, int PREV1, int CUR>
struct bg1_preload_mask2 : bg1_preload_mask2_impl<PREV2, PREV1, CUR, 0, row_degree<1, CUR>::value> {};

////////////////////////////////////////////////////////////////////////
// c2v_cache_split_band
// C2V cache with the register/shared split point placed INSIDE
// the non-core rows:
//   rows 0..3             : core storage, as in c2v_cache_split (register
//                           compressed min-sum, or shared box_plus when
//                           TStorageCore is larger than 4 words)
//   rows 4..NUM_REG-1     : per-row-sized register box_plus storage
//                           (identical to c2v_cache_split's chain)
//   rows NUM_REG..MAX_P-1 : per-row-sized SHARED box_plus storage in an
//                           SoA layout: word w of row r for thread t sits
//                           at word offset (prefix(r) + w)*Z + t from the
//                           region base. Consecutive threads touch
//                           consecutive banks (conflict-free), and with Z
//                           fixed at compile time (RT_Z = false) every
//                           ld/st.shared is a compile-time immediate off
//                           the per-thread base pointer. With RT_Z = true
//                           the stride is blockDim.x -- which IS Z for
//                           these 1-CTA/SM, blockDim==Z kernels -- so the
//                           immediate is lost but no value is carried live
//                           through the decode loop (%ntid.x is uniform
//                           and rematerializable).
//
// Rationale: the bounded runtime-p kernel's register allocation is
// static, so a trailing row's register C2V is paid at EVERY p even
// though the row only executes when p reaches it. Placing the trailing
// rows' C2V in the (otherwise unused) shared-memory headroom converts
// that always-paid register pressure into ld/st.shared executed only at
// the p values that actually run those rows. Values, operations and
// their order are unchanged -- storage placement only -- so the APP
// trajectory is bit-identical to the all-register cache.
//
// The shared region is sized by the RUNTIME p (bg1_noncore_shm_words):
// 0 bytes for p <= NUM_REG, growing per-row-sized above that.
//
// NOTE: the shared C2V region is NOT zero-initialized. The caller always
// peels iteration 0 (IS_FIRST), which elides the
// previous-C2V subtract for every row, so the initial C2V storage value
// is never read. For the shared rows the load is likewise elided on
// IS_FIRST and the (dead) store on IS_LAST.
//
// EXTENSION-COLUMN REGISTER CACHE: every BG1 non-core row has exactly one
// extension (isolated, degree-1) column whose APP is written once by the
// LLR loader and never updated afterwards (app_writer::write_non_ext
// skips it, and no other row connects to it). The baseline reloads that
// loop-invariant value from shared memory every iteration; here it is
// captured in a register on the peeled first iteration (IS_FIRST) and
// reused for the remaining iterations, removing one ld.shared per
// non-core row per subsequent iteration and making the box-plus "outside"
// seed available without waiting on the shared-memory pipe. The value
// read is bit-identical (the shared copy never changes), so the APP
// trajectory is unchanged. Because iteration 0 is always peeled and a row
// executes either in every iteration or in none (runtime p is
// iteration-invariant), the cache entry is always written before any
// read. With the ALGO202 compressed tail (HAS_TAIL) the cache covers the
// REGISTER rows 4..NUM_REG-1 only; tail rows re-load the extension APP
// from shared each iteration (see the tail branches for the audit).
//------------------------------------------------------------------
// Core parity-bidiagonal register-handoff contract, proved against the
// checked-in 38.212 descriptors over a WHOLE Z zone.
//
// Each producer row's CARRY_OUT edge and its consumer row's CARRY_IN edge
// must name the SAME variable-node column with the SAME circulant shift
// mod Z, so that the same thread owns the element in both rows; adjacency
// of the row pairs guarantees no other row touches the column in between.
// The shift equality is a per-lifting fact, so a kernel that runs at more
// than one lifting has to prove it at each of them.
//
// (It does hold at all of them, and for a structural reason rather than a
// coincidence: these are BG1's parity bidiagonal edges, whose 38.212
// shifts are 0 in every lifting set. The check is here so that stays
// verified rather than remembered.)

// nr_is_lifting(): the standard's own legality rule, Z = a*2^j with
// a in {2,3,5,7,9,11,13,15} -- so band edges stay derived (OBJECTIVES O4)
// instead of hand-listed.
constexpr bool nr_is_lifting(int Z)
{
    const int a_vals[8] = {2, 3, 5, 7, 9, 11, 13, 15};
    for(int i = 0; i < 8; ++i)
    {
        for(int z = a_vals[i]; z <= 384; z *= 2)
        {
            if(z == Z) { return true; }
        }
    }
    return false;
}

// One lifting. Non-liftings have no descriptor table and can never be
// dispatched, so there is nothing to prove for them.
template <int Z, bool IS_LIFTING = nr_is_lifting(Z)>
struct bg1_bidiag_carry_ok_at
{
    static constexpr bool value = true;
};

template <int Z>
struct bg1_bidiag_carry_ok_at<Z, true>
{
    static constexpr int ILS = set_index<Z>::value;
    static constexpr bool value =
        // rows 0->1 carry, column 23
        (vnode_index<1, 0, 18>::value == 23) &&
        (vnode_index<1, 1, 17>::value == 23) &&
        (vnode_shift<1, ILS, 0, 18>::value % Z == vnode_shift<1, ILS, 1, 17>::value % Z) &&
        // rows 1->2 carry, column 24
        (vnode_index<1, 1, 18>::value == 24) &&
        (vnode_index<1, 2, 17>::value == 24) &&
        (vnode_shift<1, ILS, 1, 18>::value % Z == vnode_shift<1, ILS, 2, 17>::value % Z) &&
        // rows 2->3 carry, column 25
        (vnode_index<1, 2, 18>::value == 25) &&
        (vnode_index<1, 3, 18>::value == 25) &&
        (vnode_shift<1, ILS, 2, 18>::value % Z == vnode_shift<1, ILS, 3, 18>::value % Z);
};

// The zone, walked in steps of 32. These kernels launch blockDim.x == Z
// and their band gate requires Z % 32 == 0 (the all-warps hard-output
// writer's precondition), so a multiple of 32 that is also a legal lifting
// is exactly the set the gate admits -- the proof and the runtime gate
// range over the same Z by construction rather than by agreement.
template <int Z_LO, int Z_HI>
struct bg1_bidiag_carry_ok_zone
{
    // Without this the recursion below runs away on a mis-specified zone
    // instead of failing, and the diagnostic would be a template depth
    // limit rather than the actual mistake.
    static_assert(Z_LO < Z_HI && ((Z_HI - Z_LO) % 32) == 0,
                  "zone must be a non-empty range of multiples of 32 (Z_LO <= Z_HI)");
    static constexpr bool value = bg1_bidiag_carry_ok_at<Z_LO>::value &&
                                  bg1_bidiag_carry_ok_zone<Z_LO + 32, Z_HI>::value;
};

template <int Z_HI>
struct bg1_bidiag_carry_ok_zone<Z_HI, Z_HI>
{
    static constexpr bool value = bg1_bidiag_carry_ok_at<Z_HI>::value;
};

template <int   BG_,
          int   Z_FIXED,
          int   NUM_REG_C2V_NODES,
          int   MAX_PARITY_NODES,
          class TC2V,
          class TStorageCore,
          class TStorageNonCore,
          class TKernelParams,
          class TStorageTail = void,
          // Extension-column register-cache row bound (ALGO202 audit):
          // rows 4..EXT_CACHE_BOUND-1 cache their loop-invariant extension
          // APP in a register; rows >= EXT_CACHE_BOUND re-load it from
          // shared each iteration (bit-identical -- the value is never
          // written back after staging). 0 (the default) caches every
          // instantiated row, preserving the
          // original behavior. ALGO202 passes its CFG_MIN_P so that only
          // rows executing at EVERY p in the zone pin an always-live
          // register; a deeper (runtime-conditional) tail row's entry
          // would tax the register plan's peak at every p while its saved
          // LDS runs only at the p values that reach the row.
          int   EXT_CACHE_BOUND = 0,
          // ALGO202 extension-column relocation (default false keeps the
          // baseline configuration byte-for-byte unchanged):
          // the degree-1 extension columns' APP is LOOP-INVARIANT (staged
          // once, never written back), yet in the baseline layout those
          // (p-4) columns occupy up to 26112 B (p21) of the shared budget
          // that deep p exhausts, purely to be re-read, while DRAM/L2 sit
          // at ~10%/~4% utilization. With EXT_GLOBAL the shared APP
          // buffer holds only the 26 read/write columns; iteration 0
          // reads the values from a TRANSIENT staging strip inside the
          // (dead until first store) tail C2V region, rows below
          // EXT_CACHE_ROWS capture them in registers, and the deeper
          // (runtime-conditional) rows re-materialize the value from the
          // global channel-LLR streams each later iteration --
          // BIT-IDENTICAL to the staged value (same PRMT lane merge,
          // same clamp_signed op on the same __float2half2_rn constant).
          bool  EXT_GLOBAL = false,
          // ALGO202 tail-row storage form: when true, tail rows store
          // their C2V EXPANDED as the URD reconstructed per-edge message
          // words at fixed URD-prefix offsets (bg1_a202_tail_eoff), so
          // the next iteration's subtract is a bare __hsub2 with no
          // per-edge decompress and no stored-signs scan accumulation.
          // BIT-IDENTICAL APP trajectory (see tail_row_expanded); fits at
          // every p in the zone only because EXT_GLOBAL freed the
          // extension columns' bytes.
          bool  TAIL_EXP = false,
          // RUNTIME lifting size. false --
          // the historical form -- pins every shared C2V plane stride to
          // Z_FIXED as a compile-time immediate. true takes the stride from
          // blockDim.x instead, so ONE compiled kernel serves every lifting
          // in [Z_MIN_, Z_FIXED]. Nothing else about the cache differs: the
          // storage plan, the row split and the register plan are identical,
          // which is why the RT_Z=false variant stays code-identical to the
          // kernel this header compiled before the parameter existed.
          bool  RT_Z = false,
          // Bottom of the Z zone, for the compile-time proofs below. Defaults
          // to Z_FIXED, i.e. the single-lifting form, so instantiations that
          // do not opt into RT_Z prove exactly what they proved before.
          int   Z_MIN_ = Z_FIXED>
struct c2v_cache_split_band
{
    //------------------------------------------------------------------
    // ZSTRIDE()
    // Shared-memory PLANE stride for the tail C2V region. Every `* Z`
    // in this struct goes through it so the two variants cannot disagree,
    // and so a runtime-Z address generator can never be paired with
    // compile-time-384 strides (which would index past the APP image).
    static __device__ __forceinline__ int ZSTRIDE()
    {
        if constexpr(RT_Z) { return static_cast<int>(blockDim.x); }
        else               { return Z_FIXED; }
    }

    static constexpr int EXT_CACHE_ROWS =
        (EXT_CACHE_BOUND <= 0) ? MAX_PARITY_NODES
                               : (EXT_CACHE_BOUND < MAX_PARITY_NODES ? EXT_CACHE_BOUND
                                                                     : MAX_PARITY_NODES);
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    typedef TStorageCore          c2v_storage_core_t;
    typedef TStorageNonCore       c2v_storage_noncore_t;
    static const int BG = BG_;

    // Optional COMPRESSED shared-memory TAIL band (ALGO202): rows >=
    // NUM_REG_C2V_NODES store their C2V as TStorageTail (fixed-size
    // compressed min-sum words) in a shared SoA region instead of the
    // per-row-sized box-plus spill below. void (the default)
    // keeps the original box-plus trailing branch untouched.
    static constexpr bool HAS_TAIL = !std::is_same<TStorageTail, void>::value;
    // Compressed tail storage words per row per thread (min0, min1,
    // signs/index) -- fixed, independent of row degree.
    static constexpr int  TAIL_WORDS_PER_ROW = 3;

    static_assert(BG_ == 1, "c2v_cache_split_band is BG1-specific (URD table)");
    static_assert(MAX_PARITY_NODES <= (HAS_TAIL ? 21 : 14),
                  "URD table covers BG1 rows 4..13 only (compressed tail extends to row 20)");
    static_assert(NUM_REG_C2V_NODES >= 4 && NUM_REG_C2V_NODES <= MAX_PARITY_NODES,
                  "register/shared split must lie inside the non-core rows");
    static_assert(has_resize<c2v_storage_noncore_t>::value,
                  "non-core storage must be per-row resizable (box_plus)");
    static_assert(!HAS_TAIL || (NUM_REG_C2V_NODES == 14),
                  "compressed tail assumes register rows 4..13 (tail rows 14..MAX-1)");
    static_assert(EXT_CACHE_ROWS >= NUM_REG_C2V_NODES,
                  "register rows 4..NUM_REG-1 always use the extension cache");
    static_assert(!EXT_GLOBAL || (HAS_TAIL && TAIL_EXP && MAX_PARITY_NODES == 21),
                  "extension relocation is implemented for the ALGO202 expanded-tail layout only");
    static_assert(!TAIL_EXP || EXT_GLOBAL,
                  "the expanded tail's shared budget requires the relocated extension columns");

    // Core rows go to shmem when storage exceeds min-sum size (4 words)
    static constexpr bool CORE_IN_SHMEM = (sizeof(c2v_storage_core_t) > 4 * sizeof(word_t));

    //------------------------------------------------------------------
    // c2v_cache_split_band() (legacy tensor interface params)
    __device__
    c2v_cache_split_band(char*                     smem,
                         const LDPC_kernel_params& params)
    {
        init_reg_c2v();
        setup_shmem_addresses(smem, params.Z_var * sizeof(app_t));
    }
    //------------------------------------------------------------------
    // c2v_cache_split_band() (transport-block interface params)
    __device__
    c2v_cache_split_band(char*                              smem,
                         const cuphyLDPCDecodeConfigDesc_t& params)
    {
        init_reg_c2v();
        const int Kb            = (1 == BG_) ? 22 : 10;
        // EXT_GLOBAL: the shared APP buffer holds only the 26 read/write
        // columns (info + core parity); the tail region starts at a
        // COMPILE-TIME offset independent of the runtime p.
        const int NUM_VAR_NODES = EXT_GLOBAL ? (Kb + 4) : (Kb + params.num_parity_nodes);
        setup_shmem_addresses(smem, params.Z * NUM_VAR_NODES * sizeof(app_t));
    }
    //------------------------------------------------------------------
    // ext_staged() (EXT_GLOBAL only)
    // Read row CHECK_IDX's loop-invariant extension value from the plane
    // the entry stager wrote (see bg1_a202_ext_stage_plane): one LDS off
    // the tail base at a compile-time offset -- exactly as cheap as the
    // baseline layout's shared extension read. Serves iteration 0 for
    // the always-executed rows (whose value then lives in the extension
    // register cache) and EVERY iteration for the runtime-conditional
    // rows 15..20 (whose plane is in the permanent survival strip).
    template <int CHECK_IDX>
    __device__ __forceinline__
    word_t ext_staged() const
    {
        return noncore_shm_[bg1_a202_ext_stage_plane(CHECK_IDX) * ZSTRIDE()];
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        // Register storage is zero-initialized in the constructor; the
        // shared region needs no initialization (see NOTE above).
    }
    //------------------------------------------------------------------
    // load_app_masked()
    // Load the APP words of row CHECK_IDX whose MASK bit is set, leaving
    // the other array entries untouched. Uses the same volatile
    // ld.shared primitive as app_loader, so values (and per-thread issue
    // order) are identical to the equivalent subset of a full load.
    // Called by the schedule BEFORE the row's barrier with a
    // bg1_preload_mask (columns proven untouched by the preceding row),
    // and by process_row_pre AFTER the barrier with the complement.
    template <int CHECK_IDX, uint32_t MASK, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void load_app_masked(word_t (&app)[NUM_APP_WORDS],
                         int    (&app_addr)[ROW_DEGREE],
                         int    smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        #pragma unroll
        for(int i = 0; i < ROW_DEGREE; ++i)
        {
            if((MASK >> i) & 1u)
            {
                app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
            }
        }
    }
    //------------------------------------------------------------------
    // store_app_masked()
    // Store the (updated) APP words of row CHECK_IDX whose MASK bit is
    // set (ALGO202 store sinking): the schedule calls this AFTER the
    // row's barrier for columns the next row(s) in the flush window
    // provably do not touch, retiring those stores out of the
    // pre-barrier tail that gates every warp's barrier arrival. Same
    // volatile st.shared primitive as app_writer::write_non_ext, so the
    // written bytes are identical to the in-place write-back.
    template <int CHECK_IDX, uint32_t MASK, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void store_app_masked(word_t (&app)[NUM_APP_WORDS],
                          int    (&app_addr)[ROW_DEGREE],
                          int    smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(0 == (MASK >> update_row_degree<BG, CHECK_IDX>::value),
                      "only write-back (non-extension) words can be store-sunk");
        #pragma unroll
        for(int i = 0; i < ROW_DEGREE; ++i)
        {
            if((MASK >> i) & 1u)
            {
                write_shared_word(app[i], smem_offset + app_addr[i]);
            }
        }
    }
    //------------------------------------------------------------------
    // write_app_sink()
    // Row write-back with the SINK_MASK words skipped (see the
    // process_row_pre SINK_MASK note). SINK_MASK == 0 degenerates to
    // exactly c2v_t::write_app (same app_writer loop, same
    // write_shared_word primitive, same order over the kept words).
    template <int CHECK_IDX, uint32_t SINK_MASK, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void write_app_sink(c2v_t& c2v,
                        word_t (&app)[NUM_APP_WORDS],
                        int    (&app_addr)[ROW_DEGREE],
                        int    smem_offset)
    {
        if constexpr(0 == SINK_MASK)
        {
            c2v.template write_app<CHECK_IDX>(app, app_addr, smem_offset);
        }
        else
        {
            constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
            #pragma unroll
            for(int i = 0; i < URD; ++i)
            {
                if(0 == ((SINK_MASK >> i) & 1u))
                {
                    write_shared_word(app[i], smem_offset + app_addr[i]);
                }
            }
        }
    }
    //------------------------------------------------------------------
    // process_row_pre()
    // process_row() with a caller-supplied prefetch hook invoked right
    // after this row's APP loads and BEFORE its check-node compute. The
    // hook is where the schedule issues the NEXT row's address
    // generation and independent APP preloads (columns this row provably
    // does not touch, see bg1_preload_mask): the loads enter the
    // shared-memory pipe while it is otherwise idle during this row's
    // compute, their latency hides behind the entire compute phase, and
    // the pre-barrier window keeps only this row's store burst. The
    // PRE_MASK words of app[] arrive already loaded by the previous
    // row's hook; the remaining words are loaded here. Only load ISSUE
    // order differs from process_row(): the check-node compute and APP
    // write-back are byte-for-byte the same calls, so the APP trajectory
    // is bit-identical. PRE_MASK == 0 with a no-op hook degenerates to
    // exactly process_row()'s behavior.
    //
    // SINK_MASK (ALGO202 store sinking, default 0 = original behavior):
    // the write-back of the masked words is SKIPPED here; their updated
    // values stay in app[] (and their addresses in app_addr[]) for the
    // schedule to retire AFTER the row's barrier via store_app_masked,
    // in the flush window whose rows provably do not touch those
    // columns. Written bytes and every kept store are identical; only
    // the sunk stores' issue point moves past the barrier.
    //
    // TAIL_PF (ALGO202 warm iterations only): this tail row's C2V record
    // -- and, above the extension register cache, its extension value --
    // arrive PREFETCHED in tail_pf_w_/ext_pf_ (issued by the schedule
    // from inside the PRECEDING row's body via tail_prefetch, per the
    // TAIL_PF_REC/EXT_ROWS policy masks); the row consumes them from
    // registers instead of loading at first use. false (the default, and
    // default) keeps the verbatim in-body loads.
    template <int CHECK_IDX, uint32_t PRE_MASK, bool IS_FIRST = false, bool IS_LAST = false, uint32_t SINK_MASK = 0, bool TAIL_PF = false, class TPrefetch, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row_pre(const TKernelParams& params,
                         word_t               (&app)[NUM_APP_WORDS],
                         int                  (&app_addr)[ROW_DEGREE],
                         int                  smem_offset,
                         TPrefetch&           prefetch,
                         bool                 is_last_iter = false)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(0 == (SINK_MASK >> update_row_degree<BG, CHECK_IDX>::value),
                      "only write-back (non-extension) words can be store-sunk");
        constexpr uint32_t REM_MASK = ~PRE_MASK; // remaining (post-barrier) words
        if constexpr (CHECK_IDX < 4)
        {
            static_assert(!CORE_IN_SHMEM,
                          "process_row_pre supports the register (compressed min-sum) core only");
            static_assert(0 == PRE_MASK,
                          "core rows are never preloaded (see PRE_MASK: CUR == 5 only)");
            // Core-row store sinking (ALGO202): the bidiagonal CARRY
            // word is never sunk (it is not stored here at all -- the
            // pair consumer re-stores the column), and the sink mask is
            // proven against the following row exactly like the
            // non-core rows. Everything else about the carry data flow
            // is unchanged.
            static_assert((0 == SINK_MASK) ||
                          (CHECK_IDX < 3 ? (0 == ((SINK_MASK >> 18) & 1u)) : true),
                          "the bidiagonal carry-out word cannot be store-sunk");
            // BG1 core parity-bidiagonal register handoff (bit-identical
            // data flow; message PLACEMENT only -- see the identical
            // process_row branch and the static_asserts below the class).
            // Columns 23/24/25 tie the adjacent core-row pairs
            // (0,1)/(1,2)/(2,3) with EQUAL circulant shifts, so thread
            // t's updated APP word for the column in row r IS exactly the
            // word thread t needs for row r+1: hand it over in a register
            // instead of an STS + (post-barrier) LDS round trip.
            constexpr int CARRY_IN  = ((CHECK_IDX == 1) || (CHECK_IDX == 2)) ? 17 :
                                      ((CHECK_IDX == 3) ? 18 : -1);
            constexpr int CARRY_OUT = (CHECK_IDX < 3) ? 18 : -1;
            #pragma unroll
            for(int i = 0; i < ROW_DEGREE; ++i)
            {
                if(i == CARRY_IN)
                {
                    app[i] = bidiag_carry_;
                }
                else
                {
                    app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                }
            }
            prefetch();
            c2v_t c2v;
            c2v.template compute_app<CHECK_IDX, TKernelParams, c2v_storage_core_t, IS_FIRST, IS_LAST>(params,
                                                                                   app,
                                                                                   c2v_storage_reg_core_[CHECK_IDX]);
            static_assert(update_row_degree<BG_, CHECK_IDX>::value == row_degree<BG, CHECK_IDX>::value,
                          "core rows write back every column");
            #pragma unroll
            for(int i = 0; i < ROW_DEGREE; ++i)
            {
                if(i == CARRY_OUT)
                {
                    bidiag_carry_ = app[i];
                }
                else if(0 == ((SINK_MASK >> i) & 1u))
                {
                    write_shared_word(app[i], smem_offset + app_addr[i]);
                }
            }
        }
        else if constexpr (CHECK_IDX < NUM_REG_C2V_NODES)
        {
            auto& st = c2v_storage_reg_.template get<CHECK_IDX>();
            using st_type = std::remove_reference_t<decltype(st)>;
            constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
            static_assert(NUM_APP_WORDS == URD + 1,
                          "BG1 non-core row must have exactly one extension column");
            static_assert(0 == (PRE_MASK >> URD),
                          "extension column must not be preloaded (register cache / staging race)");
            c2v_t c2v;
            if constexpr (IS_FIRST)
            {
                // Remaining words; capture the loop-invariant extension
                // APP for later iterations, as in process_row(). Under
                // EXT_GLOBAL the extension value comes from the transient
                // staging plane (the extension column has no APP slot).
                if constexpr (EXT_GLOBAL)
                {
                    app[URD] = ext_staged<CHECK_IDX>();
                    load_app_masked<CHECK_IDX, (REM_MASK & ((1u << URD) - 1u))>(app, app_addr, smem_offset);
                }
                else
                {
                    // (extension included: bg1_preload_mask never covers it)
                    load_app_masked<CHECK_IDX, REM_MASK>(app, app_addr, smem_offset);
                }
                prefetch();
                c2v.template compute_app<CHECK_IDX, TKernelParams, st_type, true, false>(params, app, st);
                noncore_ext_[CHECK_IDX - 4] = app[URD];
            }
            else
            {
                #pragma unroll
                for(int i = 0; i < URD; ++i)
                {
                    if((REM_MASK >> i) & 1u)
                    {
                        app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                    }
                }
                app[URD] = noncore_ext_[CHECK_IDX - 4];
                prefetch();
                c2v.template compute_app<CHECK_IDX, TKernelParams, st_type, false, IS_LAST>(params, app, st);
            }
            write_app_sink<CHECK_IDX, SINK_MASK>(c2v, app, app_addr, smem_offset);
        }
        else if constexpr (!HAS_TAIL)
        {
            // Trailing shared-C2V rows: same split as process_row()'s
            // trailing path, with the preloaded words skipped.
            constexpr int URD  = update_row_degree<BG, CHECK_IDX>::value;
            constexpr int WOFF = bg1_noncore_shm_words(NUM_REG_C2V_NODES, CHECK_IDX);
            static_assert(NUM_APP_WORDS == URD + 1,
                          "BG1 non-core row must have exactly one extension column");
            static_assert(0 == (PRE_MASK >> URD),
                          "extension column must not be preloaded (register cache / staging race)");
            typename c2v_storage_noncore_t::template resize<URD> st;
            c2v_t c2v;
            if constexpr (IS_FIRST)
            {
                load_app_masked<CHECK_IDX, REM_MASK>(app, app_addr, smem_offset);
                prefetch();
                c2v.template compute_app<CHECK_IDX, TKernelParams, decltype(st), true, false>(params, app, st);
                noncore_ext_[CHECK_IDX - 4] = app[URD];
            }
            else
            {
                #pragma unroll
                for(int i = 0; i < URD; ++i)
                {
                    if((REM_MASK >> i) & 1u)
                    {
                        app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                    }
                }
                app[URD] = noncore_ext_[CHECK_IDX - 4];
                #pragma unroll
                for(int w = 0; w < URD; ++w)
                {
                    st.w[w] = noncore_shm_[(WOFF + w) * ZSTRIDE()];
                }
                prefetch();
                c2v.template compute_app<CHECK_IDX, TKernelParams, decltype(st), false, IS_LAST>(params, app, st);
            }
            if constexpr (!IS_LAST)
            {
                #pragma unroll
                for(int w = 0; w < URD; ++w)
                {
                    noncore_shm_[(WOFF + w) * ZSTRIDE()] = st.w[w];
                }
            }
            write_app_sink<CHECK_IDX, SINK_MASK>(c2v, app, app_addr, smem_offset);
        }
        else
        {
            // Trailing tail rows (rows NUM_REG..MAX_P-1, ALGO202): common
            // tail processor (compressed or expanded storage form -- see
            // process_tail_row), preloads skipped.
            static_assert(0 == PRE_MASK,
                          "tail rows are never preloaded (see PRE_MASK: CUR == 5 only)");
            static_assert(0 == SINK_MASK,
                          "tail rows do not participate in store sinking (their write "
                          "burst shares the phase with the C2V record store; thread the "
                          "mask through process_tail_row before enabling it here)");
            static_assert(!TAIL_PF || (HAS_TAIL && !IS_FIRST),
                          "TAIL_PF applies to warm-iteration tail rows only");
            process_tail_row<CHECK_IDX, IS_FIRST, TAIL_PF>(params, app, app_addr, smem_offset,
                                                           prefetch, is_last_iter);
        }
    }
    //------------------------------------------------------------------
    // Compressed-tail storage round trip (HAS_TAIL only). Word w of row r
    // lives at noncore_shm_[(3*(r - NUM_REG) + w) * Z] (SoA planes, the
    // per-thread offset is already folded into noncore_shm_): fixed 3
    // words per row per thread regardless of row degree.
    // Compressed-record word offset: the baseline dense 3-word packing,
    // or the row's slot in the mixed expanded/compressed ALGO202 layout
    // (bg1_a202_tail_eoff) under EXT_GLOBAL.
    template <int CHECK_IDX>
    static constexpr int tail_comp_woff()
    {
        return EXT_GLOBAL ? bg1_a202_tail_eoff(CHECK_IDX)
                          : TAIL_WORDS_PER_ROW * (CHECK_IDX - NUM_REG_C2V_NODES);
    }
    template <int CHECK_IDX, class TSt>
    __device__ __forceinline__
    void tail_load(TSt& st)
    {
        constexpr int WOFF = tail_comp_woff<CHECK_IDX>();
#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
        if constexpr (c2v_storage_prenorm_packed<TSt>::value)
        {
            // 2-word pre-norm packed record: one LDS.64, thread t at
            // words {2t, 2t+1} of the slot's leading pair band
            // (conflict-free 8-byte stride).
            const uint2 r = *reinterpret_cast<const uint2*>(
                noncore_shm_ + (WOFF * ZSTRIDE()) + threadIdx.x);
            st.a.u32 = r.x;
            st.b.u32 = r.y;
        }
        else
#endif
        {
            st.min0 = noncore_shm_[(WOFF + 0) * ZSTRIDE()];
            st.min1 = noncore_shm_[(WOFF + 1) * ZSTRIDE()];
            st.signs_0_9_min0_index = noncore_shm_[(WOFF + 2) * ZSTRIDE()].u32;
        }
    }
    template <int CHECK_IDX, class TSt>
    __device__ __forceinline__
    void tail_store(const TSt& st)
    {
        constexpr int WOFF = tail_comp_woff<CHECK_IDX>();
#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
        if constexpr (c2v_storage_prenorm_packed<TSt>::value)
        {
            *reinterpret_cast<uint2*>(
                noncore_shm_ + (WOFF * ZSTRIDE()) + threadIdx.x) =
                make_uint2(st.a.u32, st.b.u32);
        }
        else
#endif
        {
            noncore_shm_[(WOFF + 0) * ZSTRIDE()] = st.min0;
            noncore_shm_[(WOFF + 1) * ZSTRIDE()] = st.min1;
            word_t sw;
            sw.u32 = st.signs_0_9_min0_index;
            noncore_shm_[(WOFF + 2) * ZSTRIDE()] = sw;
        }
    }
    //------------------------------------------------------------------
    // Expanded-tail storage round trip (TAIL_EXP only): URD per-edge
    // message words at the FIXED URD-prefix offset bg1_a202_tail_eoff
    // (compile-time -- every tail row is expanded at every p), pair-
    // interleaved two words per thread per 2-plane band (LDS.64/STS.64;
    // thread t owns words {2t, 2t+1}, so consecutive threads touch
    // consecutive 8-byte cells: conflict-free), with a trailing .32
    // plane when URD is odd.
    template <int CHECK_IDX, int URD>
    __device__ __forceinline__
    void tail_load_exp(word_t (&w)[URD])
    {
        constexpr int WOFF = bg1_a202_tail_eoff(CHECK_IDX);
        #pragma unroll
        for(int i = 0; (i + 1) < URD; i += 2)
        {
            const uint2 v = *reinterpret_cast<const uint2*>(
                noncore_shm_ + ((WOFF + i) * ZSTRIDE()) + threadIdx.x);
            w[i].u32     = v.x;
            w[i + 1].u32 = v.y;
        }
        if constexpr (URD & 1)
        {
            w[URD - 1] = noncore_shm_[(WOFF + URD - 1) * ZSTRIDE()];
        }
    }
    template <int CHECK_IDX, int URD>
    __device__ __forceinline__
    void tail_store_exp(const word_t (&w)[URD])
    {
        constexpr int WOFF = bg1_a202_tail_eoff(CHECK_IDX);
        #pragma unroll
        for(int i = 0; (i + 1) < URD; i += 2)
        {
            *reinterpret_cast<uint2*>(
                noncore_shm_ + ((WOFF + i) * ZSTRIDE()) + threadIdx.x) =
                make_uint2(w[i].u32, w[i + 1].u32);
        }
        if constexpr (URD & 1)
        {
            noncore_shm_[(WOFF + URD - 1) * ZSTRIDE()] = w[URD - 1];
        }
    }
    //------------------------------------------------------------------
    // Tail-row state prefetch policy (ALGO202, merging the earlier prefetch
    // pipeline onto the mixed expanded/packed tail): bit r of REC/EXT
    // selects whether tail row r's C2V record / extension-column value is
    // loaded one row EARLY, from inside the PRECEDING row's body (the
    // process_row_pre hook point: after that row's own APP loads, before
    // its check-node compute, where the shared pipe is otherwise idle for
    // the whole compute phase) instead of at first use. Legality at any
    // issue distance:
    //   * the record (expanded per-edge words or the 2-word packed pair)
    //     is THREAD-PRIVATE -- every band form folds threadIdx only
    //     (thread t owns words {2t, 2t+1}), written by this thread's
    //     tail_store/tail_store_exp in the previous iteration: same-
    //     thread program order needs no barrier;
    //   * the extension value of rows >= EXT_CACHE_ROWS lives in the
    //     PERMANENT survival planes, written only by iteration-0 entry
    //     staging and never rewritten, so the value is immutable across
    //     every warm iteration.
    // Values are bit-identical to the in-body loads they replace; only
    // the issue points move. 0/0 disables the pipeline entirely (and
    // every non-ALGO202 TU compiles it out via the default macros).
#if defined(LDPC2_A202_TAIL_PF_REC_MASK)
    static constexpr uint32_t TAIL_PF_REC_ROWS = LDPC2_A202_TAIL_PF_REC_MASK;
#else
    static constexpr uint32_t TAIL_PF_REC_ROWS = 0u;
#endif
#if defined(LDPC2_A202_TAIL_PF_EXT_MASK)
    static constexpr uint32_t TAIL_PF_EXT_ROWS = LDPC2_A202_TAIL_PF_EXT_MASK;
#else
    static constexpr uint32_t TAIL_PF_EXT_ROWS = 0u;
#endif
    template <int CHECK_IDX>
    static constexpr bool tail_pf_rec() { return 0 != ((TAIL_PF_REC_ROWS >> CHECK_IDX) & 1u); }
    template <int CHECK_IDX>
    static constexpr bool tail_pf_ext()
    {
        // Ext prefetch is meaningful only for the survival-strip rows
        // (>= EXT_CACHE_ROWS); ext-cached rows read a register anyway.
        return (0 != ((TAIL_PF_EXT_ROWS >> CHECK_IDX) & 1u)) &&
               (CHECK_IDX >= EXT_CACHE_ROWS) && EXT_GLOBAL;
    }
    //------------------------------------------------------------------
    // tail_prefetch()
    // Issue row CHECK_IDX's warm-iteration state reads EARLY (see the
    // policy note above): the record into tail_pf_w_ (expanded rows fill
    // words 0..URD-1; packed rows fill words 0..1; a 3-word compressed
    // record fills 0..2), the extension value into ext_pf_. Called from
    // the preceding row's body via the schedule's tail_state_prefetch
    // hook; the consuming row copies the pipeline registers out BEFORE
    // its own hook invocation refills them with the NEXT row's state.
    template <int CHECK_IDX>
    __device__ __forceinline__
    void tail_prefetch()
    {
        static_assert(HAS_TAIL && (CHECK_IDX >= NUM_REG_C2V_NODES) &&
                      (CHECK_IDX < MAX_PARITY_NODES),
                      "tail_prefetch covers the tail rows only");
        constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
        if constexpr (tail_pf_rec<CHECK_IDX>())
        {
#if defined(LDPC_MICRO_PACK_ARGMIN) && LDPC_MICRO_PACK_ARGMIN
            if constexpr (TAIL_EXP && (CHECK_IDX <= 18))
            {
                // Expanded record: same pair-band loads as tail_load_exp.
                constexpr int WOFF = bg1_a202_tail_eoff(CHECK_IDX);
                #pragma unroll
                for(int i = 0; (i + 1) < URD; i += 2)
                {
                    const uint2 v = *reinterpret_cast<const uint2*>(
                        noncore_shm_ + ((WOFF + i) * ZSTRIDE()) + threadIdx.x);
                    tail_pf_w_[i].u32     = v.x;
                    tail_pf_w_[i + 1].u32 = v.y;
                }
                if constexpr (URD & 1)
                {
                    tail_pf_w_[URD - 1] = noncore_shm_[(WOFF + URD - 1) * ZSTRIDE()];
                }
            }
            else
#endif
            {
                constexpr int WOFF = tail_comp_woff<CHECK_IDX>();
#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
                if constexpr (CHECK_IDX >= 19)
                {
                    // 2-word packed pair: same LDS.64 as tail_load.
                    const uint2 r = *reinterpret_cast<const uint2*>(
                        noncore_shm_ + (WOFF * ZSTRIDE()) + threadIdx.x);
                    tail_pf_w_[0].u32 = r.x;
                    tail_pf_w_[1].u32 = r.y;
                }
                else
#endif
                {
                    // 3-word compressed record: same planes as tail_load.
                    tail_pf_w_[0]     = noncore_shm_[(WOFF + 0) * ZSTRIDE()];
                    tail_pf_w_[1]     = noncore_shm_[(WOFF + 1) * ZSTRIDE()];
                    tail_pf_w_[2]     = noncore_shm_[(WOFF + 2) * ZSTRIDE()];
                }
            }
        }
        if constexpr (tail_pf_ext<CHECK_IDX>())
        {
            ext_pf_ = ext_staged<CHECK_IDX>();
        }
    }
    //------------------------------------------------------------------
    // tail_load_app()
    // Row APP loads shared by both tail forms: updated columns from the
    // shared APP image; the extension value from the register cache
    // (ext-cached rows after iteration 0), the iteration-0 staging plane
    // (EXT_GLOBAL first iteration), the survival plane (EXT_GLOBAL
    // deeper rows on warm iterations -- or the ext_pf_ pipeline register
    // when TAIL_PF prefetched it a row early), or the shared APP slot
    // (original layout).
    template <int CHECK_IDX, bool IS_FIRST, bool TAIL_PF = false, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void tail_load_app(word_t (&app)[NUM_APP_WORDS],
                       int    (&app_addr)[ROW_DEGREE],
                       int    smem_offset)
    {
        constexpr int  URD        = update_row_degree<BG, CHECK_IDX>::value;
        constexpr bool EXT_CACHED = (CHECK_IDX < EXT_CACHE_ROWS);
        if constexpr (IS_FIRST && !EXT_GLOBAL)
        {
            c2v_t c2v;
            c2v.template load_app<CHECK_IDX>(app, app_addr, smem_offset);
        }
        else
        {
            #pragma unroll
            for(int i = 0; i < URD; ++i)
            {
                app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
            }
            if constexpr (TAIL_PF && tail_pf_ext<CHECK_IDX>() && !IS_FIRST)
            {
                // Arrived one row early in the pipeline register (same
                // survival-plane value; see tail_prefetch).
                app[URD] = ext_pf_;
            }
            else if constexpr (EXT_GLOBAL && (IS_FIRST || !EXT_CACHED))
            {
                // Staging plane (iteration 0) / survival plane (deep rows,
                // every iteration): the same compile-time-mapped word.
                app[URD] = ext_staged<CHECK_IDX>();
            }
            else if constexpr (EXT_CACHED)
            {
                app[URD] = noncore_ext_[CHECK_IDX - 4];
            }
            else if constexpr (!EXT_GLOBAL)
            {
                app[URD] = smem_address_as<word_t>(smem_offset + app_addr[URD]);
            }
        }
    }
    //------------------------------------------------------------------
    // tail_row_compressed()
    // COMPRESSED-form tail row compute: the original cC2V row-processor
    // call (unchanged numerics/machinery) around the 3-word record round
    // trip, with the final iteration's dead record store elided at
    // RUNTIME (the record is read only by the next iteration's subtract;
    // the warm loop body serves the final iteration, so the compile-time
    // IS_LAST elision never fires).
    template <int CHECK_IDX, bool IS_FIRST, bool TAIL_PF = false, class TPrefetch, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void tail_row_compressed(const TKernelParams& params,
                             word_t               (&app)[NUM_APP_WORDS],
                             int                  (&app_addr)[ROW_DEGREE],
                             int                  smem_offset,
                             TPrefetch&           prefetch,
                             bool                 is_last_iter)
    {
#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
        // Deep rows 19/20: 2-word pre-norm packed record (one
        // LDS.64/STS.64 round trip; bit-identical reconstruction -- see
        // cC2V_storage_x2_tail_packed).
        using tail_st_t = std::conditional_t<(CHECK_IDX >= 19),
                                             cC2V_storage_x2_tail_packed,
                                             TStorageTail>;
#else
        using tail_st_t = TStorageTail;
#endif
        tail_st_t st;
        c2v_t     c2v;
        if constexpr (IS_FIRST)
        {
            tail_load_app<CHECK_IDX, IS_FIRST>(app, app_addr, smem_offset);
            prefetch();
            c2v.template compute_app<CHECK_IDX, TKernelParams, tail_st_t, true, false>(params, app, st);
        }
        else
        {
            tail_load_app<CHECK_IDX, IS_FIRST, TAIL_PF>(app, app_addr, smem_offset);
            if constexpr (TAIL_PF && tail_pf_rec<CHECK_IDX>())
            {
                // Record arrived one row early (see tail_prefetch);
                // consume the pipeline registers BEFORE the hook below
                // refills them with the NEXT row's state.
#if defined(LDPC2_A202_TAIL_PACKED2) && LDPC2_A202_TAIL_PACKED2
                if constexpr (c2v_storage_prenorm_packed<tail_st_t>::value)
                {
                    st.a = tail_pf_w_[0];
                    st.b = tail_pf_w_[1];
                }
                else
#endif
                {
                    st.min0                 = tail_pf_w_[0];
                    st.min1                 = tail_pf_w_[1];
                    st.signs_0_9_min0_index = tail_pf_w_[2].u32;
                }
            }
            else
            {
                tail_load<CHECK_IDX>(st);
            }
            prefetch();
            c2v.template compute_app<CHECK_IDX, TKernelParams, tail_st_t, false, false>(params, app, st);
        }
        if(!is_last_iter)
        {
            tail_store<CHECK_IDX>(st);
        }
    }
#if defined(LDPC_MICRO_PACK_ARGMIN) && LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // tail_row_expanded()
    // EXPANDED-form tail row: normalized min-sum with the row's C2V held
    // as its URD reconstructed per-edge messages instead of the 3-word
    // compressed record. BIT-IDENTICAL APP trajectory to the compressed
    // form, because
    //   * the stored word for edge i is `inc` from the add pass below,
    //     and inc == extract_pair(i) of the identical (min0, min1m0,
    //     argmin, signs) state the compressed form would have stored:
    //     on the add pass app[i] still holds the v2c message, so its
    //     sign IS the sign the compressed record would store for edge i
    //     (the LDPC_MICRO_PARTIAL_SIGN_ALL (B') identity), and the
    //     magnitude select/FMA reads the same min0/min1m0/argmin this
    //     scan just produced. The next iteration's subtract therefore
    //     removes bit-exactly the value the compressed path would have
    //     reconstructed -- as a bare __hsub2, with no per-edge
    //     HSET2+HFMA2+sign-reposition decompress;
    //   * the magnitude/argmin scan below replicates op-for-op the
    //     degree<16 branch of cC2V_row_context's (norm, app) constructor
    //     (packed-argmin init_row/update linear insertion) and its
    //     finalize() wrap-up (index recover/strip, sign product apply as
    //     finalize_mask -- (m & 0x7FFF7FFF) | sprod, identical for the
    //     non-clamping sign_mgr_pair_src<false> -- normalize-once,
    //     min1m0 precompute), MINUS the stored-signs accumulation that
    //     only the compressed record needs;
    //   * the row sign product is the XOR parity of the v2c words' sign
    //     bits -- identical to the popcount parity of the collected sign
    //     bits (the LDPC_MICRO_XOR_SIGNPROD identity the core rows
    //     already rely on);
    //   * the extract/APP-add pass is byte-for-byte the compressed
    //     path's (B') loop (extract_signed_magnitude + sign XOR +
    //     app_combine_x2).
    // The store is elided at RUNTIME on the final iteration (the words
    // are read only by the next iteration's subtract).
    template <int CHECK_IDX, bool IS_FIRST, bool TAIL_PF = false, class TPrefetch, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void tail_row_expanded(const TKernelParams& params,
                           word_t               (&app)[NUM_APP_WORDS],
                           int                  (&app_addr)[ROW_DEGREE],
                           int                  smem_offset,
                           TPrefetch&           prefetch,
                           bool                 is_last_iter)
    {
        constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
        constexpr int D   = URD + 1;
        static_assert(NUM_APP_WORDS == D, "tail row APP array size");
        const __half2 norm = params.norm.f16x2;
        // Previous iteration's C2V subtract (elided on the first
        // iteration exactly like the compressed path: storage is still
        // zero and never read).
        if constexpr (!IS_FIRST)
        {
            word_t w[URD];
            tail_load_app<CHECK_IDX, IS_FIRST, TAIL_PF>(app, app_addr, smem_offset);
            if constexpr (TAIL_PF && tail_pf_rec<CHECK_IDX>())
            {
                // Record arrived one row early (see tail_prefetch);
                // consume the pipeline registers BEFORE the hook below
                // refills them with the NEXT row's state.
                #pragma unroll
                for(int i = 0; i < URD; ++i)
                {
                    w[i] = tail_pf_w_[i];
                }
            }
            else
            {
                tail_load_exp<CHECK_IDX, URD>(w);
            }
            prefetch();
            #pragma unroll
            for(int i = 0; i < URD; ++i)
            {
                app[i].f16x2 = __hsub2(app[i].f16x2, w[i].f16x2);
            }
        }
        else
        {
            tail_load_app<CHECK_IDX, IS_FIRST>(app, app_addr, smem_offset);
            prefetch();
        }
        // Packed-argmin two-minimum scan (linear insertion), op-for-op
        // cC2V_row_context::init_row/update with the stored-signs
        // accumulation removed.
#if defined(LDPC_MICRO_ABS_MASK_CONST) && LDPC_MICRO_ABS_MASK_CONST
        const uint32_t abs_mask = g_ldpc2_x2_abs_mask;
#else
        const uint32_t abs_mask = 0x7FE07FE0u;
#endif
        word_t m0, m1;
        {
            word_t av0, av1;
            av0.u32 = (app[0].u32 & 0x7FE07FE0u);
            av1.u32 = (app[1].u32 & abs_mask) | 0x00010001u;
            m0.u32  = LDPC2_X2_SCAN_MINU(av0.u32, av1.u32);
            m1.u32  = LDPC2_X2_SCAN_MAXU(av0.u32, av1.u32);
        }
        #pragma unroll
        for(int i = 2; i < D; ++i)
        {
            word_t avp;
            avp.u32 = (app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16));
            word_t lo;
            lo.u32 = LDPC2_X2_SCAN_MAXU(avp.u32, m0.u32);
            m0.u32 = LDPC2_X2_SCAN_MINU(avp.u32, m0.u32);
            m1.u32 = LDPC2_X2_SCAN_MINU(lo.u32,  m1.u32);
        }
        // Row sign product: XOR parity of the v2c sign bits.
        uint32_t sprod = app[0].u32;
        #pragma unroll
        for(int i = 1; i < D; ++i)
        {
            sprod ^= app[i].u32;
        }
        sprod &= 0x80008000u;
        // finalize() replica (index recover/strip, sign-product apply as
        // finalize_mask, normalize-once, min1m0 precompute; no clamp:
        // sign_mgr_pair_src<false>).
        word_t min0_index;
        min0_index.u32 = (m0.u32 & 0x001F001F);
        m0.u32 &= 0xFFE0FFE0u;
        m1.u32 &= 0xFFE0FFE0u;
        m0.u32 = (m0.u32 & 0x7FFF7FFF) | sprod;
        m1.u32 = (m1.u32 & 0x7FFF7FFF) | sprod;
        m0.f16x2 = __hmul2(m0.f16x2, norm);
        m1.f16x2 = __hmul2(m1.f16x2, norm);
        word_t min1m0;
        min1m0.f16x2 = __hsub2(m1.f16x2, m0.f16x2);
        // Add pass: (B') signed-domain reconstruction; the inc word is
        // exactly the message the next iteration's subtract removes.
        word_t wout[URD];
        #pragma unroll
        for(int i = 0; i < URD; ++i)
        {
            word_t idx_pair;
            idx_pair.u16x2 = ushort2{static_cast<unsigned short>(i),
                                     static_cast<unsigned short>(i)};
            word_t mask_eq = hset2_bf_eq(idx_pair, min0_index);
            word_t inc;
            inc.f16x2 = __hfma2(mask_eq.f16x2, min1m0.f16x2, m0.f16x2);
            inc.u32   = inc.u32 ^ (app[i].u32 & 0x80008000u);
            wout[i]   = inc;
            app[i]    = app_combine_x2(app[i], inc, norm);
        }
        // Dead on the final iteration: read only by the next subtract.
        if(!is_last_iter)
        {
            tail_store_exp<CHECK_IDX, URD>(wout);
        }
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // process_tail_row()
    // Common ALGO202 tail-row body (rows NUM_REG..MAX_P-1): APP loads,
    // storage-FORM selection (compile-time TAIL_EXP), the extension
    // register-cache capture (rows below EXT_CACHE_ROWS only), and the
    // APP write-back. Both forms produce a bit-identical APP trajectory;
    // only the storage placement/width differs.
    template <int CHECK_IDX, bool IS_FIRST, bool TAIL_PF = false, class TPrefetch, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__ __forceinline__
    void process_tail_row(const TKernelParams& params,
                          word_t               (&app)[NUM_APP_WORDS],
                          int                  (&app_addr)[ROW_DEGREE],
                          int                  smem_offset,
                          TPrefetch&           prefetch,
                          bool                 is_last_iter)
    {
        constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
        static_assert(NUM_APP_WORDS == URD + 1,
                      "BG1 non-core row must have exactly one extension column");
        static_assert(!TAIL_PF || !IS_FIRST,
                      "the state pipeline serves warm iterations only (iteration 0 "
                      "has no previous-iteration record to read)");
#if defined(LDPC_MICRO_PACK_ARGMIN) && LDPC_MICRO_PACK_ARGMIN
        if constexpr (TAIL_EXP && (CHECK_IDX <= 18))
        {
            tail_row_expanded<CHECK_IDX, IS_FIRST, TAIL_PF>(params, app, app_addr, smem_offset,
                                                            prefetch, is_last_iter);
        }
        else
#endif
        {
            tail_row_compressed<CHECK_IDX, IS_FIRST, TAIL_PF>(params, app, app_addr, smem_offset,
                                                              prefetch, is_last_iter);
        }
        if constexpr (IS_FIRST && (CHECK_IDX < EXT_CACHE_ROWS))
        {
            // The row processor never modifies app[URD], so capturing it
            // after compute is equivalent to before.
            noncore_ext_[CHECK_IDX - 4] = app[URD];
        }
        c2v_t c2v;
        c2v.template write_app<CHECK_IDX>(app, app_addr, smem_offset);
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX, bool IS_FIRST = false, bool IS_LAST = false, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        if constexpr (CHECK_IDX < 4)
        {
            static_assert(!CORE_IN_SHMEM,
                          "GB203 band decoder keeps the compressed min-sum core C2V in registers");
            // BG1 core parity-bidiagonal register handoff (bit-identical
            // data flow; message PLACEMENT only). Columns 23/24/25 tie
            // the adjacent core-row pairs (0,1)/(1,2)/(2,3) with EQUAL
            // circulant shifts (0 on both entries), so thread t's updated
            // APP word for the column in row r IS exactly the word thread
            // t needs for row r+1: hand it over in a register instead of
            // an STS + (post-barrier) LDS round trip. Per iteration this
            // removes one STS from rows 0/1/2 and one LDS from rows 1/2/3
            // -- and the consumer's carry edge (its LAST-loading column)
            // no longer waits on shared-memory latency ahead of the
            // min-sum scan.
            // Safety of the skipped producer STS: the consumer is the
            // next row to touch the column (adjacent rows), and every
            // later reader -- non-core rows 8/11/13, the next iteration's
            // core rows, and the end-of-iteration dump / soft-output
            // paths -- reads the CONSUMER's own store of its updated
            // value, exactly as in the baseline. The end-of-iteration
            // shared APP image is bit-identical. Edge indices are proven
            // against the checked-in 38.212 descriptors by the
            // static_asserts below the class.
            constexpr int CARRY_IN  = ((CHECK_IDX == 1) || (CHECK_IDX == 2)) ? 17 :
                                      ((CHECK_IDX == 3) ? 18 : -1);
            constexpr int CARRY_OUT = (CHECK_IDX < 3) ? 18 : -1;
            // NOTE (measured, this base): extending the handoff to the
            // CROSS-ITERATION wrap pairs (col 25 row3->row2 every p, col
            // 23 row1->row0 p<=11, col 24 row2->row1 p<=8, load side only
            // with the stores kept for the dump image) was latency-
            // NEUTRAL at p5 and a REGRESSION at p12: the three
            // wrap registers live across
            // the whole iteration and the warp-uniform p compares cost
            // more than the 1-3 removed LDS return, so the reuse was
            // reverted to the within-iteration bidiagonal handoff below.
            #pragma unroll
            for(int i = 0; i < ROW_DEGREE; ++i)
            {
                if(i == CARRY_IN)
                {
                    app[i] = bidiag_carry_;
                }
                else
                {
                    app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                }
            }
            c2v_t c2v;
            c2v.template compute_app<CHECK_IDX, TKernelParams, c2v_storage_core_t, IS_FIRST, IS_LAST>(params,
                                                                                   app,
                                                                                   c2v_storage_reg_core_[CHECK_IDX]);
            static_assert(update_row_degree<BG_, CHECK_IDX>::value == row_degree<BG, CHECK_IDX>::value,
                          "core rows write back every column");
            #pragma unroll
            for(int i = 0; i < ROW_DEGREE; ++i)
            {
                if(i == CARRY_OUT)
                {
                    bidiag_carry_ = app[i];
                }
                else
                {
                    write_shared_word(app[i], smem_offset + app_addr[i]);
                }
            }
        }
        else if constexpr (CHECK_IDX < NUM_REG_C2V_NODES)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Per-row-sized register storage (as in c2v_cache_split),
            // with the extension-column APP served from the register
            // cache after the first iteration (see header NOTE).
            auto& st = c2v_storage_reg_.template get<CHECK_IDX>();
            using st_type = std::remove_reference_t<decltype(st)>;
            constexpr int URD = update_row_degree<BG, CHECK_IDX>::value;
            static_assert(NUM_APP_WORDS == URD + 1,
                          "BG1 non-core row must have exactly one extension column");
            c2v_t c2v;
            if constexpr (IS_FIRST)
            {
                // Full load (extension included -- from the staging plane
                // under EXT_GLOBAL); capture the loop-invariant extension
                // APP for later iterations. The row processor never
                // modifies app[URD], so capturing it after compute is
                // equivalent to before.
                if constexpr (EXT_GLOBAL)
                {
                    app[URD] = ext_staged<CHECK_IDX>();
                    #pragma unroll
                    for(int i = 0; i < URD; ++i)
                    {
                        app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                    }
                }
                else
                {
                    c2v.template load_app<CHECK_IDX>(app, app_addr, smem_offset);
                }
                c2v.template compute_app<CHECK_IDX, TKernelParams, st_type, true, false>(params, app, st);
                noncore_ext_[CHECK_IDX - 4] = app[URD];
            }
            else
            {
                // Updated columns from shared APP; extension column from
                // the register cache (its shared copy never changes).
                #pragma unroll
                for(int i = 0; i < URD; ++i)
                {
                    app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                }
                app[URD] = noncore_ext_[CHECK_IDX - 4];
                c2v.template compute_app<CHECK_IDX, TKernelParams, st_type, false, IS_LAST>(params, app, st);
            }
            c2v.template write_app<CHECK_IDX>(app, app_addr, smem_offset);
        }
        else if constexpr (!HAS_TAIL)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Trailing non-core row: per-row-sized SoA C2V storage in the
            // shared non-core region (the row's word offset within the
            // region is the compile-time prefix sum of the preceding
            // shared rows' update row degrees), with the same
            // extension-column register cache as the register rows.
            constexpr int URD  = update_row_degree<BG, CHECK_IDX>::value;
            constexpr int WOFF = bg1_noncore_shm_words(NUM_REG_C2V_NODES, CHECK_IDX);
            static_assert(NUM_APP_WORDS == URD + 1,
                          "BG1 non-core row must have exactly one extension column");
            typename c2v_storage_noncore_t::template resize<URD> st;
            c2v_t c2v;
            if constexpr (IS_FIRST)
            {
                // Previous iteration's C2V is only read by the subtract
                // pass, which IS_FIRST elides -- skip the (unread) C2V
                // load. Full APP load; capture the extension APP.
                c2v.template load_app<CHECK_IDX>(app, app_addr, smem_offset);
                c2v.template compute_app<CHECK_IDX, TKernelParams, decltype(st), true, false>(params, app, st);
                noncore_ext_[CHECK_IDX - 4] = app[URD];
            }
            else
            {
                #pragma unroll
                for(int i = 0; i < URD; ++i)
                {
                    app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
                }
                app[URD] = noncore_ext_[CHECK_IDX - 4];
                #pragma unroll
                for(int w = 0; w < URD; ++w)
                {
                    st.w[w] = noncore_shm_[(WOFF + w) * ZSTRIDE()];
                }
                c2v.template compute_app<CHECK_IDX, TKernelParams, decltype(st), false, IS_LAST>(params, app, st);
            }
            // The C2V written here is read only by the NEXT iteration's
            // subtract; on the final iteration (IS_LAST) the row processor
            // already elided the normalize-and-store into st, so the
            // write-back is dead and skipped.
            if constexpr (!IS_LAST)
            {
                #pragma unroll
                for(int w = 0; w < URD; ++w)
                {
                    noncore_shm_[(WOFF + w) * ZSTRIDE()] = st.w[w];
                }
            }
            c2v.template write_app<CHECK_IDX>(app, app_addr, smem_offset);
        }
        else
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Trailing tail row (rows NUM_REG..MAX_P-1, ALGO202): the
            // row's C2V lives in the shared SoA tail region as either the
            // 3-word compressed min-sum record (min-sum row processor via
            // hybrid_storage_row_map_x2, the same numerics class as the
            // E5M5 core rows) or the URD-word expanded per-edge form
            // (TAIL_EXP -- see process_tail_row for the bit-identical-
            // trajectory argument). Extension-column register cache for
            // rows below EXT_CACHE_ROWS only (see process_row_pre).
            struct no_pf { __device__ __forceinline__ void operator()() {} };
            no_pf pf;
            process_tail_row<CHECK_IDX, IS_FIRST>(params, app, app_addr, smem_offset,
                                                  pf, IS_LAST);
        }
    }

private:
    //------------------------------------------------------------------
    // setup_shmem_addresses()
    // Layout: [APP] [core C2V if CORE_IN_SHMEM] [trailing non-core C2V]
    // Must match algo202::get_app_c2v_shmem().
    __device__
    void setup_shmem_addresses(char* smem, int app_size_bytes)
    {
        int offset = app_size_bytes;

        if constexpr (CORE_IN_SHMEM)
        {
            offset = round_up_to_next(offset, static_cast<int>(alignof(c2v_storage_core_t)));
            c2v_storage_shm_core_ = reinterpret_cast<c2v_storage_core_t*>(smem + offset) + threadIdx.x;
            offset += 4 * ZSTRIDE() * static_cast<int>(sizeof(c2v_storage_core_t));
        }

        noncore_shm_ = reinterpret_cast<word_t*>(smem + offset) + threadIdx.x;
    }
    //------------------------------------------------------------------
    // init_reg_c2v()
    __device__
    void init_reg_c2v()
    {
        if constexpr (!CORE_IN_SHMEM)
        {
            #pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                c2v_storage_reg_core_[i].init();
            }
        }
        c2v_storage_reg_.init();
    }
    //------------------------------------------------------------------
    // Data
    struct empty_storage { __device__ void init() {} };
    using core_reg_element_t = std::conditional_t<CORE_IN_SHMEM, empty_storage, c2v_storage_core_t>;
    core_reg_element_t  c2v_storage_reg_core_[CORE_IN_SHMEM ? 1 : 4];
    // Rows 4..NUM_REG-1: per-row-sized register storage
    noncore_reg_chain<BG, 4, NUM_REG_C2V_NODES, c2v_storage_noncore_t> c2v_storage_reg_;
    // Rows NUM_REG..MAX_P-1: per-thread base of the SoA shared region
    word_t*             noncore_shm_;
    c2v_storage_core_t* c2v_storage_shm_core_; // only used when CORE_IN_SHMEM
    // Per-row loop-invariant extension-column APP (see header NOTE).
    // Deliberately uninitialized: written on the peeled first iteration
    // before any read; entries of rows >= runtime p are never touched.
    // Covers rows 4..EXT_CACHE_ROWS-1 only (see the EXT_CACHE_BOUND
    // parameter): rows above the bound re-load their extension APP from
    // shared instead of pinning more always-live registers
    // (bit-identical; see the tail branches). Sizing at the default bound
    // is unchanged.
    word_t              noncore_ext_[(EXT_CACHE_ROWS > 4) ? (EXT_CACHE_ROWS - 4) : 1];
    // Core parity-bidiagonal handoff register (see the core-row branch):
    // written by row r's carry edge, consumed by row r+1's carry edge
    // within the same iteration (produce always precedes consume; row 0
    // consumes nothing). Deliberately uninitialized.
    word_t              bidiag_carry_;
    // Tail-row state prefetch pipeline (see tail_prefetch / the TAIL_PF
    // consumption branches): the NEXT tail row's thread-private C2V
    // record (expanded rows fill words 0..URD-1 <= 6; the packed pair
    // fills 0..1; a 3-word compressed record fills 0..2) and, for rows
    // >= EXT_CACHE_ROWS, its immutable extension value, loaded from
    // inside the preceding row's body and consumed one row later.
    // Deliberately uninitialized: every consuming row's state was
    // written by the preceding row's hook (the schedule pairs producers
    // and consumers one-for-one via the same policy masks), and a
    // skipped (runtime-inactive-row) prefetch is never consumed. Dead --
    // and absent from SASS -- whenever the policy masks are 0 (every
    // non-ALGO202 TU).
    word_t              tail_pf_w_[6];
    word_t              ext_pf_;

    //------------------------------------------------------------------
    // Compile-time proof of the bidiagonal register-handoff contract
    // against the checked-in 38.212 descriptors: each producer row's
    // CARRY_OUT edge and its consumer row's CARRY_IN edge must name the
    // SAME variable-node column with the SAME circulant shift mod Z (so
    // the same thread owns the element in both rows). Adjacency of the
    // row pairs guarantees no other row touches the column in between.
    // Proved over EVERY lifting the kernel may see, not just one. With
    // RT_Z the same compiled kernel runs at every Z in [Z_MIN_, Z_FIXED],
    // so a proof stated at a single lifting would expire silently the
    // moment the zone widened -- which is exactly how algo201's
    // column-ownership proof went stale. With
    // Z_MIN_ defaulting to Z_FIXED this degenerates to the original
    // single-lifting check for every non-RT_Z instantiation.
    static_assert(bg1_bidiag_carry_ok_zone<Z_MIN_, Z_FIXED>::value,
                  "core parity-bidiagonal carry: producer and consumer edges must name "
                  "the same column with the same shift mod Z at EVERY lifting in the "
                  "zone, or the register handoff reads another thread's value");

};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_SPLIT_BAND_CUH_INCLUDED_)
