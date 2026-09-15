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

#if !defined(LDPC2_BOX_PLUS_X2_CUH_INCLUDED_)
#define LDPC2_BOX_PLUS_X2_CUH_INCLUDED_

#include "ldpc2.cuh"
#include "ldpc2_box_plus.cuh"
#include <type_traits>

////////////////////////////////////////////////////////////////////////
// APP update form (was CUPHY_LDPC_BP_X2_MATCH_ALGO40, removed)
//
// The APP add is app = __hfma2(bp, norm, app): one fused instruction,
// independent of the store-side hmul2, so the two issue in parallel.
//
// The alternative -- app = __hadd2(row_storage, app), reusing the rounded
// bp*norm already in row_storage -- was measured 1-5% slower here, because
// the add must wait for the hmul2. It was kept behind a build switch so a
// CICD numerical check could be bit-exact against ALGO 40; nothing in the
// tree ever enabled it, so the branch is gone. Consequence worth knowing:
// the FMA computes bp*norm with no intermediate rounding while the stored
// value IS rounded, so the add puts in a value ~1 fp16 ULP off from what
// the next iteration's subtract takes out. Bounded, does not affect BLER,
// and means output is not bit-identical to ALGO 40.

////////////////////////////////////////////////////////////////////////
// LDPC2_BP_X2_RAW_C2V
//
// Where the min-sum normalization of the STORED C2V message is applied.
// Default 0: every TU sees the verbatim baseline path (store the rounded
// normalized message row_storage.w[i] = __hmul2(loo[i], norm); next
// iteration subtracts it with __hsub2). A TU that opts in sets
// it to 1 BEFORE including this header:
//
//   store:    row_storage.w[i] = loo[i]              (RAW box-plus output;
//             the value is already in a register, so the store is free --
//             the per-column __hmul2 normalize disappears entirely)
//   subtract: app[i] = __hfma2(w[i], -norm, app[i])  (normalization folded
//             into the next iteration's subtract as a fused multiply-add
//             by the negated norm; same one-op-per-column cost as the
//             baseline __hsub2)
//
// Numerics: the subtract sees round(app - w*norm) (single rounding of the
// exact product) instead of app - round(w*norm) (two roundings). This is
// the same bounded ~1-ulp rounding-form difference as the APP add above
// (which uses the unrounded product in __hfma2 while the stored value was
// rounded), and like it does not affect BLER.
#ifndef LDPC2_BP_X2_RAW_C2V
#define LDPC2_BP_X2_RAW_C2V 0
#endif

////////////////////////////////////////////////////////////////////////
// Pre-normalized tree inputs (was LDPC2_BP_X2_NORM_INPUT, removed)
//
// Running the leave-one-out tree on pre-normalized inputs is legitimate --
// box_plus is min-magnitude/xor-sign and fp16 rounding is monotone under a
// positive scale, so tree(round(v*norm)) == round(tree(v)*norm) bit-exactly
// -- and it halves the heavy-pipe ops per row by turning the APP add into a
// lite __hadd2. It measured slower than the adopted RAW_C2V form on every
// base tried, and the two are mutually exclusive, so only RAW_C2V remains.

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_storage_x2_box_plus
// Per-row C2V storage for box-plus dual-codeword (__half2) processing.
// Stores UPDATE_ROW_DEGREE word_t values directly, where each word_t
// contains the same column's C2V message for 2 codewords (hi/lo halves).
// MAX_WORDS should be set to the maximum UPDATE_ROW_DEGREE across all
// rows that use this storage type.
template <int MAX_WORDS>
struct c2v_storage_x2_box_plus
{
    static constexpr int kMaxWords = MAX_WORDS;
    word_t w[MAX_WORDS];

    // Rebind to a different size (used by variable-size noncore storage)
    template <int N> using resize = c2v_storage_x2_box_plus<N>;

    c2v_storage_x2_box_plus() = default;

    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < MAX_WORDS; ++i)
        {
            w[i].u32 = 0;
        }
    }
};

////////////////////////////////////////////////////////////////////////
// is_box_plus_x2_storage
// Type trait to identify box_plus_x2 storage types.
template <class T>
struct is_box_plus_x2_storage : std::false_type {};

template <int N>
struct is_box_plus_x2_storage<c2v_storage_x2_box_plus<N>> : std::true_type {};

////////////////////////////////////////////////////////////////////////
// box_plus_fwd_bwd_x2_row_proc
// Row processor for box-plus with __half2 (dual-codeword) APP layout.
// Uses forward-backward prefix-suffix sweep to compute all
// "leave-one-out" box_plus products in 3*(d-1) operations.
//
// For __half2, each word_t in the APP array contains the same column
// from 2 different codewords. The box_plus() function (min.xorsign.abs.f16x2)
// operates on both halves simultaneously.
class box_plus_fwd_bwd_x2_row_proc
{
private:
#if LDPC2_C2V_SELECTIVE_WRITEBACK
    template <bool ORDERED_STORE>
    __device__ static void bg1_z256up_bp_shwin_x2_persist_word(word_t& dst, const word_t& value)
    {
        if constexpr(ORDERED_STORE)
        {
            *reinterpret_cast<volatile uint32_t*>(&dst) = value.u32;
        }
        else
        {
            dst = value;
        }
    }
#endif
    //------------------------------------------------------------------
    // bp_identity()
    // Identity element for box_plus (= min-magnitude with XOR sign): +inf
    // in both fp16 lanes. box_plus(identity, x) == x for any finite x, so
    // it is a safe seed for an empty product. (APP values are bounded by
    // clamping, so the +inf magnitude never wins a min.)
    __device__ static word_t bp_identity()
    {
        word_t w;
        w.u32 = 0x7C007C00u;
        return w;
    }

    //------------------------------------------------------------------
    // seg_reduce<LO,HI>()
    // box_plus reduction over leaves [LO, HI) of v, evaluated as a balanced
    // tree (critical path ceil(log2(HI-LO))). Used to fold the always-
    // included extension columns into a single "outside" seed. box_plus is
    // associative+commutative and exact, so the tree order is bit-identical
    // to a linear reduction.
    template <int LO, int HI, int RD>
    __device__ static word_t seg_reduce(const word_t (&v)[RD])
    {
        if constexpr(LO >= HI)            return bp_identity();
        else if constexpr(HI - LO == 1)   return v[LO];
        else
        {
            constexpr int M = LO + (HI - LO + 1) / 2;
            return box_plus(seg_reduce<LO, M>(v), seg_reduce<M, HI>(v));
        }
    }

    //------------------------------------------------------------------
    // seg_get<LO,HI>()
    // Product of leaves [LO,HI): a single leaf is v[LO] (no box_plus);
    // an internal range reads the memoized product P[LO][HI].
    template <int LO, int HI, int RD>
    __device__ static word_t seg_get(const word_t (&v)[RD], const word_t (&P)[RD][RD + 1])
    {
        if constexpr(HI - LO == 1) return v[LO];
        else                       return P[LO][HI];
    }

    //------------------------------------------------------------------
    // up_sweep<LO,HI>()
    // Computes box_plus over leaves [LO,HI) once, memoizing each internal
    // node's product into P[LO][HI]. Total box_plus ops = (HI-LO-1), the
    // minimum, with critical path ceil(log2(HI-LO)).
    // LDPC2_BP_REC_M chooses the recursive midpoint; up_sweep and down_sweep
    // MUST share it so the memoized P[LO][HI] keys align. The recursive
    // sub-tree split direction (REC_M) and per-node emission order
    // (UP_REC_SWAP/DOWN_REC_SWAP) are bit-exact reassociation/scheduling knobs
    // (box_plus is associative+commutative and exact). Swept per build on the
    // unified base; the default reproduces the baseline ordering exactly.
#ifndef LDPC2_BP_REC_M
#define LDPC2_BP_REC_M(LO, HI) ((LO) + ((HI) - (LO) + 1) / 2)
#endif
    template <int LO, int HI, int RD>
    __device__ static word_t up_sweep(const word_t (&v)[RD], word_t (&P)[RD][RD + 1])
    {
        if constexpr(HI - LO == 1) return v[LO];
        else
        {
            constexpr int M = LDPC2_BP_REC_M(LO, HI);
#if defined(LDPC2_BP_UP_REC_SWAP) && LDPC2_BP_UP_REC_SWAP
            word_t r = up_sweep<M, HI>(v, P);
            word_t l = up_sweep<LO, M>(v, P);
#else
            word_t l = up_sweep<LO, M>(v, P);
            word_t r = up_sweep<M, HI>(v, P);
#endif
            word_t p = box_plus(l, r);
            P[LO][HI] = p;
            return p;
        }
    }

    //------------------------------------------------------------------
    // down_sweep<LO,HI>()
    // Given `outside` = box_plus of every leaf NOT in [LO,HI), writes the
    // leave-one-out result for each leaf in [LO,HI) into out[]. Each leaf's
    // result is box_plus(outside, product-of-its-siblings), built top-down
    // by combining the parent's `outside` with the opposite child's
    // (already memoized) product. Critical path ceil(log2(HI-LO)).
    template <int LO, int HI, int RD, int U>
    __device__ static void down_sweep(word_t            outside,
                                       const word_t      (&v)[RD],
                                       const word_t      (&P)[RD][RD + 1],
                                       word_t            (&out)[U])
    {
        if constexpr(HI - LO == 1)
        {
            out[LO] = outside;
        }
        else
        {
            constexpr int M = LDPC2_BP_REC_M(LO, HI);
#if defined(LDPC2_BP_DOWN_REC_SWAP) && LDPC2_BP_DOWN_REC_SWAP
            down_sweep<M, HI>(box_plus(outside, seg_get<LO, M>(v, P)), v, P, out);
            down_sweep<LO, M>(box_plus(outside, seg_get<M, HI>(v, P)), v, P, out);
#else
            down_sweep<LO, M>(box_plus(outside, seg_get<M, HI>(v, P)), v, P, out);
            down_sweep<M, HI>(box_plus(outside, seg_get<LO, M>(v, P)), v, P, out);
#endif
        }
    }

public:
    //------------------------------------------------------------------
    // process_row()
    // ROW_DEGREE: total number of columns (including extension)
    // UPDATE_ROW_DEGREE: number of non-extension columns
    // TStorage: c2v_storage_x2_box_plus<MAX_WORDS>
    //
    // Computes the per-edge "leave-one-out" box_plus (normalized min-sum)
    // for every updated column with a BALANCED TREE instead of the linear
    // forward/backward prefix-suffix sweep. The box_plus operator
    // (min.xorsign.abs.f16x2 = min-magnitude with XOR sign) is exactly
    // associative and commutative with no accumulation rounding, so the
    // tree order produces BIT-IDENTICAL results to the linear sweep while
    // shortening the per-row dependency chain from O(degree) to
    // O(log degree). On this latency-bound kernel (1 CTA/SM) that is a
    // direct critical-path win on the higher-degree non-core rows, which
    // carry most of the per-iteration edges.
    //
    // For the APP update form and the rounding it implies, see the block at
    // the top of this file. In short:
    //   - Default (=0): app = __hfma2(bp, norm, app). Faster, but drifts
    //     ~1 fp16 ULP/iter from ALGO 40 because row_storage holds the
    //     rounded bp*norm while the FMA uses the unrounded version.
    //   - =1: app = __hadd2(row_storage, app). Reuses the rounded stored
    //     value so the per-iter add/subtract closes exactly and matches
    //     ALGO 40. 1-5% slower on ALGO 55.
    // BLER is unaffected either way.
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, bool IS_FIRST = false,
              bool IS_LAST = false, class TStorage
#if LDPC2_C2V_SELECTIVE_WRITEBACK
              , bool ORDERED_STORE = false
#endif
              >
    __device__
    static void process_row(word_t          (&app)[ROW_DEGREE],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        static_assert(UPDATE_ROW_DEGREE <= TStorage::kMaxWords,
                      "UPDATE_ROW_DEGREE exceeds c2v_storage_x2_box_plus MAX_WORDS");
        //--------------------------------------------------------------
        // Step 1: Subtract previous iteration's C2V from non-extension
        // APP values. Extension nodes (UPDATE_ROW_DEGREE..ROW_DEGREE-1)
        // have zero C2V and are not modified.
        // IS_FIRST (first BP iteration): C2V row storage is still zero, so
        // this subtract is a bit-exact no-op (x - (+/-0) == x in fp16) and
        // is skipped at compile time, eliminating the per-row C2V read.
        if constexpr (!IS_FIRST)
        {
#if LDPC2_BP_X2_RAW_C2V
            // Stored C2V is the RAW (un-normalized) leave-one-out box-plus
            // output; fold the normalization into the subtract as a fused
            // multiply-add by the negated norm:
            //   app - (w * norm) == __hfma2(w, -norm, app)
            // The sign flip is a single loop-invariant LOP3 on the norm
            // register (CSE'd across rows / hoisted out of the iteration
            // loop), NOT a per-column op.
            word_t nnorm;
            {
                word_t wn;
                wn.f16x2  = norm;
                nnorm.u32 = wn.u32 ^ 0x80008000u;
            }
            #pragma unroll
            for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
            {
                app[i].f16x2 = __hfma2(row_storage.w[i].f16x2, nnorm.f16x2, app[i].f16x2);
            }
#else
            #pragma unroll
            for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
            {
                app[i].f16x2 = __hsub2(app[i].f16x2, row_storage.w[i].f16x2);
            }
#endif
        }

        //--------------------------------------------------------------
        // Step 2 + 3: leave-one-out box_plus + inline normalize/APP update.
        if constexpr (UPDATE_ROW_DEGREE == 0)
        {
            // No non-extension columns to update (nothing to do).
        }
        else if constexpr (ROW_DEGREE <= 1)
        {
            // Degenerate: single column
#if LDPC2_BP_X2_RAW_C2V
            row_storage.w[0] = app[0];
#else
            row_storage.w[0].f16x2 = __hmul2(app[0].f16x2, norm);
#endif
            app[0].f16x2 = __hfma2(app[0].f16x2, norm, app[0].f16x2);
        }
        else if constexpr (ROW_DEGREE == 2)
        {
            // Trivial swap: C2V[0] = app[1], C2V[1] = app[0]
            word_t bp0 = app[1];
            word_t bp1 = app[0];
#if LDPC2_BP_X2_RAW_C2V
            row_storage.w[0] = bp0;
#else
            row_storage.w[0].f16x2 = __hmul2(bp0.f16x2, norm);
#endif
            app[0].f16x2 = __hfma2(bp0.f16x2, norm, app[0].f16x2);
            if constexpr (1 < UPDATE_ROW_DEGREE)
            {
#if LDPC2_BP_X2_RAW_C2V
                row_storage.w[1] = bp1;
#else
                row_storage.w[1].f16x2 = __hmul2(bp1.f16x2, norm);
#endif
                app[1].f16x2 = __hfma2(bp1.f16x2, norm, app[1].f16x2);
            }
        }
        else
        {
            //----------------------------------------------------------
            // Balanced-tree leave-one-out over the UPDATE_ROW_DEGREE
            // updated columns. The always-included extension columns
            // [UPDATE_ROW_DEGREE, ROW_DEGREE) fold into the initial
            // "outside" seed (a single box_plus tree of their own), so
            // every produced result already accounts for them.
            word_t (&tin)[ROW_DEGREE] = app;
            word_t loo[UPDATE_ROW_DEGREE];
            word_t outside = seg_reduce<UPDATE_ROW_DEGREE, ROW_DEGREE>(tin);
            if constexpr (UPDATE_ROW_DEGREE == 1)
            {
                loo[0] = outside;
            }
            else
            {
                // Split the root manually and combine `outside` with each
                // half-product directly. The full-row product is never used
                // for leave-one-out, and box_plus is asm volatile (not DCE'd),
                // so computing it would just waste an op per row.
                // Root split point (M0) + top-level up/down emission order are
                // bit-exact reassociation/scheduling knobs. The up->down feeding
                // is asymmetric (one half's product seeds the other half's
                // down-sweep), so the balanced split is not automatically the
                // critical-path optimum -- swept per build. Default = the
                // baseline.
                word_t P[ROW_DEGREE][ROW_DEGREE + 1];
#ifndef LDPC2_BP_M0
#define LDPC2_BP_M0(URD) (((URD) + 1) / 2)
#endif
                constexpr int M0 = LDPC2_BP_M0(UPDATE_ROW_DEGREE);
#if defined(LDPC2_BP_UP_SWAP) && LDPC2_BP_UP_SWAP
                word_t Rp = up_sweep<M0, UPDATE_ROW_DEGREE>(tin, P);
                word_t Lp = up_sweep<0,  M0>(tin, P);
#else
                word_t Lp = up_sweep<0,  M0>(tin, P);
                word_t Rp = up_sweep<M0, UPDATE_ROW_DEGREE>(tin, P);
#endif
#if defined(LDPC2_BP_DOWN_SWAP) && LDPC2_BP_DOWN_SWAP
                down_sweep<M0, UPDATE_ROW_DEGREE>(box_plus(outside, Lp), tin, P, loo);
                down_sweep<0,  M0>(box_plus(outside, Rp), tin, P, loo);
#else
                down_sweep<0,  M0>(box_plus(outside, Rp), tin, P, loo);
                down_sweep<M0, UPDATE_ROW_DEGREE>(box_plus(outside, Lp), tin, P, loo);
#endif
            }

            #pragma unroll
            for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
            {
                // IS_LAST (final BP iteration): the C2V written here is read
                // only by the NEXT iteration's previous-C2V subtract; on the
                // final iteration there is none, so this store (and its
                // __hmul2) is provably dead. Eliding it at compile time lets
                // ptxas DCE the normalize-and-store that the runtime iteration
                // loop otherwise forces it to keep. The APP update below uses
                // loo[] directly (not row_storage), so the decoded output is
                // bit-identical to a normal iteration.
                if constexpr (!IS_LAST)
                {
#if LDPC2_BP_X2_RAW_C2V
                    // Storage is a register copy -- no op. RAW_C2V keeps the
                    // un-normalized output; the next iteration's subtract
                    // applies norm via __hfma2 (see Step 1).
#if LDPC2_C2V_SELECTIVE_WRITEBACK
                    bg1_z256up_bp_shwin_x2_persist_word<ORDERED_STORE>(row_storage.w[i], loo[i]);
#else
                    row_storage.w[i] = loo[i];
#endif
#else
                    word_t next;
                    next.f16x2 = __hmul2(loo[i].f16x2, norm);
#if LDPC2_C2V_SELECTIVE_WRITEBACK
                    bg1_z256up_bp_shwin_x2_persist_word<ORDERED_STORE>(row_storage.w[i], next);
#else
                    row_storage.w[i] = next;
#endif
#endif
                }
                app[i].f16x2 = __hfma2(loo[i].f16x2, norm, app[i].f16x2);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////
// hybrid_storage_row_map_x2
// Row processing dispatch template that selects between box_plus and
// min-sum based on the storage type:
// - c2v_storage_x2_box_plus<N> → box_plus_fwd_bwd_x2_row_proc
// - Any other storage → context-based min-sum row processor
//
// Template parameters match the context_storage_row_map interface:
// BG, CHECK_IDX, TC2VStorage are from the C2V_row_proc orchestrator.
// T, TRowContext, TRowProc are bound by the kernel config.
template <int                    BG,
          int                    CHECK_IDX,
          class                  TC2VStorage,
          typename               T,
          template <class> class TRowContext,
          template <class> class TRowProc>
struct hybrid_storage_row_map_x2
{
    using row_proc_t = std::conditional_t<
        is_box_plus_x2_storage<TC2VStorage>::value,
        box_plus_fwd_bwd_x2_row_proc,
        TRowProc<TRowContext<TC2VStorage>>
    >;
};

} // namespace ldpc2

#endif // !defined(LDPC2_BOX_PLUS_X2_CUH_INCLUDED_)
