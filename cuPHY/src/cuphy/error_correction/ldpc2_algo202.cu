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

// LDPC decode kernel for BG1 / Z in {256..384} / p=12..21 (2 codewords per CTA).
//
// Deep-p half of the mid-p range: algo201 covers p=4..11 at every Z, this
// kernel covers p=12..21, and together they are contiguous.
//
// CFG_MIN_P IS 12, AND ROWS 12..14 ARE RUNTIME-CONDITIONAL BECAUSE OF IT.
// That is the whole cost of starting the band here: those three rows would
// otherwise be compile-time unconditional, and every schedule decision below
// that mentions "conditional rows" traces back to this choice. It was taken
// on measurement over p=12..21 x five liftings; do not move CFG_MIN_P without
// re-measuring the band as a whole.
//
// Rows below 14 run the inherited schedule unchanged. Audited: the bidiagonal
// parity carry re-stores every carried column through its pair consumer, so
// rows 15+ reading V25 -- and rows 16/20 reading V22 -- always see a fresh
// shared copy. What changes for p>14:
//   * rows 14..20 hold their C2V COMPRESSED (min-sum: min0/min1/signs+
//     argmin, 3 words -- cC2V_storage_x2_low_degree) in a shared-memory
//     TAIL region (c2v_cache_split_band TStorageTail band): the register
//     plan cannot absorb the ~46 extra box-plus words and a box-plus
//     shmem tail (URD words/row) exceeds the 99 KB budget past p=18,
//     while the 3-word compressed tail fits through p=21 (p=22..23 would
//     need a sub-12-byte pack: a separate numerics decision).
//     Tail rows therefore share the E5M5/min-sum numerics class of the
//     core rows (the same compressed machinery the algo26..52 family
//     uses for every non-core row).
//   * the entry stager grows to 11 rounds (MIN_COLS=32 -> 8 unconditional,
//     3 runtime-p tail rounds flushed after rows 11/12/13 -- every staging
//     live range ends inside the compile-time-unconditional row region;
//     the just-in-time flush rows 15/19 for rounds 9/10 were re-audited
//     on GB203 and only extended register live ranges without buying
//     any remaining load-latency hiding).
//   * CFG_MIN_P=12: rows 0..11 are compile-time-unconditional. (At the
//     previous CFG_MIN_P=15 that region reached row 14; the three rows it
//     gave up are the whole cost of the p=12..14 extension.)
//
// Inherited schedule and numerics, carried over unchanged:
//   * the check-node numerics / box-plus / pairwise min-sum micro-win stack
//     (TU-local toggles below). GB203-only build: core rows
//     use compressed min-sum C2V storage in registers (E5M5 packed-argmin
//     scan, which fits GB203's ~100 KB shmem). The exact-fp16
//     box_plus-in-shared-memory core (which needed ~228 KB, e.g. GH200) and
//     the legacy tensor interface have been dropped -- only the
//     transport-block kernel ldpc2_BG1_z256up_cms_shtail_x2_tb_msc remains.
//     Non-core rows 4..13 use box-plus storage in registers; rows 14..20 use
//     the compressed shared tail.
//   * a bounded schedule: only rows 0..20 are instantiated (no dead rows
//     21..45), CFG_MIN_P = 12 removes the runtime IS_LAST_ROW check for rows
//     0..11, and first-iteration elision removes the provably-dead C2V
//     subtract on iteration 0.
//     The final iteration is NOT peeled: it runs the shared (warm) iteration
//     body. A peeled MODE::last body (which elided the dead C2V pack) was
//     measured on GB203/p5 to cost ~3.8% of kernel time in cold instruction
//     fetch while its dead-work elision saved only ~0.2%: every C2V store it
//     elided is a REGISTER write here (NUM_REG_PARITY = 14 >= max p), so the
//     peel removed only ALU, not memory traffic. Running the normal body on
//     the final iteration is bit-identical (the C2V packs write registers no
//     one reads afterwards) and keeps the instruction stream in the warm loop.
//   * the hard-decision output uses the single-pass, guard-free all-warps
//     writer (22 words per warp, Z/32 warps); its precondition holds at
//     every lifting with Z % 32 == 0, which is the whole zone.
//
// Decisions worth restating (the alternatives and the measurements behind
// them are in the internal optimization report, not here):
//   * APP addressing: dp_desc -- the measured winner on a compute-bound
//     stack of this anatomy. A compile-time-immediate generator remains the
//     natural A/B candidate wherever Z is pinned.
//   * __launch_bounds__(384, 1): the all-register C2V cache at p=14 needs
//     ~168 registers; 2 CTAs/SM would cap registers at 85 and spill.
//   * Norm: global table norm, with the in-kernel E5M5_NORM_SCALE covering
//     the packed-argmin truncation to 5 mantissa bits. A per-p compensation
//     sweep is future work.

// ---------------------------------------------------------------------------
// Check-node numerics micro-wins for the BG1 compressed-min-sum core rows
// (rows 0-3, degree 19); the measured-best stack for BG1/largeZ/mb14.
// These toggles MUST be defined BEFORE
// ldpc2_c2v_x2.cuh is first included, so they are scoped to THIS translation
// unit only; every other TU that includes the header still sees the verbatim
// baseline paths (see the LDPC_MICRO_PACK_ARGMIN master toggle in
// ldpc2_c2v_x2.cuh).
#define LDPC_MICRO_PACK_ARGMIN    1
#define LDPC_MICRO_ABS_MASK_CONST 1
#define LDPC_MICRO_PARTIAL_SIGN   1
#define LDPC_MICRO_PREDIFF        1
// FUSE_NORM=1 re-measured as a regression at p12 on this base (with the
// bidiagonal carry) -- moving the 2x HMUL2 off the finalize spine does not
// pay for the per-column HADD2->HFMA2 swap + pack-side HMUL2/HSUB2 here.
// Re-measured AGAIN on the composed tree under the PAIRWISE=0 linear scan:
// still a regression. The normalize-once choice stands.
#define LDPC2_X2_FUSE_NORM        0
#define LDPC_MICRO_SPLIT_REDUCE   1
// Scan-half emission order: CORE_SWAP=1 re-measured neutral-to-noise on
// the bidiag-carry base and AGAIN on the full combined base (at both p5
// and p12). On THIS composed tree (PAIRWISE=0 linear scan) it re-measured
// WORSE at p12. Keep the default order.
#define LDPC2_X2_CORE_SWAP        0
// Core scan split shape (degree-19 two-minimum reduction): SPLIT-way
// independent partial scans merged associatively; HALF is the SPLIT==2
// split point. Re-adjudicated at p12 on the merged base, where
// HALF=8 lost 2.8% and neither HALF=10 nor SPLIT=3 improved on the
// reference. RE-adjudicated AGAIN on THIS composed tree under the inverted
// PAIRWISE=0 linear scan, where HALF=8, HALF=10, SPLIT=3 and SPLIT=4 every
// one re-measured worse -- the (2, 9) shape is the measured optimum under
// BOTH scan forms.
#define LDPC2_X2_CORE_SPLIT       2
#define LDPC2_X2_CORE_HALF        9
// Measured and rejected on this base, at p12: packed-integer LOP3 selects
// for the core reconstruction cost 6.2%, running the packed-argmin scan on
// the fp16 pipe (HMNMX2) costs 4.9%, and together 10.7% -- additive, not
// compensating. On GB203 the INT/ALU datapath is the scarce compute
// resource (half the fp16x2 rate), so moving work onto it starves the
// critical scan, and moving the scan off it lengthens the exposed
// min/2nd-min chain. The baseline scan/reconstruct pipe assignment is the
// measured optimum here.
// Pairwise-merge scan verdict INVERTED AGAIN on this composed tree (unpeel
// + narrowed entry head gate + bidiag carry + (4,5) preload + all-ceil/
// all-swap box-plus tree): the LINEAR insertion scan (0) now wins at every
// zone point measured -- p5, p10, p12 and p14 -- by a uniform ~0.7-1.3%.
// The pairwise merge saves VIMNMX ops on the m0
// chain but lengthens the per-pair dependency ladder; with the composed
// tree's shorter core-row critical path (register carry feeds the scan
// head, smaller i-footprint after the unpeel) the exposed scan latency,
// not the op count, binds -- so the shorter-latency linear insertion wins.
// Bit-identical output either way (same min/2nd-min reduction).
#define LDPC_PAIRWISE_MINSUM      0
// XOR-tree row sign product for the degree-19 core scan (bit-identical:
// parity-of-signs == XOR of sign bits). Removes the serial sign-accumulate
// OR chain + POPC fold from the core row's finalize critical path; the
// packed sign words are tree-OR built off-path for storage. See
// ldpc2_c2v_x2.cuh (E).
#define LDPC_MICRO_XOR_SIGNPROD   1
// Extend the XOR-tree sign product + balanced OR-tree sign-word build to the
// LOW-degree compressed rows (under the mixed ALGO202 tail this is rows
// 19/20, the packed-record rows -- rows 14..18 are EXPANDED and run their
// own inline XOR-parity scan in tail_row_expanded; rows 4..13 are box-plus).
// Replaces each compressed row's serial sign-OR accumulation + two-POPC
// parity fold (the 7-op serial chain that gated finalize -> pack) with the
// same balanced-tree forms the degree-19 core uses. Bit-identical storage
// word and sign product (OR/XOR associativity + GF(2) parity linearity, see
// (E'') in ldpc2_c2v_x2.cuh). This measured as REGRESSING on the
// all-packed tail but WINNING once the shallow tail rows are expanded --
// this merged tree has rows 14..18 expanded, the winning configuration;
// re-measured on the merge.
#define LDPC_MICRO_XOR_SIGNPROD_LOWDEG 1 // adopted: required composition partner of TAIL_PACKED2 (alone: net-negative)
// With the XOR-tree product, signed-domain reconstruction for EVERY add-pass
// column ends the packed sign words' live range at get_storage() (no add-loop
// reader at all) and drops the keep-set's per-column reposition shifts.
// Bit-identical (see (B') in ldpc2_c2v_x2.cuh). Re-adjudicated on the
// merged base: =0 measured 5.5% SLOWER at p12 -- keep 1.
#define LDPC_MICRO_PARTIAL_SIGN_ALL 1
// Measured and rejected: folding the +/-1.0 sign application into an HFMA2
// regressed in every mask combination, and again after a register-plan audit
// lowered the warm-loop liveness peak from 166 to 156/168. The fp16 FMA pipe
// is this kernel's top-utilized pipeline, so folding the sign into it starves
// the true bottleneck regardless of register slack.

// ---------------------------------------------------------------------------
// Box-plus leave-one-out tree split points and emission orders
// (bit-identical output, pure scheduling knobs -- box_plus is exactly
// associative/commutative, so every setting yields the same APP
// trajectory). Must be defined BEFORE ldpc2_box_plus_x2.cuh.
//
// JOINTLY re-adjudicated at p12 on THIS merged base (bidiag core carry +
// row4->5 preload + compile-time-immediate addressing + XOR sign tree);
// single-knob verdicts measured on pre-merge bases did NOT transfer.
// Starting from the combined reference (ceil REC_M + UP/DOWN_SWAP), adding
// DOWN_REC_SWAP=1, then UP_REC_SWAP=1, then M0-ceil each improved on it in
// turn, and the all-ceil/all-swap point is what is retained -- even though
// DOWN_REC_SWAP had measured as a regression pre-merge.
// Every single-bit neighbor of the retained point re-measured worse:
// M0-floor, REC_M-floor, UP_SWAP-off, DOWN_SWAP-off, UP_REC_SWAP-off and
// DOWN_REC_SWAP-off. M0-ceil alone (without the REC_SWAPs) is neutral and
// was the WORST pre-merge single -- the shape and emission knobs interact,
// so they must be swept jointly, not one-at-a-time.
// Zone check: p5 neutral (only the degree-3 stub row uses the tree), p10
// and p14 both improved.
// RE-adjudicated on the COMPOSED tree (unpeel + narrowed entry head
// gate + carry + preload, under both PAIRWISE settings): every single-bit
// neighbor is again equal or worse -- M0-floor and REC_M-floor worse,
// UP_SWAP-off worse, DOWN_SWAP-off and DOWN_REC_SWAP-off a tie (keep the
// current setting), UP_REC_SWAP-off a wash across p10/p12/p14 (keep). The
// all-ceil/all-swap point is stable across this merge.
#define LDPC2_BP_M0(URD)          (((URD)+1)/2)        // ceil root split
#define LDPC2_BP_REC_M(LO,HI)     ((LO)+((HI)-(LO)+1)/2) // ceil recursive split
#define LDPC2_BP_UP_SWAP          1
// DOWN_SWAP tie-break under the adopted RAW_C2V store form: the p12/p14
// tie either way pre-RAW_C2V resolved to a consistent
// 3-of-4-point preference for 0 once the tree's stores became free
// register copies (0 wins at p10, p12 and p14; p5 is a wash). Bit-identical
// either way (associative min-magnitude/xor-sign operator).
// UP_REC_SWAP=0 on top re-measured as an exact tie -- keep 1.
#define LDPC2_BP_DOWN_SWAP        0
#define LDPC2_BP_UP_REC_SWAP      1
#define LDPC2_BP_DOWN_REC_SWAP    1
// Non-core C2V normalize-and-store placement (both implemented in
// ldpc2_box_plus_x2.cuh, default 0 = verbatim baseline for every other
// TU). The RAW_C2V verdict INVERTED on the composed tree:
//   LDPC2_BP_X2_RAW_C2V: store the RAW tree output (free register copy
//     -- the per-column store-side __hmul2 normalize disappears) and fold
//     norm into the next iteration's subtract as __hfma2(w, -norm, app).
//     Pre-merge it measured 0.9890x at p12 -- i.e. slower (the
//     subtract moved onto the then-saturated FMA-heavy pipe). On THIS
//     composed tree (unpeel + entry overlap + PAIRWISE=0 linear scan) it
//     WINS at every zone point measured -- p5, p10, p12 and p14
//     -- with the linear scan off the int pipe's merge ladder and the
//     shorter warm-loop body, the freed store-side HMUL2 now pays more
//     than the subtract-side HSUB2->HFMA2 swap costs. ADOPTED (1).
//     Numerics: bounded ~1-ulp rounding-form difference documented in
//     ldpc2_box_plus_x2.cuh (round(app - w*norm) vs app - round(w*norm)),
//     the same form family as the APP add itself; keyed
//     only to code structure, and re-validated through the full BLER
//     p5/p10/p14 + endpoint APP/LLR gates on adoption.
//   Running the tree on pre-normalized inputs instead (bit-identical stored
//     C2V, free store, __hadd2 APP add) was the alternative and measured
//     slower at p12 both pre-merge and on the composed tree. Rejected.
#define LDPC2_BP_X2_RAW_C2V       1
// Compressed min-sum storage for the non-core rows (A/B, see the
// noncore_storage_t block below). Measured 0.8124x at p12, i.e. much
// slower: the per-column FMA-select reconstruction on both passes and
// the packed-argmin scan add ~33% static instructions while the freed
// ~30 registers are absorbed by ptxas (161 vs 167 REG) with no occupancy
// change (shared memory pins 1 CTA/SM for p>=11; 2 CTAs/SM needs <=80
// regs). The full-width box_plus form wins decisively.
// Entry-staging tail-round LDG issue placement (bit-identical either way --
// only the issue point of the predicated tail loads moves):
//   1: issue_tail() BEFORE the head-visibility barrier, in the shadow of
//      the still-in-flight head loads (the tail LDGs sit behind the head
//      loads in program order, so the head arrival that gates the barrier
//      is not delayed, while round 6 gets the full core-row span of
//      latency shadow before its after_row<3> flush).
//   0: issue_tail() AFTER the head barrier (the pre-merge default), so the
//      head drain never competes with tail traffic in the L2 queue.
// The two candidate entry designs disagreed on this point (one measured
// pre-barrier issue regressing p14), so it was re-adjudicated by measurement
// on THIS composed tree (unpeel + narrowed head gate + carry + preload):
// Post-barrier issue (PRE_BARRIER=0) wins p10/p12/p14, by the widest
// margin at p14 -- the point
// with all 3 tail rounds active, where the 6 extra pre-barrier LDGs delay
// the head drain -- and loses at p5 only within noise. KEEP 0.
#define LDPC2_A202_TAIL_PRE_BARRIER 0
// Extension-column relocation (see EXT_GLOBAL in ldpc2_c2v_cache_split_band.cuh):
// the degree-1 extension columns' APP is loop-invariant -- staged once and
// never written back -- yet in the baseline layout those (p-4) columns
// occupy up to 26112 B (p21) of the ~99 KiB shared budget that deep p
// exhausts to the byte, while DRAM sits at ~10% and L2 at ~4%. Relocate
// that state out of the persistent shared image: the shared APP buffer
// holds only the 26 read/write columns, the entry stager's tail rounds
// stage the extension values into a TRANSIENT strip inside the tail C2V
// region (planes dead until their owning row's first C2V store -- see
// bg1_a202_ext_stage_plane), rows below EXT_CACHE_BOUND keep the register
// capture, and the runtime-conditional rows above it re-materialize the
// value from the (L2-resident) channel-LLR streams each later iteration
// as clamp(interleave(ldg,ldg)) -- BIT-IDENTICAL to the staged value.
#define LDPC2_A202_EXT_GLOBAL     1
// Tail-row C2V storage form in the freed capacity (see TAIL_EXP in
// ldpc2_c2v_cache_split_band.cuh): with the extension columns out of the
// persistent shared image, the URD-word EXPANDED per-edge form fits at
// EVERY p in the zone (26 cols * 1536 + 37 words * 1536 = 96768 B <=
// 101376 B), so the per-(row,p) prefix rationing a dual-form scheme
// would need collapses to a single always-expanded layout with
// compile-time offsets. The expanded subtract is a bare __hsub2 per edge
// (no per-edge HSET2+HFMA2+sign-reposition decompress, no stored-signs
// scan accumulation); the APP trajectory is bit-identical.
#define LDPC2_A202_TAIL_EXP       1
// 2-word pre-norm packed record for the two deep compressed rows 19/20
// (see cC2V_storage_x2_tail_packed in ldpc2_c2v_cache_split_band.cuh and the
// c2v_storage_prenorm_packed machinery in ldpc2_c2v_x2.cuh): their
// record round trip becomes ONE LDS.64 + ONE STS.64 per row per
// iteration (6 -> 2 word ops each), with the argmin index and per-edge
// signs hidden in the zero low-5 mantissa bits of the pre-normalization
// E5M5 magnitudes and a bit-identical norm-aware reconstruction. Only
// p20/p21 execute these rows.
#define LDPC2_A202_TAIL_PACKED2   1 // adopted: 2-word packed rows 19/20 (+0.5% w/ sign trees; frees 2 planes = 3KB for future strip)
// The 2-word packed tail reconstructs through the packed-argmin row context,
// so it cannot be built without it. Stated here because the failure otherwise
// surfaces as a constructor-mismatch error inside ldpc2_c2v_x2.cuh, which does
// not name the cause -- and turning E5M5 off to cost it is a likely reason to
// come here.
#if !LDPC_MICRO_PACK_ARGMIN && LDPC2_A202_TAIL_PACKED2
#error "LDPC2_A202_TAIL_PACKED2 requires LDPC_MICRO_PACK_ARGMIN. Set LDPC2_A202_TAIL_PACKED2 to 0 to build this kernel without the E5M5 packed-argmin scan."
#endif
// Descriptor-proven inter-row barrier elision (see row_boundary_sync in
// schedule_p15to21): the per-row __syncthreads() between rows IDX and
// IDX+1 is removed when the checked-in BG1 vnode tables prove the two
// rows touch DISJOINT variable-node column sets (then neither ordering
// nor visibility requires the barrier -- proof at row_boundary_sync).
// Among rows 0..20 exactly two adjacent pairs are disjoint, (16,17) and
// (17,18) -- every other pair shares punctured column 0 or 1 -- and only
// ONE may be elided (the pairs share row 17, and rows 16/18 both touch
// column 1, so chaining the elisions would let warps drift across a
// three-row window with a real column conflict). Value = the IDX whose
// trailing barrier is elided (16 or 17), or -1 to disable.
//   16: live for p >= 18 (row 17 must run) -- covers p18/p20/p21; also
//       the pair upstream row_seq_sync<1,16> elides in the generic
//       dynamic schedule.
//   17: live for p >= 19 only.
#define LDPC2_A202_ELIDE_SYNC_ROW -1 // ISOLATION: off for 1&2-only test
// ---------------------------------------------------------------------------
// Descriptor-proven APP store SINKING (a direction retained from an earlier
// base, re-measured on this merged tree): a store whose NEXT READER is k >= 2
// rows away does not need to complete before the immediately following
// barrier -- only before the barrier that precedes its next reader. Such
// stores are skipped in the row's in-place write-back and retired right
// AFTER the row's barrier instead (store_app_masked), because the two
// sides of a barrier are asymmetric for the CTA: the pre-barrier issue
// tail is SERIALIZED (the barrier waits for the slowest warp, so every
// warp's store-burst queueing delay adds to the phase), while
// post-barrier issue overlaps across warps and drains under the next
// row's compute. Values (and, without REGEN_ADDR, addresses) stay in the
// row's app[]/app_addr[] registers across one barrier; written bytes are
// identical, and every store still lands inside its iteration (the
// runtime-last row flushes BEFORE the iteration-tail barrier), so the
// per-iteration dump image and the wrap semantics are unchanged.
// Eligibility (bg1_sink_mask) is proven per column from the checked-in
// BG1 vnode tables: the column must be untouched by every row of the
// flush window -- row IDX+1, plus row 17 in the one window widened by
// the elided (16,17) barrier; row 16 itself never sinks (its trailing
// barrier is elided, so there is nothing to cross), and the wrap
// boundary is NEVER crossed (the dump kernel reads the APP image at the
// iteration-tail barrier).
//   LDPC2_A202_SINK_CAP: per-row cap on sunk words (keep_highest_bits:
//     the LAST-written eligible words, i.e. the tail of the store
//     burst). 0 disables non-core sinking. Every non-core row's sink
//     measured neutral-to-negative (motion between two
//     equally LSU-dense phases is zero-sum), so the row mask stays 0;
//     under the mixed expanded tail, tail rows (14..20) would need the
//     mask threaded through process_tail_row first (static_assert).
#define LDPC2_A202_SINK_CAP 4
#define LDPC2_A202_SINK_ROW_MASK 0x0
//   LDPC2_A202_SINK_CORE_CAP: separate cap for sinking CORE row 3's
//     store-burst tail into row 4's phase. Row 4 is the degree-3 STUB
//     row -- the one barrier interval in the zone with genuinely idle
//     LSU issue slots (the same slack that makes the (4,5) preload the
//     only measured-positive preload). Row 3's 18-word store burst is
//     the largest pre-barrier tail in the iteration (PC sampling: its
//     phase carries the biggest mio bucket of the four core rows), and
//     row 4 touches only columns {0,1}, so almost the whole tail is
//     eligible. Columns preloaded for row 5 in the same window are
//     excluded from the mask (drop_cols_in) -- a sunk store and a
//     preload of the same column in one barrier interval would race.
//     The earlier frontier on ITS base (p20) put the optimum at cap 6, with
//     caps 4, 8, 9 and 12 all worse. RE-SWEPT
//     on THIS merged tree (elision + sign build + relocation base, against
//     a no-sink reference): caps 2 and 3 tie at the optimum, with caps 1,
//     4 and 6 worse -- the optimum
//     moved DOWN from 6 to 2-3 (the merged tree's row-4 stub phase
//     already hosts the survival-strip and packed-record traffic that
//     the relocation added, so the stub absorbs fewer extra stores
//     before its own queue binds). Retained 2 (tie with 3; fewer
//     values live across the barrier).
#define LDPC2_A202_SINK_CORE_CAP 0 // ISOLATION: off for 1&2-only test
//   LDPC2_A202_SINK_CORE012_CAP: cap for sinking core rows 0..2's
//     store-burst tails past their barriers (flush lands ahead of the
//     next core row's 19-word load burst; measured a regression on the
//     earlier base -- kept 0).
#define LDPC2_A202_SINK_CORE012_CAP 0
//   LDPC2_A202_SINK_REGEN_ADDR: recompute the sunk words' shared-memory
//     addresses at the flush site (a throwaway app_addr_gen::generate --
//     pure compile-time immediates + tid, absorbed by the flush window's
//     idle int-issue slots) instead of keeping the address registers
//     live across the barrier. Halves the per-word register tax of
//     sinking: only the VALUE crosses the barrier.
#define LDPC2_A202_SINK_REGEN_ADDR 1
// ---------------------------------------------------------------------------
// Tail-row state prefetch pipeline (a mechanism retained from an earlier
// base, re-derived for the merged mixed expanded/packed tail): in warm
// iterations each tail row's thread-private C2V record -- and, for the
// survival-strip rows 15..20, its immutable extension value -- is loaded
// from inside the PRECEDING row's body (the process_row_pre hook point:
// after that row's own APP loads, before its check-node compute, where
// the LSU pipe is otherwise idle for the whole compute phase) instead of
// at first use. Thread-privacy of every record band (expanded pair bands
// and packed uint2 slots both fold threadIdx only: same-thread
// store->load) and staging-immutability of the survival planes make the
// early reads legal at ANY distance with no barrier interaction (see
// c2v_cache_split_band::tail_prefetch); values are bit-identical, only
// issue points move. Per-row policy masks (bit r = tail row r), so the
// register cost of the pipeline -- up to URD+1 = 7 words live across one
// row boundary for an expanded row, vs 3 on the earlier all-compressed
// base -- can be rationed per storage form by measurement:
//   REC_MASK: prefetch the row's C2V record (URD words expanded / 2
//     packed).
//   EXT_MASK: prefetch the row's survival-strip extension value (rows
//     15..20 only; rows 4..14 serve extensions from the register cache).
// MEASURED COMPREHENSIVELY NEGATIVE at p20 on this merged tree, against a
// reference with the pipeline off, and DISABLED. Every mask tried was
// worse: REC+EXT over rows 14..20 (7 words), REC+EXT over rows 19..20 only
// (3 words), the same with the guard off, and EXT only over rows 15..20
// (1 word/row).
// The regression scales with ANY pipeline state, spill-free at REG=168
// -- the same cliff shape as the EXT_CACHE_BOUND=16 point, which
// cost measurably for ONE extra always-live register. This prefetch
// pipeline paid off the 11-13 registers of slack the EXT_CACHE_BOUND audit
// freed; the extension-column relocation SPENT that same slack on the
// expanded tail forms (w[]/wout[] per-edge arrays in the deep-row
// bodies), so on the merged tree the two wins are mutually exclusive
// consumers of one resource -- and the relocation is the better buyer
// (1.0974 vs 1.0816 on their common base). Machinery kept
// (masks 0 = zero-cost, verified byte-identical latency) for future
// bases with restored slack.
#define LDPC2_A202_TAIL_STATE_PF    1
#define LDPC2_A202_TAIL_PF_REC_MASK 0x0
#define LDPC2_A202_TAIL_PF_EXT_MASK 0x0
//   GUARD: 1 = deeper rows' hooks carry the warp-uniform runtime
//     row-activity test (the earlier shape -- REQUIRED there because
//     its shared tail was runtime-p-sized, so an inactive row's read
//     was out of bounds); 0 = hooks issue unconditionally. On THIS
//     tree the whole tail region is allocated at its fixed p21 size,
//     so an inactive row's prefetch reads in-bounds garbage that is
//     never consumed (the schedule exits at row p-1) -- dropping the
//     guard keeps the hosting row bodies straight-line (no mid-body
//     branch) at the cost of <= 3 wasted LDS per iteration at p < 21.
#define LDPC2_A202_TAIL_PF_GUARD    1
// Runtime elision of the tail rows' dead final-iteration C2V record
// store (the record is read only by the next iteration's subtract, so
// the last iteration's store has no reader; the block-uniform
// is_last_iter predicate skips it inside the shared warm body -- no
// peeled copy, no numerics change). Retained by the extension-column
// relocation; an earlier run measured its variant NEGATIVE on the
// all-packed-tail base -- a base-sensitive knob, re-measured on this merge.
// Isolated on the tree with rows 14..18 expanded and 19/20 compressed: 1.056x
// geomean p15..21, and flat in absolute terms at every p -- far more than
// the elided stores themselves; the runtime predicate makes ptxas emit a
// 168-instruction-smaller kernel (10816 vs 10984 SASS), and the reduced
// i-footprint is the p-independent component. Timed path bit-exact vs
// knob-off (identical BER/BLER at low-SNR fingerprints).
#define LDPC2_A202_TAIL_LAST_ELIDE  1

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_box_plus_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_app_address_dp_z_imm.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_c2v_cache_split_band.cuh"
#include "ldpc2_dec_output.cuh"
#include "ldpc2_algo202.hpp"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace algo202
{
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Z zone and bounded-p configuration.
    //
    // CFG_Z_MAX is both the compile-time lifting the Z-pinned variant is
    // specialized for and the worst case every shared-memory budget here is
    // sized against; CFG_Z_MIN is the bottom of the zone the runtime-Z
    // variant serves. Registers are Z-invariant (pinned by
    // __launch_bounds__) and shared memory is monotone in Z, so a kernel
    // that fits at CFG_Z_MAX fits at every lifting below it.
    static constexpr int CFG_BG    = 1;   // BG1 only
    static constexpr int CFG_Z_MIN = 256; // bottom of the zone (runtime-Z variant)
    static constexpr int CFG_Z_MAX = 384; // top of the zone; the Z-pinned variant's lifting
    static constexpr int CFG_MIN_P = 12;  // no IS_LAST_ROW check below this
    static constexpr int CFG_MAX_P = 21;  // rows 0..20 only are instantiated

    //------------------------------------------------------------------
    // TWO COMPILED VARIANTS, selected by RT_Z (the same recipe proved on
    // bg1_z256up_bp_shwin_x2).
    //
    //   RT_Z=false -> Z = CFG_Z_MAX COMPILE-TIME IMMEDIATES. Correct ONLY at
    //                 that lifting, and what it buys is not arithmetic but an
    //                 IMMEDIATE that ptxas folds into the consuming LDS/STS
    //                 [R+UR+imm] operand for zero instructions.
    //   RT_Z=true  -> runtime descriptor loads and blockDim.x strides;
    //                 correct at every lifting in the zone.
    //
    // Only the address generator, the shared-memory strides and the entry
    // stager's byte geometry depend on RT_Z. The C2V storage plan, the
    // schedule, the staging ROUND SCHEDULE (indexed in columns, not bytes)
    // and the register plan are identical in both.
    template <bool RT_Z>
    __device__ __forceinline__
    int a202_zstride()
    {
        if constexpr(RT_Z) { return static_cast<int>(blockDim.x); }
        else               { return CFG_Z_MAX; }
    }

    // Register/shared split point inside the non-core rows (see
    // ldpc2_c2v_cache_split_band.cuh). Measured on GB203/p12: the all-register
    // split (= CFG_MAX_P) is fastest -- ptxas absorbs freed registers
    // without a latency gain, so a shared move only adds ld/st -- but the
    // knob is kept for devices/zones where register relief pays. The
    // retained win of the split cache is the extension-column register
    // cache: each non-core row's loop-invariant degree-1 extension APP is
    // captured on the peeled first iteration and reused thereafter,
    // removing one ld.shared per non-core row per later iteration
    // (bit-identical: the shared copy never changes).
    // Register/shared split: rows 4..13 keep per-row box-plus
    // register storage; rows 14..20 hold COMPRESSED (min-sum, 3-word) C2V
    // in the shared tail region (see c2v_cache_split_band TStorageTail) --
    // the 168-register budget cannot hold the deep rows (~+46 box-plus
    // words) and a box-plus shmem tail (URD words/row) exceeds the 99KB
    // budget past p=18, while the compressed tail fits through p=21.
    static constexpr int CFG_NUM_REG_PARITY = 14;

    //------------------------------------------------------------------
    // E5M5 truncation compensation for the compressed-min-sum core: the
    // packed-argmin scan truncates core-row min-sum magnitudes to 5 mantissa
    // bits (~1.5% average down-bias). Scaling the table norm by the p=4
    // validated 0.80/0.79 ratio restores the effective normalization (GB203
    // A/B: TB BLER 12.08% uncompensated vs 10.19% for the exact-fp16 box-plus
    // reference at 13.88 dB). Applied in-kernel so it is idempotent across
    // the TB/graph-capture launch paths. NOTE: only the 4 core rows are
    // truncated here while the box_plus non-core rows are exact, so the
    // global scale slightly over-corrects the non-core rows; a per-p sweep or
    // core-rows-only compensation is future tuning work.
#if LDPC_MICRO_PACK_ARGMIN
    static constexpr float E5M5_NORM_SCALE = 0.80f / 0.79f;
#endif

    //------------------------------------------------------------------
    // Hard-decision output layout for the all-warps single-pass writer.
    //
    // Its precondition is out_words_per_cw == WORDS_PER_WARP *
    // (blockDim.x/32), and that holds at EVERY lifting with Z % 32 == 0,
    // not only at 384: Kb=22 gives (22*Z+31)/32 = 22*(Z/32) output words,
    // spread as 22 words/warp x (Z/32) warps when blockDim.x == Z. So the
    // per-warp word count -- the only one of these that reaches the writer
    // as a template argument -- is Z-INVARIANT and stays compile-time:
    //
    //   CFG_OUT_WORDS_PER_WARP = (Kb*Z/32) / (Z/32) = Kb = 22, for all Z.
    //
    // The Z % 32 term of a202_z_supported() is therefore a CORRECTNESS gate
    // for this writer, not a tidiness rule, and must not be dropped if the
    // zone is ever widened downward.
    static constexpr int CFG_KB                 = ldpc2::max_info_nodes<CFG_BG>::value; // 22
    static constexpr int CFG_OUT_WORDS_PER_WARP = CFG_KB;                               // 22, every Z
    static_assert(CFG_Z_MAX % 32 == 0 && CFG_Z_MIN % 32 == 0,
                  "zone bounds must be multiples of 32 (all-warps output writer)");
    static_assert((CFG_KB * CFG_Z_MAX) % 32 == 0, "info bits must be word-aligned");
    static_assert(((CFG_KB * CFG_Z_MIN) / 32) == CFG_OUT_WORDS_PER_WARP * (CFG_Z_MIN / 32) &&
                  ((CFG_KB * CFG_Z_MAX) / 32) == CFG_OUT_WORDS_PER_WARP * (CFG_Z_MAX / 32),
                  "output words must spread evenly over warps at both ends of the zone");

    //------------------------------------------------------------------
    // Non-core C2V storage: box_plus full-width register storage (max
    // update_row_degree = 9 for the BG1 non-core rows).
    //
    // Compressed min-sum storage for these rows was measured and rejected:
    // it frees ~34 always-live registers at p14, but the per-column
    // FMA-select reconstruction on both passes costs more than the freed
    // registers buy, and it would extend E5M5 truncation to rows the
    // box_plus form keeps exact.
    static constexpr int BG1_MAX_BOX_PLUS_WORDS = 9;
    typedef c2v_storage_x2_box_plus<BG1_MAX_BOX_PLUS_WORDS> noncore_storage_t;

    //------------------------------------------------------------------
    // Core C2V storage (BG1 rows 0-3, degree 19): compressed min-sum in
    // registers (subject to the E5M5 packed-argmin truncation of the toggles
    // above). GB203-only build: the exact-fp16 box_plus-in-shared-memory core
    // (which needs ~228 KB, e.g. GH200) has been dropped -- GB203's ~100 KB
    // shmem never selected it.

    //------------------------------------------------------------------
    // Sign manager for compressed C2V row processor
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    // APP address calculation: dp_z_imm -- the dp_desc HSET2+dp2a wrap-add
    // form (the measured winner once the pairwise merge makes
    // the kernel compute-bound) with every per-edge descriptor value folded
    // to a compile-time immediate, exploiting the dispatch-pinned Z = 384.
    // Removes the three LDC/LDCU descriptor loads per edge pair per row per
    // iteration and extends the shift==0 single-IADD fast path from the
    // final edge of each row to every zero-shift edge. All calculators
    // produce bit-identical byte addresses (the immediates come from the
    // same nrLDPC_templates.cuh metafunctions that generate the checked-in
    // BG_adj_desc runtime tables).
    //
    // RT_Z swaps in the runtime descriptor form (the algo55
    // scheme, as the p=22..46 band decoders use) so one kernel covers the
    // whole zone.
    template <bool RT_Z>
    using app_loc_t = std::conditional_t<RT_Z,
                                         app_loc_address_fp_dp_desc<__half2, CFG_BG>,
                                         app_loc_address_dp_z_imm<__half2, CFG_BG, CFG_Z_MAX>>;

    //------------------------------------------------------------------
    // The base-graph descriptor type and its lookup are variant-INDEPENDENT
    // (both address classes use BG_adj_desc<BG> and the same checked-in
    // tables), so host launch paths and kernel signatures do not fork.
    typedef ldpc2::BG_adj_desc<CFG_BG> a202_bg_desc_t;

    const a202_bg_desc_t* a202_get_bg_desc(int Z)
    {
        return get_adj_BG_desc<__half2, CFG_BG>(Z);
    }

    //------------------------------------------------------------------
    // a202_z_supported()
    // The Z zone: the 38.212 liftings in [CFG_Z_MIN, CFG_Z_MAX] for which
    // the 1-CTA/SM, blockDim==Z design is sensible, all multiples of 32 as
    // the all-warps hard-output writer requires. Both compiled variants are
    // covered: dispatch routes Z == CFG_Z_MAX to the Z-pinned kernel and
    // everything else to the runtime-Z one, so this gate is the union, not
    // a per-variant test. A multiple of 32 in range that is not a legal
    // lifting has no descriptor table and is refused by the launch paths
    // when a202_get_bg_desc() returns nullptr.
    bool a202_z_supported(int Z)
    {
        return (Z >= CFG_Z_MIN) && (Z <= CFG_Z_MAX) && (0 == (Z % 32));
    }

    //------------------------------------------------------------------
    template <class TStorage> using row_context_t = cC2V_row_context<__half2,
                                                                     sign_mgr_t,
                                                                     unused,
                                                                     TStorage>;
    template <class TRowContext> using cC2V_row_proc_t = cC2V_row_proc<__half2,
                                                                       TRowContext>;

    //------------------------------------------------------------------
    // schedule_p15to21
    // Bounded runtime-p schedule with boundary-iteration elision. Like
    // ldpc_schedule_dynamic_desc_lowp it only instantiates process_row<>
    // for rows 0..CFG_MAX_P-1 (so the C2V register cache can be sized to
    // CFG_MAX_P instead of the worst-case 46), and reuses the base's
    // iter_sync_check_done<> for the per-row sync/termination semantics
    // (rows below CFG_MIN_P skip the runtime IS_LAST_ROW check entirely).
    // Additionally provides do_first_iteration(), which runs the row sweep
    // with IS_FIRST set and so elides the bit-exact-no-op C2V subtract on
    // iteration 0. The final iteration deliberately reuses do_iteration()'s
    // warm body: a peeled IS_LAST body would only elide dead REGISTER
    // writes (all C2V state is register-resident at NUM_REG_PARITY = 14)
    // while paying a once-per-CTA cold instruction-fetch of its own
    // ~1.5K-instruction copy of the row sweep, measured at ~3.8% of p5
    // kernel time on GB203 (ncu PC sampling: the once-executed body was
    // dominated by no_inst stalls). Running the final iteration through
    // the hot loop is bit-identical -- the extra C2V pack writes registers
    // no one reads -- so the APP trajectory, the per-iteration dumps, and
    // the decoded output are unchanged, and every codeword still runs the
    // full max_iterations.
    enum class iter_mode { normal, first };

    //------------------------------------------------------------------
    // no_tail_flush
    // No-op after-row hook for iterations with no pending staging data
    // (see llr_stager::after_row for the real iteration-0 hook).
    struct no_tail_flush
    {
        template <int ROW>
        __device__ __forceinline__
        void after_row()
        {
        }
    };

    // MEASURED NEGATIVE (kept as a record): splitting the per-row barrier
    // into bar.arrive(1,768) + bar.sync(1,768) so the next row's address
    // generation and preloads execute inside the arrive->wait window cost
    // 3.2% at p12 even with NO preloads -- the named
    // barrier arrive/wait pair is substantially more expensive than one
    // BAR.SYNC 0 on GB203/sm120. Do not revisit split named barriers here.

    template <class TC2VCache, class TKernelParams, bool RT_Z>
    struct schedule_p15to21 :
        ldpc_schedule_dynamic_desc_base<CFG_BG,
                                        app_loc_t<RT_Z>,
                                        TC2VCache,
                                        TKernelParams,
                                        a202_bg_desc_t,
                                        CFG_MIN_P,
                                        CFG_MAX_P>
    {
        typedef ldpc_schedule_dynamic_desc_base<CFG_BG,
                                                app_loc_t<RT_Z>,
                                                TC2VCache,
                                                TKernelParams,
                                                a202_bg_desc_t,
                                                CFG_MIN_P,
                                                CFG_MAX_P> inherited_t;
        typedef a202_bg_desc_t bg_desc_t;

        __device__
        schedule_p15to21(char*                smem,
                        const TKernelParams& params,
                        const bg_desc_t&     bg_desc,
                        int                  soffset,
                        unsigned int         t_idx) :
            inherited_t(smem, params, bg_desc, soffset, t_idx)
        {
        }

        //--------------------------------------------------------------
        // Cross-barrier APP preload masks (see bg1_preload_mask in
        // ldpc2_c2v_cache_split_band.cuh): bit i of PRE_MASK<PREV, CUR> marks
        // row CUR's i-th APP word as loadable during row PREV's phase,
        // proven from the checked-in BG1 vnode tables (row PREV never
        // writes that column, so the load is race-free and returns the
        // post-barrier value). Extension columns are never preloaded.
        // Preload is restricted to non-core destination rows (CUR >= 4):
        // preloading into the degree-19 core rows extends 6-7 word live
        // ranges across the schedule's highest register-pressure windows
        // and measured slower at p12 against a baseline with core
        // preloads enabled, and slower again when re-measured on the
        // bidiag-carry base for the rows-1..3 preload -- the
        // preloadable columns are exactly the LATE scan edges whose
        // post-barrier LDS latency is already hidden under the min-sum
        // chain, the chain-HEADING columns are shared with the preceding
        // row (not hoistable), and a preload does not reduce the LSU op
        // count that actually binds).
        // keep_lowest_bits(): restrict a preload mask to its K lowest set
        // bits (the earliest-consumed eligible words) to bound the
        // per-row register/queue cost of preloading.
        static constexpr uint32_t keep_lowest_bits(uint32_t m, int k)
        {
            uint32_t r = 0;
            for(int i = 0; i < 32; ++i)
            {
                if(((m >> i) & 1u) && (k > 0)) { r |= (1u << i); --k; }
            }
            return r;
        }
        // Preload re-verified on the composed tree (unpeel + entry overlap
        // + PAIRWISE=0 linear scan): p12 is faster with the preload on than
        // off -- the (4,5) stub-row preload still pays.
        // CAP RE-SWEPT on the merged tree (relocation + elision +
        // sign build + row-3 store sink): the row-4 stub phase now also
        // hosts the sink flush, and the register plan sits on the
        // allocation cliff, so the full preload set is no longer free --
        // at p20 the optimum is cap 3 (RETAINED), with caps 1, 2, 4, 6 and
        // 32 (all) each worse. The sink cap was re-swept jointly
        // (the two masks are coupled through drop_cols_in): at cap 3
        // preload, sink 2 is the optimum (RETAINED), with sink 0, 1 and 3
        // worse -- both knobs interact and their joint optimum is
        // (preload 3, sink 2).
        static constexpr int PRELOAD_CAP = 32; // ISOLATION: baseline cap for 1&2-only test
        // Preload enabled ONLY for the (4,5) pair: row 4 is the degree-3
        // stub row whose barrier-to-barrier phase is genuinely
        // latency-exposed (too little work to cover the sync), so row
        // 5's independent loads can fill real idle issue slots there.
        // Wider preload sets measured slower.
        template <int PREV, int CUR>
        static constexpr uint32_t PRE_MASK =
            (CUR == 5) ? keep_lowest_bits(bg1_preload_mask<PREV, CUR>::value, PRELOAD_CAP) : 0u;

        // Two-row-ahead preload of row 6 in the same row-4 stub phase:
        // columns proven untouched by BOTH rows 4 and 5 (edges 1..7 of
        // row 6, cols {6,10,11,13,17,18,20}) survive the two intervening
        // barriers, register-carried (pre_ahead6_) across row 5's phase.
        // MEASURED NEGATIVE at p12 against the (4,5)-only
        // preload -- the stub phase is already filled by row 5's five
        // preloads, and the 7 extra words live across two barriers cost
        // more than they hide. Keep OFF.
        static constexpr bool     TWO_AHEAD_46 = false;
        static constexpr uint32_t PRE2_MASK_6  = bg1_preload_mask2<4, 5, 6>::value;
        word_t pre_ahead6_[app_num_words<__half2, CFG_BG, 6>::value];

        //--------------------------------------------------------------
        // bg1_rows_disjoint: TRUE iff BG1 rows A and B touch DISJOINT
        // variable-node column sets (every edge of B misses A's column
        // set; disjointness is symmetric, so one direction proves both),
        // computed from the same checked-in 38.212 vnode tables that
        // generate the runtime BG descriptors.
        template <int A, int B, int I, int DEG>
        struct bg1_rows_disjoint_impl
        {
            static constexpr bool value =
                !bg1_row_contains<A, vnode_index<CFG_BG, B, I>::value>::value &&
                bg1_rows_disjoint_impl<A, B, I + 1, DEG>::value;
        };
        template <int A, int B, int DEG>
        struct bg1_rows_disjoint_impl<A, B, DEG, DEG>
        {
            static constexpr bool value = true;
        };
        template <int A, int B>
        struct bg1_rows_disjoint :
            bg1_rows_disjoint_impl<A, B, 0, row_degree<CFG_BG, B>::value> {};

        //--------------------------------------------------------------
        // row_boundary_sync<IDX>(): the schedule's barrier between row
        // IDX and row IDX+1, ELIDED for the descriptor-proven disjoint
        // pair selected by LDPC2_A202_ELIDE_SYNC_ROW. The barrier's two
        // jobs are (a) making row IDX's APP write-backs visible to row
        // IDX+1's loads and (b) keeping row IDX+1's write-backs from
        // racing row IDX's loads; when the two rows share no variable
        // column, neither hazard exists and the rows' load/compute
        // windows may overlap freely across warps:
        //   * APP: bg1_rows_disjoint proves no shared column, so no
        //     thread reads or writes a word the other row touches. Both
        //     rows' inputs were last written at or before row IDX-1 and
        //     are covered by the (unelided) preceding barrier; the next
        //     consumer of either row's outputs sits behind the following
        //     barrier, which every warp reaches only after executing
        //     BOTH rows.
        //   * C2V state: registers (rows 4..13) or thread-private shared
        //     SoA planes (tail rows 14..20: the expanded pair bands and
        //     the packed uint2 slots both fold threadIdx only -- thread
        //     t owns words {2t, 2t+1} of each band -- so no cross-thread
        //     access exists regardless of barriers).
        //   * Extension columns (EXT_GLOBAL layout): rows 15..20 re-read
        //     their loop-invariant extension value from the PERMANENT
        //     survival planes 33..38 every iteration. Those planes are
        //     written only by the entry stager's tail rounds, flushed in
        //     the iteration-0 after_row<12>/after_row<13> hooks --
        //     strictly ahead of the (unelided) row-13 barrier -- and no
        //     C2V store ever touches them, so a drifted read across the
        //     elided boundary always returns the staged value. Rows
        //     4..14 serve extensions from the register cache after
        //     iteration 0 (no shared access at all).
        // Elisions must not chain: removing the barriers after BOTH rows
        // IDX and IDX+1 lets warps drift across a three-row window,
        // which additionally requires (IDX, IDX+2) disjointness -- false
        // for the only candidates here (rows 16 and 18 share column 1).
        // The static_asserts pin both proofs to the checked-in tables.
        // When row IDX+1 is the runtime-last row, the elision site is
        // never reached (the !IS_LAST_ROW branch guards it) and the
        // iteration-tail barrier covers the wrap into row 0 -- which is
        // never disjoint in this zone (the last row shares columns with
        // row 0 at every p in 15..21).
        template <int IDX>
        static constexpr bool ELIDE_SYNC_AFTER = (IDX == LDPC2_A202_ELIDE_SYNC_ROW);

        template <int IDX>
        __device__ __forceinline__
        void row_boundary_sync()
        {
            static_assert((!ELIDE_SYNC_AFTER<IDX>) || bg1_rows_disjoint<IDX, IDX + 1>::value,
                          "barrier elision requires descriptor-proven column disjointness of rows (IDX, IDX+1)");
            static_assert(!(ELIDE_SYNC_AFTER<IDX> && ELIDE_SYNC_AFTER<IDX + 1>),
                          "chained elisions widen the overlap window to three rows; (IDX, IDX+2) disjointness is not established");
            if constexpr(!ELIDE_SYNC_AFTER<IDX>)
            {
                __syncthreads();
            }
        }

        //--------------------------------------------------------------
        // Store-sinking masks (LDPC2_A202_SINK_CAP, see the toggle
        // comment). keep_highest_bits(): restrict a mask to its K
        // HIGHEST set bits -- the last-written eligible words, i.e. the
        // tail of the row's store burst, which is what delays barrier
        // arrival.
        static constexpr uint32_t keep_highest_bits(uint32_t m, int k)
        {
            uint32_t r = 0;
            for(int i = 31; i >= 0; --i)
            {
                if(((m >> i) & 1u) && (k > 0)) { r |= (1u << i); --k; }
            }
            return r;
        }
        // bg1_sink_mask: bit i (i < URD of ROW) is set iff row ROW's
        // i-th column is touched by NEITHER row X1 NOR row X2 (the rows
        // of the post-barrier flush window), proven from the checked-in
        // BG1 vnode tables. Pass X2 == X1 for a one-row window.
        template <int ROW, int X1, int X2, int I, int DEG>
        struct bg1_sink_mask_impl
        {
            static constexpr int  COL      = vnode_index<CFG_BG, ROW, I>::value;
            static constexpr bool ELIGIBLE = !bg1_row_contains<X1, COL>::value &&
                                             !bg1_row_contains<X2, COL>::value;
            static constexpr uint32_t value = (ELIGIBLE ? (1u << I) : 0u) |
                                              bg1_sink_mask_impl<ROW, X1, X2, I + 1, DEG>::value;
        };
        template <int ROW, int X1, int X2, int DEG>
        struct bg1_sink_mask_impl<ROW, X1, X2, DEG, DEG>
        {
            static constexpr uint32_t value = 0u;
        };
        template <int ROW, int X1, int X2>
        struct bg1_sink_mask :
            bg1_sink_mask_impl<ROW, X1, X2, 0, update_row_degree<CFG_BG, ROW>::value> {};

        // cols_of_mask: translate an edge-index mask of ROW into a
        // variable-node column bitmap (uint64_t; BG1 columns < 46), for
        // cross-row overlap static_asserts.
        template <int ROW, uint32_t MASK, int I, int DEG>
        struct cols_of_mask_impl
        {
            static constexpr uint64_t value =
                ((((MASK >> I) & 1u) != 0u) ? (1ull << vnode_index<CFG_BG, ROW, I>::value) : 0ull) |
                cols_of_mask_impl<ROW, MASK, I + 1, DEG>::value;
        };
        template <int ROW, uint32_t MASK, int DEG>
        struct cols_of_mask_impl<ROW, MASK, DEG, DEG>
        {
            static constexpr uint64_t value = 0ull;
        };
        template <int ROW, uint32_t MASK>
        struct cols_of_mask :
            cols_of_mask_impl<ROW, MASK, 0, row_degree<CFG_BG, ROW>::value> {};

        // drop_cols_in: clear the bits of MASK (edge indices of ROW)
        // whose variable-node column appears in the EXCL column bitmap.
        template <int ROW, uint32_t MASK, uint64_t EXCL, int I, int DEG>
        struct drop_cols_in_impl
        {
            static constexpr uint32_t BIT =
                ((((MASK >> I) & 1u) != 0u) &&
                 (0ull == ((EXCL >> vnode_index<CFG_BG, ROW, I>::value) & 1ull)))
                    ? (1u << I) : 0u;
            static constexpr uint32_t value = BIT | drop_cols_in_impl<ROW, MASK, EXCL, I + 1, DEG>::value;
        };
        template <int ROW, uint32_t MASK, uint64_t EXCL, int DEG>
        struct drop_cols_in_impl<ROW, MASK, EXCL, DEG, DEG>
        {
            static constexpr uint32_t value = 0u;
        };
        template <int ROW, uint32_t MASK, uint64_t EXCL>
        struct drop_cols_in :
            drop_cols_in_impl<ROW, MASK, EXCL, 0, row_degree<CFG_BG, ROW>::value> {};

        // SINK_MASK<IDX>: the words of row IDX whose in-place write-back
        // is skipped and retired after row IDX's barrier. Zero for row
        // 16 (its trailing barrier is elided -- no barrier to cross),
        // for the compile-time-last row 20 (no in-iteration flush
        // window; the wrap is never crossed), and for the tail rows
        // unless threaded through process_tail_row (static_assert in the
        // cache). A row whose flush window is widened by the elided
        // (16,17) barrier excludes BOTH rows' columns.
        template <int IDX>
        static constexpr uint32_t SINK_MASK_of()
        {
            if constexpr((IDX >= 0) && (IDX < 3) && (LDPC2_A202_SINK_CORE012_CAP > 0))
            {
                // Core rows 0..2 -> the next core row's phase. The
                // preload set of the (IDX+1, IDX+2) pair is empty for
                // these windows (core rows are never preloaded), so no
                // drop is needed; keep the dropper for symmetry.
                return keep_highest_bits(
                    drop_cols_in<IDX,
                                 bg1_sink_mask<IDX, IDX + 1, IDX + 1>::value,
                                 cols_of_mask<IDX + 2, PRE_MASK<IDX + 1, IDX + 2>>::value>::value,
                    LDPC2_A202_SINK_CORE012_CAP);
            }
            else if constexpr((IDX == 3) && (LDPC2_A202_SINK_CORE_CAP > 0))
            {
                // Core row 3 -> the row-4 stub phase (see the
                // LDPC2_A202_SINK_CORE_CAP comment). Columns preloaded
                // for row 5 in the same interval are dropped.
                return keep_highest_bits(
                    drop_cols_in<3,
                                 bg1_sink_mask<3, 4, 4>::value,
                                 cols_of_mask<5, PRE_MASK<4, 5>>::value>::value,
                    LDPC2_A202_SINK_CORE_CAP);
            }
            else if constexpr((LDPC2_A202_SINK_CAP <= 0) || (IDX < 4) ||
                         (IDX >= CFG_NUM_REG_PARITY) ||
                         (0 == ((LDPC2_A202_SINK_ROW_MASK >> IDX) & 1)) ||
                         (IDX == LDPC2_A202_ELIDE_SYNC_ROW) || ((IDX + 1) >= CFG_MAX_P))
            {
                return 0u;
            }
            else if constexpr((IDX + 1) == LDPC2_A202_ELIDE_SYNC_ROW)
            {
                static_assert(bg1_rows_disjoint<LDPC2_A202_ELIDE_SYNC_ROW,
                                                LDPC2_A202_ELIDE_SYNC_ROW + 1>::value,
                              "widened flush window exists because the (16,17) barrier is elided");
                return keep_highest_bits(bg1_sink_mask<IDX, IDX + 1, IDX + 2>::value,
                                         LDPC2_A202_SINK_CAP);
            }
            else
            {
                return keep_highest_bits(bg1_sink_mask<IDX, IDX + 1, IDX + 1>::value,
                                         LDPC2_A202_SINK_CAP);
            }
        }
        template <int IDX>
        static constexpr uint32_t SINK_MASK = SINK_MASK_of<IDX>();

        //--------------------------------------------------------------
        // sink_flush(): retire row IDX's sunk stores. Called right after
        // row_boundary_sync<IDX>() (the flush window whose rows the mask
        // excludes), or -- for the runtime-last row -- before the
        // iteration-tail barrier so the per-iteration APP image is
        // complete at the dump/wrap point.
        template <int IDX, int RD, int NW>
        __device__ __forceinline__
        void sink_flush(int (&app_addr)[RD], word_t (&app)[NW])
        {
            if constexpr(SINK_MASK<IDX> != 0u)
            {
                // No column may be both sunk into interval IDX+1 (from
                // row IDX) and preloaded in interval IDX+1 (for row
                // IDX+2): the two would race inside one barrier window.
                if constexpr((IDX + 2) < CFG_MAX_P)
                {
                    static_assert(0ull == (cols_of_mask<IDX, SINK_MASK<IDX>>::value &
                                           cols_of_mask<IDX + 2, PRE_MASK<IDX + 1, IDX + 2>>::value),
                                  "store-sink and preload column sets collide in one barrier interval");
                }
                if constexpr(LDPC2_A202_SINK_REGEN_ADDR != 0)
                {
                    // Recompute the sunk words' addresses here (throwaway
                    // array; identical values -- generate<> is a pure
                    // function of compile-time immediates and tid) so the
                    // address registers do not cross the barrier.
                    int taddr[RD];
                    (*this).app_addr_gen.template generate<IDX>(taddr);
                    (*this).c2v_cache.template store_app_masked<IDX, SINK_MASK<IDX>>(app,
                                                                                     taddr,
                                                                                     (*this).smem_offset);
                }
                else
                {
                    (*this).c2v_cache.template store_app_masked<IDX, SINK_MASK<IDX>>(app,
                                                                                     app_addr,
                                                                                     (*this).smem_offset);
                }
            }
        }

        //--------------------------------------------------------------
        // Tail-state prefetch policy plumbing (see the
        // LDPC2_A202_TAIL_STATE_PF toggle comment): row CUR's state is
        // pipelined iff any policy bit selects it; the same masks gate
        // the producing hook (below) and the consuming row body, so
        // producers and consumers pair one-for-one.
        template <int CUR>
        static constexpr bool TAIL_PF_ANY =
            (LDPC2_A202_TAIL_STATE_PF != 0) &&
            (CUR >= CFG_NUM_REG_PARITY) && (CUR < CFG_MAX_P) &&
            (0 != (((LDPC2_A202_TAIL_PF_REC_MASK | LDPC2_A202_TAIL_PF_EXT_MASK) >> CUR) & 1u));

        //--------------------------------------------------------------
        // tail_state_prefetch: warm-iteration prefetch hook for the
        // tail rows (LDPC2_A202_TAIL_STATE_PF) -- issues row CUR's
        // thread-private C2V record (and, for survival-strip rows, its
        // immutable extension value) from inside row CUR-1's body, one
        // full row of compute plus the row boundary ahead of use (see
        // c2v_cache_split_band::tail_prefetch for the any-distance
        // legality argument). rows below CFG_MIN_P execute at every p in
        // the zone, so their prefetch is unconditional; row 14 and deeper
        // (which at CFG_MIN_P = 12 includes row 14 itself) carry the
        // warp-uniform runtime row-activity guard
        // -- not for bounds (the shared allocation is fixed p21-size)
        // but to skip the wasted LSU traffic when the consuming row
        // does not execute (the schedule exits at row p-1).
        template <int CUR>
        struct tail_state_prefetch
        {
            schedule_p15to21& sched;
            __device__ __forceinline__
            void operator()()
            {
                static_assert(CUR >= CFG_NUM_REG_PARITY && CUR < CFG_MAX_P,
                              "tail-state prefetch targets the tail rows only");
                if constexpr((CUR < CFG_MIN_P) || (LDPC2_A202_TAIL_PF_GUARD == 0))
                {
                    // Unguarded: the fixed p21-size tail allocation makes
                    // an inactive row's read in-bounds (and its value is
                    // never consumed -- the schedule exits at row p-1),
                    // so the hosting body stays straight-line.
                    sched.c2v_cache.template tail_prefetch<CUR>();
                }
                else
                {
                    if(CUR < sched.params.num_parity_nodes)
                    {
                        sched.c2v_cache.template tail_prefetch<CUR>();
                    }
                }
            }
        };

        //--------------------------------------------------------------
        // row_body(): one row's load/compute/store with the APP
        // addresses AND the preload-eligible APP words supplied by the
        // caller. The PRE mask says which app words arrived preloaded
        // (issued from the PREVIOUS row's prefetch hook); the remaining
        // words are loaded here, then the prefetch hook (the NEXT row's
        // address generation + independent preloads) runs before the
        // check-node compute. Identical values and check-node math to
        // the base schedule's process_row<> -- only the load issue order
        // differs.
        template <int IDX, iter_mode MODE, uint32_t PRE, class TPrefetch, int RD, int NW>
        __device__ __forceinline__
        void row_body(int (&app_addr)[RD], word_t (&app)[NW], TPrefetch& prefetch,
                      bool is_last_iter)
        {
            static_assert(RD == row_degree<CFG_BG, IDX>::value, "address array size");
            static_assert(NW == app_num_words<__half2, CFG_BG, IDX>::value, "app array size");
            // Warm-iteration tail rows consume their prefetched state
            // (record + extension value) from the pipeline the PRECEDING
            // row's hook filled (see LDPC2_A202_TAIL_STATE_PF).
            constexpr bool TAIL_PF = (MODE != iter_mode::first) && TAIL_PF_ANY<IDX>;
            (*this).c2v_cache.template process_row_pre<IDX,
                                                       PRE,
                                                       MODE == iter_mode::first,
                                                       false,
                                                       SINK_MASK<IDX>,
                                                       TAIL_PF>((*this).params,
                                                                                app,
                                                                                app_addr,
                                                                                (*this).smem_offset,
                                                                                prefetch,
                                                                                is_last_iter);
        }

        //--------------------------------------------------------------
        // no_prefetch: no-op hook for rows with no following row to
        // prefetch (compile-time last row, and row 0's full load).
        struct no_prefetch
        {
            __device__ __forceinline__ void operator()() {}
        };

        //--------------------------------------------------------------
        // do_rows_from(): recursively unroll rows IDX..CFG_MAX_P-1.
        // Row IDX's addresses and preloaded APP words arrive from the
        // caller (produced before the barrier that precedes row IDX --
        // address generation reads only compile-time immediates and
        // thread id, and the preloaded APP columns are proven untouched
        // by the preceding row from the checked-in BG1 descriptors, so
        // hoisting both across the barrier is race-free and bit-exact).
        // The runtime last row (and compile-time max row) RETURNS
        // without a barrier; the single iteration-tail site in do_rows()
        // refills row 0's addresses for the next iteration and issues
        // the boundary barrier, preserving the base schedule's semantics
        // (one barrier after every active row, no row work for rows
        // >= p) with one static tail instead of one per exit depth.
        // The TFlush hook runs after row_body<IDX> and BEFORE the next
        // row's preload -- on the first iteration it flushes deferred
        // entry-staging rounds (columns >= 26 only, which no preload
        // mask covers) so their stores become CTA-visible strictly
        // ahead of their first consumer row.
        template <int IDX, iter_mode MODE, uint32_t PRE, class TFlush, int RD, int NW>
        __device__
        void do_rows_from(int (&app_addr)[RD], word_t (&app)[NW], TFlush& flush,
                          bool is_last_iter)
        {
            if constexpr((IDX + 1) < CFG_MAX_P)
            {
                constexpr uint32_t NPRE = (TWO_AHEAD_46 && (IDX + 1) == 6) ? PRE2_MASK_6
                                                                           : PRE_MASK<IDX, IDX + 1>;
                int    naddr[row_degree<CFG_BG, IDX + 1>::value];
                word_t napp [app_num_words<__half2, CFG_BG, IDX + 1>::value];
                // Post-store placement (measured best of the preload
                // placements): row IDX+1's address generation and
                // independent APP preloads issue in the shadow of row
                // IDX's store burst, just before the barrier.
                // Warm iterations: rows 13..19 host the NEXT tail row's
                // state prefetch inside their body (issue distance one
                // full row; see LDPC2_A202_TAIL_STATE_PF). Iteration 0
                // keeps the no-op hook: tail rows take the IS_FIRST path,
                // which has no previous-iteration record to read.
                if constexpr((MODE != iter_mode::first) && TAIL_PF_ANY<IDX + 1>)
                {
                    tail_state_prefetch<IDX + 1> pf{*this};
                    row_body<IDX, MODE, PRE>(app_addr, app, pf, is_last_iter);
                }
                else
                {
                    no_prefetch np;
                    row_body<IDX, MODE, PRE>(app_addr, app, np, is_last_iter);
                }
                flush.template after_row<IDX>();
                if constexpr((IDX + 1) < CFG_MIN_P)
                {
                    // Row IDX+1 always active (p >= CFG_MIN_P): no
                    // IS_LAST_ROW check (as before).
                    (*this).app_addr_gen.template generate<IDX + 1>(naddr);
                    (*this).c2v_cache.template load_app_masked<IDX + 1, NPRE>(napp, naddr, (*this).smem_offset);
                    row_boundary_sync<IDX>();
                    sink_flush<IDX>(app_addr, app);
                    do_rows_from<IDX + 1, MODE, NPRE>(naddr, napp, flush, is_last_iter);
                }
                else
                {
                    const bool IS_LAST_ROW = ((IDX + 1) == (*this).params.num_parity_nodes);
                    if(!IS_LAST_ROW)
                    {
                        (*this).app_addr_gen.template generate<IDX + 1>(naddr);
                        if constexpr(TWO_AHEAD_46 && (IDX + 1) == 5)
                        {
                            // Row-4 stub phase: also issue row 6's
                            // two-row-ahead loads (columns untouched by
                            // rows 4 and 5). Wasted-but-safe when p <= 6.
                            int naddr6[row_degree<CFG_BG, 6>::value];
                            (*this).app_addr_gen.template generate<6>(naddr6);
                            (*this).c2v_cache.template load_app_masked<6, PRE2_MASK_6>((*this).pre_ahead6_, naddr6, (*this).smem_offset);
                        }
                        if constexpr(TWO_AHEAD_46 && (IDX + 1) == 6)
                        {
                            // Row 6's preloaded words arrived two phases
                            // ago (register-carried across row 5).
                            #pragma unroll
                            for(int i = 0; i < (int)(sizeof(napp) / sizeof(napp[0])); ++i)
                            {
                                if((NPRE >> i) & 1u) { napp[i] = (*this).pre_ahead6_[i]; }
                            }
                        }
                        else
                        {
                            (*this).c2v_cache.template load_app_masked<IDX + 1, NPRE>(napp, naddr, (*this).smem_offset);
                        }
                        row_boundary_sync<IDX>();
                        sink_flush<IDX>(app_addr, app);
                        do_rows_from<IDX + 1, MODE, NPRE>(naddr, napp, flush, is_last_iter);
                    }
                    else
                    {
                        // Runtime-last row: retire the sunk stores here,
                        // BEFORE the iteration-tail barrier, so the
                        // per-iteration APP image is complete at the
                        // dump/wrap point. (Zero for every sinkable row
                        // in the current config -- row 3 is never the
                        // runtime-last row -- kept for mask sweeps.)
                        sink_flush<IDX>(app_addr, app);
                    }
                    // IS_LAST_ROW: fall through to do_rows()'s tail.
                }
            }
            else
            {
                // Compile-time last row (IDX + 1 == CFG_MAX_P): nothing
                // to prefetch; fall through to do_rows()'s tail.
                // (SINK_MASK<CFG_MAX_P-1> is 0 by construction; the
                // sink_flush is a structural no-op kept for symmetry.)
                no_prefetch np;
                row_body<IDX, MODE, PRE>(app_addr, app, np, is_last_iter);
                flush.template after_row<IDX>();
                sink_flush<IDX>(app_addr, app);
            }
        }

        //--------------------------------------------------------------
        // do_rows(): one iteration. addr0/app0 carry row 0's addresses
        // and (preloaded) APP words: addresses generated by gen_row0()
        // once before the first iteration and refilled here at the
        // iteration tail BEFORE the boundary barrier, so the largest
        // (degree-19) generation interleaves with the last row's store
        // burst instead of heading the next iteration's post-barrier
        // critical path. The refill after the final iteration is dead but
        // cheap (compile-time-immediate address math once per codeword);
        // keeping it unconditional keeps the final iteration on this same
        // warm body instead of a once-executed (i-fetch-cold) peeled copy.
        // Row 0 itself is not preloaded (PRE == 0): the runtime identity
        // of the last row makes its independence set p-dependent.
        template <int RD0>
        __device__ __forceinline__
        void gen_row0(int (&addr0)[RD0])
        {
            static_assert(RD0 == row_degree<CFG_BG, 0>::value, "row 0 address array size");
            (*this).app_addr_gen.template generate<0>(addr0);
        }
        template <iter_mode MODE, class TFlush, int RD0, int NW0>
        __device__
        void do_rows(int (&addr0)[RD0], word_t (&app0)[NW0], TFlush& flush,
                     bool is_last_iter)
        {
            do_rows_from<0, MODE, 0u>(addr0, app0, flush, is_last_iter);
            gen_row0(addr0);
            __syncthreads();
        }
        // is_last_iter marks the FINAL BP iteration (a block-uniform
        // runtime value): the tail rows' shared C2V record store is dead
        // there (read only by the next iteration's subtract) and is
        // elided at runtime inside the shared warm body -- no peeled
        // last-iteration copy of the row sweep (see the i-fetch note
        // above), no numerics change (the elided stores have no reader,
        // so the APP trajectory, dumps, and outputs are unchanged).
        template <int RD0, int NW0>
        __device__ void do_iteration(int (&addr0)[RD0], word_t (&app0)[NW0],
                                     bool is_last_iter)
        {
            no_tail_flush nf;
            do_rows<iter_mode::normal>(addr0, app0, nf, is_last_iter);
        }
        template <int RD0, int NW0>
        __device__ void do_first_iteration(int (&addr0)[RD0], word_t (&app0)[NW0],
                                           bool is_last_iter)
        {
            no_tail_flush nf;
            do_rows<iter_mode::first>(addr0, app0, nf, is_last_iter);
        }
        template <class TFlush, int RD0, int NW0>
        __device__ void do_first_iteration(int (&addr0)[RD0], word_t (&app0)[NW0], TFlush& flush,
                                           bool is_last_iter)
        {
            do_rows<iter_mode::first>(addr0, app0, flush, is_last_iter);
        }
    };

    //------------------------------------------------------------------
    // Kernel configuration (GB203-only: compressed-min-sum core in registers;
    // the box_plus-in-shmem core has been dropped). Non-core rows always use
    // register box_plus.
    template <class TKernelParams, bool RT_Z>
    struct algo202_kernel_config
    {
        typedef TKernelParams kernel_params_t;

        template <int   BG,
                  int   CHECK_IDX,
                  class TC2VStorage> using cC2V_row_map_t = hybrid_storage_row_map_x2<BG,
                                                                                      CHECK_IDX,
                                                                                      TC2VStorage,
                                                                                      __half2,
                                                                                      row_context_t,
                                                                                      cC2V_row_proc_t>;
        typedef C2V_row_proc<__half2,
                             CFG_BG,
                             cC2V_row_map_t,
                             app_loader,
                             app_writer> C2V_t;

        // C2V message cache (specific to this storage split): non-core rows
        // below CFG_NUM_REG_PARITY use per-row-sized register box_plus storage
        // (trailing rows would fall to a per-row-sized shared SoA region),
        // plus the loop-invariant extension-column register cache. The
        // core rows use compressed min-sum storage in registers.
        // EXT_CACHE_BOUND = 15: the extension-column register cache
        // covers register rows 4..13 plus compressed-tail row 14. The
        // runtime-conditional tail rows 15..20 re-load their
        // loop-invariant extension APP from shared each iteration instead
        // of pinning 6 more always-live registers on the static plan's
        // peak. Measured frontier on GB203 (p20 latency): bound 21 (= cache
        // all, the original) is the frozen baseline and carries a 2-word
        // warm-loop spill; bound 14 (register rows only) improves on it;
        // bound 15 (this setting) is the optimum, spill-free, with a
        // warm-loop liveness peak of 157/168; bound 16 re-crosses the
        // pressure cliff and is worse than both
        // -- one more always-live register costs far more than the one
        // conditional LDS it saves.
        // It is stated as a LITERAL rather than as CFG_MIN_P (which it
        // equalled while the band started at 15) for two reasons: the cache
        // structurally requires EXT_CACHE_BOUND >= NUM_REG_C2V_NODES = 14,
        // so it cannot follow CFG_MIN_P down to 12; and pinning it at the
        // measured optimum keeps the p=15..21 register plan unchanged, so
        // the cost of starting the band at p=12 is confined to the
        // schedule's conditional rows.
        typedef ldpc2::c2v_cache_split_band<CFG_BG,
                                            CFG_Z_MAX,
                                            CFG_NUM_REG_PARITY,
                                            CFG_MAX_P,
                                            C2V_t,
                                            typename core_storage_x2<CFG_BG>::type,
                                            noncore_storage_t,
                                            kernel_params_t,
                                            ldpc2::cC2V_storage_x2_low_degree,
                                            15, // EXT_CACHE_BOUND (see above; NOT CFG_MIN_P)
                                            (LDPC2_A202_EXT_GLOBAL != 0),
                                            (LDPC2_A202_TAIL_EXP != 0),
                                            RT_Z,
                                            CFG_Z_MIN> c2v_cache_t;

        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        typedef llr_loader_t::app_buf_t                                    app_buf_t;

        typedef schedule_p15to21<c2v_cache_t, kernel_params_t, RT_Z> sched_t;
    };

    //------------------------------------------------------------------
    // get_app_c2v_shmem()
    // APP buffer (26 SHARED-resident columns only -- the extension
    // columns live in global memory, see LDPC2_A202_EXT_GLOBAL) + the
    // expanded shared TAIL region, ALWAYS allocated at its p21 size (37
    // words per thread): the region doubles as the iteration-0
    // extension staging strip (bg1_a202_ext_stage_plane needs planes
    // 17..33 at every p), and the kernel is register-limited to 1
    // CTA/SM (168 regs x 384 threads = 64K) at every p anyway, so the
    // fixed 96768 B allocation costs no occupancy. The compressed-
    // min-sum core storage (16 bytes) and the row-4..13 box-plus C2V
    // live in registers.
    CUDA_BOTH
    int get_app_c2v_shmem(int num_parity_nodes, int Z)
    {
        const int32_t NUM_SHM_VAR_NODES = ldpc2::max_info_nodes<CFG_BG>::value + 4;
        int offset = static_cast<int>(shmem_llr_buffer_size(NUM_SHM_VAR_NODES, Z, sizeof(__half2)));
        offset += ldpc2::bg1_a202_tail_total_words() *
                  Z * static_cast<int>(sizeof(ldpc2::word_t));
        return offset;
    }

    //------------------------------------------------------------------
    // Shared-memory budget proof: the fixed layout must fit GB203's
    // 99 KiB (101376 B) opt-in limit. The mixed expanded/compressed tail
    // plus the survival strip fits only BECAUSE the extension columns
    // were relocated out of the persistent APP image
    // (26*1536 + 39*1536 = 99840 B).
    // Stated at CFG_Z_MAX, which is the worst case: this layout is
    // (26 + tail_words) * Z * 4, monotone increasing in Z, so
    // fitting at the top of the zone means fitting at every lifting below.
    // The budget has only 1,536 B of slack at Z=384 -- four kernels in this
    // family sit AT the 101,376 B limit and the failure mode when one is
    // pushed over is invisible (no warning, just a decoder that quietly
    // stops being selectable), so this guard is what makes the next byte
    // added fail at compile time instead.
    static_assert((ldpc2::max_info_nodes<CFG_BG>::value + 4) * CFG_Z_MAX * 4 +
                  ldpc2::bg1_a202_tail_total_words() * CFG_Z_MAX * 4 <= 99 * 1024,
                  "fixed mixed-tail layout exceeds GB203 opt-in shmem -- this band "
                  "will silently stop being selectable at every p");

    //------------------------------------------------------------------
    // get_shmem_required()
    // No shared tb_token slot: the TB_SCAN entry path computes its
    // transport-block token entirely in registers
    // (find_block_tb_token_allwarp / the single-TB fast path in
    // llr_stager::issue), never through the generic loader's shared
    // token word.
    int get_shmem_required(int num_parity_nodes, int Z)
    {
        return get_app_c2v_shmem(num_parity_nodes, Z);
    }

#if LDPC_DECODE_USE_TB_SCAN

    //------------------------------------------------------------------
    // find_block_tb_token_allwarp()
    // Barrier-free transport-block token search. The generic
    // llr_loader_t::load_sync_token computes the token in warp 0 only,
    // stores it to shared memory, and forces a CTA __syncthreads() so the
    // other 11 warps can read it. Instead, EVERY warp runs the identical
    // warp-level prefix scan (the inputs -- decodeDesc.llr_input[] and
    // decode_index -- are warp-uniform, so every warp gets the same
    // answer) and __shfl-broadcasts the token from the owning lane. This
    // removes one CTA barrier and the shared-memory round trip; the token
    // is bit-identical to warp_find_block_tb_token<CW_PER_CTA>'s.
    template <int CW_PER_CTA>
    __device__ __forceinline__
    tb_token find_block_tb_token_allwarp(const cuphyLDPCDecodeDesc_t& decode_desc,
                                         unsigned int                 decode_index)
    {
        static_assert(CUPHY_LDPC_DECODE_DESC_MAX_TB <= 32,
                      "CUPHY_LDPC_DECODE_DESC_MAX_TB must be <= warp size");
        // Single-TB fast path: when the decode descriptor carries exactly
        // one transport block, the owning TB is 0, blocks_sum is 0, and
        // the token reduces to (tb=0, offset=decode_index*CW_PER_CTA,
        // partial). Emitting it directly drops the scan/ballot/ffs/shfl
        // chain that gates src_gmem (and therefore the cold staging
        // loads) off the prologue critical path, with a bit-identical
        // token. Multi-TB still takes the proven warp scan below.
        if(decode_desc.num_tbs == 1)
        {
            const int          num_cw  = decode_desc.llr_input[0].num_codewords;
            const unsigned int offset  = decode_index * CW_PER_CTA;
            const bool         partial = ((offset + CW_PER_CTA) > static_cast<unsigned int>(num_cw));
            return to_token<CW_PER_CTA>(0, offset, partial);
        }
        const int    LANEID           = threadIdx.x & 0x1F;
        int          entry_num_cw     = (LANEID < decode_desc.num_tbs)
                                            ? decode_desc.llr_input[LANEID].num_codewords
                                            : 0;
        int          entry_num_blocks = (entry_num_cw + CW_PER_CTA - 1) / CW_PER_CTA;
        int          blocks_sum       = warp_exclusive_scan<int>(entry_num_blocks);
        unsigned int sum_gt           = __ballot_sync(0xFFFFFFFFu, blocks_sum > decode_index);
        unsigned int tb               = sum_gt ? (__ffs(sum_gt) - 2) : 31;
        unsigned int offset           = (decode_index - blocks_sum) * CW_PER_CTA;
        bool         partial          = ((offset + CW_PER_CTA) > entry_num_cw);
        tb_token     tok              = to_token<CW_PER_CTA>(tb, offset, partial);
        // The owning lane 'tb' read the correct TB; broadcast its token to
        // all lanes. Every warp runs this identically, so no CTA barrier
        // is needed.
        return __shfl_sync(0xFFFFFFFFu, tok, tb);
    }

    //------------------------------------------------------------------
    // llr_stager
    // Entry staging for this specialization zone
    // (BG1 / Z == blockDim.x in [CFG_Z_MIN, CFG_Z_MAX] / p in
    // [CFG_MIN_P, CFG_MAX_P]),
    // replacing the generic runtime-bound batched copy in
    // llr_loader_variable_batch<__half2>::load_sync_token. Numerically it
    // stages the identical clamped/interleaved fp16x2 APP image; only the
    // schedule of the copy changes:
    //   * barrier-free all-warp token search (see above) -- no idle wait
    //     while a single warp scans the TB descriptor;
    //   * the copy is fully unrolled to the compile-time round count for
    //     36 variable nodes, and every global load is issued before the
    //     first interleave/clamp/store consumes any of them, so all rounds
    //     overlap in the memory system instead of serializing one
    //     4-round batch (one L2/DRAM round trip) at a time;
    //   * the staging is split into issue() / complete() so the caller
    //     can construct the schedule (C2V register-cache init, descriptor
    //     reads, APP address setup) and the output parameter blocks --
    //     work with no dependency on the staged APP bytes -- between the
    //     two, i.e. inside the global-load-latency shadow instead of
    //     serialized behind the staging barrier;
    //   * the 6 rounds covering transmitted columns 2..25 (present for
    //     every p >= CFG_MIN_P) are unconditional, but they are PERMUTED
    //     (see the permutation note in the struct) so the head barrier
    //     waits only on the 5 rounds row 0 touches while the sixth --
    //     the four columns first consumed by rows 1..2 -- defers its
    //     store into the iteration-0 after_row<0> hook; only the last 3
    //     rounds carry the runtime-p bounds predicate, and on the store
    //     side that predicate guards only the STS itself (the
    //     interleave/clamp of an inactive round computes on unconsumed
    //     registers instead of paying a branch/reconvergence block);
    //   * BG1 punctured systematic columns 0,1 (2Z LLRs the 38.212 rate
    //     matcher never transmits; the derate matcher materializes them
    //     as zero LLRs) are written as zeros directly instead of being
    //     loaded from global memory -- bit-identical to clamp(0);
    //   * a lone-codeword CTA (partial token) aliases the cw1 source to
    //     cw0 with stride 0 instead of guarding every cw1 load: the .y
    //     lanes of a partial CTA are never emitted by the output/dump
    //     writers, so their staged content is as immaterial as the
    //     uninitialized registers the generic loader interleaved.
    template <bool RT_Z>
    struct llr_stager
    {
        typedef uint2 ldg_t; // 4 fp16 per codeword per LDG (LDG.64)
        typedef uint4 sts_t; // 4 fp16x2 per STS (STS.128)

        //--------------------------------------------------------------
        // Geometry: Z threads, 4 fp16 per codeword per thread per round.
        //
        // THE ROUND SCHEDULE IS Z-INVARIANT, which is what makes this
        // loader parameterizable rather than in need of redesign. The unit
        // of work is a COLUMN, not a byte: ldg_t is 4 halves and the CTA
        // has blockDim.x == Z threads, so one round moves 4*Z halves =
        // EXACTLY 4 COLUMNS at every Z. Everything that encodes the
        // schedule -- the round counts, the GATHER_A/GATHER_B permutation,
        // the aff_q<> bases, the EXT_STAGE_SHIFT_* column shifts and the
        // after_row flush rows -- is indexed in columns and does not move
        // with Z. Only BYTE STRIDES change, and every one of them is a
        // shift of blockDim.x, hence rematerializable and never live
        // across the decode loop (the runtime-Z result, applied to a loader).
        //
        // Per-thread ownership generalizes the same way: thread quarter
        // t/(Z/4) owns one column of the round and lanes t%(Z/4) its z
        // groups (96 = Z/4 at Z=384).
        static constexpr int PUNCT_COLS = 2;                                 // BG1 punctured cols 0,1

        // Byte geometry. Compile-time immediates when RT_Z is false (the
        // values in the trailing comments), shifts of blockDim.x otherwise.
        static __device__ __forceinline__ int Z_()          { return a202_zstride<RT_Z>(); }
        static __device__ __forceinline__ int LD_COL_B()    { return Z_() * (int)sizeof(__half); }   // 768:  one column, one codeword
        static __device__ __forceinline__ int ST_COL_B()    { return Z_() * (int)sizeof(__half2); }  // 1536: one column, interleaved pair
        static __device__ __forceinline__ int LD_PUNCT_B()  { return PUNCT_COLS * LD_COL_B(); }      // 1536 (skip punctured input)
        static __device__ __forceinline__ int ST_PUNCT_B()  { return PUNCT_COLS * ST_COL_B(); }      // 3072 (store base = col 2)
        // This thread's quarter/lane-group divisor (96 = Z/4 at Z=384).
        // Runtime under RT_Z, but evaluated ONCE per kernel in issue().
        static __device__ __forceinline__ int QUARTER_LANES() { return Z_() / 4; }

        // Thread index at which staging round II crosses from the transient
        // extension planes to the survival planes: NQ whole quarters, i.e.
        // NQ*(Z/4) threads. Spelled as a template so the pinned variant
        // keeps the compile-time LITERAL it had before the widening --
        // routing it through QUARTER_LANES() instead cost no instructions
        // but made ptxas re-encode the compare (ISETP.GT.U32.OR ->
        // ISETP.GE.U32.AND), which broke the SASS-identity proof that is
        // the whole point of keeping a Z-pinned variant.
        template <int NQ>
        static __device__ __forceinline__ int straddle_thread()
        {
            if constexpr(RT_Z) { return NQ * QUARTER_LANES(); }
            else               { return NQ * (CFG_Z_MAX / 4); }
        }
        static constexpr int MIN_COLS    = CFG_KB + CFG_MIN_P - PUNCT_COLS;              // 32 transmitted cols at p12
        static constexpr int MAX_COLS    = CFG_KB + CFG_MAX_P - PUNCT_COLS;              // 41 transmitted cols at p21
        static constexpr int N_LDG_MAX   = (MAX_COLS + 3) / 4;                           // 11 rounds instantiated
        static constexpr int N_LDG_FULL  = MIN_COLS / 4;                                 // 8 rounds always complete
        static constexpr int N_PREBAR    = 5;                                            // rounds stored before the head barrier (row-0 columns; independent of N_LDG_FULL)
        // Lanes needed to zero the punctured columns, one uint4 each: 192 at Z=384.
        static __device__ __forceinline__ int ZERO_LANES()  { return ST_PUNCT_B() / (int)sizeof(sts_t); }

        //--------------------------------------------------------------
        // Column -> staging-round permutation (q = transmitted column
        // index = vnode column - 2). Chosen against the checked-in BG1
        // descriptors so the PRE-BARRIER head (rounds 0..4, 20 columns)
        // covers exactly the 17 transmitted columns row 0 reads/writes
        // (cols 2,3,5,6,9..13,15,16,18..23) plus three row-1 columns
        // (4,7,8) as fillers, while the DEFERRED round 5 carries the four
        // head columns no core row before row 1 touches:
        //   q {12,15,22,23} = cols {14,17,24,25}, first consumers rows
        //   {1,1,1,2} -- flushed in the iteration-0 after_row<0> hook, so
        //   they are CTA-visible strictly before row 1.
        // The head barrier then gates on 20/24 of the head bytes and the
        // last four columns' arrival overlaps row 0's processing instead
        // of heading it. The final staged image is byte-identical; only
        // WHEN each column becomes visible changes, and every column is
        // visible before its first consumer. Rounds 0,1,2 keep the
        // identity mapping (q = 4*round + quarter) and round 4 is affine
        // (base 18), so their addressing stays a compile-time immediate;
        // only rounds 3 and 5 are 4-entry gathers.
        static constexpr int      GATHER_A = 3;                                  // q {13,14,16,17} = cols 15,16,18,19
        static constexpr int      GATHER_B = 5;                                  // q {12,15,22,23} = cols 14,17,24,25 (deferred)
        static constexpr unsigned PACK_QA  = 13u | (14u << 8) | (16u << 16) | (17u << 24);
        static constexpr unsigned PACK_QB  = 12u | (15u << 8) | (22u << 16) | (23u << 24);

        // Affine base q for the non-gather rounds: rounds 0..2 and the
        // tail rounds 6..8 are identity (4*II); round 4 is base 18.
        template <int II>
        __device__ __forceinline__
        static constexpr int aff_q()
        {
            static_assert(II != GATHER_A && II != GATHER_B, "gather rounds have no affine base");
            return (II == 4) ? 18 : (4 * II);
        }

        ldg_t       r0[N_LDG_MAX]; // in-flight LLR data, codeword 0
        ldg_t       r1[N_LDG_MAX]; // in-flight LLR data, codeword 1
        const char* in0p;          // first transmitted column (cw0)
        const char* in1p;          // first transmitted column (cw1)
        int         gA, gB;       // gather-round per-thread load offsets (bytes into the transmitted region)
        int         sA, sB;       // gather-round per-thread store offsets (bytes into smem, incl. ST_PUNCT_B)

        // Transmitted bytes per codeword for the runtime p: (20+p)*2Z.
        __device__ __forceinline__
        static int load_bytes(const cuphyLDPCDecodeDesc_t& decodeDesc)
        {
            return (CFG_KB - PUNCT_COLS + decodeDesc.config.num_parity_nodes) * LD_COL_B();
        }

        // This thread's load byte offset for staging round II. For an
        // affine round this is the historical t*8 + base*LD_COL_B form;
        // the flattening identity t*8 == (t/(Z/4))*2Z + (t%(Z/4))*8 holds
        // at EVERY Z, since (Z/4)*8 == 2Z, so this stays one IMAD rather
        // than a divide. The gather rounds return the offsets precomputed
        // in issue(). All six non-tail rounds'
        // columns satisfy q <= 23 < MIN_COLS, so they exist for every
        // p >= CFG_MIN_P; only the tail rounds need the load_bytes
        // predicate.
        template <int II>
        __device__ __forceinline__
        int ld_off() const
        {
            if constexpr(II == GATHER_A) { return gA; }
            else if constexpr(II == GATHER_B) { return gB; }
            else
            {
                return (static_cast<int>(threadIdx.x) * (int)sizeof(ldg_t)) + (aff_q<II>() * LD_COL_B());
            }
        }

        // This thread's store byte offset (from smem) for staging round
        // II, including the punctured-column base. The HEAD rounds
        // (q <= 23, cols 2..25) store into the shared APP image as
        // before; the TAIL rounds (q >= 24: the extension columns, which
        // have NO persistent APP slot under LDPC2_A202_EXT_GLOBAL) store
        // into the extension staging planes in the tail region: column
        // 22+r lands in word plane bg1_a202_ext_stage_plane(r) off the
        // tail base at (CFG_KB+4)*Z*4 -- which folds to the same affine
        // shape with a piecewise column shift (+19 for the transient
        // rows-4..14 planes at q <= 34, +24 for the survival rows-15..20
        // planes at q >= 35):
        //   (q + SHIFT)*ST_COL_B + t*16
        //     = (CFG_KB+4)*Z*4 + plane(q+2-22)*Z*4 + z*16.
        static constexpr int EXT_STAGE_SHIFT_LO = 19; // q <= 34 (rows 4..14)
        static constexpr int EXT_STAGE_SHIFT_HI = 24; // q >= 35 (rows 15..20)
        // Both sides of these are homogeneous of degree 1 in Z (every term
        // is a column or plane count times Z), so the Z factor cancels and
        // proving them at CFG_Z_MAX proves them at every lifting. They are
        // assertions about the column->plane MAPPING, not about the lifting.
        static_assert((24 + EXT_STAGE_SHIFT_LO) * (CFG_Z_MAX * (int)sizeof(__half2)) ==
                      (CFG_KB + 4) * CFG_Z_MAX * (int)sizeof(__half2) +
                      ldpc2::bg1_a202_ext_stage_plane(4) * CFG_Z_MAX * (int)sizeof(ldpc2::word_t),
                      "staging shift must map col 26 (row 4) to its transient plane");
        static_assert((35 + EXT_STAGE_SHIFT_HI) * (CFG_Z_MAX * (int)sizeof(__half2)) ==
                      (CFG_KB + 4) * CFG_Z_MAX * (int)sizeof(__half2) +
                      ldpc2::bg1_a202_ext_stage_plane(15) * CFG_Z_MAX * (int)sizeof(ldpc2::word_t),
                      "staging shift must map col 37 (row 15) to its survival plane");
        template <int II>
        __device__ __forceinline__
        int st_off() const
        {
            if constexpr(II == GATHER_A) { return sA; }
            else if constexpr(II == GATHER_B) { return sB; }
            else if constexpr(aff_q<II>() >= 24)
            {
                // Tail round: quarters whose q >= 35 (rows >= 15) land in
                // the survival planes 5 columns further up. Rounds 6/7
                // (q <= 31) and 9/10 (q >= 36) are uniform; only round 8
                // (q 32..35) straddles, costing its last thread-quarter
                // one predicated offset add.
                constexpr int Q0 = aff_q<II>();
                int off = (static_cast<int>(threadIdx.x) * (int)sizeof(sts_t)) +
                          ((Q0 + EXT_STAGE_SHIFT_LO) * ST_COL_B());
                if constexpr(Q0 >= 35)
                {
                    off += (EXT_STAGE_SHIFT_HI - EXT_STAGE_SHIFT_LO) * ST_COL_B();
                }
                else if constexpr(Q0 + 3 >= 35)
                {
                    // The straddle boundary is a COLUMN boundary: quarters
                    // 0..(35-Q0-1) of the round are transient planes and the
                    // rest survival, so the thread index that separates them
                    // is (35-Q0) whole quarters = (35-Q0)*(Z/4) threads --
                    // 96 per quarter at Z=384.
                    if(static_cast<int>(threadIdx.x) >= straddle_thread<35 - Q0>())
                    {
                        off += (EXT_STAGE_SHIFT_HI - EXT_STAGE_SHIFT_LO) * ST_COL_B();
                    }
                }
                return off;
            }
            else
            {
                return ST_PUNCT_B() + (static_cast<int>(threadIdx.x) * (int)sizeof(sts_t)) +
                       (aff_q<II>() * ST_COL_B());
            }
        }

        // One staging round's pair of global loads.
        template <int II>
        __device__ __forceinline__
        void load_round()
        {
            r0[II] = *reinterpret_cast<const ldg_t*>(in0p + ld_off<II>());
            r1[II] = *reinterpret_cast<const ldg_t*>(in1p + ld_off<II>());
        }

        //--------------------------------------------------------------
        // issue()
        // Token search, then issue every transmitted-column global load
        // up front (max MLP) and the punctured-column zero fill. Returns
        // with the loads still in flight; the caller runs load-independent
        // setup before calling complete().
        //
        // Single-TB LDG-issue fast path: when the descriptor carries one
        // transport block the owning TB is 0, so the source addressing
        // needs only llr_input[0], whose fields sit at compile-time
        // constant-bank offsets. Reading them unconditionally (entry 0 is
        // always valid) breaks the token -> tb -> indexed-LDCU
        // serialization off the LDG-issue critical path; the token is
        // built in parallel and is bit-identical to the scan's. The
        // multi-TB path recomputes the source addressing from the scanned
        // token exactly as before.
        __device__ __forceinline__
        tb_token issue(char*                        smem,
                       const cuphyLDPCDecodeDesc_t& decodeDesc,
                       unsigned int                 decodeIndex)
        {
            const int          num_cw0  = decodeDesc.llr_input[0].num_codewords;
            const int          stride0  = decodeDesc.llr_input[0].stride_elements;
            const char*        taddr0   = static_cast<const char*>(decodeDesc.llr_input[0].addr);
            const unsigned int offset0  = decodeIndex * 2;
            const bool         partial0 = ((offset0 + 2) > static_cast<unsigned int>(num_cw0));

            tb_token    tok;
            const char* in0;
            int         stride1;
            if(decodeDesc.num_tbs == 1)
            {
                tok     = to_token<2>(0, offset0, partial0);
                in0     = taddr0 + (offset0 * static_cast<int>(sizeof(__half)) * stride0);
                stride1 = partial0 ? 0 : stride0;
            }
            else
            {
                tok = find_block_tb_token_allwarp<2>(decodeDesc, decodeIndex);
                const int tb       = tb_from_token(tok);
                const int cwOffset = offset_from_token(tok);
                const int stride   = decodeDesc.llr_input[tb].stride_elements;
                in0     = static_cast<const char*>(decodeDesc.llr_input[tb].addr) +
                          (cwOffset * static_cast<int>(sizeof(__half)) * stride);
                stride1 = is_partial_from_token(tok) ? 0 : stride;
            }
            const char* in1 = in0 + (stride1 * static_cast<int>(sizeof(__half)));

            in0p = in0 + LD_PUNCT_B(); // first transmitted column (cw0)
            in1p = in1 + LD_PUNCT_B(); // first transmitted column (cw1)
            const int stOff = static_cast<int>(threadIdx.x) * (int)sizeof(sts_t);

            // Gather-round permutation offsets: this thread's quarter
            // (t/(Z/4)) selects the packed q byte; the lane group (t%(Z/4))
            // addresses its 4 z lanes within that column.
            const int qlanes = QUARTER_LANES();
            const int tq  = static_cast<int>(threadIdx.x) / qlanes;
            const int rq  = static_cast<int>(threadIdx.x) - (tq * qlanes);
            const int qsh = tq * 8;
            const int qA  = static_cast<int>((PACK_QA >> qsh) & 0xFFu);
            const int qB  = static_cast<int>((PACK_QB >> qsh) & 0xFFu);
            gA = (qA * LD_COL_B()) + (rq * (int)sizeof(ldg_t));
            gB = (qB * LD_COL_B()) + (rq * (int)sizeof(ldg_t));
            sA = ST_PUNCT_B() + (qA * ST_COL_B()) + (rq * (int)sizeof(sts_t));
            sB = ST_PUNCT_B() + (qB * ST_COL_B()) + (rq * (int)sizeof(sts_t));

            // Head + deferred rounds: the tail rounds are issued later
            // (issue_tail, just before the head barrier), behind every
            // head request in program order, so the head drain keeps its
            // place at the front of the memory queue and the barrier
            // releases as soon as row 0's columns have arrived.
            load_round<0>();
            load_round<1>();
            load_round<2>();
            load_round<3>();
            load_round<4>();
            load_round<5>();
            // Punctured columns 0,1: write 0 (no transmitted channel LLR);
            // issues while the loads above are in flight.
            if(static_cast<int>(threadIdx.x) < ZERO_LANES())
            {
                *reinterpret_cast<sts_t*>(smem + stOff) = make_uint4(0u, 0u, 0u, 0u);
            }
            return tok;
        }

        //--------------------------------------------------------------
        // issue_tail()
        // Issue the tail rounds' global loads (cols 26..21+p). Called
        // just BEFORE the head barrier (in the head-load latency shadow)
        // so these loads are in flight underneath the head drain, the
        // barrier, and iteration 0's core-row processing; their stores
        // happen in the after_row hooks (rows 3/7/11).
        __device__ __forceinline__
        void issue_tail(const cuphyLDPCDecodeDesc_t& decodeDesc)
        {
            const int LOAD_BYTES = load_bytes(decodeDesc);
            // Rounds 6/7 cover q 24..31 < MIN_COLS: unconditional at p >= 15.
            load_round<6>();
            load_round<7>();
            if(ld_off<8>()  < LOAD_BYTES) { load_round<8>();  }
            if(ld_off<9>()  < LOAD_BYTES) { load_round<9>();  }
            if(ld_off<10>() < LOAD_BYTES) { load_round<10>(); }
        }

        //--------------------------------------------------------------
        // store_round()
        // Interleave the two codewords, clamp, and store one staging
        // round. The interleave/clamp computes unconditionally (an
        // inactive round computes on unconsumed registers); only the STS
        // itself is predicated, so no branch/reconvergence block is paid.
        // Rounds 0..5 (q <= 23 < MIN_COLS) are unconditional.
        template <int II>
        __device__ __forceinline__
        void store_round(char* smem, const cuphyLDPCDecodeDesc_t& decodeDesc)
        {
            sts_t sv = llr_op_clamp<__half, sts_t>::apply(interleave_llr(r0[II], r1[II]),
                                                          decodeDesc.config.clamp_value);
            if((II < N_LDG_FULL) || (ld_off<II>() < load_bytes(decodeDesc)))
            {
                *reinterpret_cast<sts_t*>(smem + st_off<II>()) = sv;
            }
        }

        //--------------------------------------------------------------
        // complete_head()
        // Store the pre-barrier rounds 0..N_PREBAR-1 (the 17 transmitted
        // columns row 0 reads/writes plus fillers 4,7,8) and make them --
        // together with the punctured-column zeros -- visible to the
        // whole CTA. This is everything BG1 row 0 (see the permutation
        // note above) touches, so iteration 0 starts row 0 while the
        // deferred round's and the tail rounds' global loads are still
        // in flight.
        __device__ __forceinline__
        void complete_head(char* smem, const cuphyLDPCDecodeDesc_t& decodeDesc)
        {
            static_assert(N_PREBAR == 5 && N_LDG_FULL == 8 && N_LDG_MAX == 11,
                          "staging round split is hardcoded to 5 pre-barrier + 1 deferred + 2 unconditional + 3 tail rounds");
            store_round<0>(smem, decodeDesc);
            store_round<1>(smem, decodeDesc);
            store_round<2>(smem, decodeDesc);
            store_round<3>(smem, decodeDesc);
            store_round<4>(smem, decodeDesc);
            __syncthreads();
        }

        //--------------------------------------------------------------
        // after_row()
        // Iteration-0 deferred/tail-flush hook, called by the schedule
        // after process_row<ROW> and BEFORE that row's __syncthreads(),
        // so each flushed round is CTA-visible strictly before its first
        // consumer row. The flush rows are chosen against the checked-in
        // BG1 descriptors (row r >= 4 first touches vnode column 22+r;
        // deferred head cols 14,17,24 are first touched by row 1 and col
        // 25 by row 2):
        //   round 5 (cols 14,17,24,25, consumers rows 1..2, always active)
        //     -> after row 0  (row 0 executes for every p)
        //   round 6 (cols 26..29, consumers rows 4..7,  active for p >= 5)
        //     -> after row 3  (rows 0..3 execute for every p >= CFG_MIN_P)
        //   round 7 (cols 30..33, consumers rows 8..11, active for p >= 9)
        //     -> after row 7  (row 7 executes whenever p >= 8)
        //   round 8 (cols 34..37, consumers rows 12..15, active for p >= 13)
        //     -> after row 11 (just in time for row 12)
        //   round 9 (cols 38..41, consumers rows 16..19, active for p >= 17)
        //     -> after row 12 (NOT the just-in-time row 15: the loads were
        //        issued before row 0, so by row 12 they arrived long ago,
        //        and flushing inside the compile-time-unconditional region
        //        ends the r0[9]/r1[9] register live range before the
        //        runtime-p tail rows instead of spanning them)
        //   round 10 (cols 42..45, consumer row 20, active for p = 21)
        //     -> after row 13 (same audit as round 9: just-in-time row 19
        //        kept 4 registers live across nearly the whole first
        //        iteration for no remaining latency-hiding benefit)
        // This staggering keeps the deferred/tail global loads in flight
        // underneath real row compute instead of serializing the full
        // staging drain ahead of iteration 0, while ending every staging
        // live range by row 13 (the last compile-time-unconditional row).
        template <int ROW>
        __device__ __forceinline__
        void after_row(char* smem, const cuphyLDPCDecodeDesc_t& decodeDesc)
        {
            if constexpr(ROW == 0)  { store_round<5>(smem, decodeDesc); }
            if constexpr(ROW == 3)  { store_round<6>(smem, decodeDesc); }
            if constexpr(ROW == 7)  { store_round<7>(smem, decodeDesc); }
            if constexpr(ROW == 11) { store_round<8>(smem, decodeDesc); }
            if constexpr(ROW == 12) { store_round<9>(smem, decodeDesc); }
            if constexpr(ROW == 13) { store_round<10>(smem, decodeDesc); }
        }
    };

    //------------------------------------------------------------------
    // stager_tail_flush
    // Adapter binding the stager and its arguments for the schedule's
    // iteration-0 after-row hook.
    template <bool RT_Z>
    struct stager_tail_flush
    {
        llr_stager<RT_Z>&            stager;
        char*                        smem;
        const cuphyLDPCDecodeDesc_t& decodeDesc;

        template <int ROW>
        __device__ __forceinline__
        void after_row()
        {
            stager.template after_row<ROW>(smem, decodeDesc);
        }
    };
#endif // LDPC_DECODE_USE_TB_SCAN

    //------------------------------------------------------------------
    // ldpc2_BG1_z256up_cms_shtail_x2_tb_body()
    // Shared kernel body (transport block interface, GB203-only
    // compressed-min-sum core). Only iteration 0 is peeled: it elides the
    // still-zero previous-C2V subtract (bit-identical) and hosts the
    // entry-staging flush hooks and the extension-column register-cache
    // capture. Iterations 1..max-1 -- INCLUDING the final one -- run the
    // single warm do_iteration() body: a peeled final iteration would only
    // elide dead register writes here while paying a once-per-CTA cold
    // i-fetch of its own body (measured ~3.8% of p5 kernel time on GB203;
    // see the schedule comment). Every codeword still runs the full
    // max_iterations -- no early stop.
    template <bool RT_Z>
    __device__ __forceinline__
    void ldpc2_BG1_z256up_cms_shtail_x2_tb_body(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                   const a202_bg_desc_t&        bgdesc)
    {
        extern __shared__ char smem[];

        typedef algo202_kernel_config<cuphyLDPCDecodeConfigDesc_t, RT_Z> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
#error "LDPC2_A202_EXT_GLOBAL requires the TB_SCAN entry stager (the generic loader assumes the baseline APP layout)"
        kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
        ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc);
        ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
        // Entry staging split: issue() returns with the channel loads in
        // flight; every declaration between issue() and complete_head()
        // (output parameter blocks, config copy/norm scale, and the
        // schedule construction with its C2V register-cache init and APP
        // address setup) has no dependency on the staged APP bytes, so it
        // executes inside the global-load-latency shadow instead of
        // serialized behind the staging barrier.
        llr_stager<RT_Z> stager;
        tb_token tok = stager.issue(smem, decodeDesc, blockIdx.x);
        ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc, output_token(tok));
        ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
        const bool write_soft = (0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
        cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;

#if LDPC_MICRO_PACK_ARGMIN
        // E5M5 truncation compensation for the compressed-min-sum core: the
        // packed-argmin scan truncates core-row min-sum magnitudes to 5
        // mantissa bits; scale the norm to restore the effective normalization.
        //
        // Guarded by the same macro that introduces the truncation, so the
        // bias and its correction are switched together. Unguarded, setting
        // PACK_ARGMIN to 0 would leave the norm scaled up by ~1.3% with
        // nothing to cancel -- a mis-normalized decoder whose BLER would be
        // read as the cost of turning E5M5 off. algo203/algo204 guard theirs.
        config.norm.f16x2 = __hmul2(config.norm.f16x2, __float2half2_rn(E5M5_NORM_SCALE));
#endif

        typename kernel_config_t::sched_t sched(smem,
                                                config,
                                                bgdesc,
                                                static_cast<int>(__cvta_generic_to_shared(smem)),
                                                threadIdx.x);

        // Row 0 address array: filled once HERE -- before the head
        // barrier, inside the head-load latency shadow (address
        // generation reads only compile-time immediates and thread id,
        // never staged APP bytes) -- then refilled at every iteration
        // tail before the boundary barrier (see do_rows). This removes
        // the degree-19 dp2a chain from the post-barrier critical path
        // that gates row 0's first shared-memory load.
        int addr0[row_degree<CFG_BG, 0>::value];
        sched.gen_row0(addr0);

#if LDPC_DECODE_USE_TB_SCAN
#if LDPC2_A202_TAIL_PRE_BARRIER
        // Tail-round global loads: issued BEFORE the head-visibility
        // barrier, in the shadow of the still-in-flight head loads. The
        // LDGs sit behind the head loads in program order, so the head
        // rounds' arrival (which gates the barrier) is not delayed, while
        // the tail rounds enter the memory system one barrier plus the
        // post-barrier setup earlier than an after-barrier issue --
        // giving the after_row<3> flush that consumes round 6 the full
        // core-row span of latency shadow. A global load cannot be
        // hoisted across BAR.SYNC, so this placement is stable.
        stager.issue_tail(decodeDesc);
#endif
        // Pre-barrier rounds (everything row 0 touches) become visible
        // here; the deferred round and the tail rounds flush inside
        // iteration 0 via the after-row hooks, overlapping their
        // global-load completion with row processing.
        stager.complete_head(smem, decodeDesc);
#if !LDPC2_A202_TAIL_PRE_BARRIER
        // Tail rounds issued after the head barrier: the head drain keeps
        // the L2 queue to itself; the loads are still in flight underneath
        // iteration 0's core-row processing before their after_row flushes.
        stager.issue_tail(decodeDesc);
#endif
        stager_tail_flush<RT_Z> tail_flush{stager, smem, decodeDesc};
        if(config.max_iterations < 1)
        {
            // Degenerate 0-iteration decode: no row ever consumes (or
            // flushes) the deferred/tail rounds; flush them here so the
            // staged APP image matches the generic loader's before output.
            stager.template after_row<0>(smem, decodeDesc);
            stager.template after_row<3>(smem, decodeDesc);
            stager.template after_row<7>(smem, decodeDesc);
            stager.template after_row<11>(smem, decodeDesc);
            stager.template after_row<12>(smem, decodeDesc);
            stager.template after_row<13>(smem, decodeDesc);
            __syncthreads();
        }
#endif

        // app0 is row 0's APP word array (not preloaded: PRE == 0; the
        // words are loaded inside row_body). addr0 itself was filled
        // pre-barrier above.
        word_t app0 [app_num_words<__half2, CFG_BG, 0>::value];

        // First-iteration-peeled loop structure (see the schedule comment):
        // the final iteration runs the same warm do_iteration() body.
        int iter = 0;
        if(iter < config.max_iterations)
        {
#if LDPC_DECODE_USE_TB_SCAN
            sched.do_first_iteration(addr0, app0, tail_flush,
                                     LDPC2_A202_TAIL_LAST_ELIDE && (config.max_iterations == 1));
#else
            sched.do_first_iteration(addr0, app0,
                                     LDPC2_A202_TAIL_LAST_ELIDE && (config.max_iterations == 1));
#endif
            ++iter;
        }
        for(; iter < config.max_iterations; ++iter)
        {
            sched.do_iteration(addr0, app0,
                               LDPC2_A202_TAIL_LAST_ELIDE && (iter == (config.max_iterations - 1)));
        }

        ldpc_dec_output_x2_all_warps<CFG_OUT_WORDS_PER_WARP>(hard_out_params,
            reinterpret_cast<const typename kernel_config_t::app_buf_t*>(smem));
        if(write_soft)
        {
            ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const typename kernel_config_t::app_buf_t*>(smem));
        }
    }

} // namespace algo202

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_z256up_cms_shtail_x2_tb_msc()
// BG1 / Z in {256..384} / p=15..21 kernel, compressed-min-sum core +
// compressed shared tail rows (transport block interface), RUNTIME Z:
// correct at every lifting in the zone. Selected for Z < CFG_Z_MAX.
// GB203-only build: this is the only production kernel family; the
// box_plus-core and legacy tensor-interface kernels have been dropped.
extern "C"
__global__ __launch_bounds__(algo202::MAX_THREADS_PER_CTA, algo202::MIN_CTA_PER_SM)
void ldpc2_BG1_z256up_cms_shtail_x2_tb_msc(cuphyLDPCDecodeDesc_t decodeDesc, algo202::a202_bg_desc_t bgdesc)
{
    algo202::ldpc2_BG1_z256up_cms_shtail_x2_tb_body<true>(decodeDesc, bgdesc);
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384()
// Same kernel with Z pinned to CFG_Z_MAX at compile time. Selected for
// Z == CFG_Z_MAX, where the folded LDS/STS immediates are worth keeping.
extern "C"
__global__ __launch_bounds__(algo202::MAX_THREADS_PER_CTA, algo202::MIN_CTA_PER_SM)
void ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384(cuphyLDPCDecodeDesc_t decodeDesc, algo202::a202_bg_desc_t bgdesc)
{
    algo202::ldpc2_BG1_z256up_cms_shtail_x2_tb_body<false>(decodeDesc, bgdesc);
}


namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// bg1_z256up_cms_shtail_x2::bg1_z256up_cms_shtail_x2()
bg1_z256up_cms_shtail_x2::bg1_z256up_cms_shtail_x2(ldpc::decoder& dec)
{
    // Sized at the top of the zone: shared memory is monotone in Z
    // so the CFG_Z_MAX requirement covers every lifting below.
    const int MAX_SHMEM_MSC = algo202::get_shmem_required(algo202::CFG_MAX_P, algo202::CFG_Z_MAX);
    const int MAX_SHMEM     = dec.max_shmem_per_block_optin();

    // Both variants need the opt-in attribute: a kernel that never gets it
    // fails to launch the moment dispatch picks it, and the Z that selects
    // it may not be the Z anything was tested at.
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 2> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_z256up_cms_shtail_x2_tb_msc,      std::min(MAX_SHMEM_MSC, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384, std::min(MAX_SHMEM_MSC, MAX_SHMEM))
    };
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_z256up_cms_shtail_x2_tb_msc);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_cms_shtail_x2::decode()
// Legacy tensor interface dropped in the GB203-only build; only the
// transport-block interface (decode_tb) is supported.
cuphyStatus_t bg1_z256up_cms_shtail_x2::decode(ldpc::decoder&                     dec,
                                           LDPC_output_t&                     tDst,
                                           const_tensor_pair&                 tLLR,
                                           const cuphy_optional<tensor_pair>& optSoftOutputs,
                                           const cuphyLDPCDecodeConfigDesc_t& config,
                                           cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::bg1_z256up_cms_shtail_x2::decode() -- tensor interface not supported\n");
    return CUPHY_STATUS_NOT_SUPPORTED;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_cms_shtail_x2::decode_tb()
cuphyStatus_t bg1_z256up_cms_shtail_x2::decode_tb(ldpc::decoder&               dec,
                                              const cuphyLDPCDecodeDesc_t& decodeDesc,
                                              cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::bg1_z256up_cms_shtail_x2::decode_tb()\n");

    if(!can_decode_config(dec, decodeDesc.config))
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        dim3 blkDim(decodeDesc.config.Z);
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        const algo202::a202_bg_desc_t* bgdesc = algo202::a202_get_bg_desc(decodeDesc.config.Z);
        if(!bgdesc) return CUPHY_STATUS_INTERNAL_ERROR;

        // GB203-only: compressed-min-sum core kernel (the box_plus core needed
        // ~228 KB shmem that GB203 does not have, so it was dropped).
        const uint32_t SHMEM_SIZE_MSC = algo202::get_shmem_required(decodeDesc.config.num_parity_nodes,
                                                                    decodeDesc.config.Z);
        // The Z-pinned variant at the top of the zone, the runtime-Z variant
        // below it.
        const bool     z_pinned       = (algo202::CFG_Z_MAX == decodeDesc.config.Z);

        if(z_pinned)
        {
            DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384, blkDim, SHMEM_SIZE_MSC);
            ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384<<<grdDim, blkDim, SHMEM_SIZE_MSC, strm>>>(decodeDesc, *bgdesc);
        }
        else
        {
            DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_z256up_cms_shtail_x2_tb_msc, blkDim, SHMEM_SIZE_MSC);
            ldpc2_BG1_z256up_cms_shtail_x2_tb_msc<<<grdDim, blkDim, SHMEM_SIZE_MSC, strm>>>(decodeDesc, *bgdesc);
        }
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
// bg1_z256up_cms_shtail_x2::get_workspace_size()
std::pair<bool, size_t> bg1_z256up_cms_shtail_x2::get_workspace_size(const ldpc::decoder&               dec,
                                                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                                                 int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_cms_shtail_x2::can_decode_config()
// Notes on three of the checks below:
//
//   * Kb == 22 is a precondition, not a restatement of BG1. The hard-decision
//     writer is instantiated as <CFG_OUT_WORDS_PER_WARP = Kb> and writes
//     22 * Z bits whatever the descriptor says, so a smaller Kb overruns a
//     buffer the caller sized from it. Do not drop this term.
//   * The Z test is Z % 32 == 0 within [CFG_Z_MIN, CFG_Z_MAX]. In that range
//     it selects exactly the five legal 38.212 liftings, which is what lets
//     the %32 test stand in for a table lookup. The coincidence does NOT
//     survive widening the zone downward -- see the same note in algo201.
//   * The shared-memory test is per-device, so acceptance is not a property
//     of the configuration alone.
bool bg1_z256up_cms_shtail_x2::can_decode_config(const ldpc::decoder&               dec,
                                             const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    if((cfg.llr_type != CUPHY_R_16F)                     ||
       (cfg.BG != algo202::CFG_BG)                       ||
       (cfg.Kb != 22)                                    ||
       !algo202::a202_z_supported(cfg.Z)                 ||
       (cfg.num_parity_nodes < algo202::CFG_MIN_P)       ||
       (cfg.num_parity_nodes > algo202::CFG_MAX_P))
    {
        return false;
    }
    // GB203-only compressed-min-sum core: APP buffer only.
    const int SHMEM_SIZE = algo202::get_shmem_required(cfg.num_parity_nodes, cfg.Z);
    return (SHMEM_SIZE <= dec.max_shmem_per_block_optin());
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_cms_shtail_x2::get_launch_config()
cuphyStatus_t bg1_z256up_cms_shtail_x2::get_launch_config(const ldpc::decoder&           dec,
                                                      cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    if(!can_decode_config(dec, launchConfig.decode_desc.config))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    const int Z                = launchConfig.decode_desc.config.Z;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;

    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra        = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams = launchConfig.kernel_args;

    // GB203-only: compressed-min-sum core kernel (box_plus core dropped).
    const uint32_t SHMEM_SIZE = algo202::get_shmem_required(NUM_PARITY_NODES, Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const bool z_pinned = (algo202::CFG_Z_MAX == Z);
    const void* kernel_func = z_pinned ? (const void*)ldpc2_BG1_z256up_cms_shtail_x2_tb_msc_z384
                                       : (const void*)ldpc2_BG1_z256up_cms_shtail_x2_tb_msc;
    cudaError_t e = cudaGetFuncBySymbol(&deviceFunction, kernel_func);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif

    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    const algo202::a202_bg_desc_t* bgdesc = algo202::a202_get_bg_desc(Z);
    launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));

    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
