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

//#define CUPHY_DEBUG 1

// ---------------------------------------------------------------------------
// TU-scoped micro-optimization toggles (must precede the ldpc2_c2v_x2.cuh /
// ldpc2_box_plus_x2.cuh includes; every other TU sees the verbatim baseline
// paths). A/B block -- see per-knob verdicts below.
//
// RAW_C2V=1 measured 0.4-0.5% FASTER, but it changes the APP-add rounding
// form, which breaks the per-iteration --llr_check tool; left at 0 pending a
// re-A/B. Note that E5M5 packed-argmin below already perturbs APP
// trajectories, so --llr_check is not usable in this TU regardless.
//
// A mutually exclusive alternative, running the box-plus tree on
// pre-normalized inputs, measured 0.1-1.1% FASTER on this band -- a higher
// ceiling than RAW_C2V, but noisier across p22..34. Its implementation was
// removed from ldpc2_box_plus_x2.cuh because no translation unit selected
// it; recover it from history (git log -S LDPC2_BP_X2_NORM_INPUT) if the
// re-A/B is ever run, since it belongs in that comparison.
#define LDPC2_BP_X2_RAW_C2V       0

// Compact-layout residual writeback policy.  Rows 22..25 and 27 live in the
// shared C2V window, while ptxas already hides the always-active row-21 stores
// behind independent work.  Keep only the remaining optional tail rows at
// their production points so their spill stores do not collect immediately
// before the row synchronization path.
#define LDPC2_C2V_SELECTIVE_WRITEBACK 1

// [B] E5M5 packed-argmin core scan (degree-19 msc rows 0-3) + satellite
// micro-knobs, ported from algo202 (same box-plus non-core / msc-core
// anatomy). The packed (|app|<<5 | colindex) VIMNMX2 scan truncates min-sum
// magnitudes to 5 mantissa bits (E5M5) -- a real numerics change. Measured
// with PACK_ARGMIN=1 + satellites + PAIRWISE=1 (2026-07-07): latency 2.3-5.7%
// LOWER than the pre-port base, at p22 / p26 / p31 / p34; TB BLER 2-3.7x
// BETTER at a40-anchored 10% and 1% PUSCH RM points (truncation ~=
// core-rows-only norm shrink; see the E5M5_NORM_SCALE note on core-only
// compensation).
//
// ENABLED (=1). E5M5 diverges APP trajectories, so --llr_check cannot be
// used against an exact-fp16 reference while it is on; that is a property
// of this setting, not a reason it is off. The norm compensation below is
// guarded by this same macro, so setting it to 0 correctly drops both the
// truncation and its correction together.
#define LDPC_MICRO_PACK_ARGMIN      1
#define LDPC_MICRO_ABS_MASK_CONST   1
#define LDPC_MICRO_PARTIAL_SIGN     1
#define LDPC_MICRO_PREDIFF          1
#define LDPC_MICRO_SPLIT_REDUCE     1
#define LDPC_MICRO_XOR_SIGNPROD     1
#define LDPC_MICRO_PARTIAL_SIGN_ALL 1
#define LDPC2_X2_CORE_SWAP           1
#define LDPC2_X2_FUSE_NORM          0
// PAIRWISE verdict is not portable between kernels of this anatomy
// (algo202 settles on 0) -- A/B'd here.
#define LDPC_PAIRWISE_MINSUM        0

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_box_plus_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_app_address_dp_z_imm.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_algo204.hpp"
#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_crc_dispatch.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // bg1_z256up_bp_gmext_x2 band specialization (BG1 / Z=256..384 / p=35..46, 2 CW/CTA).
    // Z became runtime in this band decoder; the p=35..44
    // wording below predates the band reaching 46.
    // UPWARD band extension of bg1_z256up_bp_shwin_x2 (fork, 2026-07-11): same compact-APP
    // x2 architecture, band moved from p22..34 to p35..44. Band cap 44 is
    // the LLR staging ceiling: initial staging of 22+p columns must fit the
    // fixed 66-word shmem block (46 APP + 18 window + 2 shared-ext =
    // 101,376 B), which holds exactly 66 columns (p=44). p45/46 would need
    // ext-from-gmem staging. Extension capture is DESCENDING
    // (see capture_extensions) -- at p>=43 staging cols 64/65 alias the
    // shared_extension_ slots.
    // (Original bg1_z256up_bp_shwin_x2 note below, retained for the architecture:)
    // Fork of the ALGO 55 bp-hybrid-bigreg variant
    // (ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc: box-plus
    // non-core C2V, compressed min-sum core C2V in registers,
    // MIN_PARITY_ROWS = 22) with:
    //   * num_reg_parity<1> = 34 (bigreg uses 30): ALL in-band parity
    //     rows keep their C2V in registers/local -- the shared-memory
    //     C2V band never exists for p <= 34, so shmem holds APP only
    //     (max 86KB @ p=34 x2). This is what lifts the GB203 fit
    //     ceiling: bigreg's shmem C2V tail (40B/row/thread from
    //     c2v_storage_x2_box_plus<9>) fails the 99KB budget at p=32;
    //     with no tail, p=32..34 fit. The extra 4 rows cost ~160B/thread
    //     of local spill on top of bigreg's existing 200B (REG is
    //     already at the LB(384,1) ceiling of 168).
    //   * max_num_parity<1> = 34 (was 46): the dynamic schedule's
    //     dispatch chain is compiled only to row 33, shrinking code and
    //     relieving register/spill pressure.
    //   * MIN_PARITY_ROWS = 22 for BG1, inherited from the bigreg
    //     parent (compiled as a high-p kernel; the band gate makes the
    //     elided IS_LAST_ROW checks safe). Note this differs from the
    //     algo35-family floor lesson -- the incumbent bigreg kernel
    //     measured fastest with the floor at 22, so it is kept; the
    //     floor sweep remains open as a tuning knob.
    // BG2 values are inherited but unreachable: can_decode_config()
    // gates this decoder to BG1 / Z in {256,288,320,352,384} / 35 <= p <= 46.
    // (The '22 <= p <= 34' here was stale from the bg1_z256up_bp_shwin_x2 fork.)
    // BG1 only: this decoder's can_decode_config() rejects BG != 1, and the
    // BG2 kernels it used to compile were unreachable.
    template <int BG> struct num_reg_parity;
    template <> struct num_reg_parity<1> { static constexpr int value = 46; };
    template <int BG> struct max_num_parity;
    template <> struct max_num_parity<1> { static constexpr int value = 46; };

    //------------------------------------------------------------------
    // Non-core C2V storage type selection:
    // box_plus storage (forward-backward, eliminates sign tracking),
    // same as the ALGO 55 parent.
    static constexpr int BG1_MAX_BOX_PLUS_WORDS = 9; // max UPDATE_ROW_DEGREE for BG1 non-core rows
    template <int BG> struct noncore_storage_x2_local;
    template <> struct noncore_storage_x2_local<1> { typedef c2v_storage_x2_box_plus<BG1_MAX_BOX_PLUS_WORDS> type; };

    //------------------------------------------------------------------
    // Sign manager for compressed C2V row processor
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    // APP address calculation
    // A204_RUNTIME_Z_ADDR selects how per-edge APP addresses are formed, and
    // with it whether this kernel is Z-generic at all:
    //
    //   0 -> Z=384 COMPILE-TIME IMMEDIATES. shift/stride arithmetic folds into
    //        HSET2+dp2a immediates instead of
    //        runtime descriptor loads. Correct ONLY at Z=384.
    //   1 -> runtime descriptor loads; the band gate opens to
    //        Z in {256,288,320,352,384}.  [default]
    //
    // Measured cost of 1 vs 0 at Z=384, same GPU back to back, 2048 cw, n=10,
    // -r 20: 5.01% / 5.55% / 5.65% / 6.84% SLOWER at p=42/38/35/46. REG:168 in
    // both -- the ceiling is unchanged, the cost is spill (540 -> 704 B).
    // Adopted anyway: it buys 48 configs (p=35..46 x Z=256..352) that had no
    // band kernel at all, at 1.12-1.32x over the best generic there.
    //
    // A204_ZSTRIDE below MUST move with this knob.
#ifndef A204_RUNTIME_Z_ADDR
#define A204_RUNTIME_Z_ADDR 1
#endif
// Shared-memory ROW STRIDE. blockDim.x IS Z for this kernel (1 CTA/SM,
// blockDim == Z), so Z-genericity needs no extra live value carried through the
// decode loop -- %ntid.x is uniform and ptxas can rematerialize it, unlike a Z
// member on the C2V cache which would sit in a register across every iteration.
// Must move together with the addressing knob: a runtime-Z address generator
// over compile-time-384 strides (or vice versa) indexes past the APP image.
#if A204_RUNTIME_Z_ADDR
#define A204_ZSTRIDE (static_cast<int>(blockDim.x))
#else
#define A204_ZSTRIDE 384
#endif

#if A204_RUNTIME_Z_ADDR
    // runtime-Z (algo55) variant (measured ablation 2026-07-07:
    // costs 7-10% across the band, and at p=22 drops below
    // the a55 bp-hybrid incumbent):
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half2, BG>;
#else
    template <int BG> using app_loc_t = app_loc_address_dp_z_imm<__half2, BG, 384>;
#endif

    //------------------------------------------------------------------
    // Template alias for a half2 row context, templated ONLY on the
    // underlying storage type. (For this decoder, we will use different
    // row contexts, and thus slightly different row processors,  for
    // the "high degree" core rows.)
    template <class TStorage> using row_context_t = cC2V_row_context<__half2,
                                                                     sign_mgr_t,
                                                                     unused,
                                                                     TStorage>;
    //------------------------------------------------------------------
    // Template alias for a half2 compressed C2V row processors,
    // templated ONLY on the row context used. This will be used by the
    // row mappers, which will instantiate a cC2V_row_proc_t template
    // instance for the different row context storage types.
    template <class TRowContext> using cC2V_row_proc_t = cC2V_row_proc<__half2,
                                                                       TRowContext>;

    //------------------------------------------------------------------
    // The transport-block lookup is CTA-uniform. Resolve it in one warp and
    // broadcast the owner lane's completed token, avoiding the CTA-wide
    // shared-memory token round trip before LLR staging.
    template <int CW_PER_CTA>
    __device__ __forceinline__
    tb_token bg1_z256up_bp_gmext_x2_find_token(const cuphyLDPCDecodeDesc_t& decode_desc,
                                unsigned int                 decode_index)
    {
        if(decode_desc.num_tbs == 1)
        {
            const int          num_cw  = decode_desc.llr_input[0].num_codewords;
            const unsigned int offset  = decode_index * CW_PER_CTA;
            const bool partial = ((offset + CW_PER_CTA) > static_cast<unsigned int>(num_cw));
            return to_token<CW_PER_CTA>(0, offset, partial);
        }
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int entry_cw = (lane < decode_desc.num_tbs)
                                 ? decode_desc.llr_input[lane].num_codewords : 0;
        const int entry_blocks = (entry_cw + CW_PER_CTA - 1) / CW_PER_CTA;
        const int blocks_sum = warp_exclusive_scan<int>(entry_blocks);
        const unsigned int sum_gt = __ballot_sync(0xFFFFFFFFu, blocks_sum > decode_index);
        const unsigned int tb = sum_gt ? static_cast<unsigned int>(__ffs(sum_gt) - 2) : 31u;
        const unsigned int offset = (decode_index - blocks_sum) * CW_PER_CTA;
        const bool partial = ((offset + CW_PER_CTA) > static_cast<unsigned int>(entry_cw));
        const tb_token tok = to_token<CW_PER_CTA>(tb, offset, partial);
        return __shfl_sync(0xFFFFFFFFu, tok, tb);
    }

    //------------------------------------------------------------------
    // Columns 46..55 are the isolated extensions of optional rows 24..33.
    // They are captured once; columns 0..45 remain resident for the common
    // prefix and rows 22/23.
    static constexpr int COMPACT_APP_NODES = 46;
    static constexpr int WINDOW_FIRST = 22;
    static constexpr int WINDOW_END   = 28;
    static constexpr int WINDOW_MAX_WORDS = 18;
    static constexpr int EXT_SHARED_ROW = 24;
    static constexpr int EXT_SHARED_END = 26;
    static constexpr int EXT_SHARED_WORDS = EXT_SHARED_END - EXT_SHARED_ROW;
    static constexpr int EXT_FIRST = 24;
    static constexpr int EXT_END   = 46;
    // v1: rows 24..43 (ext cols 46..65) are captured from the staged shmem
    // image; rows 44/45 (cols 66/67) are NEVER staged -- the fixed 66-word
    // block cannot hold them -- and are captured straight from global
    // memory (capture_deep_ext_gmem) when p >= 45.
    static constexpr int EXT_SHM_CAPTURE_END = 44;
    // L2-ext hybrid (smid-slice staging, sized 2026-07-13 from a profile of
    // a p=44 configuration dominated by per-launch spill traffic and
    // long_scoreboard stalls):
    // the DEEP rows' immutable ext values (rows 28..45, the coldest 18 of
    // the 20 capture registers) move to a per-SM-recycled device slice,
    // consumed through a depth-2 register prefetch ring re-seeded by rows
    // 26/27 each iteration. Rows 24..27 stay on-chip (shared_extension_ +
    // extension_[0..1]). Frees ~18 words/thread of always-live capture
    // state against the STACK-464 spill frame. 49-sector padded plane
    // stride + odd slice walk (ablation-verified layout).
#ifndef LDPC2_A204_EXT_DEEP_FIRST
#define LDPC2_A204_EXT_DEEP_FIRST 40
#endif
    static constexpr int    EXT_DEEP_FIRST      = LDPC2_A204_EXT_DEEP_FIRST;
    static constexpr int    A204_EXT_PAD_WORDS  = 8;   // +1 32B sector per plane
    static constexpr int    A204_EXT_STRIDE     = 384 + A204_EXT_PAD_WORDS;
    static constexpr int    A204_DEEP_PLANES    = EXT_END - EXT_DEEP_FIRST; // 18
    // %smid is a PHYSICAL SM id and indexes the scratch DIRECTLY (no modulo),
    // so A204_SMID_SLOTS must be >= the physical SM count of any GPU this
    // kernel runs on. 256 is the ASSUMED MAXIMUM SM count across those GPUs;
    // only accessed slices are L2-resident, so a larger slot count costs DRAM,
    // not L2. Any device reporting more SMs is declined in can_decode_config
    // (a modulo would silently alias two SMs onto one slice -> data race).
    static constexpr int    A204_SMID_SLOTS     = 256;
    // +1 sector so the slice-to-slice walk stays odd (18*49 = 882 even).
    static constexpr size_t A204_SLICE_BYTES =
        static_cast<size_t>(A204_DEEP_PLANES) * A204_EXT_STRIDE * sizeof(word_t) + 32;
    // Total shmem columns available for initial LLR staging (fixed block).
    static constexpr int COMPACT_APP_NODES2_STAGE = 66;
    static constexpr int A204_EXT_REG_CAPTURE_END = EXT_DEEP_FIRST; // rows 24..27 on-chip

    template <int ROW>
    struct bg1_z256up_bp_gmext_x2_window_row_words
    {
        static constexpr int value =
            (ROW == 24) ? 3 :
            (ROW == 26) ? 0 :
            update_row_degree<1, ROW>::value;
    };

    template <int ROW, int CUR = WINDOW_FIRST>
    struct bg1_z256up_bp_gmext_x2_window_prefix
    {
        static constexpr int value = bg1_z256up_bp_gmext_x2_window_row_words<CUR>::value +
                                     bg1_z256up_bp_gmext_x2_window_prefix<ROW, CUR + 1>::value;
    };
    template <int ROW>
    struct bg1_z256up_bp_gmext_x2_window_prefix<ROW, ROW>
    {
        static constexpr int value = 0;
    };

    CUDA_BOTH_INLINE int bg1_z256up_bp_gmext_x2_window_words(int num_parity_nodes)
    {
        const int end = (num_parity_nodes < WINDOW_END) ? num_parity_nodes : WINDOW_END;
        int words = 0;
        for(int row = WINDOW_FIRST; row < end; ++row)
        {
            words += (row == 24 || row == 27) ? 3 :
                     (row == 26) ? 0 : 4;
        }
        return words;
    }

    //------------------------------------------------------------------
    // Capture the isolated extension channels once, then reuse their former
    // APP columns as a rows-22..30 word-major C2V window. Extension state is
    // read-only after construction; moved C2V state avoids a local store on
    // every iteration as well as its following load.
    template <class TC2V,
              class TStorageCore,
              class TStorageNonCore,
              class TKernelParams>
    struct c2v_cache_bg1_z256up_bp_gmext_x2_window
    {
        typedef TC2V                  c2v_t;
        typedef typename c2v_t::app_t app_t;
        typedef TStorageCore          c2v_storage_core_t;
        typedef TStorageNonCore       c2v_storage_noncore_t;

        static_assert(bg1_z256up_bp_gmext_x2_window_prefix<WINDOW_END>::value == WINDOW_MAX_WORDS,
                      "rows 22..25 and 27 shared window size");
        static_assert(sizeof(c2v_storage_core_t) <= 4 * sizeof(word_t),
                      "ALGO204 expects register-resident min-sum core state");

        __device__
        c2v_cache_bg1_z256up_bp_gmext_x2_window(char* smem, const TKernelParams& params)
        {
            setup_window(smem);
            capture_extensions<EXT_FIRST>(smem, params.num_parity_nodes);
            init_register_c2v();
            init_window(params.num_parity_nodes, params.Z);
        }

        __device__ void init() {}

        // Attach this CTA's ext slice (lane-adjusted) after construction and
        // BEFORE the first do_iteration(). The slice is staged by the kernel
        // body (from the already-staged shmem columns) before sched
        // construction, so values are ready.
        __device__ __forceinline__
        void attach_ext_slice(const word_t* slice_lane)
        {
            ext_slice_lane_ = slice_lane;
        }

        // v1 (p45/46 support): capture rows 44/45's isolated extension
        // channels directly from global memory with the loader's
        // interleave+clamp transform (bit-identical to what staging
        // through shared memory would have produced). Called from the
        // kernel body after construction, only when p >= 45.
        __device__ __forceinline__
        void capture_deep_ext_gmem(const __half* src0,
                                   const __half* src1,
                                   bool          paired,
                                   __half2       clamp_value,
                                   int           num_parity_nodes)
        {
            (void)src0; (void)src1; (void)paired; (void)clamp_value;
            (void)num_parity_nodes; // slice staging happens in the kernel body
        }


        template <int CHECK_IDX, bool IS_FIRST = false, bool IS_LAST = false,
                  int NUM_APP_WORDS, int ROW_DEGREE>
        __device__
        void process_row(const TKernelParams& params,
                         word_t               (&app)[NUM_APP_WORDS],
                         int                  (&app_addr)[ROW_DEGREE],
                         int                  smem_offset)
        {
            static_assert(ROW_DEGREE == row_degree<1, CHECK_IDX>::value,
                          "APP address size incorrect for row degree");
            c2v_t c2v;
            if constexpr(CHECK_IDX < 4)
            {
                c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t,
                                         IS_FIRST, IS_LAST>(params, app, app_addr,
                                                            core_[CHECK_IDX], smem_offset);
            }
            else if constexpr(CHECK_IDX < WINDOW_FIRST)
            {
                auto& st = before_.template get<CHECK_IDX>();
                using st_type = std::remove_reference_t<decltype(st)>;
                c2v.template process_row<CHECK_IDX, TKernelParams, st_type,
                                         IS_FIRST, IS_LAST>(params, app, app_addr,
                                                            st, smem_offset);
            }
            else if constexpr(CHECK_IDX < WINDOW_END && CHECK_IDX != 26)
            {
                constexpr int WOFF = bg1_z256up_bp_gmext_x2_window_prefix<CHECK_IDX>::value;
                if constexpr(CHECK_IDX == 24)
                {
                    cC2V_storage_x2_low_degree st;
                    st.min0 = window_[(WOFF + 0) * A204_ZSTRIDE];
                    st.min1 = window_[(WOFF + 1) * A204_ZSTRIDE];
                    st.signs_0_9_min0_index = window_[(WOFF + 2) * A204_ZSTRIDE].u32;
                    process_noncore<CHECK_IDX, IS_FIRST, IS_LAST>(params, app, app_addr,
                                                                   st, smem_offset, c2v);
                    window_[(WOFF + 0) * A204_ZSTRIDE] = st.min0;
                    window_[(WOFF + 1) * A204_ZSTRIDE] = st.min1;
                    window_[(WOFF + 2) * A204_ZSTRIDE].u32 = st.signs_0_9_min0_index;
                }
                else
                {
                    constexpr int WORDS = update_row_degree<1, CHECK_IDX>::value;
                    typename c2v_storage_noncore_t::template resize<WORDS> st;
                    #pragma unroll
                    for(int i = 0; i < WORDS; ++i)
                    {
                        st.w[i] = window_[(WOFF + i) * A204_ZSTRIDE];
                    }
                    if constexpr(CHECK_IDX < EXT_FIRST)
                    {
                        c2v.template process_row<CHECK_IDX, TKernelParams, decltype(st),
                                                 IS_FIRST, IS_LAST>(params, app, app_addr,
                                                                    st, smem_offset);
                    }
                    else
                    {
                        process_noncore<CHECK_IDX, IS_FIRST, IS_LAST>(params, app, app_addr,
                                                                       st, smem_offset, c2v);
                    }
                    #pragma unroll
                    for(int i = 0; i < WORDS; ++i)
                    {
                        window_[(WOFF + i) * A204_ZSTRIDE] = st.w[i];
                    }
                }
            }
            else if constexpr(CHECK_IDX == 26)
            {
                auto& st = middle_.template get<CHECK_IDX>();
                using st_type = std::remove_reference_t<decltype(st)>;
                process_noncore<CHECK_IDX, IS_FIRST, IS_LAST>(params, app, app_addr,
                                                               st, smem_offset, c2v);
            }
            else if constexpr(CHECK_IDX < 46)
            {
                auto& st = after_.template get<CHECK_IDX>();
                using st_type = std::remove_reference_t<decltype(st)>;
                process_noncore<CHECK_IDX, IS_FIRST, IS_LAST>(params, app, app_addr,
                                                               st, smem_offset, c2v);
            }
            else
            {
                using dead_storage_t = typename c2v_storage_noncore_t::template resize<
                    update_row_degree<1, CHECK_IDX>::value>;
                dead_storage_t st;
                c2v.template process_row<CHECK_IDX, TKernelParams, dead_storage_t,
                                         IS_FIRST, IS_LAST>(params, app, app_addr,
                                                            st, smem_offset);
            }
        }

    private:
        template <int CHECK_IDX, bool IS_FIRST, bool IS_LAST,
                  int NUM_APP_WORDS, int ROW_DEGREE, class TStorage>
        __device__ __forceinline__
        void process_noncore(const TKernelParams& params,
                             word_t               (&app)[NUM_APP_WORDS],
                             int                  (&app_addr)[ROW_DEGREE],
                             TStorage&            st,
                             int                  smem_offset,
                             c2v_t&               c2v)
        {
            constexpr int URD = update_row_degree<1, CHECK_IDX>::value;
            static_assert(NUM_APP_WORDS == URD + 1,
                          "BG1 non-core row has one isolated extension edge");
            if constexpr(CHECK_IDX < EXT_SHARED_END)
            {
                app[URD] = shared_extension_[(CHECK_IDX - EXT_SHARED_ROW) * A204_ZSTRIDE];
            }
            else if constexpr(CHECK_IDX < EXT_DEEP_FIRST)
            {
                app[URD] = extension_[CHECK_IDX - EXT_SHARED_END];
                // Ring seeds: the two on-chip rows immediately before the
                // deep band issue the slice loads for the first two deep
                // consumers every iteration -- a two-row shadow.
                if constexpr(CHECK_IDX >= (EXT_DEEP_FIRST - 2))
                {
                    ext_pf_[CHECK_IDX - (EXT_DEEP_FIRST - 2)] =
                        ext_slice_lane_[(CHECK_IDX + 2 - EXT_DEEP_FIRST) * A204_EXT_STRIDE];
                }
            }
            else
            {
                // Deep rows: consume the ring slot, refill 2 rows ahead.
                // Band-max-sized slice: a refill for a row beyond runtime p
                // reads in-bounds garbage nothing consumes.
                app[URD] = ext_pf_[(CHECK_IDX - EXT_DEEP_FIRST) & 1];
                if constexpr((CHECK_IDX + 2) < EXT_END)
                {
                    ext_pf_[(CHECK_IDX - EXT_DEEP_FIRST) & 1] =
                        ext_slice_lane_[(CHECK_IDX + 2 - EXT_DEEP_FIRST) * A204_EXT_STRIDE];
                }
            }
            #pragma unroll
            for(int i = 0; i < URD; ++i)
            {
                app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
            }
            static_assert(vnode_index<1, CHECK_IDX, URD>::value == 22 + CHECK_IDX,
                          "BG1 extension column must be the final edge");
            c2v.template compute_app<CHECK_IDX, TKernelParams, TStorage,
                                     IS_FIRST, IS_LAST,
                                     ((CHECK_IDX >= 29) && ((CHECK_IDX & 1) != 0))>(params, app, st);
            c2v.template write_app<CHECK_IDX>(app, app_addr, smem_offset);
        }

        // DESCENDING capture (row EXT_END-1 down to EXT_FIRST): at p >= 43
        // the initial LLR staging of columns 64/65 (rows 42/43's isolated
        // extensions) occupies the same shmem slots that shared_extension_
        // (rows 24/25) is later written to. Capturing deep rows FIRST reads
        // the staged values before the shared_extension_ writes clobber
        // them; all accesses are same-lane, so in-thread ordering suffices
        // (no barrier needed). Ascending capture corrupted rows 42/43's
        // ext at p=43/44 (llr pkerr 0.85, BLER 7884/10000 at p44).
        template <int ROW>
        __device__ __forceinline__
        void capture_extensions(char* smem, int num_parity_nodes)
        {
            if constexpr((ROW + 1) < A204_EXT_REG_CAPTURE_END)
            {
                capture_extensions<ROW + 1>(smem, num_parity_nodes);
            }
            if(ROW < num_parity_nodes)
            {
                const word_t* app = reinterpret_cast<const word_t*>(smem);
                if constexpr(ROW < EXT_SHARED_END)
                {
                    shared_extension_[(ROW - EXT_SHARED_ROW) * A204_ZSTRIDE] =
                        app[(22 + ROW) * A204_ZSTRIDE + threadIdx.x];
                }
                else
                {
                    extension_[ROW - EXT_SHARED_END] =
                        app[(22 + ROW) * A204_ZSTRIDE + threadIdx.x];
                }
            }
        }

        __device__ void setup_window(char* smem)
        {
            window_ = reinterpret_cast<word_t*>(smem) +
                      COMPACT_APP_NODES * A204_ZSTRIDE + threadIdx.x;
            shared_extension_ = window_ + WINDOW_MAX_WORDS * A204_ZSTRIDE;
        }

        __device__ void init_register_c2v()
        {
            #pragma unroll
            for(int i = 0; i < 4; ++i) core_[i].init();
            before_.init();
            middle_.init();
            after_.init();
        }

        __device__ void init_window(int num_parity_nodes, int /*Z*/)
        {
            const int words = bg1_z256up_bp_gmext_x2_window_words(num_parity_nodes);
            for(int i = 0; i < words; ++i)
            {
                window_[i * A204_ZSTRIDE].u32 = 0;
            }
        }

        c2v_storage_core_t core_[4];
        noncore_reg_chain<1, 4, WINDOW_FIRST, c2v_storage_noncore_t> before_;
        noncore_reg_chain<1, 26, 27, c2v_storage_noncore_t> middle_;
        noncore_reg_chain<1, WINDOW_END, 46, c2v_storage_noncore_t> after_;
        word_t* window_;
        word_t* shared_extension_;
        word_t        extension_[EXT_DEEP_FIRST - EXT_SHARED_END]; // rows 26/27 only
        word_t        ext_pf_[2];                                  // deep-row prefetch ring
        const word_t* ext_slice_lane_;
    };

    //------------------------------------------------------------------
    // ALGO204's bounded BG1 schedule.  The generic dynamic schedule makes
    // every possible runtime-last row conditionally perform the iteration
    // barrier before returning.  Here runtime-last rows return to one common
    // tail barrier.  Rows 0..20 have no row-existence test (p >= 22), while
    // rows 21..33 retain the shared runtime-p termination chain.
    //
    // This is bit-equivalent: there is no work between a runtime-last row's
    // return and the common barrier, and non-last rows retain exactly the
    // row_seq_sync barrier proved by the checked-in BG1 descriptors.  It also
    // remains one runtime-p body with no p-specific kernel instantiations.
    template <int                  BG,
              class                TAPPLoc,
              class                TC2VCache,
              class                TKernelParams,
              class                BGDesc,
              int                  MIN_PARITY_ROWS,
              int                  MAX_PARITY_ROWS>
    struct bg1_z256up_bp_gmext_x2_schedule;

    template <class TAPPLoc,
              class TC2VCache,
              class TKernelParams,
              class BGDesc,
              int   MIN_PARITY_ROWS,
              int   MAX_PARITY_ROWS>
    struct bg1_z256up_bp_gmext_x2_schedule<1,
                            TAPPLoc,
                            TC2VCache,
                            TKernelParams,
                            BGDesc,
                            MIN_PARITY_ROWS,
                            MAX_PARITY_ROWS> :
        ldpc2::ldpc_schedule_dynamic_desc_base<1,
                                               TAPPLoc,
                                               TC2VCache,
                                               TKernelParams,
                                               BGDesc,
                                               MIN_PARITY_ROWS,
                                               MAX_PARITY_ROWS>
    {
        typedef ldpc2::ldpc_schedule_dynamic_desc_base<1,
                                                        TAPPLoc,
                                                        TC2VCache,
                                                        TKernelParams,
                                                        BGDesc,
                                                        MIN_PARITY_ROWS,
                                                        MAX_PARITY_ROWS> inherited_t;
        typedef BGDesc bg_desc_t;

        static_assert(MIN_PARITY_ROWS == 22, "ALGO204 BG1 prefix is rows 0..21");
        static_assert(MAX_PARITY_ROWS == 46, "ALGO204 BG1 suffix ends at row 45");

        __device__
        bg1_z256up_bp_gmext_x2_schedule(char*                smem,
                         const TKernelParams& params,
                         const bg_desc_t&     bg_desc,
                         int                  soffset,
                         unsigned int         t_idx) :
            inherited_t(smem, params, bg_desc, soffset, t_idx)
        {
        }

        // The first optional pair (22,23) stays on the generic termination
        // form.  Keeping that short low-p path compact avoids extending the
        // common-tail state through rows that p=22..24 never reach.
        static constexpr int COMMON_TAIL_FIRST_ROW = 24;

        template <int ROW>
        __device__ __forceinline__
        void process_common_tail()
        {
            (*this).template process_row<ROW>();
            if constexpr((ROW + 1) < MAX_PARITY_ROWS)
            {
                const bool is_last =
                    ((ROW + 1) == (*this).params.num_parity_nodes);
                if(!is_last)
                {
                    if constexpr(ldpc2::row_seq_sync<1, ROW>::value)
                    {
                        __syncthreads();
                    }
                    process_common_tail<ROW + 1>();
                }
            }
            // ROW + 1 == MAX_PARITY_ROWS: fall through to the common tail.
        }

        __device__ __forceinline__
        void do_iteration()
        {
            (*this).template process_row<0> (); __syncthreads();
            (*this).template process_row<1> (); __syncthreads();
            (*this).template process_row<2> (); __syncthreads();
            (*this).template process_row<3> (); if((*this).template iter_sync_check_done<3> ()) return;
            (*this).template process_row<4> (); if((*this).template iter_sync_check_done<4> ()) return;
            (*this).template process_row<5> (); if((*this).template iter_sync_check_done<5> ()) return;
            (*this).template process_row<6> (); if((*this).template iter_sync_check_done<6> ()) return;
            (*this).template process_row<7> (); if((*this).template iter_sync_check_done<7> ()) return;
            (*this).template process_row<8> (); if((*this).template iter_sync_check_done<8> ()) return;
            (*this).template process_row<9> (); if((*this).template iter_sync_check_done<9> ()) return;
            (*this).template process_row<10>(); if((*this).template iter_sync_check_done<10>()) return;
            (*this).template process_row<11>(); if((*this).template iter_sync_check_done<11>()) return;
            (*this).template process_row<12>(); if((*this).template iter_sync_check_done<12>()) return;
            (*this).template process_row<13>(); if((*this).template iter_sync_check_done<13>()) return;
            (*this).template process_row<14>(); if((*this).template iter_sync_check_done<14>()) return;
            (*this).template process_row<15>(); if((*this).template iter_sync_check_done<15>()) return;
            (*this).template process_row<16>(); if((*this).template iter_sync_check_done<16>()) return;
            (*this).template process_row<17>(); if((*this).template iter_sync_check_done<17>()) return;
            (*this).template process_row<18>(); if((*this).template iter_sync_check_done<18>()) return;
            (*this).template process_row<19>(); if((*this).template iter_sync_check_done<19>()) return;
            (*this).template process_row<20>(); if((*this).template iter_sync_check_done<20>()) return;
            (*this).template process_row<21>(); if((*this).template iter_sync_check_done<21>()) return;
            (*this).template process_row<22>(); if((*this).template iter_sync_check_done<22>()) return;
            (*this).template process_row<23>(); if((*this).template iter_sync_check_done<23>()) return;
            process_common_tail<COMMON_TAIL_FIRST_ROW>();
            __syncthreads();
        }
    };

    // Unreachable under ALGO204's zone gate, but retain the BG2 entry points
    // and their original generic schedule so forced symbol resolution/builds
    // remain valid.
    template <class TAPPLoc,
              class TC2VCache,
              class TKernelParams,
              class BGDesc,
              int   MIN_PARITY_ROWS,
              int   MAX_PARITY_ROWS>
    struct bg1_z256up_bp_gmext_x2_schedule<2,
                            TAPPLoc,
                            TC2VCache,
                            TKernelParams,
                            BGDesc,
                            MIN_PARITY_ROWS,
                            MAX_PARITY_ROWS> :
        ldpc2::ldpc_schedule_dynamic_desc<2,
                                          TAPPLoc,
                                          TC2VCache,
                                          TKernelParams,
                                          BGDesc,
                                          MIN_PARITY_ROWS,
                                          MAX_PARITY_ROWS>
    {
        typedef ldpc2::ldpc_schedule_dynamic_desc<2,
                                                  TAPPLoc,
                                                  TC2VCache,
                                                  TKernelParams,
                                                  BGDesc,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> inherited_t;
        typedef BGDesc bg_desc_t;

        __device__
        bg1_z256up_bp_gmext_x2_schedule(char*                smem,
                         const TKernelParams& params,
                         const bg_desc_t&     bg_desc,
                         int                  soffset,
                         unsigned int         t_idx) :
            inherited_t(smem, params, bg_desc, soffset, t_idx)
        {
        }
    };

    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    template <int   BG_,           // base graph (1 or 2)
              class TKernelParams, // struct with kernel params
              int   MIN_P_>        // compile-time floor on parity rows
    struct ldpc2_bg1_z256up_bp_gmext_x2_kernel_config
    {
        static constexpr int BG                  = BG_;
        static constexpr int MIN_PARITY_ROWS     = MIN_P_;
        static constexpr int NUM_REG_PARITY_ROWS = num_reg_parity<BG>::value;
        static constexpr int MAX_PARITY_ROWS     = max_num_parity<BG>::value;

        typedef TKernelParams                           kernel_params_t;

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // cC2V_row_map_t
        // Hybrid row map: dispatches to box_plus for non-core rows
        // (which use c2v_storage_x2_box_plus), and compressed min-sum
        // for core rows (register storage).
        template <int   BG,
                  int   CHECK_IDX,
                  class TC2VStorage> using cC2V_row_map_t = hybrid_storage_row_map_x2<BG,
                                                                                       CHECK_IDX,
                                                                                       TC2VStorage,
                                                                                       __half2,
                                                                                       row_context_t,
                                                                                       cC2V_row_proc_t>;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V row dispatch type: uses the row map to determine which
        // C2V processor to call for each row.
        typedef C2V_row_proc<__half2,
                             BG,
                             cC2V_row_map_t,
                             app_loader,
                             app_writer> C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // BG1 moves rows 22/23 to the shared window.
        // The unreachable
        // BG2 symbols retain the generic cache for legacy resolution.
        typedef std::conditional_t<
            (BG == 1),
            c2v_cache_bg1_z256up_bp_gmext_x2_window<C2V_t,
                                      typename core_storage_x2<BG>::type,
                                      typename noncore_storage_x2_local<BG>::type,
                                      kernel_params_t>,
            ldpc2::c2v_cache_split<BG,
                                    NUM_REG_PARITY_ROWS,
                                    C2V_t,
                                    typename core_storage_x2<BG>::type,
                                    typename noncore_storage_x2_local<BG>::type,
                                    kernel_params_t>> c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef ldpc2::llr_loader_variable_batch<__half2, 6, llr_op_clamp> llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef llr_loader_t::app_buf_t                                    app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef bg1_z256up_bp_gmext_x2_schedule<BG,
                                 app_loc_t<BG>,
                                 c2v_cache_t,
                                 kernel_params_t,
                                 typename app_loc_t<BG_>::bg_desc_t,
                                 MIN_PARITY_ROWS,
                                 MAX_PARITY_ROWS> sched_t;
    };
    //------------------------------------------------------------------
    // Compile-time parity floors: BG1 kernels are high-p specialized
    // (band gate guarantees p >= 22); the unreachable BG2 kernels keep
    // the generic floor.
    const int BG1_MIN_P = 22;
    //------------------------------------------------------------------
    // get_app_c2v_shmem()
    // BG1 reuses optional extension columns for the fixed rows-22..25
    // word-major C2V window. The initial full APP staging image (at most 56
    // columns) also fits inside this 63-column allocation.
    template <int BG>
    CUDA_BOTH
    int get_app_c2v_shmem(int num_parity_nodes, int Z)
    {
        typedef typename noncore_storage_x2_local<BG>::type noncore_storage_t;

        constexpr int32_t NUM_REG_NODES = num_reg_parity<BG>::value;
        const int32_t NUM_APP_NODES = (BG == 1) ? COMPACT_APP_NODES :
                                                  (ldpc2::max_info_nodes<BG>::value + num_parity_nodes);
        int32_t offset = static_cast<int32_t>(shmem_llr_buffer_size(NUM_APP_NODES,
                                                                    Z,
                                                                    sizeof(__half2)));

        int32_t C2V_SIZE = 0;
        if constexpr(BG == 1)
        {
            C2V_SIZE = (WINDOW_MAX_WORDS + EXT_SHARED_WORDS) * Z * sizeof(word_t);
        }
        else
        {
            C2V_SIZE = (num_parity_nodes > NUM_REG_NODES) ?
                       (num_parity_nodes - NUM_REG_NODES) * Z * sizeof(noncore_storage_t) : 0;
        }
        // Pad for non-core C2V alignment
        int shmem_size = round_up_to_next(offset, static_cast<int>(alignof(noncore_storage_t))) +
                                          C2V_SIZE;
        return shmem_size;
    }
    //------------------------------------------------------------------
    // Shared-memory budget proof (algo202's convention). This layout comes to
    // EXACTLY GB203's 99 KiB (101376 B) opt-in limit at Z=384:
    //   (COMPACT_APP_NODES 46 + WINDOW_MAX_WORDS 18 + EXT_SHARED_WORDS 2)
    //     * 384 * 4 = 101376
    // i.e. zero slack. There is no runtime diagnostic for overrunning it --
    // can_decode_config() simply starts returning false and the whole band goes
    // silently undecodable. That is exactly what the unconditional 1104-byte ET
    // context did: four kernels vanished
    // and it took a full phase to find. Fail at COMPILE time instead.
    static_assert((COMPACT_APP_NODES + WINDOW_MAX_WORDS + EXT_SHARED_WORDS) *
                  384 * static_cast<int>(sizeof(word_t)) <= 99 * 1024,
                  "BG1 APP+C2V layout exceeds GB203 opt-in shmem -- this band "
                  "will silently stop being selectable at every p");

    //------------------------------------------------------------------
    // a204_z_supported()
    // The Z zone. Tied to A204_RUNTIME_Z_ADDR because the addressing and the
    // shared-memory strides must agree: with compile-time-384 addressing this
    // kernel is only correct at Z=384. The zone is the 38.212 liftings for
    // which the 1-CTA/SM, blockDim==Z design is sensible (below ~256 the SM
    // holds too few threads), all multiples of 32 as the all-warps hard-output
    // writer requires. Mirrors a115_z_supported().
    bool a204_z_supported(int Z)
    {
#if A204_RUNTIME_Z_ADDR
        return (Z == 256) || (Z == 288) || (Z == 320) || (Z == 352) || (Z == 384);
#else
        return (Z == 384);
#endif
    }

    //------------------------------------------------------------------
    // get_shmem_required()
    // Calculates the sum of the APP and C2V data storage.
    int get_shmem_required(int BG,
                           int num_parity_nodes,
                           int Z)
    {
        // BG1 only: can_decode_config() rejects BG != 1 before any caller
        // reaches here, and the BG2 kernels this decoder used to compile were
        // unreachable for the same reason.
        (void)BG;
        int shmem_size = get_app_c2v_shmem<1>(num_parity_nodes, Z);
        return static_cast<int>(ldpc2::shmem_size_with_experimental_et_context(static_cast<uint32_t>(shmem_size)));
    }
#if LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // get_token_addr()
    // Returns the address of the tb_token value used to store information
    // about the specific codeword being processed by a CTA when the
    // transport block interface is used. The token is assumed to reside
    // immediately after the APP and C2V memory.
    template <int BG>
    __device__
    tb_token* get_token_addr(int num_parity_nodes, int Z, char* smem)
    {
        if constexpr(BG == 1)
        {
            return reinterpret_cast<tb_token*>(smem +
                get_app_c2v_shmem<BG>(num_parity_nodes, Z) - Z * sizeof(word_t));
        }
        else
        {
            return reinterpret_cast<tb_token*>(smem + get_app_c2v_shmem<BG>(num_parity_nodes, Z));
        }
    }
    template <int BG>
    __device__
    tb_token* get_token_addr(const cuphyLDPCDecodeDesc_t& decodeDesc,
                             char* smem)
    {
        return get_token_addr<BG>(decodeDesc.config.num_parity_nodes,
                                  decodeDesc.config.Z,
                                  smem);
    }
#endif // if LDPC_DECODE_USE_TB_SCAN
} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_z256up_bp_gmext_x2()
// Kernel for base graph 1 (legacy tensor interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_z256up_bp_gmext_x2(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_bg1_z256up_bp_gmext_x2_kernel_config<1,                         // BG
                                        ldpc2::LDPC_kernel_params, // params struct
                                        BG1_MIN_P> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values.
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// bg1_bg1_z256up_bp_gmext_x2_tb_body()
// Shared body for the BG1 transport-block kernels.
namespace
{
__device__ __forceinline__
void bg1_bg1_z256up_bp_gmext_x2_tb_body(const cuphyLDPCDecodeDesc_t& decodeDesc, const app_loc_t<1>::bg_desc_t& bgdesc, void* scratch)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_bg1_z256up_bp_gmext_x2_kernel_config<1,                           // BG
                                        cuphyLDPCDecodeConfigDesc_t, // params struct
                                        BG1_MIN_P> kernel_config_t;
#if !LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    //------------------------------------------------------------------
    // Pre-compute output params before iterations so the compiler can
    // release the large decodeDesc (TB arrays) during the decode loop.
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = bg1_z256up_bp_gmext_x2_find_token<2>(decodeDesc, blockIdx.x);
    ldpc_dec_loader_params<__half2> loader_params(smem, decodeDesc, loader_token(tok));
    // v1: the fixed 66-word shmem block stages at most 66 columns
    // (22+p <= 66 <=> p <= 44). For p=45/46 stage only columns 0..65;
    // rows 44/45's ext columns (66/67) are captured from gmem below.
    if(decodeDesc.config.num_parity_nodes > 44)
    {
        loader_params.num_cw_elements = COMPACT_APP_NODES2_STAGE * A204_ZSTRIDE;
    }
    kernel_config_t::llr_loader_t::load_sync(loader_params);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
#if LDPC_MICRO_PACK_ARGMIN
    // E5M5 norm compensation: the
    // packed-argmin scan truncates magnitudes toward zero; scale the norm
    // up to cancel the bias. In-kernel so it is idempotent across the
    // TB/graph-capture launch paths. 1.006 retained (= bg1_z256up_bp_shwin_x2's core-only
    // optimum). This band's llr-sweep optimum is 1.004 (2026-07-11: 1.000
    // -> 0/10, 1.002 -> 5/10, 1.004 -> 10/10 worst pkerr 0.048, 1.006 ->
    // 9/10 with only p41 in the marginal-pkerr class at 0.064, 1.008 ->
    // 7/10, 1.010+ collapses), but the two criteria DIVERGE here: at
    // 1.004 TB BLER degrades +11..15% at every recorded 10% point, while
    // at 1.006 it sits ~+3% off a40 parity. BLER (the production metric)
    // wins the tie; 1.006 is not a BLER-harvest retune above the llr
    // optimum, it is the family constant. bg1_z256up_bp_shwin_x2's original note:
    // --llr_check pkerr sweep (1.0127 and 1.016 over-compensate; 12/12 TV
    // PASS at 1.006, worst pkerr 0.046).
    //
    // ANTI-PATTERN -- do not retune this constant to harvest BLER.
    // Lowering or removing it makes the four degree-19 core rows run at an
    // effectively LOWER min-sum norm than the box-plus rows, which
    // measurably improves TB BLER (up to 2-3.7x at the 10%/1% operating
    // points) -- but it silently breaks --llr_check (the per-iteration APP
    // trajectory drifts from the MATLAB reference by a compounding bias)
    // and smuggles in an unreviewed degree-dependent-normalization design
    // change. This constant exists ONLY to cancel E5M5's truncation bias
    // so --llr_check stays valid. If degree-dependent normalization is
    // wanted, propose it as its own explicit, separately reviewed knob --
    // never by adjusting this constant.
    config.norm.f16x2 = __hmul2(config.norm.f16x2, __float2half2_rn(1.006f));
#endif

    #ifdef CUPHY_EXPERIMENTAL_LDPC_ET
        //------------------------------------------------------------------
        // Early-termination context (before per-iteration loop)
#if LDPC_DECODE_USE_TB_SCAN
        int size_before_et = get_app_c2v_shmem<kernel_config_t::BG>(decodeDesc.config.num_parity_nodes, decodeDesc.config.Z);
        size_before_et = static_cast<int>(round_up_to_next(static_cast<uint32_t>(size_before_et), static_cast<uint32_t>(alignof(tb_token))) + sizeof(tb_token));
#else
        int size_before_et = get_app_c2v_shmem<kernel_config_t::BG>(decodeDesc.config.num_parity_nodes, decodeDesc.config.Z);
#endif
        ldpc_et_context_t* et_ctx = ldpc2::get_et_ctx_ptr(smem, static_cast<uint32_t>(size_before_et));
        ldpc2::early_term_initialize(et_ctx, decodeDesc, blockIdx.x);
    #endif

    //------------------------------------------------------------------
    // Perform iterations
#if LDPC_DECODE_USE_TB_SCAN
    // Locate this CTA's ext slice (%smid-recycled: ~1 CTA/SM resident, so
    // the working set is bounded to A204_SMID_SLOTS slices, permanently
    // L2-resident) and stage the DEEP rows' ext values BEFORE sched
    // construction (whose window init clobbers the staged shmem columns).
    // Rows 28..43 (cols 50..65) copy from the already-staged shmem image;
    // rows 44/45 (cols 66/67, never staged) transform from gmem (p>=45).
    unsigned int a204_smid;
    asm("mov.u32 %0, %%smid;" : "=r"(a204_smid));
    word_t* a204_slice_lane =
        reinterpret_cast<word_t*>(static_cast<char*>(scratch) +
                                  static_cast<size_t>(a204_smid) * A204_SLICE_BYTES) +
        threadIdx.x;
    {
        const word_t* staged_app = reinterpret_cast<const word_t*>(smem);
        const int     last_row   = (config.num_parity_nodes < EXT_END ?
                                    config.num_parity_nodes : EXT_END) - 1;
        #pragma unroll
        for(int row = EXT_DEEP_FIRST; row < EXT_SHM_CAPTURE_END; ++row)
        {
            if(row <= last_row)
            {
                a204_slice_lane[(row - EXT_DEEP_FIRST) * A204_EXT_STRIDE] =
                    staged_app[(22 + row) * A204_ZSTRIDE + threadIdx.x];
            }
        }
        if(config.num_parity_nodes > 44)
        {
            const __half* dsrc0 = static_cast<const __half*>(loader_params.src_gmem);
            const __half* dsrc1 = dsrc0 + loader_params.src_stride_elements;
            const bool    dpair = (loader_params.max_cta_cw_index > 0);
            const __half2 dclmp = __float2half2_rn(loader_params.clamp_value);
            #pragma unroll
            for(int row = EXT_SHM_CAPTURE_END; row < EXT_END; ++row)
            {
                if(row <= last_row)
                {
                    const int e = (22 + row) * A204_ZSTRIDE + static_cast<int>(threadIdx.x);
                    const __half v0 = dsrc0[e];
                    const __half v1 = dpair ? dsrc1[e] : v0;
                    word_t w;
                    w.f16x2 = clamp_signed(__halves2half2(v0, v1), dclmp);
                    a204_slice_lane[(row - EXT_DEEP_FIRST) * A204_EXT_STRIDE] = w;
                }
            }
        }
    }
#endif
    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
#if LDPC_DECODE_USE_TB_SCAN
    sched.c2v_cache.attach_ext_slice(a204_slice_lane);
#else
#if LDPC_DECODE_USE_TB_SCAN
    if(config.num_parity_nodes > 44)
    {
        const __half* v1_src0 = static_cast<const __half*>(loader_params.src_gmem);
        const __half* v1_src1 = v1_src0 + loader_params.src_stride_elements;
        sched.c2v_cache.capture_deep_ext_gmem(v1_src0,
                                              v1_src1,
                                              (loader_params.max_cta_cw_index > 0),
                                              __float2half2_rn(loader_params.clamp_value),
                                              config.num_parity_nodes);
    }
#endif
#endif
    int32_t iter = 0;
    #ifdef CUPHY_EXPERIMENTAL_LDPC_ET
        uint32_t crc = 1;  // Assume fail initially
    #endif
    while(iter < config.max_iterations)
    {
        sched.do_iteration();
#if LDPC_DECODE_USE_TB_SCAN
#endif
        ++iter;

        #ifdef CUPHY_EXPERIMENTAL_LDPC_ET
            if(0 != (CUPHY_LDPC_DECODE_EARLY_TERM & decodeDesc.config.flags))
            {
                crc = ldpc2::should_terminate_early_crc<kernel_config_t::BG>(decodeDesc, blockIdx.x, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), et_ctx);
                if(crc == 0) break;
            }
        #endif
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values. All-warps single-pass
    // ballot-transpose writer. Its precondition is out_words_per_cw ==
    // WORDS_PER_WARP * (blockDim.x/32), which the band gate guarantees at
    // EVERY Z in the zone, not just at 384: Kb=22 gives (22*Z+31)/32 =
    // 22*(Z/32) output words, and blockDim.x == Z with Z % 32 == 0 gives 22
    // words/warp x (Z/32) warps. The Z % 32 term of a204_z_supported() is
    // what makes this hold -- it is a correctness gate for this writer, not a
    // tidiness rule.
    ldpc_dec_output_x2_all_warps<22>(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    #ifdef CUPHY_EXPERIMENTAL_LDPC_ET
        //------------------------------------------------------------------
        // Write CRC result if early termination enabled
        if(0 != (CUPHY_LDPC_DECODE_EARLY_TERM & decodeDesc.config.flags))
        {
            ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
        }
        //------------------------------------------------------------------
        // Write iteration count if requested
        if(0 != (CUPHY_LDPC_DECODE_WRITE_ITER_COUNT & decodeDesc.config.flags))
        {
            ldpc2::ldpc_dec_iter_output(decodeDesc, blockIdx.x, iter);
        }
    #endif
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}
} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_z256up_bp_gmext_x2_tb()
// Kernel for base graph 1 (transport block interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_z256up_bp_gmext_x2_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc, void* scratch)
{
    bg1_bg1_z256up_bp_gmext_x2_tb_body(decodeDesc, bgdesc, scratch);
}


namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::decode()
cuphyStatus_t bg1_z256up_bp_gmext_x2::decode(ldpc::decoder&                     dec,
                              LDPC_output_t&                     tDst,
                              const_tensor_pair&                 tLLR,
                              const cuphy_optional<tensor_pair>& optSoftOutputs,
                              const cuphyLDPCDecodeConfigDesc_t& config,
                              cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_bg1_z256up_bp_gmext_x2()\n");
    //------------------------------------------------------------------
    // Same gate the other three entry points apply. Without it this path
    // would accept a configuration the kernel cannot serve -- in particular
    // Kb != 22, which the hard-decision writer (instantiated as
    // <CFG_OUT_WORDS_PER_WARP = Kb>, writing 22*Z bits regardless) would
    // turn into an out-of-bounds store into a buffer the caller sized from
    // a smaller Kb.
    if(!can_decode_config(dec, config))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    // Narrower still on this path: p=45/46 are TB-interface only (the tensor-path kernels stage
    // all 22+p columns through the fixed 66-word shmem block, which
    // holds at most 66; the TB body's gmem ext capture is not plumbed
    // into this path).
    if(config.num_parity_nodes > 44)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(div_round_up(NUM_CW, 2));
    // We need to be mindful of the blockDim not being a multiple of 32.
    // The hard decision output writes 32-bit words. We may need to
    // revisit the output function to allow us to truncate the threads
    // that write to the next lowest multiple of 32, but that  may also
    // mean that we need to then have the output function LOOP.
    //dim3 blkDim(((config.Z + 31) / 32) * 32);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, optSoftOutputs);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = get_shmem_required(config.BG,
                                                   config.num_parity_nodes,
                                                   config.Z);

    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_z256up_bp_gmext_x2, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_z256up_bp_gmext_x2<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
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
// bg1_z256up_bp_gmext_x2::decode_tb()
cuphyStatus_t bg1_z256up_bp_gmext_x2::decode_tb(ldpc::decoder&               dec,
                                 const cuphyLDPCDecodeDesc_t& decodeDesc,
                                 cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::bg1_z256up_bp_gmext_x2::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    if(can_decode_config(dec, decodeDesc.config))
    {
        // We need to be mindful of the blockDim not being a multiple of 32.
        // The hard decision output writes 32-bit words. We may need to
        // revisit the output function to allow us to truncate the threads
        // that write to the next lowest multiple of 32, but that  may also
        // mean that we need to then have the output function LOOP.
        //dim3 blkDim(((config.Z + 31) / 32) * 32);
        dim3 blkDim(decodeDesc.config.Z);

        //------------------------------------------------------------------
        // Launch a CTA for each codeword pair. Note that the number of CTAs
        // may be more than the total number of codewords divided by 2 -
        // there may be transport blocks with odd numbers of codewords.
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        //------------------------------------------------------------------
        // Determine the dynamic amount of shared memory
        const uint32_t SHMEM_SIZE = get_shmem_required(decodeDesc.config.BG,
                                                       decodeDesc.config.num_parity_nodes,
                                                       decodeDesc.config.Z);
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                cuphyStatus_t sw = ensure_scratch(static_cast<size_t>(A204_SMID_SLOTS) * A204_SLICE_BYTES);
                if(CUPHY_STATUS_SUCCESS != sw) { s = sw; break; }
                //------------------------------------------------------------------
                // Launch the kernel
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_z256up_bp_gmext_x2_tb, blkDim, SHMEM_SIZE);
                    ldpc2_BG1_z256up_bp_gmext_x2_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc, m_scratch_d);
                }
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
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
// bg1_z256up_bp_gmext_x2::get_workspace_size()
std::pair<bool, size_t> bg1_z256up_bp_gmext_x2::get_workspace_size(const ldpc::decoder&               dec,
                                                    const cuphyLDPCDecodeConfigDesc_t& config,
                                                    int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::bg1_z256up_bp_gmext_x2()
bg1_z256up_bp_gmext_x2::bg1_z256up_bp_gmext_x2(ldpc::decoder& dec)
{
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = get_shmem_required(1,                             // BG
                                                      max_num_parity<1>::value,      // max parity nodes
                                                      CUPHY_LDPC_MAX_LIFTING_SIZE); // lifting size  // lifting size
    //------------------------------------------------------------------
    // Maximum shared memory supported by the device
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 2> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_z256up_bp_gmext_x2,    std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_z256up_bp_gmext_x2_tb, std::min(MAX_BG1_SHMEM_SIZE, MAX_SHMEM)),
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
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_z256up_bp_gmext_x2);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_z256up_bp_gmext_x2_tb);
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::can_decode_config()
bool bg1_z256up_bp_gmext_x2::can_decode_config(const ldpc::decoder&               dec,
                                const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Band gate: BG1 / p=35..46 / fp16 only (see the band specialization
    // note at the top of this file). Z is the zone when the kernel is built
    // Z-generic, otherwise pinned at 384.
    // Kb == 22 is a precondition, not a preference: the hard decision writer
    // is ldpc_dec_output_x2_all_warps<22>, and because blockDim == Z its
    // words-per-warp template argument IS Kb. It writes 22 * Z bits whatever
    // cfg.Kb says, so a BG1 descriptor with a smaller Kb would overrun an
    // output buffer the caller sized from it.
    if((1 != cfg.BG)                                    ||
       !a204_z_supported(cfg.Z)                         ||
       (22 != cfg.Kb)                                   ||
       (CUPHY_R_16F != cfg.llr_type)                    ||
       (cfg.num_parity_nodes < 35)                      ||
       (cfg.num_parity_nodes > max_num_parity<1>::value))
    {
        return false;
    }
    // Calculate required shared memory (APP only in this band) and
    // compare to the device maximum.
    const uint32_t SHMEM_BYTES = get_shmem_required(cfg.BG,
                                                    cfg.num_parity_nodes,
                                                    cfg.Z);
    bool ok = (SHMEM_BYTES <= dec.max_shmem_per_block_optin());
    // The %smid-indexed ext scratch has A204_SMID_SLOTS slices (the assumed
    // max SM count). Decline any device that reports more SMs so the
    // dispatcher falls back rather than the kernel writing out of bounds.
    ok = ok && (dec.sm_count() <= A204_SMID_SLOTS);
    return ok;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::get_launch_config()
cuphyStatus_t bg1_z256up_bp_gmext_x2::get_launch_config(const ldpc::decoder&           dec,
                                         cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    if(!can_decode_config(dec, launchConfig.decode_desc.config))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    //------------------------------------------------------------------
    // Validate input arguments (band gate: BG1 / p=35..46)
    if((1 != BG)                                        ||
       !a204_z_supported(Z)                             ||
       (NUM_PARITY_NODES < 35)                          ||
       (NUM_PARITY_NODES > max_num_parity<1>::value))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;

    const uint32_t SHMEM_SIZE = get_shmem_required(launchConfig.decode_desc.config.BG,
                                                   launchConfig.decode_desc.config.num_parity_nodes,
                                                   launchConfig.decode_desc.config.Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const void* bg1_kernel = (const void*)ldpc2_BG1_z256up_bp_gmext_x2_tb;
    cudaError_t    e = cudaGetFuncBySymbol(&deviceFunction, bg1_kernel);
    if (e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif
    //------------------------------------------------------------------
    // Set kernel arguments:
    // arg 0: decode descriptor
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    // arg 1: base graph descriptor
    if(1 == BG)
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
        // arg 2: smid-sliced ext scratch (graph-capture path must have it
        // resident before capture; allocation is idempotent).
        cuphyStatus_t sw = ensure_scratch(static_cast<size_t>(A204_SMID_SLOTS) * A204_SLICE_BYTES);
        if(CUPHY_STATUS_SUCCESS != sw) { return sw; }
        launchConfig.kernel_args[2] = &m_scratch_d;
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::ensure_scratch()
cuphyStatus_t bg1_z256up_bp_gmext_x2::ensure_scratch(size_t bytes)
{
    if(bytes <= m_scratch_bytes)
    {
        return CUPHY_STATUS_SUCCESS;
    }
    if(nullptr != m_scratch_d)
    {
        cudaFree(m_scratch_d);
        m_scratch_d     = nullptr;
        m_scratch_bytes = 0;
    }
    // FIXME: self-managed cudaMalloc. This runs on the first decode (a blocking,
    // non-deterministic sync on the RT path) and is illegal during CUDA graph
    // capture -- hence the get_launch_config() pre-alloc workaround. The size
    // (~7 MB at 256 slots) is also hidden from the decoder's workspace
    // accounting (workspace_size() reports 0). SUGGESTION: route this scratch
    // through the framework workspace instead -- report it via
    // get_workspace_size(), have the decode path allocate it once and pass the
    // pointer to the kernel, and drop m_scratch_d/ensure_scratch and the
    // graph-capture pre-alloc.
    cudaError_t e = cudaMalloc(&m_scratch_d, bytes);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    m_scratch_bytes = bytes;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_gmext_x2::~bg1_z256up_bp_gmext_x2()
bg1_z256up_bp_gmext_x2::~bg1_z256up_bp_gmext_x2()
{
    if(nullptr != m_scratch_d)
    {
        cudaFree(m_scratch_d);
        m_scratch_d = nullptr;
    }
}

} // namespace ldpc2
