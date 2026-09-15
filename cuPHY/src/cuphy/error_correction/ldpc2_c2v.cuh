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

#if !defined(LDPC2_C2V_CUH_INCLUDED_)
#define LDPC2_C2V_CUH_INCLUDED_

#include "ldpc2.cuh"
#include "ldpc2_box_plus.cuh"

namespace ldpc2
{

template <int NUM_WORDS> struct storage_load_store_t
{
    typedef struct
    {
        word_t w[NUM_WORDS];
    } load_store_type;
    static
    __device__
    void load(load_store_type& x, const load_store_type* p)
    {
        const word_t* pw = static_cast<const word_t*>(p);
        #pragma unroll
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            x.w[i].i32 = pw[i].i32;
        }
    }
    static
    __device__
    void store(const load_store_type& x, load_store_type* p)
    {
        word_t* pw = static_cast<word_t*>(p);
        #pragma unroll
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            pw[i].i32 = x.w[i].i32;
        }
    }
};
template <> struct storage_load_store_t<2>
{
    typedef int2 load_store_type;
    static
    __device__
    void load(int2& x, const int2* p)
    {
        x = *(reinterpret_cast<const int2*>(p));
    }
    static
    __device__
    void store(const int2& x, int2* p)
    {
        *(reinterpret_cast<int2*>(p)) = x;
    }
};
template <> struct storage_load_store_t<4>
{
    typedef int4 load_store_type;
    static
    __device__
    void load(int4& x, const int4* p)
    {
        x = *(reinterpret_cast<const int4*>(p));
    }
    static
    __device__
    void store(const int4& x, int4* p)
    {
        *(reinterpret_cast<int4*>(p)) = x;
    }
};

////////////////////////////////////////////////////////////////////////
// C2V_storage_t
// Per-row "storage" for a check-to-variable (C2V) processor. This
// data structure is used to subtract APP contributions from the
// previous iteration for an individual row.
// In some cases, row processors may use different approaches.
template <typename T, int NUM_WORDS> struct C2V_storage_t;

template <int NUM_WORDS_> struct C2V_storage_t<__half, NUM_WORDS_>
{
    typedef storage_load_store_t<NUM_WORDS_>       load_store_t;
    typedef typename load_store_t::load_store_type load_store_type;
    static constexpr int NUM_WORDS = NUM_WORDS_;
    struct cC2V_t
    {
        word_t    min1_min0;
        uint32_t  signs_index;
    };
    // Compressed C2V row processors will use the c2v field. Box plus
    // row processors will use the w[] array to store sequences
    // directly.
    union value
    {
        cC2V_t          c2v;
        word_t          w[NUM_WORDS_];
        load_store_type load_store;
    };
    C2V_storage_t() = default;
    __device__
    C2V_storage_t(word_t m1_m0, uint32_t s_i)
    {
        v.c2v.min1_min0 = m1_m0;
        v.c2v.signs_index = s_i;
    }
    __device__
    C2V_storage_t(const C2V_storage_t* p)
    {
        load_store_t::load(v.load_store, &(p->v.load_store));
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            v.w[i].u32 = 0;
        }
    }
    //------------------------------------------------------------------
    // store()
    __device__
    void store(C2V_storage_t* p) const
    {
        load_store_t::store(v.load_store, &(p->v.load_store));
    }
    //------------------------------------------------------------------
    // c2v()
    __host__ __device__
    cC2V_t& c2v() { return v.c2v; }
    //------------------------------------------------------------------
    // c2v()
    __host__ __device__
    const cC2V_t& c2v() const { return v.c2v; }
    //------------------------------------------------------------------
    // Data
    value v;
};

////////////////////////////////////////////////////////////////////////
// Structure with expanded, in-register C2V representation (as opposed
// to one compressed for storage to global/shared memory), for temporary
// use in processing a single parity node.
template <typename T,
          class    TSignMgr,
          class    TMinSumUpdate,
          class    TC2VStorage,
          class    TEnable = void> class cC2V_row_context;

template <class TSignMgr, class TMinSumUpdate, class TC2VStorage>
class cC2V_row_context<__half, TSignMgr, TMinSumUpdate, TC2VStorage>
{
public:
    typedef TC2VStorage storage_t;
    //------------------------------------------------------------------
    // Construct a cC2V representation from a storage (compressed) form
    template <int ROW_DEGREE>
    __device__
    cC2V_row_context(const storage_t&                        s,
                     std::integral_constant<int, ROW_DEGREE>) :
        min1_min0(s.v.c2v.min1_min0),
        signs(sign_mgr_t::from_packed_word(s.v.c2v.signs_index)),
        min0_index(sign_mgr_t::non_sign_bits(s.v.c2v.signs_index))
    {
        setup_extract_context();
    }
    //------------------------------------------------------------------
    // Construct a cC2V representation from a sequence of values
    template <int ROW_DEGREE, int MAX_WORDS>
    __device__
    explicit cC2V_row_context(const __half2&                          norm,
                              word_t                                  (&app)[MAX_WORDS],
                              std::integral_constant<int, ROW_DEGREE>)
    {
        // Initialize with first two values
        init_row<ROW_DEGREE>(app[0]);
        // Update with "full" words
        #pragma unroll
        for(int i = 1; i < div_round_down_t<ROW_DEGREE, 2>::value; ++i)
        {
            update(app[i], i);
        }
        // Optional odd entry at the end
        if(0 != (ROW_DEGREE % 2))
        {
            update_low(app[div_round_down_t<ROW_DEGREE, 2>::value],
                       div_round_down_t<ROW_DEGREE, 2>::value);
        }
        // Post-process to prepare for extraction
        finalize(norm, div_round_up_t<ROW_DEGREE, 2>::value);
        setup_extract_context();
    }
    //------------------------------------------------------------------
    // extract_pair()
    // Extract a pair of values using the internal representation. The
    // returned word will contain two values, with the low word having
    // the sequence value with the lower index.
    __device__
    word_t extract_pair(int pair_idx) const
    {
        word_t w = (pair_idx == ex_ctx.min0_pair_index) ? ex_ctx.min0_pair_value : ex_ctx.min0_min0;
        return sign_mgr_t::apply_sign(w, signs, pair_idx);
    }
    //------------------------------------------------------------------
    // extract()
    // Use the internal compressed C2V representation to extract the
    // sequence of values.
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, bool ZERO_HIGH_EXTENSION>
    __device__
    void extract(word_t (&seq)[div_round_up_t<UPDATE_ROW_DEGREE, 2>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the compressed representation
        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            seq[i] = extract_pair(i);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // When ZERO_HIGH_EXTENSION is true, we need to handle the
        // extension values, which do not need to be updated,
        // specially here. Specifically, for odd update row degrees,
        // the low value needs have the value from the previous iteration
        // subtracted, but the high value does not.
        if((UPDATE_ROW_DEGREE != ROW_DEGREE) &&
           (0 != (UPDATE_ROW_DEGREE % 2))    &&
           ZERO_HIGH_EXTENSION)
        {
            constexpr int LAST_INDEX = div_round_up_t<UPDATE_ROW_DEGREE, 2>::value - 1;
            seq[LAST_INDEX] = set_high_zero(seq[LAST_INDEX]);
        }
    }

    //------------------------------------------------------------------
    // Return the compressed storage representation of the row context
    __device__
    storage_t get_storage() const
    {
        return storage_t(min1_min0, sign_mgr_t::to_packed_word(signs, min0_index));
    }
private:
    struct extract_context
    {
        word_t    min0_min0;
        int       min0_pair_index;
        word_t    min0_pair_value;
    };
    //------------------------------------------------------------------
    typedef TSignMgr              sign_mgr_t;
    typedef TMinSumUpdate         min_sum_update_t;
    //------------------------------------------------------------------
    // init_row()
    template <int ROW_DEGREE>
    __device__
    void init_row(word_t v1_v0)
    {
        min1_min0 = v1_v0;
        half2_sort(min1_min0, min0_index);
        sign_mgr_t::template init_row<ROW_DEGREE>(signs, v1_v0);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal representation with a new pair of values
    __device__
    void update(word_t v1_v0, int pair_idx)
    {
        min_sum_update_t::update_pair(min1_min0,
                                      min0_index,
                                      v1_v0,
                                      pair_idx);
        sign_mgr_t::update(signs, v1_v0, pair_idx);
    }
    //------------------------------------------------------------------
    // update_low()
    // Update the internal representation with a single value
    __device__
    void update_low(word_t vx_v0, int pair_idx)
    {
        min_sum_update_t::update_low(min1_min0,
                                     min0_index,
                                     vx_v0,
                                     pair_idx);
        sign_mgr_t::update_low(signs, vx_v0, pair_idx);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize(const __half2& norm, int pair_count)
    {
        // Apply normalization to both values (min1 and min 0)
        // TODO: fuse with add of update to APP value
        min1_min0.f16x2 = __hmul2(min1_min0.f16x2, norm);

        // Adjust signs of min0, min1 values (optionally)
        sign_mgr_t::finalize(min1_min0, signs, pair_count);
    }
    //------------------------------------------------------------------
    // setup_extract_context()
    // Populate a data structure in preparation for extraction of a
    // sequence of values from the compressed representation.
    __device__
    void setup_extract_context()
    {
        ex_ctx.min0_pair_index = min0_index >> 1;
        ex_ctx.min0_min0.f16x2 = __low2half2(min1_min0.f16x2);
        // If the min0 index is even, the low fp16 value for the
        // min0_pair_index will have min1 and the high order value will
        // be min0. Vice versa if the min0 index is odd.
        ex_ctx.min0_pair_value.f16x2  = (0 == (min0_index % 2))             ?
                                         __lowhigh2highlow(min1_min0.f16x2) :
                                         __half2(min1_min0.f16x2);
    }
    //------------------------------------------------------------------
    // Data
    word_t          min1_min0;
    uint32_t        signs;
    int             min0_index;
    extract_context ex_ctx;
};

//------------------------------------------------------------------
// simple_row_map
// Row processing dispatch template structure which returns the same
// processor class for all rows. (Other implementations may change
// the row processor based on, for example, the row degree. However,
// this decoder class uses the same processor for all rows.)
template <int   BG,
          int   CHECK_IDX,
          class TC2VStorage,
          class TC2V> struct simple_row_map
{
    // Return the template C2V row processor for all rows.
    typedef TC2V row_proc_t;
};

//------------------------------------------------------------------
// context_storage_row_map
template <int                    BG,
          int                    CHECK_IDX,
          class                  TC2VStorage,
          typename               T,
          template <class> class TRowContext,
          template <class> class TRowProc
          > struct context_storage_row_map
{
    // Row context using storage template parameter
    typedef TRowContext<TC2VStorage> row_context_t;
    // C2V row processor using the row_context
    typedef TRowProc<row_context_t>  row_proc_t;
};

////////////////////////////////////////////////////////////////////////
// C2V_row_proc
// Orchestrator class for check-to-variable (C2V) row processing. The
// TRowMap template parameter is used to determine which processing
// function is used for each check node row, allowing for row-dependent
// processing. (As a concrete example, we may process high-degree rows
// differently than low-degree rows.)
template <typename                            T,
          int                                 BG,
          template <int, int, class> class    TRowMap,
          template <typename, int> class      TAPPLoader,
          template <typename, int, int> class TAPPWriter>
class C2V_row_proc
{
public:
    typedef T           app_t;
    //------------------------------------------------------------------
    // C2V_row_proc()
    C2V_row_proc() = default;
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, class TKernelParams, class TC2VStorage, bool IS_FIRST = false, bool IS_LAST = false>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[app_num_words<T, BG, CHECK_IDX>::value],
                     int                  (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                     TC2VStorage&         c2v_storage,
                     int                  smem_offset)
    {
        const int ROW_DEGREE        = row_degree<BG, CHECK_IDX>::value;
        const int UPDATE_ROW_DEGREE = update_row_degree<BG, CHECK_IDX>::value;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // The row map determines which row processor will be applied
        // to the current row.
        typedef typename TRowMap<BG,
                                 CHECK_IDX,
                                 TC2VStorage>::row_proc_t row_proc_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load APP values into registers
        typedef TAPPLoader<T, ROW_DEGREE> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Perform row processing:
        // - update the given APP values
        // - update C2V storage to be used for the next iteration
        // IS_FIRST routes to the row processor's first-iteration path (skip the
        // still-zero previous-C2V subtract/decompress); IS_LAST routes to its
        // final-iteration path (skip the now-dead C2V store/pack). At most one
        // is ever true (first != last for >1 iteration). The if constexpr keeps
        // the extra template args out of every interior instantiation.
        if constexpr (IS_FIRST)
        {
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE, true, false>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
        }
        else if constexpr (IS_LAST)
        {
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE, false, true>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
        }
        else
        {
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values to shared_memory
        typedef TAPPWriter<T, ROW_DEGREE, UPDATE_ROW_DEGREE> app_writer_t;
        app_writer_t::write_non_ext(app, app_addr, smem_offset);
    }

    template <int CHECK_IDX, class TKernelParams, class TC2VStorage>
    __device__
    bool process_row_sign_change(const TKernelParams& params,
                                 word_t               (&app)[app_num_words<T, BG, CHECK_IDX>::value],
                                 int                  (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                                 TC2VStorage&         c2v_storage,
                                 int                  smem_offset)
    {
        const int ROW_DEGREE        = row_degree<BG, CHECK_IDX>::value;
        const int UPDATE_ROW_DEGREE = update_row_degree<BG, CHECK_IDX>::value;
        typedef typename TRowMap<BG, CHECK_IDX, TC2VStorage>::row_proc_t row_proc_t;

        typedef TAPPLoader<T, ROW_DEGREE> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);

        row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);

        typedef TAPPWriter<T, ROW_DEGREE, UPDATE_ROW_DEGREE> app_writer_t;
        return app_writer_t::write_non_ext_sign_change(app, app_addr, smem_offset);
    }
    //------------------------------------------------------------------
    // Phase-split process_row(): load_app / compute_app / write_app expose the
    // three steps of process_row() separately, so a caller can hoist a row's
    // shared-memory load across the preceding barrier (the split C2V cache's
    // cross-barrier APP preload) or defer its write-back.
    // process_row() runs load -> compute -> write as one block, which pins all
    // three to the same barrier interval. The arithmetic and its order are
    // byte-for-byte identical to process_row() -- only the load and the write
    // are pulled out so a caller can move THEM (not the math). compute_app is
    // IS_FIRST/IS_LAST-aware so a caller that peels the first or last iteration
    // keeps the boundary-iteration C2V subtract/store elision.
    //------------------------------------------------------------------
    // load_app(): shared-memory load of a row's APP values.
    template <int CHECK_IDX>
    __device__
    void load_app(word_t (&app)     [app_num_words<T, BG, CHECK_IDX>::value],
                  int    (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                  int    smem_offset)
    {
        const int ROW_DEGREE = row_degree<BG, CHECK_IDX>::value;
        typedef TAPPLoader<T, ROW_DEGREE> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);
    }
    //------------------------------------------------------------------
    // compute_app(): register-only check-node update on an already-loaded APP
    // (updates APP + C2V storage). No shared-memory traffic -> two disjoint
    // rows' compute_app() calls are alias-free and free to interleave.
    template <int CHECK_IDX, class TKernelParams, class TC2VStorage,
              bool IS_FIRST = false, bool IS_LAST = false
#if LDPC2_C2V_SELECTIVE_WRITEBACK
              , bool ORDERED_STORE = false
#endif
              >
    __device__
    void compute_app(const TKernelParams& params,
                     word_t       (&app)[app_num_words<T, BG, CHECK_IDX>::value],
                     TC2VStorage&   c2v_storage)
    {
        const int ROW_DEGREE        = row_degree<BG, CHECK_IDX>::value;
        const int UPDATE_ROW_DEGREE = update_row_degree<BG, CHECK_IDX>::value;
        typedef typename TRowMap<BG,
                                 CHECK_IDX,
                                 TC2VStorage>::row_proc_t row_proc_t;
        if constexpr (IS_FIRST)
        {
#if LDPC2_C2V_SELECTIVE_WRITEBACK
            if constexpr(ORDERED_STORE)
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE,
                                                 true, false, TC2VStorage,
                                                 true>(app,
                                                       c2v_storage,
                                                       params.norm.f16x2);
            }
            else
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE,
                                                 true, false>(app,
                                                              c2v_storage,
                                                              params.norm.f16x2);
            }
#else
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE, true, false>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
#endif
        }
        else if constexpr (IS_LAST)
        {
#if LDPC2_C2V_SELECTIVE_WRITEBACK
            if constexpr(ORDERED_STORE)
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE,
                                                 false, true, TC2VStorage,
                                                 true>(app,
                                                       c2v_storage,
                                                       params.norm.f16x2);
            }
            else
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE,
                                                 false, true>(app,
                                                              c2v_storage,
                                                              params.norm.f16x2);
            }
#else
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE, false, true>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
#endif
        }
        else
        {
#if LDPC2_C2V_SELECTIVE_WRITEBACK
            if constexpr(ORDERED_STORE)
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE,
                                                 false, false, TC2VStorage,
                                                 true>(app,
                                                       c2v_storage,
                                                       params.norm.f16x2);
            }
            else
            {
                row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE>(app,
                                                                                c2v_storage,
                                                                                params.norm.f16x2);
            }
#else
            row_proc_t::template process_row<ROW_DEGREE, UPDATE_ROW_DEGREE>(app,
                                                                        c2v_storage,
                                                                        params.norm.f16x2);
#endif
        }
    }
    //------------------------------------------------------------------
    // write_app(): shared-memory write-back of the updated APP values.
    template <int CHECK_IDX>
    __device__
    void write_app(word_t (&app)     [app_num_words<T, BG, CHECK_IDX>::value],
                   int    (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                   int    smem_offset)
    {
        const int ROW_DEGREE        = row_degree<BG, CHECK_IDX>::value;
        const int UPDATE_ROW_DEGREE = update_row_degree<BG, CHECK_IDX>::value;
        typedef TAPPWriter<T, ROW_DEGREE, UPDATE_ROW_DEGREE> app_writer_t;
        app_writer_t::write_non_ext(app, app_addr, smem_offset);
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_row_proc
// Row processor for a compressed C2V message implementation.
template <typename T, class TRowContext> class cC2V_row_proc;

template <class TRowContext>
class cC2V_row_proc<__half, TRowContext>
{
private:
    //------------------------------------------------------------------
    typedef TRowContext            row_context_t;
public:
    //typedef typename row_context_t::storage_t storage_t;

    //------------------------------------------------------------------
    // init()
    //__device__ static void init(storage_t& s) { s.init(); }
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              class TStorage>
    __device__
    static void process_row(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                                  const TStorage& row_storage)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a row processing context using data from the
        // previous iteration (which may be stored in registers or
        // global memory).
        row_context_t rc(row_storage, std::integral_constant<int, ROW_DEGREE>{});
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't want to update APP values for extension
        // nodes, so we ensure that those values are zero.
        const int UPDATE_NUM_WORDS = div_round_up_t<UPDATE_ROW_DEGREE, 2>::value;
        word_t    prev[UPDATE_NUM_WORDS];
        rc.template extract<ROW_DEGREE, UPDATE_ROW_DEGREE, true>(prev);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Subtract previous contribution from the APP values
        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            app[i].f16x2 = __hsub2(app[i].f16x2, prev[i].f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[row_num_words<__half, ROW_DEGREE>::value])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Construct a row context using the updated APP values (with
        // values from the previous iteration subtracted). This will
        // create a min-sum representation of the APP values.
        row_context_t rc(norm,
                         app,
                         std::integral_constant<int, ROW_DEGREE>{});
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract the sequence of values from the compressed
        // representation. We don't need to zero extension node values
        // because they won't be written back to the APP array.
        const int UPDATE_NUM_WORDS = div_round_up_t<UPDATE_ROW_DEGREE, 2>::value;
        word_t    inc[UPDATE_NUM_WORDS];
        rc.template extract<ROW_DEGREE, UPDATE_ROW_DEGREE, false>(inc);
        
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update APP values
        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            app[i].f16x2 = __hadd2(app[i].f16x2, inc[i].f16x2);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the compressed C2V representation
        row_storage = rc.get_storage();
    }
};

// Print at least NUM_VALUES from the word_t array a, assuming that each word
// holds one or more value of type TElem.
// The printing loop rounds up the number of values printed such that entire
// words are printed. For example, if NUM_VALUES is odd, this function will
// print the "extra" value in the high part of the final word.
template <int NUM_VALUES, class TElem, int ARRAY_DIM>
__host__ __device__
void print_word_array_count(int tIdx, const char* label, const word_t (&a)[ARRAY_DIM], const char* suffix = nullptr)
{
    if(threadIdx.x == tIdx)
    {
        constexpr int VALUES_PER_WORD = ldpc_traits<TElem>::values_per_word;
        printf("%s: ", label);
        for(int i = 0; i < div_round_up_t<NUM_VALUES, VALUES_PER_WORD>::value; ++i)
        {
            print_word_values_as<TElem>(a[i]);
        }
        printf("\n");
        if(suffix)
        {
            printf(suffix);
        }
    }
}

template <class TElem, int ARRAY_DIM>
__device__
void print_word_array(int tIdx, const char* label, const word_t (&a)[ARRAY_DIM], const char* suffix = nullptr)
{
    if(threadIdx.x == tIdx)
    {
        printf("%s: ", label);
        for(int i = 0; i < ARRAY_DIM; ++i)
        {
            print_word_values_as<TElem>(a[i]);
        }
        printf("\n");
        if(suffix)
        {
            printf(suffix);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// box_plus_row_proc
// Row processor for a box plus implementation
template <class TBoxPlusOp, class TLLR, class TEnable = void> class box_plus_row_proc;


// Specialization for __half LLR type
template <class TBoxPlusOp>
class box_plus_row_proc<TBoxPlusOp, __half>
{
public:
    //typedef TC2VStorage storage_t;
    //------------------------------------------------------------------
    // init()
    //__device__ static void init(storage_t& s) { s.init(); }
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              class TStorage>
    __device__
    static void process_row(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                                  const TStorage& row_storage)
    {
        //print_word_array_count<ROW_DEGREE, __half>(0, "APP", app);
        //print_word_array<__half>                  (0, "C2V", row_storage.v.w);

        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            // Relying on extension nodes in the high word being
            // set to zero in the storage structure.
            app[i].f16x2 = __hsub2(app[i].f16x2, row_storage.v.w[i].f16x2);
        }
        //print_word_array_count<ROW_DEGREE, __half>(0, "SUB", app);
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[row_num_words<__half, ROW_DEGREE>::value])
    {
        word_t bp_update_seq[div_round_up_t<UPDATE_ROW_DEGREE, 2>::value];

        typedef box_plus_seq_gen<__half, TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;
        
        box_plus_seq_gen_t::generate(bp_update_seq, app);
        //print_word_array_count<UPDATE_ROW_DEGREE, __half>(0, "SEQ", bp_update_seq);

        #pragma unroll
        for(int i = 0; i < div_round_up_t<UPDATE_ROW_DEGREE, 2>::value; ++i)
        {
            // Apply normalization and store for next iteration
            row_storage.v.w[i].f16x2 = __hmul2(bp_update_seq[i].f16x2, norm);
            // Update APP
            //app[i].f16x2 = __hfma2(bp_update_seq[i].f16x2, norm, app[i].f16x2);
            app[i].f16x2 = __hadd2(row_storage.v.w[i].f16x2, app[i].f16x2);
        }
        //print_word_array_count<UPDATE_ROW_DEGREE, __half>(0, "NEW", app, "----\n");
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_index
// (To be deprecated in favor of row_proc interfaces above)
template <typename                            T,
          int                                 BG,
          class                               TSignManager,
          class                               TMinSumUpdate,
          template <typename, int> class      TAPPLoader,
          template <typename, int, int> class TAPPWriter> struct cC2V_index;

template <int                                 BG,
          class                               TSignManager,
          class                               TMinSumUpdate,
          template <typename, int> class      TAPPLoader,
          template <typename, int, int> class TAPPWriter>
struct cC2V_index<__half, BG, TSignManager, TMinSumUpdate, TAPPLoader, TAPPWriter>
{
    typedef __half                                                               app_t;
    //typedef C2V_storage_t<__half, 2>                                             c2v_storage_t;
    typedef TSignManager                                                         sign_mgr_t;
    //------------------------------------------------------------------
    // init()
    //__device__
    //void init()
    //{
    //    c2v_storage.init();
    //}
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_sub_prev_iter(word_t             (&app)[app_num_words<__half, BG, CHECK_IDX>::value],
                           const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't need to update APP values for extension
        // nodes.
        // We need to handle the extension values, which are not updated,
        // specially here. Specifically, for odd update row degrees,
        // the low value needs have the value from the previous iteration
        // subtracted, but the high value does not.
        #pragma unroll
        for(int i = 0; i < div_round_up_t<update_row_degree<BG, CHECK_IDX>::value, 2>::value; ++i)
        {
            // Operate on a pair of values at a time
            word_t dec = rc.extract_pair(i);

            // For extension nodes with an odd update row degree, zero
            // the high word before modifying the APP value.
            if((update_row_degree<BG, CHECK_IDX>::value != row_degree<BG, CHECK_IDX>::value) &&
               (0 != (update_row_degree<BG, CHECK_IDX>::value % 2))                          &&
               (i == div_round_down_t<update_row_degree<BG, CHECK_IDX>::value, 2>::value))
            {
                dec = set_high_zero(dec);
            }

            // TODO: Fuse normalization from prev iteration?
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_update(word_t             (&app)[app_num_words<__half, BG, CHECK_IDX>::value],
                    const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the new row context and update the APP
        // values.
        #pragma unroll
        for(int i = 0; i < div_round_up_t<update_row_degree<BG, CHECK_IDX>::value, 2>::value; ++i)
        {
            word_t inc = rc.extract_pair(i);
            // OK to modify the APP value for the extension nodes - it
            // won't be written to shared memory below
            app[i].f16x2 = __hadd2(app[i].f16x2, inc.f16x2);
        }
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, class TKernelParams, class TStorage>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[app_num_words<__half, BG, CHECK_IDX>::value],
                     int                  (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                     TStorage&            c2v_storage,
                     int                  smem_offset)
    {
        typedef cC2V_row_context<__half, TSignManager, TMinSumUpdate, TStorage> row_context_t;
        
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load APP values into registers
        typedef TAPPLoader<__half, row_degree<BG, CHECK_IDX>::value> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Initialize a row processing context using data from the
            // previous iteration (which may be stored in registers or
            // global memory).
            row_context_t rc(c2v_storage, std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Update the APP values with the decrement from the previous
            // iteration.
            app_sub_prev_iter<CHECK_IDX>(app, rc);
        }
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Construct a row context using the updated APP values (with
            // values from the previous iteration subtracted). This will
            // create a min-sum representation of the APP values.
            row_context_t rcNew(params.norm.f16x2,
                                app,
                                std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Update the compressed representation for the next iteration.
            c2v_storage = rcNew.get_storage();

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Extract values from the new row context and update the APP
            // values.
            app_update<CHECK_IDX>(app, rcNew);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values to shared_memory
        typedef TAPPWriter<__half,
                           row_degree       <BG, CHECK_IDX>::value,
                           update_row_degree<BG, CHECK_IDX>::value> app_writer_t;
        app_writer_t::write_non_ext(app, app_addr, smem_offset);
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CUH_INCLUDED_)
