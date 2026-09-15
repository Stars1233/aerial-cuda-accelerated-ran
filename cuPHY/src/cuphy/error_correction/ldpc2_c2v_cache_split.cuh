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

#if !defined(LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_

#include "ldpc2.cuh"
#include <type_traits>

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// has_resize
// Type trait: true when T provides a T::resize<N> alias template.
// Used to detect box_plus storage (which supports per-row sizing).
template <class T, class = void>
struct has_resize : std::false_type {};
template <class T>
struct has_resize<T, std::void_t<typename T::template resize<1>>> : std::true_type {};

////////////////////////////////////////////////////////////////////////
// noncore_reg_chain
// Heterogeneous register storage: each non-core row gets storage sized
// to its actual update_row_degree, eliminating over-allocation.
// Only instantiated when TStorageNonCore has a resize<N> alias.
template <int BG, int IDX, int END, class TStorageNonCore, bool ACTIVE = (IDX < END)>
struct noncore_reg_chain;

// Base case: no more rows
template <int BG, int IDX, int END, class TStorageNonCore>
struct noncore_reg_chain<BG, IDX, END, TStorageNonCore, false>
{
    __device__ void init() {}
};

// Recursive case: store this row's data, then the rest
template <int BG, int IDX, int END, class TStorageNonCore>
struct noncore_reg_chain<BG, IDX, END, TStorageNonCore, true>
{
    typename TStorageNonCore::template resize<update_row_degree<BG, IDX>::value> element;
    noncore_reg_chain<BG, IDX+1, END, TStorageNonCore> rest;

    template <int CHECK_IDX>
    __device__ auto& get()
    {
        if constexpr (CHECK_IDX == IDX) return element;
        else return rest.template get<CHECK_IDX>();
    }

    __device__ void init()
    {
        element.init();
        rest.init();
    }
};

////////////////////////////////////////////////////////////////////////
// noncore_reg_array
// Homogeneous register storage wrapper (existing behavior).
// Provides the same get<CHECK_IDX>() / init() interface.
template <int START_IDX, int COUNT, class TStorageNonCore>
struct noncore_reg_array
{
    TStorageNonCore elements[COUNT];

    template <int CHECK_IDX>
    __device__ auto& get() { return elements[CHECK_IDX - START_IDX]; }

    __device__ void init()
    {
        #pragma unroll
        for(int i = 0; i < COUNT; ++i)
            elements[i].init();
    }
};

////////////////////////////////////////////////////////////////////////
// c2v_cache_split
// Check to variable (C2V) messages are stored in both shared memory
// and registers
// BG: Base graph (1 or 2)
// NUM_REG_C2V_NODES: Number of cC2V nodes stored in shared memory
// TC2V: Check to variable node class
//
// When TStorageCore is large (>16 bytes, e.g. box_plus storage for
// high-degree core rows), core C2V is stored in shared memory instead
// of registers. This is controlled by the CORE_IN_SHMEM constexpr.
template <int   BG_,
          int   NUM_REG_C2V_NODES,
          class TC2V,
          class TStorageCore,
          class TStorageNonCore,
          class TKernelParams>
struct c2v_cache_split
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    typedef TStorageCore          c2v_storage_core_t;
    typedef TStorageNonCore       c2v_storage_noncore_t;
    static const int BG = BG_;

    // Core rows go to shmem when storage exceeds min-sum size (4 words)
    static constexpr bool CORE_IN_SHMEM = (sizeof(c2v_storage_core_t) > 4 * sizeof(word_t));

    //------------------------------------------------------------------
    // c2v_cache_split()
    __device__
    c2v_cache_split(char*                     smem,
                    const LDPC_kernel_params& params)
    //: smem_(smem), Z_var_(params.Z_var)
    {
        init_reg_c2v();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        setup_shmem_addresses(smem, params.Z_var * sizeof(app_t), params.Z);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize shared memory with "zero" C2V data
        init_shared_c2v(params.num_parity_nodes, params.Z);
    }
    //------------------------------------------------------------------
    // c2v_cache_split()
    __device__
    c2v_cache_split(char*                              smem,
                    const cuphyLDPCDecodeConfigDesc_t& params)

    {
        init_reg_c2v();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the address of C2V data in shared memory (for this
        // thread).
        const int Kb              = (1 == BG_) ? 22 : 10;
        const int NUM_VAR_NODES   = Kb + params.num_parity_nodes;
        setup_shmem_addresses(smem, params.Z * NUM_VAR_NODES * sizeof(app_t), params.Z);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize shared memory with "zero" C2V data
        init_shared_c2v(params.num_parity_nodes, params.Z);
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        // Nothing done here, since we zero-initialize shared memory in
        // the constructor.
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        //thread0_dump_app(smem_, Z_var_);
        if constexpr (CHECK_IDX < 4)
        {
            if constexpr (CORE_IN_SHMEM)
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Load core C2V from shared memory
                c2v_storage_core_t st = c2v_storage_shm_core_[CHECK_IDX * params.Z];
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Initialize a C2V row processor and process the row
                c2v_t c2v;
                c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                       app,
                                                                                       app_addr,
                                                                                       st,
                                                                                       smem_offset);
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Store core C2V back to shared memory
                c2v_storage_shm_core_[CHECK_IDX * params.Z] = st;
            }
            else
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Initialize a C2V row processor and process the row
                c2v_t c2v;
                c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                       app,
                                                                                       app_addr,
                                                                                       c2v_storage_reg_core_[CHECK_IDX],
                                                                                       smem_offset);
            }
        }
        else if constexpr (CHECK_IDX < NUM_REG_C2V_NODES)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Initialize a C2V row processor and process the row.
            // Storage type may be per-row sized (variable noncore).
            auto& st = c2v_storage_reg_.template get<CHECK_IDX>();
            using st_type = std::remove_reference_t<decltype(st)>;
            c2v_t c2v;
            c2v.template process_row<CHECK_IDX, TKernelParams, st_type>(params,
                                                                        app,
                                                                        app_addr,
                                                                        st,
                                                                        smem_offset);
        }
        else
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load storage from shared memory
            c2v_storage_noncore_t st = c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z];
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Initialize a C2V row processor and process the row
            c2v_t c2v;
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                      app,
                                                                                      app_addr,
                                                                                      st,
                                                                                      smem_offset);
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Store C2V data back to shared memory, to be used during the
            // next iteration
            c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z] = st;
        }
        //thread0_dump_app(smem_, Z_var_);
    }

    //------------------------------------------------------------------
    // process_row_sign_change()
    // Process a row and return whether any APP sign changed while writing it.
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    bool process_row_sign_change(const TKernelParams& params,
                                 word_t               (&app)[NUM_APP_WORDS],
                                 int                  (&app_addr)[ROW_DEGREE],
                                 int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        if constexpr (CHECK_IDX < 4)
        {
            if constexpr (CORE_IN_SHMEM)
            {
                c2v_storage_core_t st = c2v_storage_shm_core_[CHECK_IDX * params.Z];
                c2v_t c2v;
                const bool sign_change = c2v.template process_row_sign_change<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                                                            app,
                                                                                                                            app_addr,
                                                                                                                            st,
                                                                                                                            smem_offset);
                c2v_storage_shm_core_[CHECK_IDX * params.Z] = st;
                return sign_change;
            }
            else
            {
                c2v_t c2v;
                return c2v.template process_row_sign_change<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                                          app,
                                                                                                          app_addr,
                                                                                                          c2v_storage_reg_core_[CHECK_IDX],
                                                                                                          smem_offset);
            }
        }
        else if constexpr (CHECK_IDX < NUM_REG_C2V_NODES)
        {
            auto& st = c2v_storage_reg_.template get<CHECK_IDX>();
            using st_type = std::remove_reference_t<decltype(st)>;
            c2v_t c2v;
            return c2v.template process_row_sign_change<CHECK_IDX, TKernelParams, st_type>(params,
                                                                                           app,
                                                                                           app_addr,
                                                                                           st,
                                                                                           smem_offset);
        }
        else
        {
            c2v_storage_noncore_t st = c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z];
            c2v_t c2v;
            const bool sign_change = c2v.template process_row_sign_change<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                                                           app,
                                                                                                                           app_addr,
                                                                                                                           st,
                                                                                                                           smem_offset);
            c2v_storage_shm_[(CHECK_IDX  - NUM_REG_C2V_NODES) * params.Z] = st;
            return sign_change;
        }
    }

    //------------------------------------------------------------------
    // core_shmem_size()
    // Returns the shared memory size needed for core C2V storage.
    // Returns 0 when core rows use register storage.
    static CUDA_BOTH
    int core_shmem_size(int Z)
    {
        if constexpr (CORE_IN_SHMEM)
            return 4 * Z * static_cast<int>(sizeof(c2v_storage_core_t));
        else
            return 0;
    }

private:
    //------------------------------------------------------------------
    // setup_shmem_addresses()
    // Compute shared memory addresses for core and non-core C2V data.
    // Layout: [APP] [core C2V if CORE_IN_SHMEM] [noncore C2V]
    __device__
    void setup_shmem_addresses(char* smem, int app_size_bytes, int Z)
    {
        int offset = app_size_bytes;

        if constexpr (CORE_IN_SHMEM)
        {
            offset = round_up_to_next(offset, static_cast<int>(alignof(c2v_storage_core_t)));
            c2v_storage_shm_core_ = reinterpret_cast<c2v_storage_core_t*>(smem + offset) + threadIdx.x;
            offset += 4 * Z * static_cast<int>(sizeof(c2v_storage_core_t));
        }

        offset = round_up_to_next(offset, static_cast<int>(alignof(c2v_storage_noncore_t)));
        c2v_storage_shm_ = reinterpret_cast<c2v_storage_noncore_t*>(smem + offset) + threadIdx.x;
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
    // init_shared_c2v()
    __device__
    void init_shared_c2v(int num_parity_nodes, int Z)
    {
        // Initialize core C2V in shmem if needed
        if constexpr (CORE_IN_SHMEM)
        {
            c2v_storage_core_t sZeroCore;
            sZeroCore.init();
            for(int i = 0; i < 4; ++i)
            {
                c2v_storage_shm_core_[i * Z] = sZeroCore;
            }
        }

        // Initialize non-core C2V in shmem
        const int             NUM_SHMEM_NODES = num_parity_nodes - NUM_REG_C2V_NODES;
        c2v_storage_noncore_t sZero;
        sZero.init();
        for(int i = 0; i < NUM_SHMEM_NODES; ++i)
        {
            c2v_storage_shm_[i * Z] = sZero;
        }
    }
    //------------------------------------------------------------------
    // Data
    // When CORE_IN_SHMEM, core storage is in shared memory, not registers.
    // Use a dummy type to avoid wasting register space.
    struct empty_storage { __device__ void init() {} };
    using core_reg_element_t = std::conditional_t<CORE_IN_SHMEM, empty_storage, c2v_storage_core_t>;
    core_reg_element_t     c2v_storage_reg_core_[CORE_IN_SHMEM ? 1 : 4];
    // Non-core register storage: per-row sized when storage supports resize
    // (box_plus), homogeneous array otherwise (min-sum).
    static constexpr bool USE_VARIABLE_NONCORE = has_resize<c2v_storage_noncore_t>::value;
    using noncore_container_t = std::conditional_t<
        USE_VARIABLE_NONCORE,
        noncore_reg_chain<BG, 4, NUM_REG_C2V_NODES, c2v_storage_noncore_t>,
        noncore_reg_array<4, NUM_REG_C2V_NODES - 4, c2v_storage_noncore_t>>;
    noncore_container_t    c2v_storage_reg_;
    c2v_storage_noncore_t* c2v_storage_shm_;
    c2v_storage_core_t*    c2v_storage_shm_core_; // only used when CORE_IN_SHMEM
    //char* smem_;
    //int Z_var_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_SPLIT_CUH_INCLUDED_)
