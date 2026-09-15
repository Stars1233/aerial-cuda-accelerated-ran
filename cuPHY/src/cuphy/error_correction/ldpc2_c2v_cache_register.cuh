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

#if !defined(LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_register
// Check to variable (C2V) messages are stored in registers. (No loading
// or storing required, but causes register pressure.) Assumes that all
// APP values are in shared memory.
template <int          BG_,
          unsigned int NUM_PARITY_NODES,
          class        TC2V,
          class        TC2VStorage,
          class        TKernelParams>
struct c2v_cache_register
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef TC2VStorage           c2v_storage_t;
    typedef typename c2v_t::app_t app_t;
    static const int              BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_register()
    __device__
    c2v_cache_register(const TKernelParams& /*params*/) { }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < NUM_PARITY_NODES; ++i)
        {
            c2v_storage[i].init();
        }
    }
    //------------------------------------------------------------------
    // Out-of-bounds read
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(CHECK_IDX < NUM_PARITY_NODES,
                      "Parity check index exceeds allocation");
        c2v_t c2v;
        c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_t>(params,
                                                                          app,
                                                                          app_addr,
                                                                          c2v_storage[CHECK_IDX],
                                                                          smem_offset);
    }
    //------------------------------------------------------------------
    // Out-of-bounds read with APP sign-change tracking
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    bool process_row_sign_change(const TKernelParams& params,
                                 word_t               (&app)[NUM_APP_WORDS],
                                 int                  (&app_addr)[ROW_DEGREE],
                                 int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(CHECK_IDX < NUM_PARITY_NODES,
                      "Parity check index exceeds allocation");
        c2v_t c2v;
        return c2v.template process_row_sign_change<CHECK_IDX, TKernelParams, c2v_storage_t>(params,
                                                                                            app,
                                                                                            app_addr,
                                                                                            c2v_storage[CHECK_IDX],
                                                                                            smem_offset);
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_t c2v_storage[NUM_PARITY_NODES];
};

////////////////////////////////////////////////////////////////////////
// c2v_cache_register_core
// Check to variable (C2V) messages are stored in registers, with a
// different type for C2V storage for "core" parity rows. (No loading
// or storing required, but causes register pressure.) Assumes that all
// APP values are in shared memory.
// "Core" parity check nodes in BG1 are high-degree (19), and this
// require more storage to retain the signs.
// Note that the compiler may be able to accomplish the same results
// as specifying two different storage types by eliminating unread
// variables.
template <int          BG_,
          unsigned int NUM_PARITY_NODES,
          class        TC2V,
          class        TC2VStorageCore,
          class        TC2VStorageNonCore,
          class        TKernelParams>
struct c2v_cache_register_core
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef TC2VStorageCore       c2v_storage_core_t;
    typedef TC2VStorageNonCore    c2v_storage_noncore_t;
    typedef typename c2v_t::app_t app_t;
    static const int              BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_register_core()
    __device__
    c2v_cache_register_core(const TKernelParams& /*params*/) { }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            c2v_storage_core[i].init();
        }
        #pragma unroll
        for(int i = 4; i < NUM_PARITY_NODES; ++i)
        {
            c2v_storage_noncore[i-4].init();
        }
    }
    //------------------------------------------------------------------
    // Out-of-bounds read
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        c2v_t c2v;
        if(CHECK_IDX < 4)
        {
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                   app,
                                                                                   app_addr,
                                                                                   c2v_storage_core[CHECK_IDX],
                                                                                   smem_offset);
        }
        else
        {
            // coverity[event tag:FALSE]
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                      app,
                                                                                      app_addr,
                                                                                      c2v_storage_noncore[CHECK_IDX-4],
                                                                                      smem_offset);
        }
                                                                              
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_core_t    c2v_storage_core[4];
    c2v_storage_noncore_t c2v_storage_noncore[NUM_PARITY_NODES - 4];
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_)
