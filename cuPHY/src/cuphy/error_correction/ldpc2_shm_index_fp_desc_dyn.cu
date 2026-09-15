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

#include <assert.h>
#include "ldpc2_desc.cuh"
#include "ldpc2_c2v.cuh"
#include "ldpc2_app_address_fp_desc.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"
#include "ldpc2_shm_index_fp_desc_dyn.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_shared.cuh"
#include "ldpc2_crc_dispatch.cuh"

using namespace ldpc2;

#define USE_APP_ADDR_FP_DP 1

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;
    const int NUM_STORAGE_WORDS   = 2;
    const int MIN_PARITY_ROWS     = 4;

    // Base graph-dependent maximum number of parity nodes. Limited by
    // the amount of shared memory...
    template <int BG> struct shm_max_parity_nodes;
    template <>
    struct shm_max_parity_nodes<1>
    {
        static const int value = 46;
    };
    template <>
    struct shm_max_parity_nodes<2>
    {
        static const int value = 42;
    };

    //------------------------------------------------------------------
    // Sign manager policies (for compressed C2V row processors)
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_fp,      false> sign_dst_fp_t;
    typedef ldpc2::sign_store_policy_split_dst<__half, ldpc2::split_sign_update_bit_ops, false> sign_dst_bit_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_fp,      false> sign_src_fp_t;
    typedef ldpc2::sign_store_policy_split_src<__half, ldpc2::split_sign_update_bit_ops, false> sign_src_bit_t;

    // Sign updates with FP unit A couple of microseconds faster on V100...
    typedef sign_dst_fp_t                                                                       sign_mgr_t;

    // APP address calculation
    // Using floating point instruction APP address calculation
    // sequence
#if USE_APP_ADDR_FP_DP
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half, BG>;
#else
    template <int BG> using app_loc_t = app_loc_address_fp_desc<__half, BG>;
#endif

    // C2V per-row storage. Larger storage allows faster row
    // processing, but increases register pressure (and may incur
    // register spills).
    typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS>                                     c2v_storage_t;

    //------------------------------------------------------------------
    // Alias template for compressed C2V row processors, with a template
    // parameter for the C2V row storage. The sign processor and min
    // sum updater have been chosen to be the "fastest" for some
    // architecture and lifting size combinations.
    typedef ldpc2::cC2V_row_proc<__half,
                                 ldpc2::cC2V_row_context<__half,
                                                         sign_dst_fp_t,
                                                         ldpc2::min_sum_update_half_0,
                                                         c2v_storage_t>
                                > cC2V_row_proc;
    //------------------------------------------------------------------
    // cC2V_all_row_map_t
    // The C2V_row_proc template requires a row map template with template
    // arguments BG (int), CHECK_IDX (int), TStorage (per-row storage
    // structure.
    template <int   BG,
              int   CHECK_IDX,
              class TC2VStorage> using cC2V_all_row_map_t = simple_row_map<BG,
                                                                           CHECK_IDX,
                                                                           TC2VStorage,
                                                                           cC2V_row_proc>;
    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    // BG_: base graph (1 or 2)
    // TKernelParams: Class/struct used for kernel parameters
    // TLLR: Source LLR data type (__half or float)
    // TLoader: LLR loader template struct (e.g. llr_loader_batch)
    template <int                                                                BG_,
              class                                                              TKernelParams,
              typename                                                           TLLR,
              template <typename, int, template<typename, typename> class> class TLoader>
    struct ldpc2_shm_index_fp_desc_dyn_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MAX_PARITY_ROWS = shm_max_parity_nodes<BG>::value;

        typedef TKernelParams                   kernel_params_t;

        typedef C2V_row_proc<__half,
                             BG,
                             cC2V_all_row_map_t,
                             app_loader,
                             app_writer>                                  C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (TODO: use shared memory here)
        typedef ldpc2::c2v_cache_shared<BG,
                                        C2V_t,
                                        c2v_storage_t,
                                        kernel_params_t>                        c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef TLoader<TLLR, 4, llr_op_clamp>                                  llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef typename llr_loader_t::app_buf_t                                app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };

    //------------------------------------------------------------------
    // get_shmem_required()
    // For the shared memory kernel, this is the sum of the APP storage
    // and the C2V data storage.
    int get_shmem_required(int BG,
                            int num_parity_nodes,
                            int Z)
    {
        const uint32_t NUM_VAR_NODES = ((1 == BG) ? ldpc2::max_info_nodes<1>::value : ldpc2::max_info_nodes<2>::value) +
                                       static_cast<uint32_t>(num_parity_nodes);
        const uint32_t APP_SIZE = shmem_llr_buffer_size(NUM_VAR_NODES, static_cast<uint32_t>(Z), sizeof(__half));
        const uint32_t C2V_SIZE = static_cast<uint32_t>(num_parity_nodes) * static_cast<uint32_t>(Z) * sizeof(c2v_storage_t);
        const uint32_t base_size = round_up_to_next(APP_SIZE, static_cast<uint32_t>(alignof(c2v_storage_t))) + C2V_SIZE;
        return static_cast<int>(ldpc2::shmem_size_with_et_context(base_size));
    }
} // namespace


////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_shm_index_fp_desc_dyn()
// Base graph 1 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_shm_index_fp_desc_dyn(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<1,
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

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
    // Write hard output based on APP values
    //ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_shm_index_fp_desc_dyn()
// Base graph 2 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_shm_index_fp_desc_dyn(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<2, // BG
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

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
    // Write hard output based on APP values
    ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_shm_index_fp_desc_dyn_tb()
// Base graph 1 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_shm_index_fp_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Early-termination context (before per-iteration loop)
    const uint32_t NUM_VAR_NODES = ldpc2::max_info_nodes<kernel_config_t::BG>::value + decodeDesc.config.num_parity_nodes;
    const uint32_t APP_SIZE = static_cast<uint32_t>(ldpc2::shmem_llr_buffer_size(NUM_VAR_NODES, decodeDesc.config.Z, sizeof(__half)));
    const uint32_t C2V_SIZE = decodeDesc.config.num_parity_nodes * decodeDesc.config.Z * sizeof(c2v_storage_t);
    const uint32_t size_before_et = round_up_to_next(APP_SIZE, static_cast<uint32_t>(alignof(c2v_storage_t))) + C2V_SIZE;
    ldpc_et_context_t* et_ctx = ldpc2::get_et_ctx_ptr(smem, size_before_et);
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;  // Assume fail initially
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<kernel_config_t::BG>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write CRC result if early termination enabled
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    //------------------------------------------------------------------
    // Write iteration count if requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_shm_index_fp_desc_dyn_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Early-termination context (before per-iteration loop)
    const uint32_t NUM_VAR_NODES = ldpc2::max_info_nodes<kernel_config_t::BG>::value + decodeDesc.config.num_parity_nodes;
    const uint32_t APP_SIZE = static_cast<uint32_t>(ldpc2::shmem_llr_buffer_size(NUM_VAR_NODES, decodeDesc.config.Z, sizeof(__half)));
    const uint32_t C2V_SIZE = decodeDesc.config.num_parity_nodes * decodeDesc.config.Z * sizeof(c2v_storage_t);
    const uint32_t size_before_et = round_up_to_next(APP_SIZE, static_cast<uint32_t>(alignof(c2v_storage_t))) + C2V_SIZE;
    ldpc_et_context_t* et_ctx = ldpc2::get_et_ctx_ptr(smem, size_before_et);
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;  // Assume fail initially
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<kernel_config_t::BG>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write CRC result if early termination enabled
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    //------------------------------------------------------------------
    // Write iteration count if requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_shm_index_fp_desc_dyn_tb()
// Base graph 2 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_shm_index_fp_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Early-termination context (before per-iteration loop)
    const uint32_t NUM_VAR_NODES = ldpc2::max_info_nodes<kernel_config_t::BG>::value + decodeDesc.config.num_parity_nodes;
    const uint32_t APP_SIZE = static_cast<uint32_t>(ldpc2::shmem_llr_buffer_size(NUM_VAR_NODES, decodeDesc.config.Z, sizeof(__half)));
    const uint32_t C2V_SIZE = decodeDesc.config.num_parity_nodes * decodeDesc.config.Z * sizeof(c2v_storage_t);
    const uint32_t size_before_et = round_up_to_next(APP_SIZE, static_cast<uint32_t>(alignof(c2v_storage_t))) + C2V_SIZE;
    ldpc_et_context_t* et_ctx = ldpc2::get_et_ctx_ptr(smem, size_before_et);
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;  // Assume fail initially
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<kernel_config_t::BG>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write CRC result if early termination enabled
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    //------------------------------------------------------------------
    // Write iteration count if requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_shm_index_fp_desc_dyn_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_shm_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Early-termination context (before per-iteration loop)
    const uint32_t NUM_VAR_NODES = ldpc2::max_info_nodes<kernel_config_t::BG>::value + decodeDesc.config.num_parity_nodes;
    const uint32_t APP_SIZE = static_cast<uint32_t>(ldpc2::shmem_llr_buffer_size(NUM_VAR_NODES, decodeDesc.config.Z, sizeof(__half)));
    const uint32_t C2V_SIZE = decodeDesc.config.num_parity_nodes * decodeDesc.config.Z * sizeof(c2v_storage_t);
    const uint32_t size_before_et = round_up_to_next(APP_SIZE, static_cast<uint32_t>(alignof(c2v_storage_t))) + C2V_SIZE;
    ldpc_et_context_t* et_ctx = ldpc2::get_et_ctx_ptr(smem, size_before_et);
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;  // Assume fail initially
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<kernel_config_t::BG>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write CRC result if early termination enabled
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    //------------------------------------------------------------------
    // Write iteration count if requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// shm_index_fp_desc_dyn::decode()
cuphyStatus_t shm_index_fp_desc_dyn::decode(ldpc::decoder&                     dec,
                                            LDPC_output_t&                     tDst,
                                            const_tensor_pair&                 tLLR,
                                            const cuphy_optional<tensor_pair>& optSoftOutputs,
                                            const cuphyLDPCDecodeConfigDesc_t& config,
                                            cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::shm_index_fp_desc_dyn::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(NUM_CW);
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
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_shm_index_fp_desc_dyn, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG1_shm_index_fp_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_shm_index_fp_desc_dyn, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG2_shm_index_fp_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// shm_index_fp_desc_dyn::decode_tb()
cuphyStatus_t shm_index_fp_desc_dyn::decode_tb(ldpc::decoder&               dec,
                                               const cuphyLDPCDecodeDesc_t& decodeDesc,
                                               cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::shm_index_fp_desc_dyn::decode_tb()\n");
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    dim3 grdDim(ldpc::decoder::get_total_num_codewords(decodeDesc));
    dim3 blkDim(decodeDesc.config.Z);

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = get_shmem_required(decodeDesc.config.BG,
                                                   decodeDesc.config.num_parity_nodes,
                                                   decodeDesc.config.Z);
    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_shm_index_fp_desc_dyn_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG1_shm_index_fp_desc_dyn_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_shm_index_fp_desc_dyn_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG2_shm_index_fp_desc_dyn_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
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
// shm_index_fp_desc_dyn::get_workspace_size()
std::pair<bool, size_t> shm_index_fp_desc_dyn::get_workspace_size(const ldpc::decoder&               dec,
                                                                  const cuphyLDPCDecodeConfigDesc_t& config,
                                                                  int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// shm_index_fp_x2_desc_dyn::can_decode_config()
bool shm_index_fp_desc_dyn::can_decode_config(const ldpc::decoder&               dec,
                                              const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Compare shared memory requirements to device maximum
    const uint32_t SHMEM_BYTES = shmem_size_for_selected_kernel(
        get_shmem_required(cfg.BG, cfg.num_parity_nodes, cfg.Z),
        cfg.flags);
    return SHMEM_BYTES <= dec.max_shmem_per_block_optin();
}

////////////////////////////////////////////////////////////////////////
// shm_index_fp_desc_dyn::shm_index_fp_desc_dyn()
shm_index_fp_desc_dyn::shm_index_fp_desc_dyn(ldpc::decoder& dec)
{
    //------------------------------------------------------------------
    // Determine the maximum shared memory size that a launch might
    // require, and set the kernel attribute accordingly.
    const int MAX_BG1_SHMEM_SIZE = std::min(get_shmem_required(1,
                                                               shm_max_parity_nodes<1>::value,
                                                               CUPHY_LDPC_MAX_LIFTING_SIZE),
                                            dec.max_shmem_per_block_optin());
    const int MAX_BG2_SHMEM_SIZE = std::min(get_shmem_required(2,
                                                               shm_max_parity_nodes<2>::value,
                                                               CUPHY_LDPC_MAX_LIFTING_SIZE),
                                            dec.max_shmem_per_block_optin());
    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 6> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_shm_index_fp_desc_dyn,    MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_shm_index_fp_desc_dyn,    MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_shm_index_fp_desc_dyn_tb, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_shm_index_fp_desc_dyn_tb_no_accessories, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_shm_index_fp_desc_dyn_tb, MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_shm_index_fp_desc_dyn_tb_no_accessories, MAX_BG2_SHMEM_SIZE)
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_shm_index_fp_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_shm_index_fp_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_shm_index_fp_desc_dyn_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_shm_index_fp_desc_dyn_tb);
}

////////////////////////////////////////////////////////////////////////
// shm_index_fp_desc_dyn::get_launch_config()
cuphyStatus_t shm_index_fp_desc_dyn::get_launch_config(const ldpc::decoder&           dec,
                                                       cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    //------------------------------------------------------------------
    // Validate input arguments
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    const uint32_t SHMEM_SIZE = get_shmem_required(launchConfig.decode_desc.config.BG,
                                                   launchConfig.decode_desc.config.num_parity_nodes,
                                                   launchConfig.decode_desc.config.Z);

    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);
    if(launchConfig.kernel_node_params_driver.sharedMemBytes > dec.max_shmem_per_block_optin())
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    cudaError_t    e = (BG == 1) ?  LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG1_shm_index_fp_desc_dyn_tb, launchConfig.decode_desc.config.flags):
                                    LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG2_shm_index_fp_desc_dyn_tb, launchConfig.decode_desc.config.flags);
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
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
