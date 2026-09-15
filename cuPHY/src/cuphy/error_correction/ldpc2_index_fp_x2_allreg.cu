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

// ALGO 51: All-register x2 min-sum decoder for high parity node counts.
//
// Identical to ALGO 35 except NUM_REG_PARITY_ROWS = MAX_PARITY_ROWS,
// so ALL C2V data resides in registers and shared memory is APP-only.
// This enables 2 CW/CTA (x2) decoding at parity counts where ALGO 35's
// normal C2V shmem overflow exceeds the device limit. Trades register
// spills for halved CTA count.
//
// Coverage: BG1 p up to ~43 (Z=384, 99 KB shmem). BG2 is identical to
// ALGO 35 (already all-register) and is included for completeness.

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_index_fp_x2_allreg.hpp"
#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_crc_dispatch.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // All parity rows stored in registers.
    template <int BG> struct max_num_parity;
    template <> struct max_num_parity<1> { static constexpr int value = 46; };
    template <> struct max_num_parity<2> { static constexpr int value = 42; };

    //------------------------------------------------------------------
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half2, BG>;

    //------------------------------------------------------------------
    template <class TStorage> using row_context_t = cC2V_row_context<__half2,
                                                                     sign_mgr_t,
                                                                     unused,
                                                                     TStorage>;
    //------------------------------------------------------------------
    template <class TRowContext> using cC2V_row_proc_t = cC2V_row_proc<__half2,
                                                                       TRowContext>;

    //------------------------------------------------------------------
    // Kernel configuration: NUM_REG_PARITY_ROWS = MAX_PARITY so all C2V
    // is register-resident. Shared memory holds only APP values.
    // MIN_P_ raises MIN_PARITY_ROWS so the schedule's IS_LAST_ROW
    // runtime check can be eliminated at compile time for the early
    // rows when a high-p variant is dispatched.
    template <int   BG_,
              class TKernelParams,
              int   MIN_P_ = 4>
    struct kernel_config
    {
        static constexpr int BG                  = BG_;
        static constexpr int MIN_PARITY_ROWS     = MIN_P_;
        static constexpr int NUM_REG_PARITY_ROWS = max_num_parity<BG>::value;
        static constexpr int MAX_PARITY_ROWS     = max_num_parity<BG>::value;

        typedef TKernelParams                           kernel_params_t;

        template <int   BG,
                  int   CHECK_IDX,
                  class TC2VStorage> using cC2V_row_map_t = context_storage_row_map<BG,
                                                                                    CHECK_IDX,
                                                                                    TC2VStorage,
                                                                                    __half2,
                                                                                    row_context_t,
                                                                                    cC2V_row_proc_t>;
        typedef C2V_row_proc<__half2,
                             BG,
                             cC2V_row_map_t,
                             app_loader,
                             app_writer> C2V_t;

        typedef ldpc2::c2v_cache_split<BG,
                                       NUM_REG_PARITY_ROWS,
                                       C2V_t,
                                       typename core_storage_x2<BG>::type,
                                       cC2V_storage_x2_low_degree,
                                       kernel_params_t> c2v_cache_t;

        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        typedef llr_loader_t::app_buf_t                                    app_buf_t;

        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };

    //------------------------------------------------------------------
    // Shared memory = APP only (no C2V overflow).
    template <int BG>
    CUDA_BOTH
    int get_app_shmem(int num_parity_nodes, int Z)
    {
        const int32_t NUM_VAR_NODES = ldpc2::max_info_nodes<BG>::value + num_parity_nodes;
        return static_cast<int32_t>(shmem_llr_buffer_size(NUM_VAR_NODES, Z, sizeof(__half2)));
    }
    //------------------------------------------------------------------
    CUDA_BOTH
    int get_shmem_required(int BG, int num_parity_nodes, int Z)
    {
        int shmem_size = (1 == BG) ? get_app_shmem<1>(num_parity_nodes, Z)
                                   : get_app_shmem<2>(num_parity_nodes, Z);
#if LDPC_DECODE_USE_TB_SCAN
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }
#if LDPC_DECODE_USE_TB_SCAN
    template <int BG>
    __device__
    tb_token* get_token_addr(int num_parity_nodes, int Z, char* smem)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_shmem<BG>(num_parity_nodes, Z));
    }
    template <int BG>
    __device__
    tb_token* get_token_addr(const cuphyLDPCDecodeDesc_t& decodeDesc, char* smem)
    {
        return get_token_addr<BG>(decodeDesc.config.num_parity_nodes,
                                  decodeDesc.config.Z,
                                  smem);
    }
#endif // if LDPC_DECODE_USE_TB_SCAN
} // namespace

////////////////////////////////////////////////////////////////////////
// HIGH_P_THRESHOLD: parity node count at/above which the high-p kernel
// variants are dispatched. Setting MIN_PARITY_ROWS = HIGH_P_THRESHOLD
// makes the schedule's IS_LAST_ROW runtime check vanish at compile time
// for rows below the threshold.
namespace { const int HIGH_P_THRESHOLD = 22; }

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_index_fp_x2_allreg()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<1, ldpc2::LDPC_kernel_params> kernel_config_t;

    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_index_fp_x2_allreg()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<2, ldpc2::LDPC_kernel_params> kernel_config_t;

    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_index_fp_x2_allreg_tb()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<1, cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    //------------------------------------------------------------------
    // Pre-compute output params before iterations so the compiler can
    // release the large decodeDesc (TB arrays) during the decode loop.
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<1, cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    //------------------------------------------------------------------
    // Pre-compute output params before iterations so the compiler can
    // release the large decodeDesc (TB arrays) during the decode loop.
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_index_fp_x2_allreg_tb()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<2, cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    //------------------------------------------------------------------
    // Pre-compute output params before iterations so the compiler can
    // release the large decodeDesc (TB arrays) during the decode loop.
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<2>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<2, cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    //------------------------------------------------------------------
    // Pre-compute output params before iterations so the compiler can
    // release the large decodeDesc (TB arrays) during the decode loop.
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<2>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// High-p variants. MIN_PARITY_ROWS = HIGH_P_THRESHOLD lets the schedule's
// per-row IS_LAST_ROW runtime check evaporate at compile time for rows
// below the threshold. Only safe to launch when the runtime
// num_parity_nodes >= HIGH_P_THRESHOLD; the dispatch enforces this.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_highp(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<1, ldpc2::LDPC_kernel_params, HIGH_P_THRESHOLD> kernel_config_t;

    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg_highp(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<2, ldpc2::LDPC_kernel_params, HIGH_P_THRESHOLD> kernel_config_t;

    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_highp_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<1, cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_highp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<1, cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg_highp_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<2, cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<2>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_index_fp_x2_allreg_highp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<2, cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<2>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(kernel_config_t::BG, config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < config.max_iterations)
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

    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg::decode()
cuphyStatus_t index_fp_x2_allreg::decode(ldpc::decoder&                     dec,
                                                LDPC_output_t&                     tDst,
                                                const_tensor_pair&                 tLLR,
                                                const cuphy_optional<tensor_pair>& optSoftOutputs,
                                                const cuphyLDPCDecodeConfigDesc_t& config,
                                                cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::index_fp_x2_allreg::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(div_round_up(NUM_CW, 2));
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    LDPC_kernel_params params(config, tLLR, tDst, optSoftOutputs);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    const uint32_t SHMEM_SIZE = get_shmem_required(config.BG,
                                                   config.num_parity_nodes,
                                                   config.Z);

    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                if(config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_highp, blkDim, SHMEM_SIZE);
                    ldpc2_BG1_index_fp_x2_allreg_highp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg, blkDim, SHMEM_SIZE);
                    ldpc2_BG1_index_fp_x2_allreg<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                }
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                if(config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_index_fp_x2_allreg_highp, blkDim, SHMEM_SIZE);
                    ldpc2_BG2_index_fp_x2_allreg_highp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_index_fp_x2_allreg, blkDim, SHMEM_SIZE);
                    ldpc2_BG2_index_fp_x2_allreg<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// index_fp_x2_allreg::decode_tb()
cuphyStatus_t index_fp_x2_allreg::decode_tb(ldpc::decoder&               dec,
                                                   const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                   cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::index_fp_x2_allreg::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        dim3 blkDim(decodeDesc.config.Z);
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        const uint32_t SHMEM_SIZE = get_shmem_required(decodeDesc.config.BG,
                                                       decodeDesc.config.num_parity_nodes,
                                                       decodeDesc.config.Z);
        const uint32_t TB_SHMEM_SIZE =
            shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE);
        const uint32_t SELECTED_SHMEM_SIZE =
            shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
                TB_SHMEM_SIZE, decodeDesc.config.flags);
        if(SELECTED_SHMEM_SIZE > dec.max_shmem_per_block_optin())
        {
            return s;
        }
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                if(decodeDesc.config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_highp_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_index_fp_x2_allreg_highp_tb, decodeDesc.config.flags,
                        grdDim, blkDim, TB_SHMEM_SIZE, strm, decodeDesc, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_index_fp_x2_allreg_tb, decodeDesc.config.flags,
                        grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                }
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                if(decodeDesc.config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_index_fp_x2_allreg_highp_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG2_index_fp_x2_allreg_highp_tb, decodeDesc.config.flags,
                        grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_index_fp_x2_allreg_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG2_index_fp_x2_allreg_tb, decodeDesc.config.flags,
                        grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
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
// index_fp_x2_allreg::get_workspace_size()
std::pair<bool, size_t> index_fp_x2_allreg::get_workspace_size(const ldpc::decoder&               dec,
                                                                      const cuphyLDPCDecodeConfigDesc_t& config,
                                                                      int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg::index_fp_x2_allreg()
index_fp_x2_allreg::index_fp_x2_allreg(ldpc::decoder& dec)
{
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(get_shmem_required(1,
                                                                        max_num_parity<1>::value,
                                                                        CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(get_shmem_required(2,
                                                                        max_num_parity<2>::value,
                                                                        CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();

    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 12> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg,          std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg,          std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_tb,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_tb_no_accessories,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg_tb,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg_tb_no_accessories,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_highp,    std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg_highp,    std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_highp_tb, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_highp_tb_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg_highp_tb, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_index_fp_x2_allreg_highp_tb_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM))
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_fp_x2_allreg);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_index_fp_x2_allreg);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_fp_x2_allreg_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_index_fp_x2_allreg_tb);
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg::can_decode_config()
bool index_fp_x2_allreg::can_decode_config(const ldpc::decoder&               dec,
                                                  const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    const uint32_t MAX_NUM_PARITY = (1 == cfg.BG) ? max_num_parity<1>::value
                                                   : max_num_parity<2>::value;
    const uint32_t SHMEM_BYTES    = get_shmem_required(cfg.BG,
                                                        cfg.num_parity_nodes,
                                                        cfg.Z);
    const uint32_t SELECTED_SHMEM_BYTES =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_BYTES),
            cfg.flags);
    return (cfg.num_parity_nodes <= MAX_NUM_PARITY) &&
           (SELECTED_SHMEM_BYTES <= dec.max_shmem_per_block_optin());
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg::get_launch_config()
cuphyStatus_t index_fp_x2_allreg::get_launch_config(const ldpc::decoder&           dec,
                                                           cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    //------------------------------------------------------------------
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;

    const uint32_t SHMEM_SIZE = get_shmem_required(BG, NUM_PARITY_NODES, Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE);
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);
    if(launchConfig.kernel_node_params_driver.sharedMemBytes > dec.max_shmem_per_block_optin())
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const void* bg1_kernel = (NUM_PARITY_NODES >= HIGH_P_THRESHOLD) ?
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_index_fp_x2_allreg_highp_tb, launchConfig.decode_desc.config.flags) :
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_index_fp_x2_allreg_tb, launchConfig.decode_desc.config.flags);
    const void* bg2_kernel = (NUM_PARITY_NODES >= HIGH_P_THRESHOLD) ?
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG2_index_fp_x2_allreg_highp_tb, launchConfig.decode_desc.config.flags) :
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG2_index_fp_x2_allreg_tb, launchConfig.decode_desc.config.flags);
    cudaError_t    e = (BG == 1) ? cudaGetFuncBySymbol(&deviceFunction, bg1_kernel)
                                 : cudaGetFuncBySymbol(&deviceFunction, bg2_kernel);
    if (e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif
    //------------------------------------------------------------------
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
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
