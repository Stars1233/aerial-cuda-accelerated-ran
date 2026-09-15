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

// ALGO 56: Low-p box-plus x2 decoder for BG1, p<=5.
//
// Extracted from ALGO 55 (split_index_bp_x2_desc_dyn). Uses min-sum
// for the 4 core rows (degree 19) and box-plus for non-core, with
// NUM_REG_PARITY_ROWS = MAX_PARITY_ROWS = 5 so all parity rows live
// in registers (no shmem C2V overflow). Combined with
// __launch_bounds__(384, 2), this targets 2 CTAs/SM for better
// occupancy at small parity node counts. BG1 only.

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_box_plus_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_index_bp_x2_lowp.hpp"
#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_crc_dispatch.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM_LOWP = 2;

    //------------------------------------------------------------------
    // Low-p specialization constants
    static constexpr int LOWP_NUM_REG_PARITY = 5;
    static constexpr int LOWP_MAX_PARITY_BG1 = 5;

    //------------------------------------------------------------------
    // Non-core C2V storage: box_plus for BG1 non-core rows
    static constexpr int BG1_MAX_BOX_PLUS_WORDS = 9;
    typedef c2v_storage_x2_box_plus<BG1_MAX_BOX_PLUS_WORDS> noncore_storage_t;

    //------------------------------------------------------------------
    // Sign manager for compressed C2V row processor
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    // APP address calculation
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
    // Kernel configuration for low-p variant.
    // NUM_REG_PARITY_ROWS = MAX_PARITY_ROWS = 5 (all parity rows in
    // registers), BG1 only. Min-sum core (not box-plus) to keep
    // register count low.
    template <class TKernelParams>
    struct lowp_kernel_config
    {
        static constexpr int BG                  = 1; // BG1 only
        static constexpr int MIN_PARITY_ROWS     = 4;
        static constexpr int NUM_REG_PARITY_ROWS = LOWP_NUM_REG_PARITY;
        static constexpr int MAX_PARITY_ROWS     = LOWP_MAX_PARITY_BG1;

        typedef TKernelParams                           kernel_params_t;

        template <int   BG,
                  int   CHECK_IDX,
                  class TC2VStorage> using cC2V_row_map_t = hybrid_storage_row_map_x2<BG,
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
                                       typename core_storage_x2<BG>::type, // min-sum core
                                       noncore_storage_t,
                                       kernel_params_t> c2v_cache_t;

        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        typedef llr_loader_t::app_buf_t                                    app_buf_t;

        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<1>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<1>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };

    //------------------------------------------------------------------
    // get_app_c2v_shmem()
    // Returns the number of bytes required for APP and C2V memory.
    // With LOWP_NUM_REG_PARITY=5 and max_parity=5, non-core C2V in shmem
    // is (5-5)*Z*sizeof(noncore_storage_t) = 0 (all rows in registers).
    CUDA_BOTH
    int get_app_c2v_shmem(int num_parity_nodes, int Z)
    {
        const     int32_t NUM_VAR_NODES = ldpc2::max_info_nodes<1>::value + num_parity_nodes;
        constexpr int32_t NUM_REG_NODES = LOWP_NUM_REG_PARITY;

        int32_t offset = static_cast<int32_t>(shmem_llr_buffer_size(NUM_VAR_NODES,
                                                                     Z,
                                                                     sizeof(__half2)));

        // Core is min-sum (small), stays in registers — no core shmem needed.
        // sizeof(core_storage_x2<1>::type) = 16 = 4*sizeof(word_t), so
        // CORE_IN_SHMEM = false.

        // Non-core C2V: rows beyond NUM_REG_NODES go to shmem
        const int32_t C2V_SIZE = (num_parity_nodes > NUM_REG_NODES)                                ?
                                 (num_parity_nodes - NUM_REG_NODES) * Z * sizeof(noncore_storage_t) :
                                 0;
        int shmem_size = round_up_to_next(offset, static_cast<int>(alignof(noncore_storage_t))) +
                                          C2V_SIZE;
        return shmem_size;
    }

    //------------------------------------------------------------------
    // get_shmem_required()
    CUDA_BOTH
    int get_shmem_required(int num_parity_nodes, int Z)
    {
        int shmem_size = get_app_c2v_shmem(num_parity_nodes, Z);
#if LDPC_DECODE_USE_TB_SCAN
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }

#if LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // get_token_addr()
    __device__
    tb_token* get_token_addr(int num_parity_nodes, int Z, char* smem)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_c2v_shmem(num_parity_nodes, Z));
    }
    __device__
    tb_token* get_token_addr(const cuphyLDPCDecodeDesc_t& decodeDesc, char* smem)
    {
        return get_token_addr(decodeDesc.config.num_parity_nodes,
                              decodeDesc.config.Z,
                              smem);
    }
#endif

} // namespace

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_index_bp_x2_lowp()
// Low-p BG1 kernel (legacy tensor interface): min-sum core, reduced
// register allocation, targeting 2 CTAs/SM.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM_LOWP)
void ldpc2_BG1_index_bp_x2_lowp(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef lowp_kernel_config<ldpc2::LDPC_kernel_params> kernel_config_t;

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
// ldpc2_BG1_index_bp_x2_lowp_tb()
// Low-p BG1 TB kernel: min-sum core, targeting 2 CTAs/SM.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM_LOWP)
void ldpc2_BG1_index_bp_x2_lowp_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef lowp_kernel_config<cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr(decodeDesc, smem));
    ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    // Pre-compute output params and copy config locally to break the
    // reference chain to the large decodeDesc during the iteration loop.
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(config.num_parity_nodes, config.Z));
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
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM_LOWP)
void ldpc2_BG1_index_bp_x2_lowp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef lowp_kernel_config<cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr(decodeDesc, smem));
    ldpc_dec_output_params<__half2>      hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    // Pre-compute output params and copy config locally to break the
    // reference chain to the large decodeDesc during the iteration loop.
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        get_shmem_required(config.num_parity_nodes, config.Z));
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

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// index_bp_x2_lowp::index_bp_x2_lowp()
index_bp_x2_lowp::index_bp_x2_lowp(ldpc::decoder& dec)
{
    const int MAX_SHMEM_LOWP = static_cast<int>(get_shmem_required(LOWP_MAX_PARITY_BG1,
                                                                    CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();

    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 3> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_index_bp_x2_lowp,    std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_SHMEM_LOWP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_bp_x2_lowp_tb, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_SHMEM_LOWP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_bp_x2_lowp_tb_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_SHMEM_LOWP)), MAX_SHMEM))
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_bp_x2_lowp);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_bp_x2_lowp_tb);
}

////////////////////////////////////////////////////////////////////////
// index_bp_x2_lowp::decode()
cuphyStatus_t index_bp_x2_lowp::decode(ldpc::decoder&                     dec,
                                             LDPC_output_t&                     tDst,
                                             const_tensor_pair&                 tLLR,
                                             const cuphy_optional<tensor_pair>& optSoftOutputs,
                                             const cuphyLDPCDecodeConfigDesc_t& config,
                                             cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::index_bp_x2_lowp::decode()\n");

    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    dim3 grdDim(div_round_up(NUM_CW, 2));
    dim3 blkDim(config.Z);

    LDPC_kernel_params params(config, tLLR, tDst, optSoftOutputs);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    if((llrType == CUPHY_R_16F) &&
       (config.BG == 1) &&
       (config.num_parity_nodes <= LOWP_MAX_PARITY_BG1))
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
        if(!bgdesc) return CUPHY_STATUS_INTERNAL_ERROR;

        const uint32_t SHMEM_SIZE = get_shmem_required(config.num_parity_nodes, config.Z);

        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_bp_x2_lowp, blkDim, SHMEM_SIZE);
        ldpc2_BG1_index_bp_x2_lowp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// index_bp_x2_lowp::decode_tb()
cuphyStatus_t index_bp_x2_lowp::decode_tb(ldpc::decoder&               dec,
                                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::index_bp_x2_lowp::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));

    if((decodeDesc.config.llr_type == CUPHY_R_16F) &&
       (decodeDesc.config.BG == 1) &&
       (decodeDesc.config.num_parity_nodes <= LOWP_MAX_PARITY_BG1))
    {
        dim3 blkDim(decodeDesc.config.Z);
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        const uint32_t SHMEM_SIZE = get_shmem_required(decodeDesc.config.num_parity_nodes,
                                                        decodeDesc.config.Z);
        const uint32_t TB_SHMEM_SIZE =
            shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE);
        const uint32_t SELECTED_SHMEM_SIZE =
            shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
                TB_SHMEM_SIZE, decodeDesc.config.flags);

        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
        if(bgdesc && (static_cast<int>(SELECTED_SHMEM_SIZE) * 2 <= dec.max_shmem_per_block_optin()))
        {
            DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_bp_x2_lowp_tb, blkDim, SHMEM_SIZE);
            LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_index_bp_x2_lowp_tb, decodeDesc.config.flags,
                grdDim, blkDim, TB_SHMEM_SIZE, strm, decodeDesc, *bgdesc);
            s = CUPHY_STATUS_SUCCESS;
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
// index_bp_x2_lowp::get_workspace_size()
std::pair<bool, size_t> index_bp_x2_lowp::get_workspace_size(const ldpc::decoder&               dec,
                                                                   const cuphyLDPCDecodeConfigDesc_t& config,
                                                                   int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// index_bp_x2_lowp::can_decode_config()
bool index_bp_x2_lowp::can_decode_config(const ldpc::decoder&               dec,
                                               const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // BG1 only, p<=5, and 2 CTAs must simultaneously fit in shmem
    if(cfg.BG != 1 || cfg.num_parity_nodes > LOWP_MAX_PARITY_BG1)
    {
        return false;
    }
    if(!app_loc_t<1>::get_bg_desc(cfg.Z))
    {
        return false;
    }
    const uint32_t SHMEM_SIZE = get_shmem_required(cfg.num_parity_nodes, cfg.Z);
    const uint32_t SELECTED_SHMEM_SIZE =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE),
            cfg.flags);
    return (static_cast<int>(SELECTED_SHMEM_SIZE) * 2 <= dec.max_shmem_per_block_optin());
}

////////////////////////////////////////////////////////////////////////
// index_bp_x2_lowp::get_launch_config()
cuphyStatus_t index_bp_x2_lowp::get_launch_config(const ldpc::decoder&           dec,
                                                        cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;

    //------------------------------------------------------------------
    // Validate: BG1 only, small parity count
    if((BG != 1)                                ||
       (Z < 2)                                  ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)        ||
       (NUM_PARITY_NODES < 4)                   ||
       (NUM_PARITY_NODES > LOWP_MAX_PARITY_BG1))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    //------------------------------------------------------------------
    // Set up launch geometry
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codeword_pairs(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra        = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams = launchConfig.kernel_args;

    const uint32_t SHMEM_SIZE = get_shmem_required(NUM_PARITY_NODES, Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE);
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);
    if(static_cast<int>(launchConfig.kernel_node_params_driver.sharedMemBytes) * 2 > dec.max_shmem_per_block_optin())
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(Z);
    if(!bgdesc)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    cudaError_t e = LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG1_index_bp_x2_lowp_tb, launchConfig.decode_desc.config.flags);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif

    //------------------------------------------------------------------
    // Set kernel arguments
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));

    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
