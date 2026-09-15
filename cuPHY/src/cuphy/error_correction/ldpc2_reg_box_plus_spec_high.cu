/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ldpc2_bg_desc.hpp"
#include "ldpc2_c2v.cuh"
#include "ldpc2_app_address_spec.cuh"
#include "ldpc2_box_plus.cuh"
#include "ldpc2_reg_box_plus_spec_high.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_register.cuh"
#include "ldpc2_llr_loader.cuh"
#include "ldpc2_dec_output.cuh"
#include "ldpc2_crc_dispatch.cuh"

using namespace ldpc2;

#ifndef CUPHY_LDPC_REVERSE_ROWS
#define CUPHY_LDPC_REVERSE_ROWS 0
#endif

#if CUPHY_LDPC_REVERSE_ROWS
#define CUPHY_LDPC_BOX_PLUS_SCHEDULE ldpc2::ldpc_schedule_dynamic_desc_reverse
#else
#define CUPHY_LDPC_BOX_PLUS_SCHEDULE ldpc2::ldpc_schedule_dynamic_desc
#endif

namespace
{
    // The lifting size this kernel is specialized for. It cannot be run for any
    // other lifting sizes.
    const int SPEC_Z = 384;

    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Maximum number of parity nodes supported by this kernel
    //const int MAX_NUM_PARITY_BG1      = 46;
    const int MAX_NUM_PARITY_BG1      = 37;
    const int MAX_NUM_PARITY_BG2      = 42;

    //------------------------------------------------------------------
    // Number of per-row storage words
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG1 = 10;
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG2 = 5;

    null_BG_desc_t<1> null_desc_1;
    null_BG_desc_t<2> null_desc_2;

    // APP address calculation
    template <int BG> using app_loc_t = app_loc_address_gen_spec<__half, BG, SPEC_Z>;

    //------------------------------------------------------------------
    // box_plus_all_row_map_t
    // The C2V_row_proc template requires a row map template with template
    // arguments BG (int), CHECK_IDX (int), TStorage (per-row storage
    // structure. We use simple_row_map to indicate that all rows should
    // use the same template. (Other kernels might choose differently for
    // different rows.)
    template <int   BG,
              int   CHECK_IDX,
              class TC2VStorage> using box_plus_all_row_map_t = simple_row_map<BG,
                                                                               CHECK_IDX,
                                                                               TC2VStorage,
                                                                               box_plus_row_proc<box_plus_op, __half>>;
    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    // BG_: base graph (1 or 2)
    // TKernelParams: Class/struct used for kernel parameters
    // NUM_STORAGE_WORDS: Number of storage words for each parity row
    // MAX_PARITY_ROWS: Maximum number of parity rows supported by the kernel
    template <int   BG_,
              int   NUM_STORAGE_WORDS,
              int   MAX_PARITY_ROWS,
              class TKernelParams>
    struct ldpc2_reg_box_plus_spec_high_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;

        // C2V per-row storage. Larger storage allows faster row
        // processing, but increases register pressure (and may incur
        // register spills).
        typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS> c2v_storage_t;
        typedef TKernelParams                                   kernel_params_t;

        typedef C2V_row_proc<__half,
                             BG,
                             box_plus_all_row_map_t,
                             app_loader,
                             app_writer>                                  C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (register memory here)
        typedef ldpc2::c2v_cache_register<BG,
                                          MAX_PARITY_ROWS,
                                          C2V_t,
                                          c2v_storage_t,
                                          kernel_params_t>                c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        using llr_loader_t = ldpc2::llr_loader_variable_batch<__half, 4, llr_op_clamp>;
        // Data type in APP shared memory buffer (__half or __half2)
        using app_buf_t = typename llr_loader_t::app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef CUPHY_LDPC_BOX_PLUS_SCHEDULE<BG,
                                                  app_loc_t<BG_>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
} // namespace


////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_box_plus_spec_high()
// Base graph 1 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus_spec_high(LDPC_kernel_params params, null_BG_desc_t<1> bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<1,
                                                       NUM_STORAGE_WORDS_BG1,
                                                       MAX_NUM_PARITY_BG1,
                                                       ldpc2::LDPC_kernel_params> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
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
// ldpc2_BG2_reg_box_plus_spec_high()
// Base graph 2 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_box_plus_spec_high(LDPC_kernel_params params, null_BG_desc_t<2> bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<2, // BG
                                                       NUM_STORAGE_WORDS_BG2,
                                                       MAX_NUM_PARITY_BG2,
                                                       ldpc2::LDPC_kernel_params> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
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
// ldpc2_BG1_reg_box_plus_spec_high_tb()
// Base graph 1 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus_spec_high_tb(cuphyLDPCDecodeDesc_t decodeDesc, null_BG_desc_t<1> bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<1,
                                                       NUM_STORAGE_WORDS_BG1,
                                                       MAX_NUM_PARITY_BG1,
                                                       cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    const uint32_t size_before_et = shmem_llr_buffer_size(
        decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value,
        decodeDesc.config.Z,
        sizeof(kernel_config_t::app_buf_t));
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    const int32_t iter = ldpc2::run_tb_iterations_with_early_termination<ENABLE_ACCESSORY_FEATURES, 1>(
        sched, decodeDesc, blockIdx.x,
        reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
        smem, size_before_et, crc);

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc2::write_early_termination_outputs<ENABLE_ACCESSORY_FEATURES, kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, crc, iter);
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus_spec_high_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, null_BG_desc_t<1> bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<1,
                                                       NUM_STORAGE_WORDS_BG1,
                                                       MAX_NUM_PARITY_BG1,
                                                       cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    const uint32_t size_before_et = shmem_llr_buffer_size(
        decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value,
        decodeDesc.config.Z,
        sizeof(kernel_config_t::app_buf_t));
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    const int32_t iter = ldpc2::run_tb_iterations_with_early_termination<ENABLE_ACCESSORY_FEATURES, 1>(
        sched, decodeDesc, blockIdx.x,
        reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
        smem, size_before_et, crc);

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc2::write_early_termination_outputs<ENABLE_ACCESSORY_FEATURES, kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, crc, iter);
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_box_plus_spec_high_tb()
// Base graph 2 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_box_plus_spec_high_tb(cuphyLDPCDecodeDesc_t decodeDesc, null_BG_desc_t<2> bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<2, // BG
                                                       NUM_STORAGE_WORDS_BG2,
                                                       MAX_NUM_PARITY_BG2,
                                                       cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    const uint32_t size_before_et = shmem_llr_buffer_size(
        decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value,
        decodeDesc.config.Z,
        sizeof(kernel_config_t::app_buf_t));
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    const int32_t iter = ldpc2::run_tb_iterations_with_early_termination<ENABLE_ACCESSORY_FEATURES, 2>(
        sched, decodeDesc, blockIdx.x,
        reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
        smem, size_before_et, crc);

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc2::write_early_termination_outputs<ENABLE_ACCESSORY_FEATURES, kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, crc, iter);
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_box_plus_spec_high_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, null_BG_desc_t<2> bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_spec_high_kernel_config<2, // BG
                                                       NUM_STORAGE_WORDS_BG2,
                                                       MAX_NUM_PARITY_BG2,
                                                       cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    const uint32_t size_before_et = shmem_llr_buffer_size(
        decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value,
        decodeDesc.config.Z,
        sizeof(kernel_config_t::app_buf_t));
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    const int32_t iter = ldpc2::run_tb_iterations_with_early_termination<ENABLE_ACCESSORY_FEATURES, 2>(
        sched, decodeDesc, blockIdx.x,
        reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
        smem, size_before_et, crc);

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc2::write_early_termination_outputs<ENABLE_ACCESSORY_FEATURES, kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, crc, iter);
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
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
// reg_box_plus_spec_high::decode()
cuphyStatus_t reg_box_plus_spec_high::decode(ldpc::decoder&                     dec,
                                             LDPC_output_t&                     tDst,
                                             const_tensor_pair&                 tLLR,
                                             const cuphy_optional<tensor_pair>& optSoftOutputs,
                                             const cuphyLDPCDecodeConfigDesc_t& config,
                                             cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::reg_box_plus_spec_high::decode()\n");
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
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                      params.Z,             // lifting size
                                                      sizeof(__half));      // element size
    if((llrType == CUPHY_R_16F) &&
       (config.Z == SPEC_Z) &&
       (config.num_parity_nodes >= 4) &&
       (((config.BG == 1) && (config.num_parity_nodes <= MAX_NUM_PARITY_BG1)) ||
        ((config.BG == 2) && (config.num_parity_nodes <= MAX_NUM_PARITY_BG2))))
    {
        switch(config.BG)
        {
        case 1:
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_box_plus_spec_high, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG1_reg_box_plus_spec_high<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, null_BG_desc_t<1>{});
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_box_plus_spec_high, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG2_reg_box_plus_spec_high<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, null_BG_desc_t<2>{});
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
// reg_box_plus_spec_high::decode_tb()
cuphyStatus_t reg_box_plus_spec_high::decode_tb(ldpc::decoder&               dec,
                                                const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_box_plus_spec_high::decode_tb()\n");
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    dim3 grdDim(ldpc::decoder::get_total_num_codewords(decodeDesc));
    dim3 blkDim(decodeDesc.config.Z);

    if((decodeDesc.config.llr_type == CUPHY_R_16F) &&
       (decodeDesc.config.Z == SPEC_Z) &&
       (decodeDesc.config.num_parity_nodes >= 4) &&
       (((decodeDesc.config.BG == 1) && (decodeDesc.config.num_parity_nodes <= MAX_NUM_PARITY_BG1)) ||
        ((decodeDesc.config.BG == 2) && (decodeDesc.config.num_parity_nodes <= MAX_NUM_PARITY_BG2))))
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_size_with_et_context(
                    shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value, // num shared memory nodes
                                          decodeDesc.config.Z,                                           // lifting size
                                          sizeof(__half)));                                              // element size

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_box_plus_spec_high_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG1_reg_box_plus_spec_high_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, null_BG_desc_t<1>{});
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_size_with_et_context(
                    shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value, // num shared memory nodes
                                          decodeDesc.config.Z,                                           // lifting size
                                          sizeof(__half)));                                              // element size

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_box_plus_spec_high_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG2_reg_box_plus_spec_high_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, null_BG_desc_t<2>{});
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
// reg_box_plus_spec_high::get_workspace_size()
std::pair<bool, size_t> reg_box_plus_spec_high::get_workspace_size(const ldpc::decoder&               dec,
                                                                   const cuphyLDPCDecodeConfigDesc_t& config,
                                                                   int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus_spec_high::reg_box_plus_spec_high()
reg_box_plus_spec_high::reg_box_plus_spec_high(ldpc::decoder& desc)
{
    const uint32_t MAX_VAR_NODES_BG1 = ldpc2::max_variable_nodes<1>::value;
    const uint32_t MAX_VAR_NODES_BG2 = ldpc2::max_variable_nodes<2>::value;
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(shmem_size_with_et_context(
        shmem_llr_buffer_size(MAX_VAR_NODES_BG1,           // num shared memory nodes
                              CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                              sizeof(__half))));           // element size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(shmem_size_with_et_context(
        shmem_llr_buffer_size(MAX_VAR_NODES_BG2,           // num shared memory nodes
                              CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                              sizeof(__half))));           // element size

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 6> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus_spec_high,    MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_box_plus_spec_high,    MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus_spec_high_tb, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus_spec_high_tb_no_accessories, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_box_plus_spec_high_tb, MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_box_plus_spec_high_tb_no_accessories, MAX_BG2_SHMEM_SIZE)
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_box_plus_spec_high);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_box_plus_spec_high);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_box_plus_spec_high_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_box_plus_spec_high_tb);
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus_spec_high::can_decode_config()
bool reg_box_plus_spec_high::can_decode_config(const ldpc::decoder&               dec,
                                               const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    const int MAX_NUM_PARITY = (1 == cfg.BG) ? MAX_NUM_PARITY_BG1 : MAX_NUM_PARITY_BG2;
    return (cfg.Z == SPEC_Z) && (cfg.num_parity_nodes <= MAX_NUM_PARITY);
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus_spec_high::get_launch_config()
cuphyStatus_t reg_box_plus_spec_high::get_launch_config(const ldpc::decoder&           dec,
                                                        cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)             ?
                                 MAX_NUM_PARITY_BG1 :
                                 MAX_NUM_PARITY_BG2;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    //------------------------------------------------------------------
    // Validate input arguments
    if((Z != SPEC_Z)                        ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

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
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_size_with_et_context(
        shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                              Z,               // lifting size
                              sizeof(__half))); // element size

    cudaFunction_t deviceFunction;
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);
    MemtraceDisableScope md;
    cudaError_t    e = (BG == 1) ?  LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG1_reg_box_plus_spec_high_tb, launchConfig.decode_desc.config.flags):
                                    LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG2_reg_box_plus_spec_high_tb, launchConfig.decode_desc.config.flags);
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
    if(BG == 1)
    {
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(&null_desc_1));
    }
    else
    {
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(&null_desc_2));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
