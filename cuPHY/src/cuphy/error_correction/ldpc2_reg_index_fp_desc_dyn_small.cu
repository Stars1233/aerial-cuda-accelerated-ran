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
#include "ldpc2_reg_index_fp_desc_dyn_small.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_register.cuh"
#include "ldpc2_crc_dispatch.cuh"

using namespace ldpc2;

#define USE_APP_ADDR_FP_DP 1

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 32;
    const int MIN_CTA_PER_SM      = 1;

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

    typedef C2V_storage_t<__half, 2>                                                            c2v_storage_t;
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
    struct ldpc2_reg_index_fp_desc_dyn_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;
        static constexpr int MAX_PARITY_ROWS = ldpc2::max_parity_nodes<BG>::value;

        typedef TKernelParams                   kernel_params_t;

        // We use the "checked" variants of app_loader and app_writer, since we may
        // have extra threads that are not participating in a valid codeword.
        typedef cC2V_index<__half,
                           BG,
                           sign_mgr_t,
                           min_sum_update_half_0,
                           app_loader_checked,
                           app_writer_checked> C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (register memory here)
        typedef ldpc2::c2v_cache_register<BG,
                                          MAX_PARITY_ROWS,
                                          C2V_t,
                                          c2v_storage_t,
                                          kernel_params_t>                      c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef TLoader<TLLR, 4, llr_op_clamp>                                  llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef typename llr_loader_t::app_buf_t                                app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG_>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };

    __device__ __forceinline__
    tb_token small_decode_desc_token_from_global_cw(const cuphyLDPCDecodeDesc_t& decodeDesc, int global_cw)
    {
        int cw = global_cw;
        #pragma unroll
        for(int tb = 0; tb < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++tb)
        {
            if(tb < decodeDesc.num_tbs)
            {
                const int num_cw = decodeDesc.llr_input[tb].num_codewords;
                if(cw < num_cw)
                {
                    return to_token_partial(tb, cw, false);
                }
                cw -= num_cw;
            }
        }
        return to_token_partial(0, 0, false);
    }

    __device__ __forceinline__
    uint32_t small_codeword_thread_mask(int Z, int thread_codeword_index)
    {
        return (Z == 32) ? 0xFFFFFFFFu : (((1u << Z) - 1u) << (thread_codeword_index * Z));
    }

    struct small_crc_layout
    {
        static constexpr bool needs_app_read_barrier = false;

        int total_bits_;
        int thread_codeword_index;
        int thread_sub_index;
        int Z;

        __device__ __forceinline__ int total_bits() const
        {
            return total_bits_;
        }

        __device__ __forceinline__ int total_words() const
        {
            return (total_bits_ + 31) >> 5;
        }

        __device__ __forceinline__ int word_index() const
        {
            return thread_sub_index;
        }

        __device__ __forceinline__ bool active_word() const
        {
            return word_index() < total_words();
        }

        __device__ __forceinline__ uint32_t group_mask() const
        {
            return small_codeword_thread_mask(Z, thread_codeword_index);
        }

        __device__ __forceinline__ bool any_bit_flip(bool bit_flip) const
        {
            return __any_sync(group_mask(), bit_flip);
        }

        template <cuphyLDPCCrcType_t CRCVariant, uint32_t T, uint64_t QPLUSCRC, uint32_t GSTAR>
        __device__ __forceinline__ uint32_t reduce_crc(uint32_t packed_word,
                                                        bool     active_word,
                                                        int      word_idx) const
        {
            uint32_t partial_crc = 0u;
            if(active_word)
            {
                partial_crc = crc_word_partial<CRCVariant, T, QPLUSCRC, GSTAR>(packed_word,
                                                                                crc_static_lut_value<CRCVariant>(word_idx));
            }
            return __reduce_xor_sync(group_mask(), partial_crc);
        }
    };

    template <cuphyLDPCCrcType_t CRCVariant, int BG>
    __device__ __forceinline__
    uint32_t small_crc_after_bitflip_gate(const __half* apps,
                                          int           iter,
                                          uint32_t&     prev_packed_word,
                                          int           Z,
                                          int           num_info_nodes,
                                          int           thread_codeword_index,
                                          int           thread_sub_index,
                                          bool          force_crc,
                                          bool          force_bit_flip)
    {
        (void)BG;
        const small_crc_layout layout{num_info_nodes * Z, thread_codeword_index, thread_sub_index, Z};
        return crc_after_bitflip_gate<CRCVariant>(apps, iter, prev_packed_word, layout, force_crc, force_bit_flip);
    }

    template <int BG>
    __device__ __forceinline__
    uint32_t small_should_terminate_early_crc(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                              tb_token                     tok,
                                              const __half*                apps,
                                              int                          iter,
                                              uint32_t&                    prev_packed_word,
                                              int                          thread_codeword_index,
                                              int                          thread_sub_index)
    {
        const bool final_iteration = (iter >= (decodeDesc.config.max_iterations - 1));
        const bool et_latency_debug = (0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG));
        const bool force_crc = final_iteration;
        const bool force_bit_flip = et_latency_debug && !final_iteration;

        switch(decode_desc_crc_type(decodeDesc, tok, 0))
        {
            case CUPHY_LDPC_CRC_16:
                return small_crc_after_bitflip_gate<CUPHY_LDPC_CRC_16, BG>(apps, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb,
                                                                                                  thread_codeword_index,
                                                                                                  thread_sub_index,
                                                                                                  force_crc,
                                                                                                  force_bit_flip);
            case CUPHY_LDPC_CRC_24A:
                return small_crc_after_bitflip_gate<CUPHY_LDPC_CRC_24A, BG>(apps, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb,
                                                                                                  thread_codeword_index,
                                                                                                  thread_sub_index,
                                                                                                  force_crc,
                                                                                                  force_bit_flip);
            case CUPHY_LDPC_CRC_24B:
                return small_crc_after_bitflip_gate<CUPHY_LDPC_CRC_24B, BG>(apps, iter, prev_packed_word, decodeDesc.config.Z, decodeDesc.config.Kb,
                                                                                                  thread_codeword_index,
                                                                                                  thread_sub_index,
                                                                                                  force_crc,
                                                                                                  force_bit_flip);
            case CUPHY_LDPC_CRC_NONE:
            default:
                return CRC_FAILURE_WORD;
        }
    }

    __device__ __forceinline__
    void small_ldpc_dec_crc_output(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok, uint32_t crc)
    {
        uint32_t* addr = decode_desc_crc_addr(decodeDesc, tok, 0);
        if(addr != nullptr)
        {
            *addr = crc;
        }
    }

    __device__ __forceinline__
    void small_ldpc_dec_iter_output(const cuphyLDPCDecodeDesc_t& decodeDesc, tb_token tok, int32_t iter)
    {
        int32_t* addr = decode_desc_iter_output_addr(decodeDesc, tok, 0);
        if(addr != nullptr)
        {
            *addr = iter;
        }
    }
} // namespace


////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small()
// Base graph 1 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);
    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

#if 0
    print_array_sync("LLR:", reinterpret_cast<__half*>(smem), 960);
#endif
    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output_multi(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small()
// Base graph 2 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      ldpc2::LDPC_kernel_params,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output_multi(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_tb()
// Base graph 1 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_tb()
// Base graph 2 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                  (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                  (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                  -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      __half,
                                                      ldpc2::llr_loader_variable_batch> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                  (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                  (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                  -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32()
// Base graph 1 kernel, "legacy" tensor interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      ldpc2::LDPC_kernel_params,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;
    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output_multi(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32()
// Base graph 2 kernel, "legacy" tensor interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      ldpc2::LDPC_kernel_params,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(params, blockIdx.x, params.num_codewords);

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, params, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(params) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(params,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output_multi(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32()
// Base graph 1 kernel, transport block interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<1,
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET       = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))           :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32()
// Base graph 2 kernel, transport block interface, fp32->fp16 on input
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_index_fp_desc_dyn_kernel_config<2, // BG
                                                      cuphyLDPCDecodeConfigDesc_t,
                                                      float,
                                                      ldpc2::llr_loader_variable_batch_convert> kernel_config_t;

    //------------------------------------------------------------------
    // Determine which codewords this block and this thread are assigned
    // to.
    multi_codeword_config mconfig(decodeDesc.config, blockIdx.x, get_num_codewords(decodeDesc));

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync_multi(smem, decodeDesc, mconfig);

    //------------------------------------------------------------------
    // Determine the shared memory offset of the LLR data for this
    // thread. "Unused" threads will have a negative offset, and the
    // "checked" APP loader and writer will prevent reads/writes beyond
    // the shared memory allocation.
    const int LLR_STRIDE_BYTES = round_up_to_next(get_num_LLRs(decodeDesc) * sizeof(__half),
                                                  sizeof(ldpc_traits<__half>::llr_sts_t));
    const int SMEM_OFFSET      = (mconfig.thread_codeword_index < mconfig.cta_codeword_count) ?
                                 (static_cast<int>(__cvta_generic_to_shared(smem)) +
                                 (mconfig.thread_codeword_index * LLR_STRIDE_BYTES))          :
                                 -1;

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config, bgdesc, SMEM_OFFSET, mconfig.thread_sub_index);
    int32_t iter = 0;
    uint32_t crc = CRC_FAILURE_WORD;
    uint32_t prev_packed_word = 0u;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;

        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            bool cw_crc_pass = false;
            if(mconfig.thread_codeword_index < mconfig.cta_codeword_count)
            {
                const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
                const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
                const __half* cw_apps = reinterpret_cast<const __half*>(smem + (mconfig.thread_codeword_index * LLR_STRIDE_BYTES));
                crc = small_should_terminate_early_crc<kernel_config_t::BG>(decodeDesc,
                                                                            tok,
                                                                            cw_apps,
                                                                            iter - 1,
                                                                            prev_packed_word,
                                                                            mconfig.thread_codeword_index,
                                                                            mconfig.thread_sub_index);
                cw_crc_pass = (mconfig.thread_sub_index == 0) && (crc == 0u);
            }
            if(__syncthreads_count(cw_crc_pass) == mconfig.cta_codeword_count)
            {
                break;
            }
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_multi(decodeDesc,
                                   reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                                   mconfig);
    if((mconfig.thread_codeword_index < mconfig.cta_codeword_count) &&
       (mconfig.thread_sub_index == 0))
    {
        const int global_cw = mconfig.cta_start_index + mconfig.thread_codeword_index;
        const tb_token tok = small_decode_desc_token_from_global_cw(decodeDesc, global_cw);
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            small_ldpc_dec_crc_output(decodeDesc, tok, crc);
        }
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
        {
            small_ldpc_dec_iter_output(decodeDesc, tok, iter);
        }
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output_multi(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem), mconfig);
    }
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::decode()
cuphyStatus_t reg_index_fp_desc_dyn_small::decode(ldpc::decoder&                     dec,
                                                  LDPC_output_t&                     tDst,
                                                  const_tensor_pair&                 tLLR,
                                                  const cuphy_optional<tensor_pair>& optSoftOutputs,
                                                  const cuphyLDPCDecodeConfigDesc_t& config,
                                                  cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_small::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    const int       THREADS_PER_CTA = MAX_THREADS_PER_CTA;
    const int       CW_PER_CTA = THREADS_PER_CTA / config.Z;
    //------------------------------------------------------------------
    dim3 grdDim((NUM_CW + (CW_PER_CTA - 1)) / CW_PER_CTA);
    dim3 blkDim(THREADS_PER_CTA);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, optSoftOutputs);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const uint32_t SHMEM_SIZE_PER_CW = round_up_to_next(shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                                              params.Z,             // lifting size
                                                                              sizeof(__half)),      // element size
                                                        static_cast<unsigned int>(sizeof(ldpc_traits<__half>::llr_sts_t)));
    const uint32_t SHMEM_SIZE = shmem_size_with_et_context(CW_PER_CTA * SHMEM_SIZE_PER_CW);

    //printf("grdDim = (%u), blkDim = (%u), CW_PER_CTA = %i, SHMEM_SIZE = %u, SHMEM_SIZE_PER_CW = %u, NUM_VAR_NODES = %i, func = %i, size = %u, test = %i\n",
    //       grdDim.x,
    //       blkDim.x,
    //       CW_PER_CTA,
    //       SHMEM_SIZE,
    //       SHMEM_SIZE_PER_CW,
    //       params.num_var_nodes,
    //       shmem_llr_buffer_size(params.num_var_nodes, params.Z, sizeof(__half)),
    //       static_cast<unsigned int>(sizeof(ldpc_traits<__half>::llr_sts_t)),
    //       ((68 * 2 + 15) / 16) * 16);

    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small, blkDim, SHMEM_SIZE);

                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    else if(llrType == CUPHY_R_32F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// reg_index_fp_desc_dyn_small::decode_tb()
cuphyStatus_t reg_index_fp_desc_dyn_small::decode_tb(ldpc::decoder&               dec,
                                                     const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                     cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_index_fp_desc_dyn_small::decode_tb()\n");
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    const int NUM_CW          = ldpc::decoder::get_total_num_codewords(decodeDesc);
    const int THREADS_PER_CTA = MAX_THREADS_PER_CTA;
    const int CW_PER_CTA      = THREADS_PER_CTA / decodeDesc.config.Z;
    //------------------------------------------------------------------
    dim3 grdDim((NUM_CW + (CW_PER_CTA - 1)) / CW_PER_CTA);
    dim3 blkDim(THREADS_PER_CTA);

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const int      NUM_VAR_NODES = decodeDesc.config.num_parity_nodes +
                                   ((1 == decodeDesc.config.BG) ? max_info_nodes<1>::value : max_info_nodes<2>::value);
    const uint32_t SHMEM_SIZE    = shmem_size_with_et_context(CW_PER_CTA * round_up_to_next(shmem_llr_buffer_size(NUM_VAR_NODES,       // num shared memory nodes
                                                                                       decodeDesc.config.Z, // lifting size
                                                                                       sizeof(__half)),     // element size
                                                                 static_cast<unsigned int>(sizeof(ldpc2::ldpc_traits<__half>::llr_sts_t))));

    //printf("grdDim = (%u), blkDim = (%u), CW_PER_CTA = %i, SHMEM_SIZE = %u, NUM_VAR_NODES = %i\n", grdDim.x, blkDim.x, CW_PER_CTA, SHMEM_SIZE, NUM_VAR_NODES);

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb, decodeDesc.config.flags,
                    grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    else if(decodeDesc.config.llr_type == CUPHY_R_32F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32, decodeDesc.config.flags, grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32, blkDim, SHMEM_SIZE);

                //------------------------------------------------------------------
                // Launch the kernel
                LDPC_LAUNCH_TB_KERNEL(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32, decodeDesc.config.flags, grdDim, blkDim, SHMEM_SIZE, strm, decodeDesc, *bgdesc);
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
// reg_index_fp_desc_dyn_small::get_workspace_size()
std::pair<bool, size_t> reg_index_fp_desc_dyn_small::get_workspace_size(const ldpc::decoder&               dec,
                                                                        const cuphyLDPCDecodeConfigDesc_t& config,
                                                                        int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::reg_index_fp_desc_dyn_small()
reg_index_fp_desc_dyn_small::reg_index_fp_desc_dyn_small(ldpc::decoder& desc)
{
    // Maximum shared memory configuration
    // Assume: THREADS_PER_CTA = 32
    //
    // (Changing THREADS_PER_CTA will change the analysis below. For
    // example, with Z = 12, we can do 2 codewords in 32 threads. But if
    // we do 96 threads, we can do 8 codewords.)
    //
    // We will launch THREADS_PER_CTA, and process as many codewords
    // in the main loop as we can.
    //
    // The amount of shared memory required is Z * (mb + Kb) * sizeof(T).
    // However, shared memory for each codeword needs to be aligned to
    // the STS size used by the LLR loader (currently uint4, or 8
    // elements). For the maximum case and 32 threads per warp, it
    // turns out that the amount of shared memory with padding is equal to
    // the amount of shared memory that would be required for Z = 32
    // without padding.
    // MAX_CODEWORDS_PER_CTA = (THREADS_PER_CTA / MIN_Z)  = (32 / 2) = 16
    // MAX_LLR_PER_CODEWORD: 68 * Z (BG1)
    //                       52 * Z (BG2)
    // MAX_SHMEM_PER_CTA = round_up(MAX_LLR_PER_CODEWORD, sizeof(sts_type)) * MAX_CODEWORDS_PER_CTA

    // BG1: 32 * sizeof(half) * 68 = 4352 bytes
    // BG2: 32 * sizeof(half) * 52 = 3328 bytes
    const uint32_t MAX_VAR_NODES_BG1 = ldpc2::max_variable_nodes<1>::value;
    const uint32_t MAX_VAR_NODES_BG2 = ldpc2::max_variable_nodes<2>::value;
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(shmem_size_with_et_context(shmem_llr_buffer_size(MAX_VAR_NODES_BG1, // num shared memory nodes
                                                                          32,                // lifting size
                                                                          sizeof(__half)))); // element size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(shmem_size_with_et_context(shmem_llr_buffer_size(MAX_VAR_NODES_BG2, // num shared memory nodes
                                                                          32,                // lifting size
                                                                          sizeof(__half)))); // element size
    const int MAX_BG1_SHMEM_SIZE_FP32 = static_cast<int>(shmem_size_with_et_context(shmem_llr_buffer_size(MAX_VAR_NODES_BG1,
                                                                               32,
                                                                               sizeof(float))));
    const int MAX_BG2_SHMEM_SIZE_FP32 = static_cast<int>(shmem_size_with_et_context(shmem_llr_buffer_size(MAX_VAR_NODES_BG2,
                                                                               32,
                                                                               sizeof(float))));

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 12> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small,         MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small,         MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_fp32,    MAX_BG1_SHMEM_SIZE_FP32),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_fp32,    MAX_BG2_SHMEM_SIZE_FP32),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb,      MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_no_accessories, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb,      MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_no_accessories, MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32, MAX_BG1_SHMEM_SIZE_FP32),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32, MAX_BG2_SHMEM_SIZE_FP32),
        func_attr_t((const void*)ldpc2_BG1_reg_index_fp_desc_dyn_small_tb_fp32_no_accessories, MAX_BG1_SHMEM_SIZE_FP32),
        func_attr_t((const void*)ldpc2_BG2_reg_index_fp_desc_dyn_small_tb_fp32_no_accessories, MAX_BG2_SHMEM_SIZE_FP32)
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_desc_dyn_small);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_desc_dyn_small);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_index_fp_desc_dyn_small_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_index_fp_desc_dyn_small_tb);
}

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_small::get_launch_config()
cuphyStatus_t reg_index_fp_desc_dyn_small::get_launch_config(const ldpc::decoder&           dec,
                                                             cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int NUM_CW           = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    const int THREADS_PER_CTA  = MAX_THREADS_PER_CTA;
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    const int CW_PER_CTA       = THREADS_PER_CTA / Z;

    const uint32_t SHMEM_SIZE  = shmem_size_with_et_context(CW_PER_CTA * round_up_to_next(shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                                                                                     Z,               // lifting size
                                                                                     sizeof(__half)), // element size
                                                               static_cast<unsigned int>(sizeof(ldpc2::ldpc_traits<__half>::llr_sts_t))));

    //------------------------------------------------------------------
    // Validate input arguments
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = THREADS_PER_CTA;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = div_round_up(NUM_CW, CW_PER_CTA);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = SHMEM_SIZE;
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    cudaError_t    e = (BG == 1) ?  LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG1_reg_index_fp_desc_dyn_small_tb, launchConfig.decode_desc.config.flags):
                                    LDPC_GET_TB_KERNEL_FUNCTION(deviceFunction, ldpc2_BG2_reg_index_fp_desc_dyn_small_tb, launchConfig.decode_desc.config.flags);
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
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc_small(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc_small(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
