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

// ALGO 52: Register-APP x2 min-sum decoder for BG1 p=44-46 on GB203.
//
// Extends ALGO 51 to handle parity counts that overflow GB203's 99 KB
// shmem limit. The last 1-3 parity columns (65, 66, 67) are degree-1
// extension nodes: their V2C = initial_LLR is constant across
// iterations. ALGO 52 stores these overflow columns' APP values in
// registers instead of shared memory, capping shmem at 65 variable
// nodes (22 info + 43 parity). All C2V data remains in registers
// (like ALGO 51), and x2 (2 CW/CTA) architecture is preserved.
//
// Coverage: BG1 p=4-46 (auto-selected for p=44-46 on GB203 where
// ALGO 51 can't fit APP in shmem). BG2 not needed
// (ALGO 51 already covers all BG2 cases).

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_index_fp_x2_allreg_regapp.hpp"
#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_crc_dispatch.cuh"

#define LDPC_DECODE_USE_TB_SCAN 1

using namespace ldpc2;

namespace
{
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    // Shmem holds at most 65 variable nodes (Kb=22 + 43 parity).
    // Column 65 at Z=384: 65*384*4 = 99,840 bytes APP. With tb_token
    // total ~99,844 < 101,376 (GB203 optin max).
    static constexpr int MAX_SHMEM_VAR_NODES_BG1 = 65;
    // Row 43 is the first row whose extension column (col 65) overflows.
    static constexpr int FIRST_OVERFLOW_ROW_BG1  = 43;
    // p=46 has 3 overflow columns: 65 (row 43), 66 (row 44), 67 (row 45).
    static constexpr int MAX_OVERFLOW_COLS       = 3;

    //------------------------------------------------------------------
    // All parity rows stored in registers (same as ALGO 51).
    template <int BG> struct max_num_parity;
    template <> struct max_num_parity<1> { static constexpr int value = 46; };

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
    // app_loader_skip_ext: loads ROW_DEGREE-1 APP values from shmem,
    // leaving the last element (extension column) untouched. The caller
    // (ldpc_schedule_regapp) pre-populates it from a register or shmem.
    template <typename T, int ROW_DEGREE> struct app_loader_skip_ext;

    template <int ROW_DEGREE>
    struct app_loader_skip_ext<__half2, ROW_DEGREE>
    {
        __device__
        static void load(word_t (&app)     [row_num_words<__half2, ROW_DEGREE>::value],
                         int    (&app_addr)[ROW_DEGREE],
                         int    smem_offset)
        {
            #pragma unroll
            for(int i = 0; i < ROW_DEGREE - 1; ++i)
            {
                app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
            }
            // app[ROW_DEGREE-1] left untouched — caller pre-populated it
        }
    };

    //------------------------------------------------------------------
    // ldpc_schedule_regapp: custom BG1 schedule that pre-populates the
    // extension column APP value before each row's C2V processing.
    // For rows 0-42: extension is in shmem, loaded manually.
    // For rows 43-45: extension is an overflow column, loaded from register.
    template <class TAPPLoc,
              class TC2VCache,
              class TKernelParams,
              class BGDesc,
              int   MIN_PARITY_ROWS,
              int   MAX_PARITY_ROWS>
    struct ldpc_schedule_regapp :
        ldpc_schedule_dynamic_desc_base<1,
                                        TAPPLoc,
                                        TC2VCache,
                                        TKernelParams,
                                        BGDesc,
                                        MIN_PARITY_ROWS,
                                        MAX_PARITY_ROWS>
    {
        typedef ldpc_schedule_dynamic_desc_base<1,
                                                TAPPLoc,
                                                TC2VCache,
                                                TKernelParams,
                                                BGDesc,
                                                MIN_PARITY_ROWS,
                                                MAX_PARITY_ROWS> inherited_t;
        typedef typename TC2VCache::app_t app_t;
        typedef BGDesc                    bg_desc_t;

        // Extension APP values for overflow columns, stored in registers.
        // regapp_[0] = col 65 (row 43 ext), regapp_[1] = col 66 (row 44 ext),
        // regapp_[2] = col 67 (row 45 ext). Unused slots zeroed.
        word_t regapp_[MAX_OVERFLOW_COLS];

        //--------------------------------------------------------------
        __device__
        ldpc_schedule_regapp(char*                smem,
                             const TKernelParams& params,
                             const bg_desc_t&     bg_desc,
                             int                  soffset,
                             unsigned int         t_idx,
                             const word_t         (&overflow_app)[MAX_OVERFLOW_COLS]) :
            inherited_t(smem, params, bg_desc, soffset, t_idx)
        {
            regapp_[0] = overflow_app[0];
            regapp_[1] = overflow_app[1];
            regapp_[2] = overflow_app[2];
        }

        //--------------------------------------------------------------
        // process_row: pre-populate extension APP, then delegate to C2V
        // chain. app_loader_skip_ext fills app[0..d-2] from shmem;
        // we set app[d-1] here.
        template <int CHECK_IDX>
        __device__
        void process_row()
        {
            constexpr int ROW_DEG   = row_degree<1, CHECK_IDX>::value;
            constexpr int NUM_APP_W = app_num_words<app_t, 1, CHECK_IDX>::value;
            int    app_addr[ROW_DEG];
            word_t app[NUM_APP_W];

            this->app_addr_gen.template generate<CHECK_IDX>(app_addr);

            // Pre-populate the extension column's APP value.
            // Overflow rows: extension column is NOT in shmem — use register.
            // Normal rows: extension column IS in shmem — load it.
            if constexpr (CHECK_IDX >= FIRST_OVERFLOW_ROW_BG1)
            {
                app[ROW_DEG - 1] = regapp_[CHECK_IDX - FIRST_OVERFLOW_ROW_BG1];
            }
            else
            {
                app[ROW_DEG - 1] = smem_address_as<word_t>(this->smem_offset +
                                                            app_addr[ROW_DEG - 1]);
            }

            this->c2v_cache.template process_row<CHECK_IDX>(this->params,
                                                             app,
                                                             app_addr,
                                                             this->smem_offset);
        }

        //--------------------------------------------------------------
        // do_iteration: BG1 46-row unrolled schedule (same structure as
        // ldpc_schedule_dynamic_desc<1,...>).
        __device__
        void do_iteration()
        {
            (*this).template process_row<0> (); __syncthreads();
            (*this).template process_row<1> (); __syncthreads();
            (*this).template process_row<2> (); __syncthreads();
            (*this).template process_row<3> (); if((*this).template iter_sync_check_done< 3>()) return;
            (*this).template process_row<4> (); if((*this).template iter_sync_check_done< 4>()) return;
            (*this).template process_row<5> (); if((*this).template iter_sync_check_done< 5>()) return;
            (*this).template process_row<6> (); if((*this).template iter_sync_check_done< 6>()) return;
            (*this).template process_row<7> (); if((*this).template iter_sync_check_done< 7>()) return;
            (*this).template process_row<8> (); if((*this).template iter_sync_check_done< 8>()) return;
            (*this).template process_row<9> (); if((*this).template iter_sync_check_done< 9>()) return;
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
            (*this).template process_row<24>(); if((*this).template iter_sync_check_done<24>()) return;
            (*this).template process_row<25>(); if((*this).template iter_sync_check_done<25>()) return;
            (*this).template process_row<26>(); if((*this).template iter_sync_check_done<26>()) return;
            (*this).template process_row<27>(); if((*this).template iter_sync_check_done<27>()) return;
            (*this).template process_row<28>(); if((*this).template iter_sync_check_done<28>()) return;
            (*this).template process_row<29>(); if((*this).template iter_sync_check_done<29>()) return;
            (*this).template process_row<30>(); if((*this).template iter_sync_check_done<30>()) return;
            (*this).template process_row<31>(); if((*this).template iter_sync_check_done<31>()) return;
            (*this).template process_row<32>(); if((*this).template iter_sync_check_done<32>()) return;
            (*this).template process_row<33>(); if((*this).template iter_sync_check_done<33>()) return;
            (*this).template process_row<34>(); if((*this).template iter_sync_check_done<34>()) return;
            (*this).template process_row<35>(); if((*this).template iter_sync_check_done<35>()) return;
            (*this).template process_row<36>(); if((*this).template iter_sync_check_done<36>()) return;
            (*this).template process_row<37>(); if((*this).template iter_sync_check_done<37>()) return;
            (*this).template process_row<38>(); if((*this).template iter_sync_check_done<38>()) return;
            (*this).template process_row<39>(); if((*this).template iter_sync_check_done<39>()) return;
            (*this).template process_row<40>(); if((*this).template iter_sync_check_done<40>()) return;
            (*this).template process_row<41>(); if((*this).template iter_sync_check_done<41>()) return;
            (*this).template process_row<42>(); if((*this).template iter_sync_check_done<42>()) return;
            (*this).template process_row<43>(); if((*this).template iter_sync_check_done<43>()) return;
            (*this).template process_row<44>(); if((*this).template iter_sync_check_done<44>()) return;
            (*this).template process_row<45>(); __syncthreads();
        }
    };

    //------------------------------------------------------------------
    // Kernel configuration for ALGO 52 (BG1 only).
    // Uses app_loader_skip_ext and ldpc_schedule_regapp.
    // MIN_P_ raises MIN_PARITY_ROWS so the schedule's IS_LAST_ROW
    // runtime check can be eliminated at compile time for the early
    // rows when a high-p variant is dispatched.
    template <class TKernelParams, int MIN_P_ = 4>
    struct kernel_config
    {
        static constexpr int MIN_PARITY_ROWS     = MIN_P_;
        static constexpr int NUM_REG_PARITY_ROWS = max_num_parity<1>::value;
        static constexpr int MAX_PARITY_ROWS     = max_num_parity<1>::value;

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
                             1,
                             cC2V_row_map_t,
                             app_loader_skip_ext,
                             app_writer> C2V_t;

        typedef ldpc2::c2v_cache_split<1,
                                       NUM_REG_PARITY_ROWS,
                                       C2V_t,
                                       typename core_storage_x2<1>::type,
                                       cC2V_storage_x2_low_degree,
                                       kernel_params_t> c2v_cache_t;

        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        typedef llr_loader_t::app_buf_t                                    app_buf_t;

        typedef ldpc_schedule_regapp<app_loc_t<1>,
                                     c2v_cache_t,
                                     kernel_params_t,
                                     typename app_loc_t<1>::bg_desc_t,
                                     MIN_PARITY_ROWS,
                                     MAX_PARITY_ROWS> sched_t;
    };

    //------------------------------------------------------------------
    // Shmem sizing: capped at MAX_SHMEM_VAR_NODES_BG1 (65) columns.
    CUDA_BOTH
    int get_app_shmem_regapp(int Z)
    {
        return static_cast<int32_t>(shmem_llr_buffer_size(MAX_SHMEM_VAR_NODES_BG1,
                                                          Z,
                                                          sizeof(__half2)));
    }
    //------------------------------------------------------------------
    CUDA_BOTH
    int get_shmem_required_regapp(int Z)
    {
        int shmem_size = get_app_shmem_regapp(Z);
#if LDPC_DECODE_USE_TB_SCAN
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }
    //------------------------------------------------------------------
    CUDA_BOTH
    uint32_t get_tb_shmem_required_regapp(int Z)
    {
        return shmem_size_with_et_context<ldpc_et_context_x2_t>(
            static_cast<uint32_t>(get_shmem_required_regapp(Z)));
    }
#if LDPC_DECODE_USE_TB_SCAN
    __device__
    tb_token* get_token_addr_regapp(int Z, char* smem)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_shmem_regapp(Z));
    }
#endif

    //------------------------------------------------------------------
    // Load overflow extension columns from global memory into registers.
    // Overflow columns (65, 66, 67 for BG1) are degree-1 extension
    // nodes whose APP = initial_LLR is constant across iterations.
    // This function reads their initial LLR values from both codewords,
    // interleaves (CW0 in .x, CW1 in .y), and applies clamping.
    __device__
    void load_overflow_columns(word_t       (&regapp)[MAX_OVERFLOW_COLS],
                               const void*  src_gmem,
                               int          src_stride_elements,
                               int          max_cta_cw_index,
                               int          Z,
                               float        clamp_value,
                               int          num_overflow)
    {
        const __half* base = reinterpret_cast<const __half*>(src_gmem);
        const __half  clamp_pos = __float2half(clamp_value);
        const __half  clamp_neg = __hneg(clamp_pos);

        #pragma unroll
        for(int i = 0; i < MAX_OVERFLOW_COLS; ++i)
        {
            word_t w;
            if(i < num_overflow)
            {
                const int col      = MAX_SHMEM_VAR_NODES_BG1 + i;
                const int elem_idx = col * Z + threadIdx.x;

                // CW0
                __half h0 = base[elem_idx];
                h0 = __hmax(h0, clamp_neg);
                h0 = __hmin(h0, clamp_pos);

                // CW1
                __half h1;
                if(max_cta_cw_index > 0)
                {
                    h1 = base[src_stride_elements + elem_idx];
                    h1 = __hmax(h1, clamp_neg);
                    h1 = __hmin(h1, clamp_pos);
                }
                else
                {
                    h1 = __float2half(0.0f);
                }

                w.f16x2.x = __half_as_ushort(h0);
                w.f16x2.y = __half_as_ushort(h1);
            }
            else
            {
                w.u32 = 0;
            }
            regapp[i] = w;
        }
    }

} // namespace

////////////////////////////////////////////////////////////////////////
// HIGH_P_THRESHOLD: parity node count at/above which the high-p kernel
// variants are dispatched. Setting MIN_PARITY_ROWS = HIGH_P_THRESHOLD
// makes the schedule's IS_LAST_ROW runtime check vanish at compile time
// for rows below the threshold.
namespace { const int HIGH_P_THRESHOLD = 22; }

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_index_fp_x2_allreg_regapp()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_regapp(LDPC_kernel_params params,
                                                app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<ldpc2::LDPC_kernel_params> kernel_config_t;

    // Construct LLR loader params and cap at 65 shmem columns.
    ldpc_dec_loader_params<__half2> lp(smem, params, blockIdx.x);
    const void* saved_src       = lp.src_gmem;
    int         saved_stride    = lp.src_stride_elements;
    int         saved_max_cw    = lp.max_cta_cw_index;
    lp.num_cw_elements          = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * params.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);

    // Load overflow extension columns from global memory into registers.
    const int num_overflow = params.num_var_nodes - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          params.Z,
                          params.clamp_value,
                          num_overflow);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
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
// ldpc2_BG1_index_fp_x2_allreg_regapp_tb()
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_regapp_tb(cuphyLDPCDecodeDesc_t decodeDesc,
                                                   app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Token-based LLR loading: find token, construct loader params with
    // capped num_cw_elements, load 65 columns into shmem.
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, blockIdx.x);
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#else
    // Find the TB token (same as load_sync_token, but split so we can
    // cap num_cw_elements before the LLR load).
    tb_token* pToken = get_token_addr_regapp(decodeDesc.config.Z, smem);
    if(0 == (threadIdx.x / 32))
    {
        warp_find_block_tb_token<2>(decodeDesc, blockIdx.x, pToken);
    }
    __syncthreads();
    tb_token tok = *pToken;
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, loader_token(tok));
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#endif

    //------------------------------------------------------------------
    // Load overflow extension columns from global memory into registers.
    const int Kb = 22;
    const int num_overflow = (Kb + decodeDesc.config.num_parity_nodes) - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          decodeDesc.config.Z,
                          decodeDesc.config.clamp_value,
                          num_overflow);


    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(
        smem, static_cast<uint32_t>(get_shmem_required_regapp(decodeDesc.config.Z)));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<1>(
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
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#else
    ldpc_dec_output_variable_loop(decodeDesc,
                                  tok,
                                  reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, tok, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#endif
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }

}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_regapp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc,
                                                   app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Token-based LLR loading: find token, construct loader params with
    // capped num_cw_elements, load 65 columns into shmem.
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, blockIdx.x);
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#else
    // Find the TB token (same as load_sync_token, but split so we can
    // cap num_cw_elements before the LLR load).
    tb_token* pToken = get_token_addr_regapp(decodeDesc.config.Z, smem);
    if(0 == (threadIdx.x / 32))
    {
        warp_find_block_tb_token<2>(decodeDesc, blockIdx.x, pToken);
    }
    __syncthreads();
    tb_token tok = *pToken;
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, loader_token(tok));
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#endif

    //------------------------------------------------------------------
    // Load overflow extension columns from global memory into registers.
    const int Kb = 22;
    const int num_overflow = (Kb + decodeDesc.config.num_parity_nodes) - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          decodeDesc.config.Z,
                          decodeDesc.config.clamp_value,
                          num_overflow);


    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(
        smem, static_cast<uint32_t>(get_shmem_required_regapp(decodeDesc.config.Z)));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<1>(
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
#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#else
    ldpc_dec_output_variable_loop(decodeDesc,
                                  tok,
                                  reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, tok, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#endif
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }

}

////////////////////////////////////////////////////////////////////////
// High-p variants. MIN_PARITY_ROWS = HIGH_P_THRESHOLD lets the schedule's
// per-row IS_LAST_ROW runtime check evaporate at compile time for rows
// 0..HIGH_P_THRESHOLD-2. Only safe to launch when the runtime
// num_parity_nodes >= HIGH_P_THRESHOLD; the dispatch below enforces this.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_regapp_highp(LDPC_kernel_params params,
                                                      app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef kernel_config<ldpc2::LDPC_kernel_params, HIGH_P_THRESHOLD> kernel_config_t;

    ldpc_dec_loader_params<__half2> lp(smem, params, blockIdx.x);
    const void* saved_src       = lp.src_gmem;
    int         saved_stride    = lp.src_stride_elements;
    int         saved_max_cw    = lp.max_cta_cw_index;
    lp.num_cw_elements          = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * params.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);

    const int num_overflow = params.num_var_nodes - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          params.Z,
                          params.clamp_value,
                          num_overflow);

    kernel_config_t::sched_t sched(smem,
                                   params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
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
void ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb(cuphyLDPCDecodeDesc_t decodeDesc,
                                                         app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef kernel_config<cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, blockIdx.x);
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#else
    tb_token* pToken = get_token_addr_regapp(decodeDesc.config.Z, smem);
    if(0 == (threadIdx.x / 32))
    {
        warp_find_block_tb_token<2>(decodeDesc, blockIdx.x, pToken);
    }
    __syncthreads();
    tb_token tok = *pToken;
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, loader_token(tok));
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#endif

    const int Kb = 22;
    const int num_overflow = (Kb + decodeDesc.config.num_parity_nodes) - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          decodeDesc.config.Z,
                          decodeDesc.config.clamp_value,
                          num_overflow);

    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(
        smem, static_cast<uint32_t>(get_shmem_required_regapp(decodeDesc.config.Z)));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<1>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#else
    ldpc_dec_output_variable_loop(decodeDesc,
                                  tok,
                                  reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, tok, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#endif
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }

}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc,
                                                         app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef kernel_config<cuphyLDPCDecodeConfigDesc_t, HIGH_P_THRESHOLD> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, blockIdx.x);
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#else
    tb_token* pToken = get_token_addr_regapp(decodeDesc.config.Z, smem);
    if(0 == (threadIdx.x / 32))
    {
        warp_find_block_tb_token<2>(decodeDesc, blockIdx.x, pToken);
    }
    __syncthreads();
    tb_token tok = *pToken;
    ldpc_dec_loader_params<__half2> lp(smem, decodeDesc, loader_token(tok));
    const void* saved_src    = lp.src_gmem;
    int         saved_stride = lp.src_stride_elements;
    int         saved_max_cw = lp.max_cta_cw_index;
    lp.num_cw_elements       = min(lp.num_cw_elements, MAX_SHMEM_VAR_NODES_BG1 * decodeDesc.config.Z);
    kernel_config_t::llr_loader_t::load_sync(lp);
#endif

    const int Kb = 22;
    const int num_overflow = (Kb + decodeDesc.config.num_parity_nodes) - MAX_SHMEM_VAR_NODES_BG1;
    word_t overflow_app[MAX_OVERFLOW_COLS];
    load_overflow_columns(overflow_app,
                          saved_src,
                          saved_stride,
                          saved_max_cw,
                          decodeDesc.config.Z,
                          decodeDesc.config.clamp_value,
                          num_overflow);

    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(
        smem, static_cast<uint32_t>(get_shmem_required_regapp(decodeDesc.config.Z)));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    kernel_config_t::sched_t sched(smem,
                                   decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x,
                                   overflow_app);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    auto prev_packed_word = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::prev_init;
    while(iter < decodeDesc.config.max_iterations)
    {
        sched.do_iteration();
        ++iter;
        if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
        {
            crc = ldpc2::should_terminate_early_crc<1>(
                decodeDesc,
                blockIdx.x,
                reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                et_ctx,
                iter - 1,
                prev_packed_word);
            if(crc == 0) break;
        }
    }

#if !LDPC_DECODE_USE_TB_SCAN
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#else
    ldpc_dec_output_variable_loop(decodeDesc,
                                  tok,
                                  reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, tok, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
#endif
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }

}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg_regapp::decode()
cuphyStatus_t index_fp_x2_allreg_regapp::decode(ldpc::decoder&                     dec,
                                                       LDPC_output_t&                     tDst,
                                                       const_tensor_pair&                 tLLR,
                                                       const cuphy_optional<tensor_pair>& optSoftOutputs,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::index_fp_x2_allreg_regapp::decode()\n");
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
    const uint32_t SHMEM_SIZE = get_shmem_required_regapp(config.Z);

    if(llrType == CUPHY_R_16F && config.BG == 1)
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
        if(bgdesc)
        {
            if(config.num_parity_nodes >= HIGH_P_THRESHOLD)
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_regapp_highp, blkDim, SHMEM_SIZE);
                ldpc2_BG1_index_fp_x2_allreg_regapp_highp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
            }
            else
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_regapp, blkDim, SHMEM_SIZE);
                ldpc2_BG1_index_fp_x2_allreg_regapp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
            }
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
// index_fp_x2_allreg_regapp::decode_tb()
cuphyStatus_t index_fp_x2_allreg_regapp::decode_tb(ldpc::decoder&               dec,
                                                          const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                          cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::index_fp_x2_allreg_regapp::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    if(decodeDesc.config.llr_type == CUPHY_R_16F && decodeDesc.config.BG == 1)
    {
        dim3 blkDim(decodeDesc.config.Z);
        dim3 grdDim(ldpc::decoder::get_total_num_codeword_pairs(decodeDesc));

        const uint32_t TB_SHMEM_SIZE = get_tb_shmem_required_regapp(decodeDesc.config.Z);
        const uint32_t SHMEM_SIZE = shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            TB_SHMEM_SIZE,
            decodeDesc.config.flags);

        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
        if(bgdesc)
        {
            if(decodeDesc.config.num_parity_nodes >= HIGH_P_THRESHOLD)
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb, blkDim, SHMEM_SIZE);
                LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb, decodeDesc.config.flags,
                    grdDim, blkDim, TB_SHMEM_SIZE, strm, decodeDesc, *bgdesc);
            }
            else
            {
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_index_fp_x2_allreg_regapp_tb, blkDim, SHMEM_SIZE);
                LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_index_fp_x2_allreg_regapp_tb, decodeDesc.config.flags,
                    grdDim, blkDim, TB_SHMEM_SIZE, strm, decodeDesc, *bgdesc);
            }
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
// index_fp_x2_allreg_regapp::get_workspace_size()
std::pair<bool, size_t> index_fp_x2_allreg_regapp::get_workspace_size(const ldpc::decoder&               dec,
                                                                             const cuphyLDPCDecodeConfigDesc_t& config,
                                                                             int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg_regapp::index_fp_x2_allreg_regapp()
index_fp_x2_allreg_regapp::index_fp_x2_allreg_regapp(ldpc::decoder& dec)
{
    const int MAX_SHMEM_SIZE = static_cast<int>(get_shmem_required_regapp(CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_SHMEM      = dec.max_shmem_per_block_optin();
    const int MAX_TB_SHMEM_SIZE = static_cast<int>(
        get_tb_shmem_required_regapp(CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_LEAN_SHMEM_SIZE = static_cast<int>(
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            static_cast<uint32_t>(MAX_TB_SHMEM_SIZE), 0));

    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 6> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp,                          std::min(MAX_SHMEM_SIZE,      MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp_tb,                       std::min(MAX_TB_SHMEM_SIZE,   MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp_tb_no_accessories,        std::min(MAX_LEAN_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp_highp,                    std::min(MAX_SHMEM_SIZE,      MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb,                 std::min(MAX_TB_SHMEM_SIZE,   MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb_no_accessories,  std::min(MAX_LEAN_SHMEM_SIZE, MAX_SHMEM))
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_fp_x2_allreg_regapp);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_index_fp_x2_allreg_regapp_tb);
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg_regapp::can_decode_config()
bool index_fp_x2_allreg_regapp::can_decode_config(const ldpc::decoder&               dec,
                                                         const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // BG1 only, p=4-46.
    if(cfg.BG != 1)
        return false;
    if(cfg.num_parity_nodes < 4) // MIN_PARITY_ROWS
        return false;
    if(cfg.num_parity_nodes > static_cast<uint32_t>(max_num_parity<1>::value)) // p > 46
        return false;

    const uint32_t SHMEM_BYTES = shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
        get_tb_shmem_required_regapp(cfg.Z),
        cfg.flags);
    return SHMEM_BYTES <= dec.max_shmem_per_block_optin();
}

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg_regapp::get_launch_config()
cuphyStatus_t index_fp_x2_allreg_regapp::get_launch_config(const ldpc::decoder&           dec,
                                                                   cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    //------------------------------------------------------------------
    if((BG != 1)                                ||
       (Z < 2)                                  ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)        ||
       (NUM_PARITY_NODES < 4)                   ||
       (NUM_PARITY_NODES > max_parity_nodes<1>::value))
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

    const uint32_t SHMEM_SIZE = get_tb_shmem_required_regapp(Z);
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            SHMEM_SIZE,
            launchConfig.decode_desc.config.flags);
    if(launchConfig.kernel_node_params_driver.sharedMemBytes >
       static_cast<uint32_t>(dec.max_shmem_per_block_optin()))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    const void* kernel_func = (NUM_PARITY_NODES >= HIGH_P_THRESHOLD) ?
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_index_fp_x2_allreg_regapp_highp_tb, launchConfig.decode_desc.config.flags) :
        LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_index_fp_x2_allreg_regapp_tb, launchConfig.decode_desc.config.flags);
    cudaError_t    e = cudaGetFuncBySymbol(&deviceFunction, kernel_func);
    if (e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif
    //------------------------------------------------------------------
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
