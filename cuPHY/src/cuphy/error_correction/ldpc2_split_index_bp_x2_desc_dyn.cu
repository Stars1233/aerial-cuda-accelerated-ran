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
//#define LDPC_PRINT_PARAMS 1

#include <assert.h>
#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_box_plus_x2.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_app_address_dp_desc.cuh"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_desc.cuh"
#include "ldpc2_split_index_bp_x2_desc_dyn.hpp"
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
    // Storing compressed check to variable (cC2V) data in registers may
    // not be possible for all code rates. Furthermore, squeezing a
    // larger number of parity node data into registers may actually
    // decrease performance at high code rates.
    template <int BG> struct num_reg_parity;
    template <> struct num_reg_parity<1> { static constexpr int value = 20; };
    template <> struct num_reg_parity<2> { static constexpr int value = 42; };

    //------------------------------------------------------------------
    // Non-core C2V storage type selection:
    // BG1: box_plus storage (forward-backward, eliminates sign tracking)
    // BG2: box_plus storage (max update_row_degree = 5 for non-core rows)
    static constexpr int BG1_MAX_BOX_PLUS_WORDS = 9; // max UPDATE_ROW_DEGREE for BG1 non-core rows
    static constexpr int BG2_MAX_BOX_PLUS_WORDS = 5; // max UPDATE_ROW_DEGREE for BG2 non-core rows
    template <int BG> struct noncore_storage_x2;
    template <> struct noncore_storage_x2<1> { typedef c2v_storage_x2_box_plus<BG1_MAX_BOX_PLUS_WORDS> type; };
    template <> struct noncore_storage_x2<2> { typedef c2v_storage_x2_box_plus<BG2_MAX_BOX_PLUS_WORDS> type; };

    //------------------------------------------------------------------
    // Core C2V storage type selection:
    // BG1: box_plus storage with degree 19 (stored in shmem, not registers)
    // BG2: compressed min-sum, low-degree (unchanged, stays in registers)
    static constexpr int BG1_CORE_BOX_PLUS_WORDS = 19; // UPDATE_ROW_DEGREE for BG1 core rows
    template <int BG> struct core_storage_x2_local;
    template <> struct core_storage_x2_local<1> { typedef c2v_storage_x2_box_plus<BG1_CORE_BOX_PLUS_WORDS> type; };
    template <> struct core_storage_x2_local<2> { typedef typename core_storage_x2<2>::type type; };
    template <int BG> struct max_num_parity;
    template <> struct max_num_parity<1> { static constexpr int value = 46; };
    template <> struct max_num_parity<2> { static constexpr int value = 42; };

    //------------------------------------------------------------------
    // Sign manager for compressed C2V row processor
    typedef sign_mgr_pair_src<false> sign_mgr_t;

    //------------------------------------------------------------------
    // APP address calculation
    // Using floating point with dot product instruction sequence for
    // this decoder algorithm. Note that the base graph descriptor
    // argument to the kernel needs to be the "adjusted" descriptor
    // structure.
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half2, BG>;
    // slightly slower on sm86
    //template <int BG> using app_loc_t = app_loc_address_dp_desc<__half2, BG>;

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
    // Kernel configuration structure, with typedefs for kernel execution
    //
    // Better perf at very high code rates when the MAX_PARITY_NODES
    // is smaller, but for now we'll prefer to get 2X codewords for
    // as many parity nodes as possible. (Try 32 vs. 28 to see the perf
    // difference.)
    // TODO: small, med, large parity count kernels?
    //
    // BOX_PLUS_CORE: When true, BG1 core rows (0-3) use box_plus with
    // shared memory C2V storage (~181 KB shmem). When false, core rows
    // fall back to compressed min-sum with register C2V storage (~65 KB shmem).
    // Non-core rows always use box_plus (register-based, no shmem impact).
    // (Inert for BG2: its core is always compressed min-sum -- see
    // core_storage_x2_local<2> below.)

    template <bool BOX_PLUS_CORE, int NUM_REG_NODES_BG1>
    CUDA_BOTH
    int get_shmem_required(int BG, int num_parity_nodes, int Z);

    template <int   BG_,                  // base graph (1 or 2)
              class TKernelParams,        // struct with kernel params
              bool  BOX_PLUS_CORE_ = true, // use box_plus for core rows?
              int   MIN_P_        = 4,    // raise to eliminate IS_LAST_ROW
                                          // for low-index rows when a
                                          // high-p variant is dispatched
              int   NUM_REG_PARITY_ = num_reg_parity<BG_>::value>
                                          // # non-core parity rows held in
                                          // registers; raise (the "bigreg"
                                          // variant) to free shmem and extend
                                          // the box-plus range past p=30
    struct ldpc2_split_index_bp_x2_desc_dyn_kernel_config
    {
        static constexpr int BG                  = BG_;
        static constexpr int MIN_PARITY_ROWS     = MIN_P_;
        static constexpr int NUM_REG_PARITY_ROWS = NUM_REG_PARITY_;
        static constexpr int MAX_PARITY_ROWS     = max_num_parity<BG>::value;
        CUDA_BOTH
        static int shmem_required(int num_parity_nodes, int Z)
        {
            return get_shmem_required<BOX_PLUS_CORE_, NUM_REG_PARITY_>(BG_, num_parity_nodes, Z);
        }

        typedef TKernelParams                           kernel_params_t;

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // cC2V_row_map_t
        // Hybrid row map: dispatches to box_plus for non-core rows
        // (which use c2v_storage_x2_box_plus), and compressed min-sum for core
        // rows (which use cC2V_storage_x2_high_degree_split).
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
        // Non-core storage: box_plus for both BG1 and BG2 (BG1 up to 9
        // words/row, BG2 up to 5; see noncore_storage_x2 above).
        typedef typename noncore_storage_x2<BG>::type noncore_storage_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (split between register and shared memory
        // here). Two storage types are provided: one for the "core"
        // parity rows, and one for the "non-core" rows.
        typedef ldpc2::c2v_cache_split<BG,
                                       NUM_REG_PARITY_ROWS,
                                       C2V_t,
                                       typename std::conditional_t<BOX_PLUS_CORE_,
                                                                  core_storage_x2_local<BG>,
                                                                  core_storage_x2<BG>>::type, // core cC2V storage
                                       noncore_storage_t,                   // non-core cC2V storage
                                       kernel_params_t> c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef ldpc2::llr_loader_variable_batch<__half2, 4, llr_op_clamp> llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef llr_loader_t::app_buf_t                                    app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
    //------------------------------------------------------------------
    // get_app_c2v_shmem()
    // Returns the number of bytes required for APP and C2V memory for
    // this kernel.
    template <int BG, bool BOX_PLUS_CORE = true, int NUM_REG_NODES_ = num_reg_parity<BG>::value>
    CUDA_BOTH
    int get_app_c2v_shmem(int num_parity_nodes, int Z)
    {
        typedef typename noncore_storage_x2<BG>::type noncore_storage_t;
        typedef typename std::conditional_t<BOX_PLUS_CORE,
                                            core_storage_x2_local<BG>,
                                            core_storage_x2<BG>>::type core_storage_t;

        const     int32_t NUM_VAR_NODES = ldpc2::max_info_nodes<BG>::value + num_parity_nodes;
        constexpr int32_t NUM_REG_NODES = NUM_REG_NODES_;
        int32_t offset        = static_cast<int32_t>(shmem_llr_buffer_size(NUM_VAR_NODES,     // num shared memory nodes
                                                                           Z,                 // lifting size
                                                                           sizeof(__half2))); // element size

        // Core C2V in shmem when core storage is large (box_plus)
        constexpr bool CORE_IN_SHMEM = (sizeof(core_storage_t) > 4 * sizeof(ldpc2::word_t));
        if constexpr (CORE_IN_SHMEM)
        {
            offset = round_up_to_next(offset, static_cast<int>(alignof(core_storage_t)));
            offset += 4 * Z * static_cast<int>(sizeof(core_storage_t));
        }

        // The first 'NUM_REG_NODES' of non-core C2V data will reside in registers.
        // The remainder will be in shared memory.
        const int32_t C2V_SIZE = (num_parity_nodes > NUM_REG_NODES)                                ?
                                 (num_parity_nodes - NUM_REG_NODES) * Z * sizeof(noncore_storage_t) :
                                 0;
        // Pad for non-core C2V alignment
        int shmem_size = round_up_to_next(offset, static_cast<int>(alignof(noncore_storage_t))) +
                                          C2V_SIZE;
        return shmem_size;
    }
    //------------------------------------------------------------------
    // get_shmem_required()
    // Calculates the sum of the APP and C2V data storage.
    template <bool BOX_PLUS_CORE = true, int NUM_REG_NODES_BG1 = num_reg_parity<1>::value>
    CUDA_BOTH
    int get_shmem_required(int BG,
                           int num_parity_nodes,
                           int Z)
    {
        int shmem_size = (1 == BG) ? get_app_c2v_shmem<1, BOX_PLUS_CORE, NUM_REG_NODES_BG1>(num_parity_nodes, Z)
                                   : get_app_c2v_shmem<2, BOX_PLUS_CORE>(num_parity_nodes, Z);
#if LDPC_DECODE_USE_TB_SCAN
        // When using a scan algorithm to determine the codeword for a CTA,
        // extra shared memory for the token is required.
        shmem_size = round_up_to_next(shmem_size, static_cast<int>(alignof(tb_token))) +
                     sizeof(tb_token);
#endif
        return shmem_size;
    }
    //------------------------------------------------------------------
    uint32_t get_selected_tb_shmem_required(uint32_t base_size, uint32_t flags)
    {
        return shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            shmem_size_with_et_context<ldpc_et_context_x2_t>(base_size),
            flags);
    }
#if LDPC_DECODE_USE_TB_SCAN
    //------------------------------------------------------------------
    // get_token_addr()
    // Returns the address of the tb_token value used to store information
    // about the specific codeword being processed by a CTA when the
    // transport block interface is used. The token is assumed to reside
    // immediately after the APP and C2V memory.
    template <int BG, bool BOX_PLUS_CORE = true, int NUM_REG_NODES_ = num_reg_parity<BG>::value>
    __device__
    tb_token* get_token_addr(int num_parity_nodes, int Z, char* smem)
    {
        return reinterpret_cast<tb_token*>(smem + get_app_c2v_shmem<BG, BOX_PLUS_CORE, NUM_REG_NODES_>(num_parity_nodes, Z));
    }
    template <int BG, bool BOX_PLUS_CORE = true, int NUM_REG_NODES_ = num_reg_parity<BG>::value>
    __device__
    tb_token* get_token_addr(const cuphyLDPCDecodeDesc_t& decodeDesc,
                             char* smem)
    {
        return get_token_addr<BG, BOX_PLUS_CORE, NUM_REG_NODES_>(decodeDesc.config.num_parity_nodes,
                                                                   decodeDesc.config.Z,
                                                                   smem);
    }
#endif // if LDPC_DECODE_USE_TB_SCAN
#ifdef LDPC_PRINT_PARAMS
    __device__ void print_ldpc_params_legacy(const ldpc2::LDPC_kernel_params& p, int BG)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            int Kb = p.Kb;
            int Z  = p.Z;
            int mb = p.num_parity_nodes;
            int B  = Kb * Z;
            float norm_lo = __half2float(*reinterpret_cast<const __half*>(&p.norm.f16x2.x));
            printf("=== LDPC ALGO 35 PARAMS (legacy interface) ===\n");
            printf("BG (base graph)       = %d\n", BG);
            printf("Kb (info nodes)       = %d\n", Kb);
            printf("Z  (lifting size)     = %d\n", Z);
            printf("mb (parity nodes)     = %d\n", mb);
            printf("num_var_nodes         = %d\n", p.num_var_nodes);
            printf("B  (Kb * Z)           = %d\n", B);
            printf("K  (info bits)        = %d\n", p.K);
            printf("KbZ                   = %d\n", p.KbZ);
            printf("Z_var                 = %d\n", p.Z_var);
            printf("max_iterations        = %d\n", p.max_iterations);
            printf("num_codewords         = %d\n", p.num_codewords);
            printf("Normalization         = %f\n", norm_lo);
            printf("clamp_value           = %f\n", p.clamp_value);
            printf("outputs_per_codeword  = %d\n", p.outputs_per_codeword);
            printf("input_llr_stride_elem = %d\n", p.input_llr_stride_elements);
            printf("output_stride_words   = %d\n", p.output_stride_words);
            printf("gridDim.x             = %d\n", gridDim.x);
            printf("blockDim.x            = %d\n", blockDim.x);
            printf("=== END LDPC PARAMS ===\n");
        }
    }

    __device__ void print_ldpc_params_tb(const cuphyLDPCDecodeDesc_t& d)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            const cuphyLDPCDecodeConfigDesc_t& c = d.config;
            int Kb = c.Kb;
            int Z  = c.Z;
            int mb = c.num_parity_nodes;
            int BG = c.BG;
            int B  = Kb * Z;
            int num_var = mb + ((BG == 1) ? 22 : 10);
            float norm_lo = __half2float(*reinterpret_cast<const __half*>(&c.norm.f16x2.x));

            int total_cw = 0;
            for(int i = 0; i < d.num_tbs; ++i)
                total_cw += d.llr_input[i].num_codewords;

            printf("=== LDPC ALGO 35 PARAMS (TB interface) ===\n");
            printf("BG (base graph)       = %d\n", BG);
            printf("Kb (info nodes)       = %d\n", Kb);
            printf("Z  (lifting size)     = %d\n", Z);
            printf("mb (parity nodes)     = %d\n", mb);
            printf("num_var_nodes         = %d\n", num_var);
            printf("B  (Kb * Z)           = %d\n", B);
            printf("max_iterations        = %d\n", c.max_iterations);
            printf("Normalization         = %f\n", norm_lo);
            printf("clamp_value           = %f\n", c.clamp_value);
            printf("algo                  = %d\n", c.algo);
            printf("llr_type              = %d\n", (int)c.llr_type);
            printf("flags                 = 0x%08X\n", c.flags);
            printf("num_tbs               = %d\n", d.num_tbs);
            printf("total_codewords       = %d\n", total_cw);
            for(int i = 0; i < d.num_tbs && i < 4; ++i)
            {
                printf("  TB[%d]: num_cw=%d, llr_stride=%d\n",
                       i, d.llr_input[i].num_codewords,
                       d.llr_input[i].stride_elements);
            }
            printf("gridDim.x             = %d\n", gridDim.x);
            printf("blockDim.x            = %d\n", blockDim.x);
            // Derived values for cuphy_ex_ldpc command line
            printf("\n--- Suggested cuphy_ex_ldpc command ---\n");
            printf("cuphy_ex_ldpc -f -g %d -p %d -Z %d -w %d -n %d -a 35\n",
                   BG, mb, Z, total_cw, c.max_iterations);
            printf("=== END LDPC PARAMS ===\n");
        }
    }
#endif

} // namespace

////////////////////////////////////////////////////////////////////////
// HIGH_P_THRESHOLD: parity node count at/above which the high-p kernel
// variants are dispatched. Setting MIN_PARITY_ROWS = HIGH_P_THRESHOLD
// makes the schedule's IS_LAST_ROW runtime check vanish at compile time
// for rows below the threshold.
namespace { const int HIGH_P_THRESHOLD = 22; }
// ----------------------------------------------------------------------------
// ALGO 55 variant terminology. All variants are box-plus ("bp") x2 kernels and
// share the same min-sum-family decode; they differ only in how the BG1 core
// rows' C2V is stored (which drives the shared-memory footprint), and in the
// non-core register budget:
//
//   bp-full          : core C2V kept in box-plus storage in SHARED MEMORY.
//                      Largest shmem, fastest per-iteration core. Kernels carry
//                      NO _msc suffix (..._desc_dyn, ..._highp, + _tb forms).
//   bp-hybrid        : core C2V kept as compressed min-sum in REGISTERS (non-core
//                      C2V is in registers too). Low shmem, no register spills.
//                      Kernels carry the _msc suffix (..._msc, ..._highp_msc,
//                      + _tb_msc forms).
//   bp-hybrid-bigreg : bp-hybrid with a wider non-core register budget
//                      (NUM_REG_PARITY = BIGREG_NUM_REG_PARITY), trading register
//                      spills for still less shmem. Kernels: ..._bigreg_msc
//                      (+ _tb form). BG1 only.
//
// Variant selection (see decode()/decode_tb()/can_decode_config()/get_launch_config):
//  1. FIT-BASED cascade: pick the first variant whose shmem fits the device, in
//     priority order bp-full > bp-hybrid > bp-hybrid-bigreg. This derives the
//     crossovers from the device shmem budget rather than a hand-tuned parity
//     threshold, so it adapts across architectures:
//       * 227 KB (GH200/H100): bp-hybrid fits to p=30, bp-hybrid-bigreg to p=39
//         at Z=384.
//       * 99 KB (GB203): bp-hybrid fits only to p=22, so bp-hybrid-bigreg engages
//         at p=23-31.
//     bp-hybrid-bigreg requires p>=HIGH_P_THRESHOLD (compiled as a high-p kernel);
//     p beyond its fit ceiling is left to the chooser (ALGO 40/35/51).
//  2. OCCUPANCY-AWARE refinement: bp-full is demoted to bp-hybrid when bp-hybrid
//     would achieve strictly higher occupancy (its smaller shmem allows more
//     CTAs/SM). See occupancy_favors_bp_full(). This matters at small/mid Z
//     where bp-full fits but is starved to 1 CTA/SM.
namespace { const int BIGREG_NUM_REG_PARITY = 30; }

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_split_index_bp_x2_desc_dyn()
// Kernel for base graph 1 (legacy tensor interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_legacy(params, 1);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,                                          // BG
                                                           ldpc2::LDPC_kernel_params> kernel_config_t; // params struct

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
        //thread0_dump_app(reinterpret_cast<__half2*>(smem), params.Z_var);
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
// ldpc2_BG1_split_index_bp_x2_desc_dyn_msc()  [variant: bp-hybrid]
// Fallback kernel for BG1 when shared memory is insufficient for the bp-full
// core C2V storage. Uses compressed min-sum for core rows (0-3) in registers,
// box_plus for non-core rows (register-based, no shmem impact).
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn_msc(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_legacy(params, 1);
#endif

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           ldpc2::LDPC_kernel_params,
                                                           false> kernel_config_t;

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
// ldpc2_BG2_split_index_bp_x2_desc_dyn()
// Kernel for base graph 2 (legacy tensor interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_split_index_bp_x2_desc_dyn(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_legacy(params, 2);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,                                          // BG
                                                           ldpc2::LDPC_kernel_params> kernel_config_t; // params struct

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
// ldpc2_BG1_split_index_bp_x2_desc_dyn_tb()
// Kernel for base graph 1 (transport block interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    while(iter < config.max_iterations)
    {
        const bool early_term_enabled = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM));
        bool sign_change = false;
        if(early_term_enabled)
        {
            sign_change = sched.do_iteration_first4_sign_change();
        }
        else
        {
            sched.do_iteration();
        }
        ++iter;
        if(early_term_enabled)
        {
            const bool final_iteration = (iter >= config.max_iterations);
            const bool et_latency_debug = ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(
                decodeDesc.config.flags, CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG);
            const bool any_sign_change = (__syncthreads_count(sign_change) != 0u);
            if(final_iteration || (!et_latency_debug && !any_sign_change))
            {
                crc = ldpc2::should_terminate_early_crc_no_bitflip<kernel_config_t::BG>(
                    decodeDesc,
                    blockIdx.x,
                    reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                    et_ctx,
                    iter - 1);
            }
            else
            {
                crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
            }
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(smem,
                                   config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    int32_t iter = 0;
    auto crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
    while(iter < config.max_iterations)
    {
        const bool early_term_enabled = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM));
        bool sign_change = false;
        if(early_term_enabled)
        {
            sign_change = sched.do_iteration_first4_sign_change();
        }
        else
        {
            sched.do_iteration();
        }
        ++iter;
        if(early_term_enabled)
        {
            const bool final_iteration = (iter >= config.max_iterations);
            const bool et_latency_debug = ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(
                decodeDesc.config.flags, CUPHY_LDPC_DECODE_ET_LATENCY_DEBUG);
            const bool any_sign_change = (__syncthreads_count(sign_change) != 0u);
            if(final_iteration || (!et_latency_debug && !any_sign_change))
            {
                crc = ldpc2::should_terminate_early_crc_no_bitflip<kernel_config_t::BG>(
                    decodeDesc,
                    blockIdx.x,
                    reinterpret_cast<const kernel_config_t::app_buf_t*>(smem),
                    et_ctx,
                    iter - 1);
            }
            else
            {
                crc = ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::failure;
            }
            if(crc == 0) break;
        }
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_loop(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc()
// Fallback TB kernel for BG1 with compressed min-sum core rows.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false> kernel_config_t;

#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
// ldpc2_BG2_split_index_bp_x2_desc_dyn_tb()
// Kernel for base graph 2 (transport block interface)
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_split_index_bp_x2_desc_dyn_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct

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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
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

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(write_soft)
    {
        ldpc_dec_soft_output(soft_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_split_index_bp_x2_desc_dyn_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];
    // Shared memory is allocated dynamically

#ifdef LDPC_PRINT_PARAMS
    print_ldpc_params_tb(decodeDesc);
#endif

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,                                            // BG
                                                           cuphyLDPCDecodeConfigDesc_t> kernel_config_t; // params struct

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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
    ldpc2::early_term_initialize_if_enabled<ENABLE_ACCESSORY_FEATURES, ldpc2::et_crc_traits<kernel_config_t::app_buf_t>::cw_per_cta>(et_ctx, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
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

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(hard_out_params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_EARLY_TERM))
    {
        ldpc2::ldpc_dec_crc_output(decodeDesc, blockIdx.x, crc);
    }
    if(ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_ITER_COUNT))
    {
        ldpc2::ldpc_dec_iter_output<kernel_config_t::app_buf_t>(decodeDesc, blockIdx.x, iter);
    }
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           ldpc2::LDPC_kernel_params,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;

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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_msc(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           ldpc2::LDPC_kernel_params,
                                                           false,
                                                           HIGH_P_THRESHOLD> kernel_config_t;

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
void ldpc2_BG2_split_index_bp_x2_desc_dyn_highp(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,
                                                           ldpc2::LDPC_kernel_params,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;

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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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

// BIGREG variants of the BG1 compressed-min-sum-core (_msc) kernels: identical to
// _highp_msc / _highp_tb_msc but with NUM_REG_PARITY = BIGREG_NUM_REG_PARITY,
// holding more non-core C2V rows in registers to free shmem and reach p=31-39.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_msc(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           ldpc2::LDPC_kernel_params,
                                                           false,
                                                           HIGH_P_THRESHOLD,
                                                           BIGREG_NUM_REG_PARITY> kernel_config_t;

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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false,
                                                           HIGH_P_THRESHOLD,
                                                           BIGREG_NUM_REG_PARITY> kernel_config_t;
#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false, BIGREG_NUM_REG_PARITY>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<1,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           false,
                                                           HIGH_P_THRESHOLD,
                                                           BIGREG_NUM_REG_PARITY> kernel_config_t;
#if !LDPC_DECODE_USE_TB_SCAN
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc);
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc);
#else
    tb_token tok = kernel_config_t::llr_loader_t::load_sync_token(smem,
                                                                  decodeDesc,
                                                                  blockIdx.x,
                                                                  get_token_addr<1, false, BIGREG_NUM_REG_PARITY>(decodeDesc, smem));
    ldpc_dec_output_params<__half2> hard_out_params(decodeDesc, output_token(tok));
    ldpc_dec_soft_output_params<__half2> soft_out_params(decodeDesc, output_token(tok));
#endif
    const bool write_soft = (ldpc2::accessory_flag_enabled<ENABLE_ACCESSORY_FEATURES>(decodeDesc.config.flags, CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS));
    const cuphyLDPCDecodeConfigDesc_t config = decodeDesc.config;
    ldpc_et_context_x2_t* et_ctx = ldpc2::get_et_ctx_ptr<ldpc_et_context_x2_t>(smem,
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = true;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
void ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb_no_accessories(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    [[maybe_unused]] constexpr bool ENABLE_ACCESSORY_FEATURES = false;
    extern __shared__ char smem[];

    typedef ldpc2_split_index_bp_x2_desc_dyn_kernel_config<2,
                                                           cuphyLDPCDecodeConfigDesc_t,
                                                           true,
                                                           HIGH_P_THRESHOLD> kernel_config_t;
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
        kernel_config_t::shmem_required(config.num_parity_nodes, config.Z));
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
// occupancy_favors_bp_full()
// Occupancy-aware tie-break between the bp-full and bp-hybrid BG1 variants
// (see the variant terminology note above the HIGH_P_THRESHOLD definition).
// bp-full has the fastest per-iteration core but the largest shared memory
// (core C2V in shmem); when that shmem starves it of CTAs/SM relative to the
// lighter bp-hybrid variant (core C2V compressed in registers), bp-hybrid wins
// (its higher occupancy hides memory/instruction latency). Returns true to keep
// bp-full, false to demote to bp-hybrid. Both args must be BG1 kernels of the
// same interface (tensor or _tb) -- occupancy here is shmem-bound, so the highp
// / non-highp instantiations of a given variant are interchangeable as the
// proxy kernel.
//
// NOTE: cuOccupancyMaxActiveBlocksPerMultiprocessor is a cheap host-side
// query (no kernel launch / no sync) and depends only on (kernel, blockDim=Z,
// shmem). If it ever shows up on a hot path, the result can be precomputed and
// cached per (Z, num_parity) configuration.
static bool occupancy_favors_bp_full(const void* bp_full_kernel,
                                          const void* bp_hybrid_kernel,
                                          int          Z,
                                          uint32_t     shmem_bp_full,
                                          uint32_t     shmem_bp_hybrid)
{
    // Driver-API occupancy query (cuBB uses cu* APIs). cudaGetFuncBySymbol is the
    // sanctioned bridge from a __global__ symbol to a CUfunction. On any failure
    // return true (keep bp-full, the safe default): a transient query error must
    // never demote to a slower variant, and 0/0 occupancy would be ambiguous.
    cudaFunction_t func_full = nullptr, func_hybrid = nullptr;
    if((cudaGetFuncBySymbol(&func_full,   bp_full_kernel)   != cudaSuccess) ||
       (cudaGetFuncBySymbol(&func_hybrid, bp_hybrid_kernel) != cudaSuccess))
    {
        return true;
    }
    int occ_full = 0, occ_hybrid = 0;
    if((cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ_full,   static_cast<CUfunction>(func_full),   Z, shmem_bp_full)   != CUDA_SUCCESS) ||
       (cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ_hybrid, static_cast<CUfunction>(func_hybrid), Z, shmem_bp_hybrid) != CUDA_SUCCESS))
    {
        return true;
    }
    return occ_hybrid <= occ_full; // keep bp-full unless bp-hybrid is strictly higher occupancy
}

////////////////////////////////////////////////////////////////////////
// split_index_bp_x2_desc_dyn::decode()
cuphyStatus_t split_index_bp_x2_desc_dyn::decode(ldpc::decoder&                     dec,
                                                 LDPC_output_t&                     tDst,
                                                 const_tensor_pair&                 tLLR,
                                                 const cuphy_optional<tensor_pair>& optSoftOutputs,
                                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                                 cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_split_index_bp_x2_desc_dyn()\n");
    //------------------------------------------------------------------
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
    // Determine the dynamic amount of shared memory and choose the fastest
    // ALGO 55 variant whose requirement fits the device (fit-based cascade):
    //   bp-full > bp-hybrid > bp-hybrid-bigreg (see terminology note up top).
    const int      MAX_SHMEM       = dec.max_shmem_per_block_optin();
    const uint32_t SHMEM_SIZE_BP   = get_shmem_required<true>(config.BG,
                                                              config.num_parity_nodes,
                                                              config.Z);
    const uint32_t SHMEM_SIZE_MSC  = get_shmem_required<false>(config.BG,
                                                               config.num_parity_nodes,
                                                               config.Z);
    const bool     bp_full_fits   = (static_cast<int>(SHMEM_SIZE_BP)  <= MAX_SHMEM);
    const bool     bp_hybrid_fits = (static_cast<int>(SHMEM_SIZE_MSC) <= MAX_SHMEM);
    // Occupancy-aware refinement (BG1): demote bp-full to bp-hybrid when bp-hybrid
    // would achieve strictly higher occupancy (see occupancy_favors_bp_full).
    bool           use_bp_full    = bp_full_fits;
    if((1 == config.BG) && bp_full_fits && bp_hybrid_fits)
    {
        use_bp_full = occupancy_favors_bp_full((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp,
                                                         (const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_msc,
                                                         config.Z, SHMEM_SIZE_BP, SHMEM_SIZE_MSC);
    }
    const uint32_t SHMEM_SIZE     = use_bp_full ? SHMEM_SIZE_BP : SHMEM_SIZE_MSC;

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

                //------------------------------------------------------------------
                // Fit-based cascade: bp-full > bp-hybrid > bp-hybrid-bigreg
                const bool use_highp = (config.num_parity_nodes >= HIGH_P_THRESHOLD);
                if(use_bp_full)
                {
                    // bp-full: core C2V (box-plus) in shared memory.
                    if(use_highp)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp, blkDim, SHMEM_SIZE);
                        ldpc2_BG1_split_index_bp_x2_desc_dyn_highp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                    }
                    else
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn, blkDim, SHMEM_SIZE);
                        ldpc2_BG1_split_index_bp_x2_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                    }
                    s = CUPHY_STATUS_SUCCESS;
                }
                else if(bp_hybrid_fits)
                {
                    // bp-hybrid: core C2V as compressed min-sum in registers (no spills).
                    if(use_highp)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_msc, blkDim, SHMEM_SIZE);
                        ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_msc<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                    }
                    else
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_msc, blkDim, SHMEM_SIZE);
                        ldpc2_BG1_split_index_bp_x2_desc_dyn_msc<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                    }
                    s = CUPHY_STATUS_SUCCESS;
                }
                else if(use_highp)
                {
                    // bp-hybrid-bigreg (last resort): spill more non-core C2V to
                    // registers to free shmem. Requires p>=HIGH_P_THRESHOLD.
                    const uint32_t SHMEM_BIGREG = get_shmem_required<false, BIGREG_NUM_REG_PARITY>(1,
                                                                                                  config.num_parity_nodes,
                                                                                                  config.Z);
                    if(static_cast<int>(SHMEM_BIGREG) <= MAX_SHMEM)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_msc, blkDim, SHMEM_BIGREG);
                        ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_msc<<<grdDim, blkDim, SHMEM_BIGREG, strm>>>(params, *bgdesc);
                        s = CUPHY_STATUS_SUCCESS;
                    }
                }
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                if(config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_bp_x2_desc_dyn_highp, blkDim, SHMEM_SIZE);
                    ldpc2_BG2_split_index_bp_x2_desc_dyn_highp<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_bp_x2_desc_dyn, blkDim, SHMEM_SIZE);
                    ldpc2_BG2_split_index_bp_x2_desc_dyn<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
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
// split_index_bp_x2_desc_dyn::decode_tb()
cuphyStatus_t split_index_bp_x2_desc_dyn::decode_tb(ldpc::decoder&               dec,
                                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                    cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::split_index_bp_x2_desc_dyn::decode_tb()\n");

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    if(decodeDesc.config.llr_type == CUPHY_R_16F)
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
        // Determine the dynamic amount of shared memory and choose the fastest
        // ALGO 55 variant whose requirement fits the device (fit-based cascade):
        //   bp-full > bp-hybrid > bp-hybrid-bigreg (see terminology note up top).
        const int      MAX_SHMEM       = dec.max_shmem_per_block_optin();
        const uint32_t SHMEM_SIZE_BP   = get_shmem_required<true>(decodeDesc.config.BG,
                                                                  decodeDesc.config.num_parity_nodes,
                                                                  decodeDesc.config.Z);
        const uint32_t SHMEM_SIZE_MSC  = get_shmem_required<false>(decodeDesc.config.BG,
                                                                   decodeDesc.config.num_parity_nodes,
                                                                   decodeDesc.config.Z);
        const uint32_t SELECTED_SHMEM_SIZE_BP =
            get_selected_tb_shmem_required(SHMEM_SIZE_BP, decodeDesc.config.flags);
        const uint32_t SELECTED_SHMEM_SIZE_MSC =
            get_selected_tb_shmem_required(SHMEM_SIZE_MSC, decodeDesc.config.flags);
        const bool     bp_full_fits   = (static_cast<int>(SELECTED_SHMEM_SIZE_BP)  <= MAX_SHMEM);
        const bool     bp_hybrid_fits = (static_cast<int>(SELECTED_SHMEM_SIZE_MSC) <= MAX_SHMEM);
        if((2 == decodeDesc.config.BG) && !bp_full_fits && !bp_hybrid_fits)
        {
            return s;
        }
        // Occupancy-aware refinement (BG1): demote bp-full to bp-hybrid when bp-hybrid
        // would achieve strictly higher occupancy (see occupancy_favors_bp_full).
        bool           use_bp_full    = bp_full_fits;
        if((1 == decodeDesc.config.BG) && bp_full_fits && bp_hybrid_fits)
        {
            use_bp_full = occupancy_favors_bp_full((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb,
                                                             (const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc,
                                                             decodeDesc.config.Z, SELECTED_SHMEM_SIZE_BP, SELECTED_SHMEM_SIZE_MSC);
        }
        const uint32_t SHMEM_SIZE     = use_bp_full ? SHMEM_SIZE_BP : SHMEM_SIZE_MSC;
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                //------------------------------------------------------------------
                // Fit-based cascade: bp-full > bp-hybrid > bp-hybrid-bigreg
                const bool use_highp = (decodeDesc.config.num_parity_nodes >= HIGH_P_THRESHOLD);
                if(use_bp_full)
                {
                    // bp-full: core C2V (box-plus) in shared memory.
                    if(use_highp)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb, blkDim, SHMEM_SIZE);
                        LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb, decodeDesc.config.flags,
                            grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                    }
                    else
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb, blkDim, SHMEM_SIZE);
                        LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb, decodeDesc.config.flags,
                            grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                    }
                    s = CUPHY_STATUS_SUCCESS;
                }
                else if(bp_hybrid_fits)
                {
                    // bp-hybrid: core C2V as compressed min-sum in registers (no spills).
                    if(use_highp)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc, blkDim, SHMEM_SIZE);
                        LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc, decodeDesc.config.flags, grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                    }
                    else
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc, blkDim, SHMEM_SIZE);
                        LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc, decodeDesc.config.flags, grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                    }
                    s = CUPHY_STATUS_SUCCESS;
                }
                else if(use_highp)
                {
                    // bp-hybrid-bigreg (last resort): spill more non-core C2V to
                    // registers to free shmem. Requires p>=HIGH_P_THRESHOLD.
                    const uint32_t SHMEM_BIGREG = get_shmem_required<false, BIGREG_NUM_REG_PARITY>(1,
                                                                                                  decodeDesc.config.num_parity_nodes,
                                                                                                  decodeDesc.config.Z);
                    if(static_cast<int>(get_selected_tb_shmem_required(SHMEM_BIGREG, decodeDesc.config.flags)) <= MAX_SHMEM)
                    {
                        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc, blkDim, SHMEM_BIGREG);
                        LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc, decodeDesc.config.flags, grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_BIGREG), strm, decodeDesc, *bgdesc);
                        s = CUPHY_STATUS_SUCCESS;
                    }
                }
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                if(decodeDesc.config.num_parity_nodes >= HIGH_P_THRESHOLD)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb, decodeDesc.config.flags,
                        grdDim, blkDim, shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE), strm, decodeDesc, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_split_index_bp_x2_desc_dyn_tb, blkDim, SHMEM_SIZE);
                    LDPC_LAUNCH_TB_KERNEL_X2(ldpc2_BG2_split_index_bp_x2_desc_dyn_tb, decodeDesc.config.flags,
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
// split_index_bp_x2_desc_dyn::get_workspace_size()
std::pair<bool, size_t> split_index_bp_x2_desc_dyn::get_workspace_size(const ldpc::decoder&               dec,
                                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                                       int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// split_index_bp_x2_desc_dyn::split_index_bp_x2_desc_dyn()
split_index_bp_x2_desc_dyn::split_index_bp_x2_desc_dyn(ldpc::decoder& dec)
{
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_BP   = static_cast<int>(get_shmem_required<true>(1,                             // BG (bp-full: box_plus core in shmem)
                                                                              max_num_parity<1>::value,      // max parity nodes
                                                                              CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    const int MAX_BG1_SHMEM_MSC  = static_cast<int>(get_shmem_required<false>(1,                             // BG (bp-hybrid: compressed min-sum core in regs)
                                                                               max_num_parity<1>::value,
                                                                               CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_BG1_SHMEM_BIGREG = static_cast<int>(get_shmem_required<false, BIGREG_NUM_REG_PARITY>(1, // BG (bp-hybrid-bigreg)
                                                                                                       max_num_parity<1>::value,
                                                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE));
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(get_shmem_required(2,                             // BG
                                                                       max_num_parity<2>::value,      // max parity nodes
                                                                       CUPHY_LDPC_MAX_LIFTING_SIZE)); // lifting size
    //------------------------------------------------------------------
    // Maximum shared memory supported by the device
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size.
    // Box_plus core kernels get their full shmem requirement (clamped to
    // device max). Fallback compressed min-sum core kernels get their smaller
    // requirement (always fits).
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 21> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_msc,     std::min(MAX_BG1_SHMEM_BIGREG, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc,  std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BIGREG)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BIGREG)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn,                std::min(MAX_BG1_SHMEM_BP, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_msc,            std::min(MAX_BG1_SHMEM_MSC, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn,                std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_tb,             std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_no_accessories,             std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc,         std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_MSC)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_MSC)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn_tb,             std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn_tb_no_accessories,             std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp,          std::min(MAX_BG1_SHMEM_BP, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_msc,      std::min(MAX_BG1_SHMEM_MSC, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn_highp,          std::min(MAX_BG2_SHMEM_SIZE, MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_no_accessories,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_BP)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc,   std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_MSC)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc_no_accessories, std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG1_SHMEM_MSC)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM)),
        func_attr_t((const void*)ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb_no_accessories,       std::min(static_cast<int>(shmem_size_with_et_context<ldpc_et_context_x2_t>(MAX_BG2_SHMEM_SIZE)), MAX_SHMEM))
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
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_bp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_bp_x2_desc_dyn);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_split_index_bp_x2_desc_dyn_tb);
}

////////////////////////////////////////////////////////////////////////
// split_index_bp_x2_desc_dyn::can_decode_config()
bool split_index_bp_x2_desc_dyn::can_decode_config(const ldpc::decoder&               dec,
                                                   const cuphyLDPCDecodeConfigDesc_t& cfg)
{
    // Compare shared memory requirements to device maximum, as well as
    // the maximum that the kernel was compiled for.

    // Maximum number of parity nodes, as limited by compilation, to
    // limit register usage.
    const uint32_t MAX_NUM_PARITY = (1 == cfg.BG) ? max_num_parity<1>::value : max_num_parity<2>::value;
    if(cfg.num_parity_nodes > MAX_NUM_PARITY)
    {
        return false;
    }
    // Fit-based cascade (mirrors decode()/decode_tb()): decodable if any of the
    // ALGO 55 variants fits the device shared-memory budget, in priority order
    // bp-full > bp-hybrid > bp-hybrid-bigreg (bigreg BG1 only, p>=HIGH_P_THRESHOLD).
    const int MAX_SHMEM = dec.max_shmem_per_block_optin();
    if(static_cast<int>(get_selected_tb_shmem_required(
           get_shmem_required<true>(cfg.BG, cfg.num_parity_nodes, cfg.Z),
           cfg.flags)) <= MAX_SHMEM)
    {
        return true;
    }
    if(static_cast<int>(get_selected_tb_shmem_required(
           get_shmem_required<false>(cfg.BG, cfg.num_parity_nodes, cfg.Z),
           cfg.flags)) <= MAX_SHMEM)
    {
        return true;
    }
    if((1 == cfg.BG) &&
       (cfg.num_parity_nodes >= static_cast<uint32_t>(HIGH_P_THRESHOLD)) &&
       (static_cast<int>(get_selected_tb_shmem_required(
            get_shmem_required<false, BIGREG_NUM_REG_PARITY>(1, cfg.num_parity_nodes, cfg.Z),
            cfg.flags)) <= MAX_SHMEM))
    {
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////
// split_index_bp_x2_desc_dyn::get_launch_config()
cuphyStatus_t split_index_bp_x2_desc_dyn::get_launch_config(const ldpc::decoder&           dec,
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

    // Determine shmem and kernel variant — fit-based cascade then occupancy refinement
    const int      MAX_SHMEM     = dec.max_shmem_per_block_optin();

    const uint32_t SHMEM_SIZE_BP  = get_shmem_required<true>(BG, NUM_PARITY_NODES, Z);
    const uint32_t SHMEM_SIZE_MSC = get_shmem_required<false>(BG, NUM_PARITY_NODES, Z);
    // Fit-based cascade (mirrors decode()/decode_tb()): bp-full > bp-hybrid >
    // bp-hybrid-bigreg. use_bp_full selects BG1's core-row storage: box-plus
    // in shared memory if it fits, otherwise compressed min-sum in registers.
    // BG2 has no such choice; its core is always compressed min-sum
    // (core_storage_x2_local<2>), and its <true>/<false> shmem sizes are equal,
    // so it is sized correctly on the bp-hybrid path.
    const uint32_t SELECTED_SHMEM_SIZE_BP =
        get_selected_tb_shmem_required(SHMEM_SIZE_BP, launchConfig.decode_desc.config.flags);
    const uint32_t SELECTED_SHMEM_SIZE_MSC =
        get_selected_tb_shmem_required(SHMEM_SIZE_MSC, launchConfig.decode_desc.config.flags);
    const bool     bp_full_fits   = (static_cast<int>(SELECTED_SHMEM_SIZE_BP)  <= MAX_SHMEM);
    const bool     bp_hybrid_fits = (static_cast<int>(SELECTED_SHMEM_SIZE_MSC) <= MAX_SHMEM);
    const bool     use_highp           = (NUM_PARITY_NODES >= HIGH_P_THRESHOLD);
    // Occupancy-aware refinement (BG1): demote bp-full to bp-hybrid when bp-hybrid
    // would achieve strictly higher occupancy (see occupancy_favors_bp_full).
    bool           use_bp_full    = bp_full_fits;
    if((BG == 1) && bp_full_fits && bp_hybrid_fits)
    {
        use_bp_full = occupancy_favors_bp_full((const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb,
                                                         (const void*)ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc,
                                                         Z, SELECTED_SHMEM_SIZE_BP, SELECTED_SHMEM_SIZE_MSC);
    }
    // bp-hybrid-bigreg is a BG1-only last resort, used only when neither bp-full nor
    // bp-hybrid fit; it is compiled as a high-p kernel (requires p>=HIGH_P_THRESHOLD)
    // and, like decode()/decode_tb(), only when its own reduced shmem footprint fits.
    const uint32_t SHMEM_SIZE_BIGREG    = get_shmem_required<false, BIGREG_NUM_REG_PARITY>(BG, NUM_PARITY_NODES, Z);
    const bool     bigreg_fits = (static_cast<int>(get_selected_tb_shmem_required(
        SHMEM_SIZE_BIGREG, launchConfig.decode_desc.config.flags)) <= MAX_SHMEM);
    const bool     use_bp_hybrid_bigreg = (BG == 1) && !bp_full_fits && !bp_hybrid_fits && use_highp && bigreg_fits;
    // If no BG1 variant fits, the config is undecodable (mirrors can_decode_config()
    // and decode()/decode_tb()). Bail out rather than populate sharedMemBytes with a
    // value exceeding MAX_SHMEM, which would fail at graph launch on a direct call.
    if(!bp_full_fits && !bp_hybrid_fits && !use_bp_hybrid_bigreg)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
    const uint32_t SHMEM_SIZE     = use_bp_full          ? SHMEM_SIZE_BP
                                  : use_bp_hybrid_bigreg ? SHMEM_SIZE_BIGREG
                                                         : SHMEM_SIZE_MSC;
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_size_with_et_context<ldpc_et_context_x2_t>(SHMEM_SIZE);
    launchConfig.kernel_node_params_driver.sharedMemBytes =
        shmem_size_for_selected_kernel<ldpc_et_context_x2_t>(
            launchConfig.kernel_node_params_driver.sharedMemBytes,
            launchConfig.decode_desc.config.flags);

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    // Fit-based cascade selection (mirrors decode()/decode_tb()): bp-full >
    // bp-hybrid-bigreg > bp-hybrid. Kept as an if-else chain for readability.
    const void* bg1_kernel = nullptr;
    if(use_bp_full)
    {
        bg1_kernel = use_highp ? LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb, launchConfig.decode_desc.config.flags)
                               : LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb, launchConfig.decode_desc.config.flags);
    }
    else if(use_bp_hybrid_bigreg)
    {
        bg1_kernel = LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_split_index_bp_x2_desc_dyn_bigreg_tb_msc, launchConfig.decode_desc.config.flags);
    }
    else // bp-hybrid
    {
        bg1_kernel = use_highp ? LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_split_index_bp_x2_desc_dyn_highp_tb_msc, launchConfig.decode_desc.config.flags)
                               : LDPC_TB_KERNEL_SYMBOL(ldpc2_BG1_split_index_bp_x2_desc_dyn_tb_msc, launchConfig.decode_desc.config.flags);
    }
    const void* bg2_kernel = use_highp ? LDPC_TB_KERNEL_SYMBOL(ldpc2_BG2_split_index_bp_x2_desc_dyn_highp_tb, launchConfig.decode_desc.config.flags)
                                       : LDPC_TB_KERNEL_SYMBOL(ldpc2_BG2_split_index_bp_x2_desc_dyn_tb, launchConfig.decode_desc.config.flags);
    cudaError_t    e = (BG == 1) ? cudaGetFuncBySymbol(&deviceFunction, bg1_kernel)
                                 : cudaGetFuncBySymbol(&deviceFunction, bg2_kernel);
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
