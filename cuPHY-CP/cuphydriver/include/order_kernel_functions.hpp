/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

#ifndef ORDER_KERNEL_FUNCTIONS_HPP
#define ORDER_KERNEL_FUNCTIONS_HPP

#include <cuda.h>

/**
 * @brief Resolved CUfunction handles for the non-templated order kernels.
 *
 * Populated once at init via init_order_kernel_functions() using
 * cudaGetFuncBySymbol, then used by launch_* wrappers with cuLaunchKernel.
 *
 * kernel_order is resolved per MPS context because CUfunction handles are
 * context-specific and kernel_order runs under PUSCH, PUCCH, PRACH and SRS
 * contexts (via warmupStream and aggregator constructors).
 */
struct OrderKernelFunctions final {
    CUfunction order_kernel_doca                          = nullptr;
    CUfunction order_kernel_doca_single                   = nullptr;
    CUfunction order_kernel_doca_single_srs               = nullptr;
    CUfunction order_kernel_doca_single_subSlot            = nullptr;
    CUfunction order_kernel_cpu_init_comms_single_subSlot  = nullptr;
    CUfunction kernel_order_pusch                         = nullptr;
    CUfunction kernel_order_pucch                         = nullptr;
    CUfunction kernel_order_prach                         = nullptr;
    CUfunction kernel_order_srs                           = nullptr;
    CUfunction kernel_order_mb_one_ch                     = nullptr;
    CUfunction kernel_order_mb_two_ch                     = nullptr;
    CUfunction receive_kernel_for_test_bench              = nullptr;
    CUfunction receive_process_kernel_for_test_bench      = nullptr;
    CUfunction order_kernel_printf_warmup                 = nullptr;  ///< Init-time device printf warmup kernel.

    // Ping-pong kernel instantiations: <is_test_bench, pkt_trace, srs_enable, threads, ctas>
    CUfunction pingpong_trace_srs                         = nullptr;  // <false, 1, 1, 1024, 1>
    CUfunction pingpong_trace_srs_pusch                   = nullptr;  // <false, 1, 3, 1024, 1>
    CUfunction pingpong_trace_no_srs                      = nullptr;  // <false, 1, 0, 320,  2>
    CUfunction pingpong_no_trace_srs                      = nullptr;  // <false, 0, 1, 1024, 1>
    CUfunction pingpong_no_trace_srs_pusch                = nullptr;  // <false, 0, 3, 1024, 1>
    CUfunction pingpong_no_trace_no_srs                   = nullptr;  // <false, 0, 0, 320,  2>
};

/**
 * @brief Resolve CUfunction handles for all order kernels via cudaGetFuncBySymbol.
 *
 * Populates every field in @p funcs except the per-context kernel_order_*
 * handles (PUSCH, PUCCH, PRACH, SRS), which must be resolved separately
 * with resolve_kernel_order_handle() under each MPS context.
 *
 * @param[out] funcs  Struct whose CUfunction fields are filled in.
 * @return true if all resolutions succeeded, false if any failed.
 */
[[nodiscard]] bool init_order_kernel_functions(OrderKernelFunctions& funcs);

/**
 * @brief Resolve the CUfunction handle for kernel_order in the current CUDA context.
 *
 * Must be called once per MPS context (PUSCH, PUCCH, PRACH, SRS) so that
 * cuLaunchKernel receives a context-appropriate handle.
 *
 * @param[out] out  Pointer to the CUfunction to populate.
 * @return true on success, false if cudaGetFuncBySymbol failed.
 */
[[nodiscard]] bool resolve_kernel_order_handle(CUfunction* out);

/**
 * @brief Resolved CUfunction handles used exclusively by the order kernel test bench.
 *
 * The test bench is a standalone executable with its own CUDA context, so these
 * handles must be resolved separately from the production OrderKernelFunctions.
 */
struct OrderKernelTbFunctions final {
    // Ping-pong kernel instantiations: <is_test_bench=true, pkt_trace=0, srs_enable, threads, ctas>
    CUfunction tb_pingpong_srs{};     // <true, 0, 1, SRS_THREADS, 1>
    CUfunction tb_pingpong_no_srs{};  // <true, 0, 0, NUM_THREADS, 2>
    CUfunction recv_process{};        // receive_process_kernel_for_test_bench (dual CTA)
};

/**
 * @brief Resolve CUfunction handles for the test-bench-only order kernels.
 *
 * Must be called after the test bench's CUDA context is set up.
 *
 * @param[out] funcs  Struct whose CUfunction fields are filled in.
 * @return true if all resolutions succeeded, false if any failed.
 */
[[nodiscard]] bool init_order_kernel_tb_functions(OrderKernelTbFunctions& funcs);

#endif // ORDER_KERNEL_FUNCTIONS_HPP
