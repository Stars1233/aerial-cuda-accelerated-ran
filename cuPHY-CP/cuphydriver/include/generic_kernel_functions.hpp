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

#ifndef GENERIC_KERNEL_FUNCTIONS_HPP
#define GENERIC_KERNEL_FUNCTIONS_HPP

#include <cuda.h>

/**
 * @brief Resolved CUfunction handles for the DL compression kernels.
 *
 * Grouped because these 5 handles always travel together (resolve, store, launch).
 */
struct CompressionKernelFunctions final {
    CUfunction compress_0{}; //!< kernel_compress<0> (generic / non-specialized)
    CUfunction compress_9{}; //!< kernel_compress<9>
    CUfunction compress_14{}; //!< kernel_compress<14>
    CUfunction compress_16{}; //!< kernel_compress<16>
    CUfunction mod_compression_qam{}; //!< kernel_mod_compression<QAM_Comp>
};

/**
 * @brief Resolved CUfunction handles for the generic CUDA kernels in cuphydriver.
 *
 * Populated once at init via init_generic_cuda_kernel_functions() using
 * cudaGetFuncBySymbol, then used by launch_* wrappers with cuLaunchKernel.
 */
struct GenericCudaKernelFunctions final {
    CUfunction print_complex_fp16         = nullptr;
    CUfunction print_hexbytes             = nullptr;
    CUfunction kernel_write               = nullptr;
    CUfunction kernel_read                = nullptr;
    CUfunction warmup_kernel              = nullptr;
    CUfunction kernel_wait_update         = nullptr;
    CUfunction kernel_wait_eq             = nullptr;
    CUfunction kernel_wait_neq            = nullptr;
    CUfunction kernel_wait_geq            = nullptr;
    CUfunction kernel_compare             = nullptr;
    CUfunction kernel_check_crc           = nullptr;
    CUfunction kernel_copy                = nullptr;
    CUfunction memset_kernel              = nullptr;
    CompressionKernelFunctions compression; //!< DL compression kernel handles
};

/**
 * @brief Resolve CUfunction handles for all generic CUDA kernels via cudaGetFuncBySymbol.
 *
 * Populates every field in @p funcs.  Must be called after the CUDA context
 * is established (typically during PhyDriverCtx construction).
 *
 * @param[out] funcs  Struct whose CUfunction fields are filled in.
 */
[[nodiscard]] bool init_generic_cuda_kernel_functions(GenericCudaKernelFunctions& funcs);

#endif // GENERIC_KERNEL_FUNCTIONS_HPP
