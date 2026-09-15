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

#ifndef CUDA_KERNEL_UTILS_CUH
#define CUDA_KERNEL_UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvlog.hpp"  // NVLOGE_FMT, AERIAL_CUDA_KERNEL_EVENT (via nvlog.h → aerial_event_code.h)

/**
 * Resolve a CUfunction handle from a __global__ kernel symbol.
 *
 * Wraps cudaGetFuncBySymbol with logging on failure. The resolved handle
 * is valid only within the CUDA context that was current at the time of
 * the call. This header must be included in a .cu translation unit that
 * has TAG defined before including this header (for NVLOGE_FMT).
 *
 * @tparam     tag     NVLOG integer component tag (compile-time constant, e.g. TAG).
 * @param[out] out     Pointer to the CUfunction to populate.
 * @param[in]  symbol  Address of the __global__ kernel function.
 * @param[in]  name    Human-readable kernel name (for error messages).
 * @return true on success, false if resolution failed.
 */
template <int tag>
[[nodiscard]] static inline bool resolve_kernel_func(CUfunction* out, const void* symbol, const char* name) noexcept
{
    if(!out || !symbol)
    {
        NVLOGE_FMT(tag, AERIAL_CUDA_KERNEL_EVENT, "resolve_kernel_func({}): null {} argument",
                   name ? name : "<null>", !out ? "out" : "symbol");
        return false;
    }

    cudaError_t err = cudaGetFuncBySymbol(out, symbol);
    if(err != cudaSuccess)
    {
        NVLOGE_FMT(tag, AERIAL_CUDA_KERNEL_EVENT, "cudaGetFuncBySymbol({}) failed: {}",
                   name ? name : "<unknown>", cudaGetErrorString(err));
        return false;
    }
    return true;
}

#endif // CUDA_KERNEL_UTILS_CUH
