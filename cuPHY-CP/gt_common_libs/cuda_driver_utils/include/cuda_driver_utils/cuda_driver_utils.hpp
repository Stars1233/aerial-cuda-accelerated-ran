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

/**
 * @file cuda_driver_utils.hpp
 * @brief CUDA_DRIVER_CHECK / CUDA_DRIVER_CHECK_NON_FATAL macros and PrimaryCtxGuard for CUDA Driver API usage.
 *
 * Call sites use direct CUDA driver APIs (cuMemAlloc, cuEventRecord, etc.) wrapped with
 * CUDA_DRIVER_CHECK (fatal log on failure) or CUDA_DRIVER_CHECK_NON_FATAL in destructors/teardown.
 */

#ifndef CUDA_DRIVER_UTILS_HPP
#define CUDA_DRIVER_UTILS_HPP

#include <cuda.h>

#include <sstream>
#include <stdexcept>

#include "nvlog.hpp"

#ifndef TAG_CUDA_DRV_UTILS
#define TAG_CUDA_DRV_UTILS (NVLOG_TAG_BASE_CUPHY_DRIVER + 32)
#endif

/* Use caller's TAG for component-level traceability when defined; otherwise fall back to TAG_CUDA_DRV_UTILS. */
#ifdef TAG
#define CUDA_DRIVER_CHECK_LOG_TAG TAG
#else
#define CUDA_DRIVER_CHECK_LOG_TAG TAG_CUDA_DRV_UTILS
#endif

/**
 * @brief Check CUDA driver API (CUresult) call; log on failure (numeric code and cuGetErrorString text when available).
 * Log tag: uses caller's TAG if defined (e.g. DRV.GPUDEV, DRV.CONTEXT), else TAG_CUDA_DRV_UTILS.
 */
#ifndef CUDA_DRIVER_CHECK
#define CUDA_DRIVER_CHECK(stmt)                                                           \
    do                                                                                     \
    {                                                                                      \
        CUresult cu_result1 = (stmt);                                                      \
        if(CUDA_SUCCESS != cu_result1)                                                     \
        {                                                                                  \
            const char* cu_err_str_ = nullptr;                                              \
            CUresult cu_ges_res_ = cuGetErrorString(cu_result1, &cu_err_str_);        \
            if(CUDA_SUCCESS == cu_ges_res_ && cu_err_str_ != nullptr)                      \
            {                                                                              \
                NVLOGF_FMT(CUDA_DRIVER_CHECK_LOG_TAG, AERIAL_CUDA_API_EVENT,               \
                           "[{}:{}] CUDA Driver API call failed with {} ({})",             \
                           __FILE__, __LINE__, static_cast<unsigned>(cu_result1), cu_err_str_); \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                NVLOGF_FMT(CUDA_DRIVER_CHECK_LOG_TAG, AERIAL_CUDA_API_EVENT,               \
                           "[{}:{}] CUDA Driver API call failed with {}",                  \
                           __FILE__, __LINE__, static_cast<unsigned>(cu_result1));        \
            }                                                                              \
        }                                                                                  \
    } while(0)
#endif

/**
 * @brief Same as CUDA_DRIVER_CHECK but logs at error level (NVLOGE_FMT) without fatal exit.
 * Use in destructors and other teardown paths where NVLOGF_FMT / test_trigger_exit must not run.
 */
#ifndef CUDA_DRIVER_CHECK_NON_FATAL
#define CUDA_DRIVER_CHECK_NON_FATAL(stmt)                                                 \
    do                                                                                     \
    {                                                                                      \
        CUresult cu_result_nf = (stmt);                                                    \
        if(CUDA_SUCCESS != cu_result_nf)                                                   \
        {                                                                                  \
            const char* cu_err_str_nf_ = nullptr;                                          \
            CUresult cu_ges_res_nf_ = cuGetErrorString(cu_result_nf, &cu_err_str_nf_); \
            if(CUDA_SUCCESS == cu_ges_res_nf_ && cu_err_str_nf_ != nullptr)                \
            {                                                                              \
                NVLOGE_FMT(CUDA_DRIVER_CHECK_LOG_TAG, AERIAL_CUDA_API_EVENT,               \
                           "[{}:{}] CUDA Driver API call failed with {} ({})",             \
                           __FILE__, __LINE__, static_cast<unsigned>(cu_result_nf), cu_err_str_nf_); \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                NVLOGE_FMT(CUDA_DRIVER_CHECK_LOG_TAG, AERIAL_CUDA_API_EVENT,               \
                           "[{}:{}] CUDA Driver API call failed with {}",                  \
                           __FILE__, __LINE__, static_cast<unsigned>(cu_result_nf));      \
            }                                                                              \
        }                                                                                  \
    } while(0)
#endif

/**
 * @brief Check CUDA Driver API call and throw std::runtime_error on failure.
 *
 * Use in code paths where failure should propagate as an exception
 * (constructors, initialisation). For destructors/teardown, use
 * CUDA_DRIVER_CHECK_NON_FATAL instead.
 */
#ifndef CHECK_CU_THROW
#define CHECK_CU_THROW_STR_(x) #x
#define CHECK_CU_THROW_STR(x) CHECK_CU_THROW_STR_(x)
#define CHECK_CU_THROW(expr)                                                                      \
    do                                                                                             \
    {                                                                                              \
        CUresult cu_res_throw_ = (expr);                                                           \
        if(cu_res_throw_ != CUDA_SUCCESS)                                                          \
        {                                                                                          \
            const char* cu_err_throw_ = nullptr;                                                   \
            cuGetErrorString(cu_res_throw_, &cu_err_throw_);                                       \
            std::ostringstream oss_;                                                               \
            oss_ << "[" << __FILE__ << ":" << __LINE__ << "] "                                     \
                 << "CUDA Driver API call failed with "                                            \
                 << static_cast<unsigned>(cu_res_throw_) << ": "                                   \
                 << (cu_err_throw_ ? cu_err_throw_ : "unknown")                                    \
                 << ". Failed call: " CHECK_CU_THROW_STR(expr);                                    \
            throw std::runtime_error(oss_.str());                                                  \
        }                                                                                          \
    } while(0)
#endif

/**
 * @brief RAII guard that retains the CUDA primary context for a device and releases it on destruction.
 *
 * Constructor calls cuInit, cuDeviceGet, cuDevicePrimaryCtxRetain, and cuCtxSetCurrent.
 * Destructor calls cuDevicePrimaryCtxRelease (non-fatal on failure, safe in stack unwinding).
 */
struct PrimaryCtxGuard final
{
    CUdevice dev;

    explicit PrimaryCtxGuard(int gpu_id)
    {
        CUDA_DRIVER_CHECK(cuInit(0));
        CUDA_DRIVER_CHECK(cuDeviceGet(&dev, gpu_id));
        CUcontext ctx;
        CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
        CUDA_DRIVER_CHECK(cuCtxSetCurrent(ctx));
    }

    ~PrimaryCtxGuard() noexcept
    {
        CUDA_DRIVER_CHECK_NON_FATAL(cuDevicePrimaryCtxRelease(dev));
    }

    PrimaryCtxGuard(const PrimaryCtxGuard&) = delete;
    PrimaryCtxGuard& operator=(const PrimaryCtxGuard&) = delete;
};

/**
 * Allocate CUDA managed memory and store the result in a typed pointer.
 * Wraps cuMemAllocManaged with CU_MEM_ATTACH_GLOBAL and handles the
 * CUdeviceptr-to-typed-pointer cast.
 */
template <typename T>
void managed_alloc(T** ptr, size_t size)
{
    CUdeviceptr d = 0;
    CUDA_DRIVER_CHECK(cuMemAllocManaged(&d, size, CU_MEM_ATTACH_GLOBAL));
    *ptr = reinterpret_cast<T*>(static_cast<uintptr_t>(d));
}

#endif // CUDA_DRIVER_UTILS_HPP
