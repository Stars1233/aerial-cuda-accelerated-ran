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

#ifndef CUPHY_ERR_HPP
#define CUPHY_ERR_HPP

#include <stdexcept>

#include "cuda.h"
#include "driver_types.h"
#include "cuda_runtime_api.h"

#include "cuphy.h"

namespace cuphy {

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::cuda_exception
// Exception class for errors from CUDA
class cuda_exception : public std::exception //
{
public:
    cuda_exception(cudaError_t s) : status_(s) { }
    virtual ~cuda_exception() = default;
    virtual const char* what() const noexcept { return cudaGetErrorString(status_); }
private:
    cudaError_t status_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::cuda_driver_exception
// Exception class for errors from CUDA driver
class cuda_driver_exception : public std::exception //
{
public:
    cuda_driver_exception(CUresult result, const char* pUsrStr = nullptr) : m_result(result)
    {
        const char* pResNameStr;
        CUresult e1 = cuGetErrorName(m_result, &pResNameStr);
        const char* pResDescriptionStr;
        CUresult e2 = cuGetErrorString(m_result, &pResDescriptionStr);

        m_dispStr = std::string("CUDA driver error: ");
        m_dispStr.append((e1 == CUDA_SUCCESS) ? pResNameStr : std::to_string(m_result));
        m_dispStr.append(" - ");
        m_dispStr.append((e2 == CUDA_SUCCESS) ? pResDescriptionStr : std::to_string(m_result));

        if(pUsrStr)
        {
            m_dispStr.append(", ");
            m_dispStr.append(pUsrStr);
        }
    }
    virtual ~cuda_driver_exception() = default;
    virtual const char* what() const noexcept { return m_dispStr.c_str(); }
private:
    std::string m_dispStr;
    CUresult    m_result;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::cuphy_exception
// Exception class for errors from the cuphy library
class cuphy_exception : public std::exception //
{
public:
    cuphy_exception(cuphyStatus_t s) : status_(s) { }
    virtual ~cuphy_exception() = default;
    virtual const char* what() const noexcept { return cuphyGetErrorString(status_); }
private:
    cuphyStatus_t status_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::cuphy_fn_exception
// Exception class for errors from the cuphy library, providing the
// name of the function that encountered the exception as specified in
// the constructor.
class cuphy_fn_exception : public std::exception //
{
public:
    cuphy_fn_exception(cuphyStatus_t s, const char* fn) : status_(s)
    {
        desc_ = std::string("Function ") + fn;
        desc_.append(" returned ");
        desc_.append(cuphyGetErrorName(status_));
        desc_.append(": ");
        desc_.append(cuphyGetErrorString(status_));
    }
    virtual ~cuphy_fn_exception() = default;
    virtual const char* what() const noexcept { return desc_.c_str(); }
private:
    cuphyStatus_t status_;
    std::string   desc_;
};
// clang-format on

#define CUPHY_CHECK(c)                                             \
    do                                                             \
    {                                                              \
        cuphyStatus_t s = c;                                       \
        if(s != CUPHY_STATUS_SUCCESS)                              \
        {                                                          \
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,   \
            "CUPHY_ERROR: {} ({})", __FILE__, __LINE__);           \
            throw cuphy::cuphy_exception(s);                       \
        }                                                          \
    } while(0)


#define CUDA_CHECK_EXCEPTION(c) \
_Pragma("vcast_dont_instrument_start")                             \
do {                                                               \
        cudaError_t s = c;                                         \
        if((cudaError_t)s != cudaSuccess)                          \
        {                                                          \
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,   \
            "[{}:{}] CUDA runtime error {}", \
                    __FILE__,                                      \
                    __LINE__,                                      \
                    cudaGetErrorString(s));                        \
            throw cuphy::cuda_exception(s);                        \
        }                                                          \
    } while(0)                                                     \
_Pragma("vcast_dont_instrument_end")                               \

#define CU_CHECK_EXCEPTION(c) do {                                 \
        CUresult s = c;                                            \
        if((CUresult)s != CUDA_SUCCESS)                            \
        {                                                          \
            const char* pErrStr;                                   \
            cuGetErrorString(s,&pErrStr);                          \
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,   \
            "[{}:{}] CUDA driver error {}",                        \
                    __FILE__,                                      \
                    __LINE__,                                      \
                    pErrStr);                                      \
            throw cuphy::cuda_driver_exception(s);                 \
        }                                                          \
    } while(0)


#define CUDA_CHECK_EXCEPTION_W_TAG(tag, c) do {       \
        cudaError_t s = c;                            \
        if((cudaError_t)s != cudaSuccess)             \
        {                                             \
            NVLOGE_FMT(tag, AERIAL_CUPHY_EVENT,       \
            "[{}:{}] CUDA runtime error {}",          \
                    __FILE__,                         \
                    __LINE__,                         \
                    cudaGetErrorString(s));           \
            throw cuphy::cuda_exception(s);           \
        }                                             \
    } while(0)


#define CU_CHECK_EXCEPTION_W_TAG(tag, c) do {       \
        CUresult s = c;                             \
        if((CUresult)s != CUDA_SUCCESS)             \
        {                                           \
            const char* pErrStr;                    \
            cuGetErrorString(s,&pErrStr);           \
            NVLOGE_FMT(tag, AERIAL_CUPHY_EVENT,     \
            "[{}:{}] CUDA driver error {}",         \
                    __FILE__,                       \
                    __LINE__,                       \
                    pErrStr);                       \
            throw cuphy::cuda_driver_exception(s);  \
        }                                           \
    } while(0)


// The following are CUDA check macros that print information to stderr.
// Having pycuphy only call these avoids issues with nvlog during pytest.

#define CUPHY_CHECK_PRINTF_VERSION(c)                 \
    do                                                \
    {                                                 \
        cuphyStatus_t s = c;                          \
        if(s != CUPHY_STATUS_SUCCESS)                 \
        {                                             \
            fprintf(stderr, "CUPHY_ERROR: %s (%i)\n", \
                    __FILE__, __LINE__);              \
            throw cuphy::cuphy_exception(s);          \
        }                                             \
    } while(0)

#define CUDA_CHECK_EXCEPTION_PRINTF_VERSION(c) do {   \
        cudaError_t s = c;                            \
        if((cudaError_t)s != cudaSuccess)             \
        {                                             \
            fprintf(stderr,                           \
                    "CUDA Runtime Error: %s:%i:%s\n", \
                    __FILE__,                         \
                    __LINE__,                         \
                    cudaGetErrorString(s));           \
            throw cuphy::cuda_exception(s);           \
        }                                             \
    } while(0)

#define CU_CHECK_EXCEPTION_PRINTF_VERSION(c) do {   \
        CUresult s = c;                             \
        if((CUresult)s != CUDA_SUCCESS)             \
        {                                           \
            const char* pErrStr;                    \
            cuGetErrorString(s,&pErrStr);           \
            fprintf(stderr,                         \
                    "CUDA Driver Error: %s:%i:%s\n",\
                    __FILE__,                       \
                    __LINE__,                       \
                    pErrStr);                       \
            throw cuphy::cuda_driver_exception(s);  \
        }                                           \
    } while(0)

} // namespace cuphy

#endif //CUPHY_ERR_HPP
