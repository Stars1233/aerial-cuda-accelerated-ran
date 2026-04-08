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

#if !defined(CUPHY_HPP_INCLUDED_)
#define CUPHY_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "memfoot_global.hpp"
#include "nvlog.hpp"
#include "util.hpp"
#include "type_convert.hpp"
#include <string>
#include <array>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <vector>
#include <stdexcept>
#include <typeinfo>
#include "tensor_desc.hpp"

#include "fmt/format.h"

#include "cuphy_err.hpp"

#include "tensor_desc.hpp" // For tensor_layout (typdefed to tensor_layout_t)

#ifdef __CUDACC__
#define CUPHY_BOTH __host__ __device__
#define CUPHY_BOTH_INLINE __forceinline__ __host__ __device__
#define CUPHY_INLINE __forceinline__ __device__
#else
#define CUPHY_BOTH
#define CUPHY_INLINE
#ifdef WINDOWS
#define CUPHY_BOTH_INLINE __inline
#else
#define CUPHY_BOTH_INLINE __inline__
#endif
#endif

// The base + offset values should match the tag numbers in nvlog_config.yaml and nvlog_fmt.hpp
#define NVLOG_SSB     (NVLOG_TAG_BASE_CUPHY + 1)  // "CUPHY.SSB_TX"
#define NVLOG_PDCCH   (NVLOG_TAG_BASE_CUPHY + 2)  // "CUPHY.PDCCH_TX"
#define NVLOG_PDSCH   (NVLOG_TAG_BASE_CUPHY + 4)  // "CUPHY.PDSCH_TX"
#define NVLOG_CSIRS   (NVLOG_TAG_BASE_CUPHY + 5)  // "CUPHY.CSIRS_TX"
#define NVLOG_PRACH   (NVLOG_TAG_BASE_CUPHY + 6)  // "CUPHY.PRACH_RX"
#define NVLOG_PUCCH   (NVLOG_TAG_BASE_CUPHY + 7)  // "CUPHY.PUCCH_RX"
#define NVLOG_PUSCH   (NVLOG_TAG_BASE_CUPHY + 8)  // "CUPHY.PUSCH_RX"
#define NVLOG_BFW     (NVLOG_TAG_BASE_CUPHY + 9)  // "CUPHY.BFW"
#define NVLOG_SRS     (NVLOG_TAG_BASE_CUPHY + 10) // "CUPHY.SRS_RX"
#define NVLOG_SRSTX   (NVLOG_TAG_BASE_CUPHY + 11) // "CUPHY.SRS_TX"
#define NVLOG_CSIRSRX (NVLOG_TAG_BASE_CUPHY + 12)  // "CUPHY.CSIRS_RX"

#define NVLOG_MEMFOOT   (NVLOG_TAG_BASE_CUPHY + 32) // "CUPHY.MEMFOOT"
#define NVLOG_PYAERIAL  (NVLOG_TAG_BASE_CUPHY + 35) // "CUPHY.PYAERIAL"

inline uint32_t format_as(const CUresult e) { return fmt::underlying(e); }
inline uint32_t format_as(const CUmoduleLoadingMode_enum m) { return fmt::underlying(m); }

////////////////////////////////////////////////////////////////////////////
// cuphyMemoryFootprint
class cuphyMemoryFootprint : public MemFootBase
{
public:
    cuphyMemoryFootprint():MemFootBase(MF_MODULE_CUPHY), gpu_allocated_bytes(0)
    {}

    void addGpuAllocation(size_t bytes)
    {
        //NVLOGC_FMT(NVLOG_MEMFOOT, "allocating {:d} bytes.", bytes);
        gpu_allocated_bytes += bytes;
    }
    size_t getGpuRegularSize() const
    {
        return gpu_allocated_bytes;
    }
    void printMemoryFootprint(void* channel_ptr=nullptr, std::string channel_name="") const
    {
        name = channel_name;
        if (!channel_ptr)
        {
            //printf("cuphyMemoryFootprint - GPU allocation: %.3f MiB.\n", gpu_allocated_bytes * 1.0f/ (1024 * 1024));
            NVLOGC_FMT(NVLOG_MEMFOOT, "cuphyMemoryFootprint - GPU allocation: {:.3f} MiB.", gpu_allocated_bytes * 1.0f/ (1024 * 1024));
        }
        else
        {
            //printf("cuphyMemoryFootprint - GPU allocation: %.3f MiB for cuPHY %s channel object (%p).\n", gpu_allocated_bytes * 1.0f/ (1024 * 1024), channel_name.c_str(), channel_ptr);
            NVLOGC_FMT(NVLOG_MEMFOOT, "cuphyMemoryFootprint - GPU allocation: {:.3f} MiB for cuPHY {} channel object ({:p}).", gpu_allocated_bytes * 1.0f/ (1024 * 1024), channel_name, channel_ptr);
        }
    }

    void setName(const std::string& name)
    {
        this->name = name;
    }

    std::string getName() override
    {
        return name;
    }

    size_t getGpuSize() override
    {
        return gpu_allocated_bytes;
    }
private:
    mutable std::string name = "";
    size_t gpu_allocated_bytes;
};

namespace cuphy
{

template <typename callable>
inline
#if (__cplusplus < 201703L) // pre C++17
typename std::enable_if<std::is_void<std::result_of_t<callable()>>::value, cuphyStatus_t>::type
#else
typename std::enable_if_t<std::is_void_v<std::invoke_result_t<callable>>, cuphyStatus_t>
#endif
invokeCallable(callable&& fn)
{
   fn();
   return CUPHY_STATUS_SUCCESS;
}

template <typename callable>
inline
#if (__cplusplus < 201703L) // pre C++17
typename std::enable_if<std::is_same<std::result_of_t<callable()>, cuphyStatus_t>::value, cuphyStatus_t>::type
#else
typename std::enable_if_t<std::is_same_v<std::invoke_result_t<callable>, cuphyStatus_t>, cuphyStatus_t>
#endif
invokeCallable(callable&& fn)
{
   return fn();
}

// Helper to be used in exception boundary functions. Consumes any exceptions thrown by the callable and maps them to cuPHY status codes.
// It is recommended that callables return types be limited to cuphyStatus_t and void types (only these types are supported).
// The user can override the status returned in all catch clauses by specifying a default_error_status (e.g., CUPHY_STATUS_INVALID_ARGUMENT
// to highlight a potential misconfiguration).
template <typename callable>
inline cuphyStatus_t tryCallableAndCatch(callable&& fn, cuphyStatus_t default_error_status=CUPHY_N_STATUS_CONFIGS)
{
    cuphyStatus_t ret_status;
    try
    {
        // printf("fn return type %s\n", typeid(decltype(fn())).name());
        return invokeCallable(fn);
    }
    catch(cuphy::cuda_exception const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "CUDA EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(cuphy::cuda_driver_exception const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "CUDA DRIVER EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(cuphy::cuphy_exception const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "CUPHY EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(cuphy::cuphy_fn_exception const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "CUPHY FUNC EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(std::bad_alloc const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(std::out_of_range const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_VALUE_OUT_OF_RANGE;
    }
    catch(std::invalid_argument const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INVALID_ARGUMENT;
    }
    catch (std::exception const& ex)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", ex.what());
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch (...)
    {
        ret_status = CUPHY_STATUS_INTERNAL_ERROR;
    }
    return ((default_error_status == CUPHY_N_STATUS_CONFIGS) ? ret_status : default_error_status);
}

struct context_deleter
{
    typedef cuphyContext_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyContext(p);
    }

};

////////////////////////////////////////////////////////////////////////////
// unique_ctx_ptr
using unique_ctx_ptr = std::unique_ptr<cuphyContext, context_deleter>;

////////////////////////////////////////////////////////////////////////////
// context
class context
{
public:
    //----------------------------------------------------------------------
    // context()
    context(unsigned int flags = 0)
    {
        cuphyContext_t p = nullptr;
        cuphyStatus_t  s = cuphyCreateContext(&p, flags);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_fn_exception(s, "cuphyCreateContext()");
        }
        ctx_.reset(p);
    }
    //----------------------------------------------------------------------
    // handle()
    cuphyContext_t handle() { return ctx_.get(); }
    //----------------------------------------------------------------------
    // demodulate_symbol()
    template <class TSym, class TLLR>
    void demodulate_symbol(TLLR& tLLR, TSym& tSym, int log2_QAM, float noiseVar = 1.0f, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyDemodulateSymbol(handle(), // context
                                               tLLR.desc().handle(),
                                               tLLR.addr(),
                                               tSym.desc().handle(),
                                               tSym.addr(),
                                               log2_QAM,
                                               noiseVar,
                                               strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_fn_exception(s, "cuphyDemodulateSymbol()");
        }
    }

    static uint32_t getScsKHz(uint8_t mu)
    {
        // Table mapping mu to subcarrier spacing in KHz
        //                              mu =    0, 1,   2,   3,   4
        constexpr uint32_t MU_TO_SCS_KHZ[] = { 15, 30, 60, 120, 240 };
        constexpr size_t len = std::extent<decltype(MU_TO_SCS_KHZ)>::value;
        if(mu >= len)
        {
            throw cuphy::cuphy_fn_exception(CUPHY_STATUS_INVALID_ARGUMENT, "getScsKHz()");
        }
        return MU_TO_SCS_KHZ[mu];
    }

private:
    unique_ctx_ptr ctx_;
};

////////////////////////////////////////////////////////////////////////////
// cudaContext
class cudaContext
{
public:
    cudaContext() = default;
    cudaContext(cudaContext const&) = delete;
    cudaContext& operator=(cudaContext const&) = delete;

    //----------------------------------------------------------------------
    // context creation
    void create(int gpuId)
    {
        CUdevice device;
        CUresult result = cuDeviceGet(&device, gpuId);
        if(CUDA_SUCCESS != result)
        {
            throw cuphy::cuda_driver_exception(result, "cuDeviceGet()");
        }
        // printf("GPU ordinal %d gpuId %d\n", static_cast<int>(device), gpuId);

#if CUDA_VERSION >= 13000
        CUctxCreateParams params{};
        result = cuCtxCreate(&m_cuCtx, &params, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device);
#else
        result = cuCtxCreate(&m_cuCtx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device);
#endif
        if(CUDA_SUCCESS != result)
        {
            throw cuphy::cuda_driver_exception(result, "cuCtxCreate()");
        }
        m_ctxCreated = true;
    }

    void create(int gpuId, int smCount, int* pAppliedSmCount)
    {
#if CUDART_VERSION >= 11040 // min CUDA version for MPS programmatic API
        CUdevice device;
        CUresult result = cuDeviceGet(&device, gpuId);
        if(CUDA_SUCCESS != result) throw cuphy::cuda_driver_exception(result, "cuDeviceGet()");
        // printf("GPU ordinal %d gpuId %d\n", static_cast<int>(device), gpuId);

        CUexecAffinityParam affinityPrm;
        affinityPrm.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinityPrm.param.smCount.val = smCount;
#if CUDA_VERSION >= 13000
        CUctxCreateParams params{};
        params.execAffinityParams = &affinityPrm;
        params.numExecAffinityParams = 1;
        params.cigParams = nullptr;
        result = cuCtxCreate(&m_cuCtx, &params, 0, device);
#else
        result = cuCtxCreate_v3(&m_cuCtx, &affinityPrm, 1, 0, device);
#endif
        if(CUDA_SUCCESS != result) throw cuphy::cuda_driver_exception(result, "cuCtxCreate()");

        if(nullptr != pAppliedSmCount)
        {
            CUexecAffinityParam appliedAffinityPrm;
            result = cuCtxGetExecAffinity(&appliedAffinityPrm, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
            if(CUDA_SUCCESS != result) throw cuphy::cuda_driver_exception(result, "cuCtxGetExecAffinity()");
            *pAppliedSmCount = appliedAffinityPrm.param.smCount.val;
        }

        m_ctxCreated = true;
#endif
    }

    ~cudaContext()
    {
        if(m_ctxCreated)
        {
            CUresult result = cuCtxDestroy(m_cuCtx);
            if(CUDA_SUCCESS != result)
            {
                const char* pResNameStr;
                CUresult e1 = cuGetErrorName(result, &pResNameStr);
                if (e1 != CUDA_SUCCESS) {
                    printf("cuCtxDestroy() error %d; cuGetErrorName() failed with %d\n", result, e1);
                }
                const char* pResDescriptionStr;
                CUresult e2 = cuGetErrorString(result, &pResDescriptionStr);
                if (e2 != CUDA_SUCCESS) {
                    printf("cuCtxDestroy() error %d; cuGetErrorString() failed with %d\n", result, e2);
                }

                if ((e1 == CUDA_SUCCESS) && (e2 == CUDA_SUCCESS)) {
                    printf("cuCtxDestroy() error %s: %s\n", pResNameStr, pResDescriptionStr);
                }
            }
        }
    }

    //----------------------------------------------------------------------
    // handle()
    CUcontext handle() { return m_cuCtx; }

    //----------------------------------------------------------------------
    // setCurrent()
    void bind()
    {
        if(! m_ctxCreated) return;
        CUresult result = cuCtxSetCurrent(m_cuCtx);
        if(CUDA_SUCCESS != result)
        {
            throw cuphy::cuda_driver_exception(result, "cuCtxSetCurrent()");
        }
    }

private:
    bool      m_ctxCreated = false;
    CUcontext m_cuCtx;
};

////////////////////////////////////////////////////////////////////////////
// cudaGreenContext

class cudaGreenContext
{
public:
    cudaGreenContext() = default;
    //Delete copy ctor and copy assignment operator
    cudaGreenContext(const cudaGreenContext&) = delete;
    cudaGreenContext& operator=(const cudaGreenContext&) = delete;

#if CUDA_VERSION >= 12040
    void create(int gpuId, const CUdevResource* resources, bool print_resources=false, bool use_workqueues=false, unsigned int wq_concurrency_limit=2)
    {
        if (resources == nullptr) {
            printf("Cannot call cudaGreenContext create with null CUdevResource ptr!\n");
            return;
        }
        m_SMcount = resources->sm.smCount;

        //FIXME exception w/ tag? handle errors in a consistent way for cudaGreenContext
        CUdevice device;
        CU_CHECK_EXCEPTION(cuDeviceGet(&device, gpuId));
        // printf("GPU ordinal %d gpuId %d\n", static_cast<int>(device), gpuId);

        // Generate a descriptor from the provided resource
#if CUDA_VERSION >= 13010
        if (!use_workqueues) {
            const unsigned int num_resources = 1;
            m_resources[0]      = resources[0];
            CU_CHECK_EXCEPTION(cuDevResourceGenerateDesc(&m_cuResourceDesc, &m_resources[0], num_resources));
        } else {
            const unsigned int num_resources = 2;
            m_resources[0]      = resources[0];
            m_resources[1].type = CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG;
            m_resources[1].wqConfig = {.device = device, .wqConcurrencyLimit = wq_concurrency_limit, .sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED};

            CU_CHECK_EXCEPTION(cuDevResourceGenerateDesc(&m_cuResourceDesc, &m_resources[0], num_resources));
        }
#else
            const unsigned int num_resources = 1;
            m_resources[0]      = resources[0];
            CU_CHECK_EXCEPTION(cuDevResourceGenerateDesc(&m_cuResourceDesc, &m_resources[0], num_resources));
#endif

        unsigned int default_ctx_creation_flags = CU_GREEN_CTX_DEFAULT_STREAM;
        // Create a green context for this resource descriptor
        CU_CHECK_EXCEPTION(cuGreenCtxCreate(&m_cuGreenCtx, m_cuResourceDesc, device, default_ctx_creation_flags));

        // Get primary context from green context
        CU_CHECK_EXCEPTION(cuCtxFromGreenCtx(&m_cuCtx, m_cuGreenCtx));

        m_ctxCreated = true;

        if (print_resources)
        {
            printResources();
        }
    }
#endif

    void destroy()
    {
#if CUDA_VERSION >= 12040
        //printf("green ctx will be destroyed if %d && %d %p\n", m_ctxCreated, !m_ctxDeleted, this);
        if(m_ctxCreated && !m_ctxDeleted)
        {
            CUresult result = cuGreenCtxDestroy(m_cuGreenCtx);
            if(CUDA_SUCCESS != result)
            {
                const char* pResNameStr;
                CUresult e1 = cuGetErrorName(result, &pResNameStr);
                if (e1 != CUDA_SUCCESS) {
                    printf("cuCtxDestroy() error %d; cuGetErrorName() failed with %d\n", result, e1);
                }
                const char* pResDescriptionStr;
                CUresult e2 = cuGetErrorString(result, &pResDescriptionStr);
                if (e2 != CUDA_SUCCESS) {
                    printf("cuCtxDestroy() error %d; cuGetErrorString() failed with %d\n", result, e2);
                }

                if ((e1 == CUDA_SUCCESS) && (e2 == CUDA_SUCCESS)) {
                    printf("cuCtxDestroy() error %s: %s\n", pResNameStr, pResDescriptionStr);
                }
                m_ctxDeleted = false;
            } else {
                m_ctxDeleted = true;
            }
        }
#endif
    }

    ~cudaGreenContext()
    {
        destroy();
    }

#if CUDA_VERSION >= 12040
    //----------------------------------------------------------------------
    // primary_handle()
    CUcontext primary_handle() const { return m_cuCtx; }

    //----------------------------------------------------------------------
    // handle()
    CUgreenCtx handle() const { return m_cuGreenCtx; }

    //----------------------------------------------------------------------
    // setCurrent()
    void bind() const
    {
        if(! m_ctxCreated) {
            printf("Cannot call cudaGreenContext bind() before a context has been created\n");
            return;
        }
        CUresult result = cuCtxSetCurrent(m_cuCtx);
        if(CUDA_SUCCESS != result)
        {
            throw cuphy::cuda_driver_exception(result, "cuCtxSetCurrent()");
        }
        //printf("binding %p\n", m_cuCtx);
    }
    //----------------------------------------------------------------------
    //getResources()
    void getResources(CUdevResource* resource, CUdevResourceType resource_type=CU_DEV_RESOURCE_TYPE_SM) const
    {
        if (!m_ctxCreated) {
            printf("Cannot call cudaGreenContext getResources() before a context has been created\n");
            return;
        }
        CUresult result = cuGreenCtxGetDevResource(m_cuGreenCtx, resource,  resource_type);
        if(CUDA_SUCCESS != result)
        {
            throw cuphy::cuda_driver_exception(result, "cuGreenCtxGetDevResource()");
        }
    }

    //----------------------------------------------------------------------
    //printDevResourceInfo()

    void printDevResourceInfo(CUdevResource dev_resource)
    {

        if (dev_resource.type == CU_DEV_RESOURCE_TYPE_INVALID) {
            NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "device resource has invalid type ({})", CU_DEV_RESOURCE_TYPE_INVALID);
        } else if (dev_resource.type == CU_DEV_RESOURCE_TYPE_SM) {
#if CUDA_VERSION >= 13010
            NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "device resource with CU_DEV_RESOURCE_TYPE_SM ({}) type has {} SM count, {} minSmPartitionSize and {} smCoscheduledAlignment and {} flags",
                   CU_DEV_RESOURCE_TYPE_SM, dev_resource.sm.smCount, dev_resource.sm.minSmPartitionSize, dev_resource.sm.smCoscheduledAlignment, dev_resource.sm.flags);
#else
            NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "device resource with CU_DEV_RESOURCE_TYPE_SM ({}) type has {} SM count",
                   CU_DEV_RESOURCE_TYPE_SM, dev_resource.sm.smCount);
#endif
        }
#if CUDA_VERSION >= 13010
        else if (dev_resource.type == CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG)
        {
            NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "device resource with CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG ({}) type has {} wqConcurrencyLimit and {} sharingScope",
                   CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG, dev_resource.wqConfig.wqConcurrencyLimit, dev_resource.wqConfig.sharingScope);
        }
        else if (dev_resource.type == CU_DEV_RESOURCE_TYPE_WORKQUEUE)
        {
            NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "device resource with CU_DEV_RESOURCE_TYPE_WORKQUEUE ({}) type",
                  CU_DEV_RESOURCE_TYPE_WORKQUEUE);
        }
#endif
        else
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "device resource with unexpected {} type", dev_resource.type);
        }
    }


    //----------------------------------------------------------------------
    //printResources()
    void printResources()
    {
        if (!m_ctxCreated) {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "Cannot call cudaGreenContext printResources() before a context has been created");
            return;
        }

        CUdevResource sm_resource = {};
        CU_CHECK(cuGreenCtxGetDevResource(m_cuGreenCtx, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
        printDevResourceInfo(sm_resource);

#if CUDA_VERSION >= 13010
        CUdevResource wq_config_resource = {};
        CU_CHECK(cuGreenCtxGetDevResource(m_cuGreenCtx, &wq_config_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
        printDevResourceInfo(wq_config_resource);

        CUdevResource wq_resource = {};
        CU_CHECK(cuGreenCtxGetDevResource(m_cuGreenCtx, &wq_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE));
        printDevResourceInfo(wq_resource);
#endif
    }

#endif


    //----------------------------------------------------------------------
    //getSmCount()
    int getSmCount() const
    {
        return m_SMcount;
    }

private:
    bool         m_ctxCreated = false;
    bool         m_ctxDeleted = false;
    CUcontext    m_cuCtx; //primary context from green context m_cuGreenCtx
#if CUDA_VERSION >= 12040
    CUgreenCtx   m_cuGreenCtx;
    CUdevResourceDesc m_cuResourceDesc; //not needed, could remove
    CUdevResource  m_resources[2] = {{}, {}}; // device resources
#endif
    int          m_SMcount = 0;
};

// clang-format off
////////////////////////////////////////////////////////////////////////////
// cuphy::type_traits
template <cuphyDataType_t Ttype> struct type_traits;
template <> struct type_traits<CUPHY_VOID>  { typedef void            type; };
template <> struct type_traits<CUPHY_BIT>   { typedef uint32_t        type; };
template <> struct type_traits<CUPHY_R_8I>  { typedef signed char     type; };
template <> struct type_traits<CUPHY_C_8I>  { typedef char2           type; };
template <> struct type_traits<CUPHY_R_8U>  { typedef unsigned char   type; };
template <> struct type_traits<CUPHY_C_8U>  { typedef uchar2          type; };
template <> struct type_traits<CUPHY_R_16I> { typedef short           type; };
template <> struct type_traits<CUPHY_C_16I> { typedef short2          type; };
template <> struct type_traits<CUPHY_R_16U> { typedef unsigned short  type; };
template <> struct type_traits<CUPHY_C_16U> { typedef ushort2         type; };
template <> struct type_traits<CUPHY_R_32I> { typedef int             type; };
template <> struct type_traits<CUPHY_C_32I> { typedef int2            type; };
template <> struct type_traits<CUPHY_R_32U> { typedef unsigned int    type; };
template <> struct type_traits<CUPHY_C_32U> { typedef uint2           type; };
template <> struct type_traits<CUPHY_R_16F> { typedef __half          type; };
template <> struct type_traits<CUPHY_C_16F> { typedef __half2         type; };
template <> struct type_traits<CUPHY_R_32F> { typedef float           type; };
template <> struct type_traits<CUPHY_C_32F> { typedef cuComplex       type; };
template <> struct type_traits<CUPHY_R_64F> { typedef double          type; };
template <> struct type_traits<CUPHY_C_64F> { typedef cuDoubleComplex type; };
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////////
// cuphy::type_traits
template <typename T> struct type_to_cuphy_type;
template <> struct type_to_cuphy_type<signed char>     { static constexpr cuphyDataType_t value = CUPHY_R_8I;  };
template <> struct type_to_cuphy_type<char2>           { static constexpr cuphyDataType_t value = CUPHY_C_8I;  };
template <> struct type_to_cuphy_type<unsigned char>   { static constexpr cuphyDataType_t value = CUPHY_R_8U;  };
template <> struct type_to_cuphy_type<uchar2>          { static constexpr cuphyDataType_t value = CUPHY_C_8U;  };
template <> struct type_to_cuphy_type<short>           { static constexpr cuphyDataType_t value = CUPHY_R_16I; };
template <> struct type_to_cuphy_type<short2>          { static constexpr cuphyDataType_t value = CUPHY_C_16I; };
template <> struct type_to_cuphy_type<unsigned short>  { static constexpr cuphyDataType_t value = CUPHY_R_16U; };
template <> struct type_to_cuphy_type<ushort2>         { static constexpr cuphyDataType_t value = CUPHY_C_16U; };
template <> struct type_to_cuphy_type<int>             { static constexpr cuphyDataType_t value = CUPHY_R_32I; };
template <> struct type_to_cuphy_type<int2>            { static constexpr cuphyDataType_t value = CUPHY_C_32I; };
template <> struct type_to_cuphy_type<unsigned int>    { static constexpr cuphyDataType_t value = CUPHY_R_32U; };
template <> struct type_to_cuphy_type<uint2>           { static constexpr cuphyDataType_t value = CUPHY_C_32U; };
template <> struct type_to_cuphy_type<__half>          { static constexpr cuphyDataType_t value = CUPHY_R_16F; };
template <> struct type_to_cuphy_type<__half2>         { static constexpr cuphyDataType_t value = CUPHY_C_16F; };
template <> struct type_to_cuphy_type<float>           { static constexpr cuphyDataType_t value = CUPHY_R_32F; };
template <> struct type_to_cuphy_type<cuComplex>       { static constexpr cuphyDataType_t value = CUPHY_C_32F; };
template <> struct type_to_cuphy_type<double>          { static constexpr cuphyDataType_t value = CUPHY_R_64F; };
template <> struct type_to_cuphy_type<cuDoubleComplex> { static constexpr cuphyDataType_t value = CUPHY_C_64F; };
// clang-format on

////////////////////////////////////////////////////////////////////////
// get_element_size()
// Returns the size, in bytes, of the underlying data type associated
// with a given cuPHY data type.
// Note that the size for CUPHY_BIT is the size of the word
// representation.
inline
int get_element_size(cuphyDataType_t t)
{
    int sz = 0;
    switch(t)
    {
    default:                                                        break;
    case CUPHY_VOID:                                                break;
    case CUPHY_BIT:    sz = sizeof(type_traits<CUPHY_BIT>::type);   break;
    case CUPHY_R_8I:   sz = sizeof(type_traits<CUPHY_R_8I>::type);  break;
    case CUPHY_C_8I:   sz = sizeof(type_traits<CUPHY_C_8I>::type);  break;
    case CUPHY_R_8U:   sz = sizeof(type_traits<CUPHY_R_8U>::type);  break;
    case CUPHY_C_8U:   sz = sizeof(type_traits<CUPHY_C_8U>::type);  break;
    case CUPHY_R_16I:  sz = sizeof(type_traits<CUPHY_R_16I>::type); break;
    case CUPHY_C_16I:  sz = sizeof(type_traits<CUPHY_C_16I>::type); break;
    case CUPHY_R_16U:  sz = sizeof(type_traits<CUPHY_R_16U>::type); break;
    case CUPHY_C_16U:  sz = sizeof(type_traits<CUPHY_C_16U>::type); break;
    case CUPHY_R_32I:  sz = sizeof(type_traits<CUPHY_R_32I>::type); break;
    case CUPHY_C_32I:  sz = sizeof(type_traits<CUPHY_C_32I>::type); break;
    case CUPHY_R_32U:  sz = sizeof(type_traits<CUPHY_R_32U>::type); break;
    case CUPHY_C_32U:  sz = sizeof(type_traits<CUPHY_C_32U>::type); break;
    case CUPHY_R_16F:  sz = sizeof(type_traits<CUPHY_R_16F>::type); break;
    case CUPHY_C_16F:  sz = sizeof(type_traits<CUPHY_C_16F>::type); break;
    case CUPHY_R_32F:  sz = sizeof(type_traits<CUPHY_R_32F>::type); break;
    case CUPHY_C_32F:  sz = sizeof(type_traits<CUPHY_C_32F>::type); break;
    case CUPHY_R_64F:  sz = sizeof(type_traits<CUPHY_R_64F>::type); break;
    case CUPHY_C_64F:  sz = sizeof(type_traits<CUPHY_C_64F>::type); break;
    }
    return sz;
}

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::variant
class variant : public cuphyVariant_t
{
public:
    variant()
    {
        type = CUPHY_VOID;
    }
    template <typename T>
    variant(T t)
    {
        type = type_to_cuphy_type<T>::value;
        set(t);
    }
    void set(const signed char&     sc)  { type = CUPHY_R_8I;  value.r8i  = sc;  }
    void set(const char2&           c2)  { type = CUPHY_C_8I;  value.c8i  = c2;  }
    void set(const unsigned char&   uc)  { type = CUPHY_R_8U;  value.r8u  = uc;  }
    void set(const uchar2&          uc2) { type = CUPHY_C_8U;  value.c8u  = uc2; }
    void set(const short&           s)   { type = CUPHY_R_16I; value.r16i = s;   }
    void set(const short2&          s2)  { type = CUPHY_C_16I; value.c16i = s2;  }
    void set(const unsigned short&  us)  { type = CUPHY_R_16U; value.r16u = us;  }
    void set(const ushort2&         us2) { type = CUPHY_C_16U; value.c16u = us2; }
    void set(const int&             i)   { type = CUPHY_R_32I; value.r32i = i;   }
    void set(const int2&            i2)  { type = CUPHY_C_32I; value.c32i = i2;  }
    void set(const unsigned int&    u)   { type = CUPHY_R_32U; value.r32u = u;   }
    void set(const uint2&           u2)  { type = CUPHY_C_32U; value.c32u = u2;  }
    void set(const __half&          h)   { type = CUPHY_R_16F; memcpy(&value.r16f, &h, sizeof(__half));   }
    void set(const __half2&         h2)  { type = CUPHY_C_16F; memcpy(&value.c16f, &h2, sizeof(__half2)); }
    void set(const float&           f)   { type = CUPHY_R_32F; value.r32f = f;   }
    void set(const cuComplex&       c)   { type = CUPHY_C_32F; value.c32f = c;   }
    void set(const double&          d)   { type = CUPHY_R_64F; value.r64f = d;   }
    void set(const cuDoubleComplex& dc)  { type = CUPHY_C_64F; value.c64f = dc;  }
    template <typename T> T& as();
};
// clang-format on

// clang-format off
template <> inline signed char&     variant::as<signed char>()     { if(type != CUPHY_R_8I)  throw std::runtime_error("variant type mismatch"); return value.r8i;  }
template <> inline char2&           variant::as<char2>()           { if(type != CUPHY_C_8I)  throw std::runtime_error("variant type mismatch"); return value.c8i;  }
template <> inline unsigned char&   variant::as<unsigned char>()   { if(type != CUPHY_R_8U)  throw std::runtime_error("variant type mismatch"); return value.r8u;  }
template <> inline uchar2&          variant::as<uchar2>()          { if(type != CUPHY_C_8U)  throw std::runtime_error("variant type mismatch"); return value.c8u;  }
template <> inline short&           variant::as<short>()           { if(type != CUPHY_R_16I) throw std::runtime_error("variant type mismatch"); return value.r16i; }
template <> inline short2&          variant::as<short2>()          { if(type != CUPHY_C_16I) throw std::runtime_error("variant type mismatch"); return value.c16i; }
template <> inline unsigned short&  variant::as<unsigned short>()  { if(type != CUPHY_R_16U) throw std::runtime_error("variant type mismatch"); return value.r16u; }
template <> inline ushort2&         variant::as<ushort2>()         { if(type != CUPHY_C_16U) throw std::runtime_error("variant type mismatch"); return value.c16u; }
template <> inline int&             variant::as<int>()             { if(type != CUPHY_R_32I) throw std::runtime_error("variant type mismatch"); return value.r32i; }
template <> inline int2&            variant::as<int2>()            { if(type != CUPHY_C_32I) throw std::runtime_error("variant type mismatch"); return value.c32i; }
template <> inline unsigned int&    variant::as<unsigned int>()    { if(type != CUPHY_R_32U) throw std::runtime_error("variant type mismatch"); return value.r32u; }
template <> inline uint2&           variant::as<uint2>()           { if(type != CUPHY_C_32U) throw std::runtime_error("variant type mismatch"); return value.c32u; }
template <> inline __half_raw&      variant::as<__half_raw>()      { if(type != CUPHY_R_16F) throw std::runtime_error("variant type mismatch"); return value.r16f; }
template <> inline __half2_raw&     variant::as<__half2_raw>()     { if(type != CUPHY_C_16F) throw std::runtime_error("variant type mismatch"); return value.c16f; }
template <> inline float&           variant::as<float>()           { if(type != CUPHY_R_32F) throw std::runtime_error("variant type mismatch"); return value.r32f; }
template <> inline cuComplex&       variant::as<cuComplex>()       { if(type != CUPHY_C_32F) throw std::runtime_error("variant type mismatch"); return value.c32f; }
template <> inline double&          variant::as<double>()          { if(type != CUPHY_R_64F) throw std::runtime_error("variant type mismatch"); return value.r64f; }
template <> inline cuDoubleComplex& variant::as<cuDoubleComplex>() { if(type != CUPHY_C_64F) throw std::runtime_error("variant type mismatch"); return value.c64f; }
// clang-format on

enum class tensor_flags
{
    align_default  = CUPHY_TENSOR_ALIGN_DEFAULT,  /* Use strides if provided, otherwise TIGHT                */
    align_tight    = CUPHY_TENSOR_ALIGN_TIGHT,    /* Pack tightly, regardless of stride values               */
    align_coalesce = CUPHY_TENSOR_ALIGN_COALESCE  /* Align 2nd dimension for coalesced I/O (ignore strides)  */
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::tensor_info
// Representation of the type, dimensions, and strides that are stored
// internally by a tensor descriptor. (This class can be used to cache
// the values stored by the cuPHY library for a tensor descriptor, or
// to store values to assign to a cuPHY library tensor descriptor.)
class tensor_info
{
public:
    typedef tensor_layout layout_t;
    tensor_info() : data_type_(CUPHY_VOID)
    {
    }
    tensor_info(cuphyDataType_t type, const layout_t& layout_in) :
        data_type_(type),
        layout_(layout_in)
    {
    }
    int             rank()   const { return layout_.rank(); }
    cuphyDataType_t type()   const { return data_type_;   }
    const layout_t& layout() const { return layout_;      }
    std::string to_string(bool withStride = true) const
    {
        std::string s("type: ");
        s.append(cuphyGetDataTypeString(data_type_));
        s.append(", dim: (");
        for(int i = 0; i < rank(); ++i)
        {
            if(i > 0) s.append(",");
            s.append(std::to_string(layout_.dimensions()[i]));
        }
        s.append(")");
        if(withStride)
        {
            s.append(", stride: (");
            for(int i = 0; i < rank(); ++i)
            {
                if(i > 0) s.append(",");
                s.append(std::to_string(layout_.strides()[i]));
            }
            s.append(")");
        }
        return s;
    }
private:
    cuphyDataType_t data_type_;
    layout_t        layout_;
};
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::tensor_desc
// Wrapper class for cuPHY tensor descriptor objects. Capable of
// representing a tensor of any underlying type, and any rank (up to the
// maximum supported by the library).
class tensor_desc
{
public:
    typedef tensor_info tensor_info_t;
    tensor_desc()
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, const tensor_layout& layout_in, tensor_flags flags = tensor_flags::align_default)
    {
        tensor_info_t tinfo(type, layout_in);
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        s = cuphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    template <int N>
    tensor_desc(cuphyDataType_t type, int (&dims)[N], tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     N,
                                     dims,
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, int dim0, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     1,
                                     &dim0,
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, int dim0, int dim1, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        std::array<int, 2> dims = {dim0, dim1};
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     dims.size(),
                                     dims.data(),
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, int dim0, int dim1, int dim2, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        std::array<int, 3> dims = {dim0, dim1, dim2};
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     dims.size(),
                                     dims.data(),
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        std::array<int, 4> dims = {dim0, dim1, dim2, dim3};
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     dims.size(),
                                     dims.data(),
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, int dim4, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        std::array<int, 5> dims = {dim0, dim1, dim2, dim3, dim4};
        s = cuphySetTensorDescriptor(desc_,
                                     type,
                                     dims.size(),
                                     dims.data(),
                                     nullptr,
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(const tensor_info& tinfo, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        s = cuphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(const tensor_desc& td)
    {
        cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
        }
        tensor_info_t tinfo = td.get_info();
        s = cuphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     CUPHY_TENSOR_ALIGN_DEFAULT);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    tensor_desc(tensor_desc&& td) : desc_(td.desc_) { td.desc_ = nullptr; }
    ~tensor_desc() { if(desc_) cuphyDestroyTensorDescriptor(desc_); }
    tensor_desc& operator=(const tensor_desc& td)
    {
        if(!desc_)
        {
            cuphyStatus_t s = cuphyCreateTensorDescriptor(&desc_);
            if(CUPHY_STATUS_SUCCESS != s)
            {
                throw cuphy_fn_exception(s, "cuphyCreateTensorDescriptor()");
            }
        }
        tensor_info_t tinfo = td.get_info();
        cuphyStatus_t s     = cuphySetTensorDescriptor(desc_,
                                                       tinfo.type(),
                                                       tinfo.rank(),
                                                       tinfo.layout().dimensions().begin(),
                                                       tinfo.layout().strides().begin(),
                                                       CUPHY_TENSOR_ALIGN_DEFAULT);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
        return *this;
    }
    tensor_desc& operator=(tensor_desc&& td)
    {
        if(desc_) cuphyDestroyTensorDescriptor(desc_);
        desc_    = td.desc_;
        td.desc_ = nullptr;
        return *this;
    }
    //------------------------------------------------------------------
    // Modify the underlying tensor descriptor
    void set(const tensor_info& tinfo, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s;
        s = cuphySetTensorDescriptor(desc_,
                                     tinfo.type(),
                                     tinfo.rank(),
                                     tinfo.layout().dimensions().begin(),
                                     tinfo.layout().strides().begin(),
                                     static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    void set(cuphyDataType_t type, int dim0, tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   1,
                                                   &dim0,
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    void set(cuphyDataType_t type, int dim0, int dim1, tensor_flags flags = tensor_flags::align_default)
    {
        std::array<int, 2> dims = {dim0, dim1};
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   dims.size(),
                                                   dims.data(),
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    void set(cuphyDataType_t type, int dim0, int dim1, int dim2, tensor_flags flags = tensor_flags::align_default)
    {
        std::array<int, 3> dims = {dim0, dim1, dim2};
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   dims.size(),
                                                   dims.data(),
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    void set(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, tensor_flags flags = tensor_flags::align_default)
    {
        std::array<int, 4> dims = {dim0, dim1, dim2, dim3};
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   dims.size(),
                                                   dims.data(),
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    void set(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, int dim4, tensor_flags flags = tensor_flags::align_default)
    {
        std::array<int, 5> dims = {dim0, dim1, dim2, dim3, dim4};
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   dims.size(),
                                                   dims.data(),
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    template <int N>
    void set(cuphyDataType_t type, int (&dims)[N], tensor_flags flags = tensor_flags::align_default)
    {
        cuphyStatus_t s = cuphySetTensorDescriptor(desc_,
                                                   type,
                                                   N,
                                                   dims,
                                                   nullptr,
                                                   static_cast<int>(flags));
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphySetTensorDescriptor()");
        }
    }
    //------------------------------------------------------------------
    // Make a copy, optionally overriding the alignment with flags
    tensor_desc clone(tensor_flags flags = tensor_flags::align_default) const
    {
        return tensor_desc(get_info(), flags);
    }
    //------------------------------------------------------------------
    // Retrieve a copy of the tensor info
    tensor_info_t get_info() const
    {
        cuphyDataType_t         dtype;
        int                     rank;
        vec<int, CUPHY_DIM_MAX> dimensions;
        vec<int, CUPHY_DIM_MAX> strides;
        cuphyStatus_t s = cuphyGetTensorDescriptor(desc_, // descriptor
                                                   CUPHY_DIM_MAX,
                                                   &dtype,
                                                   &rank,
                                                   dimensions.begin(),
                                                   strides.begin());
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyGetTensorDescriptor()");
        }
        return tensor_info_t(dtype, tensor_layout(rank, dimensions.begin(), strides.begin()));
    };
    //------------------------------------------------------------------
    // Get the size (in bytes) required for a tensor with this descriptor
    size_t get_size_in_bytes() const
    {
        size_t        sz = 0;
        cuphyStatus_t s  = cuphyGetTensorSizeInBytes(desc_, &sz);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyGetTensorSizeInBytes()");
        }
        return sz;
    }
    //------------------------------------------------------------------
    // Return the rank of the tensor descriptor
    int rank() const
    {
        tensor_info_t tinfo = get_info();
        return tinfo.rank();
    }
    //------------------------------------------------------------------
    // get_dim()
    int get_dim(int dim) const
    {
        tensor_info_t tinfo = get_info();
        if(dim >= tinfo.rank())
        {
            throw std::runtime_error("Dimension exceeds tensor rank");
        }
        return tinfo.layout().dimensions()[dim];
    }
    //------------------------------------------------------------------
    // get_stride()
    int get_stride(int dim) const
    {
        tensor_info_t tinfo = get_info();
        if(dim >= tinfo.rank())
        {
            throw std::runtime_error("Dimension exceeds tensor rank");
        }
        return tinfo.layout().strides()[dim];
    }
    cuphyTensorDescriptor_t handle() const { return desc_; }
private:
    cuphyTensorDescriptor_t desc_;
};
// clang-format on
////////////////////////////////////////////////////////////////////////
// index_range
// Wrapper class for a pair of values to represent a range of a dimension.
// The start value represents the first value in the range, and the end
// value represents ONE BEYOND the last element of the range (as is the
// case in Python ranges, for example). The number of elements in the
// range is thus given by (end - start), unless the end value has the
// special "end_all()" value.
// An end value of -1 is used to signify the "entire" range of values
// in a dimension.
// A group of index ranges is combined into an index group, with one
// index range for each dimension. Index ranges and index groups combine
// to allow addressing of "slices" or subsets of tensors.
class index_range
{
public:
    static constexpr int dim_invalid() { return -1; }
    static constexpr int dim_end()     { return -1; }
    //------------------------------------------------------------------
    // index_range()
    // Default constructor, used for unspecified dimensions.
    index_range() : start_(dim_invalid()), end_(dim_end())
    {
    }
    //------------------------------------------------------------------
    // index_range()
    // Constructor, used for when both values are given
    index_range(int s, int e) : start_(s), end_(e)
    {
        if(s < 0)
        {
            throw std::runtime_error(std::string("Invalid start index"));
        }
    }
    //------------------------------------------------------------------
    // index_range()
    // Constructor, used for when a single value is given, corresponding
    // to a single index value.
    index_range(int s) : start_(s), end_(s + 1)
    {
        if(s < 0)
        {
            throw std::runtime_error(std::string("Invalid start index"));
        }
    }
    //------------------------------------------------------------------
    // is_valid()
    // Returns true if a start index was provided to the constructor
    bool is_valid() const { return (start_ != dim_invalid()); }
    //------------------------------------------------------------------
    // includes()
    // Returns true if the given index is within the represented range
    bool includes(int idx) const
    {
        return (idx >= start_) && ((end_ == dim_end()) || idx < end_);
    }
    int start() const { return start_; }
    int end()   const { return end_;   }
private:
    int start_;
    int end_;
};

inline index_range dim_invalid() { return index_range(); }
inline index_range dim_all()     { return index_range(0, index_range::dim_end()); }

////////////////////////////////////////////////////////////////////////
// index_group
// A set of index_range instances (one for each dimension). Can be used
// to represent a subset (or "slice") of a tensor.
class index_group
{
public:
    //------------------------------------------------------------------
    // index_group()
    index_group(index_range i0 = dim_invalid(),
                index_range i1 = dim_invalid(),
                index_range i2 = dim_invalid(),
                index_range i3 = dim_invalid(),
                index_range i4 = dim_invalid())
    {
        static_assert(CUPHY_DIM_MAX == 5, "index_group assumes CUPHY_DIM_MAX = 5");
        ranges_[0] = i0;
        ranges_[1] = i1;
        ranges_[2] = i2;
        ranges_[3] = i3;
        ranges_[4] = i4;
    }
    //------------------------------------------------------------------
    // rank()
    int rank() const
    {
        int i = 0;
        for(; i < CUPHY_DIM_MAX; ++i) { if(!ranges_[i].is_valid()) break; }
        return i;
    }
    //------------------------------------------------------------------
    // ranges()
    const std::array<index_range, CUPHY_DIM_MAX>& ranges() const { return ranges_; }
    //------------------------------------------------------------------
    // start_indices()
    std::array<int, CUPHY_DIM_MAX> start_indices() const
    {
        std::array<int, CUPHY_DIM_MAX> s;
        for(size_t i = 0; i < CUPHY_DIM_MAX; ++i)
        {
            s[i] = ranges_[i].is_valid() ? ranges_[i].start() : 0;
        }
        return s;
    }
    //------------------------------------------------------------------
    // includes()
    // Returns true if the index range group contains the given index
    bool includes(int idx0) const
    {
        return ranges_[0].includes(idx0);
    }
    bool includes(int idx0, int idx1) const
    {
        return includes(idx0) && ranges_[1].includes(idx1);
    }
    bool includes(int idx0, int idx1, int idx2) const
    {
        return includes(idx0, idx1) && ranges_[2].includes(idx2);
    }
    bool includes(int idx0, int idx1, int idx2, int idx3) const
    {
        return includes(idx0, idx1, idx2) && ranges_[3].includes(idx3);
    }
    bool includes(int idx0, int idx1, int idx2, int idx3, int idx4) const
    {
        return includes(idx0, idx1, idx2, idx3) && ranges_[4].includes(idx4);
    }
    //------------------------------------------------------------------
    // get_tensor_desc()
    // Generates a tensor descriptor representing the index ranges in
    // this index group, using the given tensor descriptor as the base
    // descriptor.
    tensor_desc get_tensor_desc(const tensor_desc& tdesc) const
    {
        const tensor_info tinfo = tdesc.get_info();
        if(rank() > tinfo.rank())
        {
            throw std::runtime_error("index_group rank exceeds tensor rank");
        }
        // Dimensions need to be updated based on the provided index
        // ranges.
        vec<int, CUPHY_DIM_MAX> dims;
        for(int i = 0; i < tinfo.rank(); ++i)
        {
            // Resolve the actual end value if the caller wants the end of
            // the dimension
            int endIndex = (ranges_[i].end() == index_range::dim_end()) ?
                           tinfo.layout().dimensions()[i]                           :
                           ranges_[i].end();
            dims[i] = endIndex - ranges_[i].start();
        }
        // Create a tensor layout based on the new dimensions. (Strides
        // are unchanged.)
        return tensor_desc(tinfo.type(),
                           tensor_layout(tinfo.rank(),
                                         dims.begin(),
                                         tinfo.layout().strides().begin()));
    }
private:
    std::array<index_range, CUPHY_DIM_MAX> ranges_;
};

////////////////////////////////////////////////////////////////////////
// bit_to_word_layout()
// Generates an "equivalent" layout that describes a layout of type
// CUPHY_BIT as 32-bit words.
inline
tensor_layout bit_to_word_layout(const tensor_layout& layout)
{
    vec<int, CUPHY_DIM_MAX> newDims    = layout.dimensions();
    newDims[0]                         = (newDims[0] + 31) / 32;
    vec<int, CUPHY_DIM_MAX> newStrides = layout.strides();
    for(int i = 1; i < layout.rank(); ++i) // Skip first dim
    {
        newStrides[i] /= 32;
    }
    return tensor_layout(layout.rank(),
                         newDims.begin(),
                         newStrides.begin());
}

////////////////////////////////////////////////////////////////////////
// offset_generator
// Templated structure used to calculate the offset, in elements, for
// a given tuple of indices.
template <cuphyDataType_t TType>
struct offset_generator
{
    template <int N>
    static size_t get(const tensor_layout& layout, const int (&idx)[N])
    {
        return layout.offset(idx);
    }
    template <size_t N>
    static size_t get(const tensor_layout& layout, const std::array<int, N>& idx)
    {
      return layout.offset(idx);
    }
};
// Tensors of type CUPHY_BIT can't use the standard offset calculations
// to address bits, so we specialize the offset calculations for tensors
// of that type
template <>
struct offset_generator<CUPHY_BIT>
{
    template <int N>
    static size_t get(const tensor_layout& layout, const int (&idx)[N])
    {
        // Create a layout for 32-bit words and use that for
        // offset calculation. Inefficient, but only used for
        // host checks.
        return bit_to_word_layout(layout).offset(idx);
    }
    template <size_t N>
    static size_t get(const tensor_layout& layout, const std::array<int, N>& idx)
    {
        // Create a layout for 32-bit words and use that for
        // offset calculation. Inefficient, but only used for
        // host checks.
        return bit_to_word_layout(layout).offset(idx);
    }
};

////////////////////////////////////////////////////////////////////////
// get_element_address()
// Returns the address of an element in a tensor, given a base address,
// a tensor layout, and the data type. This function can be used instead
// of the offset_generator<> template when the data type is not known at
// compile time.
inline
void* get_element_address(void*                                 baseAddress,
                          cuphyDataType_t                       t,
                          const tensor_layout&                  layout,
                          const std::array<int, CUPHY_DIM_MAX>& idx)
{
    size_t obytes = 0;
    if(CUPHY_BIT == t)
    {
        obytes = offset_generator<CUPHY_BIT>::get(layout, idx) *
                 sizeof(type_traits<CUPHY_BIT>::type);
    }
    else
    {
        // Bytes offset is offset (in elements) times element size
        obytes = layout.offset(idx) * get_element_size(t);
    }
    return static_cast<char*>(baseAddress) + obytes;
}

////////////////////////////////////////////////////////////////////////
// cuphy::device_alloc
struct device_alloc
{
    static void* allocate(size_t nbytes, cuphyMemoryFootprint* pMemFootprint=nullptr)
    {
        void*       addr = nullptr;
        cudaError_t s = cudaMalloc(&addr, nbytes);
        if(cudaSuccess != s)
        {
            throw cuda_exception(s);
        }
        if(pMemFootprint)
        {
            //NVLOGC_FMT(0, "allocate {} Bytes", nbytes);
            pMemFootprint->addGpuAllocation(nbytes);
        }
        return addr;
    }
    static void deallocate(void* addr)
    {
        CUDA_CHECK(cudaFree(addr));
    }
};

////////////////////////////////////////////////////////////////////////
// cuphy::pinned_alloc
struct pinned_alloc
{
    static void* allocate(size_t nbytes, cuphyMemoryFootprint* pMemFootprint=nullptr)
    {
        void*       addr =  nullptr;
        cudaError_t s = cudaHostAlloc(&addr, nbytes, 0);
        if(cudaSuccess != s)
        {
            throw cuda_exception(s);
        }
        //No tracking of host pinned memory currently
        return addr;
    }
    static void deallocate(void* addr)
    {
        CUDA_CHECK(cudaFreeHost(addr));
    }
};

////////////////////////////////////////////////////////////////////////
// tensor_ref
// Reference to a tensor, consisting of a cuPHY tensor descriptor handle
// and a (non-owned) memory address. (This class is a "reference" to the
// buffer - it does not manage allocation or freeing of the address. It
// is assumed that the caller is managing resource allocation.)
class tensor_ref
{
public:
    tensor_ref(void* pv = nullptr) : addr_(pv) {}
    tensor_ref(const tensor_desc& desc, void* pv) :
      desc_(desc),
      addr_(pv)
    {
    }
    tensor_ref(tensor_desc&& desc, void* pv) :
      desc_(std::move(desc)),
      addr_(pv)
    {
    }
    tensor_ref(tensor_ref&& t) :
      desc_(std::move(t.desc_)),
      addr_(t.addr_)
    {
    }
    tensor_ref(const tensor_ref& t) :
      desc_(t.desc_),
      addr_(t.addr_)
    {
    }
    tensor_ref& operator=(tensor_ref&& t)
    {
        desc_ = std::move(t.desc_);
        addr_ = t.addr_;
        return *this;
    }
    tensor_ref& operator=(const tensor_ref& t)
    {
        desc_ = t.desc_;
        addr_ = t.addr_;
        return *this;
    }
    tensor_ref subset(const index_group& grp)
    {
        // For bit tensors, the "word" address is only accurate for the
        // first bit in the word.
        if(CUPHY_BIT == type())
        {
            if(0 != (grp.ranges()[0].start() % 32))
            {
                throw std::runtime_error("CUPHY_BIT subset must be word aligned");
            }
        }
        tensor_info tinfo = desc_.get_info();
        // Adjust the tensor address to the first index in the subset
        void* subset_addr = get_element_address(addr_,
                                                tinfo.type(),
                                                tinfo.layout(),
                                                grp.start_indices());
        return tensor_ref(grp.get_tensor_desc(desc_),
                          subset_addr);
    }
    tensor_desc&       desc()             { return desc_;                   }
    const tensor_desc& desc() const       { return desc_;                   }
    void*              addr()             { return addr_;                   }
    const void*        addr() const       { return addr_;                   }
    void               set_addr(void* pv) { addr_ = pv;                     }
    int                rank() const       { return desc_.rank();            }
    cuphyDataType_t    type() const       { return desc_.get_info().type(); }
private:
    tensor_desc desc_;
    void*       addr_;
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::tensor
// Class to manage a generic (non-typed) cuPHY tensor descriptor of any
// rank (up to the number supported by the cuPHY library) and an
// assocated memory allocation.
// NOTE: The tensor size is immutable (cannot be changed after construction)
template <class TAllocator = device_alloc>
class tensor //
{
public:
    typedef tensor_ref tensor_ref_t;
    tensor() : addr_(nullptr), alloc_memory(0) { }
    tensor(cuphyDataType_t type, const tensor_layout& layout_in, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, layout_in, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(cuphyDataType_t type, int dim0, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, dim0, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(cuphyDataType_t type, int dim0, int dim1, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, dim0, dim1, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(cuphyDataType_t type, int dim0, int dim1, int dim2, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, dim0, dim1, dim2, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, dim0, dim1, dim2, dim3, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, int dim4, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(type, dim0, dim1, dim2, dim3, dim4, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(desc_.get_info().layout()),
        alloc_memory(1)
    {
    }
    tensor(const tensor_info& tinfo, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(tinfo, flags),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)),
        layout_(tinfo.layout()),
        alloc_memory(1)
    {
    }
    tensor(void * pre_alloc_addr, cuphyDataType_t type, int dim0, tensor_flags flags = tensor_flags::align_default) :
        desc_(type, dim0, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(desc_.get_info().layout()),
        alloc_memory(0)
    {
    }
    tensor(void * pre_alloc_addr, cuphyDataType_t type, int dim0, int dim1, tensor_flags flags = tensor_flags::align_default) :
        desc_(type, dim0, dim1, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(desc_.get_info().layout()),
        alloc_memory(0)
    {
    }
    tensor(void * pre_alloc_addr, cuphyDataType_t type, int dim0, int dim1, int dim2, tensor_flags flags = tensor_flags::align_default) :
        desc_(type, dim0, dim1, dim2, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(desc_.get_info().layout()),
        alloc_memory(0)
    {
    }
    tensor(void * pre_alloc_addr, cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, tensor_flags flags = tensor_flags::align_default) :
        desc_(type, dim0, dim1, dim2, dim3, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(desc_.get_info().layout()),
        alloc_memory(0)
    {
    }
    tensor(void * pre_alloc_addr, cuphyDataType_t type, int dim0, int dim1, int dim2, int dim3, int dim4, tensor_flags flags = tensor_flags::align_default) :
        desc_(type, dim0, dim1, dim2, dim3, dim4, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(desc_.get_info().layout()),
        alloc_memory(0)
    {
    }
    tensor(void * pre_alloc_addr, const tensor_info& tinfo, tensor_flags flags = tensor_flags::align_default) :
        desc_(tinfo, flags),
        addr_(pre_alloc_addr), /* No allocation */
        layout_(tinfo.layout()),
        alloc_memory(0)
    {
    }
    template <class TAlloc_>
    tensor(const tensor<TAlloc_>& t) :
        desc_(t.desc()),
        addr_(TAllocator::allocate(desc_.get_size_in_bytes())), /*TODO allocation not tracked in memory footprint tracker */
        layout_(t.layout()),
        alloc_memory(1)
    {
        // Use convert() function, which will copy() if descs match
        convert(t);
    }
    tensor(const tensor& t) :
    desc_(t.desc()),
    addr_(TAllocator::allocate(desc_.get_size_in_bytes())), /*TODO allocation not tracked in memory footprint tracker */
    layout_(t.layout()),
    alloc_memory(1)
    {
        // Use convert() function, which will copy() if descs match
        convert(t);
    }
    tensor(tensor&& t) noexcept :
        desc_(std::move(t.desc_)),
        addr_(t.addr_),
        layout_(t.layout_)
    {
        t.addr_ = nullptr;
        alloc_memory = t.alloc_memory;
    }
    ~tensor() { if((addr_) && (alloc_memory == 1)) TAllocator::deallocate(addr_); }

    template <class TAlloc_>
    tensor& operator=(const tensor<TAlloc_>& t)
    {
        if(addr() != t.addr())
        {
            copy(t);
        }
        return *this;
    }
    tensor& operator=(const tensor& t)
    {
        if(this != &t)
        {
            copy(t);
        }
        return *this;
    }
    tensor& operator=(tensor&& t)
    {
        if((addr_) && (alloc_memory == 1)) TAllocator::deallocate(addr_);
        addr_ = t.addr_;
        t.addr_ = nullptr;
        layout_ = t.layout();
        desc_ = t.desc();
        alloc_memory = t.alloc_memory;
        return *this;
    }
    // Explicit copy() for callers that want to provide a stream
    template <class T>
    void copy(const T& tSrc, cudaStream_t strm = 0, cuphyMemoryFootprint* pMemoryFootprint=nullptr)
    {
        // If dimensions don't match, allocate a new descriptor.
        // If dimensions match but strides don't, assume that the
        // caller wants to keep the destination layout.
        if(dimensions() != tSrc.dimensions())
        {
            if((addr_) && (alloc_memory == 1))
            {
                TAllocator::deallocate(addr_);
                addr_ = nullptr;
            }
            desc_ = tSrc.desc();
            addr_ = TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint);
            alloc_memory = 1;
            layout_ = desc_.get_info().layout();
        }
        // Copy data.
        convert(tSrc, strm);
    }
    template <class T>
    void convert(const T& tSrc, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyConvertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyConvertTensor()");
        }
    }
    template <typename TValue>
    void fill(TValue v, cudaStream_t strm = 0)
    {
        variant       value(v);
        cuphyStatus_t s = cuphyFillTensor(desc_.handle(),
                                          addr_,
                                          &value,
                                          strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyFillTensor()");
        }
    }
    template <class TDst >
    void tile(TDst& v, int nrows, int ncols, cudaStream_t strm = 0)
    {
        std::array<int, 2> tileExtents = {{nrows, ncols}};
        cuphyStatus_t s = cuphyTileTensor(v.desc().handle(),
                                          v.addr(),
                                          desc_.handle(),
                                          addr_,
                                          tileExtents.size(),
                                          tileExtents.data(),
                                          strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTileTensor()");
        }
    }
    template <class TSrcA, class TSrcB>
    void add(TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyTensorElementWiseOperation(desc_.handle(),     // tDst
                                                          addr_,              // pDst
                                                          a.desc().handle(),  // tSrcA
                                                          a.addr(),           // pSrcA
                                                          nullptr,            // alpha
                                                          b.desc().handle(),  // tSrcB
                                                          b.addr(),           // pSrcB
                                                          nullptr,            // beta
                                                          CUPHY_ELEMWISE_ADD, // operation
                                                          strm);              // stream
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
        }
    }
    template <class TSrcA, class TSrcB>
    void xor_op(TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyTensorElementWiseOperation(desc_.handle(),         // tDst
                                                          addr_,                  // pDst
                                                          a.desc().handle(),      // tSrcA
                                                          a.addr(),               // pSrcA
                                                          nullptr,                // alpha
                                                          b.desc().handle(),      // tSrcB
                                                          b.addr(),               // pSrcB
                                                          nullptr,                // beta
                                                          CUPHY_ELEMWISE_BIT_XOR, // operation
                                                          strm);                  // stream
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
        }
    }
    tensor_ref subset(const index_group& grp)
    {
        // For bit tensors, the "word" address is only accurate for the
        // first bit in the word.
        if(CUPHY_BIT == type())
        {
            if(0 != (grp.ranges()[0].start() % 32))
            {
                throw std::runtime_error("CUPHY_BIT subset must be word aligned");
            }
        }
        tensor_info tinfo = desc_.get_info();
        // Adjust the tensor address to the first index in the subset
        void* subset_addr = get_element_address(addr_,
                                                tinfo.type(),
                                                tinfo.layout(),
                                                grp.start_indices());
        return tensor_ref(grp.get_tensor_desc(desc_),
                          subset_addr);
    }
    const tensor_desc&             desc() const       { return desc_; }
    void*                          addr() const       { return addr_; }
    int                            rank() const       { return layout_.rank(); }
    const vec<int, CUPHY_DIM_MAX>& dimensions() const { return layout_.dimensions(); }
    const vec<int, CUPHY_DIM_MAX>& strides()    const { return layout_.strides();    }
    const tensor_layout&           layout()     const { return layout_; }
    cuphyDataType_t                type()       const
    {
        tensor_info i = desc_.get_info();
        return i.type();
    }
private:
    tensor_desc   desc_;
    void*         addr_;
    tensor_layout layout_;
    int           alloc_memory;
};
// clang-format on

using tensor_device = tensor<device_alloc>;
using tensor_pinned = tensor<pinned_alloc>;

////////////////////////////////////////////////////////////////////////
// typed_tensor_ref
// Reference to a tensor, consisting of a cuPHY tensor descriptor handle
// and a (non-owned) address. (This class is a "reference" to the buffer
// - it does not manage allocation or freeing of the address. It is
// assumed that the caller is managing resource allocation.) The
// tensor descriptor is "owned" by this class.
template <cuphyDataType_t TType>
class typed_tensor_ref
{
public:
    typedef typename type_traits<TType>::type element_t;
    typedef          offset_generator<TType>  offset_gen_t;
    typed_tensor_ref(element_t* pe = nullptr) : addr_(pe) {}
    typed_tensor_ref(const tensor_desc& desc, element_t* pe) :
      desc_(desc),
      addr_(pe)
    {
    }
    typed_tensor_ref(tensor_desc&& desc, element_t* pe) :
      desc_(std::move(desc)),
      addr_(pe)
    {
    }
    typed_tensor_ref(typed_tensor_ref&& t) :
      desc_(std::move(t.desc_)),
      addr_(t.addr_)
    {
    }
    typed_tensor_ref(const typed_tensor_ref& t) :
        desc_(t.desc_),
        addr_(t.addr_)
    {
    }
    typed_tensor_ref& operator=(typed_tensor_ref&& t)
    {
        desc_ = std::move(t.desc_);
        addr_ = t.addr_;
        return *this;
    }
    typed_tensor_ref& operator=(const typed_tensor_ref& t)
    {
        desc_ = t.desc_;
        addr_ = t.addr_;
        return *this;
    }
    tensor_desc&       desc()                  { return desc_;        }
    const tensor_desc& desc() const            { return desc_;        }
    element_t*         addr()                  { return addr_;        }
    const element_t*   addr() const            { return addr_;        }
    void               set_addr(element_t* pv) { addr_ = pv;          }
    int                rank() const            { return desc_.rank(); }
    cuphyDataType_t    type() const            { return TType;        }
private:
    tensor_desc desc_;
    element_t*  addr_;
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// typed_tensor
// Class to manage a cuPHY tensor descriptor of any rank (up to the
// maximum supported by the library) and an assocated memory allocation.
template <cuphyDataType_t TType, class TAllocator = device_alloc>
class typed_tensor
{
public:
    typedef typename type_traits<TType>::type element_t;
    typedef          TAllocator               allocator_t;
    typedef          offset_generator<TType>  offset_gen_t;
    typedef          typed_tensor_ref<TType>  tensor_ref_t;
    typed_tensor() : addr_(nullptr) { }
    typed_tensor(const tensor_layout& tlayout, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(tensor_info(TType, tlayout), flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)))
    {
    }
    typed_tensor(int dim0, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(TType, dim0, flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)))
    {
    }
    typed_tensor(int dim0, int dim1, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(TType, dim0, dim1, flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)))
    {
    }
    typed_tensor(int dim0, int dim1, int dim2, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(TType, dim0, dim1, dim2, flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)))
    {
    }
    typed_tensor(int dim0, int dim1, int dim2, int dim3, tensor_flags flags = tensor_flags::align_default, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        desc_(TType, dim0, dim1, dim2, dim3, flags),
        layout_(desc_.get_info().layout()),
        addr_(static_cast<element_t*>(allocator_t::allocate(desc_.get_size_in_bytes(), pMemoryFootprint)))
    {
    }
    typed_tensor(const typed_tensor&, cudaStream_t strm = 0) = delete; // TODO
    typed_tensor(typed_tensor&& t) :
        desc_(std::move(t.desc_)),
        layout_(t.layout()),
        addr_(t.addr_)
    {
        t.addr_ = nullptr;
    }
    ~typed_tensor() { if(addr_) allocator_t::deallocate(addr_); }

    template <class TSrc>
    typed_tensor& operator=(const TSrc& tSrc)
    {
        cuphyStatus_t s = cuphyConvertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             0);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyConvertTensor()");
        }
        cudaStreamSynchronize(0);
        return *this;
    }
    typed_tensor& operator=(typed_tensor&& t)
    {
        desc_   = std::move(t.desc_);
        layout_ = t.layout();
        addr_   = t.addr_;
        t.addr_ = nullptr;
        return *this;
    }
    // Explicit copy() for callers that want to provide a stream
    template <class T>
    void copy(const T& tSrc, cudaStream_t strm = 0, cuphyMemoryFootprint* pMemoryFootprint=nullptr)
    {
        // If dimensions don't match, allocate a new descriptor.
        // If dimensions match but strides don't, assume that the
        // caller wants to keep the destination layout.
        if(dimensions() != tSrc.dimensions())
        {
            if(addr_)
            {
                TAllocator::deallocate(addr_);
                addr_ = nullptr;
            }
            desc_ = tSrc.desc();
            addr_ = static_cast<element_t*>(TAllocator::allocate(desc_.get_size_in_bytes(), pMemoryFootprint));
            layout_ = desc_.get_info().layout();
        }
        // Copy data.
        convert(tSrc, strm);
    }
    template <class TSrc>
    void convert(const tensor<TSrc>& tSrc, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyConvertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyConvertTensor()");
        }
    }
    void convert(const tensor_ref& tSrc, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyConvertTensor(desc_.handle(),
                                             addr_,
                                             tSrc.desc().handle(),
                                             tSrc.addr(),
                                             strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyConvertTensor()");
        }
    }
    template <typename TValue>
    void fill(TValue v, cudaStream_t strm = 0)
    {
        variant       value(v);
        cuphyStatus_t s = cuphyFillTensor(desc_.handle(),
                                          addr_,
                                          &value,
                                          strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyFillTensor()");
        }
    }
    template <class TDst >
    void tile(TDst& v, int nrows, int ncols, cudaStream_t strm = 0)
    {
        std::array<int, 2> tileExtents = {{nrows, ncols}};
        cuphyStatus_t s = cuphyTileTensor(v.desc().handle(),
                                          v.addr(),
                                          desc_.handle(),
                                          addr_,
                                          tileExtents.size(),
                                          tileExtents.data(),
                                          strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTileTensor()");
        }
    }
    template <class TSrcA, class TSrcB>
    void add(TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyTensorElementWiseOperation(desc_.handle(),     // tDst
                                                          addr_,              // pDst
                                                          a.desc().handle(),  // tSrcA
                                                          a.addr(),           // pSrcA
                                                          nullptr,            // alpha
                                                          b.desc().handle(),  // tSrcB
                                                          b.addr(),           // pSrcB
                                                          nullptr,            // beta
                                                          CUPHY_ELEMWISE_ADD, // operation
                                                          strm);              // stream
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
        }
    }
    template <class TSrcA, class TSrcB>
    void xor_op(TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyTensorElementWiseOperation(desc_.handle(),         // tDst
                                                          addr_,                  // pDst
                                                          a.desc().handle(),      // tSrcA
                                                          a.addr(),               // pSrcA
                                                          nullptr,                // alpha
                                                          b.desc().handle(),      // tSrcB
                                                          b.addr(),               // pSrcB
                                                          nullptr,                // beta
                                                          CUPHY_ELEMWISE_BIT_XOR, // operation
                                                          strm);                  // stream
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
        }
    }
    template <class TDst >
    void sum(TDst& v, int dim = 0, cudaStream_t strm = 0)
    {
        cuphyStatus_t s = cuphyTensorReduction(v.desc().handle(),
                                               v.addr(),
                                               desc_.handle(),
                                               addr_,
                                               CUPHY_REDUCTION_SUM,
                                               dim,
                                               0,
                                               nullptr,
                                               strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy_fn_exception(s, "cuphyTensorReduction()");
        }
    }
    const tensor_desc&             desc() const       { return desc_; }
    element_t*                     addr() const       { return addr_; }
    int                            rank() const       { return layout_.rank(); }
    const vec<int, CUPHY_DIM_MAX>& dimensions() const { return layout_.dimensions(); }
    const vec<int, CUPHY_DIM_MAX>& strides() const    { return layout_.strides();    }
    const tensor_layout&           layout() const     { return layout_; }
    // operator()
    // Indexed access, only enabled on the host for non-device allocations
    template <int N, typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    elem(const int(&idx)[N])
    {
        return addr_[offset_gen_t::get(layout_, idx)];
    }
    template <typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(int idx)
    {
        int idx_[1] = {idx};
        layout_.check_bounds(idx_);
        return addr_[offset_gen_t::get(layout_, idx_)];
    }
    template <typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(int idx0, int idx1)
    {
        int idx_[2] = {idx0, idx1};
        layout_.check_bounds(idx_);
        return addr_[offset_gen_t::get(layout_, idx_)];
    }
    template <typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(int idx0, int idx1, int idx2)
    {
        int idx_[3] = {idx0, idx1, idx2};
        layout_.check_bounds(idx_);
        return addr_[offset_gen_t::get(layout_, idx_)];
    }
    template <typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(int idx0, int idx1, int idx2, int idx3)
    {
        int idx_[4] = {idx0, idx1, idx2, idx3};
        layout_.check_bounds(idx_);
        return addr_[offset_gen_t::get(layout_, idx_)];
    }
    template <typename TAlloc = TAllocator>
    typename std::enable_if<!std::is_same<TAlloc, device_alloc>::value, element_t&>::type
    operator()(int idx0, int idx1, int idx2, int idx3, int idx4)
    {
        int idx_[5] = {idx0, idx1, idx2, idx3, idx4};
        layout_.check_bounds(idx_);
        return addr_[offset_gen_t::get(layout_, idx_)];
    }
    template <int N, typename TAlloc = TAllocator>
    typename std::enable_if<std::is_same<TAlloc, device_alloc>::value, element_t>::type
    operator()(const int(&idx)[N])
    {
        element_t elem;
        cudaError_t e = cudaMemcpy(&elem,
                                   addr_ + offset_gen_t::get(layout_, idx),
                                   sizeof(element_t),
                                   cudaMemcpyDeviceToHost);
        if(e != cudaSuccess)
        {
            throw cuda_exception(e);
        }
        return elem;
    }
    typed_tensor_ref<TType> subset(const index_group& grp)
    {
        // For bit tensors, the "word" address is only accurate for the
        // first bit in the word.
        if(CUPHY_BIT == TType)
        {
            if(0 != (grp.ranges()[0].start() % 32))
            {
                throw std::runtime_error("CUPHY_BIT subset must be word aligned");
            }
        }
        // Adjust the tensor address to the first index in the subset
        element_t*         subset_addr = addr_ +
                                         offset_gen_t::get(layout_, grp.start_indices());
        return typed_tensor_ref<TType>(grp.get_tensor_desc(desc_),
                                       subset_addr);
    }
private:
    tensor_desc   desc_;
    tensor_layout layout_;
    element_t*    addr_;
};
// clang-format on

template <class TDstAlloc, class TSrcAlloc>
struct memcpy_helper;

template <>
struct memcpy_helper<device_alloc, device_alloc>
{
    static constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
};

template <>
struct memcpy_helper<pinned_alloc, device_alloc>
{
    static constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
};

template <>
struct memcpy_helper<device_alloc, pinned_alloc>
{
    static constexpr cudaMemcpyKind kind = cudaMemcpyHostToDevice;
};

template <>
struct memcpy_helper<pinned_alloc, pinned_alloc>
{
    static constexpr cudaMemcpyKind kind = cudaMemcpyHostToHost;
};

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphy::buffer
template <typename T, class TAlloc>
class buffer
{
public:
    typedef T      element_t;
    typedef TAlloc allocator_t;

    buffer() : addr_(nullptr), size_(0) {}
    buffer(size_t numElements, cuphyMemoryFootprint* pMemoryFootprint=nullptr) :
        addr_(static_cast<element_t*>(allocator_t::allocate(numElements * sizeof(T), pMemoryFootprint))),
        size_(numElements)
    {
    };
    ~buffer() { if(addr_) allocator_t::deallocate(addr_); }
    void delete_buffer() { if(addr_) allocator_t::deallocate(addr_); addr_ = nullptr;}
    template <class TAlloc2>
    buffer(const buffer<T, TAlloc2>& b) :
        addr_(static_cast<element_t*>(allocator_t::allocate(b.size() * sizeof(T)))), /* TODO allocation not tracked in memory footprint tracker */
        size_(b.size())
    {
        cudaError_t e = cudaMemcpy(addr_,
                                   b.addr(),
                                   sizeof(T) * size_,
                                   memcpy_helper<TAlloc, TAlloc2>::kind);
        if(e != cudaSuccess)
        {
            throw cuda_exception(e);
        }
    }
    buffer(const buffer& b) :
    addr_(static_cast<element_t*>(allocator_t::allocate(b.size() * sizeof(T)))), /* TODO allocation not tracked in memory footprint tracker */
    size_(b.size())
    {
        cudaError_t e = cudaMemcpy(addr_,
                                   b.addr(),
                                   sizeof(T) * size_,
                                   memcpy_helper<TAlloc, TAlloc>::kind);
        if(e != cudaSuccess)
        {
            throw cuda_exception(e);
        }
    }
    template <class TAlloc2>
    buffer(buffer<T, TAlloc2>&& b) :
        addr_(b.addr()),
        size_(b.size())
    {
        b.size_ = 0;
        b.addr_ = nullptr;
    }
    buffer(const std::vector<T>& srcVec) :
        addr_(static_cast<element_t*>(allocator_t::allocate(srcVec.size() * sizeof(T)))), /* TODO allocation not tracked in memory footprint tracker */
        size_(srcVec.size())
    {
        cudaError_t e = cudaMemcpy(addr_,
                                   srcVec.data(),
                                   sizeof(T) * size_,
                                   cudaMemcpyDefault);
        if(e != cudaSuccess)
        {
            throw cuda_exception(e);
        }
    };
    buffer& operator=(buffer && b)
    {
       if(addr_) allocator_t::deallocate(addr_);
       addr_   = b.addr();
       size_   = b.size();
       b.addr_ = nullptr;
       return *this;
    }
    element_t*       addr()       { return addr_; }
    const element_t* addr() const { return addr_; }
    size_t           size() const { return size_; }
    // operator()
    // Indexed access, only enabled on the host for non-device allocations.
    // Use dummy template parameter for SFINAE
    template <typename Alloc = TAlloc>
    typename std::enable_if<!std::is_same<device_alloc, Alloc>::value, element_t&>::type
    operator[](size_t idx) const
    {
        assert(idx < size_);
        return addr_[idx];
    }
private:
    element_t* addr_;
    size_t     size_;
};
// clang-format on

// Create a buffer wrapper class to allow for different buffer types to be used interchangeably
class buffer_wrapper {
public:
    virtual ~buffer_wrapper() = default;
    buffer_wrapper() = default;
    buffer_wrapper(const buffer_wrapper&) = default;
    buffer_wrapper(buffer_wrapper&&) = default;
    buffer_wrapper& operator=(const buffer_wrapper&) = default;
    buffer_wrapper& operator=(buffer_wrapper&&) = default;
    virtual void* addr() = 0;
};

template<typename AllocType>
class buffer_impl : public buffer_wrapper {
public:
    explicit buffer_impl(size_t size) : _buffer(size) {}
    [[nodiscard]] void* addr() override { return _buffer.addr(); }
private:
    buffer<uint8_t, AllocType> _buffer;
};


template <class T>
struct device_deleter
{
    typedef typename std::remove_all_extents<T>::type ptr_t;
    //typedef T ptr_t;
    void operator()(ptr_t* p) const
    {
        CUDA_CHECK(cudaFree(p));
    }
};

template <class T>
struct pinned_deleter
{
    typedef typename std::remove_all_extents<T>::type ptr_t;
    //typedef T ptr_t;
    void operator()(ptr_t* p) const { CUDA_CHECK(cudaFreeHost(p)); }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter<T>>;

template <typename T>
using unique_pinned_ptr = std::unique_ptr<T, pinned_deleter<T>>;

template <typename T>
unique_device_ptr<T> make_unique_device(size_t count = 1, cuphyMemoryFootprint* pMemoryFootprint = nullptr)
{
    typedef typename unique_device_ptr<T>::pointer pointer_t;
    pointer_t                                      p = static_cast<pointer_t>(device_alloc::allocate(count * sizeof(T), pMemoryFootprint));
    return unique_device_ptr<T>(p);
}

template <typename T>
unique_pinned_ptr<T> make_unique_pinned(size_t count = 1)
{
    typedef typename unique_pinned_ptr<T>::pointer pointer_t;
    pointer_t                                      p = static_cast<pointer_t>(pinned_alloc::allocate(count * sizeof(T)));
    return unique_pinned_ptr<T>(p);
}

////////////////////////////////////////////////////////////////////////
// cuphy::device
class device
{
public:
    typedef int int3_t[3];
    device(int idx = 0) : index_(idx)
    {
        cudaError_t e = cudaGetDeviceProperties(&properties_, index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        // Retrieve attributes deprecated from device properties
        e = cudaDeviceGetAttribute(&clockRate_, cudaDevAttrClockRate, index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        e = cudaDeviceGetAttribute(&kernelExecTimeoutEnabled_, cudaDevAttrKernelExecTimeout, index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        e = cudaDeviceGetAttribute(&computeMode_, cudaDevAttrComputeMode, index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        e = cudaDeviceGetAttribute(&memoryClockRate_, cudaDevAttrMemoryClockRate, index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    void set()
    {
        cudaError_t e = cudaSetDevice(index_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    static int get_current()
    {
        int         idx = 0;
        cudaError_t e   = cudaGetDevice(&idx);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return idx;
    }
    // clang-format off
    const cudaDeviceProp& properties() const { return properties_; }
    const char*   name()                         const { return properties_.name;                        }
    size_t        total_global_mem_bytes()       const { return properties_.totalGlobalMem;              }
    size_t        shared_mem_per_block()         const { return properties_.sharedMemPerBlock;           }
    int           registers_per_block()          const { return properties_.regsPerBlock;                }
    int           warp_size()                    const { return properties_.warpSize;                    }
    size_t        memory_pitch()                 const { return properties_.memPitch;                    }
    int           max_threads_per_block()        const { return properties_.maxThreadsPerBlock;          }
    const int3_t& max_threads_dim()              const { return properties_.maxThreadsDim;               }
    const int3_t& max_grid_size()                const { return properties_.maxGridSize;                 }
    int           clock_rate_kHz()               const { return clockRate_;                              }
    size_t        total_const_mem_bytes()        const { return properties_.totalConstMem;               }
    int           major_version()                const { return properties_.major;                       }
    int           minor_version()                const { return properties_.minor;                       }
    int           multiprocessor_count()         const { return properties_.multiProcessorCount;         }
    int           kernel_timeout_enabled()       const { return kernelExecTimeoutEnabled_;               }
    int           integrated()                   const { return properties_.integrated;                  }
    int           can_map_host_memory()          const { return properties_.canMapHostMemory;            }
    int           compute_mode()                 const { return computeMode_;                            }
    int           concurrent_kernels()           const { return properties_.concurrentKernels;           }
    int           ECC_enabled()                  const { return properties_.ECCEnabled;                  }
    int           pci_bus_ID()                   const { return properties_.pciBusID;                    }
    int           pci_device_ID()                const { return properties_.pciDeviceID;                 }
    int           pci_domain_ID()                const { return properties_.pciDomainID;                 }
    int           tcc_driver()                   const { return properties_.tccDriver;                   }
    int           async_engine_count()           const { return properties_.asyncEngineCount;            }
    int           unified_addressing()           const { return properties_.unifiedAddressing;           }
    int           memory_clock_rate_kHz()        const { return memoryClockRate_;                        }
    int           memory_bus_width()             const { return properties_.memoryBusWidth;              }
    int           L2_cache_size_bytes()          const { return properties_.l2CacheSize;                 }
    int           max_threads_per_SM()           const { return properties_.maxThreadsPerMultiProcessor; }
    int           stream_prio_supported()        const { return properties_.streamPrioritiesSupported;   }
    int           global_L1_cache_supported()    const { return properties_.globalL1CacheSupported;      }
    int           local_L1_cache_supported()     const { return properties_.localL1CacheSupported;       }
    size_t        shmem_per_multiprocessor()     const { return properties_.sharedMemPerMultiprocessor;  }
    int           registers_per_multiprocessor() const { return properties_.regsPerMultiprocessor;       }
    // clang-format on
    std::string   desc() const
    {
        char buf[128];
        std::string s(properties_.name);
        snprintf(buf,
                 sizeof(buf) / sizeof(buf[0]),
                 ": %d SMs @ %.0f MHz, %.1f GiB @ %.0f MHz, Compute Capability %d.%d, PCI %04X:%02X:%02X",
                 properties_.multiProcessorCount,
                 clockRate_ / 1000.0,
                 properties_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                 memoryClockRate_ / 1000.0,
                 properties_.major,
                 properties_.minor,
                 properties_.pciDomainID,
                 properties_.pciBusID,
                 properties_.pciDeviceID);
        s.append(buf);
        return s;
    }
private:
    int            index_;
    cudaDeviceProp properties_;
    int            clockRate_;
    int            memoryClockRate_;
    int            kernelExecTimeoutEnabled_;
    int            computeMode_;
};

////////////////////////////////////////////////////////////////////////
// cuphy::stream
class stream final
{
public:
    explicit stream(unsigned int flags = cudaStreamDefault)
    {
        const auto e = cudaStreamCreateWithFlags(&stream_, flags);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }

    stream(unsigned int flags, int priority)
    {
        const auto e = cudaStreamCreateWithPriority(&stream_, flags, priority);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }

    stream(stream&& s) noexcept : stream_(std::exchange(s.stream_, nullptr)) {}
    ~stream() { if(stream_) CUDA_CHECK_NO_THROW(cudaStreamDestroy(stream_)); }

    void synchronize() const
    {
        const auto e = cudaStreamSynchronize(stream_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    void wait_event(cudaEvent_t ev) const
    {
        const auto e = cudaStreamWaitEvent(stream_, ev, 0);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    [[nodiscard]] cudaError_t query() const { return cudaStreamQuery(stream_); }
    stream& operator==(const stream&) = delete;
    stream(const stream&) = delete;
    [[nodiscard]] cudaStream_t handle() const { return stream_; }
private:
    cudaStream_t stream_{};
};

////////////////////////////////////////////////////////////////////////
// cuphy::event
class event
{
public:
    event(unsigned int flags = cudaEventDefault)
    {
        cudaError_t e = cudaEventCreateWithFlags(&ev_, flags);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    event(event&& e) : ev_(e.ev_) { e.ev_ = nullptr; }
    ~event() { if(ev_) CUDA_CHECK_NO_THROW(cudaEventDestroy(ev_)); }

    void synchronize()
    {
        cudaError_t e = cudaEventSynchronize(ev_);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    void record()
    {
        cudaError_t e = cudaEventRecord(ev_, 0);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    void record(cudaStream_t s)
    {
        cudaError_t e = cudaEventRecord(ev_, s);
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    void record(stream& s)
    {
        cudaError_t e = cudaEventRecord(ev_, s.handle());
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
    }
    cudaError_t query() { return cudaEventQuery(ev_); }
    event& operator=(const event&) = delete;
    event(const event&) = delete;
    cudaEvent_t handle() { return ev_; }
private:
    cudaEvent_t ev_;
};

////////////////////////////////////////////////////////////////////////
// cuphy::event_timer
// Class to time operations using a pair of CUDA events
//
// Usage:
// event_timer tmr;
// tmr.record_begin();             // Record begin event in stream
//   something interesting...
// tmr.record_end();               // Record end event in stream
// tmr.synchronize();              // Allow operations to finish
// float t =tmr.elapsed_time_ms(); // Retrieve time
class event_timer
{
public:
    void record_begin()               { begin_event_.record();  }
    void record_begin(stream&      s) { begin_event_.record(s); }
    void record_begin(cudaStream_t s) { begin_event_.record(s); }
    void record_end()                 { end_event_.record();    }
    void record_end(stream&      s)   { end_event_.record(s);   }
    void record_end(cudaStream_t s)   { end_event_.record(s);   }
    void synchronize() { end_event_.synchronize(); }
    cudaEvent_t begin_event_handle()  { return begin_event_.handle(); }
    cudaEvent_t end_event_handle()    { return end_event_.handle();   }
    float elapsed_time_ms()
    {
        float time_ms = 0.0f;
        cudaError_t e = cudaEventElapsedTime(&time_ms,
                                             begin_event_.handle(),
                                             end_event_.handle());
        if(cudaSuccess != e)
        {
            throw cuda_exception(e);
        }
        return time_ms;
    }
private:
    event begin_event_, end_event_;
};

////////////////////////////////////////////////////////////////////////
// cuphy::stream_pool
// Simple round-robin stream pool
class stream_pool
{
public:
    //------------------------------------------------------------------
    // Constructor
    stream_pool(size_t maxSize = 32, int priority=0) :
        current_stream_idx_(0),
        fork_event_(cudaEventDisableTiming)
    {
        //printf("stream_pool with maxSize %lu and priority %d.\n", maxSize, priority);
        for(size_t i = 0; i < maxSize; ++i)
        {
            streams_.emplace_back(cudaStreamNonBlocking, priority);
            join_events_.emplace_back(cudaEventDisableTiming);
        }
        fork_width_ = streams_.size();
        priority_   = priority;
    }

    //------------------------------------------------------------------
    // Returns the maximum size of the thread pool
    size_t max_size() { return streams_.size(); }
    //------------------------------------------------------------------
    // resize()
    void resize(size_t maxSize)
    {
        if(maxSize != streams_.size())
        {
            streams_.clear();
            join_events_.clear();
            for(size_t i = 0; i < maxSize; ++i)
            {
                streams_.emplace_back(cudaStreamNonBlocking, priority_);
                join_events_.emplace_back(cudaEventDisableTiming);
            }
            fork_width_ = streams_.size();
        }
    }

    void resize(size_t maxSize, int priority)
    {
        if((maxSize != streams_.size()) || (priority_ != priority))
        {
            // printf("stream_pool with maxSize %lu and priority %d.\n", maxSize, priority);
            streams_.clear();
            join_events_.clear();
            priority_ = priority;
            for(size_t i = 0; i < maxSize; ++i)
            {
                streams_.emplace_back(cudaStreamNonBlocking, priority_);
                join_events_.emplace_back(cudaEventDisableTiming);
            }
            fork_width_ = streams_.size();
        }
    }


    //------------------------------------------------------------------
    // fork()
    // The width specifies how many streams to use. If the width is
    // greater than the maximum size (from either construction or if the
    // resize() function is called, an exception is thrown.
    // If 0 is given, the maximum size is used.
    // This function records an event in the given stream. `width`
    // streams in the stream pool will wait on that recorded event, so
    // that the work submitted to those streams will not begin until
    // that event is signaled.
    void fork(cudaStream_t s, size_t width = 0)
    {
        if(width > streams_.size()) { throw std::runtime_error("Invalid stream pool fork width"); }
        if(0 == width)               { width = streams_.size(); }
        fork_width_ = width;
        current_stream_idx_ = 0;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Record an event in the given stream. Forked stream submissions
        // will wait for this event before continuing.
        fork_event_.record(s);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for(size_t i = 0; i < fork_width_; ++i)
        {
            streams_[i].wait_event(fork_event_.handle());
        }
    }
    //------------------------------------------------------------------
    // join()
    void join(cudaStream_t s)
    {
        for(size_t i = 0; i < fork_width_; ++i)
        {
            // Record an event in each worker stream
            join_events_[i].record(streams_[i]);
            // Force the main stream to wait for events from all worker
            // streams
            cudaError_t e = cudaStreamWaitEvent(s, join_events_[i].handle(), 0);
            if(cudaSuccess != e)
            {
                throw cuda_exception(e);
            }
        }
    }
    //------------------------------------------------------------------
    // advance()
    void advance()
    {
        current_stream_idx_ = (current_stream_idx_ + 1) % fork_width_;
    }
    //------------------------------------------------------------------
    // advance_to_start()
    void advance_to_start()
    {
        current_stream_idx_ = 0;
    }
    //------------------------------------------------------------------
    // current_stream()
    stream& current_stream() { return streams_[current_stream_idx_]; }
private:
    //------------------------------------------------------------------
    // Data
    std::vector<stream> streams_;
    size_t              current_stream_idx_;
    event               fork_event_;
    std::vector<event>  join_events_;
    size_t              fork_width_;
    int                 priority_;
};

template <size_t VALUE>
struct is_power_of_2
{
    static constexpr bool value = (0 == ((VALUE-1) & (VALUE)));
};

////////////////////////////////////////////////////////////////////////
// cuphy::linear_alloc
// Allocator to provide efficient sub-buffer allocations of a larger
// parent block. The parent block is allocated at initialization.
// Sub-allocations are not freed individually, but are freed collectively
// via the reset() function of this class. Therefore, its use is
// intended for situations when a collection of allocations have the
// same lifetime.
template <uint32_t ALLOC_ALIGN_BYTES, class TAlloc>
class linear_alloc
{
public:
    //------------------------------------------------------------------
    // linear_alloc()
    linear_alloc(size_t bufsize, cuphyMemoryFootprint* pMemoryFootprint=nullptr) : buffer_(TAlloc::allocate(bufsize, pMemoryFootprint)), size_(bufsize), offset_(0)
    {
        static_assert(is_power_of_2<ALLOC_ALIGN_BYTES>::value, "ALLOC_ALIGN_BYTES must be a power of 2");
    }
    linear_alloc(linear_alloc&& a) : buffer_(a.buffer_), size_(a.size_), offset_(a.offset_)
    {
        a.buffer_ = nullptr;
    }
    //------------------------------------------------------------------
    // ~linear_alloc()
    ~linear_alloc() { if(buffer_) TAlloc::deallocate(buffer_); }
    //------------------------------------------------------------------
    // reset()
    void reset() { offset_ = 0; }
    //------------------------------------------------------------------
    // alloc()
    void* alloc(size_t nbytes)
    {
        if((offset_ + nbytes) > size_)
        {
            throw std::runtime_error(std::string("linear_alloc::alloc(): Buffer size exceeded. offset = ") +
                                     std::to_string(offset_)                                               +
                                     std::string(", num_bytes = ")                                         +
                                     std::to_string(nbytes)                                                +
                                     std::string(", block_size = ")                                        +
                                     std::to_string(size_));
        }
        // Store the current offset for returning...
        void* p = static_cast<char*>(buffer_) + offset_;
#if 0
        printf("linear_alloc::alloc(): alloc_size = %lu, prev_offset = %lu, new_offset = %lu\n",
               nbytes,
               offset_,
               offset_ + ((nbytes + (ALLOC_ALIGN_BYTES - 1)) & ~(ALLOC_ALIGN_BYTES - 1)));
#endif
        // Increment the offset for the next allocation, aligning as
        // required
        offset_ += ((nbytes + (ALLOC_ALIGN_BYTES - 1)) & ~(ALLOC_ALIGN_BYTES - 1));
        return p;
    }

    //------------------------------------------------------------------
    // memset()
    void memset(int val, cudaStream_t strm = 0)
    {
        CUDA_CHECK(cudaMemsetAsync(buffer_, val, size_, strm));
    }
    //------------------------------------------------------------------
    // alloc()
    // Set the address of the given tensor ref based on the size of the
    // tensor ref descriptor.
    void alloc(tensor_ref& tref)
    {
        tref.set_addr(alloc(tref.desc().get_size_in_bytes()));
    }
    //------------------------------------------------------------------
    size_t size() const { return size_; }
    size_t offset() const { return offset_; }
    void*  address() const { return buffer_; }

    linear_alloc& operator=(const linear_alloc&) = delete;
    linear_alloc(const linear_alloc&) = delete;
private:
    void*  buffer_; // Parent allocation
    size_t size_;   // Total allocation size
    size_t offset_; // Offset of next allocation
};

// N_DESCRIPTORS - Number of kernel descriptors to be represented
template <size_t N_DESCRIPTORS>
class kernelDescrs
{
public:
    kernelDescrs(std::string const& name) : m_name(name) {};
    ~kernelDescrs() = default;
    kernelDescrs() = delete;
    kernelDescrs(kernelDescrs const&) = delete;
    kernelDescrs& operator=(kernelDescrs const&) = delete;

    bool alloc(std::array<size_t, N_DESCRIPTORS> const& descrSizeBytes, std::array<size_t, N_DESCRIPTORS> const& descrAlignBytes, cuphyMemoryFootprint* pMemoryFootprint=nullptr)
    {
        if(m_valid) return false;

        // set m_descrSizeBytes and m_totalDescrSizeBytes
        setAlignedSizeBytes(descrSizeBytes, descrAlignBytes);

        // allocate on CPU and GPU
        m_cpuDescrs = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_totalDescrSizeBytes));
        m_gpuDescrs = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(m_totalDescrSizeBytes, pMemoryFootprint));

        m_cpuDescrStartAddr[0] = m_cpuDescrs.addr();
        m_gpuDescrStartAddr[0] = m_gpuDescrs.addr();
        for(uint32_t i = 1; i < N_DESCRIPTORS; ++i)
        {
            m_cpuDescrStartAddr[i] = m_cpuDescrStartAddr[i-1] + m_descrSizeBytes[i-1];
            m_gpuDescrStartAddr[i] = m_gpuDescrStartAddr[i-1] + m_descrSizeBytes[i-1];
        }

        m_valid = true;
        return true;
    }

    // this will not allocate memory by calling cuphy::buffer<>(), but it updates start address and size arrays
    bool update(std::array<size_t, N_DESCRIPTORS> const& descrSizeBytes, std::array<size_t, N_DESCRIPTORS> const& descrAlignBytes)
    {
        if(!m_valid) return false;

        // set m_descrSizeBytes and m_totalDescrSizeBytes
        setAlignedSizeBytes(descrSizeBytes, descrAlignBytes);

        m_cpuDescrStartAddr[0] = m_cpuDescrs.addr();
        m_gpuDescrStartAddr[0] = m_gpuDescrs.addr();
        for(uint32_t i = 1; i < N_DESCRIPTORS; ++i)
        {
            m_cpuDescrStartAddr[i] = m_cpuDescrStartAddr[i-1] + m_descrSizeBytes[i-1];
            m_gpuDescrStartAddr[i] = m_gpuDescrStartAddr[i-1] + m_descrSizeBytes[i-1];
        }

        return true;
    }

    uint8_t* gpuPtr(){return m_gpuDescrs.addr();}
    uint8_t* cpuPtr() {return m_cpuDescrs.addr();}
    size_t   byteCount() {return m_gpuDescrs.size();};

    void asyncCpuToGpuCpy(cudaStream_t cuStream) {
        if(!m_valid) {
            throw std::runtime_error(std::string("Attempted copy of descriptor before initialization"));
        }
        CUDA_CHECK(cudaMemcpyAsync(m_gpuDescrs.addr(),
                                   m_cpuDescrs.addr(),
                                   m_gpuDescrs.size(),
                                   cudaMemcpyHostToDevice,
                                   cuStream));
    }

    std::array<uint8_t*, N_DESCRIPTORS> const& getCpuStartAddrs(void) const {
        if(!m_valid) {
            throw std::runtime_error(std::string("Attempted read of CPU descriptor start addresses before initialization"));
        }
        return m_cpuDescrStartAddr;
    };

    std::array<uint8_t*, N_DESCRIPTORS> const& getGpuStartAddrs(void) const {
        if(!m_valid) {
            throw std::runtime_error(std::string("Attempted read of GPU descriptor start addresses before initialization"));
        }
        return m_gpuDescrStartAddr;
    };

    std::array<size_t, N_DESCRIPTORS> const& getDescrSizeBytes(void) const {return m_descrSizeBytes;};
    size_t getSizeBytes(void) const {return m_totalDescrSizeBytes;};

    void displayDescrSizes(void) const {
        printf("%s size (bytes): %zu\n", m_name.c_str(), getSizeBytes());
        // Display component descriptor sizes
        for(int i = 0; i < N_DESCRIPTORS; ++i) {
            printf("%s entry[%i]: descrSizeBytes: %zu descrCpuAddr 0x%08lx descrGpuAddr 0x%08lx\n",
                 m_name.c_str(), i, m_descrSizeBytes[i], reinterpret_cast<uintptr_t>(m_cpuDescrStartAddr[i]), reinterpret_cast<uintptr_t>(m_gpuDescrStartAddr[i]));
        }
    }

private:
    std::string m_name;
    bool m_valid = false;
    size_t m_totalDescrSizeBytes = 0;
    std::array<size_t, N_DESCRIPTORS> m_descrSizeBytes = {{0}};
    std::array<uint8_t*, N_DESCRIPTORS> m_cpuDescrStartAddr = {{nullptr}};
    std::array<uint8_t*, N_DESCRIPTORS> m_gpuDescrStartAddr = {{nullptr}};
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_cpuDescrs;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_gpuDescrs;

    void setAlignedSizeBytes(std::array<size_t, N_DESCRIPTORS> const& descrSizeBytes, std::array<size_t, N_DESCRIPTORS> const& descrAlignBytes)
    {
        // allocation sizes computed so that the subsequent descriptor is self-aligned (starts at the alignment
        // required by the descriptor)
        auto alignUp = [](size_t size, size_t alignSize) {
            return ((size + (alignSize - 1)) / alignSize) * alignSize;
        };

        // Alignment required for next descriptor of non-zero size is padded to current descriptor also of non-zero size
        // Note: No alignment for the first descriptor as memory allocation is assumed to naturally align it
        std::array<size_t, N_DESCRIPTORS> nextDescrAlignBytes{};
        for(int32_t i = 0; i < N_DESCRIPTORS - 1;)
        {
            // Alignment padding added for descriptors with non-zero sizes only
            if(descrSizeBytes[i] > 0)
            {
                // find the next non-zero sized descriptor and use its alignment requirement to compute the padding for
                // current descriptor. If there is no non-zero sized descriptor found then there is no alignment needed.
                int32_t j = i + 1;
                while(j < N_DESCRIPTORS)
                {
                    if(descrSizeBytes[j] > 0)
                    {
                        nextDescrAlignBytes[i] = descrAlignBytes[j];
                        break;
                    }
                    ++j;
                }
                // j would be the next descriptor with non-zero size unless we reached the end
                i = j;
            }
            else
            {
                ++i;
            }
        }

        // for(int i = 0; i < N_DESCRIPTORS; ++i)
        // printf("descr[%d]: descrAlignBytes: %zu nextDescrAlignBytes: %zu\n", i, descrAlignBytes[i], nextDescrAlignBytes[i]);

        m_totalDescrSizeBytes = 0;
        for(uint32_t i = 0; i < N_DESCRIPTORS; ++i)
        {
            m_descrSizeBytes[i] = 0;
            if(descrSizeBytes[i] > 0)
            {
                size_t totalDescrSize        = m_totalDescrSizeBytes + descrSizeBytes[i];
                size_t alignedTotalDescrSize = (nextDescrAlignBytes[i] > 0) ? alignUp(totalDescrSize, nextDescrAlignBytes[i]) : totalDescrSize;
                // Padding to be added to the current descriptor so that the next descriptor starts per its alignment requirement
                size_t alignPadding = alignedTotalDescrSize - totalDescrSize;

                m_descrSizeBytes[i] = descrSizeBytes[i] + alignPadding;
                m_totalDescrSizeBytes += m_descrSizeBytes[i];
                // printf("%s descr[%d]: totalDescrSizeBytes: %zu nextDescrAlignBytes: %zu alignPadding: %zu\n", m_name.c_str(), i, m_totalDescrSizeBytes, nextDescrAlignBytes[i], alignPadding);
            }
        }
    }
};


////////////////////////////////////////////////////////////////////////////
// rng_deleter
struct rng_deleter
{
    typedef cuphyRNG_t ptr_t;
    void operator()(ptr_t p) const
    {
        cuphyDestroyRandomNumberGenerator(p);
    }

};

////////////////////////////////////////////////////////////////////////////
// unique_rng_ptr
using unique_rng_ptr = std::unique_ptr<cuphyRNG, rng_deleter>;

////////////////////////////////////////////////////////////////////////////
// rng
class rng
{
public:
    //----------------------------------------------------------------------
    // rng()
    rng(unsigned long long seed = 0, unsigned int flags = 0, cudaStream_t strm = 0)
    {
        cuphyRNG_t    r = nullptr;
        cuphyStatus_t s = cuphyCreateRandomNumberGenerator(&r,
                                                           seed,
                                                           flags,
                                                           strm);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_fn_exception(s, "cuphyCreateRandomNumberGenerator()");
        }
        rng_.reset(r);
    }
    //----------------------------------------------------------------------
    // normal()
    template <typename T, typename TVal>
    void normal(T& t, TVal mean, TVal stddev, cudaStream_t strm = 0)
    {
        variant m(mean);
        variant s(stddev);
        cuphyStatus_t r = cuphyRandomNormal(rng_.get(),
                                            t.desc().handle(),
                                            t.addr(),
                                            &m,
                                            &s,
                                            strm);
        if(CUPHY_STATUS_SUCCESS != r)
        {
            throw cuphy::cuphy_fn_exception(r, "cuphyRandomNormal()");
        }
    }
    //----------------------------------------------------------------------
    // uniform()
    template <typename T, typename TVal>
    void uniform(T& t, TVal min_val, TVal max_val, cudaStream_t strm = 0)
    {
        variant min_v(min_val);
        variant max_v(max_val);
        cuphyStatus_t r = cuphyRandomUniform(rng_.get(),
                                             t.desc().handle(),
                                             t.addr(),
                                             &min_v,
                                             &max_v,
                                             strm);
        if(CUPHY_STATUS_SUCCESS != r)
        {
            throw cuphy::cuphy_fn_exception(r, "cuphyRandomUniform()");
        }
    }
private:
    //----------------------------------------------------------------------
    // Data
    unique_rng_ptr rng_;
};

template <class TDst, class TSrc>
void modulate_symbol(TDst& dstSym, TSrc& srcBits, int log2_qam, cudaStream_t strm = 0)
{
    cuphyStatus_t s = cuphyModulateSymbol(dstSym.desc().handle(),
                                          dstSym.addr(),
                                          srcBits.desc().handle(),
                                          srcBits.addr(),
                                          log2_qam,
                                          strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy::cuphy_fn_exception(s, "cuphyModulateSymbol()");
    }
}

template <class TDst, typename TValue>
void tensor_fill(TDst& tDst, TValue v, cudaStream_t strm = 0)
{
    variant       value(v);
    cuphyStatus_t s = cuphyFillTensor(tDst.desc().handle(),
                                      tDst.addr(),
                                      &value,
                                      strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyFillTensor()");
    }
}

template <class TDst, class TSrc>
void tensor_convert(TDst& tDst, TSrc& tSrc, cudaStream_t strm = 0)
{
    cuphyStatus_t s = cuphyConvertTensor(tDst.desc().handle(),
                                         tDst.addr(),
                                         tSrc.desc().handle(),
                                         tSrc.addr(),
                                         strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyConvertTensor()");
    }
}

template <class TDst>
void tensor_convert(TDst&              tDst,
                    const tensor_desc& src_desc,
                    const void*        src_addr,
                    cudaStream_t       strm = 0)
{
    cuphyStatus_t s = cuphyConvertTensor(tDst.desc().handle(),
                                         tDst.addr(),
                                         src_desc.handle(),
                                         src_addr,
                                         strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyConvertTensor()");
    }
}

////////////////////////////////////////////////////////////////////////
// tensor_copy_range()
// TSrc input must have a typedef for tensor_ref_t, and support the
// subset() member function. Examples of this are the tensor and
// typed_tensor classes.
// The destination tensor must have a size that matches the subset
// indicated by the index group.
template <class TDst, class TSrc>
void tensor_copy_range(TDst&              tDst,     // destination tensor
                       TSrc&              tSrc,     // source tensor
                       const index_group& grp,      // range of source to copy
                       cudaStream_t       strm = 0)
{
    typedef typename TSrc::tensor_ref_t tensor_ref_t;
    // Generate a (temporary) reference to a subset of the source tensor.
    tensor_ref_t  src_subset = tSrc.subset(grp);

    cuphyStatus_t s          = cuphyConvertTensor(tDst.desc().handle(),
                                                              tDst.addr(),
                                                              src_subset.desc().handle(),
                                                              src_subset.addr(),
                                                              strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyConvertTensor()");
    }
}

////////////////////////////////////////////////////////////////////////
// tensor_xor()
template <class TDst, class TSrcA, class TSrcB>
void tensor_xor(TDst& dst, TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
{
    cuphyStatus_t s = cuphyTensorElementWiseOperation(dst.desc().handle(),    // tDst
                                                      dst.addr(),             // pDst
                                                      a.desc().handle(),      // tSrcA
                                                      a.addr(),               // pSrcA
                                                      nullptr,                // alpha
                                                      b.desc().handle(),      // tSrcB
                                                      b.addr(),               // pSrcB
                                                      nullptr,                // beta
                                                      CUPHY_ELEMWISE_BIT_XOR, // operation
                                                      strm);                  // stream
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
    }
}

////////////////////////////////////////////////////////////////////////
// tensor_reduction_sum()
template <class TDst, class TSrc >
void tensor_reduction_sum(TDst& v, TSrc& src, int dim = 0, cudaStream_t strm = 0)
{
    cuphyStatus_t s = cuphyTensorReduction(v.desc().handle(),
                                           v.addr(),
                                           src.desc().handle(),
                                           src.addr(),
                                           CUPHY_REDUCTION_SUM,
                                           dim,
                                           0,
                                           nullptr,
                                           strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyTensorReduction()");
    }
}

////////////////////////////////////////////////////////////////////////
// tensor_sum()
template <class TDst, class TSrcA, class TSrcB>
void tensor_sum(TDst& dst, TSrcA& a, TSrcB& b, cudaStream_t strm = 0)
{
    cuphyStatus_t s = cuphyTensorElementWiseOperation(dst.desc().handle(), // tDst
                                                      dst.addr(),          // pDst
                                                      a.desc().handle(),   // tSrcA
                                                      a.addr(),            // pSrcA
                                                      nullptr,             // alpha
                                                      b.desc().handle(),   // tSrcB
                                                      b.addr(),            // pSrcB
                                                      nullptr,             // beta
                                                      CUPHY_ELEMWISE_ADD,  // operation
                                                      strm);               // stream
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyTensorElementWiseOperation()");
    }
}

} // namespace cuphy

CUresult inline
launch_kernel(const CUDA_KERNEL_NODE_PARAMS& kernelNodeParams, cudaStream_t strm)
{
    CUresult e = cuLaunchKernel(kernelNodeParams.func,
                                kernelNodeParams.gridDimX,
                                kernelNodeParams.gridDimY,
                                kernelNodeParams.gridDimZ,
                                kernelNodeParams.blockDimX,
                                kernelNodeParams.blockDimY,
                                kernelNodeParams.blockDimZ,
                                kernelNodeParams.sharedMemBytes,
                                static_cast<CUstream>(strm),
                                kernelNodeParams.kernelParams,
                                kernelNodeParams.extra);

    return e;
}


inline double computeSnrComponents(double &avgSignalEnergy, double&avgErrorEnergy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_ref)
{
    const auto& refDim =  T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim          =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i <  T_cuphy.layout().rank(); ++i){
        dim[i] = std::min(dim[i], refDim[i]);
        tensorSize = tensorSize*dim[i];
    }

    avgSignalEnergy = 0;
    avgErrorEnergy  = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    float2 refScaler = T_ref(idx0, idx1, idx2, idx3);
                    float2 cuphyScaler  = T_cuphy(idx0, idx1, idx2, idx3);

                    avgSignalEnergy += (pow(abs(refScaler.x),2) + pow(abs(refScaler.y),2)) / static_cast<double>(tensorSize);
                    avgErrorEnergy  += (pow(abs(refScaler.x - cuphyScaler.x),2) + pow(abs(refScaler.y - cuphyScaler.y),2)) / static_cast<double>(tensorSize);
                }
            }
        }
    }

    double snr = 10*log10(avgSignalEnergy / avgErrorEnergy);
    return snr;

}

inline double computeSnrComponentsHalf(double &avgSignalEnergy, double&avgErrorEnergy, cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_ref)
{
    const auto& refDim =  T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim          =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i <  T_cuphy.layout().rank(); ++i){
        dim[i] = std::min(dim[i], refDim[i]);
        tensorSize = tensorSize*dim[i];
    }

    avgSignalEnergy = 0;
    avgErrorEnergy  = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    float2 refScaler   = T_ref(idx0, idx1, idx2, idx3);
                    float2 cuphyScaler = __half22float2(T_cuphy(idx0, idx1, idx2, idx3));

                    avgSignalEnergy += (pow(abs(refScaler.x),2) + pow(abs(refScaler.y),2)) / static_cast<double>(tensorSize);
                    avgErrorEnergy  += (pow(abs(refScaler.x - cuphyScaler.x),2) + pow(abs(refScaler.y - cuphyScaler.y),2)) / static_cast<double>(tensorSize);
                }
            }
        }
    }

    double snr = 10*log10(avgSignalEnergy / avgErrorEnergy);
    return snr;

}

inline double computeSnrComponentsShort(double &avgSignalEnergy, double&avgErrorEnergy, cuphy::typed_tensor<CUPHY_C_16I, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32I, cuphy::pinned_alloc>& T_ref)
{
    const auto& refDim =  T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim          =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i <  T_cuphy.layout().rank(); ++i){
        dim[i] = std::min(dim[i], refDim[i]);
        tensorSize = tensorSize*dim[i];
    }

    avgSignalEnergy = 0;
    avgErrorEnergy  = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    int2 refScaler   = T_ref(idx0, idx1, idx2, idx3);
                    short2 cuphyScaler = T_cuphy(idx0, idx1, idx2, idx3);

                    avgSignalEnergy += (pow(abs(refScaler.x*1.0f),2) + pow(abs(refScaler.y*1.0f),2)) / static_cast<double>(tensorSize);
                    avgErrorEnergy  += (pow(abs(refScaler.x*1.0f - cuphyScaler.x*1.0f),2) + pow(abs(refScaler.y*1.0f - cuphyScaler.y*1.0f),2)) / static_cast<double>(tensorSize);
                }
            }
        }
    }

    double snr = 10*log10(avgSignalEnergy / avgErrorEnergy);
    return snr;

}

inline double computeChEqCoefSnrComponents(double &avgSignalEnergy, double&avgErrorEnergy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_ref)
{
    const auto& refDim = T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim = T_cuphy.layout().dimensions();
    
    static constexpr int CUPHY_EXPECTED_RANK = 5;
    static constexpr int REF_EXPECTED_RANK = 4;

    
    if((T_cuphy.layout().rank()!=CUPHY_EXPECTED_RANK)||(T_ref.layout().rank()!=REF_EXPECTED_RANK))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "computeChEqCoefSnrComponents() error for wrong tensor ranks: cuPHY {} ref {}", T_cuphy.layout().rank(), T_ref.layout().rank());
        return 0.0f;
    }

    int tensorSize = 1;
    for(int i = 0; i < T_cuphy.layout().rank(); ++i){
        tensorSize = tensorSize*dim[i];
    }

    avgSignalEnergy = 0;
    avgErrorEnergy  = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    for(int idx4 = 0; idx4 < dim[4]; ++idx4)  
                    {
                        if((idx4<refDim[0])&&(idx2<refDim[1])&&(idx0<refDim[2])&&((idx3*12+idx1)<refDim[3]))
                        {
                            float2 refScaler = T_ref(idx4, idx2, idx0, idx3*CUPHY_N_TONES_PER_PRB+idx1);
                            float2 cuphyScaler  = T_cuphy(idx0, idx1, idx2, idx3, idx4);

                            avgSignalEnergy += (pow(abs(refScaler.x),2) + pow(abs(refScaler.y),2)) / static_cast<double>(tensorSize);
                            avgErrorEnergy  += (pow(abs(refScaler.x - cuphyScaler.x),2) + pow(abs(refScaler.y - cuphyScaler.y),2)) / static_cast<double>(tensorSize);
                        }
                        else
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "computeChEqCoefSnrComponents() error for out-of-bounds on reference tensor");
                            return 0.0f;
                        }
                    }
                }
            }
        }
    }

    return 10*log10(avgSignalEnergy / avgErrorEnergy);
}

inline double computeChEqLLRSnrComponents(double &avgSignalEnergy, double&avgErrorEnergy, cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>& T_ref)
{
    const auto& refDim = T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim = T_cuphy.layout().dimensions();
    
    static constexpr int CUPHY_EXPECTED_RANK = 4;
    static constexpr int REF_EXPECTED_RANK = 4;

    
    if((T_cuphy.layout().rank()!=CUPHY_EXPECTED_RANK)||(T_ref.layout().rank()!=REF_EXPECTED_RANK))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "computeChEqLLRSnrComponents() error for wrong tensor ranks: cuPHY {} ref {}", T_cuphy.layout().rank(), T_ref.layout().rank());
        return 0.0f;
    }

    int tensorSize = 1;
    for(int i = 0; i < T_cuphy.layout().rank(); ++i){
        tensorSize = tensorSize*dim[i];
    }

    avgSignalEnergy = 0.0;
    avgErrorEnergy  = 0.0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    if((idx0<refDim[0])&&(idx1<refDim[1])&&(idx2<refDim[2])&&(idx3<refDim[3]))
                    {
                        double refScaler = type_convert<double>(T_ref(idx0, idx1, idx2, idx3));
                        double cuphyScaler  = type_convert<double>(T_cuphy(idx0, idx1, idx2, idx3));
                        
                        avgSignalEnergy += (refScaler * refScaler); 
                        avgErrorEnergy  += (refScaler - cuphyScaler) * (refScaler - cuphyScaler); 
                    }
                    else
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "computeChEqLLRSnrComponents() error for out-of-bounds on reference tensor");
                        return 0.0;
                    }
                }
            }
        }
    }

    return (avgErrorEnergy == 0.0) ? 1000.0 : (10.0*log10(avgSignalEnergy / avgErrorEnergy));
}

//-------------------------------------------------------------------------------------
// function compares two tensors and computes SNR
inline double computeSnr(cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_ref)
{
    double avgSignalEnergy = 0;
    double avgErrorEnergy  = 0;
    computeSnrComponents(avgSignalEnergy, avgErrorEnergy, T_cuphy, T_ref);
    return 10*log10(avgSignalEnergy / avgErrorEnergy);
}

template<cuphyDataType_t cuphy_type, cuphyDataType_t ref_type>
inline double computeSnr(cuphy::typed_tensor<cuphy_type, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<ref_type, cuphy::pinned_alloc>& T_ref, bool verbose = false)
{
    const auto& refDim =  T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim          =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i <  T_cuphy.layout().rank(); ++i){
        dim[i] = std::min(dim[i], refDim[i]);
        if(verbose) printf("dim[%d]: resDim %d refDim %d\n", i, dim[i], refDim[i]);
        tensorSize = tensorSize*dim[i];
    }

    double avgSignalEnergy = 0;
    double avgErrorEnergy  = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    float refScaler   = type_convert<float>(T_ref(idx0, idx1, idx2, idx3));
                    float cuphyScaler = type_convert<float>(T_cuphy(idx0, idx1, idx2, idx3));

                    if(verbose) printf("[%d][%d][%d][%d] refScaler %f cuphyScaler %f\n", idx0, idx1, idx2, idx3, refScaler, cuphyScaler);

                    avgSignalEnergy += (refScaler*refScaler) / static_cast<double>(tensorSize);
                    avgErrorEnergy  += ((refScaler - cuphyScaler)*(refScaler - cuphyScaler)) / static_cast<double>(tensorSize);
                }
            }
        }
    }

    double snr = 10*log10(avgSignalEnergy / avgErrorEnergy);
    return snr;

}

inline double computeChEqCoefSnr(cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>& T_ref)
{
    double avgSignalEnergy = 0;
    double avgErrorEnergy  = 0;
    return computeChEqCoefSnrComponents(avgSignalEnergy, avgErrorEnergy, T_cuphy, T_ref);
}

inline double computeChEqLLRSnr(cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>& T_ref)
{
    double avgSignalEnergy = 0;
    double avgErrorEnergy  = 0;
    return computeChEqLLRSnrComponents(avgSignalEnergy, avgErrorEnergy, T_cuphy, T_ref);
}

template<cuphyDataType_t cuphy_type, cuphyDataType_t ref_type>
inline double computeDiff(cuphy::typed_tensor<cuphy_type, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<ref_type, cuphy::pinned_alloc>& T_ref, bool verbose = false)
{
    const auto& refDim =  T_ref.layout().dimensions();
    cuphy::vec<int, CUPHY_DIM_MAX> dim          =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i <  T_cuphy.layout().rank(); ++i){
        dim[i] = std::min(dim[i], refDim[i]);
        if(verbose) printf("dim[%d]: resDim %d refDim %d\n", i, dim[i], refDim[i]);
        tensorSize = tensorSize*dim[i];
    }

    double avgErr = 0;

    for(int idx0 = 0; idx0 < dim[0]; ++idx0){
        for(int idx1 = 0; idx1 < dim[1]; ++idx1){
            for(int idx2 = 0; idx2 < dim[2]; ++idx2){
                for(int idx3 = 0; idx3 < dim[3]; ++idx3){
                    float refScaler   = type_convert<float>(T_ref(idx0, idx1, idx2, idx3));
                    float cuphyScaler = type_convert<float>(T_cuphy(idx0, idx1, idx2, idx3));

                    if(verbose) printf("[%d][%d][%d][%d] refScaler %f cuphyScaler %f\n", idx0, idx1, idx2, idx3, refScaler, cuphyScaler);

                    avgErr += abs(refScaler - cuphyScaler);
                }
            }
        }
    }

    return avgErr/static_cast<double>(tensorSize);

}

template<cuphyDataType_t cuphy_type, cuphyDataType_t ref_type>
inline bool compareTensor(cuphy::typed_tensor<cuphy_type, cuphy::pinned_alloc>& T_cuphy, cuphy::typed_tensor<ref_type, cuphy::pinned_alloc>& T_ref, bool verbose = false)
{
    bool mismatch=false;
    const auto& refDim =  T_ref.layout().dimensions();
    const cuphy::vec<int, CUPHY_DIM_MAX> dim    =  T_cuphy.layout().dimensions();

    int tensorSize = 1;
    for(int i = 0; i < T_cuphy.layout().rank(); ++i)
    {
        if(verbose) printf("dim[%d]: resDim %d refDim %d\n", i, dim[i], refDim[i]);
        tensorSize *= dim[i];
        mismatch   |= (dim[i] != refDim[i]);
    }
    if(mismatch) return false;

    for(int idx = 0; idx < tensorSize; ++idx)
    {
        uint32_t refVal   = type_convert<uint32_t>(T_ref.addr()[idx]);
        uint32_t cuphyVal = type_convert<uint32_t>(T_cuphy.addr()[idx]);
        if(cuphyVal != refVal)
        {
            mismatch = true;
            if(verbose)
            {
                printf("[%d] ref %d cuphy %d\n",idx,refVal,cuphyVal);
            } else {
                return false;
            }
        }
    }
    return !mismatch;
}

////////////////////////////////////////////////////////////////////////////
// Timer

template <typename T, typename unit>
using duration  = std::chrono::duration<T, unit>;
using t_ns = std::chrono::nanoseconds;
using t_us = std::chrono::microseconds;

using HighResClock     = std::chrono::high_resolution_clock;
using HighResTimePoint = std::chrono::time_point<HighResClock>;


class Timer
{
public:
    Timer();
    ~Timer();

    static auto now()
    {
        return HighResClock::now();
    }

    static float elapsedTime(HighResTimePoint start, HighResTimePoint stop)
    {
        return std::chrono::duration<float, std::micro>(stop - start).count();
    }

    static t_ns nowNs()
    {
        return std::chrono::system_clock::now().time_since_epoch();
    }

    static t_us NsToUs(t_ns time)
    {
        return std::chrono::duration_cast<t_us>(time);
    }

};

class cuphyNvlogFmtHelper
{
public:
    cuphyNvlogFmtHelper(std::string nvlog_name="nvlog.log")
    {
        // Relative path from binary to default nvlog_config.yaml
        relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, nvlog_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
    }

    ~cuphyNvlogFmtHelper()
    {
        nvlog_fmtlog_close(log_thread_id);
    }


private:
    char nvlog_yaml_file[1024];
    std::string relative_path;
    pthread_t log_thread_id;
};

enum class batchedMemcpySrcHint
{
    srcIsHost,
    srcIsDevice
};

enum class batchedMemcpyDstHint
{
    dstIsHost,
    dstIsDevice
};


class cuphyBatchedMemcpyHelper
{
public:
    // Current implementation assumes that all copies have the same copy direction (known at ctor time) and a single attribute
    // Max possible number of copies is also know at this point and relevant vectors are resized accordingly.
    cuphyBatchedMemcpyHelper(size_t  max_copies, batchedMemcpySrcHint src_hint, batchedMemcpyDstHint dst_hint, bool batched_memcpy)
        : use_batched_memcpy((CUDA_VERSION >= 12080) && batched_memcpy), max_memory_copies_count(use_batched_memcpy ? max_copies : 0)
    {

        if (use_batched_memcpy) {
            memcpy_dst_addr.resize(max_memory_copies_count);
            memcpy_src_addr.resize(max_memory_copies_count);
            memcpy_size.resize(max_memory_copies_count);

#if CUDA_VERSION >= 12080
            // Could alternatively pass a cudaMemcpyKind here and infer the src and dst hints
            batched_memcpy_attributes = {CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
                                     {(src_hint == batchedMemcpySrcHint::srcIsHost) ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE, 0},
                                     {(dst_hint == batchedMemcpyDstHint::dstIsHost) ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE, 0}, // Reminder other options are CU_MEM_LOCATION_TYPE_HOST_NUM, CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT
                                     CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE};
#endif
       }

    }

    /* Launch either a cudaMemcpyAsync (fallback path if !use_batched_memcpy) or update parameters needed for next launchBatchedMemcpy
       If an error occurs (e.g., cudaMemcpyAsync returns an error or one attempts to include more copies in the batch than the max supported),
       an error count is incremented. A non-zero error count will then cause the launchBatchedMemcpy to return an error.
    */
    void updateMemcpy(void* dst_addr, void* src_addr, size_t copy_size, cudaMemcpyKind copy_kind, cudaStream_t& strm)
    {
        if (!use_batched_memcpy) {
            // If this is an older CUDA version, or batched memcpy is disabled, then calls to launchBatchedMemcpy will be a no-op and the updateMemcpy call itself will perform the cudaMemcpyAsync.
            // We could alternatively keep updateMemcpy method unchanged and loop over the array in launchBatchedMempcy, but that could delay a bit when individual memory copies take place.
            cudaError_t e = cudaMemcpyAsync(dst_addr, src_addr, copy_size, copy_kind, strm);
            if (e != cudaSuccess)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "cudaMemcpyAsync in updateMemcpy() returned {}",  (int)e);
                // No exception is thrown, but error_count is updated, which will then cause launchBatchedMemcpy to return an error.
                error_count += 1;
            }
        } else {
            if (copy_index >= max_memory_copies_count)
            {
                error_count += 1;
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "Number of batches for batched copy is {}, but current max. specified is {}.", copy_index, max_memory_copies_count);
                // No exception is thrown, but error_count is updated, which will then cause launchBatchedMemcpy to return an error.
                return;
            }

            memcpy_dst_addr[copy_index] = dst_addr;
            memcpy_src_addr[copy_index] = src_addr;
            memcpy_size[copy_index] = copy_size;
            copy_index += 1;
            memcpy_count = copy_index;
        }
    }


    /* Calling launchBatchedMemcpy will:
       - be a no-op and return CUPHY_STATUS_SUCCESS, if !useBatchedMemcpy()
       - launch the batched memcpy and return CUPHY_STATUS_SUCCESS, if the cuMemcpyBatchAsync API call was successful, if useBatchedMemcpy()
       - return CUPHY_STATUS_INVALID_ARGUMENT if a copy is attempted with 0 number of copies or the cuMemcpyBatchAsync returns an error or there is a non-zero error_count
    */
    [[nodiscard]] cuphyStatus_t launchBatchedMemcpy(cudaStream_t& strm)
    {
        if (error_count != 0)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error! There were {} errors during the batch of updateMemcpy calls preceding launchBatchedMemcpy()!", error_count);
            reset();
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        // This will be a no-op if an older CUDA version is used or batched memcpy feature is disabled.
        if (!use_batched_memcpy)
        {
            reset();
            return CUPHY_STATUS_SUCCESS;
        }

        // The memcpy_count check is not strictly needed, as this would also be caught when cuMemcpyBatchAsync is attempted. Can potentially remove.
        if (memcpy_count == 0)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Should not call cuMemcpyBatchAsync with a zero number of copies!");
            reset();
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
        MemtraceDisableScope md;
#if CUDA_VERSION >= 12080
        // updateMemcpy must have been called before, as memcpy_count cannot be zero here.
        const auto batched_memcpy_attributes_count = 1;
#if CUDA_VERSION >= 13000
        CUresult batched_memcpy_status = cuMemcpyBatchAsync((CUdeviceptr*)memcpy_dst_addr.data(), (CUdeviceptr*)memcpy_src_addr.data(), memcpy_size.data(), memcpy_count, &batched_memcpy_attributes, &batched_memcpy_attr_index, batched_memcpy_attributes_count, strm);
        if (batched_memcpy_status != CUDA_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error! cuMemcpyBatchAsync returned {}.", batched_memcpy_status);
            status = CUPHY_STATUS_INVALID_ARGUMENT;
        }
#else
        size_t fail_idx = 0;
        CUresult batched_memcpy_status = cuMemcpyBatchAsync((CUdeviceptr*)memcpy_dst_addr.data(), (CUdeviceptr*)memcpy_src_addr.data(), memcpy_size.data(), memcpy_count, &batched_memcpy_attributes, &batched_memcpy_attr_index, batched_memcpy_attributes_count, &fail_idx, strm);
        if (batched_memcpy_status != CUDA_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error! cuMemcpyBatchAsync returned {} and failed index {}.", batched_memcpy_status, fail_idx);
            status = CUPHY_STATUS_INVALID_ARGUMENT;
        }
#endif
#endif
        reset();
        return status;
    }

    [[nodiscard]] bool useBatchedMemcpy()
    {
        return use_batched_memcpy;
    }

    [[nodiscard]] size_t getMaxMemcopiesCount()
    {
        return max_memory_copies_count;
    }

    // Reminder: it is the user's responsibility to call reset() before each batch of updateMemcpy calls.
    void reset()
    {
        copy_index   = 0;
        memcpy_count = 0;
        error_count  = 0;
    }

    [[nodiscard]] size_t getMemcpyCount()
    {
       return memcpy_count;
    }

    [[nodiscard]] int getErrorCount()
    {
       return error_count;
    }

private:
    const bool use_batched_memcpy; // control whether batched async. memcpy will be used or if we fall back to regular async. memcpy
    const size_t max_memory_copies_count; // max. size of batched_memcpy_dst_addr, src_addr and batched_memcpy_count vectors; max. number of memory copies in the cuMemcpyBatchAsync call.
    size_t batched_memcpy_attr_index = 0; // current class implementation assumes a batch will contain a single attribute
    std::vector<void*> memcpy_dst_addr; // vector of dst addresses
    std::vector<void*> memcpy_src_addr; // vector of src addresses
    std::vector<size_t> memcpy_size; // vector of copy sizes in bytes
    size_t memcpy_count = 0; // number of copies in batched memcpy
    int copy_index = 0;  // running counter of copies in batched memcpy
    int error_count = 0; // number of errors encountered during updateMemcpy() calls before a launchBatchedMemcpy() or reset() was called.
#if CUDA_VERSION >= 12080
    CUmemcpyAttributes batched_memcpy_attributes; // batched memcpy attributes
#endif
};

#endif // !defined(CUPHY_HPP_INCLUDED_)
