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

#include "empty_kernels.hpp"
//#define DBG_COND_HANDLE_OR_DGL
#ifdef DBG_COND_HANDLE_OR_DGL
#include <iostream>
#endif

#include "error_correction/ldpc.hpp"
#include "crc_encode.hpp"
#include "rm_decoder.hpp"
#include "cfo_ta_est.hpp"
#include "pusch_noise_intf_est.hpp"
#include "channel_eq.hpp"
#include "crc/crc_decode.hpp"

// empty kernels used in CUDA graphs
__global__ void graphs_empty_kernel()
{
}

__global__ void graphs_empty_kernel_1_ptr_arg(void* ptr1)
{
}

__global__ void graphs_empty_kernel_2_ptr_arg(void* ptr1, void* ptr2)
{
}

__global__ void graphs_empty_kernel_3_ptr_arg(void* ptr1, void* ptr2, void* ptr3)
{
}

__global__ void graphs_empty_kernel_4_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4)
{
}

__global__ void graphs_empty_kernel_5_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5)
{
}

__global__ void graphs_empty_kernel_6_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6)
{
}

__global__ void graphs_empty_kernel_7_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7)
{
}

__global__ void graphs_empty_kernel_8_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7, void* ptr8)
{
}

__global__ void graphs_empty_kernel_9_ptr_arg(void* ptr1, void* ptr2, void* ptr3, void* ptr4, void* ptr5, void* ptr6, void* ptr7, void* ptr8, void* ptr9)
{
}

__global__ void graphs_empty_kernel_1_grid_constant_arg_32B(const __grid_constant__ struct testDescr_sz<32> desc)
{
}


__global__ void graphs_empty_kernel_1_grid_constant_arg_48B(const __grid_constant__ struct testDescr_sz<48> desc)
{
}

template <typename DynDescr>
__global__ void graphs_empty_kernel_1_grid_constant_arg_T(const __grid_constant__ DynDescr desc)
{
}

template <typename DynDescr>
__global__ void graphs_empty_kernel_1_ptr_arg_1_grid_constant_arg_T(void *, const __grid_constant__ DynDescr desc)
{
}

#if CUDA_VERSION >= 12040
// Helper kernel used to set the  CUgraphConditionalHandle condition depending on *early_exit value.
__global__ void kCondSetHandle(uint8_t* early_exit, CUgraphConditionalHandle handle)
{
    bool cond_value = (*early_exit == 0);
    cudaGraphSetConditional(handle, cond_value);
#ifdef DBG_COND_HANDLE_OR_DGL
    printf("set handle to %d from kCondSetHandle kernel\n", cond_value);
#endif
}
#endif

// Helper kernel used to launch a device graph depending on the *early_exit value.
__global__ void kDeviceGraphLauncher(uint8_t* early_exit, CUgraphExec device_graph_exec)
{
    // Ensure only a single thread launches the device graph, if launch_device_graph is set
    if (
        (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0) && \
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {

        if (*early_exit == 0) {
            // currently CUgraphExec and cudaGraphExec can be used interchangeably.
            // Note: one could also use cudaStreamGraphFireAndForget instead of cudaStreamGraphTailLaunch (tail launch).
            // The graph node with this kernel does not have any children, so expect results to be comparable.
	    cudaError_t e = cudaGraphLaunch(device_graph_exec, cudaStreamGraphTailLaunch);
	    if (e != cudaSuccess) {
                // printf("error from kDeviceGraphLauncher for tailLaunch\n"); // should be commented out in real time path
                // FIXME no handling of errors right now. It'd be a silent failure that will result in errors later,
                // as part of the pipeline won't execute.
                // Could write to a memory location, but would need to be part of cuPHY API so it's visible to cuPHY-CP.
     	    }
#ifdef DBG_COND_HANDLE_OR_DGL
	    printf("launching device graph from kernel returned %d\n", e);
#endif
        } else {
#ifdef DBG_COND_HANDLE_OR_DGL
	    printf("should not launch device graph from kernel\n");
#endif
        }
    }
}

// The internalCuphy* functions are included here rather than in cuphy.cpp so that the kernel references
// (via cudaGetFuncBySymbol) are in the same translation unit as their kernel definitions. This is necessary
// to avoid potential symbol resolution issues or linker failures.
cuphyStatus_t CUPHYWINAPI internalCuphySetGenericEmptyKernelNodeGridConstantParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, int ptrArgsCnt, uint16_t descr_size)
{
    if(pNodeParams == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    void *func = nullptr;
    // Currently, sizeof(pusch_noise_intf_est::puschRxNoiseIntfEstDynDescr_t) == sizeof(cfo_ta_est::puschRxCfoTaEstDynDescr_t),
    // so we can only include one case. If one of them is changed, then this function must be updated to include both cases
    // (assuming that both sizes remain unique)
    static_assert(sizeof(cfo_ta_est::puschRxCfoTaEstDynDescr_t) == sizeof(rmDecoderDynDescr_t),
        "sizeof(cfo_ta_est::puschRxCfoTaEstDynDescr_t) == sizeof(rmDecoderDynDescr_t)");
    if (ptrArgsCnt == 0) {
        switch (descr_size) {
            case sizeof(ldpcEncodeDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<ldpcEncodeDescr_t>);
                break;
            case sizeof(crcEncodeDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<crcEncodeDescr_t>);
                break;
            case sizeof(rmDecoderDynDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<rmDecoderDynDescr_t>);
                break;
//            case sizeof(cfo_ta_est::puschRxCfoTaEstDynDescr_t):
//                func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<cfo_ta_est::puschRxCfoTaEstDynDescr_t>);
//                break;
            // See above static assertion and comment.
            case sizeof(pusch_noise_intf_est::puschRxNoiseIntfEstDynDescr_t):
                 func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<pusch_noise_intf_est::puschRxNoiseIntfEstDynDescr_t>);
                 break;
            case sizeof(puschRxCrcDecodeDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_grid_constant_arg_T<puschRxCrcDecodeDescr_t>);
                break;
            default:
                return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    } else if (ptrArgsCnt == 1) {
        switch (descr_size) {
            case sizeof(channel_eq::puschRxChEqCoefCompDynDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_ptr_arg_1_grid_constant_arg_T<channel_eq::puschRxChEqCoefCompDynDescr_t>);
                break;
            case sizeof(channel_eq::puschRxChEqSoftDemapDynDescr_t):
                func = reinterpret_cast<void*>(graphs_empty_kernel_1_ptr_arg_1_grid_constant_arg_T<channel_eq::puschRxChEqSoftDemapDynDescr_t>);
                break;
            default:
                return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    } else {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    MemtraceDisableScope md;
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, func);
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}

// See internalCuphySetGenericEmptyKernelNodeGridConstantParams for comments.
cuphyStatus_t CUPHYWINAPI internalCuphySetGenericEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, int ptrArgsCnt, void** pKernelParams)
{
    if((pNodeParams == nullptr) || (ptrArgsCnt < 0) || (ptrArgsCnt > 9))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    void* kernel_function_ptr[10] = {reinterpret_cast<void*>(graphs_empty_kernel),
                                     reinterpret_cast<void*>(graphs_empty_kernel_1_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_2_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_3_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_4_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_5_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_6_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_7_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_8_ptr_arg),
                                     reinterpret_cast<void*>(graphs_empty_kernel_9_ptr_arg)};

    MemtraceDisableScope md;
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, kernel_function_ptr[ptrArgsCnt]);
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;
    pNodeParams->kern           = nullptr;
    pNodeParams->ctx            = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}

// See internalCuphySetGenericEmptyKernelNodeGridConstantParams for comments.
cuphyStatus_t CUPHYWINAPI internalCuphySetEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams)
{
    if(pNodeParams == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    MemtraceDisableScope md;
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, reinterpret_cast<void*>(graphs_empty_kernel));
    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = nullptr;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;
    pNodeParams->kern           = nullptr;
    pNodeParams->ctx            = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}

// See internalCuphySetGenericEmptyKernelNodeGridConstantParams for comments.
cuphyStatus_t CUPHYWINAPI internalCuphySetWorkCancelKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, uint8_t device_graph_launch)
{
    if(pNodeParams == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    MemtraceDisableScope md;
#if CUDA_VERSION >= 12040
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, (device_graph_launch == 0) ? \
                                  reinterpret_cast<void*>(kCondSetHandle) : reinterpret_cast<void*>(kDeviceGraphLauncher));
#else
    if(device_graph_launch == 0)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "Cannot use conditional graph nodes with CUDA_VERSION {}; they require CUDA 12.4. Falling back to device graphs.", CUDA_VERSION);
    }
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, reinterpret_cast<void*>(kDeviceGraphLauncher));
#endif

    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 1;
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}


__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}

__global__ void delay_kernel_us(uint32_t delay_us)
{
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    // 64-bit timer has a long range so skipping wrap around check
    while(__globaltimer() < end_time)
    {
    };
}

cuphyStatus_t CUPHYWINAPI internalCuphySetDelayKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams)
{

    if((pNodeParams == nullptr) || (pKernelParams == nullptr) || (pKernelParams[0] == nullptr))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    MemtraceDisableScope md;
    cudaError_t e               = cudaGetFuncBySymbol(&pNodeParams->func, reinterpret_cast<void*>(delay_kernel_us));

    pNodeParams->gridDimX       = 1;
    pNodeParams->gridDimY       = 1;
    pNodeParams->gridDimZ       = 1;
    pNodeParams->blockDimX      = 32; // 1 (single thread) would be fine too
    pNodeParams->blockDimY      = 1;
    pNodeParams->blockDimZ      = 1;
    pNodeParams->kernelParams   = pKernelParams;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->extra          = nullptr;
    pNodeParams->kern           = nullptr;
    pNodeParams->ctx            = nullptr;

    return (cudaSuccess != e) ? CUPHY_STATUS_INTERNAL_ERROR : CUPHY_STATUS_SUCCESS;
}
