/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_utils.h"
// #include "nvlog.hpp"

#define TAG "NVIPC.CUDAUTILS"

inline cudaError __checkLastCudaError(const char* file, int line)
{
    cudaError lastErr = cudaGetLastError();
    if(lastErr != cudaSuccess)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "Error at {} line {}: {}", file, line, cudaGetErrorString(lastErr));
    }
    return lastErr;
}
#define checkLastCudaError() __checkLastCudaError(__FILE__, __LINE__)

// Check whether CUDA driver and CUDA device exist. Return 0 if exist, else return -1
int cuda_version_check()
{
    int driverVersion  = -1;
    int runtimeVersion = -1;

    if(cudaDriverGetVersion(&driverVersion) != cudaSuccess)
    {
        // checkLastCudaError();
        // NVLOGI_FMT(TAG, "{}: cudaDriverGetVersion failed", __func__);
    }
    else
    {
        if(cudaRuntimeGetVersion(&runtimeVersion) != cudaSuccess)
        {
            // checkLastCudaError();
            // NVLOGI_FMT(TAG, "{}: cudaRuntimeGetVersion failed", __func__);
        }
    }

    // NVLOGC_FMT(TAG, "{}: driverVersion={} runtimeVersion={}", __func__, driverVersion, runtimeVersion);

    if(driverVersion > 0 && runtimeVersion > 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int cuda_is_device_pointer(const void *ptr)
{
    int in_gpu = 0;

    cudaPointerAttributes attr;
    attr.type = cudaMemoryTypeUnregistered;

    if(cudaPointerGetAttributes(&attr, ptr) != cudaSuccess)
    {
        NVLOGD_FMT(TAG, "{}: cudaPointerGetAttributes failed", __func__);
        in_gpu = 0;
    }

    if(attr.type == cudaMemoryTypeDevice) // Or cudaMemoryTypeManaged?
    {
        in_gpu = 1;
    }

    NVLOGD_FMT(TAG, "{}: {} attr: type={} device={} devicePointer=0x{} hostPointer=0x{} in_gpu={}",
            __func__, (void *)ptr, static_cast<uint32_t>(attr.type), attr.device, attr.devicePointer, attr.hostPointer, in_gpu);
    return in_gpu;
}

int cuda_get_device_count(void)
{
    int num;
    cudaError_t err = cudaGetDeviceCount (&num);
    if (err != cudaSuccess)
    {
        NVLOGW_FMT(TAG, "{}: cudaGetDeviceCount failed", __func__);
        return -1;
    }
    else
    {
        return num;
    }
}

// Check if a CPU memory buffer is host pinned memory: 1 if yes, 0 if no
int cuda_is_host_pinned_memory(void* phost)
{
    cudaPointerAttributes input_attributes;
    if(cudaPointerGetAttributes(&input_attributes, phost) != cudaSuccess)
    {
        checkLastCudaError();
        return 0;
    }

    return (input_attributes.type == cudaMemoryTypeHost) ? 1 : 0;
}

int cuda_page_lock(void* phost, size_t size)
{
    if(cuda_version_check() < 0)
    {
        NVLOGI_FMT(TAG, "{}: CUDA driver or device not exist, skip", __func__);
        return 0;
    }

    int flag = cudaHostRegisterPortable | cudaHostRegisterMapped;
    if(cudaHostRegister(phost, size, flag) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaHostRegister failed", __func__);
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
        return 0;
    }
}

int cuda_page_unlock(void* phost)
{
    if(cuda_version_check() < 0)
    {
        NVLOGI_FMT(TAG, "{}: CUDA driver or device not exist, skip", __func__);
        return 0;
    }

    if(cudaHostUnregister(phost) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaHostUnregister failed", __func__);
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
        return 0;
    }
}

int nv_ipc_memcpy_to_host(void* host, const void* device, size_t size)
{
    NVLOGV_FMT(TAG, "{}: dst_host={} src_gpu={} size={}", __func__, host, (void *)device, size);
/*
    if(cudaSetDevice(0) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaSetDevice to {} failed", __func__, 0);
        return -1;
    }
*/
    if(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaMemcpy failed", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}

int nv_ipc_memcpy_to_device(void* device, const void* host, size_t size)
{
    NVLOGV_FMT(TAG, "{}: dst_gpu={} src_host={} size={}", __func__, device, (void *)host, size);
/*
    if(cudaSetDevice(0) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaSetDevice to {} failed", __func__, 0);
        return -1;
    }
*/
    if(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        checkLastCudaError();
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cudaMemcpy failed", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}
