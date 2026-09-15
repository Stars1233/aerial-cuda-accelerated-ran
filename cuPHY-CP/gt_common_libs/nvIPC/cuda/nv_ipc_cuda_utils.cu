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

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <cuda.h>

#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_utils.h"
#include <cuda_driver_utils/cuda_driver_utils.hpp>

#define TAG "NVIPC.CUDAUTILS"

// Check whether CUDA driver and CUDA device exist. Return 0 if exist, else return -1
int cuda_version_check()
{
    int driverVersion  = -1;

    if(cuDriverGetVersion(&driverVersion) != CUDA_SUCCESS)
    {
        return -1;
    }

    if(driverVersion > 0)
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
    unsigned int mem_type = 0;

    CUresult res = cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr);
    if(res != CUDA_SUCCESS)
    {
        NVLOGD_FMT(TAG, "{}: cuPointerGetAttribute failed", __func__);
        return 0;
    }

    if(mem_type == CU_MEMORYTYPE_DEVICE)
    {
        in_gpu = 1;
    }

    NVLOGD_FMT(TAG, "{}: {} mem_type={} in_gpu={}",
            __func__, (void *)ptr, mem_type, in_gpu);
    return in_gpu;
}

int cuda_get_device_count(void)
{
    if(cuInit(0) != CUDA_SUCCESS)
    {
        return -1;
    }

    int num;
    CUresult err = cuDeviceGetCount(&num);
    if (err != CUDA_SUCCESS)
    {
        NVLOGW_FMT(TAG, "{}: cuDeviceGetCount failed", __func__);
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
    unsigned int mem_type = 0;
    CUresult res = cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)phost);
    if(res != CUDA_SUCCESS)
    {
        return 0;
    }

    return (mem_type == CU_MEMORYTYPE_HOST) ? 1 : 0;
}

int cuda_page_lock(void* phost, size_t size)
{
    if(cuda_version_check() < 0)
    {
        NVLOGI_FMT(TAG, "{}: CUDA driver or device not exist, skip", __func__);
        return 0;
    }

    CUdevice dev;
    CUcontext ctx;
    if(cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
       cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS || cuCtxSetCurrent(ctx) != CUDA_SUCCESS)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to establish CUDA context", __func__);
        return -1;
    }

    unsigned int flag = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
    if(cuMemHostRegister(phost, size, flag) != CUDA_SUCCESS)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemHostRegister failed", __func__);
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

    PrimaryCtxGuard ctx_guard(0);

    if(cuMemHostUnregister(phost) != CUDA_SUCCESS)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemHostUnregister failed", __func__);
        return -1;
    }

    // Balance the extra cuDevicePrimaryCtxRetain from cuda_page_lock
    CUdevice dev;
    if(cuDeviceGet(&dev, 0) == CUDA_SUCCESS)
    {
        cuDevicePrimaryCtxRelease(dev);
    }

    NVLOGI_FMT(TAG, "{}: OK", __func__);
    return 0;
}

int nv_ipc_memcpy_to_host(void* host, const void* device, size_t size)
{
    NVLOGV_FMT(TAG, "{}: dst_host={} src_gpu={} size={}", __func__, host, (void *)device, size);

    if(cuMemcpyDtoH(host, (CUdeviceptr)device, size) != CUDA_SUCCESS)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyDtoH failed", __func__);
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

    if(cuMemcpyHtoD((CUdeviceptr)device, host, size) != CUDA_SUCCESS)
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyHtoD failed", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}
