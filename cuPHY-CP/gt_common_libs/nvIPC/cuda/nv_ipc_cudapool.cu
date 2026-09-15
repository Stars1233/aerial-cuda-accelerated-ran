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
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "nv_ipc_cudapool.h"
#include "nv_ipc_utils.h"

#define TAG "NVIPC.CUDAPOOL"

#define CONFIG_CREATE_CUDA_STREAM 1

typedef struct
{
    CUipcMemHandle   memHandle;
    CUipcEventHandle eventHandle;
} cuda_ipc_info_t;

typedef struct
{
    int primary;

    // CUDA device ID for CUDA memory case
    int device_id;

    size_t size;

    CUdevice  dev;
    CUcontext dev_ctx;

    cudaEvent_t event;

    cudaStream_t stream;

    // For store the CUDA info communicated between different processes, should be in CPU SHM
    cuda_ipc_info_t* ipc_info;
    CUdeviceptr      cuda_addr;

} priv_data_t;

static inline priv_data_t* get_private_data(nv_ipc_cudapool_t* cudapool)
{
    return (priv_data_t*)((char*)cudapool + sizeof(nv_ipc_cudapool_t));
}

static int cudapool_create(priv_data_t* priv_data)
{
    CUresult res;

    res = cuMemAlloc(&priv_data->cuda_addr, priv_data->size);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to allocate device memory", __func__);
        return -1;
    }

    if (priv_data->primary)
    {
        if (cuMemsetD8(priv_data->cuda_addr, 0, priv_data->size) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to memset device memory", __func__);
            return -1;
        }
    }

    memset(priv_data->ipc_info, 0, sizeof(cuda_ipc_info_t));

    res = cuIpcGetMemHandle(&priv_data->ipc_info->memHandle, priv_data->cuda_addr);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to create memory handler", __func__);
        return -1;
    }

    res = cuEventCreate(&priv_data->event, CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to create event", __func__);
        return -1;
    }

    res = cuIpcGetEventHandle(&priv_data->ipc_info->eventHandle, priv_data->event);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to get event handler", __func__);
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: sizeof(cuda_ipc_info_t)={} device_id={} cuda_addr=0x{:x} OK", __func__, sizeof(cuda_ipc_info_t), priv_data->device_id, static_cast<unsigned long long>(priv_data->cuda_addr));
        return 0;
    }
}

static int cudapool_lookup(priv_data_t* priv_data)
{
    CUresult res;

    res = cuIpcOpenMemHandle(&priv_data->cuda_addr, priv_data->ipc_info->memHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to lookup memory handler", __func__);
        return -1;
    }

    res = cuIpcOpenEventHandle(&priv_data->event, priv_data->ipc_info->eventHandle);
    if(res != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to lookup event handler", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}

static int cudapool_close(priv_data_t* priv_data)
{
    if(cuIpcCloseMemHandle(priv_data->cuda_addr) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to close memory handler", __func__);
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
        return 0;
    }
}

static int cudapool_destroy(priv_data_t* priv_data)
{
    if(cuMemFree(priv_data->cuda_addr) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: failed to free memory", __func__);
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
        return 0;
    }
}

static void* ipc_get_cudapool_addr(nv_ipc_cudapool_t* cudapool)
{
    if(cudapool == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: instance not exist", __func__);
        return NULL;
    }
    priv_data_t* priv_data = get_private_data(cudapool);
    return (void*)(uintptr_t)priv_data->cuda_addr;
}

static int ipc_memcpy_to_host(nv_ipc_cudapool_t* cudapool, void* host, const void* device, size_t size)
{
    NVLOGV_FMT(TAG, "{}: dst_host={} src_gpu={} size={}", __func__, host, (void *)device, size);

    if(cudapool == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(cudapool);

    if(cuCtxSetCurrent(priv_data->dev_ctx) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuCtxSetCurrent failed for device {}", __func__, priv_data->device_id);
        return -1;
    }

    int ret = 0;

    if(CONFIG_CREATE_CUDA_STREAM)
    {
        if(cuMemcpyDtoHAsync(host, (CUdeviceptr)device, size, priv_data->stream) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyDtoHAsync failed", __func__);
            ret = -1;
        }
        else if(cuStreamSynchronize(priv_data->stream) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuStreamSynchronize failed", __func__);
            ret = -1;
        }
    }
    else
    {
        if(cuMemcpyDtoH(host, (CUdeviceptr)device, size) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyDtoH failed", __func__);
            ret = -1;
        }
    }

    return ret;
}

static int ipc_memcpy_to_device(nv_ipc_cudapool_t* cudapool, void* device, const void* host, size_t size)
{
    NVLOGV_FMT(TAG, "{}: dst_gpu={} src_host={} size={}", __func__, device, (void *)host, size);

    if(cudapool == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(cudapool);

    if(cuCtxSetCurrent(priv_data->dev_ctx) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuCtxSetCurrent failed for device {}", __func__, priv_data->device_id);
        return -1;
    }

    int ret = 0;

    if(CONFIG_CREATE_CUDA_STREAM)
    {
        if(cuMemcpyHtoDAsync((CUdeviceptr)device, host, size, priv_data->stream) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyHtoDAsync failed", __func__);
            ret = -1;
        }
        else if(cuStreamSynchronize(priv_data->stream) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuStreamSynchronize failed", __func__);
            ret = -1;
        }
    }
    else
    {
        if(cuMemcpyHtoD((CUdeviceptr)device, host, size) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuMemcpyHtoD failed", __func__);
            ret = -1;
        }
    }

    return ret;
}

static int ipc_cudapool_close(nv_ipc_cudapool_t* cudapool)
{
    if(cudapool == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(cudapool);

    int ret = 0;
    if(cuCtxSetCurrent(priv_data->dev_ctx) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuCtxSetCurrent failed for device {}", __func__, priv_data->device_id);
        ret = -1;
    }

    if(CONFIG_CREATE_CUDA_STREAM)
    {
        if(cuStreamDestroy(priv_data->stream) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuStreamDestroy failed", __func__);
            ret = -1;
        }
    }

    if(priv_data->primary)
    {
        ret = cudapool_destroy(priv_data);
    }
    else
    {
        ret = cudapool_close(priv_data);
    }

    if(cuDevicePrimaryCtxRelease(priv_data->dev) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuDevicePrimaryCtxRelease failed", __func__);
        ret = -1;
    }

    free(cudapool);

    if(ret == 0)
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
    }
    return ret;
}

static int ipc_cudapool_open(priv_data_t* priv_data)
{
    if(cuInit(0) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuInit failed", __func__);
        return -1;
    }

    if(cuDeviceGet(&priv_data->dev, priv_data->device_id) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuDeviceGet failed for device {}", __func__, priv_data->device_id);
        return -1;
    }

    if(cuDevicePrimaryCtxRetain(&priv_data->dev_ctx, priv_data->dev) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuDevicePrimaryCtxRetain failed", __func__);
        return -1;
    }

    if(cuCtxSetCurrent(priv_data->dev_ctx) != CUDA_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuCtxSetCurrent failed", __func__);
        return -1;
    }

    if(CONFIG_CREATE_CUDA_STREAM)
    {
        if(cuStreamCreate(&priv_data->stream, 0) != CUDA_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: cuStreamCreate failed", __func__);
            return -1;
        }
    }

    if(priv_data->primary)
    {
        return cudapool_create(priv_data);
    }
    else
    {
        return cudapool_lookup(priv_data);
    }
}

nv_ipc_cudapool_t* nv_ipc_cudapool_open(int primary, void* shm, size_t size, int device_id)
{
    if(shm == NULL || size <= 0 || device_id < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: invalid parameter", __func__);
        return NULL;
    }

    int                struct_size = sizeof(nv_ipc_cudapool_t) + sizeof(priv_data_t);
    nv_ipc_cudapool_t* cudapool    = (nv_ipc_cudapool_t*)malloc(struct_size);
    if(cudapool == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: memory malloc failed", __func__);
        return NULL;
    }
    memset(cudapool, 0, struct_size);

    priv_data_t* priv_data = get_private_data(cudapool);
    priv_data->primary     = primary;
    priv_data->device_id   = device_id;
    priv_data->size        = size;
    priv_data->ipc_info    = (cuda_ipc_info_t*)shm;

    cudapool->get_cudapool_addr = ipc_get_cudapool_addr;
    cudapool->memcpy_to_host    = ipc_memcpy_to_host;
    cudapool->memcpy_to_device  = ipc_memcpy_to_device;
    cudapool->close             = ipc_cudapool_close;

    if(ipc_cudapool_open(priv_data) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: Failed", __func__);
        ipc_cudapool_close(cudapool);
        return NULL;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: OK", __func__);
        return cudapool;
    }
}
