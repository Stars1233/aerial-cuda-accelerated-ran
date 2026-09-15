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
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>

#include "array_queue.h"
#include "nv_ipc_shm.h"
#include "nv_ipc_mempool.h"
#include "nv_ipc_cudapool.h"
#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_gpudatapool.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 13) //"NVIPC.MEMPOOL"

typedef enum
{
    STATE_FREE = 0,
    STATE_USED = 1
} buf_state_t;

typedef struct
{
    int primary;
    int cuda_device_id;

    int32_t buf_size;
    int32_t pool_len;

    nv_ipc_shm_t* shmpool;

#ifdef NVIPC_CUDA_ENABLE
    nv_ipc_cudapool_t* cudapool;
#endif
#ifdef NVIPC_GDRCPY_ENABLE
	nv_ipc_gpudatapool_t *gpupool;
#endif

    // Memory pool header should be shared between primary and secondary by CPU SHM
    array_queue_t* queue;

    // Memory buffers, can be CPU memory or CUDA memory
    int8_t* body;

} priv_data_t;

static inline priv_data_t* get_private_data(nv_ipc_mempool_t* mempool)
{
    return (priv_data_t*)((int8_t*)mempool + sizeof(nv_ipc_mempool_t));
}

static int32_t ipc_mempool_alloc(nv_ipc_mempool_t* mempool)
{
    priv_data_t* priv_data = get_private_data(mempool);
    return priv_data->queue->dequeue(priv_data->queue);
}

// Set page lock/unlock for host memory
int nv_ipc_page_lock(void* phost, size_t size)
{
    int ret = 0;
#ifdef NVIPC_CUDA_ENABLE
    if(cuda_get_device_count() > 0) {
        ret = cuda_page_lock(phost, size);
    } else {
        NVLOGC(TAG, "%s: CUDA-capable device not exist, skip", __func__);
    }
#else
    NVLOGC(TAG, "%s: CUDA not enabled in build, skip", __func__);
#endif
    return ret;
}

int nv_ipc_page_unlock(void* phost)
{
    int ret = 0;
#ifdef NVIPC_CUDA_ENABLE
    if(cuda_get_device_count() > 0) {
        ret = cuda_page_unlock(phost);
    } else {
        NVLOGC(TAG, "%s: CUDA-capable device not exist, skip", __func__);
    }
#else
    NVLOGC(TAG, "%s: CUDA not enabled in build, skip", __func__);
#endif
    return ret;
}

static int ipc_mempool_free(nv_ipc_mempool_t* mempool, int32_t index)
{
    priv_data_t* priv_data = get_private_data(mempool);
    if(index < 0 || index >= priv_data->pool_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid index %d", __func__, index);
        return -1;
    }
    else
    {
        return priv_data->queue->enqueue(priv_data->queue, index);
    }
}

static int ipc_mempool_get_index(nv_ipc_mempool_t* mempool, void* buf)
{
    if(mempool == NULL || buf == NULL)
    {
        return -1;
    }
    else
    {
        priv_data_t* priv_data = get_private_data(mempool);
        return ((int8_t*)buf - priv_data->body) / priv_data->buf_size;
    }
}

static void* ipc_mempool_get_addr(nv_ipc_mempool_t* mempool, int32_t index)
{
    if(mempool == NULL || index < 0)
    {
        return NULL;
    }
    else
    {
        priv_data_t* priv_data = get_private_data(mempool);
        if(index >= priv_data->pool_len)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid index %d", __func__, index);
            return NULL;
        }
        return priv_data->body + (size_t)priv_data->buf_size * index;
    }
}

static int ipc_mempool_get_buf_size(nv_ipc_mempool_t* mempool)
{
    return mempool == NULL ? -1 : get_private_data(mempool)->buf_size;
}

static int ipc_mempool_get_pool_len(nv_ipc_mempool_t* mempool)
{
    return mempool == NULL ? -1 : get_private_data(mempool)->pool_len;
}

static int ipc_mempool_get_free_count(nv_ipc_mempool_t* mempool)
{
    priv_data_t* priv_data = get_private_data(mempool);
    return mempool == NULL ? -1 : priv_data->queue->get_count(priv_data->queue);
}

static void* ipc_mempool_get_free_queue(nv_ipc_mempool_t* mempool)
{
    priv_data_t* priv_data = get_private_data(mempool);
    return mempool == NULL ? NULL : priv_data->queue;
}

static int ipc_memcpy_to_host(nv_ipc_mempool_t* mempool, void* host, const void* device, size_t size)
{
    if(mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(mempool);

    if(size > priv_data->buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: size exceeds boundary", __func__);
        return -1;
    }

    if(priv_data->cuda_device_id == NV_IPC_MEMPOOL_NO_CUDA_DEV)
    {
        // No CUDA device, the device address is in CPU memory, fall back to CPU memory copy
        memcpy(host, device, size);
        return 0;
    }
#ifdef NVIPC_CUDA_ENABLE
    else if (priv_data->cuda_device_id == NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS)
    {
        nv_ipc_memcpy_to_host(host, device, size);
        return 0;
    }
    else if(priv_data->cudapool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: CUDA memory pool is NULL", __func__);
        return -1;
    }
    else
    {
        return priv_data->cudapool->memcpy_to_host(priv_data->cudapool, host, device, size);
    }
#elif NVIPC_GDRCPY_ENABLE
    if(priv_data->gpupool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: GPU memory pool is NULL", __func__);
        return -1;
    }
    else
    {
        return priv_data->gpupool->memcpy_to_host(priv_data->gpupool, host, device, size);
    }
#else
    return 0;
#endif
}

static int ipc_memcpy_to_device(nv_ipc_mempool_t* mempool, void* device, const void* host, size_t size)
{
    if(mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(mempool);

    if(size > priv_data->buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: size exceeds boundary", __func__);
        return -1;
    }

    if(priv_data->cuda_device_id == NV_IPC_MEMPOOL_NO_CUDA_DEV)
    {
        // No CUDA device, the device address is in CPU memory, fall back to CPU memory copy
        memcpy(device, host, size);
        return 0;
    }
#ifdef NVIPC_CUDA_ENABLE
    else if (priv_data->cuda_device_id == NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS)
    {
        nv_ipc_memcpy_to_device(device, host, size);
        return 0;
    }
    else if(priv_data->cudapool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: CUDA memory pool is NULL", __func__);
        return -1;
    }
    else
    {
        return priv_data->cudapool->memcpy_to_device(priv_data->cudapool, device, host, size);
    }
#endif    
#ifdef NVIPC_GDRCPY_ENABLE
	if(priv_data->gpupool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: GPU memory pool is NULL", __func__);
        return -1;
    }
    else
    {
        return priv_data->gpupool->memcpy_to_device(priv_data->gpupool, device, host, size);
    }
#else
    return 0;
#endif
}

static int ipc_mempool_close(nv_ipc_mempool_t* mempool)
{
    if(mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    // Destroy the nv_ipc_mempool_t instance
    priv_data_t* priv_data = get_private_data(mempool);
    int          ret       = 0;

    if(priv_data->queue != NULL)
    {
        if(priv_data->queue->close(priv_data->queue) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: close queue failed", __func__);
            ret = -1;
        }
    }

#ifdef NVIPC_CUDA_ENABLE
    if(priv_data->cudapool != NULL)
    {
        if(priv_data->cudapool->close(priv_data->cudapool) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: close CUDA pool failed", __func__);
            ret = -1;
        }
    }
#endif

#ifdef NVIPC_GDRCPY_ENABLE
		if(priv_data->gpupool != NULL)
		{
			if(priv_data->gpupool->close(priv_data->gpupool) < 0)
			{
				NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: close GPU pool failed", __func__);
				ret = -1;
			}
		}
#endif

    if(priv_data->shmpool != NULL)
    {
        if(priv_data->shmpool->close(priv_data->shmpool) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: close SHM pool failed", __func__);
            ret = -1;
        }
    }

    free(mempool);

    if(ret == 0)
    {
        NVLOGI(TAG, "%s: OK", __func__);
    }
    return ret;
}

#ifdef NVIPC_GDRCPY_ENABLE
static int ipc_mempool_reInit(nv_ipc_mempool_t* mempool)
{
    int res = 0;
	priv_data_t* priv_data = get_private_data(mempool);
	//NVLOGD(TAG,"[%s]RE-INIT GDR POOL primary %d gpudatapool = 0x%x",__func__,priv_data->primary,priv_data->gpupool);
	res = nv_ipc_gpudatapool_reInit(priv_data->gpupool);
    if(res == -1)
        return -1;
    priv_data->body = priv_data->gpupool->get_gpudatapool_addr(priv_data->gpupool);
	NVLOGD(TAG,"[%s] GPU POOL primary %d priv_data->body = %p",__func__,priv_data->primary,priv_data->body);
	return 0;
}
#endif

static int ipc_mempool_open(priv_data_t* priv_data, const char* name, int cuda_device_id)
{
    size_t header_size = ARRAY_QUEUE_HEADER_SIZE(priv_data->pool_len);
    size_t body_size   = (size_t)priv_data->buf_size * priv_data->pool_len;

    size_t shm_size;

    if(cuda_device_id == NV_IPC_MEMPOOL_NO_CUDA_DEV)
    {
        shm_size = header_size + body_size;
    }
    else if (cuda_device_id == NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS)
    {
        shm_size = header_size;
    }
    else
    {
        shm_size = header_size + CUDA_INFO_SIZE + GPUDATA_INFO_SIZE;
    }

    // Create a shared memory pool
    if((priv_data->shmpool = nv_ipc_shm_open(priv_data->primary, name, shm_size)) == NULL)
    {
        return -1;
    }
    int8_t* shm_addr = priv_data->shmpool->get_mapped_addr(priv_data->shmpool);

    // Create array queue as memory manager
    if((priv_data->queue = array_queue_open(priv_data->primary, name, shm_addr, priv_data->pool_len)) == NULL)
    {
        return -1;
    }

    // Enqueue the whole memory pool
    if(priv_data->primary)
    {
        int i;
        for(i = 0; i < priv_data->pool_len; i++)
        {
            if(priv_data->queue->enqueue(priv_data->queue, i) < 0)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue initial value %d failed", __func__, i);
                return -1;
            }
        }
    }

    // Memory pool body
    priv_data->body = shm_addr + header_size;
#ifdef NVIPC_CUDA_ENABLE
    if(cuda_device_id < 0)
    {
        priv_data->cudapool = NULL;
    }
    else
    {
        if((priv_data->cudapool = nv_ipc_cudapool_open(priv_data->primary, priv_data->body, body_size, cuda_device_id)) == NULL)
        {
            return -1;
        }
        priv_data->body = priv_data->cudapool->get_cudapool_addr(priv_data->cudapool);
    }
#endif

#ifdef NVIPC_GDRCPY_ENABLE
	if(cuda_device_id < 0)
    {
        priv_data->gpupool = NULL;
    }
    else
    {
        if((priv_data->gpupool = nv_ipc_gpudatapool_open(priv_data->primary, priv_data->body, body_size, cuda_device_id)) == NULL)
        {
            return -1;
        }
        priv_data->body = priv_data->gpupool->get_gpudatapool_addr(priv_data->gpupool);
        NVLOGI(TAG,"[%s] GDR POOL primary %d priv_data->body = %p",__func__,priv_data->primary,priv_data->body);
    }
#endif

    return 0;
}

int nv_ipc_mempool_set_ext_bufs(nv_ipc_mempool_t *mempool, void *gpu_addr, void *cpu_addr) {
    if(mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(mempool);
    priv_data->body = gpu_addr;
    NVLOGC(TAG, "%s: set external mempool: gpu_addr=%p in_gpu=%d | cpu_addr=%p in_gpu=%d",
            __func__, gpu_addr, is_device_pointer(gpu_addr), cpu_addr, is_device_pointer(cpu_addr));
    return 0;
}

int nv_ipc_mempool_reset(nv_ipc_mempool_t *mempool)
{
    if (mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t *priv_data = get_private_data(mempool);

    int i;
    // Dequeue existing elements in queue
    for (i = 0; i < priv_data->pool_len; i++)
    {
        if (priv_data->queue->dequeue(priv_data->queue) < 0)
        {
            break;
        }
    }

    // Re-enqueue the whole memory pool
    for (i = 0; i < priv_data->pool_len; i++)
    {
        if (priv_data->queue->enqueue(priv_data->queue, i) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue initial value %d failed", __func__, i);
            return -1;
        }
    }

    return 0;
}

nv_ipc_mempool_t* nv_ipc_mempool_open(int primary, const char* name, int buf_size, int pool_len, int cuda_device_id)
{
    if(name == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid parameter", __func__);
        return NULL;
    }

    int               size    = sizeof(nv_ipc_mempool_t) + sizeof(priv_data_t);
    nv_ipc_mempool_t* mempool = (nv_ipc_mempool_t*)malloc(size);
    if(mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }
    memset(mempool, 0, size);

    priv_data_t* priv_data    = get_private_data(mempool);
    priv_data->primary        = primary;
    priv_data->buf_size       = buf_size;
    priv_data->pool_len       = pool_len;

    if (cuda_device_id >= 0)
    {
        int cuda_dev_count = 0;
#ifdef NVIPC_CUDA_ENABLE
        cuda_dev_count = cuda_get_device_count();
#endif
        if (cuda_device_id >= cuda_dev_count) {
            NVLOGC(TAG, "%s: %s CUDA device not found for ID=%d, dev_cout=%d. fall-back to CPU memory", __func__, name, cuda_device_id, cuda_dev_count);
            cuda_device_id = -1;
        }
    }
    priv_data->cuda_device_id = cuda_device_id;

    mempool->alloc          = ipc_mempool_alloc;
    mempool->free           = ipc_mempool_free;
    mempool->get_index      = ipc_mempool_get_index;
    mempool->get_addr       = ipc_mempool_get_addr;
    mempool->get_buf_size   = ipc_mempool_get_buf_size;
    mempool->get_pool_len   = ipc_mempool_get_pool_len;
    mempool->get_free_count = ipc_mempool_get_free_count;
    mempool->get_free_queue = ipc_mempool_get_free_queue;

    mempool->memcpy_to_host   = ipc_memcpy_to_host;
    mempool->memcpy_to_device = ipc_memcpy_to_device;

    mempool->close = ipc_mempool_close;
#ifdef NVIPC_GDRCPY_ENABLE
	mempool->poolReInit = ipc_mempool_reInit;
#endif
    if(ipc_mempool_open(priv_data, name, cuda_device_id) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: name=%s size=%d*%d cuda_device_id=%d Failed",
                __func__, name, priv_data->buf_size, priv_data->pool_len, priv_data->cuda_device_id);
        ipc_mempool_close(mempool);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: name=%s body=%p size=%d*%d cuda_device_id=%d in_gpu=%d OK", __func__,
                name, priv_data->body, priv_data->buf_size, priv_data->pool_len, priv_data->cuda_device_id, is_device_pointer(priv_data->body));
        return mempool;
    }
}
