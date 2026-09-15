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

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>    /* For O_* constants */
#include <sys/stat.h> /* For mode constants */
#include <sys/types.h>

#include "nv_ipc_shm.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 6) //"NVIPC.SHM"

// Sync the name length with nv_ipc_t definition
#define NV_SHM_NAME_MAX_LEN (32 + 16)

typedef struct
{
    int    primary;
    int    shm_fd;
    size_t size;
    char   name[NV_SHM_NAME_MAX_LEN];
    void*  mapped_addr;
} priv_data_t;

static inline priv_data_t* get_private_data(nv_ipc_shm_t* ipc_shm)
{
    return (priv_data_t*)((int8_t*)ipc_shm + sizeof(nv_ipc_shm_t));
}

static void* ipc_get_mapped_addr(nv_ipc_shm_t* ipc_shm)
{
    if(ipc_shm == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return NULL;
    }

    priv_data_t* priv_data = get_private_data(ipc_shm);
    return priv_data->mapped_addr;
}

static size_t ipc_get_size(nv_ipc_shm_t* ipc_shm)
{
    if(ipc_shm == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return 0;
    }

    priv_data_t* priv_data = get_private_data(ipc_shm);
    return priv_data->size;
}

static int ipc_shm_close(nv_ipc_shm_t* ipc_shm)
{
    if(ipc_shm == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc_shm);
    int          ret       = 0;

    if(priv_data->mapped_addr)
    {
        if(munmap(priv_data->mapped_addr, priv_data->size) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: munmap failed", __func__);
            ret = -1;
        }
    }
    if(priv_data->shm_fd)
    {
        if(close(priv_data->shm_fd) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: close shm_fd failed", __func__);
            ret = -1;
        }
    }
    if(priv_data->primary)
    {
        if(shm_unlink(priv_data->name) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: shm_unlink %s failed", __func__, priv_data->name);
            ret = -1;
        }
    }

    free(ipc_shm);

    if(ret == 0)
    {
        NVLOGI(TAG, "%s: OK", __func__);
    }
    return ret;
}

static int ipc_shm_open(priv_data_t* priv_data)
{
    int open_flag;
    if(priv_data->primary)
    {
        open_flag = O_RDWR | O_CREAT;
    }
    else
    {
        open_flag = O_RDWR;
    }

    if((priv_data->shm_fd = shm_open(priv_data->name, open_flag, 0660)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: shm_open %s failed error %d", __func__, priv_data->name,priv_data->shm_fd);
        return -1;
    }

    if(priv_data->primary)
    {
        if(ftruncate(priv_data->shm_fd, priv_data->size))
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: ftruncate failed. size=%lu", __func__, priv_data->size);
            return -1;
        }
    }
    else
    {
        struct stat buffer;
        if(fstat(priv_data->shm_fd, &buffer) == 0)
        {
            priv_data->size = buffer.st_size;
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fstat get size failed", __func__);
            return -1;
        }
    }

    priv_data->mapped_addr = mmap(NULL, priv_data->size, PROT_READ | PROT_WRITE, MAP_SHARED, priv_data->shm_fd, 0);
    if(priv_data->mapped_addr == MAP_FAILED)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: mmap failed: name=%s size=%lu errno=%d %s", __func__,
                priv_data->name, (unsigned long)priv_data->size, errno, strerror(errno));
        priv_data->mapped_addr = NULL;
        return -1;
    }

    if (priv_data->primary)
    {
        memset(priv_data->mapped_addr, 0, priv_data->size);
    }

    return 0;
}

nv_ipc_shm_t* nv_ipc_shm_open(int primary, const char* name, size_t size)
{
    if(name == NULL || (primary != 0 && size <= 0))
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid configuration", __func__);
        return NULL;
    }

    int           ipc_size = sizeof(nv_ipc_shm_t) + sizeof(priv_data_t);
    nv_ipc_shm_t* ipc_shm  = malloc(ipc_size);
    if(ipc_shm == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }

    memset(ipc_shm, 0, ipc_size);

    priv_data_t* priv_data = get_private_data(ipc_shm);
    priv_data->primary     = primary;
    priv_data->size        = size;
    nvlog_safe_strncpy(priv_data->name, name, NV_SHM_NAME_MAX_LEN);

    ipc_shm->get_mapped_addr = ipc_get_mapped_addr;
    ipc_shm->get_size        = ipc_get_size;
    ipc_shm->close           = ipc_shm_close;

    if(ipc_shm_open(priv_data) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: primary=%d name=%s size=%lu Failed", __func__, primary, name, priv_data->size);
        ipc_shm_close(ipc_shm);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: primary=%d name=%s size=%lu OK", __func__, primary, name, priv_data->size);
        return ipc_shm;
    }
}
