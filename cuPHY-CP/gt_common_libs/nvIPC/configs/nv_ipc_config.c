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
#include <string.h>

#include "nv_ipc.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 17) //"NVIPC.CONF"

#define ENABLE_SHM_FORWARDER 0

/*********** Default configurations for SHM IPC **************/
#define SHM_RING_QUEUE_LEN 8192

// MAX MSG size is 2692. Reserve 8 bytes for data pool_id and buf_id. And align to 8 bytes.
#define SHM_MSG_BUF_SIZE (8192)
#define SHM_MSG_POOL_LEN (4096)

// There's 2 PDUs in each TX MSG, we allocate 1 buffer for them.
#define SHM_DATA_BUF_SIZE (576000) // PDU buffer size
#define SHM_DATA_LARGE_SIZE (4096000) // Large data buffer size
#define SHM_CPU_DATA_POOL_LEN (1024)
#define SHM_CPU_LARGE_POOL_LEN (64)
#define SHM_CUDA_DATA_POOL_LEN (0)
#define SHM_GPU_DATA_POOL_LEN (0)

// Semaphore names
//static const char SEMAPHORE_NAME_UL[] = "sem_shm_ul_%d";
//static const char SEMAPHORE_NAME_DL[] = "sem_shm_dl_%d";

/*********** Default configurations for UDP IPC **************/
#define UDP_MAC_PORT 38555
#define UDP_PHY_PORT 38556

// Forwarder configuration, for test only
#define UDP_FW_MAC_PORT 39555
#define UDP_FW_PHY_PORT 39556
#define UDP_FW_UE_PHY_PORT 38555
#define UDP_FW_UE_MAC_PORT 38556

static void set_default_udp_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type)
{
    nv_ipc_config_udp_t* udp_config = &cfg->transport_config.udp;

    // Max FAPI message size and data(TB) size
    udp_config->msg_buf_size  = 152; // Align to 8 byte
    udp_config->data_buf_size = 150 * 1024;

    // TODO: To support multiple cells, need to set different UDP socket ports for different cells
    udp_config->local_port  = is_module_primary(module_type) ? UDP_PHY_PORT : UDP_MAC_PORT;
    udp_config->remote_port = is_module_primary(module_type) ? UDP_MAC_PORT : UDP_PHY_PORT;

    sprintf(udp_config->local_addr, "127.0.0.1");
    sprintf(udp_config->remote_addr, "127.0.0.1");
}

static void set_default_mempools_size(nv_ipc_mempool_size_t mempools[]) {
    mempools[NV_IPC_MEMPOOL_CPU_MSG].buf_size = SHM_MSG_BUF_SIZE;
    mempools[NV_IPC_MEMPOOL_CPU_MSG].pool_len = SHM_MSG_POOL_LEN;

    mempools[NV_IPC_MEMPOOL_CPU_DATA].buf_size = SHM_DATA_BUF_SIZE;
    mempools[NV_IPC_MEMPOOL_CPU_DATA].pool_len = SHM_CPU_DATA_POOL_LEN;

    mempools[NV_IPC_MEMPOOL_CPU_LARGE].buf_size = SHM_DATA_LARGE_SIZE;
    mempools[NV_IPC_MEMPOOL_CPU_LARGE].pool_len = SHM_CPU_LARGE_POOL_LEN;

    mempools[NV_IPC_MEMPOOL_CUDA_DATA].buf_size = SHM_DATA_BUF_SIZE;
    mempools[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = SHM_CUDA_DATA_POOL_LEN;

    mempools[NV_IPC_MEMPOOL_GPU_DATA].buf_size = SHM_DATA_BUF_SIZE;
    mempools[NV_IPC_MEMPOOL_GPU_DATA].pool_len = SHM_GPU_DATA_POOL_LEN;
}

static void set_default_shm_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type)
{
    nv_ipc_config_shm_t* shm_config = &cfg->transport_config.shm;

    // The name prefix of a nv_ipc_t instance. Set unique name prefix for each nv_ipc_t instance.
    if(ENABLE_SHM_FORWARDER)
    {
        nvlog_safe_strncpy(shm_config->prefix, is_module_primary(module_type) ? "nvipc_ue" : "nvipc_gnb", NV_NAME_MAX_LEN);
    }
    else
    {
        nvlog_safe_strncpy(shm_config->prefix, "nvipc", NV_NAME_MAX_LEN);
    }
    shm_config->ring_len = SHM_RING_QUEUE_LEN;

    // CUDA device ID for CUDA memory pool. Can set to -1 to fall back to CPU memory pool
    shm_config->cuda_device_id = -1;

    set_default_mempools_size(shm_config->mempool_size);
}

static void set_default_dpdk_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type)
{
    nv_ipc_config_dpdk_t* dpdk_config = &cfg->transport_config.dpdk;

    dpdk_config->primary = is_module_primary(module_type);

    dpdk_config->cuda_device_id = -1;

    dpdk_config->lcore_id = 5;
    nvlog_safe_strncpy(dpdk_config->prefix, "nvipc", NV_NAME_MAX_LEN);

    nvlog_safe_strncpy(dpdk_config->local_nic_pci, "b5:00.1", NV_NAME_MAX_LEN);
    nvlog_safe_strncpy(dpdk_config->peer_nic_mac, "00:00:00:00:00:00", NV_NAME_MAX_LEN);

    set_default_mempools_size(dpdk_config->mempool_size);
}

static void set_default_doca_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type)
{
    nv_ipc_config_doca_t* doca_config = &cfg->transport_config.doca;

    doca_config->primary = is_module_primary(module_type);

    doca_config->cuda_device_id = -1;

    doca_config->cpu_core = 5;
    nvlog_safe_strncpy(doca_config->prefix, "nvipc", NV_NAME_MAX_LEN);

    nvlog_safe_strncpy(doca_config->host_pci, "ca:00.0", NV_NAME_MAX_LEN);
    nvlog_safe_strncpy(doca_config->dpu_pci, "b3:00.0", NV_NAME_MAX_LEN);

    set_default_mempools_size(doca_config->mempool_size);
}

int set_nv_ipc_default_config(nv_ipc_config_t* cfg, nv_ipc_module_t module_type)
{
    cfg->module_type = module_type;

    // cfg->cell_id = 0;
    // cfg->ipc_transport = NV_IPC_TRANSPORT_UDP;

    if(cfg->ipc_transport == NV_IPC_TRANSPORT_UDP)
    {
        set_default_udp_config(cfg, module_type);
    }
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        set_default_shm_config(cfg, module_type);
    }
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        set_default_dpdk_config(cfg, module_type);
    }
    else if(cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        set_default_doca_config(cfg, module_type);
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unsupported transport: transport=%d", __func__, cfg->ipc_transport);
        return -1;
    }
    memset(&cfg->debug_configs, 0, sizeof(nv_ipc_debug_config_t));
    return 0;
}
