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
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <sys/queue.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "test_cuda.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_cuda_utils.h"
#include "cuda_driver_utils/cuda_driver_utils.hpp"

// Log level: NVLOG_ERROR, NVLOG_CONSOLE, NVLOG_WARN, NVLOG_INFO, NVLOG_DEBUG, NVLOG_VERBOSE
#define DEFAULT_TEST_LOG_LEVEL NVLOG_CONSOLE
#define DEFAULT_TEST_LOG_LEVEL_CONSOLE NVLOG_CONSOLE

#define TEST_DUPLEX_TRANSFER 1

// Configure whether to sync by TTI or sync by one single message.
#define CONFIG_SYNC_BY_TTI 1

#define TEST_MSG_COUNT 1000
#define MAX_EVENTS 10
#define TEST_DATA_BUF_LEN 256

#define SHM_MSG_BUF_SIZE (512)
#define SHM_DATA_BUF_SIZE (2 * 150 * 1024L)

#define TEST_CPU_DATA_POOL_LEN 2048
#define TEST_CUDA_DATA_POOL_LEN 1024

#define UDP_PACKET_MAX_SIZE 65000
#define UDP_DATA_BUF_SIZE (UDP_PACKET_MAX_SIZE - SHM_MSG_BUF_SIZE) // PDU buffer size

// The CUDA device ID. Defaults to 0; override via argv[2].
int test_cuda_device_id = 0;

typedef struct
{
    int32_t msg_id;
    int32_t cell_id;
    int32_t msg_len;
    int32_t data_len;
    int32_t data_pool;
} test_msg_t;

#define TAG "NVIPC.TESTCUDA"

nv_ipc_transport_t ipc_transport;
nv_ipc_module_t    module_type;

nv_ipc_t* ipc;

int blocking_flag = 1;

// SCF FAPI definition
#define FAPI_SLOT_INDATION 0x82
#define FAPI_DL_TTI_REQUEST 0x80
#define FAPI_UL_TTI_REQUEST 0x81
#define FAPI_TX_DATA_REQUEST 0x84
#define FAPI_RX_DATA_INDICATION 0x85

test_msg_t test_phy_tx_msg[TEST_MSG_COUNT] = {
    {FAPI_RX_DATA_INDICATION, 0, SHM_MSG_BUF_SIZE, SHM_DATA_BUF_SIZE, NV_IPC_MEMPOOL_CPU_DATA},
    {FAPI_RX_DATA_INDICATION, 1, SHM_MSG_BUF_SIZE, SHM_DATA_BUF_SIZE, NV_IPC_MEMPOOL_CUDA_DATA},
    {FAPI_SLOT_INDATION, 2, SHM_MSG_BUF_SIZE, 0, -1}};

test_msg_t test_mac_tx_msg[TEST_MSG_COUNT] = {
    {FAPI_TX_DATA_REQUEST, 0, SHM_MSG_BUF_SIZE, SHM_DATA_BUF_SIZE, NV_IPC_MEMPOOL_CPU_DATA},
    {FAPI_TX_DATA_REQUEST, 1, SHM_MSG_BUF_SIZE, SHM_DATA_BUF_SIZE, NV_IPC_MEMPOOL_CUDA_DATA},
    {FAPI_DL_TTI_REQUEST, 2, SHM_MSG_BUF_SIZE, 0, -1}};

cudaEvent_t send_start, send_end;
cudaEvent_t recv_start, recv_end;
void*       gpu_buf_send;
void*       gpu_buf_recv;
float       send_copy_time = 0;
float       recv_copy_time = 0;

char* cpu_buf_send;
char* cpu_buf_recv;

static void init_test_messages()
{
    int i;
    for(i = 0; i < TEST_MSG_COUNT; i++)
    {
        test_phy_tx_msg[i].msg_id    = FAPI_RX_DATA_INDICATION;
        test_phy_tx_msg[i].cell_id   = 0;
        test_phy_tx_msg[i].msg_len   = SHM_MSG_BUF_SIZE;
        test_phy_tx_msg[i].data_len  = SHM_DATA_BUF_SIZE;
        test_phy_tx_msg[i].data_pool = NV_IPC_MEMPOOL_CPU_DATA;

        test_mac_tx_msg[i].msg_id    = FAPI_TX_DATA_REQUEST;
        test_mac_tx_msg[i].cell_id   = 0;
        test_mac_tx_msg[i].msg_len   = SHM_MSG_BUF_SIZE;
        test_mac_tx_msg[i].data_len  = SHM_DATA_BUF_SIZE;
        test_mac_tx_msg[i].data_pool = NV_IPC_MEMPOOL_CPU_DATA;
    }
}

////////////////////////////////////////////////////////////////////////
// Handle an RX message
static int ipc_handle_rx_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(msg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: ERROR: buffer is empty", __func__);
        return -1;
    }

    int32_t* p_fapi = (int32_t*)msg->msg_buf;
    msg->msg_id     = *p_fapi;
    char* str       = (char*)(p_fapi + 1);

    char* p_cpu_data = NULL;
    if(msg->data_buf != NULL)
    {
        int gpu = 0;
        if(msg->data_pool == NV_IPC_MEMPOOL_CUDA_DATA)
        {
            gpu = 1;
        }
        else if(msg->data_pool == NV_IPC_MEMPOOL_CPU_DATA)
        {
            gpu = 0;
            nv_ipc_memcpy_to_device(gpu_buf_recv, msg->data_buf, SHM_DATA_BUF_SIZE);
            CUDA_DRIVER_CHECK(cuEventSynchronize(recv_end));
        }

        test_cuda_to_lower_case(test_cuda_device_id, (char*)msg->data_buf, TEST_DATA_BUF_LEN, gpu);

        if(gpu)
        {
            p_cpu_data = cpu_buf_recv;
            memset(cpu_buf_recv, 0, TEST_DATA_BUF_LEN);
            ipc->cuda_memcpy_to_host(ipc, p_cpu_data, msg->data_buf, TEST_DATA_BUF_LEN);
        }
        else
        {
            p_cpu_data = (char*)msg->data_buf;
        }
    }

    NVLOGI_FMT(TAG, "Receive: cell_id={} {}; {} <<<", msg->cell_id, str, p_cpu_data == NULL ? "DATA: NULL" : p_cpu_data);
    return 0;
}

////////////////////////////////////////////////////////////////////////
// Build a TX message
static int ipc_build_tx_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    static int counter = 0;

    if(msg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: ERROR: buffer is empty", __func__);
        return -1;
    }

    int32_t* p_fapi = (int32_t*)msg->msg_buf;
    *p_fapi         = msg->msg_id;
    char* str       = (char*)(p_fapi + 1);
    sprintf(str, "[counter=%d] MSG Sent By %s", counter++, TAG);

    char* p_cpu_data = NULL;
    if(msg->data_buf != NULL)
    {
        if(msg->data_pool == NV_IPC_MEMPOOL_CUDA_DATA)
        {
            memset(cpu_buf_send, 0, TEST_DATA_BUF_LEN);
            sprintf(cpu_buf_send, "CUDA DATA Sent By %s", TAG);
            ipc->cuda_memcpy_to_device(ipc, msg->data_buf, cpu_buf_send, TEST_DATA_BUF_LEN);
            p_cpu_data = cpu_buf_send;
        }
        else if(msg->data_pool == NV_IPC_MEMPOOL_CPU_DATA)
        {
            nv_ipc_memcpy_to_host(msg->data_buf, gpu_buf_send, SHM_DATA_BUF_SIZE);
            CUDA_DRIVER_CHECK(cuEventSynchronize(send_end));
            p_cpu_data = (char*)msg->data_buf;
        }
    }

    NVLOGI_FMT(TAG, "Send: cell_id={} {}; {} >>>", msg->cell_id, str, p_cpu_data == NULL ? "DATA: NULL" : p_cpu_data);

    return 0;
}

// Always allocate message buffer, but allocate data buffer only when data_len > 0
static int test_nv_ipc_send_msg(nv_ipc_t* ipc, nv_ipc_msg_t* send_msg, test_msg_t* test_msg)
{
    send_msg->msg_id    = test_msg->msg_id;
    send_msg->cell_id   = test_msg->cell_id;
    send_msg->msg_len   = test_msg->msg_len;
    send_msg->data_len  = test_msg->data_len;
    send_msg->data_pool = test_msg->data_pool;

    // Allocate buffer for TX message
    if(ipc->tx_allocate(ipc, send_msg, 0) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{} error: allocate TX buffer failed", __func__);
        return -1;
    }

    // Build the message
    if(ipc_build_tx_msg(ipc, send_msg))
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{} error: build FAPI message failed", __func__);
    }

    // Send the message
    ipc->tx_send_msg(ipc, send_msg);

    return 0;
}

// Always allocate message buffer, but allocate data buffer only when data_len > 0
static int test_nv_ipc_recv_msg(nv_ipc_t* ipc, nv_ipc_msg_t* recv_msg)
{
    recv_msg->msg_buf  = NULL;
    recv_msg->data_buf = NULL;

    // Allocate buffer for TX message
    if(ipc->rx_recv_msg(ipc, recv_msg) < 0)
    {
        NVLOGV_FMT(TAG, "{}: no more message available", __func__);
        return -1;
    }
    ipc_handle_rx_msg(ipc, recv_msg);

    ipc->rx_release(ipc, recv_msg);

    return 0;
}

int is_tti_end(nv_ipc_msg_t* msg)
{
    if(msg != NULL && (msg->msg_id == FAPI_SLOT_INDATION || msg->msg_id == FAPI_DL_TTI_REQUEST))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void blocking_send_task(void)
{
    NVLOGI_FMT(TAG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

    test_msg_t* test_tx_msg;
    if(module_type == NV_IPC_MODULE_PRIMARY)
    {
        test_tx_msg = test_phy_tx_msg;
    }
    else
    {
        test_tx_msg = test_mac_tx_msg;
    }

    CUDA_DRIVER_CHECK(cuEventRecord(send_start, 0));

    int i = 0;
    for(i = 0; i < TEST_MSG_COUNT; i++)
    {
        nv_ipc_msg_t msg;
        test_nv_ipc_send_msg(ipc, &msg, test_tx_msg);
        test_tx_msg++;

        // Sync message one by one
        if(!CONFIG_SYNC_BY_TTI)
        {
            ipc->tx_tti_sem_post(ipc);
        }
    }

    CUDA_DRIVER_CHECK(cuEventRecord(send_end, 0));
    CUDA_DRIVER_CHECK(cuEventSynchronize(send_end));

    CUDA_DRIVER_CHECK(cuEventElapsedTime(&send_copy_time, send_start, send_end));
    NVLOGC_FMT(TAG, ">>> send: count={} GPU->CPU copy: {}MB {:4.1f}ms", i, i * SHM_DATA_BUF_SIZE / 1048576, send_copy_time);

    // Sync message by TTI
    if(CONFIG_SYNC_BY_TTI)
    {
        ipc->tx_tti_sem_post(ipc);
    }
}

void* blocking_recv_task(void* arg)
{
    nv_ipc_msg_t recv_msg;

    while(1)
    {
        NVLOGI_FMT(TAG, "{}: wait for TTI synchronization ...", __func__);
        ipc->rx_tti_sem_wait(ipc);

        NVLOGI_FMT(TAG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");

        int count = 0;

        CUDA_DRIVER_CHECK(cuEventRecord(recv_start, 0));
        while(test_nv_ipc_recv_msg(ipc, &recv_msg) >= 0)
        {
            count++;
            // Loop until all messages are received
            if(module_type == NV_IPC_MODULE_PRIMARY && is_tti_end(&recv_msg) && TEST_DUPLEX_TRANSFER)
            {
                blocking_send_task();
            }
        }
        CUDA_DRIVER_CHECK(cuEventRecord(recv_end, 0));
        CUDA_DRIVER_CHECK(cuEventSynchronize(recv_end));

        CUDA_DRIVER_CHECK(cuEventElapsedTime(&recv_copy_time, recv_start, recv_end));
        NVLOGC_FMT(TAG, "<<< recv: count={} CPU->GPU copy: {}MB {:4.1f}ms", count, count * SHM_DATA_BUF_SIZE / 1048576, recv_copy_time);

        if(module_type == NV_IPC_MODULE_PRIMARY && TEST_DUPLEX_TRANSFER)
        {
            blocking_send_task();
        }
    }
    return NULL;
}

int create_recv_thread(void)
{
    pthread_t thread_id;

    void* (*recv_task)(void*);
    if(blocking_flag)
    {
        recv_task = blocking_recv_task;
    }
    else
    {
        return -1;
    }

    // recv_task thread
    int ret = pthread_create(&thread_id, NULL, recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{} failed, ret = {}", __func__, ret);
    }
    // set_thread_priority(79);
    return ret;
}

int main(int argc, char** argv)
{
    int primary;
    if(argc < 2 || (primary = atoi(argv[1])) < 0)
    {
        fprintf(stderr, "Usage: test_ipc <module> [gpu_id]\n");
        fprintf(stderr, "    module:    0 - secondary;  1 - primary.\n");
        fprintf(stderr, "    gpu_id:    CUDA device_id. Default is 0\n");
        exit(1);
    }

    if(argc > 2)
    {
        test_cuda_device_id = atoi(argv[2]);
    }
    if(test_cuda_device_id < 0)
    {
        fprintf(stderr, "Invalid gpu_id: %d\n", test_cuda_device_id);
        exit(1);
    }

    NVLOGC_FMT(TAG, "{}: INIT argc={}, module_type={}, gpu_id={}", __func__, argc, primary, test_cuda_device_id);

    ipc_transport = NV_IPC_TRANSPORT_SHM;

    if(primary)
    {
        module_type = NV_IPC_MODULE_PRIMARY;
    }
    else
    {
        module_type = NV_IPC_MODULE_SECONDARY;
    }

    // Create configuration
    nv_ipc_config_t config;
    config.ipc_transport = ipc_transport;
    if(set_nv_ipc_default_config(&config, module_type) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: set configuration failed", __func__);
        return -1;
    }

    // Override default CUDA device ID for SHM
    if(ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        config.transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_DATA].buf_size  = SHM_DATA_BUF_SIZE;
        config.transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CUDA_DATA].buf_size = SHM_DATA_BUF_SIZE;

        config.transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_DATA].pool_len = TEST_CPU_DATA_POOL_LEN;

        config.transport_config.shm.cuda_device_id = test_cuda_device_id;
        if(test_cuda_device_id >= 0)
        {
            config.transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = TEST_CUDA_DATA_POOL_LEN;
        }
        else
        {
            config.transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = 0;
        }
    }
    else if(ipc_transport == NV_IPC_TRANSPORT_UDP)
    {
        config.transport_config.udp.msg_buf_size  = SHM_MSG_BUF_SIZE;
        config.transport_config.udp.data_buf_size = UDP_DATA_BUF_SIZE;
    }

    if((ipc = create_nv_ipc_interface(&config)) == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: create IPC interface failed", __func__);
        return -1;
    }

    init_test_messages();

    if(!init_test_cuda_kernels())
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "{}: init_test_cuda_kernels failed", __func__);
        return -1;
    }

    CUDA_DRIVER_CHECK(cuEventCreate(&send_start, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&send_end, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&recv_start, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&recv_end, 0));

    CUdeviceptr gpu_send_dptr = 0, gpu_recv_dptr = 0;
    CUDA_DRIVER_CHECK(cuMemAlloc(&gpu_send_dptr, SHM_DATA_BUF_SIZE));
    CUDA_DRIVER_CHECK(cuMemAlloc(&gpu_recv_dptr, SHM_DATA_BUF_SIZE));
    gpu_buf_send = (void*)(uintptr_t)gpu_send_dptr;
    gpu_buf_recv = (void*)(uintptr_t)gpu_recv_dptr;

    cpu_buf_send = (char*)malloc(SHM_DATA_BUF_SIZE);
    cpu_buf_recv = (char*)malloc(SHM_DATA_BUF_SIZE);

    sprintf(cpu_buf_send, "CPU DATA Sent By %s", TAG);
    nv_ipc_memcpy_to_device(gpu_buf_send, cpu_buf_send, SHM_DATA_BUF_SIZE);

    create_recv_thread();

    NVLOGD_FMT(TAG, "{}: Initiation finished", __func__);
    NVLOGD_FMT(TAG, "========================================");

    while(1)
    {
        usleep(3 * 1000 * 1000);

        if(module_type != NV_IPC_MODULE_PRIMARY)
        {
            if(blocking_flag)
            {
                blocking_send_task();
            }
        }
    }

    return 0;
}
