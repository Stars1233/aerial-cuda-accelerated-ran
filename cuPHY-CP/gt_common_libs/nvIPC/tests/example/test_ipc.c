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

#define _GNU_SOURCE
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
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/epoll.h>

#include "nv_utils.h"
#include "test_cuda.h"
#include "nv_ipc.h"
#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_utils.h"

#ifdef NVIPC_DPDK_ENABLE
#include "nv_ipc_dpdk_utils.h"
#endif

#ifdef BUILD_NVIPC_ONLY
#define YAML_CONFIG_PATH "../../../../nvIPC/tests/example/"
#else
#define YAML_CONFIG_PATH "../../../../../../cuPHY-CP/gt_common_libs/nvIPC/tests/example/"
#endif

// Log level: NVLOG_ERROR, NVLOG_CONSOLE, NVLOG_WARN, NVLOG_INFO, NVLOG_DEBUG, NVLOG_VERBOSE
#define DEFAULT_TEST_LOG_LEVEL NVLOG_INFO
#define DEFAULT_TEST_LOG_LEVEL_CONSOLE NVLOG_CONSOLE

#define TEST_DUPLEX_TRANSFER 1

// Configure whether to sync by TTI or sync by one single message.
#define CONFIG_SYNC_BY_TTI 1

#define TEST_MSG_COUNT 4
#define MAX_EVENTS 10
#define TEST_DATA_BUF_LEN 256

#define UDP_PACKET_MAX_SIZE 65000
#define SHM_MSG_BUF_SIZE (5000)
#define SHM_DATA_BUF_SIZE (UDP_PACKET_MAX_SIZE - SHM_MSG_BUF_SIZE) // PDU buffer size

#define IPC_DATA_SIZE (100 * 1024)

#define NIC_PCI_ADDR "b5:00.1"
#define ETH_MAC_PRIMARY "b8:ce:f6:33:fe:23"
#define ETH_MAC_SECONDARY "00:00:00:00:00:00" // No need to configure secondary MAC

#define MAX_PATH_LEN (1024)

// The CUDA device ID, get from nvipc_config
int test_cuda_device_id = -1;

pthread_t recv_thread_id = 0;

uint64_t total_test_slots = 10000;

atomic_ulong poll_counter;

typedef struct
{
    int32_t msg_id;
    int32_t cell_id;
    int32_t msg_len;
    int32_t data_len;
    int32_t data_pool;
} test_msg_t;

// Log TAG to be configured at main()
static int TAG = (NVLOG_TAG_BASE_NVIPC + 27); // "INIT"
char module_name[32] = "INIT";

nv_ipc_transport_t ipc_transport;
nv_ipc_module_t    module_type;

nv_ipc_t* ipc = NULL;

int blocking_flag;

char cpu_buf_send[TEST_DATA_BUF_LEN];
char cpu_buf_recv[TEST_DATA_BUF_LEN];

#define FAPI_SLOT_INDATION 0x82
#define FAPI_DL_TTI_REQUEST 0x80
#define FAPI_UL_TTI_REQUEST 0x81
#define FAPI_TX_DATA_REQUEST 0x84
#define FAPI_RX_DATA_INDICATION 0x85
#define FAPI_RESERVED_MSG1 0xF1
#define FAPI_RESERVED_MSG2 0xF2

test_msg_t test_phy_tx_msg[TEST_MSG_COUNT] = {
    {FAPI_RX_DATA_INDICATION, 0, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CPU_DATA},
    {FAPI_RX_DATA_INDICATION, 0, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CPU_LARGE},
    {FAPI_RESERVED_MSG1, 1, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CUDA_DATA},
    {FAPI_SLOT_INDATION, 2, SHM_MSG_BUF_SIZE, 0, -1}};

test_msg_t test_mac_tx_msg[TEST_MSG_COUNT] = {
    {FAPI_TX_DATA_REQUEST, 0, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CPU_DATA},
    {FAPI_TX_DATA_REQUEST, 0, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CPU_LARGE},
    {FAPI_RESERVED_MSG2, 1, SHM_MSG_BUF_SIZE, IPC_DATA_SIZE, NV_IPC_MEMPOOL_CUDA_DATA},
    {FAPI_DL_TTI_REQUEST, 2, SHM_MSG_BUF_SIZE, 0, -1}};

////////////////////////////////////////////////////////////////////////
// Handle an RX message
static int ipc_handle_rx_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: ERROR: buffer is empty", __func__);
        return -1;
    }

    int32_t* p_fapi = msg->msg_buf;
    msg->msg_id     = *p_fapi;
    char* str       = (char*)(p_fapi + 1);

    char* p_cpu_data = NULL;
    if(msg->data_buf != NULL && msg->data_len > 0)
    {
        if (msg->data_pool == NV_IPC_MEMPOOL_CUDA_DATA) {
            if (is_device_pointer(msg->data_buf)) {
#ifdef NVIPC_CUDA_ENABLE
                // Test: call CUDA functions to change all string to lower case
                cuda_to_lower_case(msg->data_buf, TEST_DATA_BUF_LEN, test_cuda_device_id);
#endif
                // For buffer in CUDA GPU memory, copy it to CPU memory for log print
                p_cpu_data = cpu_buf_recv;
                memset(cpu_buf_recv, 0, TEST_DATA_BUF_LEN);
                ipc->cuda_memcpy_to_host(ipc, p_cpu_data, msg->data_buf, TEST_DATA_BUF_LEN);
                // nv_ipc_memcpy_to_host(p_cpu_data, msg->data_buf, TEST_DATA_BUF_LEN);
                snprintf(p_cpu_data + strlen(p_cpu_data), 50, " | RX in GPU");
            } else {
                p_cpu_data = msg->data_buf;
                snprintf(p_cpu_data + strlen(p_cpu_data), 50, " | RX Fall-Back to CPU");
            }
        } else {
            p_cpu_data = msg->data_buf;
            snprintf(p_cpu_data + strlen(p_cpu_data), 50, " | RX in CPU");
        }
    }

    NVLOGC(TAG, "RECV: [0x%02X %3d %6d] %s; %s", msg->msg_id, msg->msg_len, msg->data_len, str, p_cpu_data == NULL ? "DATA: NULL" : p_cpu_data);
    return 0;
}

////////////////////////////////////////////////////////////////////////
// Build a TX message
static int ipc_build_tx_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    static int counter = 0;

    if(msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: ERROR: buffer is empty", __func__);
        return -1;
    }

    int32_t* p_fapi = msg->msg_buf;
    *p_fapi         = msg->msg_id;
    char* str       = (char*)(p_fapi + 1);
    sprintf(str, "[N=%d] MSG from %s data_poll=%d", counter++, module_name, msg->data_pool);

    char* p_cpu_data = NULL;
    if(msg->data_buf != NULL)
    {
        if(msg->data_pool == NV_IPC_MEMPOOL_CUDA_DATA)
        {
            if (is_device_pointer(msg->data_buf)) {
                memset(cpu_buf_send, 0, TEST_DATA_BUF_LEN);
                sprintf(cpu_buf_send, "CUDA_DATA from %s | TX from GPU", module_name);
                ipc->cuda_memcpy_to_device(ipc, msg->data_buf, cpu_buf_send, TEST_DATA_BUF_LEN);
                // nv_ipc_memcpy_to_device(msg->data_buf, cpu_buf_send, TEST_DATA_BUF_LEN);
                p_cpu_data = cpu_buf_send;
#ifdef NVIPC_CUDA_ENABLE
                // Test: call CUDA functions to change all string to lower case
                cuda_to_lower_case(msg->data_buf, TEST_DATA_BUF_LEN, test_cuda_device_id);
#endif
            } else {
                sprintf(msg->data_buf, "CUDA_DATA from %s | TX Fall-Back to CPU", module_name);
                p_cpu_data = msg->data_buf;
            }
        }
        else
        {
            sprintf(msg->data_buf, "CPU_DATA  from %s | TX from CPU", module_name);
            p_cpu_data = msg->data_buf;
        }
    }

    NVLOGC(TAG, "SEND: [0x%02X %3d %6d] %s; %s", msg->msg_id, msg->msg_len, msg->data_len, str, p_cpu_data == NULL ? "DATA: NULL" : p_cpu_data);
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

    if (send_msg->data_pool == NV_IPC_MEMPOOL_CUDA_DATA && test_cuda_device_id < 0)
    {
        // Fall back to CPU data pool if CUDA device is not available
        send_msg->data_pool = NV_IPC_MEMPOOL_CPU_DATA;
    }

    // Allocate buffer for TX message
    if(ipc->tx_allocate(ipc, send_msg, 0) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s error: allocate TX buffer failed", __func__);
        return -1;
    }

    // Build the message
    if(ipc_build_tx_msg(ipc, send_msg))
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s error: build FAPI message failed", __func__);
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
        NVLOGV(TAG, "%s: no more message available", __func__);
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
    NVLOGC(TAG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

    test_msg_t* test_tx_msg;
    if(module_type == NV_IPC_MODULE_PRIMARY)
    {
        test_tx_msg = test_phy_tx_msg;
    }
    else
    {
        test_tx_msg = test_mac_tx_msg;
    }

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

    // Sync message by TTI
    if(CONFIG_SYNC_BY_TTI)
    {
        ipc->tx_tti_sem_post(ipc);
    }
}

void* blocking_recv_task(void* arg)
{
    pthread_setname_np(pthread_self(), "blocking_recv");

    nv_ipc_msg_t recv_msg;

    while(1)
    {
        NVLOGI(TAG, "%s: wait for TTI synchronization ...", __func__);
        ipc->rx_tti_sem_wait(ipc);

        NVLOGC(TAG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
        while(test_nv_ipc_recv_msg(ipc, &recv_msg) >= 0)
        {
            // Loop until all messages are received
            if(module_type == NV_IPC_MODULE_PRIMARY && is_tti_end(&recv_msg) && TEST_DUPLEX_TRANSFER)
            {
                blocking_send_task();
            }
        }
    }
    return NULL;
}

#ifdef NVIPC_DPDK_ENABLE
int dpdk_recv_task(void* arg)
{
    nv_ipc_msg_t recv_msg;

    int msg_count = 0;
    dpdk_print_lcore("dpdk_recv_task start running");
    while(1)
    {
        ipc->rx_tti_sem_wait(ipc);

        atomic_fetch_add(&poll_counter, 1);
        while(test_nv_ipc_recv_msg(ipc, &recv_msg) >= 0)
        {
            if(msg_count == 0)
            {
                NVLOGC(TAG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
            }
            msg_count = (msg_count + 1) % TEST_MSG_COUNT;

            // Loop until all messages are received
            if(module_type == NV_IPC_MODULE_PRIMARY && is_tti_end(&recv_msg) && TEST_DUPLEX_TRANSFER)
            {
                blocking_send_task();
            }
        }
    }
    return 0;
}
#endif

// Test send task
void epoll_send_task(void)
{
    NVLOGC(TAG, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

    test_msg_t* test_tx_msg;
    if(module_type == NV_IPC_MODULE_PRIMARY)
    {
        test_tx_msg = test_phy_tx_msg;
    }
    else
    {
        test_tx_msg = test_mac_tx_msg;
    }

    int i = 0;
    for(i = 0; i < TEST_MSG_COUNT; i++)
    {
        nv_ipc_msg_t msg;
        test_nv_ipc_send_msg(ipc, &msg, test_tx_msg);
        test_tx_msg++;

        // Sync message one by one
        if(!CONFIG_SYNC_BY_TTI)
        {
            ipc->notify(ipc, 1);
        }
    }

    // Sync message by TTI
    if(CONFIG_SYNC_BY_TTI)
    {
        ipc->notify(ipc, TEST_MSG_COUNT);
    }
}

// Test receiver task
void* epoll_recv_task(void* arg)
{
    pthread_setname_np(pthread_self(), "epoll_recv");

    struct epoll_event ev, events[MAX_EVENTS];

    int epoll_fd = epoll_create1(0);
    if(epoll_fd == -1)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s epoll_create failed", __func__);
    }

    int ipc_rx_event_fd = ipc->get_fd(ipc);
    ev.events           = EPOLLIN;
    ev.data.fd          = ipc_rx_event_fd;
    if(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ev.data.fd, &ev) == -1)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s epoll_ctl failed", __func__);
    }

    while(1)
    {
        NVLOGI(TAG, "%s: epoll_wait fd_rx=%d ...", __func__, ipc_rx_event_fd);

        int nfds;
        do
        {
            // epoll_wait() may return EINTR when get unexpected signal SIGSTOP from system
            nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        } while(nfds == -1 && errno == EINTR);

        if(nfds < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "epoll_wait failed: epoll_fd=%d nfds=%d err=%d - %s", epoll_fd, nfds, errno, strerror(errno));
        }

        NVLOGC(TAG, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
        int n = 0;
        for(n = 0; n < nfds; ++n)
        {
            if(events[n].data.fd == ipc_rx_event_fd)
            {
                ipc->get_value(ipc);
                nv_ipc_msg_t recv_msg;
                while(test_nv_ipc_recv_msg(ipc, &recv_msg) == 0)
                {
                    if(module_type == NV_IPC_MODULE_PRIMARY && is_tti_end(&recv_msg) && TEST_DUPLEX_TRANSFER)
                    {
                        epoll_send_task();
                    }
                }
            }
        }
    }
    // close(epoll_fd);
    return NULL;
}

int create_recv_thread(void)
{
    void* (*recv_task)(void*);
    if(blocking_flag)
    {
        recv_task = blocking_recv_task;
    }
    else
    {
        recv_task = epoll_recv_task;
    }

    // epoll_recv_task
    int ret = pthread_create(&recv_thread_id, NULL, recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret = %d", __func__, ret);
    }
    // set_thread_priority(79);
    return ret;
}
void dpdk_main(void)
{
    long loop_counter = 0;

    struct timespec ts_now, ts_last;
    nvlog_gettime(&ts_last);
    while(1)
    {
        // recv_poll_task(1);

        nvlog_gettime(&ts_now);
        if(nvlog_timespec_interval(&ts_last, &ts_now) >= 1000L * 1000 * 1000 * 2)
        {
            ts_last = ts_now;
            loop_counter++;
            NVLOGI(TAG, "Loop: loop_counter=%ld recv_poll_counter=%lu", loop_counter, atomic_load(&poll_counter));
            if(module_type != NV_IPC_MODULE_PRIMARY)
            {
                blocking_send_task();
            }
        }
    }

    // rte_eal_mp_wait_lcore();
}

int get_cuda_device_id_config(nv_ipc_config_t* config)
{
    int cuda_device_id = -1;
    if(config->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        cuda_device_id = config->transport_config.shm.cuda_device_id;
    }
    else if(config->ipc_transport == NV_IPC_TRANSPORT_UDP)
    {
        cuda_device_id = -1;
    }
    else if(config->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        cuda_device_id = config->transport_config.dpdk.cuda_device_id;
    }
    else if(config->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        cuda_device_id = config->transport_config.doca.cuda_device_id;
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid configuration: ipc_transport=%d cuda_device_id=%d", __func__, ipc_transport, cuda_device_id);
    }
    return cuda_device_id;
}

int load_hard_code_config(nv_ipc_config_t* config, int primary, nv_ipc_transport_t _transport)
{
    // Create configuration
    config->ipc_transport = _transport;
    if(set_nv_ipc_default_config(config, module_type) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: set configuration failed", __func__);
        return -1;
    }

    // Override default CUDA device ID for SHM
    if(_transport == NV_IPC_TRANSPORT_SHM)
    {
        config->transport_config.shm.cuda_device_id = 0;
        config->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = 128;
    }
    else if(_transport == NV_IPC_TRANSPORT_UDP)
    {
        config->transport_config.udp.msg_buf_size  = SHM_MSG_BUF_SIZE;
        config->transport_config.udp.data_buf_size = SHM_DATA_BUF_SIZE;
    }
    else if(_transport == NV_IPC_TRANSPORT_DPDK)
    {
        config->transport_config.dpdk.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].pool_len  = 4096;
        config->transport_config.dpdk.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].buf_size  = 8192;
        config->transport_config.dpdk.mempool_size[NV_IPC_MEMPOOL_CPU_DATA].pool_len = 1024;
        config->transport_config.dpdk.mempool_size[NV_IPC_MEMPOOL_CPU_DATA].buf_size = 576000;

        config->transport_config.dpdk.cuda_device_id = 0;
        config->transport_config.dpdk.need_eal_init  = 1;
        config->transport_config.dpdk.lcore_id       = 11;
        nvlog_safe_strncpy(config->transport_config.dpdk.prefix, "nvipc", NV_NAME_MAX_LEN);
        nvlog_safe_strncpy(config->transport_config.dpdk.local_nic_pci, NIC_PCI_ADDR, NV_NAME_MAX_LEN);
        if(primary)
        {
            nvlog_safe_strncpy(config->transport_config.dpdk.peer_nic_mac, ETH_MAC_SECONDARY, NV_NAME_MAX_LEN);
        }
        else
        {
            nvlog_safe_strncpy(config->transport_config.dpdk.peer_nic_mac, ETH_MAC_PRIMARY, NV_NAME_MAX_LEN);
        }
    }
    else if(_transport == NV_IPC_TRANSPORT_DOCA)
    {
        config->transport_config.doca.cuda_device_id = 0;
        config->transport_config.doca.cpu_core       = 11;
    }
    return 0;
}

int main(int argc, char** argv)
{
    int primary, transport;
    if(argc < 4 || (transport = atoi(argv[1])) < 0 || (blocking_flag = atoi(argv[2])) < 0 || (primary = atoi(argv[3])) < 0)
    {
        fprintf(stderr, "Usage: test_ipc <transport> <blocking_flag> <module> [config_yaml_file]\n");
        fprintf(stderr, "    transport:      0 - UDP;    1 - SHM;    2 - DPDK;    3 - Config by YAML\n");
        fprintf(stderr, "    blocking_flag:  0 - epoll;  1 - blocking.\n");
        fprintf(stderr, "    module:         0 - secondary;  1 - primary.\n");
        exit(1);
    }
    else
    {
        NVLOGC(TAG, "%s: argc=%d, blocking=%d, transport=%d, module_type=%d", __func__, argc, blocking_flag, transport, primary);
    }

    pthread_setname_np(pthread_self(), "main_init");

    int use_yaml_config = 0;
    switch(transport)
    {
    case 0:
        ipc_transport = NV_IPC_TRANSPORT_UDP;
        break;
    case 1:
        ipc_transport = NV_IPC_TRANSPORT_SHM;
        break;
    case 2:
        ipc_transport = NV_IPC_TRANSPORT_DPDK;
        break;
    case 3:
        use_yaml_config = 1;
        break;
    default:
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: unsupported transport: argc=%d, blocking=%d, transport=%d, module_type=%d", __func__, argc, blocking_flag, transport, primary);
        return 0;
    }

    if(primary)
    {
        module_type = NV_IPC_MODULE_PRIMARY;
        TAG = (NVLOG_TAG_BASE_NVIPC + 29); // "PHY"
        snprintf(module_name, 32, "PHY");
    }
    else
    {
        module_type = NV_IPC_MODULE_SECONDARY;
        TAG = (NVLOG_TAG_BASE_NVIPC + 30); // "MAC"
        snprintf(module_name, 32, "MAC");
    }

    // Get nvipc configuration
    nv_ipc_config_t config;
    if(use_yaml_config)
    {
        char yaml_path[MAX_PATH_LEN];

        if(argc < 5 || argv[4] == NULL)
        {
            nv_get_absolute_path(yaml_path, YAML_CONFIG_PATH);
            strncat(yaml_path, primary ? "nvipc_primary.yaml" : "nvipc_secondary.yaml", MAX_PATH_LEN - strlen(yaml_path) - 1);
        }
        else
        {
            // Use input YAML configuration file
            nvlog_safe_strncpy(yaml_path, argv[4], MAX_PATH_LEN);
        }

        NVLOGC(TAG, "YAML configuration file: %s", yaml_path);
        load_nv_ipc_yaml_config(&config, yaml_path, module_type);
        ipc_transport = config.ipc_transport;
    }
    else
    {
        load_hard_code_config(&config, primary, ipc_transport);
    }

    if ((ipc = create_nv_ipc_interface(&config)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create IPC interface failed", __func__);
        return -1;
    }

    test_cuda_device_id = get_cuda_device_id_config(&config);

    NVLOGC(TAG, "%s: Initiation finished test_cuda_device_id=%d", __func__, test_cuda_device_id);
    NVLOGD(TAG, "========================================");

#ifdef NVIPC_DPDK_ENABLE
    if(ipc_transport == NV_IPC_TRANSPORT_DPDK && config.transport_config.dpdk.lcore_id > 0)
    {
        atomic_store(&poll_counter, 0);
        create_dpdk_task(dpdk_recv_task, NULL, config.transport_config.dpdk.lcore_id);
        dpdk_main();
    }
    else
#endif
    {
        create_recv_thread();
        while(total_test_slots > 0)
        {
            total_test_slots--;
            usleep(3 * 1000 * 1000);

            if(module_type != NV_IPC_MODULE_PRIMARY)
            {
                if(blocking_flag)
                {
                    blocking_send_task();
                }
                else
                {
                    epoll_send_task();
                }
            }
        }
    }

    if(recv_thread_id != 0)
    {
        // Cancel the receiver thread
        NVLOGI(TAG, "%s: canceling receiver thread", __func__);
        if(pthread_cancel(recv_thread_id) != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_cancel failed, stderr=%s", __func__, strerror(errno));
        }

        // Wait for the receiver thread to finish
        NVLOGI(TAG, "%s: waiting for receiver thread to exit", __func__);
        int ret = pthread_join(recv_thread_id, NULL);
        if(ret != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_join failed, stderr=%s", __func__, strerror(ret));
            return -1;
        }
    }

    nvlog_c_close();
    return 0;
}
