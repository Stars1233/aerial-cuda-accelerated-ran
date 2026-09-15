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
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>

#include <pthread.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include "nv_ipc.h"
#include "nv_ipc_epoll.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 26) // "NVIPC.UDP"

#define FIXED_MSG_BUF_SIZE 1

// Set the socket to non-blocking and create epoll handler
#define UDP_NON_BLOCKING 1

typedef struct
{
    int msg_buf_size;
    int data_buf_size;

    int          local_sock;
    unsigned int counter;

    struct sockaddr_in local_addr;
    struct sockaddr_in remote_addr;

    nv_ipc_epoll_t* ipc_epoll;
} priv_data_t;

static inline priv_data_t* get_private_data(nv_ipc_t* ipc)
{
    return (priv_data_t*)((int8_t*)ipc + sizeof(nv_ipc_t));
}

static int udp_peer_addr_match(const struct sockaddr_in* from, const struct sockaddr_in* expected)
{
    return from != NULL && expected != NULL && from->sin_family == AF_INET
        && from->sin_addr.s_addr == expected->sin_addr.s_addr
        && from->sin_port == expected->sin_port;
}

static int setnonblocking(int sockfd)
{
    int flag = fcntl(sockfd, F_GETFL, 0);
    if(flag < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: get flag failed: %d", __func__, flag);
        return flag;
    }
    if(fcntl(sockfd, F_SETFL, flag | O_NONBLOCK) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: set flag failed: %d", __func__, flag);
    }
    return 0;
}

static int create_udp_socket(uint32_t address, uint16_t port)
{
    // Attempt to initialize socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s create socket failed: sock=%d", __func__, sock);
        return sock;
    }

    if(UDP_NON_BLOCKING)
    {
        setnonblocking(sock);
    }

    // Return socket now if binding is not needed
    if(address == 0 && port == 0)
    {
        return sock;
    }

    // Initialize address structure
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    if(address == 0)
    {
        addr.sin_addr.s_addr = INADDR_ANY;
    }
    else
    {
        addr.sin_addr.s_addr = htonl(address);
    }
    addr.sin_port = htons(port);

    // Attempt to bind socket to the port/address specified
    int ret = bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    if(ret < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s bind socket failed: addr=%u, port=%d, ret=%d", __func__, address, port, ret);
        close(sock);
        return -1;
    }

    NVLOGI(TAG, "Created UDP socket: fd=%d, addr=%lu port=%d", sock, (long unsigned int)address, port);

    return sock;
}

static void set_sockaddr_in(struct sockaddr_in* addr, char* ip, uint16_t port)
{
    bzero(addr, sizeof(struct sockaddr_in));
    addr->sin_family      = AF_INET;
    addr->sin_port        = htons(port);
    addr->sin_addr.s_addr = inet_addr(ip);
}

static int udp_ipc_init(nv_ipc_t* ipc, nv_ipc_config_udp_t* udp_config)
{
    priv_data_t* priv_data = get_private_data(ipc);

    priv_data->msg_buf_size  = udp_config->msg_buf_size;
    priv_data->data_buf_size = udp_config->data_buf_size;

    set_sockaddr_in(&priv_data->local_addr, udp_config->local_addr, udp_config->local_port);
    set_sockaddr_in(&priv_data->remote_addr, udp_config->remote_addr, udp_config->remote_port);
    priv_data->local_sock = create_udp_socket(0, 0);
    if(priv_data->local_sock < 0)
    {
        return -1;
    }

    if(bind(priv_data->local_sock, (struct sockaddr*)&priv_data->local_addr, sizeof(priv_data->local_addr)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s bind socket failed: addr=%s port=%d", __func__,
                udp_config->local_addr, udp_config->local_port);
        close(priv_data->local_sock);
        return -1;
    }

    if(UDP_NON_BLOCKING)
    {
        priv_data->ipc_epoll = ipc_epoll_create(10, priv_data->local_sock);
        if(priv_data->ipc_epoll == NULL)
        {
            return -1;
        }
    }
    else
    {
        priv_data->ipc_epoll = NULL;
    }

    return 0;
}

static int udp_ipc_destroy(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);

    // Close epoll FD if exist
    if(priv_data->ipc_epoll != NULL)
    {
        ipc_epoll_destroy(priv_data->ipc_epoll);
    }

    close(priv_data->local_sock);

    // Destroy the nv_ipc_t instance
    free(ipc);

    return 0;
}

static int get_msg_size(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);
    if(FIXED_MSG_BUF_SIZE == 1)
    {
        return priv_data->msg_buf_size;
    }
    else
    {
        return msg->msg_len;
    }
}

/* For UDP, buffer should be allocated and released for both TX and RX.
 * Just call the system malloc and free functions. */
static int udp_allocate(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options)
{
    msg->msg_len = get_msg_size(ipc, msg);
    msg->msg_buf = malloc(msg->msg_len + msg->data_len);
    if(msg->msg_buf == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: allocate message buffer failed", __func__);
        return -1;
    }

    if(msg->data_len > 0)
    {
        msg->data_buf = (int8_t*)msg->msg_buf + msg->msg_len;
    }
    else
    {
        msg->data_buf = NULL;
        msg->data_len = 0;
    }
    return 0;
}

static int udp_release(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: msg=NULL", __func__);
        return -1;
    }

    if(msg->msg_buf != NULL)
    {
        free(msg->msg_buf);
    }
    return 0;
}

// Send all messages of a TTI and call tx_tti_sem_post() at the end of the TTI
static int udp_tx_send_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);
    int          size      = sendto(priv_data->local_sock, msg->msg_buf, get_msg_size(ipc, msg) + msg->data_len, 0, (struct sockaddr*)(&priv_data->remote_addr), sizeof(struct sockaddr));

    // Free buffer after sent
    free(msg->msg_buf);

    NVLOGV(TAG, "%s: total_len=%d msg_len=%d data_len=%d", __func__, size, msg->msg_len, msg->data_len);

    if(size < 0)
    {
        return size;
    }
    else
    {
        return 0;
    }
}

static int udp_tx_tti_sem_post(nv_ipc_t* ipc)
{
    // Do nothing for UDP IPC
    return 0;
}

static int udp_rx_tti_sem_wait(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    if(priv_data->ipc_epoll != NULL)
    {
        return ipc_epoll_wait(priv_data->ipc_epoll);
    }
    else
    {
        // Do nothing for UDP IPC
        return 0;
    }
}

static int udp_rx_recv_msg(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    struct sockaddr_in from_addr;
    socklen_t          addr_len  = sizeof(from_addr);
    priv_data_t*       priv_data = get_private_data(ipc);

    // Allocate buffer before receive
    size_t  max_size = priv_data->msg_buf_size + priv_data->data_buf_size;
    int8_t* buf      = malloc(max_size);
    if (buf == NULL)
    {
        return -1;
    }

    msg->msg_buf     = buf;
    msg->data_buf    = buf + priv_data->msg_buf_size;

    int size = recvfrom(priv_data->local_sock, msg->msg_buf, max_size, 0, (struct sockaddr*)&from_addr,
            &addr_len);

    if(size < 0)
    {
        free(buf);
        return size;
    }

    if(!udp_peer_addr_match(&from_addr, &priv_data->remote_addr))
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: datagram from unauthorized peer", __func__);
        free(buf);
        return -1;
    }

    int msg_hdr_size = get_msg_size(ipc, msg);
    if(size < msg_hdr_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: datagram too short: size=%d msg_hdr_size=%d", __func__,
                size, msg_hdr_size);
        free(buf);
        return -1;
    }

    int data_len = size - msg_hdr_size;
    msg->msg_len = msg_hdr_size;

    NVLOGV(TAG, "%s: total_len=%d msg_len=%d data_len=%d", __func__, size, msg->msg_len, data_len);

    if(data_len > 0)
    {
        if(data_len > priv_data->data_buf_size)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: data_len %d exceeds buffer %d", __func__, data_len,
                    priv_data->data_buf_size);
            free(buf);
            return -1;
        }
        msg->data_len = data_len;
        msg->data_buf = (int8_t*)msg->msg_buf + msg_hdr_size;
    }
    else
    {
        msg->data_len = 0;
        msg->data_buf = NULL;
    }

    return 0;
}

static int udp_get_fd(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    NVLOGD(TAG, "%s: fd=%d", __func__, priv_data->local_sock);
    return priv_data->local_sock;
}

static int udp_notify(nv_ipc_t* ipc, int value)
{
    return 0;
}

static int udp_get_value(nv_ipc_t* ipc)
{
    return 1;
}

static int cpu_memcpy(nv_ipc_t* ipc, void* dst, const void* src, size_t size)
{
    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc);

    if(size > priv_data->data_buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: size exceeds boundary", __func__);
        return -1;
    }
    else
    {
        // No CUDA device, the data_buf address is in CPU memory, fall back to CPU memory copy
        memcpy(dst, src, size);
        return 0;
    }
}

nv_ipc_t* create_udp_nv_ipc_interface(const nv_ipc_config_t* cfg)
{
    if(cfg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: configuration is NULL", __func__);
        return NULL;
    }

    int       size = sizeof(nv_ipc_t) + sizeof(priv_data_t);
    nv_ipc_t* ipc  = malloc(size);
    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }

    memset(ipc, 0, size);

    priv_data_t* priv_data = get_private_data(ipc);

    ipc->ipc_destroy = udp_ipc_destroy;
    ipc->tx_allocate = udp_allocate;
    ipc->tx_release  = udp_release;
    ipc->rx_allocate = udp_allocate;
    ipc->rx_release  = udp_release;
    ipc->tx_send_msg = udp_tx_send_msg;
    ipc->rx_recv_msg = udp_rx_recv_msg;

    // Semaphore synchronization
    ipc->tx_tti_sem_post = udp_tx_tti_sem_post; // Empty function
    ipc->rx_tti_sem_wait = udp_rx_tti_sem_wait; // Empty function

    // FD synchronization
    ipc->get_fd    = udp_get_fd;    // Return the UDP socket FD
    ipc->get_value = udp_get_value; // Empty function
    ipc->notify    = udp_notify;    // Empty function

    ipc->cuda_memcpy_to_host   = cpu_memcpy;
    ipc->cuda_memcpy_to_device = cpu_memcpy;

    if(udp_ipc_init(ipc, (nv_ipc_config_udp_t*)&cfg->transport_config.udp) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed", __func__);
        udp_ipc_destroy(ipc);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: OK", __func__);
        return ipc;
    }
}
