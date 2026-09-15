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
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <pthread.h>
#include <sys/queue.h>
#include <sys/eventfd.h>
#include <sys/time.h>

#include <rte_common.h>
#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_atomic.h>
#include <rte_branch_prediction.h>
#include <rte_ring.h>
#include <rte_log.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_bus_pci.h>

#if (DOCA_GPU_DPDK_OLD == 1)
#include <rte_bus_auxiliary.h>
#else
#include <rte_bus.h>
#endif

#include "nv_ipc.h"
#include "nv_ipc_mempool.h"
#include "nv_ipc_ring.h"
#include "nv_ipc_epoll.h"
#include "nv_ipc_dpdk.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_dpdk_utils.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_debug.h"
#include "nv_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 20) //"NVIPC.DPDK"

#define DEBUG_DPDK_IPC 0

#define USE_SHM_MEMPOOL_RX
#define USE_SHM_MEMPOOL_TX
// #define MULTI_SEGS_SEND
// #define MBUF_CHAIN_LINK

#define CONFIG_DPDK_DL_TTI_SYNC 0

#define MSG_ID_DL_TTI_SYNC (0x70)
#define MSG_ID_CONNECT (-3)

#define CONNECT_TEST_COUNT 1

#ifdef BUILD_NVIPC_ONLY
#define RTE_LAUNCH_THREAD 0
#else
#define RTE_LAUNCH_THREAD 0
#endif

#define CUST_ETHER_TYPE_NVIPC 0xAE55

#define NV_NAME_SUFFIX_MAX_LEN 16

#define BURST_SIZE 512
#define BURST_QUEUE_SIZE (BURST_SIZE * 16)
#define MBUF_CACHE_SIZE 250

#define RX_BUF_X 8

#define TX_RING_SIZE 4096
#define RX_RING_SIZE (TX_RING_SIZE * RX_BUF_X)

#define CONFIG_EPOLL_EVENT_MAX 1

#define IPC_TEST_EN 0

typedef struct
{
    int primary;

    struct rte_mempool* rxpool;
    struct rte_mempool* txpool;

    // rx_burst receive cache
    nv_ipc_ring_t* rx_ring;
    nv_ipc_ring_t* tx_ring;

    nv_ipc_ring_t* rx_free_ring;

    // Large buffer memory pool for DATA part
    nv_ipc_mempool_t* mempool;

    nv_ipc_epoll_t* ipc_epoll;

    int32_t msg_payload_size;
    int32_t data_payload_size;
    int32_t max_chain_length;
    int32_t mbuf_payload_size;
    int32_t max_rx_pkt_len;

    uint16_t nic_port;
    uint16_t nic_mtu;

    struct rte_ether_addr src;
    struct rte_ether_addr dst;

    int efd_rx;
    int cuda_device_id;

    uint16_t  lcore_id;
    pthread_t thread_id;

    pthread_mutex_t tx_lock;
    pthread_mutex_t rx_lock;

    uint64_t poll_counter;
    uint64_t rx_pkts;

    // For debug
    struct timeval tv_last;

    nv_ipc_debug_t* ipc_debug;

} priv_data_t;

static priv_data_t* get_private_data(nv_ipc_t* ipc)
{
    return (priv_data_t*)((int8_t*)ipc + sizeof(nv_ipc_t));
}

#define MUTEX_LOCK(lock_ptr)                                  \
    do                                                        \
    {                                                         \
        if(pthread_mutex_lock(lock_ptr) != 0)                 \
        {                                                     \
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: mutex lock failed", __func__); \
            return -1;                                        \
        }                                                     \
    } while(0)

#define MUTEX_UNLOCK(lock_ptr)                                  \
    do                                                          \
    {                                                           \
        if(pthread_mutex_unlock(lock_ptr) != 0)                 \
        {                                                       \
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: mutex unlock failed", __func__); \
            return -1;                                          \
        }                                                       \
    } while(0)

#if 1
#define TX_LOCK() MUTEX_LOCK(&priv_data->tx_lock)
#define TX_UNLOCK() MUTEX_UNLOCK(&priv_data->tx_lock)
#define RX_LOCK() MUTEX_LOCK(&priv_data->rx_lock)
#define RX_UNLOCK() MUTEX_UNLOCK(&priv_data->rx_lock)
#else
#define TX_LOCK()
#define TX_UNLOCK()
#endif

typedef struct
{
    struct rte_mbuf* head;
    struct rte_mbuf* tail;
} mbuf_list_t;

static int mbuf_release(priv_data_t* priv_data, struct rte_mbuf* mbuf)
{
    //    if (pthread_mutex_lock(&priv_data->rx_lock) != 0)
    //    {
    //        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: mutex rx lock failed", __func__);
    //        return -1;
    //    }

    if(mbuf == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: mbuf=null", __func__);
        return -1;
    }

    nvipc_hdr_t* head     = get_nvipc_hdr(mbuf);
    int          cell_id  = head->cell_id;
    int          msg_id   = head->msg_id;
    int          msg_len  = head->msg_len;
    int          data_len = head->data_len;
    int          count    = head->nb_segs;
    if(count > priv_data->max_chain_length)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "FREE: error: count=%d max=%d msg_id=0x%02X msg_len=%d data_len=%d", count, priv_data->max_chain_length, msg_id, msg_len, data_len);
    }

    int              index;
    struct rte_mbuf* curr = mbuf;
    for(index = 0; index < count && curr != NULL; index++)
    {
        struct rte_mbuf* next = get_nvipc_hdr(curr)->next;
        rte_pktmbuf_free(curr);
        curr = next;
    }

    NVLOGD(TAG, "FREE: msg_id=0x%02X msg_len=%d data_len=%d count=%d-%d", msg_id, msg_len, data_len, count, index);

    //    if (pthread_mutex_unlock(&priv_data->rx_lock) != 0)
    //    {
    //        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: mutex rx unlock failed", __func__);
    //        return -1;
    //    }

    return 0;
}

int nvipc_hdr_valid(struct rte_mbuf* mbuf)
{
    nvipc_hdr_t* curr = get_nvipc_hdr(mbuf);
    nvipc_hdr_t* next = get_nvipc_hdr(curr->next);
    if((curr->next == NULL && curr->nb_segs != 1) || (curr->next != NULL && curr->nb_segs == 1))
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Invalid mbuf[%p-%u]: next[%p-%u]", mbuf, curr->nb_segs, curr->next, next == NULL ? 0 : next->nb_segs);
        return 0;
    }
    else
    {
        return 1;
    }
}

void list_init(mbuf_list_t* list)
{
    list->head = NULL;
    list->tail = NULL;
}

int list_is_empty(mbuf_list_t* list)
{
    return (list->head == NULL && list->tail == NULL) ? 1 : 0;
}

int list_is_ended(mbuf_list_t* list)
{
    if(list->head == NULL || list->tail == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Empty mbuf list");
        return 0;
    }
    nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(list->tail);
    // Check
    //    if((nvipc_hdr->next == NULL && nvipc_hdr->nb_segs != 1) || (nvipc_hdr->next != NULL && nvipc_hdr->nb_segs == 1))
    //    {
    //        nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(list->head);
    //        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Invalid mbuf_list[%p-%u]: %p-%u", list->head, nvipc_hdr->nb_segs, nvipc_hdr->next, nvipc_hdr->nb_segs);
    //    }
    return (nvipc_hdr->next == NULL && nvipc_hdr->nb_segs == 1) ? 1 : 0;
}

void list_append(mbuf_list_t* list, struct rte_mbuf* mbuf)
{
    // Check
    nvipc_hdr_valid(mbuf);

    if(list->tail == NULL)
    {
        list->head = mbuf;
        list->tail = mbuf;
    }
    else
    {
        nvipc_hdr_t* old_hdr = get_nvipc_hdr(list->tail);
        nvipc_hdr_t* new_hdr = get_nvipc_hdr(mbuf);

        // Check
        // if(old_hdr->next == NULL || old_hdr->nb_segs != new_hdr->nb_segs + 1 || (new_hdr->nb_segs == 1 && new_hdr->next != NULL))
        if(old_hdr->nb_segs != new_hdr->nb_segs + 1)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "appending invalid mbuf: %u-%u %p-%p", old_hdr->nb_segs, new_hdr->nb_segs, list->tail, mbuf);
        }

        old_hdr->next = mbuf;
        new_hdr->next = NULL;
        list->tail    = mbuf;
    }
}

void list_dump(mbuf_list_t* list)
{
    if(list == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "LIST is NULL");
        return;
    }

    struct rte_mbuf* curr  = list->head;
    int              count = 0;
    for(curr = list->head; curr != NULL; curr = get_nvipc_hdr(curr)->next)
    {
        nvipc_hdr_t* hdr = get_nvipc_hdr(curr);
        NVLOGI(TAG, "LIST[%u-%u]: msg_id=0x%02X msg_len=%d data_len=%d mbuf_len=%u-%u", hdr->seg_num, hdr->seg_id, hdr->msg_id, hdr->msg_len, hdr->data_len, curr->pkt_len, curr->data_len);
        count++;
    }
    NVLOGI(TAG, "LIST dump ended: count=%d", count);
}

static struct rte_eth_conf port_conf_default = {
    .rxmode = {
#if !DOCA_GPU_DPDK
        .mtu = RTE_ETHER_MAX_LEN,
#endif
    },
#ifdef MULTI_SEGS_SEND
    .txmode = {
        .offloads = DEV_TX_OFFLOAD_MULTI_SEGS,
    },
#endif
};

static int mac_str_to_eth_addr(const char* mac_str, struct rte_ether_addr* eth_addr)
{
    char log[32];
    char str[8];
    for(int i = 0; i < 6; i++)
    {
        char* err_ptr           = NULL;
        str[0]                  = mac_str[i * 3];
        str[1]                  = mac_str[i * 3 + 1];
        str[2]                  = '\0';
        eth_addr->addr_bytes[i] = strtol(str, &err_ptr, 16); // Octal, Decimal, Hex
        snprintf(log + i * 3, 4, "%02x:", eth_addr->addr_bytes[i]);
    }
    // NVLOGI(TAG, "%s: %s ==> %s", __func__, s, log);
    return 0;
}

static int eth_addr_to_mac_str(char* mac_str, struct rte_ether_addr* eth_addr)
{
    int offset = 0;
    for(int i = 0; i < 5; i++)
    {
        offset += snprintf(mac_str + offset, 4, "%02x:", eth_addr->addr_bytes[i]);
    }
    offset += snprintf(mac_str + offset, 3, "%02x", eth_addr->addr_bytes[5]);
    return offset;
}

static void print_eth_hdr(priv_data_t* priv_data, struct rte_mbuf* mbuf, const char* info)
{
    char         buf[128];
    nvipc_hdr_t* hdr = get_nvipc_hdr(mbuf);

    int offset = 0;
    offset += snprintf(buf + offset, 6, "ETH: ");
    for(int i = 0; i < 6; i++)
    {
#if (DOCA_GPU_DPDK == 0 || DOCA_GPU_DPDK_OLD == 0)
        offset += snprintf(buf + offset, 3, "%02x", hdr->eth_hdr.src_addr.addr_bytes[i]);
#else
        offset += snprintf(buf + offset, 3, "%02x", hdr->eth_hdr.s_addr.addr_bytes[i]);
#endif
        if(i < 5)
        {
            offset += snprintf(buf + offset, 2, ":");
        }
    }
    offset += snprintf(buf + offset, 6, " ==> ");
    for(int i = 0; i < 6; i++)
    {
#if (DOCA_GPU_DPDK == 0 || DOCA_GPU_DPDK_OLD == 0)
        offset += snprintf(buf + offset, 3, "%02x", hdr->eth_hdr.dst_addr.addr_bytes[i]);
#else
        offset += snprintf(buf + offset, 3, "%02x", hdr->eth_hdr.d_addr.addr_bytes[i]);
#endif
        if(i < 5)
        {
            offset += snprintf(buf + offset, 2, ":");
        }
    }
    offset += snprintf(buf + offset, 13, " type=0x%04X", hdr->eth_hdr.ether_type);
    NVLOGD(TAG, "%s: %s data_len=%u pkt_len=%u nb_segs=%u next=%p", info, buf, mbuf->data_len, mbuf->pkt_len, mbuf->nb_segs, mbuf->next);
}

static void list_all_nic_ports()
{
    uint16_t port;
    char     tmp_buf[64];
    RTE_ETH_FOREACH_DEV(port)
    {
        // Get PCI address
        struct rte_eth_dev_info dev_info;
#if (DOCA_GPU_DPDK_OLD == 1)
        struct rte_pci_device*  pci_dev;
#else
        struct rte_pci_addr     pci_addr;
#endif
        struct rte_bus*         bus = NULL;
        int                     offset = 0;

        rte_eth_dev_info_get(port, &dev_info);
        if(dev_info.device)
        {
            bus = rte_bus_find_by_device(dev_info.device);
        }

#if (DOCA_GPU_DPDK_OLD == 1)
        if(bus && !strcmp(bus->name, "pci"))
        {
            pci_dev = RTE_DEV_TO_PCI(dev_info.device);
            offset += snprintf(tmp_buf, 20, "pci %04x:%02x:%02x.%x", pci_dev->addr.domain, pci_dev->addr.bus, pci_dev->addr.devid, pci_dev->addr.function);
        }
        else if(bus && !strcmp(bus->name, "auxiliary"))
        {
            struct rte_auxiliary_device* aux_dev = RTE_DEV_TO_AUXILIARY(dev_info.device);
            NVLOGC(TAG, "Port %u: auxiliary %s", port, aux_dev->name);
            continue;
        }
        else
        {
            NVLOGC(TAG, "Port %u: bus->name=%s", port, bus->name);
            continue;
        }
#else
        if(bus && !strcmp(rte_bus_name(bus), "pci"))
        {
            rte_pci_addr_parse(rte_dev_name(dev_info.device), &pci_addr);
            offset += snprintf(tmp_buf, 20, "pci %04x:%02x:%02x.%x", pci_addr.domain, pci_addr.bus, pci_addr.devid, pci_addr.function);
        }
        else
        {
            NVLOGC(TAG, "Port %u: bus->name=%s", port, rte_bus_name(bus));
            continue;
        }
#endif

        // Get MAC address
        struct rte_ether_addr mac;
        int                   retval = rte_eth_macaddr_get(port, &mac);
        if(rte_eth_macaddr_get(port, &mac) == 0)
        {
            offset += snprintf(tmp_buf + offset, 20, " %02x:%02x:%02x:%02x:%02x:%02x", mac.addr_bytes[0], mac.addr_bytes[1], mac.addr_bytes[2], mac.addr_bytes[3], mac.addr_bytes[4], mac.addr_bytes[5]);
        }

        if(retval != 0)
        {
            NVLOGC(TAG, "Port %u: rte_eth_macaddr_get failed, skip", port);
            continue;
        }

        NVLOGC(TAG, "Port %u: %s", port, tmp_buf);
    }
}

static int port_init(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);

    uint16_t port   = priv_data->nic_port;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;

    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t      rx_rings = 1, tx_rings = 1;
#if !DOCA_GPU_DPDK
    port_conf.rxmode.mtu = priv_data->max_rx_pkt_len;
#endif
    // port_conf.rxmode.max_rx_pkt_len = priv_data->mbuf_data_room_size - RTE_PKTMBUF_HEADROOM;
    // port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_JUMBO_FRAME;

    int                     retval = 0;
    uint16_t                q;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf   txconf;

    if(!rte_eth_dev_is_valid_port(port))
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_dev_is_valid_port: (port %u) info: %s", port, strerror(-retval));
        return -1;
    }

    retval = rte_eth_dev_info_get(port, &dev_info);
    if(retval != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error during getting device (port %u) info: %s", port, strerror(-retval));
        return retval;
    }

#if (DOCA_GPU_DPDK_OLD == 1)
    if(dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    {
        NVLOGC(TAG, "tx_offload_capa=0x%X DEV_TX_OFFLOAD_MBUF_FAST_FREE supported", dev_info.tx_offload_capa);
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    } else {
        NVLOGC(TAG, "tx_offload_capa=0x%X DEV_TX_OFFLOAD_MBUF_FAST_FREE not supported", dev_info.tx_offload_capa);
    }
#else
    if(dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE)
    {
        NVLOGC(TAG, "tx_offload_capa=0x%lX RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE supported", dev_info.tx_offload_capa);
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
    } else {
        NVLOGC(TAG, "tx_offload_capa=0x%lX RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE not supported", dev_info.tx_offload_capa);
    }
#endif
    /* Configure the Ethernet device. */
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if(retval != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_dev_configure: (port %u) info: %s", port, strerror(-retval));
        return retval;
    }

    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if(retval != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_dev_adjust_nb_rx_tx_desc: (port %u) info: %s", port, strerror(-retval));
        return retval;
    }

    /* Allocate and set up 1 RX queue per Ethernet port. */
    for(q = 0; q < rx_rings; q++)
    {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd, rte_eth_dev_socket_id(port), NULL, priv_data->rxpool);
        if(retval < 0)
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_rx_queue_setup: (port %u) info: %s", port, strerror(-retval));
            return retval;
        }
    }

    txconf          = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    /* Allocate and set up 1 TX queue per Ethernet port. */
    for(q = 0; q < tx_rings; q++)
    {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd, rte_eth_dev_socket_id(port), &txconf);
        if(retval < 0)
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_tx_queue_setup: (port %u) info: %s", port, strerror(-retval));
            return retval;
        }
    }

    /* Starting Ethernet port. 8< */
    retval = rte_eth_dev_start(port);
    /* >8 End of starting of ethernet port. */
    if(retval < 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_tx_queue_setup: (port %u) info: %s", port, strerror(-retval));
        return retval;
    }

    /* Display the port MAC address. */
    struct rte_ether_addr addr;
    retval = rte_eth_macaddr_get(port, &addr);
    if(retval != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "Error: rte_eth_macaddr_get: (port %u) info: %s", port, strerror(-retval));
        return retval;
    }

    NVLOGC(TAG, "Use port %u MAC: %02x:%02x:%02x:%02x:%02x:%02x", port, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2], addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5]);

    /* Enable RX in promiscuous mode for the Ethernet device. */
    // retval = rte_eth_promiscuous_enable(port);
    // if(retval != 0)
    //    return retval;

    return 0;
}

static inline int nvipc_hdr_filter(priv_data_t* priv_data, struct rte_mbuf* mbuf)
{
    nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(mbuf);
    if(nvipc_hdr->vlan_hdr.eth_proto == rte_cpu_to_be_16(CUST_ETHER_TYPE_NVIPC))
    {
        return 0;
    }
    else
    {
        // print_eth_hdr(priv_data, mbuf, "skip incoming packet");
        NVLOGD(TAG, "%s: skip incoming packet: protocol=0x%04X", __func__, nvipc_hdr->vlan_hdr.eth_proto);
        return -1;
    }
}

static int copy_buf_to_mbuf(priv_data_t* priv_data, struct rte_mbuf** mbufs, uint8_t* src, uint32_t size)
{
    int count = (size + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
    for(uint32_t index = 0; index < count; index++)
    {
        struct rte_mbuf* mbuf         = mbufs[index];
        uint32_t         payload_size = index < count - 1 ? priv_data->mbuf_payload_size : size % priv_data->mbuf_payload_size;
        uint32_t         offset       = priv_data->mbuf_payload_size * index;

        nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(mbuf);
        memcpy(nvipc_hdr->payload, src + offset, payload_size);

        mbuf->data_len = payload_size + sizeof(nvipc_hdr_t);
        mbuf->pkt_len  = mbuf->data_len;
    }
    return count;
}

static struct rte_mbuf* copy_mbuf_to_buf(priv_data_t* priv_data, struct rte_mbuf* mbuf, uint8_t* dest, uint32_t size, uint32_t dest_size)
{
    // nvipc_hdr_t* head = get_nvipc_hdr(mbuf);
    if(size > dest_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: copy size %u exceeds destination size %u", __func__, size, dest_size);
        return (struct rte_mbuf*)-1;
    }

    uint32_t count = (size + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;

    uint32_t         index = 0;
    struct rte_mbuf* curr  = mbuf;
    for(index = 0; index < count && curr != NULL; index++)
    {
        uint32_t payload_size = index < count - 1 ? priv_data->mbuf_payload_size : size % priv_data->mbuf_payload_size;
        uint32_t offset       = priv_data->mbuf_payload_size * index;
        if(curr->data_len != payload_size + sizeof(nvipc_hdr_t))
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "mbuf size doens't match [%u-%u]: data_len=%u payload_size=%u sizeof(nvipc_hdr_t)=%lu", count, index, mbuf->data_len, payload_size, sizeof(nvipc_hdr_t));
            return (struct rte_mbuf*)-1;
        }

        nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(curr);
        memcpy(dest + offset, nvipc_hdr->payload, payload_size);
        curr = nvipc_hdr->next;
    }
    if(index != count)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "mbuf size doens't match: count=%u index=%u", count, index);
        return (struct rte_mbuf*)-1;
    }
    return curr;
}

static int copy_mbufs_from_nvipc(priv_data_t* priv_data, struct rte_mbuf** mbufs, nv_ipc_msg_t* msg)
{
    int count = 1;
    if(msg->msg_len > priv_data->mbuf_payload_size)
    {
        // Copy from the second mbuf for MSG part
        count += copy_buf_to_mbuf(priv_data, mbufs + count, (uint8_t*)msg->msg_buf + priv_data->mbuf_payload_size, msg->msg_len - priv_data->mbuf_payload_size);
    }

    if(msg->data_len > 0)
    {
        count += copy_buf_to_mbuf(priv_data, mbufs + count, msg->data_buf, msg->data_len);
    }
    return count;
}

static int copy_mbufs_to_nvipc(priv_data_t* priv_data, nvipc_hdr_t* head)
{
    struct rte_mbuf* next = head->next;
    if(head->msg_len < 0 || head->msg_len > priv_data->msg_payload_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_len=%d msg_payload_size=%d",
                __func__, head->msg_len, priv_data->msg_payload_size);
        return -1;
    }
    if(head->data_len < 0 || head->data_len > priv_data->data_payload_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_len=%d data_payload_size=%d",
                __func__, head->data_len, priv_data->data_payload_size);
        return -1;
    }
    if(head->msg_len > priv_data->mbuf_payload_size)
    {
        // Copy from the second mbuf for MSG part
        next = copy_mbuf_to_buf(priv_data, next, head->payload + priv_data->mbuf_payload_size,
                head->msg_len - priv_data->mbuf_payload_size, priv_data->msg_payload_size - priv_data->mbuf_payload_size);
        if(next == (struct rte_mbuf*)-1)
        {
            return -1;
        }
    }
    if(head->data_len > 0)
    {
        next = copy_mbuf_to_buf(priv_data, next, head->data_buf, head->data_len, priv_data->data_payload_size);
        if(next == (struct rte_mbuf*)-1)
        {
            return -1;
        }
    }
    if(next != NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: copy not ended: nb_segs=%u", __func__, get_nvipc_hdr(next)->nb_segs);
    }
    return 0;
}

static int edit_nvipc_header(priv_data_t* priv_data, struct rte_mbuf* mbuf, nv_ipc_msg_t* msg, uint16_t seg_num, uint16_t seg_id)
{
    nvipc_hdr_t* nvipc_hdr = get_nvipc_hdr(mbuf);
#if (DOCA_GPU_DPDK == 0 || DOCA_GPU_DPDK_OLD == 0)
    NVLOGV(TAG, "%s: msg_id=0x%02x mbuf=%p nvipc_hdr=%p dst_addr=%p src_addr=%p", __func__, msg->msg_id, mbuf, nvipc_hdr, &nvipc_hdr->eth_hdr.dst_addr, &nvipc_hdr->eth_hdr.src_addr);
#else
    NVLOGV(TAG, "%s: msg_id=0x%02x mbuf=%p nvipc_hdr=%p dst_addr=%p src_addr=%p", __func__, msg->msg_id, mbuf, nvipc_hdr, &nvipc_hdr->eth_hdr.d_addr, &nvipc_hdr->eth_hdr.s_addr);
#endif

    // Ethernet header
#if (DOCA_GPU_DPDK == 0 || DOCA_GPU_DPDK_OLD == 0)
    memcpy(&nvipc_hdr->eth_hdr.dst_addr, &priv_data->dst, sizeof(struct rte_ether_addr));
    memcpy(&nvipc_hdr->eth_hdr.src_addr, &priv_data->src, sizeof(struct rte_ether_addr));
#else
    memcpy(&nvipc_hdr->eth_hdr.d_addr, &priv_data->dst, sizeof(struct rte_ether_addr));
    memcpy(&nvipc_hdr->eth_hdr.s_addr, &priv_data->src, sizeof(struct rte_ether_addr));
#endif
    nvipc_hdr->eth_hdr.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    // VLAN header
    nvipc_hdr->vlan_hdr.vlan_tci  = rte_cpu_to_be_16(2);
    nvipc_hdr->vlan_hdr.eth_proto = rte_cpu_to_be_16(CUST_ETHER_TYPE_NVIPC); // RTE_ETHER_TYPE_IPV4

    nvipc_hdr->seg_num = seg_num; // The segment count
    nvipc_hdr->seg_id  = seg_id;  // The segment index

    nvipc_hdr->msg_id    = msg->msg_id;
    nvipc_hdr->cell_id   = msg->cell_id;
    nvipc_hdr->msg_len   = msg->msg_len;
    nvipc_hdr->data_len  = msg->data_len;
    nvipc_hdr->data_pool = msg->data_pool;

    mbuf->nb_segs = 1;
    mbuf->next    = NULL;

    return 0;
}

static int check_mbuf_chain(priv_data_t* priv_data, struct rte_mbuf* mbuf_head, nv_ipc_msg_t* msg, const char* info)
{
    if(DEBUG_DPDK_IPC == 0)
    {
        return 0;
    }

    int ret = 0;
#ifdef MBUF_CHAIN_LINK
    struct rte_mbuf* head = mbuf_head;
    struct rte_mbuf* next = head->next;
    struct rte_mbuf* curr = head;
#else
    nvipc_hdr_t* head = get_nvipc_hdr(mbuf_head);
    nvipc_hdr_t* next = get_nvipc_hdr(head->next);
    nvipc_hdr_t* curr = head;
#endif
    struct rte_mbuf* mbuf_curr = mbuf_head;
    struct rte_mbuf* mbuf_next = curr->next;

    for(uint32_t index = 0; index < head->nb_segs; index++)
    {
#ifdef MBUF_CHAIN_LINK
        next = curr->next;
#else
        next = get_nvipc_hdr(curr->next);
#endif
        mbuf_next = curr->next;
        if(curr == NULL || head->nb_segs == 0 || curr->nb_segs + index != head->nb_segs || (next == NULL && index != head->nb_segs - 1))
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: chain[%p-%u-%u]: i=%u curr=%p next=%p next->nb_segs=%u Error", info, head, head->nb_segs, msg->data_len, index, mbuf_curr, mbuf_next, next == NULL ? 0 : next->nb_segs);
            return -1;
        }
        else
        {
            NVLOGD(TAG, "%s: chain[%p-%u-%u]: i=%u curr=%p next=%p next->nb_segs=%u", info, head, head->nb_segs, msg->data_len, index, mbuf_curr, mbuf_next, next == NULL ? 0 : next->nb_segs);
        }
        mbuf_curr = curr->next;
        curr      = next;
    }
    uint32_t count = (msg->msg_len + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size + 1;
    count += (msg->data_len + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size + 1;
    if(head->nb_segs != count || curr != NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: chain[%p-%u-%u]: end=%p Error", info, head, head->nb_segs, msg->data_len, index, curr);
        ret = -1;
    }
    return ret;
}

static int dpdk_nic_send(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);

    struct rte_mbuf* mbuf_head = get_rte_mbuf(msg->msg_buf);
    struct rte_mbuf* mbuf      = mbuf_head;

    if(msg->msg_buf == NULL || mbuf_head == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: error: msg->msg_buf=%p mbuf_head=%p", __func__, msg->msg_buf, mbuf_head);
        return -1;
    }

    mbuf_head->data_len = msg->msg_len < priv_data->mbuf_payload_size ? msg->msg_len : priv_data->mbuf_payload_size;
    mbuf_head->data_len += sizeof(nvipc_hdr_t);
    mbuf_head->pkt_len = mbuf_head->data_len;

#ifdef MBUF_CHAIN_LINK
    struct rte_mbuf* head = mbuf_head;
    struct rte_mbuf* tail = head;
    struct rte_mbuf* curr = head;
#else
    nvipc_hdr_t* head = get_nvipc_hdr(mbuf_head);
    nvipc_hdr_t* tail = head;
    nvipc_hdr_t* curr = head;
#endif

    int count = (msg->msg_len + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
    if(msg->data_pool > 0 && msg->data_pool < NV_IPC_MEMPOOL_NUM)
    {
        count += (msg->data_len + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
    }

    NVLOGI(TAG, "SEND: cell_id=%d msg_id=0x%02X msg_len=%u data_len=%u count=%d", msg->cell_id, msg->msg_id, msg->msg_len, msg->data_len, count);

    struct rte_mbuf** send_mbufs = malloc(sizeof(struct rte_mbuf*) * count);
    if (send_mbufs == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: malloc failed", __func__);
        return -1;
    }

    send_mbufs[0]                = mbuf_head;
    if(count > 1)
    {
        TX_LOCK();
        if(rte_pktmbuf_alloc_bulk(priv_data->txpool, send_mbufs + 1, count - 1) != 0)
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_pktmbuf_alloc_bulk failed: count=%d msg_id=0x%02X data_len=%d", __func__, count, msg->msg_id, msg->data_len);
            free(send_mbufs);
            TX_UNLOCK();
            return -1;
        }
        TX_UNLOCK();
    }

#ifdef USE_SHM_MEMPOOL_TX
    int copied_count = copy_mbufs_from_nvipc(priv_data, send_mbufs, msg);
    if(copied_count != count)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: mbuf count doesn't match: copied=%d count=%d", __func__, copied_count, count);
    }

    if(msg->data_buf != NULL)
    {
        int data_index = priv_data->mempool->get_index(priv_data->mempool, msg->data_buf);
        if(priv_data->mempool->free(priv_data->mempool, data_index) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: failed to free SHM: data_buf=%p data_index=%d", __func__, msg->data_buf, data_index);
        }
    }
#endif

    // Set mbuf_self pointer and link as chain if there's multiple mbufs
    for(uint32_t index = 0; index < count; index++)
    {
        struct rte_mbuf* curr_mbuf = send_mbufs[index];
        if(index < count - 1 && curr_mbuf == send_mbufs[index + 1])
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "same: chain[%p-%u] OK: next=%p mbufs[%d]=%p", curr_mbuf, curr_mbuf->nb_segs, curr_mbuf->next, index + 1, send_mbufs[index + 1]);
        }
        set_mbuf_self(curr_mbuf);

#ifdef MBUF_CHAIN_LINK
        curr_mbuf->nb_segs = count - index;
        curr_mbuf->next    = index == count - 1 ? NULL : send_mbufs[index + 1];
        NVLOGV(TAG, "link: chain[%p-%u] OK: next->nb_segs=%u", curr_mbuf, curr_mbuf->nb_segs, curr_mbuf->next == NULL ? 0 : curr_mbuf->next->nb_segs);
#else
        nvipc_hdr_t* hdr = get_nvipc_hdr(curr_mbuf);
        hdr->nb_segs     = count - index;
        hdr->next        = index == count - 1 ? NULL : send_mbufs[index + 1];
        NVLOGV(TAG, "link: chain[%p-%u] i=%u curr=%p next=%p next->nb_segs=%u", curr_mbuf, hdr->nb_segs, index, curr_mbuf, hdr->next, hdr->next == NULL ? 0 : get_nvipc_hdr(hdr->next)->nb_segs);
#endif
        edit_nvipc_header(priv_data, curr_mbuf, msg, count, index);
    }

    if(head->nb_segs > priv_data->max_chain_length)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: too large segments: max_chain_len=%u nb_segs=%u ", __func__, priv_data->max_chain_length, head->nb_segs);
        free(send_mbufs);
        return -1;
    }

    if(check_mbuf_chain(priv_data, mbuf_head, msg, "SEND") < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: check failed. head->nb_segs=%u", __func__, head->nb_segs);
        free(send_mbufs);
        return -1;
    }

    TX_LOCK();

    int ret = 0;
#if 1 // burst send
    uint16_t nb_tx = rte_eth_tx_burst(priv_data->nic_port, 0, send_mbufs, count);
    if(nb_tx != count)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: failed nb_tx=%u nb_segs=%u", __func__, nb_tx, head->nb_segs);
        ret = -1;
    }
#else // send one by one
    for(int index = 0; index < count; index++)
    {
        mbuf             = send_mbufs[index];
        nvipc_hdr_t* hdr = get_nvipc_hdr(mbuf);

        uint16_t nb_tx;
        if((nb_tx = rte_eth_tx_burst(priv_data->nic_port, 0, &mbuf, 1)) != 1)
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "SEND: RAW: %d-%d mbuf_segs=%u msg_id=0x%02X msg_len=%d data_len=%d failed: nb_tx=%d", count, index, mbuf->nb_segs, msg->msg_id, msg->msg_len, msg->data_len, nb_tx);
            ret = -1;
            break;
        }
        NVLOGV(TAG, "SEND: RAW: %d-%d mbuf_len=%u msg_id=0x%02X msg_len=%d data_len=%d OK", hdr->nb_segs, hdr->seg_id, mbuf->data_len, msg->msg_id, msg->msg_len, msg->data_len);
    }
#endif

    // mbuf_release(mbuf_head);

    TX_UNLOCK();

    free(send_mbufs);
    return ret;
}

static int send_tti_sync_event(nv_ipc_t* ipc, int value)
{
    priv_data_t* priv_data = get_private_data(ipc);

    nv_ipc_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    if(ipc->tx_allocate(ipc, &msg, 0) < 0)
    {
        return -1;
    }

    msg.msg_id  = MSG_ID_DL_TTI_SYNC;
    msg.msg_len = sizeof(value);
    memcpy(msg.msg_buf, &value, sizeof(value));

    if(ipc->tx_send_msg(ipc, &msg) < 0)
    {
        return -1;
    }
    return 0;
}

int recv_efd_notify(priv_data_t* priv_data, struct rte_mbuf* mbuf)
{
    nvipc_hdr_t* hdr = get_nvipc_hdr(mbuf);

    int ret;
    if(CONFIG_DPDK_DL_TTI_SYNC && hdr->msg_id == MSG_ID_DL_TTI_SYNC)
    {
        NVLOGI(TAG, "RECV: TTI SYNC notification: cell_id=%d msg_id=0x%02X value=%d", hdr->cell_id, hdr->msg_id, *(int*)get_nvipc_payload(mbuf));
        ret = mbuf_release(priv_data, mbuf);
    }
    else
    {
        NVLOGI(TAG, "RECV: enqueue: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%u", hdr->cell_id, hdr->msg_id, hdr->msg_len, hdr->data_len);
        ret = priv_data->rx_ring->enqueue(priv_data->rx_ring, &mbuf);
    }

    if(CONFIG_DPDK_DL_TTI_SYNC == 0 || priv_data->primary == 0 || (priv_data->primary && hdr->msg_id == MSG_ID_DL_TTI_SYNC))
    {
        uint64_t efd_value = 1;
        ssize_t  size      = write(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
        if(size < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: efd write failed: size=%lu", __func__, size);
            return -1;
        }
    }

    return ret;
}

int dpdk_nic_recv_poll(priv_data_t* priv_data)
{
    uint16_t nb_rx;

    mbuf_list_t list;
    list_init(&list);

    do
    {
        struct rte_mbuf* to_free;
        while(priv_data->rx_free_ring->dequeue(priv_data->rx_free_ring, &to_free) == 0)
        {
            mbuf_release(priv_data, to_free);
        }

        struct rte_mbuf* recv_mbufs[BURST_SIZE];
        nb_rx = rte_eth_rx_burst(priv_data->nic_port, 0, recv_mbufs, BURST_SIZE);
        if(nb_rx > BURST_SIZE)
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: received count nb_rx=%u > BURST_SIZE=%u", __func__, nb_rx, BURST_SIZE);
            nb_rx = BURST_SIZE;
        }

        // Periodic logging info
        priv_data->poll_counter++;
        priv_data->rx_pkts += nb_rx;
        int available = rte_mempool_avail_count(priv_data->rxpool);
        if(priv_data->poll_counter % 10000 == 0 || available < priv_data->max_chain_length)
        {
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            int64_t interval = nvlog_timeval_interval(&priv_data->tv_last, &tv_now);
            if(nvlog_timeval_interval(&priv_data->tv_last, &tv_now) > 5 * 1000L * 1000)
            {
                NVLOGI(TAG, "POLL: available=%d nb_rx=%u rx_pkts=%lu poll_counter=%lu", available, nb_rx, priv_data->rx_pkts, priv_data->poll_counter);
                priv_data->tv_last = tv_now;
            }
        }

        struct rte_mbuf* mbuf_curr;
        for(uint16_t index = 0; index < nb_rx; index++)
        {
            mbuf_curr = recv_mbufs[index];
            // print_eth_hdr(priv_data, mbuf_curr, "RECV-RAW");

            // Filter out non NVIPC message
            if(nvipc_hdr_filter(priv_data, mbuf_curr) != 0)
            {
                rte_pktmbuf_free(mbuf_curr);
                continue;
            }

            set_mbuf_self(mbuf_curr);

            nvipc_hdr_t* hdr = get_nvipc_hdr(mbuf_curr);
            NVLOGV(TAG, "RECV: RAW: %d-%d mbuf_segs=%u mbuf_len=%u next=%p msg_id=0x%02X msg_len=%d data_len=%d", hdr->seg_num, hdr->seg_id, mbuf_curr->nb_segs, mbuf_curr->data_len, mbuf_curr->next, hdr->msg_id, hdr->msg_len, hdr->data_len);

            if(hdr->seg_num == 1)
            {
                // No DATA part message
                recv_efd_notify(priv_data, mbuf_curr);
            }
            else if(hdr->seg_num > priv_data->max_chain_length)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid seg_num %d", __func__, hdr->seg_num);
                rte_pktmbuf_free(mbuf_curr);
                continue;
            }
            else
            {
                if(hdr->seg_id == 0)
                {
                    // MSG and DATA parts exist
                    if(!list_is_empty(&list))
                    {
                        list_dump(&list);
                        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "RECV: List is not empty, packet lost! Receiving %d-%d msg_id=0x%02X msg_len=%d data_len=%d", hdr->seg_num, hdr->seg_id, hdr->msg_id, hdr->msg_len, hdr->data_len);
                        mbuf_release(priv_data, list.head);
                        list_init(&list);
                    }
                    list_append(&list, mbuf_curr);
                }
                else
                {
                    list_append(&list, mbuf_curr);
                    if(hdr->seg_num == hdr->seg_id + 1)
                    {
                        // The last mbuf segment received
#ifdef USE_SHM_MEMPOOL_RX
                        nvipc_hdr_t* head = get_nvipc_hdr(list.head);
                        if(head->msg_len < 0 || head->msg_len > priv_data->msg_payload_size ||
                                head->data_len < 0 || head->data_len > priv_data->data_payload_size)
                        {
                            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT,
                                    "%s: invalid RX lengths msg_len=%d/%d data_len=%d/%d",
                                    __func__, head->msg_len, priv_data->msg_payload_size,
                                    head->data_len, priv_data->data_payload_size);
                            mbuf_release(priv_data, list.head);
                            list_init(&list);
                            continue;
                        }

                        int data_mbuf_count = (head->data_len + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
                        int data_index = -1;
                        if(data_mbuf_count)
                        {
                            data_index = priv_data->mempool->alloc(priv_data->mempool);
                            if(data_index < 0)
                            {
                                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: alloc SHM data buf failed: data_index=%d", __func__, data_index);
                                mbuf_release(priv_data, list.head);
                                list_init(&list);
                                continue;
                            }
                            head->data_buf = priv_data->mempool->get_addr(priv_data->mempool, data_index);
                            if(head->data_buf == NULL)
                            {
                                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: get_addr SHM data buf failed: data_index=%d", __func__, data_index);
                                priv_data->mempool->free(priv_data->mempool, data_index);
                                mbuf_release(priv_data, list.head);
                                list_init(&list);
                                continue;
                            }
                        }

                        if(copy_mbufs_to_nvipc(priv_data, head) != 0)
                        {
                            if(data_index >= 0)
                            {
                                priv_data->mempool->free(priv_data->mempool, data_index);
                                head->data_buf = NULL;
                            }
                            mbuf_release(priv_data, list.head);
                            list_init(&list);
                            continue;
                        }
#endif
                        recv_efd_notify(priv_data, list.head);
                        list_init(&list);
                    }
                }
            }
        }
    } while(nb_rx > 0 || !list_is_empty(&list));
    return 0;
}

static void* nic_poll_thread(void* arg)
{
    priv_data_t* priv_data = (priv_data_t*)arg;

    NVLOGC(TAG, "%s: nvipc_nic_poll thread start ...", __func__);

    char thread_name[NVLOG_NAME_MAX_LEN + 16];
    snprintf(thread_name, NVLOG_NAME_MAX_LEN + 16, "nvipc_nic_poll");
    if(pthread_setname_np(pthread_self(), thread_name) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set thread name %s failed", __func__, thread_name);
    }

    if(nv_assign_thread_cpu_core(priv_data->lcore_id) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set cpu core failed: lcore_id=%u", __func__, priv_data->lcore_id);
    }

    if(RTE_LAUNCH_THREAD != 1)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if(rte_thread_register() != 0)
#pragma GCC diagnostic pop
        {
            NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_thread_register failed: %s", __func__, rte_strerror(rte_errno));
        }
    }

    dpdk_print_lcore("nvipc_nic_poll thread running");

    int tx_free = rte_mempool_avail_count(priv_data->txpool);
    int rx_free = rte_mempool_avail_count(priv_data->rxpool);
    NVLOGC(TAG, "%s: mbuf mempool free count: tx_free=%d rx_free=%d", __func__, tx_free, rx_free);

    priv_data->rx_pkts         = 0;
    priv_data->poll_counter    = 0;
    priv_data->tv_last.tv_sec  = 0;
    priv_data->tv_last.tv_usec = 0;

    while(1)
    {
        dpdk_nic_recv_poll(priv_data);
    }

    return NULL;
}

int nic_poll_lcore_func(void* arg)
{
    nic_poll_thread(arg);
    return 0;
}

static int dpdk_nic_recv(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);

    // rte_mempool_free_count();

    struct rte_mbuf* mbuf;
    if(priv_data->rx_ring->dequeue(priv_data->rx_ring, &mbuf) == 0)
    {
        // print_eth_hdr(priv_data, mbuf, "RECV");
        nvipc_hdr_t* hdr = get_nvipc_hdr(mbuf);
        check_mbuf_chain(priv_data, mbuf, msg, "RECV");

        msg->msg_id    = hdr->msg_id;
        msg->cell_id   = hdr->cell_id;
        msg->msg_len   = hdr->msg_len;
        msg->data_len  = hdr->data_len;
        msg->data_pool = hdr->data_pool;

        msg->msg_buf  = hdr->payload;
        msg->data_buf = hdr->data_buf;
        NVLOGI(TAG, "RECV: cell_id=%d msg_id=0x%02X msg_len=%u data_len=%u msg_buf=%p data_buf=%p", msg->cell_id, msg->msg_id, msg->msg_len, msg->data_len, msg->msg_buf, msg->data_buf);
        return 0;
    }
    else
    {
        return -1;
    }
}

static int get_port_by_pci_addr(const char* pci_addr, uint16_t* nic_port)
{
    struct rte_eth_dev_info dev_info;
#if (DOCA_GPU_DPDK_OLD == 1)
    struct rte_pci_device*  pci_dev;
#else
    struct rte_pci_addr     pci_dev_addr;
#endif
    struct rte_bus*         bus = NULL;

    int compare_offset;
    if(strstr(pci_addr, ":") == pci_addr + 4)
    {
        compare_offset = 0; // pci_addr contains domain
    }
    else
    {
        compare_offset = 5; // pci_addr doesn't contain domain
    }

    char                  tmp_buf[64];
    int                   ret = -1;
    uint16_t              port;
    struct rte_ether_addr mac;

    RTE_ETH_FOREACH_DEV(port)
    {
        rte_eth_dev_info_get(port, &dev_info);
        if(dev_info.device)
            bus = rte_bus_find_by_device(dev_info.device);
#if (DOCA_GPU_DPDK_OLD == 1)
        if(bus && !strcmp(bus->name, "pci"))
        {
            pci_dev    = RTE_DEV_TO_PCI(dev_info.device);
            int offset = snprintf(tmp_buf, 20, "%04x:%02x:%02x.%x", pci_dev->addr.domain, pci_dev->addr.bus, pci_dev->addr.devid, pci_dev->addr.function);
#else
        if(bus && !strcmp(rte_bus_name(bus), "pci"))
        {
            rte_pci_addr_parse(rte_dev_name(dev_info.device), &pci_dev_addr);
            int offset = snprintf(tmp_buf, 20, "%04x:%02x:%02x.%x", pci_dev_addr.domain, pci_dev_addr.bus, pci_dev_addr.devid, pci_dev_addr.function);
#endif

            if(rte_eth_macaddr_get(port, &mac) == 0)
            {
                offset += snprintf(tmp_buf + offset, 20, " %02x:%02x:%02x:%02x:%02x:%02x", mac.addr_bytes[0], mac.addr_bytes[1], mac.addr_bytes[2], mac.addr_bytes[3], mac.addr_bytes[4], mac.addr_bytes[5]);
            }

            if(strncasecmp(tmp_buf + compare_offset, pci_addr, strlen(pci_addr)) == 0)
            {
                offset += snprintf(tmp_buf + offset, 20, " - matched");
                *nic_port = port;
                ret       = 0;
            }
            NVLOGC(TAG, "Port %u: %s", port, tmp_buf);
        }
    }
    return ret;
}

static int dpdk_mbuf_allocate(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options)
{
    priv_data_t* priv_data = get_private_data(ipc);
    if(priv_data->txpool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: MSG memory pool is NULL", __func__);
        return -1;
    }

    struct rte_mbuf* msg_mbuf;

    TX_LOCK();
    if(rte_pktmbuf_alloc_bulk(priv_data->txpool, &msg_mbuf, 1) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_pktmbuf_alloc_bulk msg_buf failed", __func__);
        return -1;
    }
    TX_UNLOCK();

    set_mbuf_self(msg_mbuf);

    msg->msg_buf = get_nvipc_payload(msg_mbuf);

    if(msg->data_pool > 0 && msg->data_pool < NV_IPC_MEMPOOL_NUM)
    {
#ifdef USE_SHM_MEMPOOL_TX
        int data_index = priv_data->mempool->alloc(priv_data->mempool);
        if(data_index < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: alloc SHM data buf failed: data_index=%d", __func__, data_index);
            return -1;
        }
        msg->data_buf = priv_data->mempool->get_addr(priv_data->mempool, data_index);
        if(msg->data_buf == NULL)
        {
            // Should never run to here
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: get_addr SHM data buf failed: data_index=%d", __func__, data_index);
        }
#else
        if(msg->data_pool == NV_IPC_MEMPOOL_CPU_DATA && msg->data_len <= 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: please set data_len before call allocate", __func__);
            return -1;
        }
        // TODO
        // msg->data_buf = get_nvipc_payload(mbufs[1]);
#endif
    }
    else
    {
        msg->data_buf = NULL;
    }

    return 0;
}

static int dpdk_rx_alloc(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options)
{
    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: API not available", __func__);
    return -1;
}

static int dpdk_tx_release(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: API not available", __func__);
    return -1;
}

static int dpdk_mbuf_release(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(ipc == NULL || msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    NVLOGD(TAG, "FREE: msg_id=0x%02X msg_len=%d data_len=%d msg_buf=%p", msg->msg_id, msg->msg_len, msg->data_len, msg->msg_buf);

    priv_data_t* priv_data = get_private_data(ipc);
    if(msg->msg_buf != NULL)
    {
        struct rte_mbuf* mbuf = get_rte_mbuf(msg->msg_buf);
        check_mbuf_chain(priv_data, mbuf, msg, "FREE");

        // mbuf_release(priv_data, mbuf);
        if(priv_data->rx_free_ring->enqueue(priv_data->rx_free_ring, &mbuf) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: failed enqueue for free", __func__);
        }

#ifdef USE_SHM_MEMPOOL_RX
        if(msg->data_buf != NULL)
        {
            int data_index = priv_data->mempool->get_index(priv_data->mempool, msg->data_buf);
            if(priv_data->mempool->free(priv_data->mempool, data_index) < 0)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: failed to free SHM data_buf=%p: data_index=%d", __func__, msg->data_buf, data_index);
            }
        }
#endif
    }
    else
    {
        NVLOGV(TAG, "FREE double called: msg_id=0x%02X msg_len=%d data_len=%d msg_buf=%p", msg->msg_id, msg->msg_len, msg->data_len, msg->msg_buf);
    }

    // memset(msg, 0, sizeof(nv_ipc_msg_t));

    return 0;
}

static int dpdk_efd_get_fd(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    return priv_data->efd_rx;
}

static int dpdk_efd_notify(nv_ipc_t* ipc, int value)
{
    priv_data_t* priv_data = get_private_data(ipc);

    if(CONFIG_DPDK_DL_TTI_SYNC && priv_data->primary == 0)
    {
        return send_tti_sync_event(ipc, value);
    }
    else
    {
        return 0;
    }
}

static int dpdk_efd_get_value(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);

    uint64_t efd_value;
    ssize_t  size = read(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
    NVLOGV(TAG, "%s: size=%ld efd_value=%lu", __func__, size, efd_value);

    if(size < 0)
    {
        return -1;
    }
    else
    {
        return (int)efd_value;
    }
}

static int dpdk_tx_tti_sem_post(nv_ipc_t* ipc)
{
    return dpdk_efd_notify(ipc, 1);
}

static int dpdk_rx_tti_sem_wait(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    int          ret       = ipc_epoll_wait(priv_data->ipc_epoll);

    uint64_t efd_value;
    ssize_t  size = read(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
    NVLOGV(TAG, "%s: size=%ld efd_value=%lu", __func__, size, efd_value);

    return ret;
}

static int cpu_memcpy(nv_ipc_t* ipc, void* dst, const void* src, size_t size)
{
    return 0;
}

static int dpdk_ipc_close(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    int          ret       = 0;

    // Close epoll FD if exist
    if(priv_data->ipc_epoll != NULL)
    {
        if(ipc_epoll_destroy(priv_data->ipc_epoll) < 0)
        {
            ret = -1;
        }
    }

#ifdef NVIPC_CUDA_ENABLE
    if(priv_data->cuda_device_id >= 0)
    {
        if(nv_ipc_page_unlock(priv_data->mempool->get_addr(priv_data->mempool, 0)) < 0)
        {
            ret = -1;
        }
    }
#endif

    // Destroy the nv_ipc_t instance
    free(ipc);

    if(ret == 0)
    {
        NVLOGI(TAG, "%s: OK", __func__);
    }
    return ret;
}

static int64_t ts_ptp_diff = 0;

static int recv_ipc_connect(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);

    char mac_str[20];
    int  count = 0;
    ipc->rx_tti_sem_wait(ipc);
    while(ipc->rx_recv_msg(ipc, msg) >= 0)
    {
        if(priv_data->primary)
        {
            memcpy(&priv_data->dst, msg->msg_buf, sizeof(struct rte_ether_addr));
            eth_addr_to_mac_str(mac_str, &priv_data->dst);
            // NVLOGC(TAG, "connected from %s", mac_str);
        }
        eth_addr_to_mac_str(mac_str, &priv_data->dst);

        struct timespec* p_ts_send = (struct timespec* )((uint8_t*)msg->msg_buf + msg->msg_len - sizeof(struct timespec));
        struct timespec ts_recv;
        nvlog_gettime_rt(&ts_recv);
        ts_ptp_diff = nvlog_timespec_interval(p_ts_send, &ts_recv);

        if(msg->msg_id != MSG_ID_CONNECT)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Error connect msg_id=0x%02X from %s", msg->msg_id, mac_str);
        }
        ipc->rx_release(ipc, msg);
        count++;
    }
    if(count > 1)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Error connect count=%d from %s", count, mac_str);
        return -1;
    }
    else
    {
        return 0;
    }
}

static int send_ipc_connect(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    priv_data_t* priv_data = get_private_data(ipc);

    memset(msg, 0, sizeof(nv_ipc_msg_t));
    if(ipc->tx_allocate(ipc, msg, 0) < 0)
    {
        return -1;
    }

    msg->msg_id  = MSG_ID_CONNECT;
    msg->msg_len = sizeof(struct rte_ether_addr);
    memcpy(msg->msg_buf, &priv_data->src, sizeof(struct rte_ether_addr));

    struct timespec* ts = (struct timespec*)((uint8_t*)msg->msg_buf + msg->msg_len);
    nvlog_gettime_rt(ts);
    msg->msg_len += sizeof(struct timespec);

    if(ipc->tx_send_msg(ipc, msg) < 0)
    {
        return -1;
    }

    ipc->tx_tti_sem_post(ipc);

    return 0;
}

int ipc_connect(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);

    char mac_str[20];
    eth_addr_to_mac_str(mac_str, &priv_data->src);

    int          count = 0;
    nv_ipc_msg_t msg;
    if(priv_data->primary)
    {
        NVLOGC(TAG, "local_mac: %s wait for connect ...", mac_str);

        while(count < CONNECT_TEST_COUNT)
        {
            if(recv_ipc_connect(ipc, &msg) == 0)
            {
                send_ipc_connect(ipc, &msg);
            }

            count++;
        }

        eth_addr_to_mac_str(mac_str, &priv_data->dst);
        NVLOGC(TAG, "Connected from %s", mac_str);
    }
    else
    {
        char dst_str[20];
        eth_addr_to_mac_str(dst_str, &priv_data->dst);
        NVLOGC(TAG, "Try connect from %s to %s ...", mac_str, dst_str);

        int64_t         total = 0;
        struct timespec start, end;
        while(count < CONNECT_TEST_COUNT)
        {
            nvlog_gettime_rt(&start);
            send_ipc_connect(ipc, &msg);
            if(recv_ipc_connect(ipc, &msg) == 0)
            {
            }
            count++;
            nvlog_gettime_rt(&end);
            int64_t interval = nvlog_timespec_interval(&start, &end);
            total += interval;
            NVLOGI(TAG, "Connection delay test: interval=%ld ptp_diff=%ld", interval, ts_ptp_diff);
        }

        NVLOGC(TAG, "Connected to %s. avarage_delay=%ld", dst_str, total / CONNECT_TEST_COUNT);
    }
    return 0;
}

int ipc_test(nv_ipc_t* ipc)
{
    priv_data_t *priv_data = get_private_data(ipc);

    int ret = 0;
    int total = 1000000;


    int bulk = 32;
    // struct rte_mbuf **send_mbufs = malloc(sizeof(struct rte_mbuf*) * count);

    int index = 0;
    if (priv_data->primary == 0) {
        NVLOGC(TAG, "%s: SEND total=%d start ...", __func__, total);
        struct rte_mbuf* send_mbufs[BURST_SIZE];
        while (total > index) {
            int count = total - index;
            count = count > bulk ? bulk : count;
            if (rte_pktmbuf_alloc_bulk(priv_data->txpool, send_mbufs, count) != 0) {
                NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: SEND rte_pktmbuf_alloc_bulk failed: count=%d index=%d", __func__,
                        count, index);
                ret = -1;
            }
            for (int i = 0; i < count; i++)
            {
                int* payload = get_nvipc_payload(send_mbufs[i]);
                *payload = index;
                index++;
            }
            uint16_t nb_tx = rte_eth_tx_burst(priv_data->nic_port, 0, send_mbufs, count);
            if (nb_tx != count) {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: SEND failed nb_tx=%u index=%d", __func__, nb_tx, index);
                ret = -1;
            }
            for (int i = 0; i < count; i++)
            {
                rte_pktmbuf_free(send_mbufs[i]);
            }
            NVLOGI(TAG, "%s: SEND index=%d", __func__, index);
        }
        NVLOGC(TAG, "%s: SEND total=%d finished", __func__, total);
    } else {
        NVLOGC(TAG, "%s: RECV total=%d start ...", __func__, total);
        struct rte_mbuf *recv_mbufs[BURST_SIZE];
        while (total > index) {
            uint16_t nb_rx;
            while ((nb_rx = rte_eth_rx_burst(priv_data->nic_port, 0, recv_mbufs, BURST_SIZE)) == 0){
                NVLOGD(TAG, "%s: RECV poll ... nb_rx=0 index=%d", __func__, index);
            }

            if (nb_rx > BURST_SIZE) {
                NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: RECV received count nb_rx=%u > BURST_SIZE=%u index=%d", __func__, nb_rx,
                        BURST_SIZE, index);
                nb_rx = BURST_SIZE;
                ret = -1;
            }
            for (int i = 0; i < nb_rx; i++) {
                int *payload = get_nvipc_payload(recv_mbufs[i]);
                if (*payload != index) {
                    NVLOGW(TAG, "%s: RECV unexpect packet: nb_rx=%u index: expected=%d payload=%d", __func__, nb_rx, index,
                            *payload);
                }
                index++;
                rte_pktmbuf_free(recv_mbufs[i]);
            }
            NVLOGI(TAG, "%s: RECV index=%d nb_rx=%u", __func__, index, nb_rx);
        }
        NVLOGC(TAG, "%s: RECV total=%d finished", __func__, total);
    }
    return ret;
}

static int add_hotplug_device(priv_data_t* priv_data, const nv_ipc_config_dpdk_t* cfg)
{
    int                ret = 0;
    NVLOGC(TAG, "%s: rte_eal_hotplug_add nic %s", __func__, cfg->local_nic_pci);
    if(rte_eal_hotplug_add("pci", cfg->local_nic_pci, "tx_pp=500,txq_inline_max=0") == 0)
    {
        NVLOGC(TAG, "%s: rte_eal_hotplug_add pci %s OK", __func__, cfg->local_nic_pci);
    }
    else if(rte_eal_hotplug_add("auxiliary", cfg->local_nic_pci, "tx_pp=500,txq_inline_max=0") == 0)
    {
        NVLOGC(TAG, "%s: rte_eal_hotplug_add auxiliary %s OK", __func__, cfg->local_nic_pci);
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_eal_hotplug_add %s failed", __func__, cfg->local_nic_pci);
        return -1;
    }

    list_all_nic_ports();

    if((ret = rte_eth_dev_get_port_by_name(cfg->local_nic_pci, &priv_data->nic_port)) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_eth_dev_get_port_by_name(%s) failed: Error: %d - %s", __func__, cfg->local_nic_pci, ret, strerror(ret));
        return -1;
    }

    NVLOGC(TAG, "%s: rte_eth_dev_get_port_by_name(%s): port=%u", __func__, cfg->local_nic_pci, priv_data->nic_port);
    return 0;
}

static int dpdk_ipc_open(nv_ipc_t* ipc, const nv_ipc_config_dpdk_t* cfg)
{
    int ret = 0;

    priv_data_t* priv_data    = get_private_data(ipc);
    priv_data->primary        = cfg->primary;
    priv_data->lcore_id       = cfg->lcore_id;
    priv_data->cuda_device_id = cfg->cuda_device_id;
    priv_data->nic_mtu        = cfg->nic_mtu;

    NVLOGC(TAG, "%s: primary=%d lcore_id=%d cuda_device_id=%d", __func__, priv_data->primary, priv_data->lcore_id, priv_data->cuda_device_id);

    if(pthread_mutex_init(&priv_data->tx_lock, NULL) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: create tx mutex failed", __func__);
        return -1;
    }
    if(pthread_mutex_init(&priv_data->rx_lock, NULL) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: create rx mutex failed", __func__);
        return -1;
    }

    // Create efd_rx locally and get efd_tx from remote party
    int flag = EFD_SEMAPHORE;
    if((priv_data->efd_rx = eventfd(0, flag)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: create efd_rx failed", __func__);
        return -1;
    }

    // Create epoll wrapper for converting to blocking-wait API interface
    if((priv_data->ipc_epoll = ipc_epoll_create(CONFIG_EPOLL_EVENT_MAX, priv_data->efd_rx)) == NULL)
    {
        return -1;
    }

    if(cfg->need_eal_init == 1)
    {
        nv_ipc_dpdk_init("nvipc_dpdk", cfg);
    }

    list_all_nic_ports();
    if(add_hotplug_device(priv_data, cfg) != 0)
    {
        return -1;
    }

    if(rte_eth_macaddr_get(priv_data->nic_port, &priv_data->src) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: failed to get MAC for port %u", __func__, priv_data->nic_port);
        return -1;
    }

    if(mac_str_to_eth_addr(cfg->peer_nic_mac, &priv_data->dst) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid peer MAC configuration: %s", __func__, cfg->peer_nic_mac);
        return -1;
    }

    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);

    int msg_buf_size = cfg->mempool_size[NV_IPC_MEMPOOL_CPU_MSG].buf_size;
    int msg_pool_len = cfg->mempool_size[NV_IPC_MEMPOOL_CPU_MSG].pool_len;

    int data_buf_size = cfg->mempool_size[NV_IPC_MEMPOOL_CPU_DATA].buf_size;
    int data_pool_len = cfg->mempool_size[NV_IPC_MEMPOOL_CPU_DATA].pool_len;

    if((priv_data->mempool = nv_ipc_mempool_open(1, "nvipc_data", data_buf_size, data_pool_len, -1)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create SHM mempool nvipc_data failed", __func__);
        return -1;
    }

#ifdef NVIPC_CUDA_ENABLE
    if(priv_data->cuda_device_id >= 0)
    {
        if(nv_ipc_page_lock(priv_data->mempool->get_addr(priv_data->mempool, 0), (size_t)data_buf_size * data_pool_len) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_CUDA_API_EVENT, "%s: failed to set SHM mempool to host pinned memory", __func__);
            return -1;
        }
    }
#endif

    // int data_room_size = msg_buf_size + RTE_PKTMBUF_HEADROOM;
    int data_room_size = 8192;
    NVLOGC(TAG, "%s: create dpdk mempools: data_room_size=%d pool_len=%d msg_buf_size=%d data_buf_size=%d", __func__, data_room_size, msg_pool_len, msg_buf_size, data_buf_size);
    if((priv_data->txpool = rte_pktmbuf_pool_create("nvipc_tx", msg_pool_len, MBUF_CACHE_SIZE, 0, data_room_size, rte_socket_id())) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: create TX mempool failed", __func__);
        return -1;
    }

    if((priv_data->rxpool = rte_pktmbuf_pool_create("nvipc_rx", msg_pool_len * RX_BUF_X, MBUF_CACHE_SIZE, 0, data_room_size, rte_socket_id())) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: create RX mempool pool failed", __func__);
        return -1;
    }

    struct rte_mbuf* mbuf_test;
    if(rte_pktmbuf_alloc_bulk(priv_data->txpool, &mbuf_test, 1) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: rte_pktmbuf_alloc_bulk failed", __func__);
        return -1;
    }

    int tx_free = rte_mempool_avail_count(priv_data->txpool);
    int rx_free = rte_mempool_avail_count(priv_data->rxpool);

    priv_data->msg_payload_size  = msg_buf_size;
    priv_data->data_payload_size = data_buf_size;
    priv_data->mbuf_payload_size = priv_data->nic_mtu - 18 - sizeof(nvipc_hdr_t);
    priv_data->max_chain_length  = (msg_buf_size + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
    priv_data->max_chain_length += (data_buf_size + priv_data->mbuf_payload_size - 1) / priv_data->mbuf_payload_size;
    priv_data->max_rx_pkt_len = priv_data->nic_mtu - 18; // (RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN)

    NVLOGC(TAG, "%s: mbuf pools: mtu=%u buf_len=%u mbuf_payload_size=%d data_payload_size=%d max_chain_len=%d tx_free=%d rx_free=%d", __func__, priv_data->nic_mtu, mbuf_test->buf_len, priv_data->mbuf_payload_size, priv_data->data_payload_size, priv_data->max_chain_length, tx_free, rx_free);

    rte_pktmbuf_free(mbuf_test);

    // Create TX cache queue
    if((priv_data->tx_ring = nv_ipc_ring_open(1, "nvipc_tx_ring", msg_pool_len, sizeof(nv_ipc_msg_t))) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed to create nvipc_tx_ring", __func__);
        return -1;
    }

    // Create RX cache queue
    if((priv_data->rx_ring = nv_ipc_ring_open(1, "nvipc_rx_ring", msg_pool_len, sizeof(struct rte_mbuf*))) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed to create nvipc_rx_ring", __func__);
        return -1;
    }

    if((priv_data->rx_free_ring = nv_ipc_ring_open(1, "mbuf_free_ring", msg_pool_len, sizeof(struct rte_mbuf*))) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed to create mbuf_free_ring", __func__);
        return -1;
    }

    // Init NIC port
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if(port_init(ipc) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: Failed to init port %u, nb_ports=%u", __func__, priv_data->nic_port, nb_ports);
        return -1;
    }

    if(rte_eth_dev_set_mtu(priv_data->nic_port, priv_data->nic_mtu) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_DPDK_API_EVENT, "%s: failed to set MTU %u for port %u", __func__, priv_data->nic_mtu, priv_data->nic_port);
        return -1;
    }

    if(RTE_LAUNCH_THREAD == 1)
    {
        create_dpdk_task(nic_poll_lcore_func, priv_data, priv_data->lcore_id);
    }
    else
    {
        // Create a background thread to write SHM cache to file stream
        if(pthread_create(&priv_data->thread_id, NULL, nic_poll_thread, priv_data) != 0)
        {
            NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: thread create failed", __func__);
            return -1;
        }
    }

    if(ipc_connect(ipc) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: fail to connect with peer", __func__);
        return -1;
    }

    if(IPC_TEST_EN && ipc_test(ipc) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: fail to test dpdk ipc", __func__);
        return -1;
    }

    char log[96];
    int  offset = snprintf(log, sizeof(log), "MAC: local=");
    offset += eth_addr_to_mac_str(log + offset, &priv_data->src);
    offset += snprintf(log + offset, sizeof(log) - offset, " peer=");
    offset += eth_addr_to_mac_str(log + offset, &priv_data->dst);
    NVLOGC(TAG, "%s: port %u-%u %s", __func__, nb_ports, priv_data->nic_port, log);
    return 0;
}

nv_ipc_t* create_dpdk_nv_ipc_interface(const nv_ipc_config_t* cfg)
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
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }

    memset(ipc, 0, size);

    priv_data_t* priv_data = get_private_data(ipc);

    ipc->ipc_destroy = dpdk_ipc_close;

    ipc->tx_allocate = dpdk_mbuf_allocate;
    ipc->rx_release  = dpdk_mbuf_release;

    ipc->tx_release  = dpdk_tx_release;
    ipc->rx_allocate = dpdk_rx_alloc;

    ipc->tx_send_msg = dpdk_nic_send;
    ipc->rx_recv_msg = dpdk_nic_recv;

    // Semaphore synchronization
    ipc->tx_tti_sem_post = dpdk_tx_tti_sem_post;
    ipc->rx_tti_sem_wait = dpdk_rx_tti_sem_wait;

    // Event FD synchronization
    ipc->get_fd    = dpdk_efd_get_fd;
    ipc->get_value = dpdk_efd_get_value;
    ipc->notify    = dpdk_efd_notify;

    ipc->cuda_memcpy_to_host   = cpu_memcpy;
    ipc->cuda_memcpy_to_device = cpu_memcpy;

    if((priv_data->ipc_debug = nv_ipc_debug_open(ipc, cfg)) == NULL)
    {
        free(ipc);
        return NULL;
    }

    if(dpdk_ipc_open(ipc, &cfg->transport_config.dpdk) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed", __func__);
        dpdk_ipc_close(ipc);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: OK", __func__);
        return ipc;
    }
}
