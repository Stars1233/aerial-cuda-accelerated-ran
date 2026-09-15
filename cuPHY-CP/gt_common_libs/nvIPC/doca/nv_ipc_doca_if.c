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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef NVIPC_DOCA_GPUNETIO
#include <rte_eal.h>
#include <doca_gpunetio.h>
#endif

#include "nv_ipc.h"
#include "nv_ipc_debug.h"
#include "nv_ipc_efd.h"
#include "nv_ipc_sem.h"
#include "nv_ipc_epoll.h"
#include "nv_ipc_shm.h"
#include "nv_ipc_mempool.h"
#include "nv_ipc_utils.h"
#include "array_queue.h"
#include "nv_ipc_cuda_utils.h"
#include "nv_ipc_config.h"
#include "nv_ipc_forward.h"
#include "nv_ipc_cuda_utils.h"

#include "nv_ipc_ring.h"
#include "nv_ipc_doca.h"
#include "nv_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 18) //"NVIPC.DOCA"

#define DEBUG_DOCA_IPC 0

#define USE_SHM_MEMPOOL_RX
#define USE_SHM_MEMPOOL_TX
// #define MULTI_SEGS_SEND
// #define MBUF_CHAIN_LINK

#define CONFIG_DL_EXPLICIT_PACKET_SYNC 0
#define CONFIG_UL_EXPLICIT_PACKET_SYNC 0

#define MSG_ID_DL_TTI_SYNC (0x70)
#define MSG_ID_CONNECT (-3)

#define CONNECT_TEST_COUNT 10

#define CUST_ETHER_TYPE_NVIPC 0xAE55

#define NV_NAME_SUFFIX_MAX_LEN 16

#define BURST_SIZE 512
#define BURST_QUEUE_SIZE (BURST_SIZE * 16)
#define MBUF_CACHE_SIZE 250

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

#define CONFIG_DL_TTI_SYNC_EVENT 1

#define CONFIG_EPOLL_EVENT_MAX 1

#define IPC_TEST_EN 0

#define CONFIG_ENABLE_HOST_PAGE_LOCK 1

#define container_of(ptr, type, member) ({                      \
                const typeof( ((type *)0)->member ) *__mptr = (ptr);    \
                (type *)( (char *)__mptr - offsetof(type,member) );})

static const char shm_suffix[]                                               = "_shm";
static const char forward_suffix[]                                           = "_fw";
static const char ring_suffix[4][NV_NAME_SUFFIX_MAX_LEN]                     = {"_tx_msg", "_rx_msg", "_tx_dma", "_rx_dma"};
static const char tx_pool_suffix[NV_IPC_MEMPOOL_NUM][NV_NAME_SUFFIX_MAX_LEN] = {"_tx_cpu_msg", "_tx_cpu_data", "_tx_cpu_large", "_tx_cuda_data", "_tx_gpu_data"};
static const char rx_pool_suffix[NV_IPC_MEMPOOL_NUM][NV_NAME_SUFFIX_MAX_LEN] = {"_rx_cpu_msg", "_rx_cpu_data", "_rx_cpu_large", "_rx_cuda_data", "_rx_gpu_data"};

static const char comm_suffix[] = "_comm";

static inline uint64_t nvipc_rdtsc() {
#ifdef _MSC_VER
    return __rdtsc();
#elif defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
    return __builtin_ia32_rdtsc();
#elif defined(__aarch64__)
    uint64_t ret;  // unsigned 64-bit value
    asm volatile("mrs %0, cntvct_el0" : "=r" (ret));
    return ret;
#else
    return rdsysns();
#endif
}

#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
#define TSC_200_US (500L * 1000 * 1000)
#elif defined(__aarch64__)
#define TSC_200_US (60L * 1000 * 1000)
#endif

typedef struct
{
    atomic_uint forward_started; // Flag indicates forwarding status
    atomic_uint msg_buf_count;   // MSG buffer count in fw_ring
    atomic_uint data_buf_count;  // DATA buffer count in fw_ring
    atomic_uint ipc_total;       // Total IPC message count to forward
    atomic_uint ipc_forwarded;   // Forwarded IPC message count since forwarding start
    atomic_uint ipc_lost;        // Lost IPC message count since forwarding start
    int32_t     queue_header[];
} forwarder_data_t;

typedef struct {
    mmap_info_t dl_descs[NV_IPC_MEMPOOL_NUM];
    mmap_info_t ul_descs[NV_IPC_MEMPOOL_NUM];
} mmap_descs_t;

typedef struct {
    dma_info_t tx_pools[NV_IPC_MEMPOOL_NUM];
    dma_info_t rx_pools[NV_IPC_MEMPOOL_NUM];
} dma_pools_t;

typedef struct {
    int32_t type;
    int32_t pkt_id;
} comm_msg_t;

typedef struct
{
    int primary;
    char prefix[NV_NAME_MAX_LEN];

    // SHM info

    // CUDA device ID for CUDA memory pool
    int cuda_device_id;

    int32_t ring_len;

    // The TX and RX rings
    nv_ipc_shm_t* shmpool;

    array_queue_t* tx_msg;
    array_queue_t* rx_msg;

    array_queue_t* rx_dma;
    array_queue_t* tx_dma;

    nv_ipc_ring_t* cmsg_async;

    /* forward_enable: configured from yaml file
     * 0: disabled;
     * 1: enabled but doesn't start forwarding at initial;
     * -1: start forwarding at initial with count = 0;
     * Other positive number: start forwarding at initial with count = forward_enable.
     */
    int32_t forward_enable;

    int32_t           fw_max_msg_buf_count;
    int32_t           fw_max_data_buf_count;
    nv_ipc_shm_t*     fw_shmpool;
    forwarder_data_t* fw_data;
    array_queue_t*    fw_ring;
    sem_t*            fw_sem;

    packet_info_t* tx_pkt_infos;
    packet_info_t* rx_pkt_infos;

    // Lock-less memory pool for MSG, CPU DATA and CUDA DATA
    nv_ipc_mempool_t* txpools[NV_IPC_MEMPOOL_NUM];

    nv_ipc_mempool_t* rxpools[NV_IPC_MEMPOOL_NUM];

    mmap_descs_t mmap_descs;
    dma_pools_t dma_pools;

    dma_job_t* tx_job_base;
    dma_job_t* rx_job_base;

    // For synchronization between the two processes
    nv_ipc_efd_t*   ipc_efd;
    nv_ipc_sem_t*   ipc_sem;
    nv_ipc_epoll_t* ipc_epoll;

    // DOCA
    doca_info_t doca_info;

    // Large buffer memory pool for DATA part
    // nv_ipc_mempool_t* mempool;

    int32_t msg_payload_size;
    int32_t data_payload_size;
    int32_t max_chain_length;
    int32_t mbuf_payload_size;
    int32_t max_rx_pkt_len;

    uint16_t nic_port;
    uint16_t nic_mtu;

    int efd_rx;

    uint16_t  cpu_core;
    pthread_t thread_id;

    pthread_mutex_t tx_lock;
    pthread_mutex_t rx_lock;

    uint64_t poll_counter;
    uint64_t rx_pkts;

    // For debug
    struct timeval tv_last;

    nv_ipc_debug_t* ipc_debug;

    uint8_t cmsg_buf[NVIPC_DOCA_CC_MAX_MSG_SIZE];

    dma_job_t* read_jobs;
    dma_job_t* write_jobs;

    int32_t msg_buf_size;
    int32_t msg_pool_len;

    uint8_t* tx_msg_pool_addr;
    uint8_t* rx_msg_pool_addr;

} priv_data_t;

#define IPC_DUMPING_CHECK(priv_data)                                      \
    if (atomic_load(&priv_data->ipc_debug->shm_data->ipc_dumping))        \
    {                                                                     \
        NVLOGI(TAG, "%s line %d: ipc dumping, skip", __func__, __LINE__); \
        return -1;                                                        \
    }

#define IPC_DUMPING_CHECK_BLOCKING(priv_data)                             \
    if (atomic_load(&priv_data->ipc_debug->shm_data->ipc_dumping))        \
    {                                                                     \
        NVLOGI(TAG, "%s line %d: ipc dumping, wait", __func__, __LINE__); \
        sleep(100000);                                                    \
        return 0;                                                        \
    }

static int tx_buf_free(priv_data_t* priv_data, nv_ipc_msg_t* msg);

static inline priv_data_t* get_private_data(nv_ipc_t* ipc)
{
    return (priv_data_t*)((int8_t*)ipc + sizeof(nv_ipc_t));
}

static inline int doca_cc_send_ipc_pkt_async(priv_data_t* priv_data, int type, packet_info_t* pkt) {
    pkt->cc_msg.type = type;
    pkt->cc_msg.len = sizeof(packet_info_t);
    return priv_data->cmsg_async->enqueue(priv_data->cmsg_async, pkt);
}

static inline int doca_cc_send_ipc_pkt(priv_data_t* priv_data, int type, packet_info_t* pkt) {
//    pkt->cc_msg.type = type;
//    pkt->cc_msg.len = sizeof(packet_info_t);
//    return doca_cc_send(&priv_data->doca_info, &pkt->cc_msg);
    return doca_cc_send_ipc_pkt_async(priv_data, type, pkt);
}

static inline void* get_ipc_buf_addr(nv_ipc_mempool_t* msgpool, int msg_index) {
    return msgpool->get_addr(msgpool, msg_index);
}

static inline int get_ipc_buf_index(nv_ipc_mempool_t* msgpool, void* _payload) {
    return msgpool->get_index(msgpool, _payload);
}

static inline packet_info_t* get_tx_packet_info(priv_data_t* priv_data, int pkt_id) {
    return priv_data->tx_pkt_infos + pkt_id;
}

static inline packet_info_t* get_rx_packet_info(priv_data_t* priv_data, int pkt_id) {
    return priv_data->rx_pkt_infos + pkt_id;
}

static inline int nvipc_pkt_has_data_payload(const packet_info_t* pkt)
{
    return pkt != NULL
        && pkt->data_pool > NV_IPC_MEMPOOL_CPU_MSG
        && pkt->data_pool < NV_IPC_MEMPOOL_NUM
        && pkt->data_len > 0;
}

/**
 * Validate peer-supplied packet_info_t indices and payload sizes against local pools.
 *
 * @param[in] priv_data Transport private data (ring length, buffer sizes).
 * @param[in] pkt       Peer packet metadata to validate.
 * @param[in] pools     Mempool table indexed by nv_ipc_mempool_id_t.
 * @param[in] context   Caller tag for error logs.
 * @return 0 on success, -1 on validation failure.
 */
static int validate_packet_info_indices(priv_data_t* priv_data, const packet_info_t* pkt,
        nv_ipc_mempool_t** pools, const char* context)
{
    if(pkt == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL packet", context);
        return -1;
    }
    if(pkt->msg_index < 0 || pkt->msg_index >= priv_data->ring_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_index=%d ring_len=%d",
                context, pkt->msg_index, priv_data->ring_len);
        return -1;
    }
    if(pkt->msg_len < 0 || pkt->msg_len > priv_data->msg_buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_len=%d msg_buf_size=%d",
                context, pkt->msg_len, priv_data->msg_buf_size);
        return -1;
    }

    if(pkt->data_len < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_len=%d", context, pkt->data_len);
        return -1;
    }

    if(!nvipc_pkt_has_data_payload(pkt))
    {
        return 0;
    }

    if(pools == NULL || pools[pkt->data_pool] == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: DATA pool %d is not configured", context,
                pkt->data_pool);
        return -1;
    }

    nv_ipc_mempool_t* datapool = pools[pkt->data_pool];
    int32_t           pool_len = datapool->get_pool_len(datapool);
    int32_t           buf_size = datapool->get_buf_size(datapool);

    if(pkt->data_index < 0 || pkt->data_index >= pool_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_index=%d pool_len=%d", context,
                pkt->data_index, pool_len);
        return -1;
    }

    if(pkt->data_len > buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_len=%d buf_size=%d", context,
                pkt->data_len, buf_size);
        return -1;
    }

    return 0;
}

// 1 - started; 0 - stopped
static int get_forward_started(priv_data_t* priv_data)
{
    if(priv_data->forward_enable)
    {
        return atomic_load(&priv_data->fw_data->forward_started);
    }
    else
    {
        return 0;
    }
}

static int send_ipc_event(nv_ipc_t* ipc, int value)
{
    priv_data_t* priv_data = get_private_data(ipc);

    packet_info_t pkt;
    pkt.msg_id = value;

    return doca_cc_send_ipc_pkt_async(priv_data, CC_MSG_IPC_PACKET_EVENT, &pkt);
}

static int write_efd_value(priv_data_t* priv_data, uint64_t efd_value)
{
    ssize_t size = write(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
    if (size < 0) {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: efd write failed: size=%lu", __func__, size);
        return -1;
    }
    return 0;
}

static int recv_ipc_event(priv_data_t* priv_data, int value) {
    NVLOGD(TAG, "RECV: TTI SYNC notification: value=%d", value);

    uint64_t efd_value = value;
    int ret = 0;
    if (priv_data->primary != 0 && CONFIG_DL_EXPLICIT_PACKET_SYNC) {
        ret = write_efd_value(priv_data, efd_value);
    } else if (priv_data->primary == 0 && CONFIG_UL_EXPLICIT_PACKET_SYNC) {
        ret = write_efd_value(priv_data, efd_value);
    }
    return ret;
}

static int enqueue_incoming_packet(priv_data_t* priv_data, packet_info_t *pkt)
{
    int ret = 0;

    NVLOGD(TAG, "RECV: enqueue: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%u", pkt->cell_id,
            pkt->msg_id, pkt->msg_len, pkt->data_len);
    if ((ret = priv_data->rx_msg->enqueue(priv_data->rx_msg, pkt->msg_index)) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: rx_msg queue is full", __func__);
        return -1;
    }

    if (priv_data->primary != 0 && CONFIG_DL_EXPLICIT_PACKET_SYNC == 0) {
        ret = write_efd_value(priv_data, 1);
    } else if (priv_data->primary == 0 && CONFIG_UL_EXPLICIT_PACKET_SYNC == 0) {
        ret = write_efd_value(priv_data, 1);
    }
    return ret;
}

// /** Available jobs for DMA. */
// enum doca_dma_job_types {
// 	DOCA_DMA_JOB_MEMCPY = DOCA_ACTION_DMA_FIRST + 1,
// };

// struct doca_dma_job_memcpy {
// 	struct doca_job base;		 /**< Common job data */
// 	struct doca_buf *dst_buff;	 /**< Destination data buffer */
// 	struct doca_buf const *src_buff; /**< Source data buffer */
// };

static inline int submit_dma_job(struct doca_ctx *ctx, struct doca_workq *workq,
        struct doca_buf *src_buf, struct doca_buf *dst_buf, size_t len, buf_info_t ipc_buf) {
    doca_error_t result;
#if 0
    // Construct DMA job
    struct doca_dma_job_memcpy dma_job = { 0 };
    dma_job.base.type = DOCA_DMA_JOB_MEMCPY;
    dma_job.base.flags = DOCA_JOB_FLAGS_NONE;
    dma_job.base.ctx = ctx;

    dma_job.base.user_data.u64 = ipc_buf.u64;
    dma_job.src_buff = src_buf;
    dma_job.dst_buff = dst_buf;
    doca_buf_reset_data_len(dst_buf);

    void *data;
    if ((doca_buf_get_data(src_buf, &data)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to get data address from DOCA buffer: %s",
                doca_error_get_descr(result));
        return -1;
    }
    result = doca_buf_set_data(src_buf, data, len);
    if (result != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to set data for DOCA buffer: %s",
                doca_error_get_descr(result));
        return -1;
    }

    // Enqueue DMA job
    result = doca_workq_submit(workq, &dma_job.base);
    if (result != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to submit job: pkt_id=%d pool_id=%d buf_id=%d %s",
                ipc_buf.i32.pkt_id, ipc_buf.i32.pool_id, ipc_buf.i32.buf_id, doca_error_get_descr(result));
        return -1;
    }
#endif
    return 0;
}

static int dma_read_start(priv_data_t *priv_data, packet_info_t *pkt) {
    dma_info_t *pool1 = &priv_data->dma_pools.rx_pools[NV_IPC_MEMPOOL_CPU_MSG];
    dma_job_t *job1 = pool1->jobs + pkt->msg_index;
    atomic_store(&job1->status_mask, 0x1);

    buf_info_t buf1 = { .i32 = { .pkt_id = pkt->msg_index, .pool_id = NV_IPC_MEMPOOL_CPU_MSG,
            .buf_id = pkt->msg_index } };
    if (submit_dma_job(priv_data->doca_info.ctx, pool1->workq, job1->remote_buf, job1->local_buf,
            pkt->msg_len, buf1) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT,
                "%s: submit RX MSG DMA job failed: pkt_id=%d msg_len=%d", __func__, pkt->msg_index,
                pkt->msg_len);
        return -1;
    } else {
        pool1->inflight ++;
    }

    if (nvipc_pkt_has_data_payload(pkt)) {
        atomic_fetch_or(&job1->status_mask, 0x2);
        dma_info_t *pool2 = &priv_data->dma_pools.rx_pools[pkt->data_pool];
        dma_job_t *job2 = pool2->jobs + pkt->data_index;
        buf_info_t buf2 = { .i32 = { .pkt_id = pkt->msg_index, .pool_id = pkt->data_pool,
                .buf_id = pkt->data_index } };
        if (submit_dma_job(priv_data->doca_info.ctx, pool2->workq, job2->remote_buf, job2->local_buf,
                pkt->data_len, buf2) != 0) {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT,
                    "%s: submit RX DATA DMA job failed: pkt_id=%d data_pool=%d data_index=%d data_len=%d inflight=%d",
                    __func__, pkt->msg_index, pkt->data_pool, pkt->data_index, pkt->data_len, pool2->inflight);
            return -1;
        } else {
            pool2->inflight ++;
        }
    }

    // Enqueue job for status poll
    if (priv_data->rx_dma->enqueue(priv_data->rx_dma, pkt->msg_index) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: rx_dma queue is full", __func__);
        return -1;
    }
    return 0;
}

static int dma_write_start(priv_data_t *priv_data, packet_info_t *pkt) {
    dma_info_t *pool1 = &priv_data->dma_pools.tx_pools[NV_IPC_MEMPOOL_CPU_MSG];
    dma_job_t *job1 = pool1->jobs + pkt->msg_index;
    atomic_store(&job1->status_mask, 0x1);

    buf_info_t buf1 = { .i32 = { .pkt_id = pkt->msg_index, .pool_id = NV_IPC_MEMPOOL_CPU_MSG,
            .buf_id = pkt->msg_index } };
    if (submit_dma_job(priv_data->doca_info.ctx, pool1->workq, job1->local_buf, job1->remote_buf,
            pkt->msg_len, buf1) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT,
                "%s: submit TX MSG DMA job failed: pkt_id=%d msg_len=%d", __func__, pkt->msg_index,
                pkt->msg_len);
        return -1;
    } else {
        pool1->inflight ++;
    }

    if (nvipc_pkt_has_data_payload(pkt)) {
        atomic_fetch_or(&job1->status_mask, 0x2);
        dma_info_t *pool2 = &priv_data->dma_pools.tx_pools[pkt->data_pool];
        dma_job_t *job2 = pool2->jobs + pkt->data_index;
        buf_info_t buf2 = { .i32 = { .pkt_id = pkt->msg_index, .pool_id = pkt->data_pool, .buf_id =
                pkt->data_index } };
        if (submit_dma_job(priv_data->doca_info.ctx, pool2->workq, job2->local_buf,
                job2->remote_buf, pkt->data_len, buf2) != 0) {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT,
                    "%s: submit TX DATA DMA job failed: pkt_id=%d data_pool=%d data_index=%d data_len=%d inflight=%d",
                    __func__, pkt->msg_index, pkt->data_pool, pkt->data_index, pkt->data_len,
                    pool2->inflight);
            return -1;
        } else {
            pool2->inflight++;
        }
    }

    // Enqueue job for status poll
    if (priv_data->tx_dma->enqueue(priv_data->tx_dma, pkt->msg_index) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: tx_dma queue is full", __func__);
        return -1;
    }
    return 0;
}

static int poll_dma_status(dma_info_t* pool, dma_job_t* job_base) {
#if 0
    struct doca_workq *workq = pool->workq;
    if (workq == NULL) {
        // Skip non-exist pools
        return 0;
    }

    struct doca_event event;
    doca_error_t result;
    if ((result = doca_workq_progress_retrieve(workq, &event, DOCA_WORKQ_RETRIEVE_FLAGS_NONE))
            == DOCA_SUCCESS) {

        buf_info_t ipc_buf;
        ipc_buf.u64 = event.user_data.u64;

        if (event.result.u64 != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: event.result fail: pool_id=%d ipc_buf=%d %s",
                    __func__, ipc_buf.i32.pool_id, ipc_buf.i32.pkt_id, doca_error_get_descr(result));
            return -1;
        }

        NVLOGD(TAG, "%s: DMA DONE: pkt_id=%d pool_id=%d buf_id=%d inflight=%d", __func__,
                ipc_buf.i32.pkt_id, ipc_buf.i32.pool_id, ipc_buf.i32.buf_id, pool->inflight);

        if (ipc_buf.i32.pool_id < 0 || ipc_buf.i32.pool_id >= NV_IPC_MEMPOOL_NUM) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: invalid pool_id=%d buf_id=%d", __func__,
                    ipc_buf.i32.pool_id, ipc_buf.i32.buf_id);
            return -1;
        }

        dma_job_t* job = job_base + ipc_buf.i32.pkt_id;
        if (ipc_buf.i32.pool_id == NV_IPC_MEMPOOL_CPU_MSG) {
            pool->inflight --;
            atomic_fetch_and(&job->status_mask, 0x2); // Set bit_0 to 0
        } else if (ipc_buf.i32.pool_id > NV_IPC_MEMPOOL_CPU_MSG
                && ipc_buf.i32.pool_id < NV_IPC_MEMPOOL_NUM) {
            pool->inflight --;
            atomic_fetch_and(&job->status_mask, 0x1); // Set bit_1 to 0
        } else {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: error pool_id: %d", __func__, ipc_buf.i32.pool_id);
            return -1;
        }
    } else if (result != DOCA_ERROR_AGAIN) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: Failed to poll DMA status: %s", __func__, doca_error_get_descr(result));
        return -1;
    }
#endif
    return 0;
}

static int get_dma_write_status(priv_data_t* priv_data, int index) {
    dma_job_t* job = priv_data->tx_job_base + index;
    return atomic_load(&job->status_mask);
}

static int get_dma_read_status(priv_data_t* priv_data, int index) {
    dma_job_t* job = priv_data->rx_job_base + index;
    return atomic_load(&job->status_mask);
}

static int doca_poll_tasks(priv_data_t* priv_data)
{
    uint16_t nb_rx;

    uint8_t cc_msb_buf[NVIPC_DOCA_CC_MAX_MSG_SIZE];
    cc_msg_t *cmsg = (cc_msg_t *)cc_msb_buf;
    doca_info_t* di = &priv_data->doca_info;
    packet_info_t* pkt;

    packet_info_t async_cmsg;

    uint32_t loop_counter = 0;

    int dma_rx_pkt_index = -1;
    int dma_tx_pkt_index = -1;
    int tx_pkt_index = -1;

    uint64_t last_tsc = nvipc_rdtsc();

    while (1) {

        loop_counter ++;

        if (priv_data->cmsg_async->dequeue(priv_data->cmsg_async, &async_cmsg) == 0) {
            doca_cc_send(di, &async_cmsg.cc_msg);
        }

        // Poll doca_comm incoming messages
        if (doca_cc_recv(di, cmsg) == 0) {
            switch (cmsg->type) {
            case CC_MSG_IPC_PACKET_EVENT: // DPU <-> HOST bi-direction
                pkt = (packet_info_t*)cmsg;
                NVLOGD(TAG, "DOCA_CC RECV: type=%d len=%d value=%d", cmsg->type, cmsg->len, pkt->msg_id);
                recv_ipc_event(priv_data, pkt->msg_id);
                break;

            case CC_MSG_UL_IPC_MSG_READY: // DPU -> HOST
                pkt = (packet_info_t*)cmsg;
                NVLOGD(TAG, "DOCA_CC RECV: type=%d len=%d msg_id=0x%02X msg_index=%d", cmsg->type,
                        cmsg->len, pkt->msg_id, pkt->msg_index);
                // UL step 4: received CC message, enqueue to rx_msg queue and write an event_fd event
                if (validate_packet_info_indices(priv_data, pkt, priv_data->rxpools, __func__) != 0)
                {
                    break;
                }
                packet_info_t * ul_rx_pkt = get_rx_packet_info(priv_data, pkt->msg_index);
                memcpy(ul_rx_pkt, pkt, sizeof(packet_info_t));
                enqueue_incoming_packet(priv_data, ul_rx_pkt);
                break;

            case CC_MSG_DL_IPC_MSG_READY: // HOST -> DPU
                pkt = (packet_info_t*)cmsg;
                NVLOGD(TAG, "DOCA_CC RECV: type=%d len=%d msg_id=0x%02X msg_index=%d", cmsg->type,
                        cmsg->len, pkt->msg_id, pkt->msg_index);
                // DL step 2: HOST -> DPU IPC message is ready, DPU to start DMA read

                if (validate_packet_info_indices(priv_data, pkt, priv_data->rxpools, __func__) != 0)
                {
                    break;
                }
                packet_info_t *dl_rx_pkt = get_rx_packet_info(priv_data, pkt->msg_index);
                memcpy(dl_rx_pkt, pkt, sizeof(packet_info_t));
                dma_read_start(priv_data, dl_rx_pkt);
                break;

            case CC_MSG_IPC_MSG_FREE:
                pkt = (packet_info_t*)cmsg;
                NVLOGD(TAG, "DOCA_CC RECV: type=%d len=%d msg_id=0x%02X msg_index=%d", cmsg->type,
                        cmsg->len, pkt->msg_id, pkt->msg_index);

                nv_ipc_msg_t msg;
                msg.cell_id = pkt->cell_id;
                msg.msg_id = pkt->msg_id;
                msg.msg_len = pkt->msg_len;
                msg.data_len = pkt->data_len;
                msg.data_pool = pkt->data_pool;
                if (validate_packet_info_indices(priv_data, pkt, priv_data->txpools, __func__) != 0)
                {
                    break;
                }
                msg.msg_buf = get_ipc_buf_addr(priv_data->txpools[NV_IPC_MEMPOOL_CPU_MSG], pkt->msg_index);
                msg.data_buf = NULL;
                if(nvipc_pkt_has_data_payload(pkt))
                {
                    msg.data_buf = get_ipc_buf_addr(priv_data->txpools[pkt->data_pool], pkt->data_index);
                }
                tx_buf_free(priv_data, &msg);
                break;
            default:
                NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "CC message not supported: %d", cmsg->type);
                break;
            }
        }

        if ((tx_pkt_index = priv_data->tx_msg->dequeue(priv_data->tx_msg)) >= 0) {
            pkt = get_tx_packet_info(priv_data, tx_pkt_index);
            if (priv_data->primary) {
                // UL step 1: DPU start DMA write
                dma_write_start(priv_data, pkt);
                // ret = priv_data->tx_dma->enqueue(priv_data->tx_dma, msg_index);
            } else {
                // DL step 1: HOST send doca_comm message to DPU to start DMA read
                doca_cc_send_ipc_pkt(priv_data, CC_MSG_DL_IPC_MSG_READY, pkt);
            }
        }

        if (priv_data->primary) {

            for (int pool_id = 0; pool_id < NV_IPC_MEMPOOL_NUM; pool_id ++) {
                poll_dma_status(&priv_data->dma_pools.rx_pools[pool_id], priv_data->rx_job_base);
                poll_dma_status(&priv_data->dma_pools.tx_pools[pool_id], priv_data->tx_job_base);
            }

            // DL step 3: Poll DMA read done from rx_dma
            // static int dma_read_index = -1;
            if (dma_rx_pkt_index < 0) {
                dma_rx_pkt_index = priv_data->rx_dma->dequeue(priv_data->rx_dma);
            }

            if(dma_rx_pkt_index >= 0 && get_dma_read_status(priv_data, dma_rx_pkt_index) == 0) {
                // DL step 4: enqueue to rx_msg queue
                packet_info_t * dl_rx_pkt = get_rx_packet_info(priv_data, dma_rx_pkt_index);
                enqueue_incoming_packet(priv_data, dl_rx_pkt);
                dma_rx_pkt_index = -1;
            }

            // UL step 2: Poll DMA write done from tx_dma
            // static int dma_write_index = -1;
            if (dma_tx_pkt_index < 0) {
                dma_tx_pkt_index = priv_data->tx_dma->dequeue(priv_data->tx_dma);
            }

            if(dma_tx_pkt_index >= 0 && get_dma_write_status(priv_data, dma_tx_pkt_index) == 0) {
                // UL step 3: CC message to notify HOST with buf_index
                packet_info_t* ul_tx_pkt = get_tx_packet_info(priv_data, dma_tx_pkt_index);
                doca_cc_send_ipc_pkt(priv_data, CC_MSG_UL_IPC_MSG_READY, ul_tx_pkt);
                dma_tx_pkt_index = -1;
            }
        }

        uint64_t curr_tsc = nvipc_rdtsc();
        if (curr_tsc - last_tsc > TSC_200_US) {
            NVLOGI(TAG, "POLL: counter=%u dma_tx_pkt_index=%d dma_rx_pkt_index=%d", loop_counter,
                    dma_tx_pkt_index, dma_rx_pkt_index);
            last_tsc = curr_tsc;
        }
    }
    return 0;
}

static void* nvipc_poll_thread(void* arg)
{
    priv_data_t* priv_data = (priv_data_t*)arg;

    NVLOGC(TAG, "%s: nvipc_poll_thread thread start ...", __func__);

    char thread_name[NVLOG_NAME_MAX_LEN + 16];
    snprintf(thread_name, NVLOG_NAME_MAX_LEN + 16, "%s_poll", priv_data->prefix);
    if(pthread_setname_np(pthread_self(), thread_name) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set thread name %s failed", __func__, thread_name);
    }

    if(nv_assign_thread_cpu_core(priv_data->cpu_core) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set cpu core failed: cpu_core=%u", __func__, priv_data->cpu_core);
    }

    doca_print_cpu_core();

    while(1)
    {
        doca_poll_tasks(priv_data);
    }

    return NULL;
}

static int tx_mem_alloc(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options)
{
    if(ipc == NULL || msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    // Allocate MSG buffer
    nv_ipc_mempool_t* msgpool   = priv_data->txpools[NV_IPC_MEMPOOL_CPU_MSG];
    int32_t           msg_index = msgpool->alloc(msgpool);
    if(msg_index < 0)
    {
        NVLOGW(TAG, "%s: MSG pool is full", __func__);
        return -1;
    }
    msg->msg_buf = get_ipc_buf_addr(msgpool, msg_index);

    packet_info_t* pkt = get_tx_packet_info(priv_data, msg_index);
    pkt->msg_index = msg_index;

    // Allocate DATA buffer if has data
    if(msg->data_pool > 0 && msg->data_pool < NV_IPC_MEMPOOL_NUM)
    {
        nv_ipc_mempool_t* datapool = priv_data->txpools[msg->data_pool];
        if(datapool == NULL)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: DATA pool %d is not configured", __func__, msg->data_pool);
            return -1;
        }

        int32_t data_index = datapool->alloc(datapool);
        if(data_index < 0)
        {
            // If MSG buffer allocation failed, free the MSG buffer and return error
            msgpool->free(msgpool, msg_index);
            NVLOGW(TAG, "%s: DATA pool %d is full", __func__, msg->data_pool);
            return -1;
        }
        msg->data_buf    = datapool->get_addr(datapool, data_index);
        pkt->data_pool  = msg->data_pool;
        pkt->data_index = data_index;
    }
    else
    {
        // No DATA buffer
        msg->data_buf    = NULL;
        pkt->data_pool  = NV_IPC_MEMPOOL_CPU_MSG;
        pkt->data_index = -1;
    }

    priv_data->ipc_debug->alloc_hook(priv_data->ipc_debug, msg, msg_index);

    // NVLOGD(TAG, "%s: msg_buf=%p data_buf=%p", __func__, msg->msg_buf, msg->data_buf);
    return 0;
}

static int mem_free_dummy(nv_ipc_t* ipc, nv_ipc_msg_t* msg) {
    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid API call", __func__);
    return 0;
}

static int mem_alloc_dummy(nv_ipc_t* ipc, nv_ipc_msg_t* msg, uint32_t options) {
    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid API call", __func__);
    return 0;
}

static int rx_mem_free(nv_ipc_t *ipc, nv_ipc_msg_t *msg) {
    if(ipc == NULL || msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    if(msg->msg_buf == NULL) {
        return 0;
    }

    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    // DL step 5: notify HOST to free the buffer
    // UL step 5: notify DPU to free the buffer
    nv_ipc_mempool_t *msgpool = priv_data->rxpools[NV_IPC_MEMPOOL_CPU_MSG];
    int32_t msg_index = get_ipc_buf_index(msgpool, msg->msg_buf);
    packet_info_t *rx_pkt = get_rx_packet_info(priv_data, msg_index);
    NVLOGD(TAG, "rx_free: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%d data_pool=%d",
            msg->cell_id, msg->msg_id, msg->msg_len, msg->data_len, msg->data_pool);

    return doca_cc_send_ipc_pkt_async(priv_data, CC_MSG_IPC_MSG_FREE, rx_pkt);;
}

static int tx_mem_free(nv_ipc_t *ipc, nv_ipc_msg_t *msg) {
    if(ipc == NULL || msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    if(msg->msg_buf == NULL) {
        return 0;
    }

    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    return tx_buf_free(priv_data, msg);
}

static int tx_buf_free(priv_data_t* priv_data, nv_ipc_msg_t* msg)
{
    if(priv_data == NULL || msg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    int ret1 = 0, ret2 = 0;
    if(msg->msg_buf != NULL)
    {
        nv_ipc_mempool_t* msgpool   = priv_data->txpools[NV_IPC_MEMPOOL_CPU_MSG];
        // int32_t           msg_index = msgpool->get_index(msgpool, msg->msg_buf);
        int32_t           msg_index = get_ipc_buf_index(msgpool, msg->msg_buf);
        priv_data->ipc_debug->free_hook(priv_data->ipc_debug, msg, msg_index);

        if(get_forward_started(priv_data))
        {
            int fw_ret = -1;

            // Move to fw_ring if forwarded used MSG buffers and DATA buffers doesn't exceed the max allowed count
            if(atomic_load(&priv_data->fw_data->msg_buf_count) < priv_data->fw_max_msg_buf_count && (msg->data_buf == NULL || atomic_load(&priv_data->fw_data->data_buf_count) < priv_data->fw_max_data_buf_count))
            {
                uint32_t forwarded = atomic_fetch_add(&priv_data->fw_data->ipc_forwarded, 1);
                uint32_t total     = atomic_load(&priv_data->fw_data->ipc_total);
                if(total == 0 || total > forwarded)
                {
                    if((fw_ret = priv_data->fw_ring->enqueue(priv_data->fw_ring, msg_index)) < 0)
                    {
                        // The fw_ring size is large enough, normally should not run to here
                        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: fw_ring enqueue error", __func__);
                    }
                }
                else
                {
                    // Stop forwarding
                    atomic_fetch_sub(&priv_data->fw_data->ipc_forwarded, 1);
                    atomic_store(&priv_data->fw_data->forward_started, 0);
                }
            }
            else
            {
                // Forwarder buffers is full, skip, go to normal free
                atomic_fetch_add(&priv_data->fw_data->ipc_lost, 1);
            }

            if(fw_ret == 0)
            {
                atomic_fetch_add(&priv_data->fw_data->msg_buf_count, 1);
                if(msg->data_buf != NULL)
                {
                    atomic_fetch_add(&priv_data->fw_data->data_buf_count, 1);
                }

                NVLOGD(TAG, "Forwarder: enqueued msg_id=0x%02X", msg->msg_id);

                if(sem_post(priv_data->fw_sem) < 0)
                {
                    NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: sem_post error", __func__);
                }
                // Forwarded the message to fw_ring, do not free the buffers, skip
                return 0;
            }
        }

        ret1 = msgpool->free(msgpool, msg_index);
    }

    if(msg->data_buf != NULL)
    {
        if(msg->data_pool > 0 && msg->data_pool < NV_IPC_MEMPOOL_NUM)
        {
            nv_ipc_mempool_t* datapool = priv_data->txpools[msg->data_pool];
            if(datapool == NULL)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: DATA pool %d is not configured", __func__, msg->data_pool);
                return -1;
            }

            int32_t data_index = datapool->get_index(datapool, msg->data_buf);
            ret2               = datapool->free(datapool, data_index);
        }
        else
        {
            ret2 = -1;
        }
    }

    return (ret1 == 0 && ret2 == 0) ? 0 : -1;
}

static int ipc_tx_send(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(ipc == NULL || msg == NULL)
    {
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    nv_ipc_mempool_t* msgpool   = priv_data->txpools[NV_IPC_MEMPOOL_CPU_MSG];
    int32_t           msg_buf_id = get_ipc_buf_index(msgpool, msg->msg_buf);

    if(msg_buf_id < 0 || msg_buf_id >= priv_data->ring_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: error msg_buf_id %d", __func__, msg_buf_id);
        return -1;
    }

    priv_data->ipc_debug->send_hook(priv_data->ipc_debug, msg, msg_buf_id);

    packet_info_t* pkt = get_tx_packet_info(priv_data, msg_buf_id);
    pkt->msg_id       = msg->msg_id;
    pkt->cell_id      = msg->cell_id;
    pkt->msg_len      = msg->msg_len;
    pkt->data_len     = msg->data_len;

    NVLOGD(TAG, "send: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%d data_pool=%d", msg->cell_id, msg->msg_id, msg->msg_len, msg->data_len, msg->data_pool);

    if (priv_data->tx_msg->enqueue(priv_data->tx_msg, msg_buf_id) != 0) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: tx_msg queue is full", __func__);
        return -1;
    }

//    int ret = 0;
//    if (priv_data->primary) {
//        // UL step 1: DPU start DMA write
//        dma_write_start(priv_data, pkt);
//        // ret = priv_data->tx_dma->enqueue(priv_data->tx_dma, msg_index);
//    } else {
//        // DL step 1: HOST send doca_comm message to DPU to start DMA read
//        ret = doca_cc_send_ipc_pkt(&priv_data->doca_info, CC_MSG_DL_IPC_MSG_READY, pkt);
//    }
    return 0;
}

static int ipc_rx_recv(nv_ipc_t* ipc, nv_ipc_msg_t* msg)
{
    if(ipc == NULL || msg == NULL)
    {
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    int32_t msg_index = priv_data->rx_msg->dequeue(priv_data->rx_msg);
    if(msg_index < 0 || msg_index >= priv_data->ring_len)
    {
        return -1;
    }
    else
    {
        nv_ipc_mempool_t* msgpool = priv_data->rxpools[NV_IPC_MEMPOOL_CPU_MSG];
        msg->msg_buf              = get_ipc_buf_addr(msgpool, msg_index);

        packet_info_t* info = get_rx_packet_info(priv_data, msg_index);

        if(info->data_pool > 0 && info->data_pool < NV_IPC_MEMPOOL_NUM)
        {
            nv_ipc_mempool_t* datapool = priv_data->rxpools[info->data_pool];
            if(datapool == NULL)
            {
                // Should never run to here
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_pool %d", __func__, info->data_pool);
            }
            msg->data_pool = info->data_pool;
            msg->data_buf  = datapool->get_addr(datapool, info->data_index);
            if(msg->data_buf == NULL)
            {
                // Should never run to here
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_index %d for pool %d",
                        __func__, info->data_index, info->data_pool);
            }
        }
        else
        {
            msg->data_pool = NV_IPC_MEMPOOL_CPU_MSG;
            msg->data_buf  = NULL;
        }

        msg->msg_id   = info->msg_id;
        msg->cell_id  = info->cell_id;
        msg->msg_len  = info->msg_len;
        msg->data_len = info->data_len;

        priv_data->ipc_debug->recv_hook(priv_data->ipc_debug, msg, msg_index);

        NVLOGD(TAG, "recv: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%d data_pool=%d msg->data_buf=%p in_gpu=%d",
                msg->cell_id, msg->msg_id, msg->msg_len, msg->data_len, msg->data_pool, msg->data_buf, is_device_pointer(msg->data_buf));

        return 0;
    }
}

static int doca_efd_get_fd(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)
    return priv_data->efd_rx;
}

static int doca_efd_notify(nv_ipc_t* ipc, int value)
{
    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    if(CONFIG_DL_EXPLICIT_PACKET_SYNC && priv_data->primary == 0)
    {
        return send_ipc_event(ipc, value);
    }
    else if(CONFIG_UL_EXPLICIT_PACKET_SYNC && priv_data->primary != 0)
    {
        return send_ipc_event(ipc, value);
    }
    else
    {
        return 0;
    }
}

static int doca_efd_get_value(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK_BLOCKING(priv_data)

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

static int doca_tx_tti_sem_post(nv_ipc_t* ipc)
{
    return doca_efd_notify(ipc, 1);
}

static int doca_rx_tti_sem_wait(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    IPC_DUMPING_CHECK_BLOCKING(priv_data)

    int ret = ipc_epoll_wait(priv_data->ipc_epoll);

    uint64_t efd_value;
    ssize_t  size = read(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
    NVLOGV(TAG, "%s: size=%ld efd_value=%lu", __func__, size, efd_value);

    return ret;
}

static int cpu_memcpy(nv_ipc_t* ipc, void* dst, const void* src, size_t size)
{
    return 0;
}

static int doca_ipc_close(nv_ipc_t* ipc)
{
    priv_data_t* priv_data = get_private_data(ipc);
    int          ret       = 0;

    IPC_DUMPING_CHECK(priv_data)

    // Close epoll FD if exist
    if(priv_data->ipc_epoll != NULL)
    {
        if(ipc_epoll_destroy(priv_data->ipc_epoll) < 0)
        {
            ret = -1;
        }
    }

    // TODO: clean up DOCA COMM, BUF_INV, NMAP, DMA

    // Destroy the nv_ipc_t instance
    free(ipc);

    if(ret == 0)
    {
        NVLOGC(TAG, "%s: OK", __func__);
    } else {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: failed with ret=%d", __func__, ret);
    }
    return ret;
}

static int64_t ts_diff = 0;

static int recv_doca_comm_connect(nv_ipc_t* ipc, void* buf)
{
    priv_data_t *priv_data = get_private_data(ipc);

    int count = 0;

    size_t size = NVIPC_DOCA_CC_MAX_MSG_SIZE;
    if (doca_comm_recv(&priv_data->doca_info, buf, &size) == 0) {
        nv_ipc_msg_t *msg = (nv_ipc_msg_t*) buf;
        if(msg->msg_len < (int32_t)sizeof(nv_ipc_msg_t)
                || msg->msg_len > (int32_t)NVIPC_DOCA_CC_MAX_MSG_SIZE)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid connect msg_len=%d", __func__,
                    msg->msg_len);
            return -1;
        }
        msg->msg_buf = buf;

        struct timespec *p_ts_send = (struct timespec*) ((uint8_t*) msg->msg_buf + msg->msg_len
                - sizeof(struct timespec));
        struct timespec ts_recv;
        nvlog_gettime_rt(&ts_recv);
        ts_diff = nvlog_timespec_interval(p_ts_send, &ts_recv);

        if (msg->msg_id != MSG_ID_CONNECT) {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Error connect msg_id=0x%02X", msg->msg_id);
        }
        return 0;
    } else {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Error connect count=%d", count);
        return -1;
    }
}

static int send_doca_comm_connect(nv_ipc_t* ipc, void* buf)
{
    priv_data_t* priv_data = get_private_data(ipc);

    nv_ipc_msg_t* msg = (nv_ipc_msg_t*)buf;

    memset(msg, 0, sizeof(nv_ipc_msg_t));
    msg->msg_id  = MSG_ID_CONNECT;
    msg->msg_len += sizeof(nv_ipc_msg_t);
    msg->msg_buf = buf;

    struct timespec* ts = (struct timespec*)((uint8_t*)msg->msg_buf + msg->msg_len);
    nvlog_gettime_rt(ts);
    msg->msg_len += sizeof(struct timespec);

    doca_comm_send(&priv_data->doca_info, buf, msg->msg_len);
    return 0;
}

static int doca_comm_connect(nv_ipc_t* ipc)
{
    priv_data_t *priv_data = get_private_data(ipc);

    int count = 0;
    uint8_t connect_buf[NVIPC_DOCA_CC_MAX_MSG_SIZE];
    if (priv_data->primary) {
        NVLOGC(TAG, "local_pci: %s representor_pci: %s wait for connect ...",
                priv_data->doca_info.dev_pci_str, priv_data->doca_info.rep_pci_str);

        while (count < CONNECT_TEST_COUNT) {
            if (recv_doca_comm_connect(ipc, connect_buf) == 0) {
                send_doca_comm_connect(ipc, connect_buf);
            }
            count++;
            NVLOGI(TAG, "received message: count=%d ts_diff=%ld", count, ts_diff);
        }
        NVLOGC(TAG, "Connected from %s. ts_diff=%ld", priv_data->doca_info.rep_pci_str, ts_diff);
    } else {
        NVLOGC(TAG, "Test connection from %s", priv_data->doca_info.dev_pci_str);

        int64_t min = 1000 * 1000 * 1000L;
        int64_t max = 0;
        int64_t total = 0;
        struct timespec start, end;
        while (count < CONNECT_TEST_COUNT) {
            nvlog_gettime_rt(&start);
            send_doca_comm_connect(ipc, connect_buf);
            while (recv_doca_comm_connect(ipc, connect_buf) != 0) {
            }
            count++;
            nvlog_gettime_rt(&end);
            int64_t interval = nvlog_timespec_interval(&start, &end);
            total += interval;
            max = max < interval ? interval : max;
            min = min > interval ? interval : min;
            NVLOGI(TAG, "Connection delay test: interval=%ld ts_diff=%ld", interval, ts_diff);
        }

        NVLOGC(TAG, "Connected. delay: min=%ld avg=%ld max=%ld ts_diff=%ld", min,
                total / CONNECT_TEST_COUNT, max, ts_diff);
    }
    return 0;
}

int doca_dma_device_open(priv_data_t *priv_data) {
    doca_error_t result = DOCA_ERROR_UNKNOWN;
    doca_info_t *di = &priv_data->doca_info;

    NVLOGC(TAG, "%s", __func__);

    // Find and open DMA device with capability DOCA_DMA_JOB_MEMCPY
    for (uint32_t i = 0; i < di->nb_devs; i++) {
        // if (doca_dma_job_get_supported(di->dev_list[i], DOCA_DMA_JOB_MEMCPY) != DOCA_SUCCESS) {
        //     continue;
        // }
        if (doca_dev_open(di->dev_list[i], &di->dma_dev) == DOCA_SUCCESS) {
            di->devinfo_dma = di->dev_list[i];
            result = 0;
            break;
        }
    }
    if (result != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "DMA device not found.");
        return -1;
    }

    if (priv_data->primary) {
        if ((result = doca_dma_create(di->dma_dev, &di->dma_ctx)) != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to create DMA engine: %s",
                    doca_error_get_descr(result));
            return -1;
        }

        if ((di->ctx = doca_dma_as_ctx(di->dma_ctx)) == NULL) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: doca_dma_as_ctx failed", __func__);
            return -1;
        }

        // TODO: upgrade for DOCA 2.5
        // result = doca_ctx_dev_add(di->ctx, di->dma_dev);
        // if (result != DOCA_SUCCESS) {
        //     NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to register device with DMA context: %s",
        //             doca_error_get_descr(result));
        //     return result;
        // }

        result = doca_ctx_start(di->ctx);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to start DMA context: %s",
                    doca_error_get_descr(result));
            return result;
        }
    }
    return 0;
}

int doca_mp_init_host(doca_info_t *di, nv_ipc_mempool_t *mp, mmap_info_t *mmap, dma_info_t *dma) {
    doca_error_t result;

    size_t buf_size = mp->get_buf_size(mp);
    size_t pool_size = buf_size * mp->get_pool_len(mp);
    void *pool_addr = mp->get_addr(mp, 0);

    dma->inflight = 0;

    size_t max_dma_buf_size;
    if ((result = doca_dma_cap_task_memcpy_get_max_buf_size(di->devinfo_dma, &max_dma_buf_size)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to doca_dma_cap_task_memcpy_get_max_buf_size: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if (buf_size > max_dma_buf_size) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: DMA buffer size limited: max=%lu expected=%lu",
                __func__, max_dma_buf_size, buf_size);
        return -1;
    }

    // DOCA local MMAP create and initiate
    if ((result = doca_mmap_create(&dma->mmap_local)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to create mmap: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if ((result = doca_mmap_add_dev(dma->mmap_local, di->dma_dev)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to add device to mmap: %s",
                doca_error_get_descr(result));
        return -1;
    }

    // TODO: upgrade for DOCA 2.5
    if ((result = doca_mmap_set_permissions(dma->mmap_local, DOCA_ACCESS_FLAG_RDMA_WRITE))
            != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to set access permissions of memory map: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if ((result = doca_mmap_set_memrange(dma->mmap_local, pool_addr, pool_size)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to set memrange of memory map: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if ((result = doca_mmap_start(dma->mmap_local)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to start memory map: %s",
                doca_error_get_descr(result));
        return -1;
    }

    // Export memory map to allow access to this memory region from DPU
    const void *export_desc = NULL;
    // TODO: upgrade for DOCA 2.5
    if ((result = doca_mmap_export_pci(dma->mmap_local, di->dma_dev, &export_desc, &mmap->desc_len))
            != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to export DOCA mmap: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if (mmap->desc_len > MAX_MMAP_EXPORT_DESC_LEN) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "desc_len exceeds limitation: %lu > %d",
                mmap->desc_len, MAX_MMAP_EXPORT_DESC_LEN);
        return -1;
    }

    // Host prepare the memory map export descriptor and send to DPU later
    mmap->mmap_addr = pool_addr;
    mmap->mmap_size = pool_size;
    memcpy(mmap->desc, export_desc, mmap->desc_len);

    NVLOGC(TAG, "%s: HOST MMAP exported: addr=%p size=%lu desc_len=%lu max_dma_buf_size=%lu",
            __func__, mmap->mmap_addr, mmap->mmap_size, mmap->desc_len, max_dma_buf_size);

    return 0;
}

int doca_mp_init_dpu(doca_info_t *di, nv_ipc_mempool_t *mp, mmap_info_t *mmap, dma_info_t *dma) {
    doca_error_t result;

    size_t buf_size = mp->get_buf_size(mp);
    size_t pool_len = mp->get_pool_len(mp);
    size_t pool_size = buf_size * pool_len;
    void *pool_addr = mp->get_addr(mp, 0);

    dma->inflight = 0;

    size_t max_dma_buf_size;
    if ((result = doca_dma_cap_task_memcpy_get_max_buf_size(di->devinfo_dma, &max_dma_buf_size)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to doca_dma_cap_task_memcpy_get_max_buf_size: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if (buf_size > max_dma_buf_size) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "%s: DMA buffer size limited: max=%lu expected=%lu",
                __func__, max_dma_buf_size, buf_size);
        return -1;
    }

    // DOCA local MMAP create and initiate
    if ((result = doca_mmap_create(&dma->mmap_local)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to create mmap: %s",
                doca_error_get_descr(result));
        return -1;
    }

    if ((result = doca_mmap_add_dev(dma->mmap_local, di->dma_dev)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to add device to mmap: %s",
                doca_error_get_descr(result));
        return -1;
    }

    // DOCA buffer inventory create and initiate
    if ((result = doca_buf_inventory_create(pool_len * 2, &dma->buf_inv)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to create buffer inventory: %s",
                doca_error_get_descr(result));
        return result;
    }

    // DOCA DMA work queue
    // TODO: upgrade for DOCA 2.5
    // result = doca_workq_create(pool_len, &dma->workq);
    // if (result != DOCA_SUCCESS) {
    //     NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to create work queue: %s",
    //             doca_error_get_descr(result));
    //     return result;
    // }

    result = doca_buf_inventory_start(dma->buf_inv);
    if (result != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to start buffer inventory: %s",
                doca_error_get_descr(result));
        return result;
    }

    // TODO: upgrade for DOCA 2.5
    // doca_ctx_dev_add, doca_ctx_start
    // result = doca_ctx_workq_add(di->ctx, dma->workq);
    // if (result != DOCA_SUCCESS) {
    //     NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to register work queue with context: %s",
    //             doca_error_get_descr(result));
    //     return result;
    // }

    //======================================================
    // TODO: upgrade for DOCA 2.5
    if ((doca_mmap_set_permissions(dma->mmap_local, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE))
            != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to set access permissions of memory map: %s",
                doca_error_get_descr(result));
        return result;
    }

    if (mmap->mmap_size != pool_size) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: MMAP size not match: export=%lu expected=%lu",
                __func__, mmap->mmap_size, pool_size);
        return -1;
    }

    if ((doca_mmap_set_memrange(dma->mmap_local, pool_addr, mmap->mmap_size)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to set memrange of memory map: %s",
                doca_error_get_descr(result));
        return result;
    }

    result = doca_mmap_start(dma->mmap_local);
    if (result != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to start memory map: %s",
                doca_error_get_descr(result));
        return result;
    }

    // Create a local DOCA mmap from export descriptor
    if ((doca_mmap_create_from_export(NULL, mmap->desc, mmap->desc_len, di->dma_dev,
            &dma->mmap_remote)) != DOCA_SUCCESS) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to create memory map from export descriptor");
        return result;
    }

    dma->jobs = malloc(sizeof(dma_job_t) * pool_len);

    for (int i = 0; i < pool_len; i++) {
        dma_job_t *job = dma->jobs + i;

        // Construct DOCA buffer for remote (Host) address range
        uint8_t *remote_ipc_buf = mmap->mmap_addr + buf_size * i;
        result = doca_buf_inventory_buf_get_by_addr(dma->buf_inv, dma->mmap_remote, remote_ipc_buf,
                buf_size, &job->remote_buf);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to acquire DOCA remote buffer: %s",
                    doca_error_get_descr(result));
            return result;
        }

        result = doca_buf_set_data(job->remote_buf, remote_ipc_buf, 0);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to set data for DOCA buffer: %s",
                    doca_error_get_descr(result));
            return result;
        }

        /* Construct DOCA buffer for local (DPU) address range */
        uint8_t *local_ipc_buf = pool_addr + buf_size * i;
        result = doca_buf_inventory_buf_get_by_addr(dma->buf_inv, dma->mmap_local, local_ipc_buf,
                buf_size, &job->local_buf);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Unable to acquire DOCA local buffer: %s",
                    doca_error_get_descr(result));
            return result;
        }

        result = doca_buf_set_data(job->local_buf, local_ipc_buf, 0);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to set data for DOCA buffer: %s",
                    doca_error_get_descr(result));
            return result;
        }
    }

    NVLOGC(TAG, "%s: DPU MMAP imported: addr=%p size=%lu desc_len=%lu max_dma_buf_size=%lu",
            __func__, mmap->mmap_addr, mmap->mmap_size, mmap->desc_len, max_dma_buf_size);
    return 0;
}

int negotiate_mmap_descriptors(priv_data_t *priv_data, int step) {
    doca_info_t *di = &priv_data->doca_info;
    struct timespec ts = { .tv_nsec = SLEEP_IN_NANOS, };

    NVLOGC(TAG, "%s: primary=%d step=%d start ...", __func__, priv_data->primary, step);

    doca_error_t result;
    if (di->mode == DMA_COPY_MODE_HOST && step == 2) {
        // Send the memory map export descriptor to DPU
        while ((result = doca_comm_channel_ep_sendto(di->ep, &priv_data->mmap_descs,
                sizeof(mmap_descs_t), DOCA_CC_MSG_FLAG_NONE, di->peer_addr)) == DOCA_ERROR_AGAIN) {
            nanosleep(&ts, &ts);
        }
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to send config files to DPU: %s",
                    doca_error_get_descr(result));
            return -1;
        }
        NVLOGC(TAG, "Host sent MMAP descriptor: total_size=%lu, waiting for response ...",
                sizeof(mmap_descs_t));

        result = wait_for_successful_status_msg(di->ep, &di->peer_addr);
        if (result != DOCA_SUCCESS) {
            return -1;
        }
        NVLOGC(TAG, "HOST MMAP negotiated done: total_size=%lu", sizeof(mmap_descs_t));
    } else if (di->mode == DMA_COPY_MODE_DPU) {
        if (step == 1) {
            // Receive exported descriptor from Host
            char recv_buf[NVIPC_DOCA_CC_MAX_MSG_SIZE];
            size_t msg_len = NVIPC_DOCA_CC_MAX_MSG_SIZE;
            while ((result = doca_comm_channel_ep_recvfrom(di->ep, recv_buf, &msg_len,
                    DOCA_CC_MSG_FLAG_NONE, &di->peer_addr)) == DOCA_ERROR_AGAIN) {
                nanosleep(&ts, &ts);
                msg_len = NVIPC_DOCA_CC_MAX_MSG_SIZE;
            }
            if (result != DOCA_SUCCESS) {
                NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT,
                        "Failed to receive export descriptor from Host: %s",
                        doca_error_get_descr(result));
                return -1;
            }
            if (msg_len != sizeof(mmap_descs_t)) {
                NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT,
                        "Export descriptor size not match: received=%lu expected=%lu", msg_len,
                        sizeof(mmap_descs_t));
                return -1;
            }
            memcpy(&priv_data->mmap_descs, recv_buf, msg_len);
            NVLOGC(TAG, "DPU received MMAP descriptor: total_size=%lu", sizeof(mmap_descs_t));
        } else {
            usleep(100);
            result = send_status_msg(di->ep, &di->peer_addr, true);
            NVLOGC(TAG, "DPU MMAP negotiated done: total_size=%lu", sizeof(mmap_descs_t));
        }
    }

    NVLOGC(TAG, "%s: primary=%d step=%d done", __func__, priv_data->primary, step);
    return 0;
}

static int doca_info_init(nv_ipc_t* ipc, const nv_ipc_config_doca_t* cfg) {
    priv_data_t *priv_data = get_private_data(ipc);
    doca_info_t *di = &priv_data->doca_info;

    print_doca_sdk_version();

    // nvipc_doca_sample_test(cfg);

    // DOCA initiate
    memset(&priv_data->doca_info, 0, sizeof(doca_info_t));

    priv_data->doca_info.mode = cfg->primary ? 1 : 0;

    nvlog_safe_strncpy(priv_data->doca_info.name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(priv_data->doca_info.name, "_doca", NV_NAME_MAX_LEN);

    if (cfg->primary)
    {
        nvlog_safe_strncpy(priv_data->doca_info.dev_pci_str, cfg->dpu_pci, PCI_ADDR_LEN);
        nvlog_safe_strncpy(priv_data->doca_info.rep_pci_str, cfg->host_pci, PCI_ADDR_LEN);
    } else {
        nvlog_safe_strncpy(priv_data->doca_info.dev_pci_str, cfg->host_pci, PCI_ADDR_LEN);
    }

    NVLOGC(TAG, "%s: dev_pci=%s rep_pci=%s", __func__, di->dev_pci_str, di->rep_pci_str);

#ifdef NVIPC_DOCA_GPUNETIO
    if (priv_data->primary) {
        char *eal_param[3] = { "nvipc_doca", "-a", "00:00.0" };
        int ret = rte_eal_init(3, eal_param);
        if (ret < 0) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "DPDK init failed: %d", ret);
        }

        /* Initialize DOCA GPU instance */
        nvlog_safe_strncpy(di->gpu_pcie_addr, "0000:06:00.0", NV_NAME_MAX_LEN);
        doca_error_t result = doca_gpu_create(di->gpu_pcie_addr, &di->gpu_dev);
        if (result != DOCA_SUCCESS) {
            NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "doca_gpu_create failed: %s",
                    doca_error_get_descr(result));
            return -1;
        }
    }
#endif

    if (doca_comm_channel_init(di) != 0) {
        return -1;
    }

    if(doca_comm_connect(ipc) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: fail to connect with peer", __func__);
        return -1;
    }

    // Open DOCA DMA device
    if (doca_dma_device_open(priv_data) != 0) {
        NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "Failed to open DMA device");
        return -1;
    }

    return 0;
}

static int debug_get_msg(priv_data_t* priv_data, nv_ipc_msg_t* msg, int msg_index, int tx)
{
    if(msg_index < 0 || msg_index >= priv_data->ring_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_index=%d ring_len=%d", __func__,
                msg_index, priv_data->ring_len);
        return -1;
    }

    nv_ipc_mempool_t** mempools = tx ? priv_data->txpools : priv_data->rxpools;
    nv_ipc_mempool_t* msgpool = mempools[NV_IPC_MEMPOOL_CPU_MSG];
    if((msg->msg_buf = msgpool->get_addr(msgpool, msg_index)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_index=%d", __func__, msg_index);
        return -1;
    }

    packet_info_t* info = tx ? priv_data->tx_pkt_infos + msg_index : priv_data->rx_pkt_infos + msg_index;
    msg->msg_id        = info->msg_id;
    msg->cell_id       = info->cell_id;
    msg->msg_len       = info->msg_len;
    msg->data_len      = info->data_len;
    msg->data_pool     = info->data_pool;

    if(msg->msg_len < 0 || msg->msg_len > priv_data->msg_buf_size)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid msg_len=%d", __func__, msg->msg_len);
        return -1;
    }

    if(msg->data_pool > 0 && msg->data_pool < NV_IPC_MEMPOOL_NUM)
    {
        nv_ipc_mempool_t* datapool = mempools[msg->data_pool];
        if(datapool != NULL)
        {
            msg->data_buf = datapool->get_addr(datapool, info->data_index);
            if(msg->data_buf == NULL)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_index %d for pool %d",
                        __func__, info->data_index, msg->data_pool);
                return -1;
            }
            if(msg->data_len < 0 || msg->data_len > datapool->get_buf_size(datapool))
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid data_len=%d", __func__,
                        msg->data_len);
                return -1;
            }
        }
        else
        {
            return -1;
        }
    }
    else
    {
        msg->data_pool = NV_IPC_MEMPOOL_CPU_MSG;
        msg->data_buf  = NULL;
    }
    return 0;
}

static int debug_dump_queue(priv_data_t *priv_data, array_queue_t *queue, int32_t *mempool_status,
        int32_t mempool_size, int32_t flag, const char *info, int tx) {
    char *queue_name = queue->get_name(queue);
    int32_t count = queue->get_count(queue);
    int32_t max_length = queue->get_max_length(queue);
    if(count < 0)
    {
        count = 0;
    }
    else if(count > max_length)
    {
        count = max_length;
    }
    int32_t base = -1, counter = 0;
    NVLOGC(TAG, "%s: count=%d max_length=%d", info, count, max_length);

    int32_t msg_index;
    while ((msg_index = queue->get_next(queue, base)) >= 0 && counter < count) {
        nv_ipc_msg_t msg;
        if (debug_get_msg(priv_data, &msg, msg_index, tx) == 0) {
            nv_ipc_dump_msg(priv_data->ipc_debug, &msg, msg_index, info);
        }

        if (msg_index < mempool_size) {
            *(mempool_status + msg_index) |= flag;
        } else {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: msg_index=%d < mempool_size=%d - error",
                    info, msg_index, mempool_size);
        }
        base = msg_index;
        counter++;
    }
    return 0;
}

static int debug_dump_mempools(priv_data_t *priv_data, nv_ipc_mempool_t *mempool,
        int32_t *mempool_status, int32_t mempool_size, const char *info, int tx) {
    array_queue_t*    queue   = mempool->get_free_queue(mempool);

    char*   queue_name = queue->get_name(queue);
    int32_t count      = queue->get_count(queue);
    int32_t max_length = queue->get_max_length(queue);
    if(count < 0)
    {
        count = 0;
    }
    else if(count > max_length)
    {
        count = max_length;
    }
    int32_t base = -1, counter = 0;

    NVLOGC(TAG, "%s: mempool_size=%d free_count=%d max_length=%d", info, mempool_size, count, max_length);

    int32_t msg_index;
    while((msg_index = queue->get_next(queue, base)) >= 0 && counter < count)
    {
        if(msg_index < mempool_size)
        {
            *(mempool_status + msg_index) |= 0x8;
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: msg_index=%d < mempool_size=%d - error", info, msg_index, mempool_size);
        }
        base = msg_index;
        counter++;
    }

    for(msg_index = 0; msg_index < mempool_size; msg_index++)
    {
        if(*(mempool_status + msg_index) == 0)
        {
            nv_ipc_msg_t msg;
            if(debug_get_msg(priv_data, &msg, msg_index, tx) == 0)
            {
                nv_ipc_dump_msg(priv_data->ipc_debug, &msg, msg_index, info);
            }
        }
    }

    return 0;
}

typedef enum {
    IPC_BUF_ALLOCATED = 0x01,
    IPC_BUF_IN_QUEUE = 0x02,
} ipc_buf_status_t;

int doca_ipc_dump(nv_ipc_t* ipc)
{
    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }
    priv_data_t* priv_data = get_private_data(ipc);
    atomic_fetch_add(&priv_data->ipc_debug->shm_data->ipc_dumping, 1);

    int ring_len = priv_data->ring_len;

    // Status of each MSG buffer: bit1 - allocated; bit2 - in DL queue; bit3 - in UL queue; bit4 - free
    int32_t* mempool_status = malloc(ring_len * sizeof(int32_t));
    if (mempool_status == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: malloc failed", __func__);
        return -1;
    }

    NVLOGC(TAG, "========== Dump TX queue: TX ongoing ======================");
    memset(mempool_status, 0, ring_len * sizeof(int32_t));
    debug_dump_queue(priv_data, priv_data->tx_msg, mempool_status, ring_len, 0x01, "TX_MSG", 1);
    debug_dump_queue(priv_data, priv_data->tx_dma, mempool_status, ring_len, 0x02, "TX_DMA", 1);
    NVLOGC(TAG, "========== Dump memory pool: buffers allocated but not send OR received but not released ==================");
    debug_dump_mempools(priv_data, priv_data->txpools[NV_IPC_MEMPOOL_CPU_MSG], mempool_status, ring_len, "TX_POOL", 0);
    NVLOGC(TAG, "========== Dump TX finished =========================");

    NVLOGC(TAG, "========== Dump RX queue: RX ongoing ======================");
    memset(mempool_status, 0, ring_len * sizeof(int32_t));
    debug_dump_queue(priv_data, priv_data->rx_msg, mempool_status, ring_len, 0x04, "RX_MSG", 0);
    debug_dump_queue(priv_data, priv_data->rx_dma, mempool_status, ring_len, 0x08, "RX_DMA", 0);
    NVLOGC(TAG, "========== Dump memory pool: buffers allocated but not send OR received but not released ==================");
    debug_dump_mempools(priv_data, priv_data->rxpools[NV_IPC_MEMPOOL_CPU_MSG], mempool_status, ring_len, "RX_POOL", 0);
    NVLOGC(TAG, "========== Dump RX finished =========================");

    if(priv_data->forward_enable)
    {
        NVLOGC(TAG, "========== Dump FW queue: FW not dequeued ======================");
        uint32_t started   = atomic_load(&priv_data->fw_data->forward_started);
        uint32_t forwarded = atomic_load(&priv_data->fw_data->ipc_forwarded);
        uint32_t lost      = atomic_load(&priv_data->fw_data->ipc_lost);
        uint32_t total     = atomic_load(&priv_data->fw_data->ipc_total);
        uint32_t msg_buf   = atomic_load(&priv_data->fw_data->msg_buf_count);
        uint32_t data_buf  = atomic_load(&priv_data->fw_data->data_buf_count);
        NVLOGC(TAG, "FW status: started=%u forwarded=%u total=%u lost=%u msg_buf=%u data_buf=%u", started, forwarded, total, lost, msg_buf, data_buf);

        debug_dump_queue(priv_data, priv_data->fw_ring, mempool_status, ring_len, 0x10, "FW", 2);
    }

    free(mempool_status);
    return 0;
}

static int doca_ipc_open(nv_ipc_t* ipc, const nv_ipc_config_t* nvipc_config)
{
    int ret = 0;
    const nv_ipc_config_doca_t* cfg = &nvipc_config->transport_config.doca;

    priv_data_t* priv_data    = get_private_data(ipc);
    priv_data->primary        = cfg->primary;
    priv_data->cpu_core       = cfg->cpu_core;
    priv_data->cuda_device_id = cfg->cuda_device_id;
    priv_data->nic_mtu        = cfg->nic_mtu;
    priv_data->ring_len       = cfg->mempool_size[NV_IPC_MEMPOOL_CPU_MSG].pool_len;

    nvlog_safe_strncpy(priv_data->prefix, cfg->prefix, NV_NAME_MAX_LEN);

    // Check prefix string length
    size_t prefix_len = strnlen(cfg->prefix, NV_NAME_MAX_LEN);
    if(prefix_len <= 0 || prefix_len >= NV_NAME_MAX_LEN)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s error prefix string length %lu", __func__, prefix_len);
        return -1;
    }

    int shm_primary = 1;
    if (nvipc_config->module_type == NV_IPC_MODULE_IPC_DUMP) {
        shm_primary = 0;
    }

    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];

    // Create a shared memory pool for the TX and RX queue
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, shm_suffix, NV_NAME_SUFFIX_MAX_LEN);
    size_t ring_queue_size = ARRAY_QUEUE_HEADER_SIZE(priv_data->ring_len);
    size_t ring_objs_size  = sizeof(packet_info_t) * priv_data->ring_len * 2;
    size_t shm_size        = ring_queue_size * 4 + ring_objs_size;
    if((priv_data->shmpool = nv_ipc_shm_open(shm_primary, name, shm_size)) == NULL)
    {
        return -1;
    }
    int8_t* shm_addr        = priv_data->shmpool->get_mapped_addr(priv_data->shmpool);

    // TX MSG ring
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, ring_suffix[0], NV_NAME_SUFFIX_MAX_LEN);
    if((priv_data->tx_msg = array_queue_open(shm_primary, name, shm_addr + 0 * ring_queue_size, priv_data->ring_len)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create tx_msg queue failed", __func__);
        return -1;
    }

    // RX MSG ring
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, ring_suffix[1], NV_NAME_SUFFIX_MAX_LEN);
    if((priv_data->rx_msg = array_queue_open(shm_primary, name, shm_addr + 1 * ring_queue_size, priv_data->ring_len)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create rx_msg queue failed", __func__);
        return -1;
    }

    // TX DMA ring
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, ring_suffix[2], NV_NAME_SUFFIX_MAX_LEN);
    if((priv_data->tx_dma = array_queue_open(shm_primary, name, shm_addr + 2 * ring_queue_size, priv_data->ring_len)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create tx_dma queue failed", __func__);
        return -1;
    }

    // RX DMA ring
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, ring_suffix[3], NV_NAME_SUFFIX_MAX_LEN);
    if((priv_data->rx_dma = array_queue_open(shm_primary, name, shm_addr + 3 * ring_queue_size, priv_data->ring_len)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create rx_dma queue failed", __func__);
        return -1;
    }

    // TX, RX packet_info_t array
    priv_data->tx_pkt_infos = (packet_info_t*) (shm_addr + 4 * ring_queue_size);
    priv_data->rx_pkt_infos = priv_data->tx_pkt_infos + priv_data->ring_len;

    // DOCA common channel ring
    nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
    strncat(name, comm_suffix, NV_NAME_SUFFIX_MAX_LEN);
    if((priv_data->cmsg_async = nv_ipc_ring_open(shm_primary, name, priv_data->ring_len * 10, sizeof(packet_info_t))) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create rx_dma queue failed", __func__);
        return -1;
    }

    // Runs in IPC dump APP
    if (shm_primary == 0) {
        for(int pool_id = 0; pool_id < NV_IPC_MEMPOOL_NUM; pool_id++)
        {
            NVLOGD(TAG, "%s: shm_primary=%d pool_id=%d buff_size=%d pool_size=%d ", __func__,
                    shm_primary, pool_id, cfg->mempool_size[pool_id].buf_size,
                    cfg->mempool_size[pool_id].pool_len);
            int cuda_device_id = NV_IPC_MEMPOOL_NO_CUDA_DEV;
            if((pool_id == NV_IPC_MEMPOOL_CUDA_DATA)||(pool_id == NV_IPC_MEMPOOL_GPU_DATA))
            {
                cuda_device_id = cfg->cuda_device_id;
            }
#ifdef NVIPC_DOCA_GPUNETIO
            if (priv_data->primary && pool_id == NV_IPC_MEMPOOL_CUDA_DATA) {
                cuda_device_id = NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS;
            }
#endif
            if(cfg->mempool_size[pool_id].buf_size > 0 && cfg->mempool_size[pool_id].pool_len > 0)
            {
                nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
                strncat(name, tx_pool_suffix[pool_id], NV_NAME_SUFFIX_MAX_LEN);
                if ((priv_data->txpools[pool_id] = nv_ipc_mempool_open(shm_primary, name,
                        cfg->mempool_size[pool_id].buf_size, cfg->mempool_size[pool_id].pool_len,
                        cuda_device_id)) == NULL) {
                    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create memory pool %s failed",
                            __func__, name);
                    return -1;
                }

                nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
                strncat(name, rx_pool_suffix[pool_id], NV_NAME_SUFFIX_MAX_LEN);
                if ((priv_data->rxpools[pool_id] = nv_ipc_mempool_open(shm_primary, name,
                        cfg->mempool_size[pool_id].buf_size, cfg->mempool_size[pool_id].pool_len,
                        cuda_device_id)) == NULL) {
                    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create memory pool %s failed",
                            __func__, name);
                    return -1;
                }
            }
        }
        // Return for IPC dump
        return 0;
    }

    if(doca_info_init(ipc, cfg)) {
        return -1;
    }

    negotiate_mmap_descriptors(priv_data, 1);

    for(int pool_id = 0; pool_id < NV_IPC_MEMPOOL_NUM; pool_id++)
    {
        NVLOGC(TAG, "%s: primary=%d pool_id=%d buff_size=%d pool_len=%d ", __func__, cfg->primary,
                pool_id, cfg->mempool_size[pool_id].buf_size, cfg->mempool_size[pool_id].pool_len);
        if(cfg->mempool_size[pool_id].buf_size > 0 && cfg->mempool_size[pool_id].pool_len > 0)
        {
            int cuda_device_id = NV_IPC_MEMPOOL_NO_CUDA_DEV;
            if((pool_id == NV_IPC_MEMPOOL_CUDA_DATA)||(pool_id == NV_IPC_MEMPOOL_GPU_DATA))
            {
                cuda_device_id = cfg->cuda_device_id;
            }
#ifdef NVIPC_DOCA_GPUNETIO
            if (priv_data->primary && pool_id == NV_IPC_MEMPOOL_CUDA_DATA) {
                cuda_device_id = NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS;
            }
#endif
            nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
            strncat(name, tx_pool_suffix[pool_id], NV_NAME_SUFFIX_MAX_LEN);
            if ((priv_data->txpools[pool_id] = nv_ipc_mempool_open(shm_primary, name,
                    cfg->mempool_size[pool_id].buf_size, cfg->mempool_size[pool_id].pool_len,
                    cuda_device_id)) == NULL) {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create memory pool %s failed",
                        __func__, name);
                return -1;
            }

            nvlog_safe_strncpy(name, cfg->prefix, NV_NAME_MAX_LEN);
            strncat(name, rx_pool_suffix[pool_id], NV_NAME_SUFFIX_MAX_LEN);
            if ((priv_data->rxpools[pool_id] = nv_ipc_mempool_open(shm_primary, name,
                    cfg->mempool_size[pool_id].buf_size, cfg->mempool_size[pool_id].pool_len,
                    cuda_device_id)) == NULL) {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: create memory pool %s failed",
                        __func__, name);
                return -1;
            }

            if(CONFIG_ENABLE_HOST_PAGE_LOCK && cfg->cuda_device_id >= 0 && pool_id == NV_IPC_MEMPOOL_CPU_DATA)
            {
                size_t size = cfg->mempool_size[pool_id].buf_size * cfg->mempool_size[pool_id].pool_len;
                if(nv_ipc_page_lock(priv_data->txpools[pool_id]->get_addr(priv_data->txpools[pool_id], 0), size) < 0)
                {
                    return -1;
                }
                if(nv_ipc_page_lock(priv_data->rxpools[pool_id]->get_addr(priv_data->rxpools[pool_id], 0), size) < 0)
                {
                    return -1;
                }
            }

#ifdef NVIPC_DOCA_GPUNETIO
            if (cuda_device_id == NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS) {
                doca_error_t result;
                void *gpu_addr;
                void *cpu_addr;
                size_t size = cfg->mempool_size[pool_id].buf_size
                        * cfg->mempool_size[pool_id].pool_len;

                // Allocate TX DOCA GPU memory pool
                result = doca_gpu_mem_alloc(priv_data->doca_info.gpu_dev, size, 4096,
                        DOCA_GPU_MEM_GPU, &gpu_addr, &cpu_addr);
                if (result != DOCA_SUCCESS || gpu_addr == NULL) {
                    NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "doca_gpu_mem_alloc failed for TX %s",
                            doca_error_get_descr(result));
                    return -1;
                }
                nv_ipc_mempool_set_ext_bufs(priv_data->txpools[pool_id], gpu_addr, cpu_addr);

                // Allocate RX DOCA GPU memory pool
                result = doca_gpu_mem_alloc(priv_data->doca_info.gpu_dev, size, 4096,
                        DOCA_GPU_MEM_GPU, &gpu_addr, &cpu_addr);
                if (result != DOCA_SUCCESS || gpu_addr == NULL) {
                    NVLOGE_NO(TAG, AERIAL_DOCA_API_EVENT, "doca_gpu_mem_alloc failed for RX %s",
                            doca_error_get_descr(result));
                    return -1;
                }
                nv_ipc_mempool_set_ext_bufs(priv_data->rxpools[pool_id], gpu_addr, cpu_addr);
            }
#endif

            if (priv_data->primary) {
                // DESC UL <-> DPU TX
                doca_mp_init_dpu(&priv_data->doca_info, priv_data->txpools[pool_id],
                        &priv_data->mmap_descs.ul_descs[pool_id],
                        &priv_data->dma_pools.tx_pools[pool_id]);
                doca_mp_init_dpu(&priv_data->doca_info, priv_data->rxpools[pool_id],
                        &priv_data->mmap_descs.dl_descs[pool_id],
                        &priv_data->dma_pools.rx_pools[pool_id]);
            } else {
                // Host TX <-> DESC DL
                doca_mp_init_host(&priv_data->doca_info, priv_data->txpools[pool_id],
                        &priv_data->mmap_descs.dl_descs[pool_id],
                        &priv_data->dma_pools.tx_pools[pool_id]);
                doca_mp_init_host(&priv_data->doca_info, priv_data->rxpools[pool_id],
                        &priv_data->mmap_descs.ul_descs[pool_id],
                        &priv_data->dma_pools.rx_pools[pool_id]);
            }
        }
    }

    priv_data->tx_job_base =  priv_data->dma_pools.tx_pools[NV_IPC_MEMPOOL_CPU_MSG].jobs;
    priv_data->rx_job_base =  priv_data->dma_pools.rx_pools[NV_IPC_MEMPOOL_CPU_MSG].jobs;

    NVLOGC(TAG, "%s: primary=%d cpu_core=%d cuda_device_id=%d", __func__, priv_data->primary, priv_data->cpu_core, priv_data->cuda_device_id);

    negotiate_mmap_descriptors(priv_data, 2);

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

    // Create a background thread
    if(pthread_create(&priv_data->thread_id, NULL, nvipc_poll_thread, priv_data) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: thread create failed", __func__);
        return -1;
    }

    return 0;
}

static int cuda_memcpy_to_host(nv_ipc_t* ipc, void* host, const void* device, size_t size)
{
    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }
    priv_data_t*      priv_data    = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    nv_ipc_mempool_t* cuda_mempool = priv_data->rxpools[NV_IPC_MEMPOOL_CUDA_DATA];
    if(cuda_mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: CUDA memory pool not exist", __func__);
        return -1;
    }
    else
    {
        return cuda_mempool->memcpy_to_host(cuda_mempool, host, device, size);
    }
}

static int cuda_memcpy_to_device(nv_ipc_t* ipc, void* device, const void* host, size_t size)
{
    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }
    priv_data_t*      priv_data    = get_private_data(ipc);
    IPC_DUMPING_CHECK(priv_data)

    nv_ipc_mempool_t* cuda_mempool = priv_data->txpools[NV_IPC_MEMPOOL_CUDA_DATA];
    if(cuda_mempool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: CUDA memory pool not exist", __func__);
        return -1;
    }
    else
    {
        return cuda_mempool->memcpy_to_device(cuda_mempool, device, host, size);
    }
}

nv_ipc_t* create_doca_nv_ipc_interface(const nv_ipc_config_t* cfg)
{
    NVLOGC(TAG, "%s: start ...", __func__);

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

    ipc->ipc_destroy = doca_ipc_close;

    ipc->tx_allocate = tx_mem_alloc;
    ipc->rx_release  = rx_mem_free;

    ipc->tx_release  = tx_mem_free;
    ipc->rx_allocate = mem_alloc_dummy;

    ipc->tx_send_msg = ipc_tx_send;
    ipc->rx_recv_msg = ipc_rx_recv;

    // Semaphore synchronization
    ipc->tx_tti_sem_post = doca_tx_tti_sem_post;
    ipc->rx_tti_sem_wait = doca_rx_tti_sem_wait;

    // Event FD synchronization
    ipc->get_fd    = doca_efd_get_fd;
    ipc->get_value = doca_efd_get_value;
    ipc->notify    = doca_efd_notify;

    ipc->cuda_memcpy_to_host   = cuda_memcpy_to_host;
    ipc->cuda_memcpy_to_device = cuda_memcpy_to_device;

    if((priv_data->ipc_debug = nv_ipc_debug_open(ipc, cfg)) == NULL)
    {
        free(ipc);
        return NULL;
    }

    if(doca_ipc_open(ipc, cfg) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed", __func__);
        doca_ipc_close(ipc);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: OK", __func__);
        return ipc;
    }
}
