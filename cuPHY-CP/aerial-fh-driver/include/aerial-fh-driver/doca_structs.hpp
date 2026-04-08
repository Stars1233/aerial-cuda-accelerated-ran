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

#ifndef AERIAL_FH_DRIVER_DOCA_STRUCTS__
#define AERIAL_FH_DRIVER_DOCA_STRUCTS__

#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>

#ifndef ACCESS_ONCE
    #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

static constexpr uint32_t RX_QUEUE_SYNC_LIST_ITEMS = 2048;
static constexpr uint32_t RX_QUEUE_SYNC_LIST_OBJ_ITEMS = 1024;
static constexpr uint32_t CK_ORDER_PKTS_BUFFERING = RX_QUEUE_SYNC_LIST_OBJ_ITEMS;
static constexpr uint32_t CK_ORDER_PKTS_BUFFERING_NS = 40000;
static constexpr uint32_t SYMBOL_DURATION_NS = 35714;
static constexpr uint32_t CK_ORDER_PKTS_BUFFERING_FLOW_NS = 10000;

static constexpr uint32_t SYNC_PACKET_STATUS_FREE = 0;
static constexpr uint32_t SYNC_PACKET_STATUS_READY = 1;
static constexpr uint32_t SYNC_PACKET_STATUS_DONE = 2;
static constexpr uint32_t SYNC_PACKET_STATUS_EXIT = 2;

static constexpr uint32_t ORDER_KERNEL_RECV_TIMEOUT_MS = 4;
static constexpr uint32_t ORDER_KERNEL_WAIT_TIMEOUT_MS = (ORDER_KERNEL_RECV_TIMEOUT_MS * 2);
static constexpr uint32_t ORDER_KERNEL_MB = 4;
static constexpr uint32_t ORDER_KERNEL_MAX_PKTS_BLOCK = (RX_QUEUE_SYNC_LIST_OBJ_ITEMS / ORDER_KERNEL_MB);
static constexpr uint32_t ORDER_KERNEL_TIMEOUT_ERROR_LOG_INTERVAL_NS = 1000000000; //1s
static constexpr uint32_t ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM = 100; //Setting to 100 as a safety net value considering we receive 64 SRS packets per symbol with 64TR config

static constexpr uint32_t MAX_PKTS_PER_SLOT_OK_TB = 400;
static constexpr uint32_t MAX_PKTS_SIZE_OK_TB = 1520; //Set to MTU size aligned to 16 byte for now (TODO:Read from NIC config)
static constexpr uint32_t MAX_UL_SLOTS_OK_TB = 16; //In a 80ms time frame window
static constexpr uint32_t MAX_TEST_SLOTS_OK_TB = 600000;

static constexpr uint32_t MAX_PKTS_PER_PCAP_BUFFER = MAX_PKTS_PER_SLOT_OK_TB * MAX_UL_SLOTS_OK_TB;

static constexpr uint32_t ORAN_FRAME_WRAP = 256;
static constexpr uint32_t ORAN_SUBFRAME_WRAP = 10;
static constexpr uint32_t ORAN_SLOT_WRAP = 2;
static constexpr uint32_t ORAN_SLOTS_PER_FRAME = ORAN_SUBFRAME_WRAP * ORAN_SLOT_WRAP;

static constexpr uint32_t ORAN_PUSCH_SYMBOLS_X_SLOT = 14;
static constexpr uint32_t ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL = 272;
static constexpr uint32_t ORAN_PRACH_B4_SYMBOLS_X_SLOT = 12;
static constexpr uint32_t ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL = 12;
static constexpr uint32_t ORAN_SRS_SYMBOLS_X_SLOT = 14;
static constexpr uint32_t ORAN_SRS_PRBS_X_PORT_X_SYMBOL = 272;
static constexpr uint32_t ORAN_PDSCH_PRBS_X_PORT_X_SYMBOL = 273;
static constexpr uint32_t ORAN_RE = 12;
static constexpr uint32_t ORAN_MAX_PRB = 273;
static constexpr uint32_t ORAN_MAX_SYMBOLS = 14;
static constexpr uint32_t ORAN_MAX_SRS_SYMBOLS = 6;
static constexpr uint32_t ORAN_PRACH_PRB = 24; //Max PRACH PRBs per 3GPP Spec 38.211 (Table 6.3.3.2-1)
static constexpr uint32_t ORAN_PRACH_REPETITIONS = 12;
static constexpr uint32_t UL_SRS_MAX_CELLS_PER_SLOT = 16;
static constexpr uint32_t MAX_RX_ANT_4T4R = 4;
static constexpr uint32_t MAX_RX_ANT_PUSCH_PUCCH_PRACH_64T64R = 16;
static constexpr uint32_t MAX_RX_ANT_SRS_64T64R = 64;
static constexpr uint32_t DEFAULT_PRB_STRIDE = 48;


static constexpr uint32_t UL_ST1_AP_BUF_SIZE = ORAN_MAX_PRB * ORAN_RE * ORAN_MAX_SYMBOLS * sizeof(uint32_t);
static constexpr uint32_t UL_ST2_AP_BUF_SIZE = ORAN_MAX_PRB * ORAN_RE * ORAN_MAX_SRS_SYMBOLS * sizeof(uint32_t);
static constexpr uint32_t UL_ST3_AP_BUF_SIZE = ORAN_PRACH_PRB * ORAN_RE * ORAN_PRACH_REPETITIONS * sizeof(uint32_t);

static constexpr uint32_t ORDER_KERNEL_IDLE = 0;
static constexpr uint32_t ORDER_KERNEL_START = 1;
static constexpr uint32_t ORDER_KERNEL_ABORT = 2;

static constexpr uint32_t NS_X_US = 1000ULL;
static constexpr uint32_t NS_X_MS = 1000000ULL;
static constexpr uint32_t NS_X_S = 1000000000ULL;

constexpr uint32_t MAX_SEM_ITEMS = 4096;     //!< Maximum semaphore items


enum order_kernel_exit_code {
    ORDER_KERNEL_RUNNING = 0,
    ORDER_KERNEL_EXIT_PRB = 1,
    ORDER_KERNEL_EXIT_ERROR_LEGACY = 2,
    ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT = 3,
    ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT = 4,
    ORDER_KERNEL_EXIT_ERROR1 = 5,
    ORDER_KERNEL_EXIT_ERROR2 = 6,
    ORDER_KERNEL_EXIT_ERROR3 = 7,
    ORDER_KERNEL_EXIT_ERROR4 = 8,
    ORDER_KERNEL_EXIT_ERROR5 = 9,
    ORDER_KERNEL_EXIT_ERROR6 = 10,
    ORDER_KERNEL_EXIT_ERROR7 = 11,
};

/**
 * Semaphore list of possible statuses.
 */
 enum aerial_fh_gpu_semaphore_status {
	/* Semaphore is free and can be (re)used. */
	AERIAL_FH_GPU_SEMAPHORE_STATUS_FREE = 0,
	AERIAL_FH_GPU_SEMAPHORE_STATUS_READY = 1,
	AERIAL_FH_GPU_SEMAPHORE_STATUS_DONE = 2,
	AERIAL_FH_GPU_SEMAPHORE_STATUS_HOLD = 3,
	AERIAL_FH_GPU_SEMAPHORE_STATUS_ERROR = 4,
	AERIAL_FH_GPU_SEMAPHORE_STATUS_EXIT = 5,
};

struct order_sem_info
{
    uint32_t pkts;
};

struct aerial_fh_gpu_semaphore_packet {
	enum aerial_fh_gpu_semaphore_status status; /**< status */
	uint32_t num_packets;		       /**< num_packets */
	uint64_t doca_buf_idx_start;	       /**< doca_buf_idx_start */
};

struct aerial_fh_gpu_semaphore_gpu {
	struct aerial_fh_gpu_semaphore_packet pkt_info_gpu[MAX_SEM_ITEMS]; /**< Packet info, GPU pointer */
	uint32_t num_items;				/**< Number of items in semaphore */
};

//need to expose this for the order kernel in cuphydriver
typedef struct doca_rx_items {
    struct doca_gpu *gpu_dev;               /* GPU device */
    struct doca_dev *ddev;                  /* Network DOCA device */
    struct doca_ctx *eth_rxq_ctx;           /* DOCA Ethernet receive queue context */
    struct doca_eth_rxq *eth_rxq_cpu;       /* DOCA Ethernet receive queue CPU handler */
    struct doca_gpu_eth_rxq *eth_rxq_gpu;   /* DOCA Ethernet receive queue GPU handler */
    struct doca_mmap *pkt_buff_mmap;        /* DOCA mmap to receive packet with DOCA Ethernet queue */
    void *gpu_pkt_addr;                     /* DOCA mmap GPU memory address */
    void *cpu_pkt_addr;			/* DOCA mmap CPU memory address */
    uint16_t dpdk_queue_idx;
    uint32_t hw_queue_idx;

    struct doca_gpu_semaphore *sem_cpu;     /* One semaphore per queue to report stats, CPU handler*/
    struct doca_gpu_semaphore_gpu *sem_gpu; /* One semaphore per queue to report stats, GPU handler*/
    struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh; /* One semaphore per queue to report stats, GPU handler*/
    int nitems;
    int dmabuf_fd;                          /* DMABuf file descriptor */
} doca_rx_items_t;

#endif