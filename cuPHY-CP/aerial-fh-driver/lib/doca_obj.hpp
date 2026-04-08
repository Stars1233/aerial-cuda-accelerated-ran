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

#include "aerial-fh-driver/api.hpp"
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_mmap.h>
#include <doca_buf_array.h>
#include <doca_log.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_pe.h>

#ifndef AERIAL_FH_DOCAOBJ_HPP__
#define AERIAL_FH_DOCAOBJ_HPP__

typedef struct doca_tx_buf {
	struct doca_gpu *gpu_dev;		/* GPU device */
	struct doca_dev *ddev;			/* Network DOCA device */
	uint32_t num_packets;			/* Number of packets in the buffer */
	enum doca_gpu_mem_type mtype;		/* Memory type */
	uint32_t max_pkt_sz;			/* Max size of each packet in the buffer */
	uint32_t pkt_nbytes;			/* Effective bytes in each packet */
	uint8_t *gpu_pkt_addr;			/* GPU memory address of the buffer */
	uint8_t *cpu_pkt_addr;			/* CPU memory address of the buffer */
	struct doca_mmap *mmap;			/* DOCA mmap around GPU memory buffer for the DOCA device */
	struct doca_buf_arr *buf_arr;		/* DOCA buffer array object around GPU memory buffer */
	struct doca_gpu_buf_arr *buf_arr_gpu;	/* DOCA buffer array GPU handle */
	int dmabuf_fd;				/* DMABuf file descriptor */
	uint32_t pkt_buff_mkey;			/* mkey value of the buffer */

	/* CPU comms defs */
	struct doca_mmap *cpu_comms_mmap;
	uint8_t *cpu_comms_cpu_pkt_addr;
	uint8_t *cpu_comms_gpu_pkt_addr; // Not used but required by DOCA
	struct doca_buf_arr *cpu_comms_buf_arr;
	struct doca_gpu_buf_arr *cpu_comms_buf_arr_gpu;	/* DOCA buffer array GPU handle */
	uint32_t cpu_comms_pkt_buff_mkey;			/* mkey value of the CPU comms buffer */
} doca_tx_buf_t;

typedef struct doca_tx_items
{
	struct doca_gpu *gpu_dev;		/* GPU device */
	struct doca_dev *ddev;			/* Network DOCA device */
	struct doca_ctx *eth_txq_ctx;		/* DOCA Ethernet send queue context */
	struct doca_eth_txq *eth_txq_cpu;	/* DOCA Ethernet send queue CPU handler */
	struct doca_gpu_eth_txq *eth_txq_gpu;	/* DOCA Ethernet send queue GPU handler */
	struct doca_pe *eth_txq_pe;		/* DOCA Ethernet progress engine */
	char dump_cqe_file[128];		/* File to dump CQE info */
	int txq_id;				/* Txq Aerial ID */
    uint8_t     frame_id;  /*ORAN frame ID [0~255]*/
    uint16_t    subframe_id; /*ORAN sub-frame ID [0~9]*/
    uint16_t    slot_id; /*ORAN slot ID [0~1]*/
    int cell_idx;
    void* gCommsHdl;
} doca_tx_items_t;

doca_error_t doca_init_logger(void);
doca_error_t doca_create_rx_queue(struct doca_rx_items *item, struct doca_gpu *gpu, struct doca_dev *ddev, int ndescr, enum doca_gpu_mem_type mtype, int max_pkt_size, int num_pkts, uint8_t enable_gpu_comm_via_cpu);
doca_error_t doca_destroy_rx_queue(struct doca_gpu *gpu, struct doca_rx_items *item);
doca_error_t doca_create_semaphore(struct doca_rx_items *item, struct doca_gpu *gpu, int nitems, enum doca_gpu_mem_type sem_mtype, enum doca_gpu_mem_type custom_mtype, int custom_nbytes);
doca_error_t doca_destroy_semaphore(struct doca_gpu_semaphore *sem_cpu);
doca_error_t doca_create_tx_queue(struct doca_tx_items *item, struct doca_gpu *gpu_dev, struct doca_dev *ddev, int ndescr, int txq_id, enum doca_gpu_mem_type mtype);
doca_error_t doca_destroy_tx_queue(struct doca_tx_items *item);
doca_error_t doca_create_tx_buf(struct doca_tx_buf *buf, struct doca_gpu *gpu_dev, struct doca_dev *ddev, enum doca_gpu_mem_type mtype, uint32_t num_packets, uint32_t max_pkt_sz,uint8_t enable_gpu_comm_via_cpu);
void error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data);
void debug_send_packet_cqe_cb(struct doca_eth_txq_gpu_event_notify_send_packet *event_notify, union doca_data event_user_data);

#endif
