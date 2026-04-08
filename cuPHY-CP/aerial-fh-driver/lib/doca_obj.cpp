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

#include "doca_obj.hpp"
#include "utils.hpp"
#include "gpu_comm.hpp"
#include "aerial-fh-driver/api.hpp"
#include "app_config.hpp"


#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)

#define TAG "FH.DOCA"

/*
 * DOCA PE callback to be invoked if any Eth Txq get an error
 * sending packets.
 *
 * @event_error [in]: DOCA PE event error handler
 * @event_user_data [in]: custom user data set at registration time
 */
void
error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
	uint16_t packet_index;
	doca_tx_items_t *txq = (doca_tx_items_t *)event_user_data.ptr;

	doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
	NVLOGI_FMT(TAG,"Error in send queue {}, packet {} for F{}S{}S{}. Gracefully killing the app",
			txq->txq_id, packet_index,txq->frame_id,txq->subframe_id,txq->slot_id);
}

/*
 * DOCA PE callback to be invoked on Eth Txq to get the debug info
 * when sending packets
 *
 * @event_notify [in]: DOCA PE event debug handler
 * @event_user_data [in]: custom user data set at registration time
 */
void
debug_send_packet_cqe_cb(struct doca_eth_txq_gpu_event_notify_send_packet *event_notify, union doca_data event_user_data)
{
	uint16_t packet_index;
	uint64_t packet_timestamp;
	uint64_t ts_diff = 0;
    uint64_t enqueue_time=0;
	doca_tx_items_t *txq = (doca_tx_items_t *)event_user_data.ptr;
    aerial_fh::GpuComm* gpuCommsHandle = static_cast<aerial_fh::GpuComm*>(txq->gCommsHdl);
    aerial_fh::PacketStartDebugInfo* pkt_dbg_info;

	doca_eth_txq_gpu_event_notify_send_packet_get_position(event_notify, &packet_index);
	doca_eth_txq_gpu_event_notify_send_packet_get_timestamp(event_notify, &packet_timestamp);

    pkt_dbg_info= gpuCommsHandle->getPkt_start_debug(txq->cell_idx,packet_index);
    enqueue_time = gpuCommsHandle->getDlSlotTriggerTs(pkt_dbg_info->frame_id*20+pkt_dbg_info->subframe_id*2+pkt_dbg_info->slot_id);

    NVLOGI_FMT(TAG,"Txq debug event: Frame:{} Subframe:{} Slot:{} Queue {} packet {} Cell index {} sent at {} time scheduled at time {} Enqueue time {} for Frame:{} Subframe:{} Slot:{} symbol:{} Packet size {}",
               txq->frame_id,txq->subframe_id,txq->slot_id,txq->txq_id, packet_index, txq->cell_idx,packet_timestamp,pkt_dbg_info->ptp_time,enqueue_time,pkt_dbg_info->frame_id,pkt_dbg_info->subframe_id,pkt_dbg_info->slot_id,pkt_dbg_info->sym,pkt_dbg_info->packet_size);

#if DUMP_WQE_INFO
	//This code will dump the WQE and CQE buffer for debugging. Please leave in here
	FILE *fp = fopen(txq->dump_cqe_file, "a");
	fprintf(fp, "CQE: %d Ts: %lx\n", packet_index, packet_timestamp);
	fclose(fp);
#endif
}

doca_error_t
doca_init_logger(void)
{
	doca_error_t result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_BAD_STATE;

	/* Enable only to get more library debug logs */
	struct doca_log_backend *stdout_logger = NULL;

	/* Create a logger backend that prints to the standard output */
	result = doca_log_backend_create_with_file_sdk(stdout, &stdout_logger);
	if (result != DOCA_SUCCESS)
		return DOCA_ERROR_BAD_STATE;

	// set DEBUG for more logs
	doca_log_backend_set_sdk_level(stdout_logger, DOCA_LOG_LEVEL_WARNING);

	return result;
}

doca_error_t
doca_create_rx_queue(struct doca_rx_items *item, struct doca_gpu *gpu_dev, struct doca_dev *ddev, int ndescr, enum doca_gpu_mem_type mtype, int max_pkt_size, int num_pkts, uint8_t enable_gpu_comm_via_cpu)
{
	doca_error_t result;
	uint32_t cyclic_buffer_size = 0;

	if (item == NULL || gpu_dev == NULL || ddev == NULL || ndescr == 0 || num_pkts == 0) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Can't create UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	NVLOGD_FMT(TAG, "Creating DOCA Eth Rxq with {} pkts {}B size {} descr\n", num_pkts, max_pkt_size, ndescr);

	item->gpu_dev = gpu_dev;
	item->ddev = ddev;

	result = doca_eth_rxq_create(item->ddev, num_pkts, max_pkt_size, &(item->eth_rxq_cpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_create: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_set_type(item->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_set_type: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}
	result = doca_eth_rxq_gpu_set_rq_mem_type(item->eth_rxq_cpu, mtype);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_gpu_set_rq_mem_type: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}
	result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, max_pkt_size, num_pkts, 0,  0,  0, &cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to get eth_rxq cyclic buffer size: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_mmap_create(&item->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to create mmap: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_mmap_add_dev(item->pkt_buff_mmap, item->ddev);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to add dev to mmap: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	if(mtype==DOCA_GPU_MEM_TYPE_CPU_GPU){
		result = doca_gpu_mem_alloc(item->gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_CPU_GPU, &item->gpu_pkt_addr, &item->cpu_pkt_addr);
	}
	else
	{
		result = doca_gpu_mem_alloc(item->gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_GPU, &item->gpu_pkt_addr,NULL);
	}
	if (result != DOCA_SUCCESS || item->gpu_pkt_addr == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to allocate gpu memory {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}
	memfoot_add_gpu_size(MF_TAG_FH_DOCA_RX, cyclic_buffer_size);

	if (!enable_gpu_comm_via_cpu) {
		/* Map GPU memory buffer used to receive packets with DMABuf */
		result = doca_gpu_dmabuf_fd(item->gpu_dev, item->gpu_pkt_addr, cyclic_buffer_size, &(item->dmabuf_fd));
		if (result != DOCA_SUCCESS) {
			NVLOGI_FMT(TAG, "Mapping receive queue buffer ({} size {}B) with nvidia-peermem mode",
					(void*)item->gpu_pkt_addr, cyclic_buffer_size);

			/* If failed, use nvidia-peermem legacy method */
			result = doca_mmap_set_memrange(item->pkt_buff_mmap, item->gpu_pkt_addr, cyclic_buffer_size);
			if (result != DOCA_SUCCESS) {
				NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set memrange for mmap {}", doca_error_get_descr(result));
				return result;
			}
		} else {
			NVLOGI_FMT(TAG, "Mapping receive queue buffer ({} size {}B dmabuf fd {}) with dmabuf mode",
				(void*)item->gpu_pkt_addr, cyclic_buffer_size, item->dmabuf_fd);

			result = doca_mmap_set_dmabuf_memrange(item->pkt_buff_mmap, item->dmabuf_fd, item->gpu_pkt_addr, 0, cyclic_buffer_size);
			if (result != DOCA_SUCCESS) {
				NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set dmabuf memrange for mmap {}", doca_error_get_descr(result));
				return result;
			}
		}		
	}
	else {
		result = doca_mmap_set_memrange(item->pkt_buff_mmap, item->cpu_pkt_addr, cyclic_buffer_size);
		if (result != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set memrange for mmap {}", doca_error_get_descr(result));
			return result;
		}	
	}

	result = doca_mmap_set_permissions(item->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set permissions for mmap {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_mmap_start(item->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to start mmap {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_set_pkt_buf(item->eth_rxq_cpu, item->pkt_buff_mmap, 0, cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set cyclic buffer  {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	item->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(item->eth_rxq_cpu);
	if (item->eth_rxq_ctx == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_as_doca_ctx: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_set_datapath_on_gpu(item->eth_rxq_ctx, item->gpu_dev);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_set_datapath_on_gpu: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_start(item->eth_rxq_ctx);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_start: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_get_gpu_handle(item->eth_rxq_cpu, &(item->eth_rxq_gpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_get_gpu_handle: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	//Get the HW queue index from DOCA here
	result = doca_eth_rxq_get_hw_queue_num(item->eth_rxq_cpu, &(item->hw_queue_idx));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_get_hw_queue_num: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}	

	return DOCA_SUCCESS;
}

/**
 * Free device memory for aerial_fh_gpu_semaphore_gpu structure
 *
 * @param[in] sem_gpu_aerial_fh Device pointer to aerial_fh_gpu_semaphore_gpu structure to free
 *
 * @return DOCA_SUCCESS on success, DOCA_ERROR_** otherwise
 */
static doca_error_t
free_aerial_fh_semaphore_gpu(struct doca_gpu *gpu_dev, struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh)
{
	doca_error_t result;

	if (sem_gpu_aerial_fh == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid parameters for aerial_fh semaphore free");
		return DOCA_ERROR_INVALID_VALUE;
	}

	result = doca_gpu_mem_free(gpu_dev, sem_gpu_aerial_fh);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to free aerial_fh_gpu_semaphore_gpu: {}", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/**
 * Allocate device memory for aerial_fh_gpu_semaphore_gpu structure
 *
 * @param[out] sem_gpu_aerial_fh Device pointer to allocated aerial_fh_gpu_semaphore_gpu structure
 * @param[in] nitems Number of items for packet info array
 *
 * @return DOCA_SUCCESS on success, DOCA_ERROR_** otherwise
 */
static doca_error_t
allocate_aerial_fh_semaphore_gpu(struct doca_gpu *gpu_dev, struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh, const int nitems)
{
	doca_error_t result;

	if (sem_gpu_aerial_fh == NULL || nitems <= 0) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid parameters for aerial_fh semaphore allocation");
		return DOCA_ERROR_INVALID_VALUE;
	}

	result = doca_gpu_mem_alloc(gpu_dev, sizeof(struct aerial_fh_gpu_semaphore_gpu), sysconf(_SC_PAGESIZE), DOCA_GPU_MEM_TYPE_GPU, (void **)sem_gpu_aerial_fh, nullptr);	
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to allocate device memory for aerial_fh_gpu_semaphore_gpu: {}", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_destroy_rx_queue(struct doca_rx_items *item)
{
	doca_error_t result;

	if (item == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Can't destroy UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	NVLOGI_FMT(TAG, "Destroying Rx queue");

	result = doca_ctx_stop(item->eth_rxq_ctx);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_stop: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_destroy(item->eth_rxq_cpu);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_rxq_destroy: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_mmap_destroy(item->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to destroy mmap: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_gpu_mem_free(item->gpu_dev, item->gpu_pkt_addr);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to free gpu memory: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	if(item->cpu_pkt_addr!=NULL){
		result = doca_gpu_mem_free(item->gpu_dev, item->cpu_pkt_addr);
		if (result != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to free cpu memory: {}", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	// Free device memory for aerial_fh GPU semaphore structure
	if (item->sem_gpu_aerial_fh != NULL) {
		result = free_aerial_fh_semaphore_gpu(item->gpu_dev, item->sem_gpu_aerial_fh);
		if (result != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to free aerial_fh_gpu_semaphore_gpu: {}", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
		item->sem_gpu_aerial_fh = NULL;
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_create_semaphore(struct doca_rx_items *item, struct doca_gpu *gpu_dev, int nitems, enum doca_gpu_mem_type sem_mtype, enum doca_gpu_mem_type custom_mtype, int custom_nbytes)
{
	doca_error_t result;

	result = doca_gpu_semaphore_create(gpu_dev, &(item->sem_cpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_create: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/*
	 * Semaphore memory reside on CPU visibile from GPU.
	 * CPU will poll in busy wait on this semaphore (multiple reads)
	 * while GPU access each item only once to update values.
	 */
	result = doca_gpu_semaphore_set_memory_type(item->sem_cpu, sem_mtype);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_set_memory_type: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	item->nitems = nitems;

	result = doca_gpu_semaphore_set_items_num(item->sem_cpu, nitems);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_set_items_num: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/*
	 * Semaphore memory reside on CPU visibile from GPU.
	 * The CPU reads packets info from this structure.
	 * The GPU access each item only once to update values.
	 */
	if (custom_nbytes > 0) {
		result = doca_gpu_semaphore_set_custom_info(item->sem_cpu, custom_nbytes, custom_mtype);
		if (result != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_set_custom_info: {}", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	result = doca_gpu_semaphore_start(item->sem_cpu);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_start: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_gpu_semaphore_get_gpu_handle(item->sem_cpu, &(item->sem_gpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_get_gpu_handle: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	// Allocate device memory for aerial_fh GPU semaphore structure
	result = allocate_aerial_fh_semaphore_gpu(gpu_dev,&(item->sem_gpu_aerial_fh), nitems);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to allocate aerial_fh_gpu_semaphore_gpu: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_destroy_semaphore(struct doca_gpu_semaphore *sem_cpu)
{
	doca_error_t result;

	if (sem_cpu == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Can't destroy semaphore, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	NVLOGI_FMT(TAG, "Destroying semaphore");

	result = doca_gpu_semaphore_stop(sem_cpu);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_start: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_gpu_semaphore_destroy(sem_cpu);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_gpu_semaphore_destroy: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_create_tx_buf(struct doca_tx_buf *buf, struct doca_gpu *gpu_dev, struct doca_dev *ddev, enum doca_gpu_mem_type mtype, uint32_t num_packets, uint32_t max_pkt_sz,uint8_t enable_gpu_comm_via_cpu)
{
	doca_error_t status;

	if (buf == NULL || gpu_dev == NULL || ddev == NULL || num_packets == 0 || max_pkt_sz == 0) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid input arguments. Args are: buf {:p}, gpu_dev {:p}, ddev {:p}, num_packets {}, max_pkt_sz {}",
                                                            (void*)buf, (void*)gpu_dev, (void*)ddev, num_packets, max_pkt_sz);
		return DOCA_ERROR_INVALID_VALUE;
	}

	buf->gpu_dev = gpu_dev;
	buf->ddev = ddev;
	buf->mtype = mtype;
	buf->num_packets = num_packets;
	buf->max_pkt_sz = max_pkt_sz;
        size_t size = (size_t)buf->num_packets * (size_t)buf->max_pkt_sz;

	status = doca_mmap_create(&(buf->mmap));
	if (status != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to create doca_buf: failed to create mmap");
		return status;
	}

	status = doca_mmap_add_dev(buf->mmap, buf->ddev);
	if (status != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to add dev to buf: doca mmap internal error");
		return status;
	}
	if(enable_gpu_comm_via_cpu==1){
		status = doca_gpu_mem_alloc(buf->gpu_dev, size , sysconf(_SC_PAGESIZE), buf->mtype, (void **)&(buf->gpu_pkt_addr), (void **)&(buf->cpu_pkt_addr));
	}
	else
	{
		status = doca_gpu_mem_alloc(buf->gpu_dev, size , sysconf(_SC_PAGESIZE), buf->mtype, (void **)&(buf->gpu_pkt_addr), NULL);
	}
	if ((status != DOCA_SUCCESS) || (buf->gpu_pkt_addr == NULL)) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to alloc txbuf: failed to allocate gpu memory");
		return status;
	}
	memfoot_add_gpu_size(MF_TAG_FH_DOCA_TX, size);


	if(!enable_gpu_comm_via_cpu)
	{
		/* Map GPU memory buffer used to receive packets with DMABuf */
		#if 0   // Set to 1, if you want to temporarily force nvidia-peermem path
		//status = doca_gpu_dmabuf_fd(buf->gpu_dev, buf->gpu_pkt_addr, size, &(buf->dmabuf_fd));
		if (true) {
	#else
		status = doca_gpu_dmabuf_fd(buf->gpu_dev, buf->gpu_pkt_addr, size, &(buf->dmabuf_fd));
			NVLOGC_FMT(TAG, "doca_gpu_dmabuf_fd returned {} (DOCA_SUCCESS is {})", +status, +DOCA_SUCCESS);
		if (status != DOCA_SUCCESS) {
	#endif
			NVLOGC_FMT(TAG, "Mapping transmit queue buffer ({:p} size {}B) with nvidia-peermem mode",
					(void*)buf->gpu_pkt_addr, size);

			/* If failed, use nvidia-peermem legacy method */
			status = doca_mmap_set_memrange(buf->mmap, buf->gpu_pkt_addr, size);
			if (status != DOCA_SUCCESS) {
				NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set memrange for mmap {}", doca_error_get_descr(status));
				return status;
			}
		} else {
			NVLOGC_FMT(TAG, "Mapping transmit queue buffer ({:p} size {}B dmabuf fd {}) with dmabuf mode",
				(void*)buf->gpu_pkt_addr, size, buf->dmabuf_fd);

			status = doca_mmap_set_dmabuf_memrange(buf->mmap, buf->dmabuf_fd, buf->gpu_pkt_addr, 0, size);
			if (status != DOCA_SUCCESS) {
				NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set dmabuf memrange for mmap {}", doca_error_get_descr(status));
				return status;
			}
		}

		status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "doca_mmap_set_permissions error");
			return status;
		}

		status = doca_mmap_start(buf->mmap);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "doca_mmap_start error");
			return status;
		}

		status = doca_mmap_get_mkey(buf->mmap, buf->ddev, &buf->pkt_buff_mkey);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to get mmap mkey %s", doca_error_get_descr(status));
			return status;
		}
		// N.B. mkey must be in network byte order
		buf->pkt_buff_mkey = htobe32(buf->pkt_buff_mkey);	
	}


	if(enable_gpu_comm_via_cpu)
	{
		status = doca_mmap_create(&(buf->cpu_comms_mmap));
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to create doca_buf: failed to create mmap");
			return status;
		}

		status = doca_mmap_add_dev(buf->cpu_comms_mmap, buf->ddev);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to add dev to buf: doca mmap internal error");
			return status;
		}
		status = doca_gpu_mem_alloc(buf->gpu_dev, size , sysconf(_SC_PAGESIZE), DOCA_GPU_MEM_TYPE_CPU_GPU, (void **)&(buf->cpu_comms_gpu_pkt_addr), (void **)&(buf->cpu_comms_cpu_pkt_addr));
		if ((status != DOCA_SUCCESS) || (buf->cpu_comms_gpu_pkt_addr == NULL)) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to alloc txbuf: failed to allocate cpu memory");
			return status;
		}
		memfoot_add_gpu_size(MF_TAG_FH_DOCA_TX, size);

		/* If failed, use nvidia-peermem legacy method */
		status = doca_mmap_set_memrange(buf->cpu_comms_mmap, buf->cpu_comms_cpu_pkt_addr, size);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to set memrange for mmap {}", doca_error_get_descr(status));
			return status;
		}

		status = doca_mmap_set_permissions(buf->cpu_comms_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "doca_mmap_set_permissions error");
			return status;
		}

		status = doca_mmap_start(buf->cpu_comms_mmap);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "doca_mmap_start error");
			return status;
		}

		status = doca_mmap_get_mkey(buf->cpu_comms_mmap, buf->ddev, &buf->cpu_comms_pkt_buff_mkey);
		if (status != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to get mmap mkey %s", doca_error_get_descr(status));
			return status;
		}
		// N.B. mkey must be in network byte order
		buf->cpu_comms_pkt_buff_mkey = htobe32(buf->cpu_comms_pkt_buff_mkey);		
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_create_tx_queue(struct doca_tx_items *item, struct doca_gpu *gpu_dev, struct doca_dev *ddev, int ndescr, int txq_id, enum doca_gpu_mem_type mtype)
{
	doca_error_t result;
	uint32_t cyclic_buffer_size = 0;
	union doca_data event_user_data = {0};

	if (item == NULL || gpu_dev == NULL || ddev == NULL || ndescr == 0) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Can't create UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	NVLOGI_FMT(TAG, "Creating DOCA Eth Txq with {} descr", ndescr);

	item->gpu_dev = gpu_dev;
	item->ddev = ddev;
	item->txq_id = txq_id;
	snprintf(item->dump_cqe_file, 128, "txq_cqe_%d.txt", item->txq_id);
	event_user_data.ptr = (void*)item;

	result = doca_eth_txq_create(item->ddev, ndescr, &(item->eth_txq_cpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_create: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_gpu_set_sq_mem_type(item->eth_txq_cpu, mtype);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_gpu_set_sq_mem_type: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	if (mtype == DOCA_GPU_MEM_TYPE_CPU_GPU) {
		result = doca_eth_txq_gpu_set_uar_on_cpu(item->eth_txq_cpu);
		if (result != DOCA_SUCCESS) {
			NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_gpu_set_uar_on_cpu: {}", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}	

	result = doca_eth_txq_set_wait_on_time_offload(item->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		// DOCA_LOG_ERR("Failed to set eth_txq num packets: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	item->eth_txq_ctx = doca_eth_txq_as_doca_ctx(item->eth_txq_cpu);
	if (item->eth_txq_ctx == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_as_doca_ctx: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_set_datapath_on_gpu(item->eth_txq_ctx, item->gpu_dev);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_set_datapath_on_gpu: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_pe_create(&item->eth_txq_pe);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to create pe queue: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_gpu_event_error_send_packet_register(item->eth_txq_cpu,
									error_send_packet_cb, event_user_data);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to set DOCA progress engine callback: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_gpu_event_notify_send_packet_register(item->eth_txq_cpu,
									debug_send_packet_cqe_cb, event_user_data);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to set DOCA progress engine callback: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_pe_connect_ctx(item->eth_txq_pe, item->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unable to set DOCA progress engine to DOCA Eth Txq: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_ctx_start(item->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_start: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_get_gpu_handle(item->eth_txq_cpu, &(item->eth_txq_gpu));
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_get_gpu_handle: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

doca_error_t
doca_destroy_tx_queue(struct doca_tx_items *item)
{
	doca_error_t result;

	if (item == NULL) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Can't destroy UDP queues, invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	NVLOGI_FMT(TAG, "Destroying Tx queue");

	result = doca_ctx_stop(item->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_ctx_stop: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_destroy(item->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed doca_eth_txq_destroy: {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_pe_destroy(item->eth_txq_pe);
	if (result != DOCA_SUCCESS) {
		NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Function doca_pe_destroy returned {}", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}
