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

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "doca_utils.hpp"
#include "order_entity.hpp"
#include "aerial-fh-driver/oran.hpp"

#pragma nv_diag_suppress 177 // warning #177-D: variable "wqe_id" was declared but never referenced
#pragma nv_diag_suppress 550 // #550-D: variable "wqe_id_last" was set but never used
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#pragma nv_diag_default 177
#pragma nv_diag_default 550

#ifndef ACCESS_ONCE
    #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

namespace fh_gen
{
// doca_error_t kernel_receive_persistent(cudaStream_t stream)
// {
//     cudaError_t result = cudaSuccess;

//     return DOCA_SUCCESS;
// }

#ifdef __cplusplus
extern "C" {
#endif
__global__ void kernel_write(uint32_t* addr, uint32_t value)
{
    ACCESS_ONCE(*addr) = value;
}

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}

void launch_kernel_write(cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    cudaError_t result = cudaSuccess;

    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }

    kernel_write<<<1, 1, 0, stream>>>(addr, value);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void _kernel_receive_persistent_(
    /* DOCA objects */
	struct doca_gpu_eth_rxq **rxq_info_gpu,
    uint32_t *exit_flag
)
{
    const uint64_t timeout_ns = 100000;
    const uint32_t  max_rx_pkts = 512;
    //int sem_idx_rx = 0;
	__shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    unsigned long long int rx_pkt_total_thread = 0;
    __shared__ unsigned long long int rx_pkt_total;
    int cell_idx=blockIdx.x;
    if (threadIdx.x == 0)
    {
        DOCA_GPUNETIO_VOLATILE(rx_pkt_total) = 0;
        DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
    }
    doca_error_t ret;
    struct doca_gpu_eth_rxq* rxq_info_gpu_cell=*(rxq_info_gpu+cell_idx);
    while (DOCA_GPUNETIO_VOLATILE(*(exit_flag)) == 0) {
        if (threadIdx.x == 0) {
            ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(rxq_info_gpu_cell,max_rx_pkts,timeout_ns,&rx_buf_idx,&rx_pkt_num,NULL);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                printf("Exit from rx kernel block %d threadIdx %d ret %d\n",
                            blockIdx.x, threadIdx.x, ret);
            }
         
            if(DOCA_GPUNETIO_VOLATILE(rx_pkt_num) != 0)
                rx_pkt_total_thread += rx_pkt_num;
        }
        __threadfence();
        __syncthreads();
    }

    __syncthreads();
    atomicAdd(&rx_pkt_total, rx_pkt_total_thread);
    if(threadIdx.x == 0)
        printf("Receive kernel exit report. Block %d, received %lu pkts\n", blockIdx.x, rx_pkt_total);
}



__global__ void _kernel_receive_slot_(
    /* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
    // uint32_t **last_sem_idx_rx_h,
    uint32_t *exit_flag,
    int frame_id,
    int subframe_id,
    int slot_id,
    uint64_t slot_t0,
    uint64_t slot_duration_ns,
    uint32_t** order_kernel_exit_cond_d,
    int* prb_x_slot,
    uint64_t* ta4_min_ns,
    uint64_t* ta4_max_ns,
    uint32_t **early_rx_packets,
    uint32_t **on_time_rx_packets,
    uint32_t **late_rx_packets,
    uint32_t **next_slot_early_rx_packets,
    uint32_t **next_slot_on_time_rx_packets,
    uint32_t **next_slot_late_rx_packets,
    uint32_t **next_slot_num_prb,
    uint64_t **rx_packets_ts,
    uint32_t **rx_packets_count,
    uint64_t **rx_packets_ts_earliest,
    uint64_t **rx_packets_ts_latest,
    uint64_t **next_slot_rx_packets_ts,
    uint32_t **next_slot_rx_packets_count
)
{
    const uint64_t timeout_ns = 100000;
    const uint32_t  max_rx_pkts = 512;
    unsigned long long kernel_start = __globaltimer();
    unsigned long long current_time = 0;
    //int sem_idx_rx = 0;
    unsigned long long int rx_pkt_total_thread = 0;
    int cell_idx=blockIdx.x;
    uint32_t* exit_cond_d_cell=*(order_kernel_exit_cond_d+cell_idx);
    doca_error_t ret;
    struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
    struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
    uint16_t ecpri_payload_length;
    uint8_t* section_buf;
    uint16_t num_prb = 0;
    uint16_t compressed_prb_size = 28;
    uint16_t current_length;
    int prb_x_slot_x_cell = prb_x_slot[cell_idx];
    uint64_t ta4_min_ns_cell = ta4_min_ns[cell_idx];
    uint64_t ta4_max_ns_cell = ta4_max_ns[cell_idx];

    int sem_idx_rx = 0;//(int)(*(*(last_sem_idx_rx_h+cell_idx)));
    uint8_t *pkt_thread = NULL;
    uint8_t pkt_frame_id = 0;
    uint8_t pkt_subframe_id = 0;
    uint8_t pkt_slot_id = 0;
    uint8_t pkt_sym_id = 0;
    uint64_t packet_early_thres = 0;
    uint64_t packet_late_thres  = 0;
    uint64_t rx_timestamp;
    uint32_t* early_rx_packets_cell = *(early_rx_packets+cell_idx);
    uint32_t* on_time_rx_packets_cell = *(on_time_rx_packets+cell_idx);
    uint32_t* late_rx_packets_cell = *(late_rx_packets+cell_idx);
    uint32_t* next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets+cell_idx);
    uint32_t* next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets+cell_idx);
    uint32_t* next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets+cell_idx);
    uint32_t* next_slot_num_prb_cell = *(next_slot_num_prb+cell_idx);
    uint32_t prev_rx_prbs;
    uint64_t* next_slot_rx_packets_ts_cell=*(next_slot_rx_packets_ts+cell_idx);
    uint32_t* next_slot_rx_packets_count_cell=*(next_slot_rx_packets_count+cell_idx);

    uint64_t* rx_packets_ts_cell=*(rx_packets_ts+cell_idx);
    uint32_t* rx_packets_count_cell=*(rx_packets_count+cell_idx);
    uint64_t* rx_packets_ts_earliest_cell = *(rx_packets_ts_earliest+cell_idx);
    uint64_t* rx_packets_ts_latest_cell = *(rx_packets_ts_latest+cell_idx);

    int max_pkt_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_ALL_SYMBOLS;
    int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0;
    __shared__ uint32_t num_prb_rx;
    __shared__ uint32_t next_slot_num_prb_rx;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;
    __shared__ unsigned long long int rx_pkt_total;
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_ALL_SYMBOLS];
    __shared__ uint32_t rx_packets_count_sh[ORAN_ALL_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_ALL_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_ALL_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_ALL_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_ALL_SYMBOLS];
    if (threadIdx.x == 0)
    {
        DOCA_GPUNETIO_VOLATILE(rx_pkt_total) = 0;
        DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
        DOCA_GPUNETIO_VOLATILE(num_prb_rx) = 0;
        prev_rx_prbs = DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_cell);
        DOCA_GPUNETIO_VOLATILE(next_slot_num_prb_rx) = 0;

        early_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;
    }

    if(threadIdx.x < ORAN_ALL_SYMBOLS)
    {
        rx_packets_count_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x]);
        rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
        rx_packets_ts_latest_sh[threadIdx.x]=0;
        next_slot_rx_packets_count_sh[threadIdx.x]=0;
    }
    __syncthreads();
    for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
    {
        uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
        rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
        __threadfence_block();
        if(rx_packets_ts_sh[pkt_idx]!=0)
            atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
        atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
        __threadfence_block();
        next_slot_rx_packets_ts_sh[pkt_idx]=0;
    }
    __syncthreads();
    if(threadIdx.x < ORAN_ALL_SYMBOLS)
    {
        if(rx_packets_ts_earliest_sh[threadIdx.x]==0)
            rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
    }
    __syncthreads();
    // if(threadIdx.x == 0)
    // {
    //     printf("Cell %d Receiving for F%uS%uS%u t0 %lu exit cond %d %p\n", cell_idx, frame_id, subframe_id, slot_id, slot_t0, DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell), exit_cond_d_cell);
    // }
    while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING)
    {
        if (threadIdx.x == 0)
        {
            DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
        }

        __syncthreads();
        if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
        {
            break;
        }

        if (threadIdx.x == 0)
        {
            current_time = __globaltimer();
            if ((current_time - kernel_start) > 3000000)
            {
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
                printf("Exit from rx kernel block %d threadIdx %d rx timeout\n", blockIdx.x, threadIdx.x);
            }
        }    
        ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, nullptr);
        /* If any thread returns receive error, the whole execution stops */
        if (ret != DOCA_SUCCESS)
        {
            printf("Exit from rx kernel block %d threadIdx %d ret %d\n", blockIdx.x, threadIdx.x, ret);
            DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
        }

        if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0)
        {
            if (threadIdx.x == 0)
            {
                rx_pkt_total_thread += rx_pkt_num;
                doca_gpu_dev_semaphore_set_packet_info(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
            }

            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<rx_pkt_num;pkt_idx+=blockDim.x)
            {
                pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
                section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                ecpri_payload_length = min(oran_umsg_get_ecpri_payload(pkt_thread),ORAN_ECPRI_MAX_PAYLOAD_LEN);
                // Network endianness to CPU endianness
                // ecpri_payload_length = (0x00ff & (ecpri_payload_length >> 8)) | (0xff00 & (ecpri_payload_length << 8));
                pkt_frame_id    = oran_umsg_get_frame_id(pkt_thread);
                pkt_subframe_id = oran_umsg_get_subframe_id(pkt_thread);
                pkt_slot_id     = oran_umsg_get_slot_id(pkt_thread);
                pkt_sym_id      = oran_umsg_get_symbol_id(pkt_thread);
                // Validate symbol ID to prevent buffer overrun
                if (unlikely(pkt_sym_id >= ORAN_ALL_SYMBOLS))
                {
                    printf("pkt_sym_id %d out of bounds, max %d\n", pkt_sym_id, ORAN_ALL_SYMBOLS - 1);
                    continue;
                }
                rx_timestamp=doca_gpu_dev_eth_rxq_get_pkt_ts(doca_rxq_cell, rx_buf_idx + pkt_idx);
                if(pkt_frame_id == frame_id && pkt_subframe_id == subframe_id && pkt_slot_id == slot_id)
                {
                    uint16_t section_buf_size = 0;
                    current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    uint16_t num_sections = 0;
                    uint16_t pkt_num_prb = 0;
                    while(current_length < ecpri_payload_length)
                    {
                        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                        if(num_prb == 0)
                        {
                            num_prb = ORAN_MAX_PRB_X_SLOT;
                        }
                        atomicAdd(&num_prb_rx,num_prb);
                        pkt_num_prb += num_prb;
                        __threadfence_block();
                        section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        current_length += section_buf_size;
                        section_buf += section_buf_size;
                        ++num_sections;
                    }
                    packet_early_thres = slot_t0 + ta4_min_ns_cell + (slot_duration_ns * pkt_sym_id / ORAN_ALL_SYMBOLS);
                    packet_late_thres  = slot_t0 + ta4_max_ns_cell + (slot_duration_ns * pkt_sym_id / ORAN_ALL_SYMBOLS);
                    if (rx_timestamp < packet_early_thres)
                    {
                        atomicAdd(&early_rx_packets_count_sh, 1);
                    }
                    else if (rx_timestamp > packet_late_thres)
                    {
                        atomicAdd(&late_rx_packets_count_sh, 1);
                    }
                    else
                    {
                        atomicAdd(&on_time_rx_packets_count_sh, 1);
                    }
                    __threadfence_block();
                    {
                        rx_packets_ts_idx = atomicAdd(&rx_packets_count_sh[pkt_sym_id], 1);
                        __threadfence_block();
                        rx_packets_ts_idx+=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*pkt_sym_id;
                        rx_packets_ts_sh[rx_packets_ts_idx]=rx_timestamp;
                        atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[pkt_sym_id],(unsigned long long) rx_timestamp);
                        atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[pkt_sym_id],(unsigned long long) rx_timestamp);
                        __threadfence_block();
                    }
                }
                else if(pkt_frame_id * ORAN_MAX_SLOT_X_SUBFRAME_ID + pkt_subframe_id * ORAN_MAX_SLOT_ID + pkt_slot_id == 
                    ((frame_id * ORAN_MAX_SLOT_X_SUBFRAME_ID + subframe_id * ORAN_MAX_SLOT_ID + slot_id) + 1) % (ORAN_MAX_FRAME_ID * ORAN_MAX_SLOT_X_SUBFRAME_ID * ORAN_MAX_SLOT_ID))
                {
                    uint16_t section_buf_size = 0;
                    current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    uint16_t num_sections = 0;
                    uint16_t pkt_num_prb = 0;
                    while(current_length < ecpri_payload_length)
                    {
                        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                        if(num_prb==0)
                            num_prb=ORAN_MAX_PRB_X_SLOT;
                        atomicAdd(&next_slot_num_prb_rx,num_prb);
                        pkt_num_prb += num_prb;
                        __threadfence_block();
                        section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        current_length += section_buf_size;
                        section_buf += section_buf_size;
                        ++num_sections;
                    }
                    packet_early_thres = slot_t0 + slot_duration_ns + ta4_min_ns_cell + (slot_duration_ns * pkt_sym_id / ORAN_ALL_SYMBOLS);
                    packet_late_thres  = slot_t0 + slot_duration_ns + ta4_max_ns_cell + (slot_duration_ns * pkt_sym_id / ORAN_ALL_SYMBOLS);
                    if (rx_timestamp < packet_early_thres)
                    {
                        atomicAdd(&next_slot_early_rx_packets_count_sh, 1);
                    }
                    else if (rx_timestamp > packet_late_thres)
                    {
                        atomicAdd(&next_slot_late_rx_packets_count_sh, 1);
                    }
                    else
                    {
                        atomicAdd(&next_slot_on_time_rx_packets_count_sh, 1);
                    }
                    __threadfence_block();
                    {
                        next_slot_rx_packets_ts_idx = atomicAdd(&next_slot_rx_packets_count_sh[pkt_sym_id], 1);
                        __threadfence_block();
                        next_slot_rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * pkt_sym_id;
                        next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                    }
                }
                else
                {
                    printf("Packet received more than +1 slot, order entity F%uS%uS%u %d pkt F%uS%uS%u %d\n", 
                        frame_id, subframe_id, slot_id, (frame_id * ORAN_MAX_SLOT_X_SUBFRAME_ID + subframe_id * ORAN_MAX_SLOT_ID + slot_id) + 1,
                        pkt_frame_id, pkt_subframe_id, pkt_slot_id, pkt_frame_id * ORAN_MAX_SLOT_X_SUBFRAME_ID + pkt_subframe_id * ORAN_MAX_SLOT_ID + pkt_slot_id);
                }
            }
        }
        __syncthreads();
        if(threadIdx.x == 0)
        {
            if(num_prb_rx + prev_rx_prbs >= prb_x_slot_x_cell)
            {
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
                // printf("F%uS%uS%u Cell %d Received all %u+%u = %u /%d PRBs\n", frame_id, subframe_id, slot_id, cell_idx, num_prb_rx, prev_rx_prbs, num_prb_rx + prev_rx_prbs, prb_x_slot_x_cell);
                break;
            }
        }
        __threadfence_block();
    }

    if(threadIdx.x<ORAN_ALL_SYMBOLS)
    {
        DOCA_GPUNETIO_VOLATILE(rx_packets_count_cell[threadIdx.x])=rx_packets_count_sh[threadIdx.x];
        DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x])=next_slot_rx_packets_count_sh[threadIdx.x];
        DOCA_GPUNETIO_VOLATILE(rx_packets_ts_earliest_cell[threadIdx.x])=rx_packets_ts_earliest_sh[threadIdx.x];
        DOCA_GPUNETIO_VOLATILE(rx_packets_ts_latest_cell[threadIdx.x])=rx_packets_ts_latest_sh[threadIdx.x];
    }
    __syncthreads();
    for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
    {
        DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
        DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
    }
    __syncthreads();

    atomicAdd(&rx_pkt_total, rx_pkt_total_thread);
    if(threadIdx.x == 0)
    {
        DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell) = next_slot_early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell) = next_slot_on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell) = next_slot_late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_cell) = next_slot_num_prb_rx;
        // printf("F%uS%uS%u Cell %d Receive kernel exit report. Received %lu pkts %u+%u=%u/%d PRBs expected early %u ontime %u late %u next slot early %u ontime %u late %u, next slot numPrb %u\n", 
        //     frame_id, subframe_id, slot_id, cell_idx, rx_pkt_total, num_prb_rx, prev_rx_prbs, num_prb_rx + prev_rx_prbs, prb_x_slot[cell_idx], early_rx_packets_count_sh, on_time_rx_packets_count_sh, late_rx_packets_count_sh,
        //     next_slot_early_rx_packets_count_sh, next_slot_on_time_rx_packets_count_sh, next_slot_late_rx_packets_count_sh, next_slot_num_prb_rx);
    }
}

doca_error_t kernel_receive_persistent(cudaStream_t stream, int num_cells,
    /* DOCA objects */
    struct doca_gpu_eth_rxq **rxq_info_gpu,
    uint32_t *exit_flag)
{
    cudaError_t result = cudaSuccess;
    int cuda_blocks = num_cells;

    if(num_cells == 0)
    {
        NVLOGC_FMT(TAG, "Zero peers defined, not launching any UL RX kernel");
        return DOCA_SUCCESS;
    }
    _kernel_receive_persistent_<<<cuda_blocks, 32, 0, stream>>>(rxq_info_gpu, exit_flag);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t kernel_receive_slot(cudaStream_t stream, orderKernelConfigParams_t* params)
{
    cudaError_t result = cudaSuccess;
    int cuda_blocks = params->num_cells;
    _kernel_receive_slot_<<<cuda_blocks, 320, 0, stream>>>(
        params->rxq_info_gpu,
        params->sem_gpu,
        params->exit_flag_d,
        params->frame_id,
        params->subframe_id,
        params->slot_id,
        params->slot_t0,
        params->slot_duration,
        params->order_kernel_exit_cond_d,
        params->prb_x_slot,
        params->ta4_min_ns,
        params->ta4_max_ns,
        params->early_rx_packets,
        params->on_time_rx_packets,
        params->late_rx_packets,
        params->next_slot_early_rx_packets,
        params->next_slot_on_time_rx_packets,
        params->next_slot_late_rx_packets,
        params->next_slot_num_prb,
        params->rx_packets_ts,
        params->rx_packets_count,
        params->rx_packets_ts_earliest,
        params->rx_packets_ts_latest,
        params->next_slot_rx_packets_ts,
        params->next_slot_rx_packets_count
    );

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}


#ifdef __cplusplus
} /* extern C */
#endif


}
