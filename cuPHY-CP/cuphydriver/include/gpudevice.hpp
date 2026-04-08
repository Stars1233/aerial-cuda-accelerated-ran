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

#ifndef GPUDEVICE_H
#define GPUDEVICE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <vector>
#include <array>
#include <stdio.h>
#include <memory>
#include <cstring>
#include "locks.hpp"
#include <gdrapi.h>
#include "nvlog.hpp"
#include "memfoot.hpp"
#include "cuphydriver_api.hpp"

#if USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors_phydrv[]   = {0xffff0000, 0xff00ffff, 0xffff00ff, 0xffffff00, 0xff00ff00, 0xff0000ff, 0xffffffff};
const int      num_colors_phydrv = sizeof(colors_phydrv) / sizeof(uint32_t);

#define PUSH_RANGE_PHYDRV(name, cid)                                       \
    {                                                                      \
        int color_id                      = cid;                           \
        color_id                          = color_id % num_colors_phydrv;  \
        nvtxEventAttributes_t eventAttrib = {0};                           \
        eventAttrib.version               = NVTX_VERSION;                  \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;               \
        eventAttrib.color                 = colors_phydrv[color_id];       \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;       \
        eventAttrib.message.ascii         = name;                          \
        nvtxRangePushEx(&eventAttrib);                                     \
    }
#define POP_RANGE_PHYDRV nvtxRangePop();
#else
#define PUSH_RANGE_PHYDRV(name, cid)
#define POP_RANGE_PHYDRV
#endif

#ifndef TAG
#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 7) // "DRV.GPUDEV"
#endif

#define CUDA_CHECK_PHYDRIVER(stmt)                   \
    do                                               \
    {                                                \
        cudaError_t result1 = (stmt);                 \
        if(cudaSuccess != result1)                    \
        {                                            \
            NVLOGW_FMT(TAG,"[{}:{}] cuda failed with result1 {} ",__FILE__,__LINE__,cudaGetErrorString(result1));    \
            cudaError_t result2 = cudaGetLastError();             \
            if(cudaSuccess != result2)                            \
            {                                                     \
                NVLOGW_FMT(TAG,"[{}:{}] cuda failed with result2 {} result1 {}",__FILE__,__LINE__,cudaGetErrorString(result2),cudaGetErrorString(result1));    \
                cudaError_t result3 = cudaGetLastError();/*check for stickiness*/             \
                if(cudaSuccess != result3)                    \
                {                                            \
                    NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with result3 {} result2 {} result1 {}", \
                           __FILE__,                         \
                           __LINE__,                         \
                           cudaGetErrorString(result3),      \
                           cudaGetErrorString(result2),      \
                           cudaGetErrorString(result1));      \
                }                                            \
            }                  \
         }                                            \
    } while(0)

#define CUDA_CHECK_PHYDRIVER_NONFATAL(stmt,id)                   \
    do                                               \
    {                                                \
        cudaError_t result1 = (stmt);                 \
        if(cudaSuccess != result1)                    \
        {                                            \
            NVLOGW_FMT(TAG,"[{}:{}] cuda failed with result1 {} for Obj {:x}",__FILE__,__LINE__,cudaGetErrorString(result1),id);    \
            cudaError_t result2 = cudaGetLastError();             \
            if(cudaSuccess != result2)                            \
            {                                                     \
                NVLOGW_FMT(TAG,"[{}:{}] cuda failed with result2 {} result1 {} for Obj {:x}",__FILE__,__LINE__,cudaGetErrorString(result2),cudaGetErrorString(result1),id);    \
                cudaError_t result3 = cudaGetLastError();/*check for stickiness*/             \
                if(cudaSuccess != result3)                    \
                {                                            \
                    NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with result3 {} result2 {} result1 {} for Obj {:x}", \
                           __FILE__,                         \
                           __LINE__,                         \
                           cudaGetErrorString(result3),      \
                           cudaGetErrorString(result2),      \
                           cudaGetErrorString(result1),id);      \
                }                                            \
            }                                            \
         }                                            \
    } while(0)

#define CU_CHECK_PHYDRIVER(stmt)                   \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cu failed with {} ", \
                   __FILE__,                       \
                   __LINE__,                       \
                   +result);                        \
        }                                          \
    } while(0)

#define CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(stmt)  \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] CUDA Driver API call failed with {}", \
                   __FILE__,                       \
                   __LINE__,                       \
                   +result);                        \
        }                                          \
    } while(0)

#ifdef DEVICE_TEGRA
#define GPU_PAGE_SHIFT 12                                      ///< GPU page size shift (12 = 4KB pages for Tegra)
#else
#define GPU_PAGE_SHIFT 16                                      ///< GPU page size shift (16 = 64KB pages for other architectures)
#endif
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)                  ///< GPU page size in bytes (4KB or 64KB depending on platform)

#ifdef DEVICE_TEGRA
#define GPU_MIN_PIN_SIZE GPU_PAGE_SIZE                         ///< Minimum GDR pin size (4KB for Tegra)
#else
#define GPU_MIN_PIN_SIZE 4                                     ///< Minimum GDR pin size (4 bytes for other architectures)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OKI_MAX_READ_CALLS 50                              ///< Maximum number of read call samples in order kernel instrumentation
#define OKI_MAX_PROC_CALLS 50                              ///< Maximum number of process call samples in order kernel instrumentation

/**
 * @brief Order Kernel Instrumentation structure
 *
 * Captures timing and packet count information for UL order kernel operations.
 * Used for performance profiling and debugging of packet receive and processing phases.
 */
typedef struct OKInstrumentation {
    uint64_t ok_read_ref_time;                             ///< Reference time for read phase measurements
    uint64_t start_read_time[OKI_MAX_READ_CALLS];          ///< Start time for each read operation
    uint64_t stop_read_time[OKI_MAX_READ_CALLS];           ///< Stop time for each read operation
    short read_packets[OKI_MAX_READ_CALLS];                ///< Number of packets read in each operation
    uint64_t ok_proc_ref_time;                             ///< Reference time for processing phase measurements
    uint64_t start_proc_time[OKI_MAX_PROC_CALLS];          ///< Start time for each processing operation
    uint64_t stop_proc_time[OKI_MAX_PROC_CALLS];           ///< Stop time for each processing operation
    short proc_packets[OKI_MAX_PROC_CALLS];                ///< Number of packets processed in each operation
    int num_reads;                                         ///< Total number of read operations recorded
    int num_procs;                                         ///< Total number of process operations recorded
} OKInstrumentation;

void force_loading_generic_cuda_kernels();
void force_loading_order_kernels();

void launch_memset_kernel(void* d_buffers_addr, int num_cells, size_t max_buffer_size, cudaStream_t strm);


void launch_kernel_order(
    cudaStream_t          stream,
    int                   fake_run,
    int                   cell_id,
    uint8_t*              order_completed_d,
    uint8_t*              order_completed_h,
    uint32_t*             order_start_gdr_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    uint8_t*              buffer,
    int                   prb_x_slot,
    int                   prb_x_symbol,
    int                   prb_x_symbol_x_antenna,
    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,
    uint16_t*             eAxC_map,
    int                   eAxC_num,
    int                   comp_meth,
    int                   bit_width,
    int                   prb_size,
    float                 beta);

int launch_kernel_order_mb(
    cudaStream_t          stream,
    int                   fake_run,
    int                   cell_id,
    uint32_t*             order_kernel_end_cuphy_d,
    uint32_t*             order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    uint32_t              timeout_no_pkt_ns,
    uint32_t              timeout_first_pkt_ns,

    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,
    int                   comp_meth,
    int                   bit_width,
    int                   prb_size,
    float                 beta,
    int*                  barrier_flag,
    uint8_t*              done_shared,
    int*                  ready_shared,
    int*                  rx_queue_index,

    uint32_t*             early_rx_packets,
    uint32_t*             on_time_rx_packets,
    uint32_t*             late_rx_packets,
    uint64_t              slot_start,
    uint64_t              Ta4_min_ns,
    uint64_t              Ta4_max_ns,
    uint64_t              slot_duration,
    
    uint16_t*             ST1_eAxC_map,
    int                   ST1_eAxC_num,
    uint8_t*              ST1_buffer,
    int                   ST1_prb_x_slot,
    int                   ST1_prb_x_symbol,
    int                   ST1_prb_x_symbol_x_antenna,
    uint32_t              ST1_prb_stride,
    uint32_t*             ST1_ordered_prbs,

    uint16_t*             prach_eAxC_map,
    int                   prach_eAxC_num,
    // uint8_t*              prach_buffer,
    // uint16_t              prach_section_id,
    uint8_t * prach_buffer_o0, uint8_t * prach_buffer_o1, uint8_t * prach_buffer_o2, uint8_t * prach_buffer_o3,
    uint16_t prach_section_id_o0, uint16_t prach_section_id_o1, uint16_t prach_section_id_o2, uint16_t prach_section_id_o3,
    int                   prach_prb_x_slot,
    int                   prach_prb_x_symbol,
    int                   prach_prb_x_symbol_x_antenna,
    uint32_t              prach_prb_stride,
    uint32_t*             prach_ordered_prbs
);

int launch_order_kernel_doca(
	cudaStream_t          stream,

    struct doca_gpu_eth_rxq *doca_rxq,
	struct doca_gpu_semaphore_gpu *sem_gpu,
	const uint16_t sem_order_num,

    int                   cell_id,
    int                   ru_type,

	uint32_t*             start_cuphy_d,
	uint32_t*             order_kernel_exit_cond_d,
	uint32_t		*last_sem_idx_rx_h,
    uint32_t		*last_sem_idx_order_h,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
	uint32_t              max_rx_pkts,
    
	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int                   comp_meth,
    int                   bit_width,
	int                   prb_size,
	float                 beta,
	int*                  barrier_flag,
	uint8_t*              done_shared,
    
	uint32_t*             early_rx_packets,
	uint32_t*             on_time_rx_packets,
	uint32_t*             late_rx_packets,
	uint64_t              slot_start,
	uint64_t              ta4_min_ns,
	uint64_t              ta4_max_ns,
	uint64_t              slot_duration,
	
	uint16_t*             pusch_eAxC_map,
	int                   pusch_eAxC_num,
	uint8_t*              pusch_buffer,
	int                   pusch_prb_x_slot,
	int                   pusch_prb_x_symbol,
	int                   pusch_prb_x_symbol_x_antenna,
	uint32_t              pusch_prb_stride,
	uint32_t*             pusch_ordered_prbs,
    
	uint16_t*             prach_eAxC_map,
	int                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t * prach_buffer_0,
    uint8_t * prach_buffer_1,
    uint8_t * prach_buffer_2,
    uint8_t * prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int                   prach_prb_x_slot,
	int                   prach_prb_x_symbol,
	int                   prach_prb_x_symbol_x_antenna,
	uint32_t              prach_prb_stride,
	uint32_t*             prach_ordered_prbs
    );

int launch_order_kernel_doca_single(
	cudaStream_t          stream,

    struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

    int*                   cell_id,
    int*                   ru_type,
    bool*                  cell_health,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    uint32_t		**last_sem_idx_order_h,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
	uint32_t              timeout_log_interval_ns,
	uint8_t               timeout_log_enable,   
	uint32_t              max_rx_pkts,
    uint32_t              rx_pkts_timeout_ns,
    bool                  commViaCpu,
    
	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,
    
	uint32_t**             early_rx_packets,
	uint32_t**             on_time_rx_packets,
	uint32_t**             late_rx_packets,
	uint32_t**             next_slot_early_rx_packets,
	uint32_t**             next_slot_on_time_rx_packets,
	uint32_t**             next_slot_late_rx_packets,	
	uint64_t*              slot_start,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
	uint64_t*              slot_duration,
	uint64_t**             order_kernel_last_timeout_error_time,
	uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**             rx_packets_count,
    uint32_t**             rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,
	
	uint16_t**             pusch_eAxC_map,
	int*                   pusch_eAxC_num,
	uint8_t**              pusch_buffer,
	int*                   pusch_prb_x_slot,
	int*                   pusch_prb_x_symbol,
	int*                   pusch_prb_x_symbol_x_antenna,
	uint32_t*              pusch_prb_stride,
	uint32_t**             pusch_ordered_prbs,

	uint16_t**             prach_eAxC_map,
	int*                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t** prach_buffer_0,
    uint8_t** prach_buffer_1,
    uint8_t** prach_buffer_2,
    uint8_t** prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int*                   prach_prb_x_slot,
	int*                   prach_prb_x_symbol,
	int*                   prach_prb_x_symbol_x_antenna,
	uint32_t*              prach_prb_stride,
	uint32_t**             prach_ordered_prbs,
	uint8_t num_order_cells,

    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size
    );

int launch_order_kernel_doca_single_subSlot(
	cudaStream_t stream,

	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
    struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh,
	const uint16_t* sem_order_num,

    int*                   cell_id,
    int*                   ru_type,
    bool*                  cell_health,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
    uint32_t              **last_sem_idx_rx_h,
    uint32_t              **last_sem_idx_order_h,
    uint8_t               ul_order_kernel_mode,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
    uint32_t              timeout_log_interval_ns,
    uint8_t               timeout_log_enable,
    uint32_t              max_rx_pkts,
    uint32_t              rx_pkts_timeout_ns,
    bool                  commViaCpu,

	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,

	uint32_t**             early_rx_packets,
	uint32_t**             on_time_rx_packets,
	uint32_t**             late_rx_packets,
	uint32_t**             next_slot_early_rx_packets,
	uint32_t**             next_slot_on_time_rx_packets,
	uint32_t**             next_slot_late_rx_packets,	
	uint64_t*              slot_start,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
	uint64_t*              slot_duration,
    uint64_t**             order_kernel_last_timeout_error_time,
	uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**              rx_packets_count,
    uint32_t**              rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
    uint32_t**             rx_packets_dropped_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,	

    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
    uint32_t* 			   num_order_cells_sym_mask_arr,
    uint8_t                pusch_prb_non_zero,

	uint16_t**             pusch_eAxC_map,
	int*                   pusch_eAxC_num,
	uint8_t**              pusch_buffer,
	int*                   pusch_prb_x_slot,
	int*                   pusch_prb_x_symbol,
	int*                   pusch_prb_x_symbol_x_antenna,
	uint32_t*              pusch_prb_stride,
	uint32_t**             pusch_ordered_prbs,

	uint16_t**             prach_eAxC_map,
	int*                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t** prach_buffer_0,
    uint8_t** prach_buffer_1,
    uint8_t** prach_buffer_2,
    uint8_t** prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int*                   prach_prb_x_slot,
	int*                   prach_prb_x_symbol,
	int*                   prach_prb_x_symbol_x_antenna,
	uint32_t*              prach_prb_stride,
	uint32_t**             prach_ordered_prbs,

	uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
    uint32_t*              srs_prb_stride,
	uint32_t		**srs_ordered_prbs,
    uint8_t*          srs_start_sym,

    uint8_t num_order_cells,
    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size,
    uint8_t srs_enable
    );

int launch_order_kernel_cpu_init_comms_single_subSlot(
cudaStream_t stream,

uint32_t**             start_cuphy_d,
uint32_t**             order_kernel_exit_cond_d,
uint32_t**             ready_list,
struct aerial_fh::rx_queue_sync** rx_queue_sync_list,
uint32_t**                  last_ordered_item_h,
uint16_t*              sem_order_num,

uint8_t               frameId,
uint8_t               subframeId,
uint8_t               slotId,
int*                   comp_meth,
int*                   bit_width,
int                   prb_size,
float*                 beta,
int*                  barrier_flag,
uint8_t**              done_shared,

uint32_t              timeout_no_pkt_ns,
uint32_t              timeout_first_pkt_ns,

uint32_t*              sym_ord_done_sig_arr,
uint32_t*              sym_ord_done_mask_arr,
uint32_t*              pusch_prb_symbol_map,
uint32_t*              num_order_cells_sym_mask_arr,

/* Timer */
uint32_t 		**early_rx_packets,
uint32_t 		**on_time_rx_packets,
uint32_t 		**late_rx_packets,
uint32_t 		**next_slot_early_rx_packets,
uint32_t 		**next_slot_on_time_rx_packets,
uint32_t 		**next_slot_late_rx_packets,	
uint64_t*		slot_start,
uint64_t*		ta4_min_ns,
uint64_t*		ta4_max_ns,
uint64_t*		slot_duration,
uint8_t                ul_rx_pkt_tracing_level,
uint64_t**             rx_packets_ts,
uint32_t**              rx_packets_count,
uint32_t**              rx_bytes_count,
uint64_t**             rx_packets_ts_earliest,
uint64_t**             rx_packets_ts_latest,	
uint64_t**             next_slot_rx_packets_ts,
uint32_t**             next_slot_rx_packets_count,
uint32_t**             next_slot_rx_bytes_count,

uint16_t**             pusch_eAxC_map,
int*                   pusch_eAxC_num,
uint8_t**              pusch_buffer,
int*                   pusch_prb_x_slot,
int*                   pusch_prb_x_symbol,
int*                   pusch_prb_x_symbol_x_antenna,
uint32_t*              pusch_prb_stride,
uint32_t**             pusch_ordered_prbs,

uint16_t**             prach_eAxC_map,
int*                   prach_eAxC_num,
// uint8_t*              prach_buffer,
// uint16_t              prach_section_id,

uint8_t** prach_buffer_0,
uint8_t** prach_buffer_1,
uint8_t** prach_buffer_2,
uint8_t** prach_buffer_3,

uint16_t prach_section_id_0,
uint16_t prach_section_id_1,
uint16_t prach_section_id_2,
uint16_t prach_section_id_3,

int*                   prach_prb_x_slot,
int*                   prach_prb_x_symbol,
int*                   prach_prb_x_symbol_x_antenna,
uint32_t*              prach_prb_stride,
uint32_t**             prach_ordered_prbs,
uint8_t num_order_cells
);

int launch_order_kernel_doca_single_srs(
	cudaStream_t          stream,

    struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

    int*                   cell_id,
    int*                   ru_type,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    uint32_t		**last_sem_idx_order_h,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
    uint32_t              max_rx_pkts,
    
	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,
	uint32_t**             early_rx_packets_srs,
	uint32_t**             on_time_rx_packets_srs,
	uint32_t**             late_rx_packets_srs,
	uint32_t**             next_slot_early_rx_packets_srs,
	uint32_t**             next_slot_on_time_rx_packets_srs,
	uint32_t**             next_slot_late_rx_packets_srs,
    uint32_t**             rx_packets_count_srs,
    uint32_t**             rx_bytes_count_srs,
    uint32_t**             next_slot_rx_packets_count_srs,
    uint32_t**             next_slot_rx_bytes_count_srs,
    uint8_t                ul_rx_pkt_tracing_level,
    uint64_t**             rx_packets_ts_srs,
    uint32_t**             rx_packets_count_per_sym_srs,
    uint64_t**             rx_packets_ts_earliest_srs,
    uint64_t**             rx_packets_ts_latest_srs,	
    uint64_t**             next_slot_rx_packets_ts_srs,
    uint32_t**             next_slot_rx_packets_count_per_sym_srs,     
	uint64_t*              slot_start_srs,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
	uint64_t*              slot_duration,
	uint16_t**             srs_eAxC_map,
	int*                   srs_eAxC_num,
	uint8_t**              srs_buffer,
	int*                   srs_prb_x_slot,
	uint32_t*              srs_prb_stride,
	uint32_t**             srs_ordered_prbs,
	uint8_t*                srs_start_sym, 
	uint8_t num_order_cells
    );

int launch_receive_kernel_for_test_bench(
	cudaStream_t stream,

	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
    const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    const int*		bit_width,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t  max_rx_pkts,
    const uint32_t  max_pkt_size,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

    uint32_t**             rx_packets_count,
	uint32_t**             next_slot_rx_packets_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

    uint8_t**               tb_fh_buf,
    uint8_t**               tb_fh_buf_next_slot,

	int*			pusch_prb_x_slot,
	int*		prach_prb_x_slot,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    int*                   srs_prb_x_slot,
	uint8_t num_order_cells
    );    

void launch_kernel_wait_update(cudaStream_t stream, uint32_t* addr, uint32_t expected, uint32_t updated);
void launch_kernel_wait_eq(cudaStream_t stream, uint32_t* addr, uint32_t value);
void launch_kernel_wait_geq(cudaStream_t stream, uint32_t* addr, uint32_t value);
void launch_kernel_wait_neq(cudaStream_t stream, uint32_t* addr, uint32_t value);
void launch_kernel_write(cudaStream_t stream, uint32_t* addr, uint32_t value);
void launch_kernel_warmup(cudaStream_t stream);
void launch_kernel_compare(cudaStream_t stream, uint8_t* addr1, uint8_t* addr2, int size);
void launch_kernel_check_crc(cudaStream_t stream, const uint32_t* i_buf, size_t i_elems, uint32_t* out);
void launch_kernel_read(cudaStream_t stream, uint8_t* addr);
void launch_kernel_copy(cudaStream_t stream, uint8_t* input_buffer, uint8_t* output_buffer, int bytes);

void launch_kernel_print_hex(cudaStream_t stream, uint8_t* addr, int offset, int num_bytes);
void launch_receive_process_kernel_for_test_bench(
    cudaStream_t stream,
	/* Cell */
	const int*		cell_id,
	uint32_t		**exit_cond_d,
    const uint16_t* sem_order_num,
    const int*		ru_type,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

    const int		prb_size,
    const int*		comp_meth,
    const int*		bit_width,
    const float*		beta,
    uint32_t		**last_sem_idx_order_h,
    
    uint32_t*       rx_pkt_num_slot,
    uint8_t**       tb_fh_buf,
    const uint32_t  max_pkt_size,

    uint32_t         **early_rx_packets,
    uint32_t         **on_time_rx_packets,
    uint32_t         **late_rx_packets,
    uint32_t         **next_slot_early_rx_packets,
    uint32_t         **next_slot_on_time_rx_packets,
    uint32_t         **next_slot_late_rx_packets,
    uint32_t         **rx_packets_dropped_count,
    bool             *cell_health,
    uint32_t         **start_cuphy_d,

    /* Sub-slot processing*/
    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
	uint32_t* 			   num_order_cells_sym_mask_arr,    
    
    /*PUSCH*/
    uint8_t**       pusch_buffer,
	uint16_t		**pusch_eAxC_map,
	int*			pusch_eAxC_num,    
    int			pusch_symbols_x_slot,
    uint32_t*			pusch_prb_x_port_x_symbol,
    uint32_t		**pusch_ordered_prbs,
    int*			pusch_prb_x_slot,

    /*PRACH*/
	uint16_t 	**prach_eAxC_map,
	int*		prach_eAxC_num,
	uint8_t		**prach_buffer_0,
	uint8_t		**prach_buffer_1,
	uint8_t 	**prach_buffer_2,
	uint8_t 	**prach_buffer_3,
    int*	    prach_prb_x_slot,
    int			prach_symbols_x_slot,
    uint32_t*   prach_prb_x_port_x_symbol,
    uint32_t	**prach_ordered_prbs,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    uint8_t num_order_cells,

    /*SRS*/
    uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
    uint32_t*              srs_prb_stride,	
	uint32_t		**srs_ordered_prbs,
	uint8_t*          srs_start_sym,

    /*Receive CTA params*/
    const uint32_t	timeout_no_pkt_ns,
    const uint32_t	timeout_first_pkt_ns,
	const uint32_t  timeout_log_interval_ns,
	const uint8_t   timeout_log_enable,
    uint64_t      **order_kernel_last_timeout_error_time,
    uint32_t		**last_sem_idx_rx_h,
    bool            commViaCpu,
    struct doca_gpu_eth_rxq **doca_rxq,
    const uint32_t  max_rx_pkts,
    const uint32_t  rx_pkts_timeout_ns,
    struct doca_gpu_semaphore_gpu **sem_gpu,
    struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh,
	uint64_t*		slot_start,
	uint64_t*		ta4_min_ns,
	uint64_t*		ta4_max_ns,
	uint64_t*		slot_duration,
    uint8_t                ul_rx_pkt_tracing_level,
    uint8_t                ul_order_kernel_mode,
    uint8_t                enable_srs
);


void launch_kernel_compression(
    cudaStream_t stream,
    const std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array);

#ifdef __cplusplus
}
#endif

/**
 * @brief Host pinned memory allocator
 *
 * Allocator for host (CPU) pinned memory using cudaHostAlloc/cudaFreeHost.
 * Pinned memory enables faster DMA transfers between CPU and GPU.
 * Used with IOBuf template for host-side buffer management.
 */
struct hpinned_alloc
{
    /**
     * @brief Allocate host pinned memory
     *
     * @param nbytes - Number of bytes to allocate
     * @return Pointer to allocated pinned host memory
     */
    static void* allocate(size_t nbytes)
    {
        void* addr;
        // CUDA_CHECK_PHYDRIVER(cudaMallocHost(&addr, nbytes));
        CUDA_CHECK_PHYDRIVER(cudaHostAlloc(&addr, nbytes, cudaHostAllocDefault | cudaHostAllocPortable));
        return addr;
    }

    /**
     * @brief Free host pinned memory
     *
     * @param addr - Pointer to pinned host memory to free
     */
    static void deallocate(void* addr)
    {
        // NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "hpinned_alloc CUDA_CHECK_PHYDRIVER(cudaFreeHost"));
        CUDA_CHECK_PHYDRIVER(cudaFreeHost(addr));
    }

    /**
     * @brief Clear host pinned memory to zero
     *
     * @param addr   - Pointer to host memory to clear
     * @param nbytes - Number of bytes to clear
     */
    static void clear(void* addr, size_t nbytes)
    {
        memset(addr, 0, nbytes);
    }
};

/**
 * @brief GPU Direct RDMA (GDR) pinned buffer
 *
 * Provides zero-copy access to GPU memory from CPU using GPUDirect RDMA.
 * Manages GPU memory allocation, GDR pinning, and host memory mapping for
 * low-latency CPU-GPU communication without going through system memory.
 */
typedef struct gpinned_buffer
{
public:
    /**
     * @brief Construct a GDR pinned buffer
     *
     * @param _g           - GDR context handle
     * @param _size_input  - Requested buffer size in bytes (will be rounded to GPU page size)
     */
    gpinned_buffer(gdr_t* _g, size_t _size_input, bool _is_rdma_supported) :
        g(_g),
        size_input(_size_input),
        is_rdma_supported(_is_rdma_supported)
    {
        CUdeviceptr        dev_addr = 0;
        void*              host_ptr = NULL;
        const unsigned int FLAG     = 1;
        size_t             pin_size, alloc_size, rounded_size;

        if(g == nullptr || size_input == 0)
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "gpinned_buffer bad input arguments");

        // If RDMA is supported - proceed to using GDRCopy library for further allocation
        // If not, use CUDA pinned host memory allocation. 
        if (!is_rdma_supported)
        {

            host_ptr = hpinned_alloc::allocate(size_input); 
            // In a system with full unified memory, the host and the device pointer _may_ match.
            CU_CHECK_PHYDRIVER(cuMemHostGetDevicePointer(&dev_addr, host_ptr, 0));

            addr_d    = (uintptr_t)dev_addr;
            addr_h    = (uintptr_t)host_ptr;
            return; 
        }

        if(size_input < GPU_MIN_PIN_SIZE)
            size_input = GPU_MIN_PIN_SIZE;

        // GDRDRV and the GPU driver require GPU page size-aligned address and size
        // arguments to gdr_pin_buffer, so we need to be paranoid here and allocate
        // an extra page so we can safely pass the rounded size
        rounded_size = (size_input + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        pin_size     = rounded_size;
        alloc_size   = rounded_size + GPU_PAGE_SIZE;

/*----------------------------------------------------------------*
            * Allocate device memory.                                        */
#ifdef DEVICE_TEGRA
        void* cudaHost_A;

        CUresult e = cuMemHostAlloc(&cudaHost_A, alloc_size, 0);
        if(CUDA_SUCCESS != e)
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "cuMemHostAlloc");

        e = cuMemHostGetDevicePointer(&dev_addr, cudaHost_A, 0);
        if(CUDA_SUCCESS != e)
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "cuMemHostGetDevicePointer");
#else
        // CU_CHECK_PHYDRIVER(cuMemAlloc(&dev_addr, alloc_size));
        CU_CHECK_PHYDRIVER(cuMemAlloc(&dev_addr, alloc_size));
#endif

        addr_free = (uintptr_t)dev_addr;
        // Offset into a page-aligned address if necessary
        if(dev_addr % GPU_PAGE_SIZE)
        {
            dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
        }

        /*----------------------------------------------------------------*
            * Set attributes for the allocated device memory.                */
        CU_CHECK_PHYDRIVER(cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr));
        // if(CUDA_SUCCESS != cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr))
        // {
        //     cuMemFree(dev_addr);
        //     // gdr_close(g);
        //     PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "cuPointerSetAttribute");
        // }
        /*----------------------------------------------------------------*
            * Pin the device buffer                                          */
        if(0 != gdr_pin_buffer(*g, dev_addr, pin_size, 0, 0, &mh))
        {
            CU_CHECK_PHYDRIVER(cuMemFree(dev_addr));
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "gdr_pin_buffer");
        }
        /*----------------------------------------------------------------*
            * Map the buffer to user space                                   */
        if(0 != gdr_map(*g, mh, &host_ptr, pin_size))
        {
            gdr_unpin_buffer(*g, mh);
            CU_CHECK_PHYDRIVER(cuMemFree(dev_addr));
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "gdr_map");
        }
        /*----------------------------------------------------------------*
            * Retrieve info about the mapping                                */
        if(0 != gdr_get_info(*g, mh, &info))
        {
            gdr_unmap(*g, mh, host_ptr, pin_size);
            gdr_unpin_buffer(*g, mh);
            CU_CHECK_PHYDRIVER(cuMemFree(dev_addr));
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "gdr_get_info");
        }

        addr_d    = (uintptr_t)dev_addr;
        addr_h    = (uintptr_t)host_ptr;
        size_free = pin_size;
        size_alloc = alloc_size;
    };

    /**
     * @brief Destroy GDR pinned buffer
     *
     * Unmaps host memory, unpins GPU buffer, and frees GPU device memory.
     */
    ~gpinned_buffer()
    {
        if (is_rdma_supported) {
            gdr_unmap(*g, mh, (void*)addr_h, size_free);
            gdr_unpin_buffer(*g, mh);
            CU_CHECK_PHYDRIVER(cuMemFree((CUdeviceptr)addr_free));
        } else {
            hpinned_alloc::deallocate((void*)addr_h); 
        }
    };

    /**
     * @brief Get host (CPU) address for zero-copy access
     *
     * @return Host memory address mapped to GPU memory
     */
    void* addrh()
    {
        return (void*)addr_h;
    }

    /**
     * @brief Get device (GPU) address
     *
     * @return GPU device memory address
     */
    void* addrd()
    {
        return (void*)addr_d;
    }

    /**
     * @brief Get buffer size
     *
     * @return Buffer size in bytes (requested size, not rounded)
     */
    size_t size()
    {
        return size_input;
    }

    // Used to measure memory footprint
    size_t     size_free;                                  ///< Actual allocated size (rounded to GPU page size) for memory footprint tracking
    size_t     size_alloc;                                 ///< Requested buffer size in bytes for memory footprint tracking

protected:
    gdr_t*     g;                                          ///< GDR context handle
    gdr_mh_t   mh;                                         ///< GDR memory handle for this buffer
    gdr_info_t info;                                       ///< GDR buffer information (mapping offset, etc.)
    uintptr_t  addr_d;                                     ///< GPU device memory address (page-aligned)
    uintptr_t  addr_h;                                     ///< Host memory address (mapped from GPU memory)
    uintptr_t  addr_free;                                  ///< Original GPU address for freeing (may be unaligned)
    size_t     size_input;                                 ///< Requested buffer size in bytes
    bool       is_rdma_supported;                          ///< Denotes if GPU RDMA is supported in the target
} gpinned_buffer;

/**
 * @brief GPU device manager
 *
 * Manages a single CUDA GPU device including device selection, properties,
 * stream synchronization, GDR buffer allocation, and memory footprint tracking.
 */
class GpuDevice {
public:
    /**
     * @brief Construct GPU device manager
     *
     * @param _pdh      - Physical layer driver handle
     * @param _id       - CUDA device ID (0-based)
     * @param init_gdr  - Initialize GPUDirect RDMA support (true=enable GDR)
     */
    GpuDevice(phydriver_handle _pdh, uint32_t _id, bool init_gdr);
    
    /**
     * @brief Destroy GPU device manager
     *
     * Closes GDR context if initialized.
     */
    ~GpuDevice();

    phydriver_handle       getPhyDriverHandler(void) const;    ///< Get physical layer driver handle
    void                   setDevice();                        ///< Set this device as active CUDA device for current thread
    uint32_t               getId();                            ///< Get CUDA device ID
    void                   print_info();                       ///< Print GPU device properties to log
    void                   synchronizeStream(cudaStream_t stream); ///< Synchronize CUDA stream (blocking wait for completion)
    
    /**
     * @brief Run GPU warmup kernels
     *
     * @param n      - Number of warmup iterations
     * @param stream - CUDA stream for warmup kernels
     * @return 0
     */
    int                    runWarmup(int n, cudaStream_t stream);
    
    /**
     * @brief Allocate new GDR pinned buffer
     *
     * @param size - Buffer size in bytes (will be rounded to GPU page size)
     * @return Pointer to allocated GDR buffer, nullptr on failure
     */
    struct gpinned_buffer* newGDRbuf(size_t size);
    
    gdr_t*                 getGDRhandler();                    ///< Get GDR context handle for this device
    
    MemFoot                mf;                                 ///< Memory footprint tracker for this GPU device

private:
    phydriver_handle      pdh;                                 ///< Physical layer driver handle
    uint32_t              id;                                  ///< CUDA device ID (0-based)
    int                   tot_devs;                            ///< Total number of CUDA devices in system
    struct cudaDeviceProp deviceProp;                          ///< CUDA device properties (name, compute capability, memory, etc.)
    int                   device_attr_clock_rate;              ///< GPU clock rate in kHz
    int                   device_is_direct_rdma_supported;     ///< Denotes if GPU Direct RDMA is supported in target
    gdr_t                 gdrc_h;                              ///< GPUDirect RDMA context handle
    bool                  init_gdr;                            ///< Flag indicating GDR was initialized for this device
};

/**
 * @brief GPU device memory allocator
 *
 * Allocator for device (GPU) memory using cudaMalloc/cudaFree.
 * Used with IOBuf template for device-side buffer management.
 */
struct device_alloc
{
    /**
     * @brief Allocate device memory
     *
     * @param nbytes - Number of bytes to allocate
     * @return Pointer to allocated device memory
     */
    static void* allocate(size_t nbytes)
    {
        void* addr;
        CUDA_CHECK_PHYDRIVER(cudaMalloc(&addr, nbytes));
        return addr;
    }
    
    /**
     * @brief Free device memory
     *
     * @param addr - Pointer to device memory to free
     */
    static void deallocate(void* addr)
    {
        // NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "device_alloc CUDA_CHECK_PHYDRIVER(cudaFree"));
        CUDA_CHECK_PHYDRIVER(cudaFree(addr));
    }

    /**
     * @brief Clear device memory to zero
     *
     * @param addr   - Pointer to device memory to clear
     * @param nbytes - Number of bytes to clear
     */
    static void clear(void* addr, size_t nbytes)
    {
        CUDA_CHECK_PHYDRIVER(cudaMemset(addr, 0, nbytes));
    }
};

/**
 * @brief Generic I/O buffer with templated allocator
 *
 * RAII wrapper for memory buffers with configurable allocation strategy.
 * Supports both device (GPU) and host (CPU) memory through TAllocator template parameter.
 *
 * @tparam T          - Element type (e.g., uint8_t for byte buffers)
 * @tparam TAllocator - Allocator type (device_alloc or hpinned_alloc)
 */
template <typename T, class TAllocator>
class IOBuf {
public:
    /**
     * @brief Construct empty buffer (no allocation)
     */
    IOBuf() :
        _addr(nullptr),
        _size(0),
        size_alloc(0),
        gDev(nullptr) {}
    
    /**
     * @brief Construct and allocate buffer
     *
     * @param numElements - Number of elements (T) to allocate
     * @param _gDev       - GPU device pointer (for tracking purposes)
     */
    IOBuf(size_t numElements, GpuDevice* _gDev) :
        // addr(static_cast<T*>(TAllocator::allocate(numElements * sizeof(T)))),
        _size(numElements),
        size_alloc(numElements * sizeof(T)),
        gDev(_gDev)
    {
        _addr = static_cast<T*>(TAllocator::allocate(size_alloc));
        // std::cout << "Allocated a new buffer of " << _size << "bytes" << std::endl;
    };

    /**
     * @brief Destroy buffer and free memory
     */
    ~IOBuf()
    {
        if(_addr)
        {
            // NVLOGI_FMT(TAG, "Free the IOBuf of {} bytes", _size);
            TAllocator::deallocate(_addr);
        }
    }

    T*     addr() { return _addr; }                            ///< Get buffer address
    size_t size() const { return _size; }                      ///< Get buffer size in elements
    void   clear() { TAllocator::clear(_addr, _size); }       ///< Clear buffer to zero

    size_t     size_alloc;                                     ///< Actual allocated size in bytes for memory footprint tracking

private:
    T*         _addr;                                          ///< Buffer address (device or host depending on TAllocator)
    size_t     _size;                                          ///< Buffer size in elements (not bytes)
    GpuDevice* gDev;                                           ///< GPU device pointer for tracking
};

typedef IOBuf<uint8_t, device_alloc>  dev_buf;                ///< Device (GPU) memory buffer (uint8_t)
typedef IOBuf<uint8_t, hpinned_alloc> host_buf;               ///< Host (CPU) pinned memory buffer (uint8_t)

#endif
