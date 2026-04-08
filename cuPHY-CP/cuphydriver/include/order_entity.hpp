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

#ifndef ORDERENTITY_H
#define ORDERENTITY_H

#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include "gpudevice.hpp"
#include "time.hpp"
#include "constant.hpp"
#include "fh.hpp"
#include "mps.hpp"
#include <slot_command/slot_command.hpp>
#include "cuphydriver_api.hpp"
#include "cuphy_api.h"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "aerial-fh-driver/oran.hpp"

/**
 * @brief Configuration parameters for GPU-based UL packet ordering kernel (PUSCH/PUCCH/PRACH/SRS)
 *
 * Contains all parameters needed by the GPU order kernel to process uplink packets
 * from DOCA GPUDirect Ethernet receive queues. The kernel reads packets, decompresses
 * IQ data, and organizes PRBs into output buffers for cuPHY processing. Supports
 * PUSCH, PUCCH, PRACH, and SRS channels with per-cell configuration and timing validation.
 */
typedef struct orderKernelConfigParams{
    struct doca_gpu_eth_rxq *rxq_info_gpu[UL_MAX_CELLS_PER_SLOT];          ///< DOCA GPU receive queue handles for direct GPU packet access
    struct doca_gpu_semaphore_gpu *sem_gpu[UL_MAX_CELLS_PER_SLOT];         ///< DOCA GPU semaphores for RX/order kernel synchronization
    struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh[UL_MAX_CELLS_PER_SLOT]; ///< Aerial FH GPU semaphores for RX/order kernel synchronization
    uint16_t sem_order_num[UL_MAX_CELLS_PER_SLOT];                         ///< Semaphore value to signal after ordering complete
    int                   cell_id[UL_MAX_CELLS_PER_SLOT];                  ///< Physical cell IDs
    int                   comp_meth[UL_MAX_CELLS_PER_SLOT];                ///< Compression method
    int                   bit_width[UL_MAX_CELLS_PER_SLOT];                ///< IQ sample bit width for compressed data
    int                   ru_type[UL_MAX_CELLS_PER_SLOT];                  ///< RU type identifier
    bool                  cell_health[UL_MAX_CELLS_PER_SLOT];              ///< Cell health status (true=healthy, false=unhealthy/skip)
    float                 beta[UL_MAX_CELLS_PER_SLOT];                     ///< BFP compression beta parameter
    uint64_t              slot_start[UL_MAX_CELLS_PER_SLOT];               ///< Slot start time (nanoseconds since epoch)
    uint64_t              ta4_min_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 min timing constraint (earliest expected packet arrival)
    uint64_t              ta4_max_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 max timing constraint (latest acceptable packet arrival)
    uint64_t              slot_duration[UL_MAX_CELLS_PER_SLOT];            ///< Slot duration (nanoseconds)
    int                   pusch_eAxC_num[UL_MAX_CELLS_PER_SLOT];           ///< Number of PUSCH eAxC IDs (antenna streams) per cell
    uint8_t*              pusch_buffer[UL_MAX_CELLS_PER_SLOT];             ///< PUSCH output buffer addresses
    int                   pusch_prb_x_slot[UL_MAX_CELLS_PER_SLOT];         ///< Total PUSCH PRBs per slot
    int                   pusch_prb_x_symbol[UL_MAX_CELLS_PER_SLOT];       ///< PUSCH PRBs per symbol
    int                   pusch_prb_x_symbol_x_antenna[UL_MAX_CELLS_PER_SLOT]; ///< PUSCH PRBs per symbol per antenna
    uint32_t              pusch_prb_stride[UL_MAX_CELLS_PER_SLOT];         ///< PUSCH buffer stride (bytes between PRBs)
    int                   prach_eAxC_num[UL_MAX_CELLS_PER_SLOT];           ///< Number of PRACH eAxC IDs per cell
    uint8_t * prach_buffer_0[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 0 (GPU memory)
    uint8_t * prach_buffer_1[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 1 (GPU memory)
    uint8_t * prach_buffer_2[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 2 (GPU memory)
    uint8_t * prach_buffer_3[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 3 (GPU memory)
    int                   prach_prb_x_slot[UL_MAX_CELLS_PER_SLOT];         ///< Total PRACH PRBs per slot
    int                   prach_prb_x_symbol[UL_MAX_CELLS_PER_SLOT];       ///< PRACH PRBs per symbol
    int                   prach_prb_x_symbol_x_antenna[UL_MAX_CELLS_PER_SLOT]; ///< PRACH PRBs per symbol per antenna
    uint32_t              prach_prb_stride[UL_MAX_CELLS_PER_SLOT];         ///< PRACH buffer stride (bytes between PRBs)
    int                   srs_eAxC_num[UL_MAX_CELLS_PER_SLOT];             ///< Number of SRS eAxC IDs per cell
    uint8_t*              srs_buffer[UL_MAX_CELLS_PER_SLOT];               ///< SRS output buffer addresses (GPU memory)
    int                   srs_prb_x_slot[UL_MAX_CELLS_PER_SLOT];           ///< Total SRS PRBs per slot
    uint32_t              srs_prb_stride[UL_MAX_CELLS_PER_SLOT];           ///< SRS buffer stride (bytes between PRBs)

    // Order kernel two-buffer (OK TB) specific parameters for double-buffering
    uint8_t*              fh_buf_ok_tb_slot[UL_MAX_CELLS_PER_SLOT];        ///< Fronthaul buffer for current slot (order kernel test bench)
    uint8_t*              fh_buf_ok_tb_next_slot[UL_MAX_CELLS_PER_SLOT];   ///< Fronthaul buffer for next slot (order kernel test bench)
    uint8_t*              pcap_buffer[UL_MAX_CELLS_PER_SLOT];              ///< PCAP packet capture buffer (debug/analysis)
    uint8_t*              pcap_buffer_ts[UL_MAX_CELLS_PER_SLOT];           ///< PCAP timestamp buffer (debug/analysis)
    uint32_t*             pcap_buffer_index[UL_MAX_CELLS_PER_SLOT];        ///< PCAP buffer write index (atomic)
    
    // GDR (GPU Direct RDMA) specific parameters for zero-copy CPU-GPU communication
    uint32_t* start_cuphy_d[UL_MAX_CELLS_PER_SLOT];                        ///< GDR flag: GPU sets when ordering complete, CPU reads to start cuPHY
    uint32_t* order_kernel_exit_cond_d[UL_MAX_CELLS_PER_SLOT];             ///< GDR flag: Order kernel exit condition (0=normal, >0=timeout/error)
    int* barrier_flag;                                                      ///< GDR barrier flag for multi-cell synchronization
    uint8_t* done_shared[UL_MAX_CELLS_PER_SLOT];                           ///< GPU flag to detect arrival of packets from the next slot(0: next slot packets received, 1: no next slot packets received)
    uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: Packets arrived before T_A4 min window
    uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];                   ///< GDR counter: Packets arrived within T_A4 window
    uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];                      ///< GDR counter: Packets arrived after T_A4 max window
    uint32_t* next_slot_early_rx_packets[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: Early packets for next slot (lookahead)
    uint32_t* next_slot_on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];         ///< GDR counter: On-time packets for next slot
    uint32_t* next_slot_late_rx_packets[UL_MAX_CELLS_PER_SLOT];            ///< GDR counter: Late packets for next slot
    uint64_t* rx_packets_ts[UL_MAX_CELLS_PER_SLOT];                        ///< GDR array: Receive timestamps for all packets (per-packet)
    uint32_t* rx_packets_count[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: Total packets received
    uint32_t* rx_bytes_count[UL_MAX_CELLS_PER_SLOT];                       ///< GDR counter: Total bytes received
    uint32_t* rx_packets_dropped_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR counter: Packets dropped
    uint64_t* rx_packets_ts_earliest[UL_MAX_CELLS_PER_SLOT];               ///< GDR timestamp: Earliest packet arrival time
    uint64_t* rx_packets_ts_latest[UL_MAX_CELLS_PER_SLOT];                 ///< GDR timestamp: Latest packet arrival time
    OKInstrumentation* oki_array[UL_MAX_CELLS_PER_SLOT];                   ///< GDR instrumentation: Order kernel profiling data (read/process phases)
    uint64_t* next_slot_rx_packets_ts[UL_MAX_CELLS_PER_SLOT];              ///< GDR array: Receive timestamps for next slot packets
    uint32_t* next_slot_rx_packets_count[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: Packet count for next slot
    uint32_t* next_slot_rx_bytes_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR counter: Byte count for next slot
	uint32_t* next_slot_num_prb_ch1[UL_MAX_CELLS_PER_SLOT];                ///< GDR counter: PRB count for PUSCH/PUCCH (next slot)
	uint32_t* next_slot_num_prb_ch2[UL_MAX_CELLS_PER_SLOT];                ///< GDR counter: PRB count for PRACH (next slot)
    
    // eAxC mapping and PRB ordering arrays
    uint16_t* pusch_eAxC_map[UL_MAX_CELLS_PER_SLOT];                       ///< GDR array: PUSCH eAxC ID mapping table
    uint32_t* pusch_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                   ///< GDR array: PUSCH ordered PRB indices
    uint16_t* prach_eAxC_map[UL_MAX_CELLS_PER_SLOT];                       ///< GDR array: PRACH eAxC ID mapping table
    uint32_t* prach_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                   ///< GDR array: PRACH ordered PRB indices
    uint16_t* srs_eAxC_map[UL_MAX_CELLS_PER_SLOT];                         ///< GDR array: SRS eAxC ID mapping table
    uint32_t* srs_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                     ///< GDR array: SRS ordered PRB indices
    uint8_t srs_start_sym[UL_MAX_CELLS_PER_SLOT];                          ///< SRS starting symbol index within slot

    uint32_t* last_sem_idx_rx_h[UL_MAX_CELLS_PER_SLOT];                    ///< GDR: Last semaphore index from RX
    uint32_t* last_sem_idx_order_h[UL_MAX_CELLS_PER_SLOT];                 ///< GDR: Last semaphore index from ordering
    uint64_t* order_kernel_last_timeout_error_time[UL_MAX_CELLS_PER_SLOT]; ///< GDR timestamp: Last timeout error occurrence

    // Sub-slot (per-symbol) processing specific parameters
    uint32_t* pusch_prb_symbol_map_d;                                       ///< GDR array: PUSCH PRB allocation per symbol (symbol-level granularity)
    uint32_t* sym_ord_done_sig_arr;                                         ///< GDR array: Per-symbol ordering completion signals
    uint32_t* sym_ord_done_mask_arr;                                        ///< GDR array: Per-symbol ordering completion masks
    uint32_t* num_order_cells_sym_mask_arr;                                 ///< GDR array: Number of cells to order per symbol
}orderKernelConfigParams_t;

/**
 * @brief Configuration parameters for GPU-based SRS-only packet ordering kernel
 *
 * Specialized variant of order kernel configuration that handles only SRS (Sounding Reference Signal)
 * channel packets. Uses separate timing windows and includes per-symbol packet counting for SRS
 * symbol-level processing. Launched on separate CUDA stream from main PUSCH/PRACH ordering.
 */
typedef struct orderKernelConfigParamsSrs{
    struct doca_gpu_eth_rxq *rxq_info_gpu[UL_MAX_CELLS_PER_SLOT];          ///< DOCA GPU receive queue handles for direct GPU packet access
    struct doca_gpu_semaphore_gpu *sem_gpu[UL_MAX_CELLS_PER_SLOT];         ///< DOCA GPU semaphores for RX/order kernel synchronization
    struct aerial_fh_gpu_semaphore_gpu *sem_gpu_aerial_fh[UL_MAX_CELLS_PER_SLOT]; ///< Aerial FH GPU semaphores for RX/order kernel synchronization
    uint16_t sem_order_num[UL_MAX_CELLS_PER_SLOT];                         ///< Semaphore value to signal after ordering complete
    int                   cell_id[UL_MAX_CELLS_PER_SLOT];                  ///< Physical cell IDs
    int                   comp_meth[UL_MAX_CELLS_PER_SLOT];                ///< Compression method
    int                   bit_width[UL_MAX_CELLS_PER_SLOT];                ///< IQ sample bit width for compressed data
    int                   ru_type[UL_MAX_CELLS_PER_SLOT];                  ///< RU type identifier
    bool                  cell_health[UL_MAX_CELLS_PER_SLOT];              ///< Cell health status (true=healthy, false=unhealthy/skip)
    float                 beta[UL_MAX_CELLS_PER_SLOT];                     ///< BFP compression beta parameter
    uint64_t              slot_start[UL_MAX_CELLS_PER_SLOT];               ///< Slot start time (nanoseconds since epoch)
    uint64_t              ta4_min_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 min timing constraint for SRS (earliest expected packet)
    uint64_t              ta4_max_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 max timing constraint for SRS (latest acceptable packet)
    uint64_t              slot_duration[UL_MAX_CELLS_PER_SLOT];            ///< Slot duration (nanoseconds)
    int                   srs_eAxC_num[UL_MAX_CELLS_PER_SLOT];             ///< Number of SRS eAxC IDs per cell
    uint8_t*              srs_buffer[UL_MAX_CELLS_PER_SLOT];               ///< SRS output buffer addresses (GPU memory)
    int                   srs_prb_x_slot[UL_MAX_CELLS_PER_SLOT];           ///< Total SRS PRBs per slot
    uint32_t              srs_prb_stride[UL_MAX_CELLS_PER_SLOT];           ///< SRS buffer stride (bytes between PRBs)

    // Order kernel two-buffer (OK TB) specific parameters for double-buffering
    uint8_t*              fh_buf_ok_tb_slot[UL_MAX_CELLS_PER_SLOT];        ///< Fronthaul buffer for current slot (order kernel test bench)
    uint8_t*              fh_buf_ok_tb_next_slot[UL_MAX_CELLS_PER_SLOT];   ///< Fronthaul buffer for next slot (order kernel test bench)

    // GDR (GPU Direct RDMA) specific parameters for zero-copy CPU-GPU communication
    uint32_t* start_cuphy_d[UL_MAX_CELLS_PER_SLOT];                        ///< GDR flag: GPU sets when SRS ordering complete, CPU reads to start cuPHY
    uint32_t* order_kernel_exit_cond_d[UL_MAX_CELLS_PER_SLOT];             ///< GDR flag: Order kernel exit condition
    int* barrier_flag;                                                      ///< GDR barrier flag for multi-cell synchronization
    uint8_t* done_shared[UL_MAX_CELLS_PER_SLOT];                           ///< GPU flag to detect arrival of packets from the next slot(0: next slot packets received, 1: no next slot packets received)
    uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: SRS packets arrived before T_A4 min window
    uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];                   ///< GDR counter: SRS packets arrived within T_A4 window
    uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];                      ///< GDR counter: SRS packets arrived after T_A4 max window
    uint32_t* next_slot_early_rx_packets[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: Early SRS packets for next slot
    uint32_t* next_slot_on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];         ///< GDR counter: On-time SRS packets for next slot
    uint32_t* next_slot_late_rx_packets[UL_MAX_CELLS_PER_SLOT];            ///< GDR counter: Late SRS packets for next slot
    uint64_t* rx_packets_ts[UL_MAX_CELLS_PER_SLOT];                        ///< GDR array: Receive timestamps for all SRS packets
    uint32_t* rx_packets_count[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: Total SRS packets received
    uint32_t* rx_packets_per_sym_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR array: SRS packet count per symbol (for multi-symbol SRS)
    uint32_t* rx_bytes_count[UL_MAX_CELLS_PER_SLOT];                       ///< GDR counter: Total SRS bytes received
    uint32_t* rx_packets_dropped_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR counter: SRS packets dropped (overflow/errors)
    uint64_t* rx_packets_ts_earliest[UL_MAX_CELLS_PER_SLOT];               ///< GDR timestamp: Earliest SRS packet arrival time
    uint64_t* rx_packets_ts_latest[UL_MAX_CELLS_PER_SLOT];                 ///< GDR timestamp: Latest SRS packet arrival time
    uint64_t* next_slot_rx_packets_ts[UL_MAX_CELLS_PER_SLOT];              ///< GDR array: Receive timestamps for next slot SRS packets
    uint32_t* next_slot_rx_packets_count[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: SRS packet count for next slot
    uint32_t* next_slot_rx_packets_per_sym_count[UL_MAX_CELLS_PER_SLOT];   ///< GDR array: SRS packet count per symbol for next slot
    uint32_t* next_slot_rx_bytes_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR counter: SRS byte count for next slot
    uint16_t* srs_eAxC_map[UL_MAX_CELLS_PER_SLOT];                         ///< GDR array: SRS eAxC ID mapping table
    uint32_t* srs_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                     ///< GDR array: SRS ordered PRB indices
    uint8_t srs_start_sym[UL_MAX_CELLS_PER_SLOT];                          ///< SRS starting symbol index within slot
    uint32_t* next_slot_num_prb_ch1[UL_MAX_CELLS_PER_SLOT];                ///< GDR counter: PRB count for channel 1 (next slot)

    uint32_t* last_sem_idx_rx_h[UL_MAX_CELLS_PER_SLOT];                    ///< GDR: Last semaphore index from RX
    uint32_t* last_sem_idx_order_h[UL_MAX_CELLS_PER_SLOT];                 ///< GDR: Last semaphore index from order kernel
    uint64_t* order_kernel_last_timeout_error_time[UL_MAX_CELLS_PER_SLOT]; ///< GDR timestamp: Last timeout error occurrence
}orderKernelConfigParamsSrs_t;

/**
 * @brief Configuration parameters for CPU-initiated UL ordering (non-DOCA mode)
 *
 * Alternative order kernel configuration for CPU-based packet processing mode, used when
 * DOCA GPUDirect is not available or CPU ordering is preferred. CPU reads packets from
 * NIC, initiates ordering, and uses ready_list/rx_queue_sync for CPU-GPU coordination.
 * Supports PUSCH and PRACH channels only (SRS handled separately).
 */
typedef struct orderKernelConfigParamsCpuInitComms{

    int                   comp_meth[UL_MAX_CELLS_PER_SLOT];                ///< Compression method
    int                   bit_width[UL_MAX_CELLS_PER_SLOT];                ///< IQ sample bit width for compressed data
    float                 beta[UL_MAX_CELLS_PER_SLOT];                     ///< BFP compression beta parameter
    uint16_t              sem_order_num[UL_MAX_CELLS_PER_SLOT];            ///< Semaphore value to signal after ordering complete
    
    uint64_t              slot_start[UL_MAX_CELLS_PER_SLOT];               ///< Slot start time (nanoseconds since epoch)
    uint64_t              ta4_min_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 min timing constraint (earliest expected packet arrival)
    uint64_t              ta4_max_ns[UL_MAX_CELLS_PER_SLOT];               ///< T_A4 max timing constraint (latest acceptable packet arrival)
    uint64_t              slot_duration[UL_MAX_CELLS_PER_SLOT];            ///< Slot duration (nanoseconds)
        
    int                   pusch_eAxC_num[UL_MAX_CELLS_PER_SLOT];           ///< Number of PUSCH eAxC IDs (antenna streams) per cell
    uint8_t*              pusch_buffer[UL_MAX_CELLS_PER_SLOT];             ///< PUSCH output buffer addresses (GPU memory)
    int                   pusch_prb_x_slot[UL_MAX_CELLS_PER_SLOT];         ///< Total PUSCH PRBs per slot
    int                   pusch_prb_x_symbol[UL_MAX_CELLS_PER_SLOT];       ///< PUSCH PRBs per symbol
    int                   pusch_prb_x_symbol_x_antenna[UL_MAX_CELLS_PER_SLOT]; ///< PUSCH PRBs per symbol per antenna
    uint32_t              pusch_prb_stride[UL_MAX_CELLS_PER_SLOT];         ///< PUSCH buffer stride (bytes between PRBs)
    
    int                   prach_eAxC_num[UL_MAX_CELLS_PER_SLOT];           ///< Number of PRACH eAxC IDs per cell
    uint8_t * prach_buffer_0[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 0 (GPU memory)
    uint8_t * prach_buffer_1[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 1 (GPU memory)
    uint8_t * prach_buffer_2[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 2 (GPU memory)
    uint8_t * prach_buffer_3[UL_MAX_CELLS_PER_SLOT];                       ///< PRACH output buffer for occasion 3 (GPU memory)
    int                   prach_prb_x_slot[UL_MAX_CELLS_PER_SLOT];         ///< Total PRACH PRBs per slot
    int                   prach_prb_x_symbol[UL_MAX_CELLS_PER_SLOT];       ///< PRACH PRBs per symbol
    int                   prach_prb_x_symbol_x_antenna[UL_MAX_CELLS_PER_SLOT]; ///< PRACH PRBs per symbol per antenna
    uint32_t              prach_prb_stride[UL_MAX_CELLS_PER_SLOT];         ///< PRACH buffer stride (bytes between PRBs)

    // GDR (GPU Direct RDMA) specific parameters for zero-copy CPU-GPU communication
    uint32_t* start_cuphy_d[UL_MAX_CELLS_PER_SLOT];                        ///< GDR flag: GPU sets when ordering complete, CPU reads to start cuPHY
    uint32_t* order_kernel_exit_cond_d[UL_MAX_CELLS_PER_SLOT];             ///< GDR flag: Order kernel exit condition
    uint32_t* ready_list[UL_MAX_CELLS_PER_SLOT];                           ///< GDR array: Ready list for CPU→GPU packet availability signaling
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list[UL_MAX_CELLS_PER_SLOT]; ///< GDR array: RX queue sync structures for CPU-GPU coordination
    uint32_t* last_sem_idx_order_h[UL_MAX_CELLS_PER_SLOT];                 ///< GDR: Last semaphore index from order kernel
    int* barrier_flag;                                                      ///< GDR barrier flag for multi-cell synchronization
    uint8_t* done_shared[UL_MAX_CELLS_PER_SLOT];                           ///< GPU flag to detect arrival of packets from the next slot(0: next slot packets received, 1: no next slot packets received)

    uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: Packets arrived before T_A4 min window
    uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];                   ///< GDR counter: Packets arrived within T_A4 window
    uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];                      ///< GDR counter: Packets arrived after T_A4 max window
    uint32_t* next_slot_early_rx_packets[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: Early packets for next slot
    uint32_t* next_slot_on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];         ///< GDR counter: On-time packets for next slot
    uint32_t* next_slot_late_rx_packets[UL_MAX_CELLS_PER_SLOT];            ///< GDR counter: Late packets for next slot
    uint64_t* rx_packets_ts[UL_MAX_CELLS_PER_SLOT];                        ///< GDR array: Receive timestamps for all packets
    uint32_t* rx_packets_count[UL_MAX_CELLS_PER_SLOT];                     ///< GDR counter: Total packets received
    uint32_t* rx_bytes_count[UL_MAX_CELLS_PER_SLOT];                       ///< GDR counter: Total bytes received
    uint64_t* rx_packets_ts_earliest[UL_MAX_CELLS_PER_SLOT];               ///< GDR timestamp: Earliest packet arrival time
    uint64_t* rx_packets_ts_latest[UL_MAX_CELLS_PER_SLOT];                 ///< GDR timestamp: Latest packet arrival time
    uint64_t* next_slot_rx_packets_ts[UL_MAX_CELLS_PER_SLOT];              ///< GDR array: Receive timestamps for next slot packets
    uint32_t* next_slot_rx_packets_count[UL_MAX_CELLS_PER_SLOT];           ///< GDR counter: Packet count for next slot
    uint32_t* next_slot_rx_bytes_count[UL_MAX_CELLS_PER_SLOT];             ///< GDR counter: Byte count for next slot
     
    uint16_t* pusch_eAxC_map[UL_MAX_CELLS_PER_SLOT];                       ///< GDR array: PUSCH eAxC ID mapping table
    uint32_t* pusch_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                   ///< GDR array: PUSCH ordered PRB indices
    uint16_t* prach_eAxC_map[UL_MAX_CELLS_PER_SLOT];                       ///< GDR array: PRACH eAxC ID mapping table
    uint32_t* prach_ordered_prbs[UL_MAX_CELLS_PER_SLOT];                   ///< GDR array: PRACH ordered PRB indices

    // Sub-slot (per-symbol) processing specific parameters
    uint32_t* pusch_prb_symbol_map_d;                                       ///< GDR array: PUSCH PRB allocation per symbol (symbol-level granularity)
    uint32_t* sym_ord_done_sig_arr;                                         ///< GDR array: Per-symbol ordering completion signals
    uint32_t* sym_ord_done_mask_arr;                                        ///< GDR array: Per-symbol ordering completion masks
    uint32_t* num_order_cells_sym_mask_arr;                                 ///< GDR array: Number of cells to order per symbol
}orderKernelConfigParamsCpuInitComms_t;


/**
 * @brief UL packet ordering entity for GPU-based ORAN fronthaul processing
 *
 * Manages GPU-based ordering of uplink ORAN packets (PUSCH, PUCCH, PRACH, SRS) from fronthaul
 * receive queues into cuPHY input buffers. Allocates GDR buffers for zero-copy CPU-GPU
 * synchronization, configures and launches order kernels, and provides statistics on
 * packet timing and counts. Supports both DOCA GPUDirect and CPU-initiated modes.
 * One OrderEntity instance can manage multiple cells concurrently.
 */
class OrderEntity {
public:
    /**
     * @brief Construct order entity
     *
     * @param _pdh  - Cuphydriver handle
     * @param _gDev - GPU device structure pointer
     */
    OrderEntity(phydriver_handle _pdh, GpuDevice* _gDev);
    
    /**
     * @brief Destroy order entity and free all allocated resources
     */
    ~OrderEntity();
    
    /**
     * @brief Get cuphydriver handle
     *
     * @return Cuphydriver handle associated with this entity
     */
    phydriver_handle    getPhyDriverHandler(void) const;
    
    /**
     * @brief Get unique entity identifier
     *
     * @return Unique ID for this order entity (timestamp-based)
     */
    uint64_t            getId() const;
    
    /**
     * @brief Get cell ID
     *
     * @return Cell ID of first cell in the order entity
     */
    cell_id_t           getCellId();
    
    /**
     * @brief Reserve order entity for specified cells
     *
     * Allocates GDR buffers, initializes eAxC mappings, and configures entity for
     * multi-cell ordering. Must be called before running order kernel.
     *
     * @param cell_idx_list      - Array of cell indices to reserve
     * @param cell_idx_list_size - Number of cells in the list
     * @param new_order_entity   - True if this is a new entity, false for reuse
     * @return 0 on success, error code on failure
     */
    int                 reserve(int32_t* cell_idx_list, uint8_t cell_idx_list_size, bool new_order_entity);
    
    /**
     * @brief Release order entity
     *
     * Releases resources and makes entity available for reuse. Does not free
     * GDR buffers (use cleanup() for complete teardown).
     */
    void                release();
    
    /**
     * @brief Clean up all resources
     *
     * Frees all GDR buffers and GPU memory. Called during destruction or
     * reconfiguration.
     */
    void                cleanup();
    
    /**
     * @brief Flush GPU memory to prepare for next slot
     *
     * Resets GPU buffers and prepares for next ordering operation.
     *
     * @return 0 on success, error code on failure
     */
    int                 flushMemory();
    
    /**
     * @brief CPU side signaling function to control the order kernel on the GPU.
     *
     * Sets writes start_type to the order_kernel_exit_cond_gdr[cell_idx] GDR buffer to
     * indicate order kernel to start or stop ordering.
     *
     * @param cell_idx   - Cell index within entity
     * @param start_type - Start type
     */
    void                enableOrder(int cell_idx,int start_type);
    
    /**
     * @brief Get GDR flag address for GPU stream wait
     *
     * @param cell_idx - Cell index
     * @return Pointer to GDR flag for ordering completion
     */
    uint32_t*           getWaitSingleOrderGpuFlag(int cell_idx);
    
    /**
     * @brief Check order completion on CPU (polling)
     *
     * Polls GDR flag to check if order kernel has completed.
     *
     * @param isSrs - True for SRS order kernel, false for PUSCH/PUCCH/PRACH
     * @return 1 if complete, 0 if not complete
     */
    int                 checkOrderCPU(bool isSrs);
    
    /**
     * @brief Wait for order completion on CPU with timeout
     *
     * Busy-waits on GDR flag with timeout.
     *
     * @param wait_ns - Timeout in nanoseconds
     * @return 0 on success, error code on timeout
     */
    int                 waitOrder(int wait_ns);
    
    /**
     * @brief Set order complete flag from CPU
     *
     * Manually sets GDR completion flag (for CPU-initiated mode).
     */
    void                setOrderCPU();
    
    /**
     * @brief Get order launched status for PUSCH/PUCCH/PRACH
     *
     * @return True if order kernel has been launched, false otherwise
     */
    bool                getOrderLaunchedStatus(); 
    
    /**
     * @brief Set order launched status for PUSCH/PUCCH/PRACH
     *
     * @param val - True to mark as launched, false otherwise
     */
    void                setOrderLaunchedStatus(bool val);
    
    /**
     * @brief Get order launched status for SRS
     *
     * @return True if SRS order kernel has been launched, false otherwise
     */
    bool                getOrderLaunchedStatusSrs(); 
    
    /**
     * @brief Set order launched status for SRS
     *
     * @param val - True to mark as launched, false otherwise
     */
    void                setOrderLaunchedStatusSrs(bool val);
    
    /**
     * @brief Wait for order kernel launch (PUSCH/PUCCH/PRACH) with timeout
     *
     * Waits for order_launched flag to be set, indicating kernel has been launched.
     *
     * @param wait_ns - Timeout in nanoseconds
     * @return 0 on success, error code on timeout
     */
    int                 waitOrderLaunched(int wait_ns);
    
    /**
     * @brief Wait for order kernel launch (SRS) with timeout
     *
     * Waits for order_launched_srs flag to be set, indicating SRS kernel has been launched.
     *
     * @param wait_ns - Timeout in nanoseconds
     * @return 0 on success, error code on timeout
     */
    int                 waitOrderLaunchedSrs(int wait_ns);
    
    /**
     * @brief Launch order kernel to process uplink packets
     *
     * Main entry point to execute GPU order kernel. Configures kernel parameters,
     * launches ordering on GPU, and records timing events. Handles PUSCH, PUCCH, PRACH,
     * and SRS channels with per-cell timing validation and per-symbol ordering support.
     *
     * @param oran_ind                  - ORAN slot indication with timing info
     * @param puschNumPrb               - PUSCH PRB counts per cell
     * @param buf_st_1                  - PUSCH (PUSCH/PUCCH) buffer pointers per cell
     * @param buf_pcap_capture          - PCAP capture buffers (optional, for debug)
     * @param buf_pcap_capture_ts       - PCAP timestamp buffers (optional, for debug)
     * @param prachNumPrb               - PRACH PRB counts per cell
     * @param buf_st_3_o0               - PRACH occasion 0 buffers per cell
     * @param buf_st_3_o1               - PRACH occasion 1 buffers per cell
     * @param buf_st_3_o2               - PRACH occasion 2 buffers per cell
     * @param buf_st_3_o3               - PRACH occasion 3 buffers per cell
     * @param prachSectionId_o0         - PRACH section ID for occasion 0
     * @param prachSectionId_o1         - PRACH section ID for occasion 1
     * @param prachSectionId_o2         - PRACH section ID for occasion 2
     * @param prachSectionId_o3         - PRACH section ID for occasion 3
     * @param srsNumPrb                 - SRS PRB counts per cell
     * @param buf_srs                   - SRS buffer pointers per cell
     * @param slot_start                - Slot start times per cell (nanoseconds)
     * @param ta4_min_ns                - T_A4 min constraints for PUSCH/PUCCH/PRACH per cell
     * @param ta4_max_ns                - T_A4 max constraints for PUSCH/PUCCH/PRACH per cell
     * @param ta4_min_ns_srs            - T_A4 min constraints for SRS per cell
     * @param ta4_max_ns_srs            - T_A4 max constraints for SRS per cell
     * @param num_order_cells           - Number of cells to order
     * @param srsCellMask               - Bitmask of cells with SRS enabled
     * @param srs_start_symbol          - SRS starting symbol indices per cell
     * @param nonSrsUlCellMask          - Bitmask of cells with PUSCH/PRACH only
     * @param sym_ord_arr_addr          - Symbol ordering array address (sub-slot mode)
     * @param pusch_prb_symbol_map      - PUSCH PRB allocation per symbol per cell
     * @param num_order_cells_sym_mask  - Number of cells to order per symbol
     * @param pusch_prb_non_zero        - Flag indicating non-zero PUSCH PRBs
     * @param slot_map_id               - Slot mapping ID for buffer rotation
     * @return 0 on success, error code on failure
     */
    int                 runOrder(
    slot_command_api::oran_slot_ind oran_ind,
    uint16_t* puschNumPrb, uint8_t ** buf_st_1, uint8_t** buf_pcap_capture,  uint8_t** buf_pcap_capture_ts,
    uint16_t* prachNumPrb, 
    uint8_t ** buf_st_3_o0, uint8_t ** buf_st_3_o1, uint8_t ** buf_st_3_o2, uint8_t ** buf_st_3_o3,
    uint16_t prachSectionId_o0, uint16_t prachSectionId_o1, uint16_t prachSectionId_o2, uint16_t prachSectionId_o3,
    uint32_t* srsNumPrb, uint8_t **buf_srs,
    std::array<t_ns,UL_MAX_CELLS_PER_SLOT>& slot_start, 
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_min_ns, std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_max_ns,
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_min_ns_srs, std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_max_ns_srs,
    uint8_t num_order_cells,uint32_t srsCellMask,std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& srs_start_symbol,
    uint32_t nonSrsUlCellMask,uint32_t* sym_ord_arr_addr,
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT>& pusch_prb_symbol_map,
    std::array<uint32_t,ORAN_PUSCH_SYMBOLS_X_SLOT>& num_order_cells_sym_mask,uint8_t pusch_prb_non_zero,uint32_t slot_map_id);
    
    /**
     * @brief Get GPU idle time for PUSCH/PUCCH/PRACH ordering (milliseconds)
     *
     * @return Time between previous order completion and current order start (ms)
     */
    float               getGPUIdleTime();
    
    /**
     * @brief Get GPU order execution time for PUSCH/PUCCH/PRACH (milliseconds)
     *
     * @return Time from order start to completion (ms)
     */
    float               getGPUOrderTime();
    
    /**
     * @brief Get GPU idle time for SRS ordering (milliseconds)
     *
     * @return Time between previous SRS order completion and current SRS order start (ms)
     */
    float               getGPUIdleTimeSrs();
    
    /**
     * @brief Get GPU order execution time for SRS (milliseconds)
     *
     * @return Time from SRS order start to completion (ms)
     */
    float               getGPUOrderTimeSrs();
    
    /**
     * @brief Get CUDA event for PUSCH/PUCCH/PRACH order completion
     *
     * @return CUDA event recorded when order kernel completes
     */
    cudaEvent_t         getRunCompletionEvt(void) const { return end_order; };
    
    /**
     * @brief Get CUDA event for SRS order completion
     *
     * @return CUDA event recorded when SRS order kernel completes
     */
    cudaEvent_t         getSrsRunCompletionEvt(void) const { return end_order_srs; };

    // Statistics getters for PUSCH/PRACH ordering
    uint32_t            getEarlyRxPackets(int cell_idx);                     ///< Get early packet count (before T_A4 min) for cell
    uint32_t            getOnTimeRxPackets(int cell_idx);                    ///< Get on-time packet count (within T_A4 window) for cell
    uint32_t            getLateRxPackets(int cell_idx);                      ///< Get late packet count (after T_A4 max) for cell
    uint32_t            getRxPacketCount(int cell_idx,int sym_idx);         ///< Get total packet count for cell and symbol
    uint32_t            getRxByteCount(int cell_idx,int sym_idx);           ///< Get total byte count for cell and symbol
    uint32_t            getRxPacketsDroppedCount(int cell_idx);             ///< Get dropped packet count for cell
    uint64_t            getRxPacketTs(int cell_idx,int sym_idx,int pkt_idx); ///< Get timestamp for specific packet
    uint64_t            getRxPacketTsEarliest(int cell_idx,int sym_idx);    ///< Get earliest packet timestamp for cell and symbol
    uint64_t            getRxPacketTsLatest(int cell_idx,int sym_idx);      ///< Get latest packet timestamp for cell and symbol
    OKInstrumentation   getOKInstrumentation(int cell_idx);                  ///< Get order kernel instrumentation data for cell
    uint32_t            getOrderExitCondition(int cell_idx);                 ///< Get order kernel exit condition
    
    // Statistics getters for SRS ordering
    uint32_t            getOrderSrsExitCondition(int cell_idx);              ///< Get SRS order kernel exit condition
    uint32_t            getEarlyRxPacketsSRS(int cell_idx);                  ///< Get early SRS packet count for cell
    uint32_t            getOnTimeRxPacketsSRS(int cell_idx);                 ///< Get on-time SRS packet count for cell
    uint32_t            getLateRxPacketsSRS(int cell_idx);                   ///< Get late SRS packet count for cell
    uint32_t            getRxPacketCountSRS(int cell_idx);                   ///< Get total SRS packet count for cell
    uint32_t            getRxByteCountSRS(int cell_idx);                     ///< Get total SRS byte count for cell
    uint32_t            getRxPacketsDroppedCountSRS(int cell_idx);           ///< Get dropped SRS packet count for cell
    uint32_t            getRxPacketCountPerSymSRS(int cell_idx,int sym_idx); ///< Get SRS packet count per symbol for cell
    uint64_t            getRxPacketTsSRS(int cell_idx,int sym_idx,int pkt_idx); ///< Get timestamp for specific SRS packet
    uint64_t            getRxPacketTsEarliestSRS(int cell_idx,int sym_idx); ///< Get earliest SRS packet timestamp for cell and symbol
    uint64_t            getRxPacketTsLatestSRS(int cell_idx,int sym_idx);   ///< Get latest SRS packet timestamp for cell and symbol
    
    MemFoot             mf;  ///< Memory footprint tracker for this entity

protected:
    phydriver_handle                        pdh;                                ///< cuphydriver handle
    uint64_t                                id;                                 ///< Unique entity identifier (timestamp-based)
    void*                                   addr_d;                             ///< GPU device memory address (generic pointer)
    void*                                   addr_h;                             ///< Host memory address (generic pointer)
    std::atomic<bool>                       active;                             ///< Entity active flag (true=in use, false=available)
    Mutex                                   mlock;                              ///< Mutex for thread-safe access to entity
    std::array<cell_id_t,UL_MAX_CELLS_PER_SLOT>  cell_id;                      ///< Physical cell IDs for each slot position
    GpuDevice*                              gDev;                               ///< GPU device structure pointer
    std::array<int,UL_MAX_CELLS_PER_SLOT>                                     rx_pkts;            ///< Receive packet count per cell (current slot)
    std::array<int,UL_MAX_CELLS_PER_SLOT>                                     tot_rx_pkts;        ///< Total receive packet count per cell (accumulated)
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> flush_gmem;               ///< GDR flush buffers per cell (for memory reset)
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        order_kernel_exit_cond_gdr;      ///< GDR: PUSCH/PUCCH/PRACH order kernel exit conditions
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        start_cuphy_gdr;                 ///< GDR: PUSCH/PUCCH/PRACH ordering complete flags (GPU→CPU)
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        order_kernel_srs_exit_cond_gdr;  ///< GDR: SRS order kernel exit conditions
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        start_cuphy_srs_gdr;             ///< GDR: SRS ordering complete flags (GPU→CPU)
    std::unique_ptr<host_buf>              start_cuphy_cpu_h;                  ///< Host buffer: PUSCH/PUCCH/PRACH start flags (CPU-initiated mode)
    std::unique_ptr<host_buf>              start_cuphy_srs_cpu_h;              ///< Host buffer: SRS start flags (CPU-initiated mode)

    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> ordered_prbs_prach;       ///< GDR: PRACH ordered PRB arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> ordered_prbs_pusch;       ///< GDR: PUSCH ordered PRB arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> ordered_prbs_srs;         ///< GDR: SRS ordered PRB arrays per cell
    std::unique_ptr<gpinned_buffer>        order_barrier_flag;                 ///< GDR: Barrier flag for multi-cell synchronization
    std::array<std::unique_ptr<dev_buf>,UL_MAX_CELLS_PER_SLOT>               done_shared;       ///< GPU flag to detect arrival of packets from the next slot(0: next slot packets received, 1: no next slot packets received)
    std::array<std::unique_ptr<dev_buf>,UL_MAX_CELLS_PER_SLOT>               done_shared_srs;   ///< GPU flag to detect arrival of packets from the next slot(0: next slot packets received, 1: no next slot packets received)
    std::unique_ptr<dev_buf>               ready_shared;                       ///< GPU memory: Ready flag for CPU-GPU coordination
    std::unique_ptr<dev_buf>               rx_queue_index;                     ///< GPU memory: RX queue index tracking
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        eAxC_map_gdr;      ///< GDR: PUSCH eAxC ID mapping tables per cell
    std::array<int,UL_MAX_CELLS_PER_SLOT>                                    eAxC_num;          ///< Number of PUSCH eAxC IDs per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        eAxC_prach_map_gdr; ///< GDR: PRACH eAxC ID mapping tables per cell
    std::array<int,UL_MAX_CELLS_PER_SLOT>                                     eAxC_prach_num;    ///< Number of PRACH eAxC IDs per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT>        eAxC_srs_map_gdr;  ///< GDR: SRS eAxC ID mapping tables per cell
    std::array<int,UL_MAX_CELLS_PER_SLOT>                                     eAxC_srs_num;      ///< Number of SRS eAxC IDs per cell
    int                                    cell_num;                           ///< Number of active cells in this entity

    t_ns reserve_t;                                                            ///< Timestamp when entity was last reserved

    cudaEvent_t start_idle;                                                    ///< CUDA event: Previous PUSCH/PUCCH/P  RACH order completion (for idle time)
    cudaEvent_t start_order;                                                   ///< CUDA event: Current PUSCH/PUCCH/PRACH order start
    cudaEvent_t end_order;                                                     ///< CUDA event: Current PUSCH/PUCCH/PRACH order completion

    cudaEvent_t start_idle_srs;                                                ///< CUDA event: Previous SRS order completion (for idle time)
    cudaEvent_t start_order_srs;                                               ///< CUDA event: Current SRS order start
    cudaEvent_t end_order_srs;                                                 ///< CUDA event: Current SRS order completion

    // Sub-slot (per-symbol) processing parameters
    std::unique_ptr<gpinned_buffer> pusch_prb_symbol_map_gdr;                  ///< GDR: PUSCH PRB allocation map per symbol
    std::unique_ptr<gpinned_buffer> sym_ord_done_mask_arr;                     ///< GDR: Symbol ordering completion mask array
    std::unique_ptr<gpinned_buffer> num_order_cells_sym_mask_arr;              ///< GDR: Number of cells to order per symbol array

    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> pusch_prbs_x_slot;              ///< PUSCH PRB count per slot per cell
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> prach_prbs_x_slot;              ///< PRACH PRB count per slot per cell
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> srs_prbs_x_slot;                ///< SRS PRB count per slot per cell

    // PUSCH/PRACH packet statistics (GDR buffers for zero-copy access)
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> early_rx_packets;         ///< GDR: Early packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> on_time_rx_packets;       ///< GDR: On-time packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> late_rx_packets;          ///< GDR: Late packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts;            ///< GDR: Packet timestamp arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_count;         ///< GDR: Packet count arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_bytes_count;           ///< GDR: Byte count arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_dropped_count; ///< GDR: Dropped packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts_earliest;   ///< GDR: Earliest packet timestamps per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts_latest;     ///< GDR: Latest packet timestamps per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> oki_array;                ///< GDR: Order kernel instrumentation arrays per cell
    
    // SRS packet statistics (GDR buffers for zero-copy access)
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> early_rx_packets_srs;         ///< GDR: SRS early packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> on_time_rx_packets_srs;       ///< GDR: SRS on-time packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> late_rx_packets_srs;          ///< GDR: SRS late packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_count_srs;         ///< GDR: SRS packet count arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_bytes_count_srs;           ///< GDR: SRS byte count arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_dropped_count_srs; ///< GDR: SRS dropped packet counters per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts_srs;            ///< GDR: SRS packet timestamp arrays per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_count_per_sym_srs; ///< GDR: SRS per-symbol packet counts per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts_earliest_srs;   ///< GDR: SRS earliest packet timestamps per cell
    std::array<std::unique_ptr<gpinned_buffer>,UL_MAX_CELLS_PER_SLOT> rx_packets_ts_latest_srs;     ///< GDR: SRS latest packet timestamps per cell

    std::atomic<bool> order_launched;                                          ///< Atomic flag: PUSCH/PRACH order kernel launched
    std::atomic<bool> order_launched_srs;                                      ///< Atomic flag: SRS order kernel launched
    int32_t cell_order_list[UL_MAX_CELLS_PER_SLOT];                            ///< List of cell indices assigned to this entity
    uint32_t cell_order_list_size;                                             ///< Number of cells in cell_order_list
    orderKernelConfigParams_t* order_kernel_config_params;                     ///< Configuration parameters for PUSCH/PRACH/SRS order kernel
    orderKernelConfigParamsSrs_t* order_kernel_config_params_srs;              ///< Configuration parameters for SRS-only order kernel
    orderKernelConfigParamsCpuInitComms_t* orderKernelConfigParamsCpuInitComms; ///< Configuration parameters for CPU-initiated order kernel
};

#endif
