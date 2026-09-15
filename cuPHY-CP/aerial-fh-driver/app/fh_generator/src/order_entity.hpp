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
#include <memory>
#include "utils.hpp"
#include "doca_utils.hpp"
#include "gpudevice.hpp"

#ifdef DOCA_GPU_DPDK
#include <doca_gpunetio.h>
#include <doca_log.h>
#endif

namespace fh_gen
{

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

typedef struct orderKernelConfigParams{
    struct doca_gpu_eth_rxq *rxq_info_gpu[kMaxCells];
    struct doca_gpu_semaphore_gpu *sem_gpu[kMaxCells];
    uint16_t sem_order_num[kMaxCells];
    int                   cell_id[kMaxCells];
    int                   bit_width[kMaxCells];
    int                   prb_x_slot[kMaxCells];
    float                 beta[kMaxCells];
    uint64_t              slot_start[kMaxCells];
    uint64_t              ta4_min_ns[kMaxCells];
    uint64_t              ta4_max_ns[kMaxCells];

    //GDR specific params
    uint32_t* order_kernel_exit_cond_d[kMaxCells];
    uint32_t* last_sem_idx_order_d[kMaxCells];
    uint32_t* early_rx_packets[kMaxCells];
    uint32_t* on_time_rx_packets[kMaxCells];
    uint32_t* late_rx_packets[kMaxCells];
    uint32_t* next_slot_early_rx_packets[kMaxCells];
    uint32_t* next_slot_on_time_rx_packets[kMaxCells];
    uint32_t* next_slot_late_rx_packets[kMaxCells];
    uint32_t* next_slot_num_prb[kMaxCells];

    uint64_t* rx_packets_ts[kMaxCells];
    uint32_t* rx_packets_count[kMaxCells];
    uint64_t* rx_packets_ts_earliest[kMaxCells];
    uint64_t* rx_packets_ts_latest[kMaxCells];
    uint64_t* next_slot_rx_packets_ts[kMaxCells];
    uint32_t* next_slot_rx_packets_count[kMaxCells];

    uint16_t frame_id;
    uint8_t subframe_id;
    uint8_t slot_id;
    uint16_t SFN;
    uint64_t slot_t0;
    uint64_t slot_duration;

    uint8_t num_cells;
    uint32_t* exit_flag_d;
} orderKernelConfigParams_t;


/*
 * Generic GPU memory buffer useful to store output from
 * different DL channels
 */

class OrderEntity {
public:
    OrderEntity(GpuDevice* _gDev, uint64_t _id);
    ~OrderEntity();
    uint64_t            getId() const;
    int                 runOrder(cudaStream_t stream);
    int                 reserve();
    void                release();
    void                cleanup();
    bool                isActive();
    uint32_t            getOrderExitCondition(int cell_idx);
    void                setOrderExitCondition(int cell_idx, uint32_t order_status);
    int                 checkOrderCPU();
    uint32_t            getEarlyRxPackets(int cell_idx);
    uint32_t            getOnTimeRxPackets(int cell_idx);
    uint32_t            getLateRxPackets(int cell_idx);
    uint64_t            getRxPacketTsEarliest(int cell_idx,int sym_idx);
    uint64_t            getRxPacketTsLatest(int cell_idx,int sym_idx);
    orderKernelConfigParams_t* order_kernel_config_params;
protected:
    GpuDevice*                              gDev;
    uint64_t                                id;
    std::atomic<bool>                       active;

    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> order_kernel_exit_cond_gdr;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> ordered_prbs;

    std::array<uint32_t,kMaxCells> prbs_x_slot;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> early_rx_packets;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> on_time_rx_packets;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> late_rx_packets;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> rx_packets_ts;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> rx_packets_count;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> rx_packets_ts_earliest;
    std::array<std::unique_ptr<gpinned_buffer>,kMaxCells> rx_packets_ts_latest;

    std::unique_ptr<host_buf> order_end;
    cudaEvent_t start_order;
    cudaEvent_t end_order;
};
#ifdef __cplusplus
extern "C" {
#endif
void launch_kernel_write(cudaStream_t stream, uint32_t* addr, uint32_t value);
doca_error_t kernel_receive_slot(cudaStream_t stream, orderKernelConfigParams_t* params);
/// Resolve CUfunction handles for fh_generator CUDA kernels via cudaGetFuncBySymbol.
/// Must be called once before any kernel launch; thread-safe via std::call_once.
/// @return true if all kernels resolved successfully, false otherwise.
[[nodiscard]] bool resolve_fhgen_cuda_kernels();
#ifdef __cplusplus
}
#endif

}
#endif
