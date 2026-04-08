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

#ifndef AERIAL_FH_GPU_COMM__
#define AERIAL_FH_GPU_COMM__

#include <doca_gpunetio.h>

#include "aerial-fh-driver/api.hpp"
#include "queue.hpp"
#include "dpdk.hpp"

#define MLX5_TX_COMP_MAX_CQE 32
#define CPU_TO_BE32(v) \
        ((((uint32_t)(v) & (UINT32_C(0x000000ff))) << 24) | (((uint32_t)(v) & (UINT32_C(0x0000ff00))) << 8) | \
         (((uint32_t)(v) & (UINT32_C(0x00ff0000))) >> 8) | (((uint32_t)(v) & (UINT32_C(0xff000000))) >> 24))

namespace aerial_fh
{
    /* CQE status. */
enum mlx5_cqe_status {
	MLX5_CQE_STATUS_SW_OWN = -1,
	MLX5_CQE_STATUS_HW_OWN = -2,
	MLX5_CQE_STATUS_ERR = -3,
};

class Gpu;
class Nic;
struct UplaneSlotInfo;
typedef struct UplaneSlotInfo UplaneSlotInfo_t;

struct PrepareCellParams {
    struct doca_gpu_eth_txq *eth_txq_gpu;
    UplaneSlotInfo_t *d_slot_info;
    UplaneSymbolInfoHost *h_slot_info;
    PartialUplaneSlotInfo_t *partial_slot_info;
    size_t mtu;
    size_t prb_size_upl;
    size_t prbs_per_pkt_upl;
    uint32_t* flow_d_info;
    uint32_t* flow_sym_d_info;
    uint32_t* block_count;
    FlowPtrInfo* flow_hdr_size_info;
    uint32_t* flow_d_ecpri_seq_id;
    uint32_t* flow_d_hdr_template_info;
    uint16_t max_num_prb_per_symbol;
    PacketStartDebugInfo *start_dbg_info;
};

struct FlowPtrInfo;

struct CleanupCellParams {
    UplaneSlotInfo_t *d_slot_info;
    uint4* d_buf_addr;
    size_t buf_size;
};

struct CleanupParams {
    CleanupCellParams cell_params[API_MAX_NUM_CELLS];
};

struct PrepareParams {
    PrepareCellParams cell_params[API_MAX_NUM_CELLS];
    uint32_t hdr_len;
    uint16_t ecpri_payload_overhead;
    uint8_t frame_id;
    uint8_t num_cells;
    uint16_t subframe_id;
    uint16_t slot_id;
    PreparePRBInfo payload_info;
    uint32_t* host_pinned_error_flag;
    uint8_t   enable_gpu_comm_via_cpu;
};

struct FlowPrepCellParams {
    uint8_t* pkt_hdr_gpu[MAX_DL_EAXCIDS];
    size_t   pkt_size_rnd[MAX_DL_EAXCIDS];
};

struct FlowPrepParams {
    FlowPrepCellParams cell_params[API_MAX_NUM_CELLS];
};

struct TriggerCellParams {
    struct doca_gpu_eth_txq *eth_txq_gpu;
    UplaneSymbolInfoHost *h_slot_info;
    UplaneSlotInfo_t *d_slot_info;
};

struct TriggerParams {
    TriggerCellParams cell_params[API_MAX_NUM_CELLS];
    uint32_t *ready_flag;
    uint32_t wait_val;
    uint8_t num_cells;
    bool    disable_empw;
};

struct PacketCopyCellParams {
    void *d_src_addr;
    void *h_dst_addr;
    int pkt_offset[kMaxFlows];
    uint32_t num_pkts;
    uint32_t num_pkts_per_flow[kMaxFlows];
    uint32_t num_flows;
};

struct PacketCopyParams {
    PacketCopyCellParams cell_params[API_MAX_NUM_CELLS];
    uint32_t pkt_size;
    uint32_t max_pkts;
    uint8_t frame_id;
    uint16_t subframe_id;
    uint16_t slot_id;
};

using num_packets_per_flow_t = std::array<uint32_t,kMaxFlows>;

void launch_memset_kernel(int num_cells, size_t max_buffer_size, CleanupParams &params, cudaStream_t strm);
int gpu_comm_warmup(cudaStream_t cstream);
int gpucomm_pre_prepare_send(PrepareParams &params, cudaStream_t cstream);
void launch_packet_memcpy_kernel(PacketCopyParams, int num_cells, cudaStream_t strm);
int gpucomm_prepare_send(PrepareParams &params, cudaStream_t cstream);
int gpucomm_trigger_send(TriggerParams &params, cudaStream_t cstream, bool cx6);
int gpucomm_ring_doorbell_for_cells(doca_gpu_eth_txq** d_txq_handlers, const uint32_t* d_wqe_indices, const uint32_t num_cells, cudaStream_t cstream);
void force_loading_gpu_comm_kernels();

class GpuComm {
public:
    GpuComm(Nic* nic);
    ~GpuComm();
    int txq_init(Txq* txq);
    int txq_poll_cq(Txq* txq,uint8_t* txq_addr);
    void traceCqe(TxRequestGpuPercell *pTxRequestGpuPercell);
    int send(TxRequestGpuPercell *pTxRequestGpuPercell, uint32_t hdr_len, uint16_t ecpri_payload_overhead, PreparePRBInfo &prb_info);
    int cpu_send(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info);
    uint32_t getErrorFlag();
    PacketStartDebugInfo* getPkt_start_debug(int index,uint16_t packet_idx);
    void setDlSlotTriggerTs(uint32_t slotIdx,uint64_t trigger_ts);
    uint64_t getDlSlotTriggerTs(uint32_t slotIdx);
    Nic*     getNic() const;
protected:
    Nic*                nic_;
    cudaStream_t        cstream_;
    cudaStream_t        cstream_pkt_copy_;
    std::array<cudaEvent_t,ORAN_ALL_SYMBOLS>         pkt_copy_done_evt;
    std::array<cudaEvent_t,ORAN_ALL_SYMBOLS>         trigger_end;
    void*               bf_db_dev_;
    volatile uint64_t*  bf_db_host_;
    uint32_t pkt_counts_flow[API_MAX_NUM_CELLS][kMaxFlows];
    uint32_t*           host_pinned_error_flag;
    std::array<PacketStartDebugInfo*, API_MAX_NUM_CELLS> pkt_start_debug;
    std::array<uint64_t,MAX_DL_SLOTS_TRIGGER_TIME> dl_slot_trigger_ts;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_GPU_COMM__
