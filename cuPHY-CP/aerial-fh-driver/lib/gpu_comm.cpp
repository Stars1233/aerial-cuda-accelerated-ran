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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu_comm.hpp"
#include "gpu.hpp"
#include "nic.hpp"
#include "utils.hpp"
#include "fronthaul.hpp"
#include "time.hpp"
#include "cuphy_pti.hpp"
#include "cuphy.hpp"
#include "mlx5.hpp"
#include <doca_eth_txq.h>

#ifndef DOCA_GPUNETIO_WQE_PI_MASK
#define DOCA_GPUNETIO_WQE_PI_MASK 0xFFFF
#endif

#define TAG (NVLOG_TAG_BASE_FH_DRIVER+13) //FH.GPU_COMM

#define DUMP_WQE_INFO 0

namespace aerial_fh
{

int gpu_comm_cpu_poll_cq(void * cq_buf_, uint32_t * cq_ci, uint16_t cqe_s, uint16_t cqe_m, uint32_t qp_num_8s, uintptr_t cq_db_addr, uint32_t* wqe_pi, uint8_t* sq);

GpuComm::GpuComm(Nic* nic) :
    nic_{nic},
    bf_db_host_{nullptr},
    bf_db_dev_{nullptr}
{
    /* Assume CUDA and context is set from calling thread */
    ASSERT_CUDA_FH(cudaStreamCreateWithPriority(&cstream_, cudaStreamNonBlocking, -5));
    ASSERT_CUDA_FH(cudaStreamCreateWithPriority(&cstream_pkt_copy_, cudaStreamNonBlocking, -5));
    gpu_comm_warmup(cstream_);
    gpu_comm_warmup(cstream_pkt_copy_);

    const char* memtrace_env = std::getenv("AERIAL_MEMTRACE");
    if((memtrace_env != nullptr) && std::atoi(memtrace_env) == 1)
    {
        force_loading_gpu_comm_kernels();
    }
    int flags=cudaEventDisableTiming;

    for (int sym = 0; sym < pkt_copy_done_evt.size(); sym++) {
        ASSERT_CUDA_FH(cudaEventCreateWithFlags(&pkt_copy_done_evt[sym],flags));
        ASSERT_CUDA_FH(cudaEventCreate(&trigger_end[sym]));
    }

    PrepareParams prepare{};
    prepare.num_cells = 16;
    for (int cell = 0; cell < 16; cell++) {
        prepare.cell_params[cell].d_slot_info = nullptr;
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(0);
    gpucomm_pre_prepare_send(prepare, cstream_);
    gpucomm_prepare_send(prepare, cstream_);
    cudaStreamSynchronize(cstream_);

    ASSERT_CUDA_FH(cudaMallocHost((void**)&host_pinned_error_flag, sizeof(uint32_t)));

    *host_pinned_error_flag = 0;
    for (int q = 0; q < API_MAX_NUM_CELLS; q++) {
        ASSERT_CUDA_FH(cudaMallocHost(&pkt_start_debug[q], MAX_PKT_DEBUG * sizeof(*pkt_start_debug[q])));
     }

    memset(&pkt_counts_flow, 0, sizeof(pkt_counts_flow));
}

GpuComm::~GpuComm()
{
    bf_db_host_ = nullptr;
    ASSERT_CUDA_FH(cudaFreeHost(host_pinned_error_flag));
    ASSERT_CUDA_FH(cudaStreamDestroy(cstream_));
    ASSERT_CUDA_FH(cudaStreamDestroy(cstream_pkt_copy_));
    for (int q = 0; q < API_MAX_NUM_CELLS; q++) {
        ASSERT_CUDA_FH(cudaFreeHost(pkt_start_debug[q]));
    }
}

Nic* GpuComm::getNic() const
{
    return nic_;
}

PacketStartDebugInfo* GpuComm::getPkt_start_debug(int index, uint16_t packet_idx)
{
    return (pkt_start_debug[index]+packet_idx);
}

int GpuComm::txq_init(Txq* txq)
{
    ASSERT_CUDA_FH(cudaStreamSynchronize(cstream_));
    return 0;
}


void GpuComm::setDlSlotTriggerTs(uint32_t slotIdx,uint64_t trigger_ts)
{
    dl_slot_trigger_ts[slotIdx%MAX_DL_SLOTS_TRIGGER_TIME]=trigger_ts;
}

uint64_t GpuComm::getDlSlotTriggerTs(uint32_t slotIdx)
{
    return dl_slot_trigger_ts[slotIdx%MAX_DL_SLOTS_TRIGGER_TIME];
}

void GpuComm::traceCqe(TxRequestGpuPercell *pTxRequestGpuPercell)
{
        TxRequestUplaneGpuComm* requests[kGpuCommSendPeers];
        doca_tx_items_t *txh[kGpuCommSendPeers];
        for(int idx = 0; idx < kGpuCommSendPeers; idx++) {
            requests[idx] = nullptr;
            txh[idx] = nullptr;
            if(idx < pTxRequestGpuPercell->size) {
                requests[idx] = static_cast<TxRequestUplaneGpuComm*>(pTxRequestGpuPercell->tx_v_per_nic[idx]);
                txh[idx] = requests[idx]->txq->get_doca_tx_items();
                txh[idx]->frame_id = requests[idx]->frame_id;
                txh[idx]->subframe_id = requests[idx]->subframe_id;
                txh[idx]->slot_id = requests[idx]->slot_id;
                txh[idx]->gCommsHdl = static_cast<void*>(this);
                txh[idx]->cell_idx = idx;
                txq_poll_cq(requests[idx]->txq,NULL);
            }
        }
}
#include <sys/time.h>

int GpuComm::cpu_send(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info) {
    doca_tx_items_t *txh[kGpuCommSendPeers];
    TxRequestUplaneGpuComm* requests[kGpuCommSendPeers];
    NVLOGI_FMT(TAG,"Starting cpu_send, pTxRequestGpuPercell->size={}", pTxRequestGpuPercell->size);
    packet_timing_info.cpu_send_start_timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    for(int idx = 0; idx < kGpuCommSendPeers; idx++) {
        requests[idx] = nullptr;
        txh[idx] = nullptr;
        if(idx < pTxRequestGpuPercell->size) {
            requests[idx] = static_cast<TxRequestUplaneGpuComm*>(pTxRequestGpuPercell->tx_v_per_nic[idx]);
            txh[idx] = requests[idx]->txq->get_doca_tx_items();
            txh[idx]->frame_id = requests[idx]->frame_id;
            txh[idx]->subframe_id = requests[idx]->subframe_id;
            txh[idx]->slot_id = requests[idx]->slot_id;
            txh[idx]->gCommsHdl = static_cast<void*>(this);
            txh[idx]->cell_idx = idx;

            NVLOGD_FMT(TAG, "GpuComm on TXQ[{}] = {} DOCA txq info {}",
                    idx, requests[idx]->txq->get_id(), reinterpret_cast<void*>(txh[idx]->eth_txq_gpu));
            NVLOGI_FMT(TAG, "requests[{}]->h_up_slot_info_->old_wqe_pi={} ptr={:p}",
                idx, requests[idx]->h_up_slot_info_->old_wqe_pi, (void*)&requests[idx]->h_up_slot_info_->old_wqe_pi);
        }
    }

    if (prb_info.compression_stop_evt != nullptr) {
        ASSERT_CUDA_FH(cudaStreamWaitEvent(cstream_pkt_copy_, prb_info.compression_stop_evt));
    }

    int check_symbol = 0;
    int copies_finished = 0;

    uint64_t time1[ORAN_ALL_SYMBOLS], time2[ORAN_ALL_SYMBOLS];
    float diff=0.0;
    PacketCopyParams copy_params;
    uint32_t num_packets;
    //Reset all timestamps in packet_timing_info to 0 before loop start
    packet_timing_info.pkt_copy_done_timestamp.fill(0);
    packet_timing_info.trigger_done_timestamp.fill(0);
    packet_timing_info.pkt_copy_launch_timestamp.fill(0);
    packet_timing_info.num_packets_per_symbol.fill(0);

    while (check_symbol < 14) {
        if (copies_finished < 14) {
            copy_params.max_pkts = 0;
            copy_params.pkt_size = nic_->get_buf_size();
            uint32_t flow_offset=0;
            num_packets = 0;

            for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
                auto txr = static_cast<TxRequestUplaneGpuComm*>(pTxRequestGpuPercell->tx_v_per_nic[cell]);
                copy_params.cell_params[cell].d_src_addr = (void*)((uint8_t*)nic_->get_flow_comm_buf()->gpu_pkt_addr + nic_->get_buf_size()*flow_offset*kMaxPktsFlow); //* copy_params.pkt_size + pkt_counts_flow[cell] * copy_params.pkt_size;
                copy_params.cell_params[cell].h_dst_addr = (void*)((uint8_t*)nic_->get_flow_comm_buf()->cpu_comms_cpu_pkt_addr + nic_->get_buf_size()*flow_offset*kMaxPktsFlow); //* copy_params.pkt_size + pkt_counts_flow[cell] * copy_params.pkt_size;
                copy_params.cell_params[cell].num_pkts   = requests[cell]->partial_slot_info->ttl_pkts;
                copy_params.cell_params[cell].num_flows  = requests[cell]->partial_slot_info->flowInfo_slot.num_flows;
                for(int i=0;i<copy_params.cell_params[cell].num_flows;i++)
                {
                    copy_params.cell_params[cell].num_pkts_per_flow[i]  = txr->partial_slot_info->flowInfo_slot.sym_flow_packet_count[i][copies_finished]; //requests[cell]->partial_slot_info->flowInfo_slot.flow_packet_count[i];
                    copy_params.max_pkts                                = std::max(copy_params.max_pkts,copy_params.cell_params[cell].num_pkts_per_flow[i]);
                    num_packets += copy_params.cell_params[cell].num_pkts_per_flow[i];
                    copy_params.cell_params[cell].pkt_offset[i]         = pkt_counts_flow[cell][i] + txr->partial_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[i][copies_finished];
                    NVLOGI_FMT(TAG,"{} : F{}S{}S{} flow index = {} Packets per flow={} pkt_offset per flow={} max_pkts={} pkt_cnts_flow={} cumul={}",
                            __func__,requests[0]->frame_id,requests[0]->subframe_id,requests[0]->slot_id,i,
                            copy_params.cell_params[cell].num_pkts_per_flow[i],copy_params.cell_params[cell].pkt_offset[i],copy_params.max_pkts, pkt_counts_flow[cell][i], txr->partial_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[i][copies_finished]);
                }
                copy_params.frame_id        =   requests[0]->frame_id;
                copy_params.subframe_id     =   requests[0]->subframe_id;
                copy_params.slot_id         =   requests[0]->slot_id;
                flow_offset                 +=  requests[cell]->partial_slot_info->total_num_flows;
                NVLOGI_FMT(TAG,"{} : F{}S{}S{} Total packets for copy kernel={} total num flows={} for cell id {} flow_offset {}",
                        __func__,requests[0]->frame_id,requests[0]->subframe_id,requests[0]->slot_id,copy_params.cell_params[cell].num_pkts,requests[cell]->partial_slot_info->total_num_flows,cell,flow_offset);
            }
            packet_timing_info.num_packets_per_symbol[copies_finished] = num_packets;

            if (copy_params.max_pkts == 0) {
                NVLOGI_FMT(TAG, "Skipping F{}S{}S{} symbol {} with 0 packets", requests[0]->frame_id,requests[0]->subframe_id,requests[0]->slot_id, copies_finished);
                time1[copies_finished] = std::chrono::steady_clock::now().time_since_epoch().count();
            }
            else {
                NVLOGI_FMT(TAG, "Launching copy kernel for symbol {}", copies_finished);
                time1[copies_finished] = std::chrono::steady_clock::now().time_since_epoch().count();
                if (prb_info.use_copy_kernel_for_d2h)
                {
                    launch_packet_memcpy_kernel(copy_params, pTxRequestGpuPercell->size, cstream_pkt_copy_);
                }
                else
                {
                    MemtraceDisableScope md;
                    // Use batched memcpy helper to batch all cell/flow copies for this symbol
                    // Note: helper falls back to individual cudaMemcpyAsync calls on older CUDA or if batching disabled
                    size_t max_copies = 0;
                    for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
                        max_copies += static_cast<size_t>(copy_params.cell_params[cell].num_flows) * 2U; // worst-case wrap requires 2 copies per flow
                    }
                    cuphyBatchedMemcpyHelper memcpy_helper(max_copies, batchedMemcpySrcHint::srcIsDevice, batchedMemcpyDstHint::dstIsHost, true);
                    memcpy_helper.reset();

                    const uint32_t pkt_size_bytes = copy_params.pkt_size;
                    const uint32_t flow_capacity_bytes = kMaxPktsFlow * pkt_size_bytes;

                    for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
                        const uint32_t flows = copy_params.cell_params[cell].num_flows;
                        const uint8_t* d_cell_base = reinterpret_cast<const uint8_t*>(copy_params.cell_params[cell].d_src_addr);
                        uint8_t* h_cell_base = reinterpret_cast<uint8_t*>(copy_params.cell_params[cell].h_dst_addr);
                        for (uint32_t flow = 0; flow < flows; ++flow) {
                            const uint32_t num_pkts_flow = copy_params.cell_params[cell].num_pkts_per_flow[flow];
                            if (num_pkts_flow == 0) {
                                continue;
                            }
                            const uint32_t pkt_offset = static_cast<uint32_t>(copy_params.cell_params[cell].pkt_offset[flow] % kMaxPktsFlow);
                            const uint8_t* d_base = d_cell_base + static_cast<size_t>(flow) * flow_capacity_bytes;
                            uint8_t* h_base = h_cell_base + static_cast<size_t>(flow) * flow_capacity_bytes;
                            const uint32_t contiguous_bytes = num_pkts_flow * pkt_size_bytes;
                            const uint32_t byte_offset = pkt_offset * pkt_size_bytes;

                            if (byte_offset + contiguous_bytes <= flow_capacity_bytes) {
                                memcpy_helper.updateMemcpy(h_base + byte_offset, const_cast<uint8_t*>(d_base + byte_offset), contiguous_bytes, cudaMemcpyDeviceToHost, cstream_pkt_copy_);
                            } else {
                                const uint32_t first_chunk = flow_capacity_bytes - byte_offset;
                                const uint32_t second_chunk = contiguous_bytes - first_chunk;
                                memcpy_helper.updateMemcpy(h_base + byte_offset, const_cast<uint8_t*>(d_base + byte_offset), first_chunk, cudaMemcpyDeviceToHost, cstream_pkt_copy_);
                                memcpy_helper.updateMemcpy(h_base, const_cast<uint8_t*>(d_base), second_chunk, cudaMemcpyDeviceToHost, cstream_pkt_copy_);
                            }
                        }
                    }

                    cuphyStatus_t status = memcpy_helper.launchBatchedMemcpy(cstream_pkt_copy_);
                    if(status != CUPHY_STATUS_SUCCESS)
                    {
                        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Launching batched memcpy for packet copy returned an error");
                        return status;
                    }
                }
            }
            if(packet_timing_info.num_packets_per_symbol[copies_finished] > 0)
            {
                packet_timing_info.pkt_copy_launch_timestamp[copies_finished] = std::chrono::system_clock::now().time_since_epoch().count();
            }
            ASSERT_CUDA_FH(cudaEventRecord(pkt_copy_done_evt[copies_finished], cstream_pkt_copy_));
            copies_finished++;
        }

        do {
            if (cudaEventQuery(pkt_copy_done_evt[check_symbol]) == cudaSuccess) {
                if(packet_timing_info.num_packets_per_symbol[check_symbol] > 0)
                {
                    packet_timing_info.pkt_copy_done_timestamp[check_symbol] = std::chrono::system_clock::now().time_since_epoch().count();
                }
                time2[check_symbol]                                        = std::chrono::steady_clock::now().time_since_epoch().count();
                diff                                                       = (time2[check_symbol] - time1[check_symbol]) / 1000.0;
                prb_info.p_packet_mem_copy_per_symbol_dur_us[check_symbol] = diff;
                NVLOGI_FMT(TAG, "Processing symbol {} after D2H copy completion", check_symbol);
                packet_timing_info.frame_id = requests[0]->frame_id;
                packet_timing_info.subframe_id = requests[0]->subframe_id;
                packet_timing_info.slot_id = requests[0]->slot_id;
                                
                if(packet_timing_info.num_packets_per_symbol[check_symbol] > 0)
                {
                    packet_timing_info.trigger_done_timestamp[check_symbol] = std::chrono::system_clock::now().time_since_epoch().count();
                }
                check_symbol++;
            }
            else {
                //NVLOGI_FMT(TAG, "Symbol {} not ready", check_symbol);
                break;
            }
        } while (check_symbol < copies_finished);
    }

    // Call CPU proxy progress for each cell after D2H copy completes
    // Note: doca_gpu_dev_eth_txq_submit_proxy is now called from within gpu_comm_prepare_send_doca kernel
    for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
        const doca_error_t result = doca_eth_txq_gpu_cpu_proxy_progress(txh[cell]->eth_txq_cpu);
        if (result != DOCA_SUCCESS) {
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "doca_eth_txq_gpu_cpu_proxy_progress failed for cell {} : {}", cell, doca_error_get_descr(result));
        }
    }

    // Accumulate the total packets per flow for the next copy round
    for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
        auto txr = static_cast<TxRequestUplaneGpuComm*>(pTxRequestGpuPercell->tx_v_per_nic[cell]);
        for(int i=0;i<copy_params.cell_params[cell].num_flows;i++) {
            pkt_counts_flow[cell][i]   = (pkt_counts_flow[cell][i] + txr->partial_slot_info->flowInfo_slot.flow_packet_count[i]) % kMaxPktsFlow;
        }
    }

    //     printf("WQE %p\n",(struct mlx5_wqe *)requests[0]->h_up_slot_info_->wqe_addr[0]);
    // for (int w = 0; w < wqe_pi; w++) {
    //     printf("%003d: ", w);
    //     for (int i = 0; i < 64; i++) {
    //         auto first_wqe_addr = (struct mlx5_wqe *)(requests[0]->h_up_slot_info_->wqe_addr[0]);
    //         printf("%02x ", *(((uint8_t*)&first_wqe_addr[w-1]) + i));
    //     }
    //     printf("\n");
    // }


    if (prb_info.trigger_end_evt != nullptr) {
        MemtraceDisableScope md;
        ASSERT_CUDA_FH(cudaEventRecord(prb_info.trigger_end_evt, cstream_));
    }

    return 0;
}

int GpuComm::send(TxRequestGpuPercell *pTxRequestGpuPercell, uint32_t hdr_len, uint16_t ecpri_payload_overhead, PreparePRBInfo &prb_info)
{
    //NVLOGC_FMT(TAG, "GpuComm::send");
    TxRequestUplaneGpuComm* requests[kGpuCommSendPeers];
    int ret = 0;

    if (pTxRequestGpuPercell->size == 0)
        return -EINVAL;

    if (pTxRequestGpuPercell->size > kGpuCommSendPeers)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "GpuComm can't work with more than {} peers", kGpuCommSendPeers);
        return -ENOTSUP;
    }

    doca_tx_items_t *txh[kGpuCommSendPeers];
    for(int idx = 0; idx < kGpuCommSendPeers; idx++) {
        requests[idx] = nullptr;
        txh[idx] = nullptr;
        if(idx < pTxRequestGpuPercell->size) {
            requests[idx] = static_cast<TxRequestUplaneGpuComm*>(pTxRequestGpuPercell->tx_v_per_nic[idx]);
            txh[idx] = requests[idx]->txq->get_doca_tx_items();
            txh[idx]->frame_id = requests[idx]->frame_id;
            txh[idx]->subframe_id = requests[idx]->subframe_id;
            txh[idx]->slot_id = requests[idx]->slot_id;
            txh[idx]->gCommsHdl = static_cast<void*>(this);
            txh[idx]->cell_idx = idx;

            NVLOGD_FMT(TAG, "GpuComm on TXQ[{}] = {} DOCA txq info {}",
                    idx, requests[idx]->txq->get_id(), reinterpret_cast<void*>(txh[idx]->eth_txq_gpu));
        }
    }

    // Reset all PRB pointers by using a memset kernel. The performance of a memset kernel was shown to be significantly higher
    // than launching N cudaMemsets where N is the number of cells.
    CleanupParams cleanup{};
    size_t max_buffer_size = 0;
    for (int cell = 0; cell < pTxRequestGpuPercell->size ; cell++) {
        requests[cell]->max_num_prb_per_symbol = prb_info.max_num_prb_per_symbol[cell];
        size_t buf_size = prb_info.max_num_prb_per_symbol[cell] * SLOT_NUM_SYMS * prb_info.num_antennas[cell] * sizeof(uint8_t*);
        max_buffer_size = std::max(max_buffer_size, buf_size);
        //h_buffers_addr[cell] = {(uint4*)prb_info.prb_ptrs[cell], buf_size};
        cleanup.cell_params[cell] = {requests[cell]->d_slot_info, (uint4*)prb_info.prb_ptrs[cell], buf_size};
        // The memset kernel expects prb_info.prb_ptr[cell] to be uint4 aligned. An error message is printed otherwise.
        if((reinterpret_cast<uintptr_t>(prb_info.prb_ptrs[cell]) & 0xF) != 0) { // Raise an error if not uint4 aligned
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "GpuComm PRB pointer {} for cell {} is not properly aligned", reinterpret_cast<void*>((uint4*)prb_info.prb_ptrs[cell]), cell);
            //FIXME should we return?
        }
    }

    //Create event that lets us know when comms sequence starts
    if(prb_info.enable_prepare_tracing) {
        ASSERT_CUDA_FH(cudaEventRecord(prb_info.comm_start_evt, cstream_));
    }
    launch_memset_kernel(pTxRequestGpuPercell->size, max_buffer_size, cleanup, cstream_);
    if(prb_info.enable_prepare_tracing) {
        ASSERT_CUDA_FH(cudaEventRecord(prb_info.comm_copy_evt, cstream_));
    }
    //NVLOGC_FMT(TAG, "GpuComm after memset launch");

    //FIXME: super ugly but functional for the moment
    PrepareParams prepare{};
    prepare.hdr_len                 = hdr_len;
    prepare.ecpri_payload_overhead  = ecpri_payload_overhead;
    prepare.frame_id                = requests[0]->frame_id;
    prepare.num_cells               = pTxRequestGpuPercell->size;
    prepare.subframe_id             = requests[0]->subframe_id;
    prepare.slot_id                 = requests[0]->slot_id;
    prepare.payload_info =          prb_info;
    prepare.host_pinned_error_flag  = host_pinned_error_flag;
    prepare.enable_gpu_comm_via_cpu = getNic()->get_fronthaul()->get_info().enable_gpu_comm_via_cpu;
    for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
		prepare.cell_params[cell] = { txh[cell]->eth_txq_gpu,
										requests[cell]->d_slot_info,
                                        requests[cell]->h_up_slot_info_,
										requests[cell]->partial_slot_info,
                                        getNic()->get_mtu(),
										requests[cell]->prb_size_upl,
										requests[cell]->prbs_per_pkt_upl,
										requests[cell]->flow_d_info,
                                        requests[cell]->flow_sym_d_info,
                                        requests[cell]->block_count,
										requests[cell]->flow_hdr_size_info,
										requests[cell]->flow_d_ecpri_seq_id,
										requests[cell]->flow_d_hdr_template_info,
                                        requests[cell]->max_num_prb_per_symbol,
                                        pkt_start_debug[cell]};
    }

    ret = gpucomm_pre_prepare_send(prepare, cstream_);
    if(ret) {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "GpuComm pre prepare send failed on GPU");
        return -1;
    }

    // Used to tell compression it can go
    // Note: compression waits on this event
    if (prb_info.comm_preprep_stop_evt != nullptr) {
        MemtraceDisableScope md;
        ASSERT_CUDA_FH(cudaEventRecord(prb_info.comm_preprep_stop_evt, cstream_));
    }

    ret = gpucomm_prepare_send(prepare, cstream_);
    if(ret) {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "GpuComm prepare send failed on GPU");
        return -1;
    }

    if (prb_info.comm_stop_evt != nullptr) {
        MemtraceDisableScope md;
        ASSERT_CUDA_FH(cudaEventRecord(prb_info.comm_stop_evt, cstream_));
    }

    if(getNic()->get_fronthaul()->get_info().enable_gpu_comm_via_cpu==0)
    {
        TriggerParams trigger{};
        trigger.ready_flag  = prb_info.ready_flag;
        trigger.wait_val    = prb_info.wait_val;
        trigger.num_cells   = pTxRequestGpuPercell->size;
        trigger.disable_empw  = prb_info.disable_empw;
        for (int cell = 0; cell < pTxRequestGpuPercell->size; cell++) {
            trigger.cell_params[cell] = {txh[cell]->eth_txq_gpu,NULL,
                                        requests[cell]->d_slot_info};
        }

        if (prb_info.compression_stop_evt != nullptr) {
            ASSERT_CUDA_FH(cudaStreamWaitEvent(cstream_, prb_info.compression_stop_evt));
        }

        ret = gpucomm_trigger_send(trigger, cstream_, nic_->is_cx6());

        if(ret) {
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "GpuComm trigger send failed on GPU");
            return -1;
        }


        if (prb_info.trigger_end_evt != nullptr) {
            MemtraceDisableScope md;
            ASSERT_CUDA_FH(cudaEventRecord(prb_info.trigger_end_evt, cstream_));
        }
    }

    return 0;
}

int GpuComm::txq_poll_cq(Txq* txq,uint8_t* txq_addr)
{
    doca_tx_items_t* doca_tx_h = txq->get_doca_tx_items();
    unsigned int count = MLX5_TX_COMP_MAX_CQE;
    do {
        doca_pe_progress(doca_tx_h->eth_txq_pe);
        // nanosleep(&ts, &ts);
        /*
         * We have to restrict the amount of processed CQEs
         * in one tx_burst routine call. The CQ may be large
         * and many CQEs may be updated by the NIC in one
         * transaction. Buffers freeing is time consuming,
         * multiple iterations may introduce significant latency.
         */
        if (likely(--count == 0))
            break;
    } while(true);
    return 0;
}


 static inline enum mlx5_cqe_status
 check_cqe(volatile struct mlx5_cqe64 *cqe, const uint16_t cqes_n, const uint16_t ci)
 {
     const uint16_t idx = ci & cqes_n;
     const uint8_t op_own = cqe->op_own;
     const uint8_t op_owner = MLX5_CQE_OWNER(op_own);
     const uint8_t op_code = MLX5_CQE_OPCODE(op_own);

     if (unlikely((op_owner != (!!(idx))) || (op_code == MLX5_CQE_INVALID)))
         return MLX5_CQE_STATUS_HW_OWN;
     rte_io_rmb();
     if (unlikely(op_code == MLX5_CQE_RESP_ERR ||
              op_code == MLX5_CQE_REQ_ERR))
         return MLX5_CQE_STATUS_ERR;
     return MLX5_CQE_STATUS_SW_OWN;
 }


int gpu_comm_cpu_poll_cq(void * cq_buf_, uint32_t * cq_ci, uint16_t cqe_s, uint16_t cqe_m, uint32_t qp_num_8s, uintptr_t cq_db_addr, uint32_t* wqe_pi, uint8_t* sq)
{
    struct mlx5_cqe64 * cq_buf = (struct mlx5_cqe64 *) cq_buf_;
    // volatile struct mlx5_cqe64 *last_cqe = NULL;
    int ret;
    unsigned int count = MLX5_TX_COMP_MAX_CQE;
    // printf("Starting polling with qp_num_8s %d cq_ci %d addr %p cqe_s %d cqe_m %d\n",
    //  qp_num_8s, cq_ci[0], cq_buf, cqe_s, cqe_m);
    do {
        volatile struct mlx5_cqe64 *cqe;
        volatile struct mlx5_err_cqe * cqe_err;

        cqe = &(cq_buf[cq_ci[0] & cqe_m]);
        // printf("CQE cq_ci %d index %d\n", cq_ci[0], cq_ci[0] & cqe_m);

        ret = check_cqe(cqe, cqe_s, cq_ci[0]);
        if (unlikely(ret != MLX5_CQE_STATUS_SW_OWN)) {
            if (likely(ret != MLX5_CQE_STATUS_ERR)) {
                /* No new CQEs in completion queue. */
                if(ret != MLX5_CQE_STATUS_HW_OWN) {
                    NVLOGF_FMT(TAG, AERIAL_DPDK_API_EVENT, "Unexpected CQE status: {} != {}", ret, MLX5_CQE_STATUS_HW_OWN);
                }
                break;
            }
            /*
            * Some error occurred, try to restart.
            * We have no barrier after WQE related Doorbell
            * written, make sure all writes are completed
            * here, before we might perform SQ reset.
            */
            rte_wmb();
            cqe_err = (volatile struct mlx5_err_cqe *)cqe;
            //NVLOGW_FMT(TAG,"Unexpected CQE error syndrome {} CQN = {} SQN = {} wqe_counter = {} cq_ci = {}. Err CQE:",cqe_err->syndrome, cqe_s, qp_num_8s >> 8,rte_be_to_cpu_16(cqe_err->wqe_counter), cq_ci[0]);
            NVLOGE_FMT(TAG,AERIAL_DPDK_API_EVENT,"Unexpected CQE error syndrome {} CQN = {} SQN = {} wqe_counter = {} cq_ci = {}. Err CQE:",(uint8_t)cqe_err->syndrome,cqe_s, qp_num_8s >> 8,rte_be_to_cpu_16(cqe_err->wqe_counter), cq_ci[0]);

            for(int x = 0; x < 64; x++) {
                uint8_t * tmp = (uint8_t *)cqe_err;
                NVLOGI_FMT(TAG, "{} ", tmp[x]);
                if(x > 0 && x % 16 == 0) fprintf(stderr, "\n");
            }

#if DUMP_WQE_INFO
            //This code will dump the WQE and CQE buffer for debugging. Please leave in here
            {
                FILE *fp = fopen("/tmp/wqe.txt", "w");
                uint32_t pi;
                cudaMemcpy(&pi, wqe_pi, sizeof(pi), cudaMemcpyDeviceToHost);
                NVLOGI_FMT(TAG,"PI is {}", pi);
                uint8_t *buf;
                pi=8192;
                cudaMallocHost((void**)&buf, 64*pi);
                cudaMemcpy(buf, (void*)sq, 64*pi, cudaMemcpyDeviceToHost);
                fprintf(fp, "PI %u\n", pi);
                NVLOGI_FMT(TAG,"CQES {} idx {}", cqe_m, cq_ci[0]);
                uint8_t *wqe = (uint8_t*)buf;
                for (int i = 0; i < pi; i++) {
                    for (int j = 0; j < 64; j++) {
                        fprintf(fp, "%02X ", *wqe++);
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                NVLOGI_FMT(TAG,"done");
            }
#endif
            /*
            * We are going to fetch all entries with
            * MLX5_CQE_SYNDROME_WR_FLUSH_ERR status.
            * The send queue is supposed to be empty.
            */
            cq_ci[0]++;
            cq_ci[0] = cq_ci[0] & 0xFFFF;

            NVLOGI_FMT(TAG,"CQE inner loop cq_ci {} index {}", cq_ci[0], cq_ci[0] & cqe_m);

            // last_cqe = NULL;
            continue;
        }

#if 0
        printf("CQE %d / TOT %d: \n"
                "\t Owner: %d\n"
                "\t Last WQE %d\n"
                "\t Timestamp %llx - %llu\n"
                ,
                cq_ci[0] & cqe_m, cq_ci[0],
                cqe->op_own,
                rte_be_to_cpu_16(cqe->wqe_counter),
                rte_be_to_cpu_64(cqe->timestamp),
                rte_be_to_cpu_64(cqe->timestamp)
        );
#endif

        cq_ci[0]++;
        cq_ci[0] = cq_ci[0] & 0xFFFF;
        // last_cqe = cqe;
        /*
        * We have to restrict the amount of processed CQEs
        * in one tx_burst routine call. The CQ may be large
        * and many CQEs may be updated by the NIC in one
        * transaction. Buffers freeing is time consuming,
        * multiple iterations may introduce significant latency.
        */
        if (likely(--count == 0))
            break;
    } while (true);

    /* Ring doorbell to notify hardware. */
    rte_compiler_barrier();
    *((volatile uint32_t * )cq_db_addr) = rte_cpu_to_be_32(cq_ci[0]);
    // printf("Polling done, cq_ci %d count %d\n", cq_ci[0], count);

    return 0;
}


uint32_t GpuComm::getErrorFlag()
{
    return *host_pinned_error_flag;
}

#ifndef ENABLE_DOCA_GPU_COMM

// stub definitions when DOCA GPU communication is disabled

void launch_memset_kernel(int num_cells, size_t max_buffer_size, CleanupParams &params, cudaStream_t strm)
{
    NVLOGW_FMT(TAG, "launch_memset_kernel called but DOCA GPU communication is disabled");
}

int gpu_comm_warmup(cudaStream_t cstream)
{
    NVLOGW_FMT(TAG, "gpu_comm_warmup called but DOCA GPU communication is disabled");
    return 0;
}

int gpucomm_pre_prepare_send(PrepareParams &params, cudaStream_t cstream)
{
    NVLOGW_FMT(TAG, "gpucomm_pre_prepare_send called but DOCA GPU communication is disabled");
    return 0;
}

void launch_packet_memcpy_kernel(PacketCopyParams params, int num_cells, cudaStream_t strm)
{
    NVLOGW_FMT(TAG, "launch_packet_memcpy_kernel called but DOCA GPU communication is disabled");
}

int gpucomm_prepare_send(PrepareParams &params, cudaStream_t cstream)
{
    NVLOGW_FMT(TAG, "gpucomm_prepare_send called but DOCA GPU communication is disabled");
    return 0;
}

int gpucomm_trigger_send(TriggerParams &params, cudaStream_t cstream, bool cx6)
{
    NVLOGW_FMT(TAG, "gpucomm_trigger_send called but DOCA GPU communication is disabled");
    return 0;
}

void force_loading_gpu_comm_kernels()
{
    NVLOGW_FMT(TAG, "force_loading_gpu_comm_kernels called but DOCA GPU communication is disabled");
}

#endif // !ENABLE_DOCA_GPU_COMM

} // namespace aerial_fh
