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

#include <doca_gpunetio.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <doca_gpunetio_dev_eth_txq.cuh>

//The relevant memcpy_async code below should also be guarded with an else clause.
//However, not having it allowed us to catch a case where the old non-memcpy code path was exercised
//when moving to CUDA 13.0. So left it as is for now.
#if((__CUDACC_VER_MAJOR__ >= 12) || (__CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 1)))
#include <cooperative_groups/memcpy_async.h>
#endif

#include "gpu_comm.hpp"
#include "gpu.hpp"
#include "nic.hpp"
#include "utils.hpp"
#include "cuphy_pti.hpp"

#define TAG "FH.DOCA"

#define FENCE_DEV() do { __threadfence(); } while(0)
#define FENCE_SYS() do { __threadfence_system(); } while(0)

#define MAX(v1, v2)    ((v1) > (v2) ? (v1) : (v2))
#define MIN(v1, v2)    ((v1) < (v2) ? (v1) : (v2))

#define BSWAP16(x) (x >> 8) | (x << 8)

#define BSWAP32(x) \
    ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |           \
    (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))

#define BSWAP64(x) \
    ((((x) & 0xff00000000000000ull) >> 56)    \
    | (((x) & 0x00ff000000000000ull) >> 40)    \
    | (((x) & 0x0000ff0000000000ull) >> 24)    \
    | (((x) & 0x000000ff00000000ull) >> 8)    \
    | (((x) & 0x00000000ff000000ull) << 8)    \
    | (((x) & 0x0000000000ff0000ull) << 24)    \
    | (((x) & 0x000000000000ff00ull) << 40)    \
    | (((x) & 0x00000000000000ffull) << 56))
namespace cg = cooperative_groups;

#define EMPW_DSEG_MAX_NUM 61
#define SEGMENTS_PER_WQE 4

//#define DBG_GPU_COMM

namespace aerial_fh
{

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}

__global__ void warmup()
{
    FENCE_DEV();
    FENCE_SYS();
}

int gpu_comm_warmup(cudaStream_t cstream)
{
    cudaError_t result = cudaSuccess;

    warmup<<<2, 512, 0, cstream>>>();

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching warmup kernel failed with {}", __FILE__, __LINE__, cudaGetErrorString(result));
    return 0;
}
__global__ void memset_kernel(CleanupParams params) {

    const uint4 val = {0, 0, 0, 0};
    int cell = blockIdx.y;

    uint4* d_buffer_addr = params.cell_params[cell].d_buf_addr; // assumption that d_buffer_addr is uint4 aligned (OK since each is a new allocation)
    size_t d_buffer_size = params.cell_params[cell].buf_size;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < (d_buffer_size >> 4)) { // 4 is to divide by sizeof(uint4) as buf_size is in bytes
        d_buffer_addr[tid] = val;
    }

    //Handle leftover bytes
    int leftover_bytes = d_buffer_size & 0xF; // modulo sizeof(uint4) = 16
    if (tid < leftover_bytes) {
        uint8_t* d_buffer_byte_addr = (uint8_t*)d_buffer_addr + (d_buffer_size - leftover_bytes);
        d_buffer_byte_addr[tid] = 0;
    }

    UplaneSlotInfo_t* this_slot_info    = params.cell_params[cell].d_slot_info;

    if (threadIdx.x == 0) {
        for (int i = blockIdx.x; i < 14; i+= gridDim.x) {
            this_slot_info->gpu_symbol_info[i].previous_pkts = 0;
            this_slot_info->gpu_symbol_info[i].previous_waits = 0;
            this_slot_info->gpu_symbol_info[i].previous_wqebbs = 0;
        }
    }
}

void launch_memset_kernel(int num_cells, size_t max_buffer_size, CleanupParams& cleanup, cudaStream_t strm) {

    int num_threads = 1024;
    // max_buffer_size is in bytes
    int blocks = (max_buffer_size + sizeof(uint4)*num_threads - 1) / (sizeof(uint4)*num_threads);
    memset_kernel<<<dim3(blocks, num_cells), num_threads, 0, strm>>>(cleanup);
    cudaError_t result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching memset kernel failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}


__global__ void packet_memcpy_kernel(PacketCopyParams params) {
    int cell = blockIdx.z;
    int flow_num = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < ((params.cell_params[cell].num_pkts_per_flow[flow_num]*params.pkt_size) >> 4)) {
            const int offset = (params.cell_params[cell].pkt_offset[flow_num]*params.pkt_size + sizeof(uint4)*tid) % (kMaxPktsFlow * params.pkt_size); // 16 bytes per load/store

            uint4* d_buffer_addr = reinterpret_cast<uint4*>((uint8_t*)params.cell_params[cell].d_src_addr + (flow_num*kMaxPktsFlow*params.pkt_size)+offset);
            uint4* h_buffer_addr = reinterpret_cast<uint4*>((uint8_t*)params.cell_params[cell].h_dst_addr + (flow_num*kMaxPktsFlow*params.pkt_size)+offset);

              *h_buffer_addr = *d_buffer_addr;
#if 0
            if (tid <10)
                        printf("F%dS%dS%d cell %d tid %d flow_num %d num_pkts_flow=%d pkt_offset=%d pkt_size=%d kMaxPktsFlow =%d d=%p h=%p offset=%d adj d=%p h=%p %08lx\n", params.frame_id, params.subframe_id,params.slot_id,cell, tid,flow_num,params.cell_params[cell].num_pkts_per_flow[flow_num], params.cell_params[cell].pkt_offset[flow_num], params.pkt_size,kMaxPktsFlow,
                                params.cell_params[cell].d_src_addr, params.cell_params[cell].h_dst_addr, offset, d_buffer_addr, h_buffer_addr, *((uint64_t*)d_buffer_addr));
#endif
    }
}


void launch_packet_memcpy_kernel(PacketCopyParams params, int num_cells, cudaStream_t strm)
{
    int num_threads = 512;
    // max_buffer_size is in bytes
    int blocks = (params.max_pkts*params.pkt_size + sizeof(uint4)*num_threads - 1) / (sizeof(uint4)*num_threads);
    packet_memcpy_kernel<<<dim3(blocks,kMaxFlows,num_cells), num_threads, 0, strm>>>(params);
    cudaError_t result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

/* DOCA Tx comm from here */

template <bool enable_gpu_comm_via_cpu=false>
__global__ void gpu_comm_pre_prepare_send_doca(PrepareParams params, struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);
    auto block  = cg::this_thread_block();
    int sym                             = blockIdx.y;
    int cell_id                         = blockIdx.x;

    UplaneSlotInfo_t* this_slot_info         = params.cell_params[cell_id].d_slot_info;
    if (!this_slot_info) return; // for warmup call to this kernel (added for when running SSB only)
    uint32_t* cell_flow_d_info               = params.cell_params[cell_id].flow_d_info;
    uint32_t* cell_flow_sym_d_info           = params.cell_params[cell_id].flow_sym_d_info;
    uint32_t* block_count                                         = params.cell_params[cell_id].block_count;
    FlowPtrInfo* h_flow_hdr_size_info        = params.cell_params[cell_id].flow_hdr_size_info; // host-pinned

    uint32_t*  cell_flow_d_ecpri_seq_id      = params.cell_params[cell_id].flow_d_ecpri_seq_id;
    uint16_t max_num_prb_per_symbol          = params.cell_params[cell_id].max_num_prb_per_symbol;
    size_t mtu = params.cell_params[cell_id].mtu;

    PartialUplaneSlotInfo_t* this_partial_slot_info    = params.cell_params[cell_id].partial_slot_info;
    UplaneSymbolInfoGpu_t* this_slot_gpu_symbol_info = this_slot_info->gpu_symbol_info + sym; //copy to shared

    __shared__ PartialSectionInfoPerMessagePerSymbol_t message_info_per_symbol;
    __shared__ FlowPtrInfo flow_info_across_all_eaxcids[MAX_DL_EAXCIDS];

    //Copy section info for this symbol to shared memory
    cg::memcpy_async(block, &message_info_per_symbol, &(this_partial_slot_info->message_info[sym]), sizeof(PartialSectionInfoPerMessagePerSymbol_t));
    cg::wait(block);

    cg::memcpy_async(block, &(flow_info_across_all_eaxcids[0]), h_flow_hdr_size_info, sizeof(FlowPtrInfo)* MAX_DL_EAXCIDS);
    cg::wait(block);

    __shared__ uint32_t tot_pkts;
    __shared__ uint16_t symbol_num_messages;

    //Initially use partial_slot_info to populate the slot_info necessary in global memory for the prepare kernel to use
    if (threadIdx.x == 0) {
        // this_slot_gpu_symbol_info->ts = this_partial_slot_info->section_info[sym].ts;
        // this_slot_gpu_symbol_info->qp_clock_id = this_partial_slot_info->qp_clock_id; // could avoid replicating across all symbols
        this_slot_gpu_symbol_info->ts = this_partial_slot_info->section_info[sym].ts;
        this_slot_gpu_symbol_info->ptp_ts = this_partial_slot_info->section_info[sym].ptp_ts;
        symbol_num_messages = this_partial_slot_info->section_info[sym].num_messages;
        tot_pkts = 0;
        this_slot_info->last_wqebb_ctrl_idx[sym] = -1;
    }

    for (int tid = threadIdx.x; tid < MAX_DL_EAXCIDS; tid += blockDim.x) {
        cell_flow_sym_d_info[sym*MAX_DL_EAXCIDS + tid] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int count = 0;
        for (int msg_index = 0; msg_index < symbol_num_messages; ++msg_index) {
            count += (int)max(1, message_info_per_symbol.num_packets[msg_index]);
        }
        if(count > MAX_PACKETS_PER_SYM)
        {
            *params.host_pinned_error_flag = 1;
        }
    }

    uint32_t frame_8b_subframe_4b_slot_6b = this_partial_slot_info->frame_8b_subframe_4b_slot_6b;
    uint8_t slot = frame_8b_subframe_4b_slot_6b & 0x3F;
    uint8_t subframe = (frame_8b_subframe_4b_slot_6b >> 6) & 0xF;
    uint8_t frame = (frame_8b_subframe_4b_slot_6b >> 10) & 0xFF;
    size_t prbs_per_pkt_upl = params.cell_params[cell_id].prbs_per_pkt_upl;
    size_t prb_size_upl = params.cell_params[cell_id].prb_size_upl;
    int num_packets  = 0;

    // Go over messages. Currently a thread processes a message
    // FIXME poor access patterns
    for (int msg_index = threadIdx.x; msg_index < symbol_num_messages; msg_index += blockDim.x) {

        int msg_num_prbu = message_info_per_symbol.num_prbu[msg_index];
        int msg_start_prbu = message_info_per_symbol.start_prbu[msg_index];
        int msg_section_id = message_info_per_symbol.section_id[msg_index];
        int msg_num_packets = message_info_per_symbol.num_packets[msg_index];
        uint8_t msg_rb = message_info_per_symbol.rb[msg_index];

        uint8_t eaxcid = message_info_per_symbol.flow_index_info[msg_index];
#ifdef DBG_GPU_COMM
        if (eaxcid >= MAX_DL_EAXCIDS)
        {
            printf("gpu_comm_pre_prepare_send_doca: cell_id %d eaxcid %d out of bounds, max %d\n", cell_id, eaxcid, MAX_DL_EAXCIDS - 1);
            continue;
        }
#endif
        //Get ptr to location where the atomicInc will take place to get pkt_header_index
        uint32_t* msg_flow_d_info = cell_flow_d_info + eaxcid;
        uint32_t* msg_flow_sym_d_info = cell_flow_sym_d_info + sym*MAX_DL_EAXCIDS + eaxcid;

        FlowPtrInfo* msg_flow_hdr_size_info = &flow_info_across_all_eaxcids[eaxcid];

        uint32_t* msg_d_ecpri_seq_id = cell_flow_d_ecpri_seq_id + eaxcid;

        num_packets  = (int)max(1, msg_num_packets);
        int old_tot_packets = atomicAdd(&tot_pkts, num_packets);

        int prb_cnt = 0;
        bool mod_comp_enabled = false;

        if(message_info_per_symbol.mod_comp_params != nullptr)
        {
            mod_comp_enabled = (message_info_per_symbol.mod_comp_params->mod_comp_enabled[msg_index] == 1);
        }

        if(mod_comp_enabled)
        {
            prb_size_upl = message_info_per_symbol.mod_comp_params->prb_size_upl[msg_index];
            prbs_per_pkt_upl = (mtu - ORAN_IQ_HDR_SZ) / prb_size_upl;
        }
        // A thread handles all packets for this message
        for (int pkt_idx_in_msg = 0; pkt_idx_in_msg < num_packets; pkt_idx_in_msg++) {
            int pkt_num_prbs = (int)min(msg_num_prbu - prb_cnt, (int)prbs_per_pkt_upl);
            if(pkt_num_prbs > ORAN_MAX_PRB_X_SECTION && pkt_num_prbs < max_num_prb_per_symbol)
            {
                pkt_num_prbs = ORAN_MAX_PRB_X_SECTION;
            }

            int i = old_tot_packets + pkt_idx_in_msg;
            uint32_t old_hdr_idx;
            uint16_t old_sym_pkts;
            if(enable_gpu_comm_via_cpu)
            {
                old_sym_pkts = atomicAdd(msg_flow_sym_d_info, 1);
                old_hdr_idx = (*msg_flow_d_info + this_partial_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[eaxcid][sym] + old_sym_pkts) % kMaxPktsFlow;
            }
            else
            {
                old_hdr_idx = atomicInc(msg_flow_d_info, kMaxPktsFlow-1);
            }

            //FIXME new sizeof(UplaneSymbolPacketInfo) is 16B, i.e., 128bits.
            uint32_t start_prbu_num_prbu = (msg_rb == 0)? (((int)pkt_num_prbs << 16) |(msg_start_prbu + prb_cnt)):(((int)pkt_num_prbs << 16) |(msg_start_prbu + 2 * prb_cnt)) ;
            this_slot_gpu_symbol_info->packet_info[i].pkt_buff_mkey = (enable_gpu_comm_via_cpu) ? msg_flow_hdr_size_info->cpu_comms_pkt_buff_mkey : msg_flow_hdr_size_info->pkt_buff_mkey;
            this_slot_gpu_symbol_info->packet_info[i].rb   = msg_rb;
            this_slot_gpu_symbol_info->packet_info[i].prb_size   = prb_size_upl;
            // printf("sym %d pkt %d start_prbu %d hdr_stride %d old_hdr_idx %d\n",
            //     sym, i, start_prbu_num_prbu, msg_flow_hdr_size_info->hdr_stride, old_hdr_idx);
            this_slot_gpu_symbol_info->packet_info[i].hdr_stride = msg_flow_hdr_size_info->hdr_stride + old_hdr_idx;

            *((uint32_t*)&this_slot_gpu_symbol_info->packet_info[i].start_prbu) = start_prbu_num_prbu;

            if(enable_gpu_comm_via_cpu)
            {
                this_slot_gpu_symbol_info->packet_info[i].hdr_addr_tx = ((uint64_t)(msg_flow_hdr_size_info->cpu_pkt_addr) + (uint64_t)(this_slot_gpu_symbol_info->packet_info[i].hdr_stride*msg_flow_hdr_size_info->max_pkt_sz));
                //printf("gpu_comm_pre_prepare_send_doca: hdr_addr_tx %lu cpu_pkt_addr %lu\n",this_slot_gpu_symbol_info->packet_info[i].hdr_addr_tx,(uint64_t)(msg_flow_hdr_size_info->cpu_pkt_addr));
                this_slot_gpu_symbol_info->packet_info[i].hdr_addr = (uintptr_t)((uint64_t)(msg_flow_hdr_size_info->gpu_pkt_addr) + (uint64_t)(this_slot_gpu_symbol_info->packet_info[i].hdr_stride*msg_flow_hdr_size_info->max_pkt_sz));
            }
            else
            {
                this_slot_gpu_symbol_info->packet_info[i].hdr_addr_tx = ((uint64_t)(msg_flow_hdr_size_info->gpu_pkt_addr) + (uint64_t)(this_slot_gpu_symbol_info->packet_info[i].hdr_stride*msg_flow_hdr_size_info->max_pkt_sz));
                this_slot_gpu_symbol_info->packet_info[i].hdr_addr =   (uintptr_t)this_slot_gpu_symbol_info->packet_info[i].hdr_addr_tx;      
            }
            

            // if(threadIdx.x == 0)
            // {
            //     auto& pkt_hdr_gpu = this_slot_gpu_symbol_info->packet_info[i].hdr_addr;
            //     printf("F%dS%dS%d cell %d hdr_addr %p DST %02X:%02X:%02X:%02X:%02X:%02X SRC %02X:%02X:%02X:%02X:%02X:%02X comm_buf %p hdr_stride %d + old_hdr_idx %d = %d\n", frame, subframe, slot, cell_id, (uint8_t*)pkt_hdr_gpu,
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[0],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[1],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[2],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[3],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[4],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[5],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[0],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[1],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[2],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[3],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[4],
            //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[5], this_slot_gpu_symbol_info->packet_info[i].comm_buf, msg_flow_hdr_size_info->hdr_stride, old_hdr_idx, this_slot_gpu_symbol_info->packet_info[i].hdr_stride);
            // }
/*
            if(i==0)
            {
                printf("gpu_comm_pre_prepare_send_doca: cell_id %d eaxcid %d threadIdx.x %d sym id %d hdr addr %p",cell_id,eaxcid,threadIdx.x,sym,(void*)this_slot_gpu_symbol_info->packet_info[i].hdr_addr);
            }
*/

            oran_umsg_hdrs *oran = (oran_umsg_hdrs*)(this_slot_gpu_symbol_info->packet_info[i].hdr_addr);
//             printf("pkt %d sym=%d flow=%d starting at index %u with old sym %d cumulative %d stride %d ptr %p\n", old_tot_packets, sym, eaxcid, old_hdr_idx, old_sym_pkts, this_partial_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[eaxcid][sym], this_slot_gpu_symbol_info->packet_info[i].hdr_stride,
// oran);

            //apply rte_cpu_to_be_16 equivalent to tmp_ecpriPayload
            uint16_t tmp_ecpriPayload = (4 + sizeof(oran_umsg_iq_hdr) + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + prb_size_upl*pkt_num_prbs);
            if(mod_comp_enabled)
            {
                tmp_ecpriPayload += ORAN_IQ_SECTION_COMPRESSION_HDR_OVERHEAD;
                this_slot_gpu_symbol_info->packet_info[i].mod_comp_enabled = 1;
            }
            else
            {
                this_slot_gpu_symbol_info->packet_info[i].mod_comp_enabled = 0;
            }
            oran->ecpri.ecpriPayload = ((tmp_ecpriPayload & 0xFF) << 8) | ((tmp_ecpriPayload & 0xFF00) >> 8);

            //There is no constraint on the ecpriSeqId, i.e., it does not need to be incremented frequency first, time second or in any particular order within a symbol.
            //It simply needs to be unique for a flow
            oran->ecpri.ecpriSeqid = (uint8_t)atomicInc(msg_d_ecpri_seq_id, 255); // Because in CPU code ecpri_seq_id is uint8_t

            oran->iq_hdr.frameId      = frame;
            oran->iq_hdr.subframeId   = subframe;
            oran->iq_hdr.slotId       = slot;
            oran->iq_hdr.symbolId     = sym;

            //Some preprocessing currently done on CPU, but the per-cell u-plane preparation time is significantly lower, e.g., 7-8us per cell from ~18-20us.
            oran->sec_hdr.numPrbu     = (pkt_num_prbs > ORAN_MAX_PRB_X_SECTION && pkt_num_prbs == max_num_prb_per_symbol) ? 0 : pkt_num_prbs;
            oran->sec_hdr.startPrbu   = (msg_rb == 0)? (msg_start_prbu + prb_cnt): (msg_start_prbu + 2*prb_cnt) ;
            oran->sec_hdr.rb          = msg_rb;
            oran->sec_hdr.sectionId   = msg_section_id;
            if(mod_comp_enabled)
            {
                oran->comp_hdr[0].udCompMeth = 0b0100;
                oran->comp_hdr[0].udIqWidth = this_slot_gpu_symbol_info->packet_info[i].prb_size / 3;
                oran->comp_hdr[0].reserved = 0;  // Initialize reserved field
            }
            prb_cnt += pkt_num_prbs;
        } // end of loop over packets in message
    } // all messages processed for this symbol

    __syncthreads();

    if (threadIdx.x == 0) {
        this_slot_info->gpu_symbol_info[sym].tot_pkts   = tot_pkts;

        uint32_t wqebbs;
        if (tot_pkts > 0) {
            int whole_wqes = tot_pkts / EMPW_DSEG_MAX_NUM;
            int remainder  = tot_pkts % EMPW_DSEG_MAX_NUM;
            wqebbs = whole_wqes * (((EMPW_DSEG_MAX_NUM + 2) + SEGMENTS_PER_WQE-1) / SEGMENTS_PER_WQE) +
                ((remainder + 2 + SEGMENTS_PER_WQE-1) / SEGMENTS_PER_WQE);
            //printf("WQEBB sym=%d %d\n", sym, wqebbs);
        }
        else {
            wqebbs = 0;
        }

        this_slot_info->gpu_symbol_info[sym].tot_wqebbs = wqebbs;

        for(int idx = sym + 1; idx < kPeerSymbolsInfo; idx++) {
            if (tot_pkts > 0) {
                atomicAdd(&this_slot_info->gpu_symbol_info[idx].previous_waits, 1);
            }

            atomicAdd(&this_slot_info->gpu_symbol_info[idx].previous_wqebbs, wqebbs);
            atomicAdd(&this_slot_info->gpu_symbol_info[idx].previous_pkts, tot_pkts);
        }
    }

  __syncthreads();

  auto prb_pkt = cg::tiled_partition<THREAD_PER_PACKET_PRB_PHASE>(block); //8 threads group
  uint8_t * hdr_ptr;
  struct oran_umsg_hdrs * hdr_format;

  // Walk all PRBs in a packet to update where the PRB pointer should point for the compression kernel
  int packet_id                        = prb_pkt.meta_group_rank();
  int prb_id                       = prb_pkt.thread_rank();
  for (int pkt = packet_id; pkt < this_slot_gpu_symbol_info->tot_pkts; pkt += prb_pkt.meta_group_size()) {
      UplaneSymbolPacketInfo_t *packet_ptr    = &(this_slot_gpu_symbol_info->packet_info[pkt]);
      hdr_ptr                                 = (uint8_t*)packet_ptr->hdr_addr;
      hdr_format                              = (struct oran_umsg_hdrs *) hdr_ptr;
      uint16_t antenna_id                          = (hdr_format->ecpri.ecpriPcid >> 8) & 0x1f;
      uint16_t antenna_idx                         = params.payload_info.eAxCMap[cell_id][antenna_id];
      //printf("gpu_comm_pre_prepare_send_doca: cell_id=%d ecpriPcid=%d antenna_id=%u antenna_idx=%u\n", cell_id, hdr_format->ecpri.ecpriPcid, antenna_id, antenna_idx);
      int packet_hdr_size = ORAN_UMSG_IQ_HDR_SIZE;
      int packet_prb_size = prb_size_upl;
      if(packet_ptr->mod_comp_enabled)
      {
          packet_hdr_size += ORAN_IQ_SECTION_COMPRESSION_HDR_OVERHEAD;
          packet_prb_size = packet_ptr->prb_size;
      }
      if(packet_ptr->rb)
      {

          int prb = packet_ptr->start_prbu + prb_id;
          int  prb2 = packet_ptr->start_prbu + prb_id * 2;
          int offset = sym*params.payload_info.num_antennas[cell_id]*max_num_prb_per_symbol +  antenna_idx*max_num_prb_per_symbol;
          for (;(prb < packet_ptr->start_prbu + packet_ptr->num_prbu) && (prb2 < packet_ptr->start_prbu + packet_ptr->num_prbu * 2); prb += prb_pkt.num_threads(), prb2 += prb_pkt.num_threads() *2 ) {
               params.payload_info.prb_ptrs[cell_id][offset + prb2] = &hdr_ptr[packet_hdr_size + (prb - packet_ptr->start_prbu) * packet_prb_size];
           }
       }
       else
       {
            int offset = sym*params.payload_info.num_antennas[cell_id]*max_num_prb_per_symbol +  antenna_idx*max_num_prb_per_symbol;
           for (int prb = packet_ptr->start_prbu + prb_id; prb < packet_ptr->start_prbu + packet_ptr->num_prbu; prb += prb_pkt.num_threads()) {
               params.payload_info.prb_ptrs[cell_id][offset + prb] = &hdr_ptr[packet_hdr_size + (prb - packet_ptr->start_prbu) * packet_prb_size];
           }
       }
  }

    if(enable_gpu_comm_via_cpu)
    {
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t oldBlock = atomicAdd(block_count, 1);

            if (oldBlock == gridDim.y - 1) {
                for (int flow = 0; flow < this_partial_slot_info->total_num_flows; flow++) {
                    uint32_t* msg_flow_d_info = cell_flow_d_info + flow;
                    uint32_t new_val = (*msg_flow_d_info + this_partial_slot_info->flowInfo_slot.flow_packet_count[flow]) % (kMaxPktsFlow);
                    *msg_flow_d_info = new_val;
                }


                *block_count = 0;

            }
        }
    }
}

int gpucomm_pre_prepare_send(PrepareParams &params, cudaStream_t cstream)
{
    cudaError_t result = cudaSuccess;

    struct cuphy_pti_activity_stats_t activity_stats;
    cuphy_pti_get_record_activity(activity_stats,CUPHY_PTI_ACTIVITY_PREPREP);
    if(params.enable_gpu_comm_via_cpu == 1)
    {
        const bool enable_gpu_comm_via_cpu = true;
        gpu_comm_pre_prepare_send_doca<enable_gpu_comm_via_cpu><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
            params,
            activity_stats);
    }
    else
    {
        const bool enable_gpu_comm_via_cpu = false;
        gpu_comm_pre_prepare_send_doca<enable_gpu_comm_via_cpu><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
            params,
            activity_stats);
    }

    result = cudaGetLastError();
    if(cudaSuccess != result) {
                NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching pre_prepare_send* kernel failed with {}", __FILE__, __LINE__, cudaGetErrorString(result));
                return 1;
        }

    return 0;
}

template <bool enable_gpu_comm_via_cpu=false, bool enable_dl_cqe_tracing=false>
__global__ void gpu_comm_prepare_send_doca(PrepareParams params, struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);
    doca_error_t ret_doca;
    auto block  = cg::this_thread_block();
    uint32_t wqebb_idx_start = 0,wqebb_idx = 0;
    uint16_t prbu_buf_len = 0;
    __shared__ UplaneSymbolInfoGpu_t symbol_info;
    int sym                             = blockIdx.y;
    int cell_id                         = blockIdx.x;
    int packet_id;
    int packet_size;
    doca_gpu_eth_send_flags txq_send_flag = DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;
    uint32_t slot_id_abs = params.subframe_id*2 + params.slot_id;
    bool cqe_gen_en = ((params.payload_info.cqe_trace_cell_mask & (0x1<<cell_id)) && (params.payload_info.cqe_trace_slot_mask & (0x1<<slot_id_abs)));
    struct doca_gpu_dev_eth_txq_wqe *wqe_ptr=NULL;

    if(enable_dl_cqe_tracing)
    {
        if(cqe_gen_en)
        {
            txq_send_flag=DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY;
        }
    }

#ifdef DBG_GPU_COMM
    PartialUplaneSlotInfo_t* this_partial_slot_info    = params.cell_params[cell_id].partial_slot_info;
    uint32_t frame_8b_subframe_4b_slot_6b = this_partial_slot_info->frame_8b_subframe_4b_slot_6b;
    uint8_t slot = frame_8b_subframe_4b_slot_6b & 0x3F;
    uint8_t subframe = (frame_8b_subframe_4b_slot_6b >> 6) & 0xF;
    uint8_t frame = (frame_8b_subframe_4b_slot_6b >> 10) & 0xFF;
#endif

    UplaneSlotInfo_t* this_slot_info    = params.cell_params[cell_id].d_slot_info;
    UplaneSymbolInfoHost* h_slot_info    = params.cell_params[cell_id].h_slot_info;
    struct doca_gpu_eth_txq *eth_txq_gpu = (struct doca_gpu_eth_txq *)params.cell_params[cell_id].eth_txq_gpu;
    int prb_size_upl                    = params.cell_params[cell_id].prb_size_upl;
    uint32_t* block_count                                         = params.cell_params[cell_id].block_count;

    if (this_slot_info == NULL)
        return;

    // Copy symbol info to shm
    cg::memcpy_async(block, &symbol_info, &(this_slot_info->gpu_symbol_info[sym]), sizeof(UplaneSymbolInfoGpu_t));
    cg::wait(block);

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, tot_pkts %d, previous_pkts %d, prev_waits %d vs %d\n",
        //slot, subframe, frame, cell_id, sym, symbol_info.tot_pkts, symbol_info.previous_pkts, symbol_info.previous_waits, this_slot_info->gpu_symbol_info[sym].previous_waits,
        //this_slot_info->gpu_symbol_info + sym);

        printf("slot %d, subframe %d, frame %d, cell %d, symbol %d\n", slot, subframe, frame, cell_id, sym);
    }
    #endif

    if(symbol_info.tot_pkts == 0)
        return;

    wqebb_idx_start=static_cast<uint32_t>(doca_gpu_dev_eth_atomic_read<uint64_t, DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>(&eth_txq_gpu->wqe_pi));
    wqebb_idx_start += symbol_info.previous_wqebbs + symbol_info.previous_waits;
    //printf("gpu_comm_prepare_send_doca: wqebb_idx_start %d sym %d\n", wqebb_idx_start, sym);

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
        printf("[gpu_comm_prepare_send_doca]slot %d, subframe %d, frame %d, cell %d, symbol %d, will continue after wqebb_idx_start %d\n", slot, subframe, frame, cell_id, sym, wqebb_idx_start);
#endif

    // One CTA per symbol/cell, so the first thread needs to set the sending timestamp
    if (threadIdx.x == 0) {
        wqe_ptr=doca_gpu_dev_eth_txq_get_wqe_ptr(eth_txq_gpu, wqebb_idx_start & 0xFFFF);
        ret_doca = doca_gpu_dev_eth_txq_wqe_prepare_wait_time(eth_txq_gpu,wqe_ptr,wqebb_idx_start & 0xFFFF,symbol_info.ts,txq_send_flag);
        if (ret_doca != DOCA_SUCCESS) {
            printf("Error %d doca gpunetio enqueue wait on time\n", ret_doca);
        }
        if(enable_dl_cqe_tracing){
            if(cqe_gen_en)
            {
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].time        = symbol_info.ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].ptp_time    = symbol_info.ptp_ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].frame_id    = params.frame_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].subframe_id = params.subframe_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].slot_id     = params.slot_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].packet_size = 0;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].sym         = sym;
#ifdef DBG_GPU_COMM
                printf("[1] slot %d, subframe %d, frame %d, cell %d, symbol %d, wqebb_idx_start %d, ts %lx ptp_ts %lu\n",
                        slot, subframe, frame, cell_id, sym, wqebb_idx_start, symbol_info.ts, symbol_info.ptp_ts);
#endif
            }
        }
#ifdef DBG_GPU_COMM
        if (ret_doca != DOCA_SUCCESS)
            printf("doca_gpu_device_wait_time_weak_thread returned %d\n", ret_doca);
#endif
    }

    // Step over the first WAIT
    wqebb_idx_start++;

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
        printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, will start wqe creation for %d packets\n",
                slot, subframe, frame, cell_id, sym, symbol_info.tot_pkts);
#endif

    /* Create the WQEs for each packet */
    const uint32_t last_wqe_idx = symbol_info.tot_pkts / EMPW_DSEG_MAX_NUM;
    constexpr uint32_t full_bbs = ((EMPW_DSEG_MAX_NUM + 2) + SEGMENTS_PER_WQE-1) /SEGMENTS_PER_WQE;
    int32_t pkt_wqe_idx = -1;
    int32_t rel_pkt_id = -1;
    uint32_t pkts_in_wqe,rel_wqebb_idx;

    for (packet_id = block.thread_rank(); packet_id < symbol_info.tot_pkts; packet_id += block.size()) {
        UplaneSymbolPacketInfo_t *packet_ptr = &(symbol_info.packet_info[packet_id]);
        prb_size_upl = packet_ptr->prb_size;
        prbu_buf_len = packet_ptr->num_prbu * prb_size_upl;
        packet_size = ORAN_UMSG_IQ_HDR_SIZE + prbu_buf_len;
        if(packet_ptr->mod_comp_enabled == 1)
        {
            packet_size += ORAN_IQ_SECTION_COMPRESSION_HDR_OVERHEAD;
        }
        pkt_wqe_idx             = packet_id / EMPW_DSEG_MAX_NUM;
        rel_pkt_id              = packet_id % EMPW_DSEG_MAX_NUM;
        rel_wqebb_idx           = (pkt_wqe_idx * full_bbs) + ((rel_pkt_id + 2) / SEGMENTS_PER_WQE);
        //uint32_t ds_idx         = (rel_pkt_id + 2) % SEGMENTS_PER_WQE;
        pkts_in_wqe             = (pkt_wqe_idx == last_wqe_idx) ? symbol_info.tot_pkts % EMPW_DSEG_MAX_NUM : EMPW_DSEG_MAX_NUM;
        wqebb_idx               = (wqebb_idx_start + rel_wqebb_idx) & 0xFFFF;
        if(enable_dl_cqe_tracing){
            if(cqe_gen_en)
            {
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].time        = symbol_info.ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].ptp_time    = symbol_info.ptp_ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].frame_id    = params.frame_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].subframe_id = params.subframe_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].slot_id     = params.slot_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].packet_size = packet_size;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx].sym         = sym;
#ifdef DBG_GPU_COMM
                printf("[2] slot %d, subframe %d, frame %d, cell %d, symbol %d, wqebb_idx %d, rel_pkt_id %d, ts %lx ptp_ts %lu\n",
                        slot, subframe, frame, cell_id, sym, wqebb_idx, rel_pkt_id, symbol_info.ts, symbol_info.ptp_ts);
#endif
            }
        }
        // if(threadIdx.x == 0)
        // {
        //     auto& pkt_hdr_gpu = packet_ptr->hdr_addr;
        //     printf("prepare_send_doca cell %d DST %02X:%02X:%02X:%02X:%02X:%02X SRC %02X:%02X:%02X:%02X:%02X:%02X \n", cell_id,
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[0],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[1],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[2],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[3],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[4],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->dst_addr.addr_bytes[5],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[0],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[1],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[2],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[3],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[4],
        //     ((struct oran_ether_hdr*)(pkt_hdr_gpu))->src_addr.addr_bytes[5]);
        // }
        wqe_ptr=doca_gpu_dev_eth_txq_get_wqe_ptr(eth_txq_gpu, wqebb_idx);
        ret_doca = doca_gpu_dev_eth_txq_wqe_prepare_send_empw(eth_txq_gpu,wqe_ptr,wqebb_idx,pkts_in_wqe, rel_pkt_id,(uint64_t)packet_ptr->hdr_addr_tx,packet_ptr->pkt_buff_mkey,packet_size,txq_send_flag);

#ifdef DBG_GPU_COMM
        if (ret_doca != DOCA_SUCCESS) {
            printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, packet_id %d, wqe_new_send returned %d\n",
                    slot, subframe, frame, cell_id, sym, packet_id, ret_doca);
        }
#endif
    }

    // If this is the last packet (and therefor the last WQE), we need to save off the WQEBB index
    // to use in the trigger kernel
    if(enable_gpu_comm_via_cpu)
    {
        if ((last_wqe_idx == pkt_wqe_idx)   &&
            (rel_pkt_id == 0)) {
                this_slot_info->last_wqebb_ctrl_idx[sym] = wqebb_idx;
                h_slot_info->wqebbs_per_cell[sym]        = this_slot_info->gpu_symbol_info[sym].tot_wqebbs + (this_slot_info->gpu_symbol_info[sym].tot_pkts > 0 ? 1 : 0);
                //h_slot_info->sq_db_rec                                = eth_txq_gpu->sq_db_rec; //TODO: Check for the replacement in DOCA 3.2
                h_slot_info->wqe_mask                                  = eth_txq_gpu->wqe_mask;
                uint32_t wqe_idx = wqebb_idx & eth_txq_gpu->wqe_mask;
                h_slot_info->test_wqe[sym] = wqe_idx;
                //h_slot_info->wqe[sym] = *((uint64_t *)(&(((struct mlx5_wqe_eth *)eth_txq_gpu->wqe_addr)[wqe_idx]))); //TODO: Check for the replacement in DOCA 3.2
                //h_slot_info->wqe_addr[sym] = &(((struct mlx5_wqe_eth *)eth_txq_gpu->wqe_addr)[wqe_idx]); //TODO: Check for the replacement in DOCA 3.2
                //h_slot_info->sq_db = eth_txq_gpu->sq_db; //TODO: Check for the replacement in DOCA 3.2
        }

        __syncthreads();
        __threadfence();
        if (threadIdx.x == 0) {
            uint32_t oldBlock = atomicAdd(block_count, 1);

            if (oldBlock == params.cell_params[cell_id].partial_slot_info->syms_with_packets - 1) {
                    const uint32_t wqe_pi = static_cast<uint32_t>(doca_gpu_dev_eth_atomic_read<uint64_t, DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>(&eth_txq_gpu->wqe_pi));

                    h_slot_info->old_wqe_pi = wqe_pi;
                    uint32_t wqebbs_per_cell = 0;
                    for (int isym = 0; isym < 14; isym++) {
                        wqebbs_per_cell += this_slot_info->gpu_symbol_info[isym].tot_wqebbs + (this_slot_info->gpu_symbol_info[isym].tot_pkts > 0 ? 1 : 0);
                    }
                    
                    // Submit all WQEs for this cell by calling submit_proxy with the final producer index
                    const uint32_t final_wqe_pi = wqe_pi + wqebbs_per_cell;
                    doca_gpu_dev_eth_txq_submit_proxy<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(eth_txq_gpu, final_wqe_pi);
                    //printf("gpu_comm_prepare_send_doca: wqe_pi %d wqebbs_per_cell %d final_wqe_pi %d cell %d sym %d\n", wqe_pi, wqebbs_per_cell, final_wqe_pi, cell_id, sym);
                *block_count = 0;
            }
        }
    }
    else
    {
        if ((last_wqe_idx == pkt_wqe_idx)   &&
            (rel_pkt_id == 0)               &&
            (sym == params.cell_params[cell_id].partial_slot_info->last_sym_with_packets)) {
                this_slot_info->last_wqebb_ctrl_idx[0] = wqebb_idx;
        }
    }
}

__global__ void gpu_comm_prepare_send_doca_nonEmpw(PrepareParams params, struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);
    doca_error_t ret_doca;
    auto block  = cg::this_thread_block();
    uint32_t wqebb_idx_start = 0;
    uint16_t prbu_buf_len = 0;
    __shared__ UplaneSymbolInfoGpu_t symbol_info;
    int sym                             = blockIdx.y;
    int cell_id                         = blockIdx.x;
    int packet_id;
    doca_gpu_eth_send_flags txq_send_flag = DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;
    uint32_t slot_id_abs = params.subframe_id*2 + params.slot_id;
    bool cqe_gen_en = ((params.payload_info.cqe_trace_cell_mask & (0x1<<cell_id)) && (params.payload_info.cqe_trace_slot_mask & (0x1<<slot_id_abs)));
    struct doca_gpu_dev_eth_txq_wqe *wqe_ptr=NULL;

    if(params.payload_info.enable_dl_cqe_tracing)
    {
        if(cqe_gen_en)
        {
            txq_send_flag=DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY;
        }
    }

#ifdef DBG_GPU_COMM
    PartialUplaneSlotInfo_t* this_partial_slot_info    = params.cell_params[cell_id].partial_slot_info;
    uint32_t frame_8b_subframe_4b_slot_6b = this_partial_slot_info->frame_8b_subframe_4b_slot_6b;
    uint8_t slot = frame_8b_subframe_4b_slot_6b & 0x3F;
    uint8_t subframe = (frame_8b_subframe_4b_slot_6b >> 6) & 0xF;
    uint8_t frame = (frame_8b_subframe_4b_slot_6b >> 10) & 0xFF;
#endif

    UplaneSlotInfo_t* this_slot_info    = params.cell_params[cell_id].d_slot_info;
    struct doca_gpu_eth_txq *eth_txq_gpu = (struct doca_gpu_eth_txq *)params.cell_params[cell_id].eth_txq_gpu;
    int prb_size_upl                    = params.cell_params[cell_id].prb_size_upl;

    if (this_slot_info == NULL)
        return;

    // Copy symbol info to shm
    cg::memcpy_async(block, &symbol_info, &(this_slot_info->gpu_symbol_info[sym]), sizeof(UplaneSymbolInfoGpu_t));
    cg::wait(block);

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, tot_pkts %d, previous_pkts %d, prev_waits %d vs %d\n",
        //slot, subframe, frame, cell_id, sym, symbol_info.tot_pkts, symbol_info.previous_pkts, symbol_info.previous_waits, this_slot_info->gpu_symbol_info[sym].previous_waits,
        //this_slot_info->gpu_symbol_info + sym);

        printf("slot %d, subframe %d, frame %d, cell %d, symbol %d\n", slot, subframe, frame, cell_id, sym);
    }
    #endif

    if(symbol_info.tot_pkts == 0)
        return;

    wqebb_idx_start=static_cast<uint32_t>(doca_gpu_dev_eth_atomic_read<uint64_t, DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>(&eth_txq_gpu->wqe_pi));
    wqebb_idx_start += symbol_info.previous_pkts + symbol_info.previous_waits;

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
        printf("[gpu_comm_prepare_send_doca_nonEmpw]slot %d, subframe %d, frame %d, cell %d, symbol %d, will continue after wqebb_idx_start %d\n", slot, subframe, frame, cell_id, sym, wqebb_idx_start);
#endif

    // One CTA per symbol/cell, so the first thread needs to set the sending timestamp
    if (threadIdx.x == 0) {
        wqe_ptr=doca_gpu_dev_eth_txq_get_wqe_ptr(eth_txq_gpu, wqebb_idx_start & 0xFFFF);
        ret_doca = doca_gpu_dev_eth_txq_wqe_prepare_wait_time(eth_txq_gpu,wqe_ptr,wqebb_idx_start & 0xFFFF,symbol_info.ts,txq_send_flag);        
        if (ret_doca != DOCA_SUCCESS) {
            printf("Error %d doca gpunetio enqueue wait on time\n", ret_doca);
        }
        if(params.payload_info.enable_dl_cqe_tracing){
            if(cqe_gen_en)
            {
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].time        = symbol_info.ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].ptp_time    = symbol_info.ptp_ts;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].frame_id    = params.frame_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].subframe_id = params.subframe_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].slot_id     = params.slot_id;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].packet_size = 0;
                params.cell_params[cell_id].start_dbg_info[wqebb_idx_start].sym         = sym;
#ifdef DBG_GPU_COMM
                printf("[1] slot %d, subframe %d, frame %d, cell %d, symbol %d, wqebb_idx_start %d, ts %lx ptp_ts %lu\n",
                        slot, subframe, frame, cell_id, sym, wqebb_idx_start, symbol_info.ts, symbol_info.ptp_ts);
#endif
            }
        }
#ifdef DBG_GPU_COMM
        if (ret_doca != DOCA_SUCCESS)
            printf("doca_gpu_device_wait_time_weak_thread returned %d\n", ret_doca);
#endif
    }

    if((sym == params.cell_params[cell_id].partial_slot_info->last_sym_with_packets) && (threadIdx.x == 0))
    {
        if(params.enable_gpu_comm_via_cpu == 1)
            this_slot_info->last_wqebb_ctrl_idx[sym]=wqebb_idx_start+symbol_info.tot_pkts;
        else
            this_slot_info->last_wqebb_ctrl_idx[0]=wqebb_idx_start+symbol_info.tot_pkts;
    }


    // Step over the first WAIT
    wqebb_idx_start++;

#ifdef DBG_GPU_COMM
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
        printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, will start wqe creation for %d packets\n",
                slot, subframe, frame, cell_id, sym, symbol_info.tot_pkts);
#endif

    /* Create the WQEs for each packet */
    int wqe_idx;

    for (packet_id = block.thread_rank(); packet_id < symbol_info.tot_pkts; packet_id += block.size()) {
        UplaneSymbolPacketInfo_t *packet_ptr = &(symbol_info.packet_info[packet_id]);
        prbu_buf_len = packet_ptr->num_prbu * prb_size_upl;
        wqe_idx = (wqebb_idx_start + packet_id) & 0xFFFF;
        wqe_ptr=doca_gpu_dev_eth_txq_get_wqe_ptr(eth_txq_gpu, wqe_idx);
        ret_doca = doca_gpu_dev_eth_txq_wqe_prepare_send(eth_txq_gpu,wqe_ptr,wqe_idx,(uint64_t)packet_ptr->hdr_addr_tx,packet_ptr->pkt_buff_mkey,ORAN_UMSG_IQ_HDR_SIZE + prbu_buf_len,txq_send_flag);        

#ifdef DBG_GPU_COMM
        if (ret_doca != DOCA_SUCCESS) {
            printf("slot %d, subframe %d, frame %d, cell %d, symbol %d, packet_id %d, wqe_new_send returned %d\n",
                    slot, subframe, frame, cell_id, sym, packet_id, ret_doca);
        }
#endif
    }
}


int gpucomm_prepare_send(PrepareParams &params, cudaStream_t cstream)
{
    cudaError_t result = cudaSuccess;

    struct cuphy_pti_activity_stats_t activity_stats;
    cuphy_pti_get_record_activity(activity_stats,CUPHY_PTI_ACTIVITY_PREP);
    if(!params.payload_info.disable_empw){
        if(params.enable_gpu_comm_via_cpu == 1)
        {
            const bool enable_gpu_comm_via_cpu = true;
            if(params.payload_info.enable_dl_cqe_tracing)
            {
                const bool enable_dl_cqe_tracing = true;
                gpu_comm_prepare_send_doca<enable_gpu_comm_via_cpu, enable_dl_cqe_tracing><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
                    params,
                    activity_stats);
            }
            else
            {
                const bool enable_dl_cqe_tracing = false;
                gpu_comm_prepare_send_doca<enable_gpu_comm_via_cpu, enable_dl_cqe_tracing><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
                    params,
                    activity_stats);
            }
        }
        else
        {
            const bool enable_gpu_comm_via_cpu = false;
            if(params.payload_info.enable_dl_cqe_tracing)
            {
                const bool enable_dl_cqe_tracing = true;
                gpu_comm_prepare_send_doca<enable_gpu_comm_via_cpu, enable_dl_cqe_tracing><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
                    params,
                    activity_stats);
            }
            else
            {
                const bool enable_dl_cqe_tracing = false;
                gpu_comm_prepare_send_doca<enable_gpu_comm_via_cpu, enable_dl_cqe_tracing><<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
                    params,
                    activity_stats);
            }
        }
    }
    else
    {
        gpu_comm_prepare_send_doca_nonEmpw<<<dim3(params.num_cells, kPeerSymbolsInfo), 128, 0, cstream>>>(
               params,
               activity_stats);
    }

    result = cudaGetLastError();
    if(cudaSuccess != result) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching gpu_comm_prepare* kernel failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }
    return 0;
}
// Simple kernel to ring doorbell for each cell
// Each thread handles one cell based on threadIdx.x
__global__ void gpucomm_ring_doorbell_per_cell(doca_gpu_eth_txq** txq_handlers, const uint32_t* wqe_indices, const uint32_t num_cells)
{
    const uint32_t cell_id = threadIdx.x;
    
    if (cell_id < num_cells) {
        doca_gpu_dev_eth_txq_submit_proxy<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(txq_handlers[cell_id], wqe_indices[cell_id]);
        //printf("gpucomm_ring_doorbell_per_cell cell_id %d wqe_indices[cell_id] %d\n", cell_id, wqe_indices[cell_id]);
    }
}

// Host function to launch the ring doorbell kernel
int gpucomm_ring_doorbell_for_cells(doca_gpu_eth_txq** d_txq_handlers, const uint32_t* d_wqe_indices, const uint32_t num_cells, cudaStream_t cstream)
{
    if (num_cells == 0) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] num_cells is 0", __FILE__, __LINE__);
        return -1;
    }
    
    if (num_cells > 1024) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] num_cells {} exceeds maximum block size", __FILE__, __LINE__, num_cells);
        return -1;
    }
    
    // Launch with 1 block and num_cells threads (CTA size based on number of cells)
    gpucomm_ring_doorbell_per_cell<<<1, num_cells, 0, cstream>>>(d_txq_handlers, d_wqe_indices, num_cells);
    
    const cudaError_t result = cudaGetLastError();
    if (cudaSuccess != result) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching gpucomm_ring_doorbell_per_cell kernel failed with {}", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }
    
    return 0;
}

__global__ void gpu_comm_trigger_send_doca_cx6(TriggerParams params, const uint32_t ncells, uint32_t *wait_flag, const uint32_t wait_val,struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);

    doca_gpu_dev_eth_txq_update_dbr(params.cell_params[threadIdx.x].eth_txq_gpu,params.cell_params[threadIdx.x].d_slot_info->last_wqebb_ctrl_idx[0]);
    // Assuming less than 32 cells
    __syncwarp();

    if (threadIdx.x == 0) {
        while(DOCA_GPUNETIO_VOLATILE(*wait_flag) != wait_val);
        for (int i = 0; i < blockDim.x; i++)
            doca_gpu_dev_eth_txq_ring_db(params.cell_params[i].eth_txq_gpu,params.cell_params[i].d_slot_info->last_wqebb_ctrl_idx[0]); 
            // doca_gpu_device_send_trigger_thread_cx6(params.cell_params[i].txq_info_gpu);
    }
}

__global__ void gpu_comm_trigger_send_doca_cx7(TriggerParams params, const uint32_t ncells, uint32_t *wait_flag, const uint32_t wait_val, struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);

    //printf("gpu_comm_trigger_send_doca_cx7 threadIdx.x %d params.cell_params[i].d_slot_info->last_wqebb_ctrl_idx[0] %d\n", threadIdx.x, params.cell_params[threadIdx.x].d_slot_info->last_wqebb_ctrl_idx[0]);
    uint32_t wqebbs_per_cell;
    wqebbs_per_cell = params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS -1].previous_wqebbs +
                    params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].tot_wqebbs +
                    params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].previous_waits +
                    (params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].tot_pkts > 0 ? 1 : 0);
    doca_gpu_dev_eth_txq_update_dbr(params.cell_params[threadIdx.x].eth_txq_gpu,wqebbs_per_cell);
    doca_gpu_dev_eth_atomic_add<uint64_t, DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>(&params.cell_params[threadIdx.x].eth_txq_gpu->wqe_pi, wqebbs_per_cell);

    // Assuming less than 32 cells
    if (wait_flag && threadIdx.x == 0)
        while(DOCA_GPUNETIO_VOLATILE(*wait_flag) != wait_val);
    __syncwarp();

#ifdef __aarch64__
    // Temporary workaround for https://redmine.mellanox.com/issues/3638896
    if (threadIdx.x == 0) {
        while(DOCA_GPUNETIO_VOLATILE(*wait_flag) != wait_val);
        for (int i = 0; i < blockDim.x; i++)
            doca_gpu_dev_eth_txq_ring_db(params.cell_params[i].eth_txq_gpu,params.cell_params[i].eth_txq_gpu->wqe_pi); 
            // doca_gpu_device_send_trigger_thread_cx6(params.cell_params[i].txq_info_gpu);
    }
#else
    doca_gpu_dev_eth_txq_ring_db(params.cell_params[threadIdx.x].eth_txq_gpu,params.cell_params[threadIdx.x].eth_txq_gpu->wqe_pi);
#endif
}

__global__ void gpu_comm_trigger_send_doca_nonEmpw_cx7(TriggerParams params, const uint32_t ncells, uint32_t *wait_flag, const uint32_t wait_val, struct cuphy_pti_activity_stats_t activity_stats)
{
    CuphyPtiRecordStartStopTimeScope scoped_record_start_stop_time(activity_stats);
    uint32_t wqebbs_per_cell;
    wqebbs_per_cell = params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS -1].previous_pkts +
                    params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].tot_pkts +
                    params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].previous_waits +
                    (params.cell_params[threadIdx.x].d_slot_info->gpu_symbol_info[SLOT_NUM_SYMS-1].tot_pkts > 0 ? 1 : 0);    
    doca_gpu_dev_eth_txq_update_dbr(params.cell_params[threadIdx.x].eth_txq_gpu,wqebbs_per_cell);
    doca_gpu_dev_eth_atomic_add<uint64_t, DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>(&params.cell_params[threadIdx.x].eth_txq_gpu->wqe_pi, wqebbs_per_cell);
    // Assuming less than 32 cells

    if (wait_flag && threadIdx.x == 0)
        while(DOCA_GPUNETIO_VOLATILE(*wait_flag) != wait_val);
    __syncwarp();

#ifdef __aarch64__
    // Temporary workaround for https://redmine.mellanox.com/issues/3638896
    if (threadIdx.x == 0) {
        while(DOCA_GPUNETIO_VOLATILE(*wait_flag) != wait_val);
        for (int i = 0; i < blockDim.x; i++)
            doca_gpu_dev_eth_txq_ring_db(params.cell_params[i].eth_txq_gpu,params.cell_params[i].d_slot_info->last_wqebb_ctrl_idx[0]);
            // doca_gpu_device_send_trigger_thread_cx6(params.cell_params[i].txq_info_gpu);
    }
#else
    doca_gpu_dev_eth_txq_ring_db(params.cell_params[threadIdx.x].eth_txq_gpu,params.cell_params[threadIdx.x].d_slot_info->last_wqebb_ctrl_idx[0]);
#endif
}


int gpucomm_trigger_send(TriggerParams &params, cudaStream_t cstream, bool cx6)
{
    cudaError_t result = cudaSuccess;
    struct cuphy_pti_activity_stats_t activity_stats;
    cuphy_pti_get_record_activity(activity_stats,CUPHY_PTI_ACTIVITY_TRIGGER);
    if (cx6 == true)
        gpu_comm_trigger_send_doca_cx6<<<1, params.num_cells, 0, cstream>>>(
            params,
            params.num_cells,
            params.ready_flag,
            params.wait_val,
            activity_stats);
    else
    {
        if(!params.disable_empw)
        {
            gpu_comm_trigger_send_doca_cx7<<<1, params.num_cells, 0, cstream>>>(
                params,
                params.num_cells,
                params.ready_flag,
                params.wait_val,
                activity_stats);
        }
        else
        {
            gpu_comm_trigger_send_doca_nonEmpw_cx7<<<1, params.num_cells, 0, cstream>>>(
                params,
                params.num_cells,
                params.ready_flag,
                params.wait_val,
                activity_stats);
        }
    }

    result = cudaGetLastError();
    if(cudaSuccess != result) {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] launching gpu_comm_trigger* kernel failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }
    return 0;
}

void force_loading_gpu_comm_kernels()
{
    std::array<void*, 14> gpu_comm_kernels = {
     (void*)warmup,
     (void*)memset_kernel,
     (void*)packet_memcpy_kernel,
     (void*)gpu_comm_pre_prepare_send_doca<true>,
     (void*)gpu_comm_pre_prepare_send_doca<false>,
     (void*)gpu_comm_prepare_send_doca<false, false>,
     (void*)gpu_comm_prepare_send_doca<false, true>,
     (void*)gpu_comm_prepare_send_doca<true, false>,
     (void*)gpu_comm_prepare_send_doca<true, true>,
     (void*)gpu_comm_prepare_send_doca_nonEmpw,
     (void*)gpucomm_ring_doorbell_per_cell,
     (void*)gpu_comm_trigger_send_doca_cx6,
     (void*)gpu_comm_trigger_send_doca_cx7,
     (void*)gpu_comm_trigger_send_doca_nonEmpw_cx7};

     for(auto& gpu_comm_kernel : gpu_comm_kernels)
     {
         cudaFuncAttributes attr;
         cudaError_t e = cudaFuncGetAttributes(&attr, static_cast<const void*>(gpu_comm_kernel));
         if(cudaSuccess != e)
         {
             NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cudaFuncGetAttributes call failed with {} ", __FILE__, __LINE__, cudaGetErrorString(e));
         }
     }
}


} // namespace aerial_fh

