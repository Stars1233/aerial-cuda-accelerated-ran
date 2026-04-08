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

#ifndef AERIAL_FH_INSTANCE_HPP__
#define AERIAL_FH_INSTANCE_HPP__

#include <vector>
#include <cuda/std/array>
#include "aerial-fh-driver/api.hpp"
#include "fh_stats_dump.hpp"
#include "dpdk.hpp"
#include "gpu.hpp"
#include "utils.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <doca_gpunetio.h>
#include <doca_log.h>
#include "doca_obj.hpp"
#pragma GCC diagnostic pop

namespace aerial_fh
{
class GpuMempool;
class Nic;
class PdumpClient;
class Peer;
class Txq;
class RivermaxPrx;
struct TxRequestUplane;
struct TxRequestCplane;

using TxRequestCallback = void (*)(TxRequestUplane* tx_request);                 //!< TX completion callback function type
using UniqueGpuMap      = std::unordered_map<GpuId, std::unique_ptr<Gpu>>;   //!< Map of GPU ID to GPU instance

/**
 * U-plane TX request descriptor
 *
 * Encapsulates mbufs and metadata for transmitting U-plane packets.
 * Used for asynchronous TX with completion notifications.
 *
 * The mbuf array acts as a ring buffer with two indices:
 * - prepare_index: where to write next prepared packets
 * - send_index: where the current batch to send starts
 * - Batch size = prepare_index - send_index
 */
struct TxRequestUplane
{
    Peer*     peer;                          //!< Destination peer
    size_t    preallocated_mbuf_count;       //!< Total number of pre-allocated mbufs
    size_t    prepare_index;                 //!< Index where next prepare operation writes
    size_t    send_index;                    //!< Index where current prepared batch starts
    rte_mbuf* mbufs[kTxPktBurstUplane];      //!< Array of mbuf pointers
    Ns        tx_window_start;               //!< TX window start time (nanoseconds)
};

/**
 * C-plane TX request descriptor
 *
 * Encapsulates mbufs for transmitting C-plane packets.
 */
struct TxRequestCplane
{
    Peer*     peer;                        //!< Destination peer
    size_t    mbuf_count;                  //!< Number of mbufs in request
    rte_mbuf* mbufs[kTxPktBurstCplane];    //!< Array of mbuf pointers
};

/**
 * U-plane PRB box information
 *
 * Describes a "box" of PRBs for GPU-accelerated packet preparation.
 * Used in massive MIMO scenarios with beamforming.
 */
typedef struct UplanePrbBoxInfo {
    uint32_t hdr_stride;                   //!< Header stride in bytes
    struct doca_tx_buf *tx_buf;            //!< DOCA TX buffer

    uintptr_t pkt_addr;                    //!< Packet buffer address
    uint32_t pkt_lkey;                     //!< RDMA local key
    uint32_t pkt_num;                      //!< Number of packets in box
    uint8_t start_sequence_id;             //!< Starting eCPRI sequence ID
    uint32_t previous_pkts;                //!< Packets sent in previous boxes
    struct UPlaneSectionInfo section_info; //!< Section info for this box
 } UplanePrbBoxInfo_t;

/**
 * U-plane symbol packet information for GPU communication
 *
 * Contains buffer and PRB information for a single packet within a symbol.
 */
typedef struct UplaneSymbolPacketInfo {
    uintptr_t hdr_addr;                             //!< Header address
    uint64_t hdr_addr_tx;                           //!< Header address for transmission
    uint32_t hdr_stride;                            //!< Header stride in bytes
    uint32_t pkt_buff_mkey;                         //!< mkey value of the buffer

    uint16_t  start_prbu;                           //!< Starting PRB
    uint16_t  num_prbu;                             //!< Number of PRBs
    uint16_t  rb;                                   //!< Resource block indicator
    uint8_t   prb_size;                             //!< PRB size in bytes
    uint8_t   mod_comp_enabled;                     //!< Modulation compression enabled flag
} UplaneSymbolPacketInfo_t;

/**
 * U-plane symbol information
 *
 * Contains all packet/PRB information for transmitting one OFDM symbol.
 */
typedef struct UplaneSymbolInfo {
    UplanePrbBoxInfo_t prb_box_info[kPeerSymbolPrbBox];  //!< PRB box info array
    uint32_t prb_box_num;                                 //!< Number of boxes per symbol
    uint32_t tot_pkts;                                    //!< Total packets in symbol
    uint32_t previous_pkts;                               //!< Packets sent in previous symbols
    uint32_t previous_waits;                              //!< Previous wait count
    uint16_t symbol_id;                                   //!< Symbol ID (0-13)
    uint64_t ts;                                          //!< Timestamp
} UplaneSymbolInfo_t;

#define MAX_PACKETS_PER_ANT 16                                      //!< Maximum packets per antenna
#define MAX_PACKETS_PER_SYM (API_MAX_ANTENNAS * MAX_PACKETS_PER_ANT)  //!< Maximum packets per symbol

/**
 * U-plane symbol information for GPU
 *
 * Optimized for GPU access with packet-level granularity.
 */
typedef struct UplaneSymbolInfoGpu {
    UplaneSymbolPacketInfo_t packet_info[MAX_PACKETS_PER_SYM];  //!< Packet info array
    uint32_t tot_pkts;                                           //!< Total packets in symbol
    uint32_t tot_wqebbs;                                         //!< Total Work Queue Element Basic Blocks
    uint32_t previous_pkts;                                      //!< Packets sent in previous symbols
    uint32_t previous_waits;                                     //!< Previous wait count
    uint32_t previous_wqebbs;                                    //!< Previous WQEBBs count
    uint16_t symbol_id;                                          //!< Symbol ID (0-13)
    uint64_t ts;                                                 //!< Timestamp
    uint64_t ptp_ts;                                             //!< PTP timestamp
} UplaneSymbolInfoGpu_t;

/**
 * U-plane symbol information (host-side)
 *
 * Contains send queue (SQ) doorbell and work queue element (WQE) information
 * for managing GPU-accelerated transmission from host.
 */
typedef struct UplaneSymbolInfoHost {
    uintptr_t sq_db;                        //!< Send queue doorbell address
    uint64_t wqe[14];                       //!< Work queue elements (one per symbol)
    void * wqe_addr[14];                    //!< WQE addresses
    uint64_t test_wqe[14];                  //!< Test WQE array
    uint32_t old_wqe_pi;                    //!< Old WQE producer index
    uint32_t wqe_mask;                      //!< WQE mask
    uintptr_t sq_db_rec;                    //!< Send queue doorbell record
    int32_t last_wqebb_ctrl_idx[14];        //!< Last WQEBB control index per symbol
    uint32_t wqebbs_per_cell[14];           //!< WQEBBs per cell per symbol
} UplaneSlotInfoHost_t;

#define TMP_MAX_MESSAGES_PER_SYMBOL 240  //!< Maximum messages per symbol (temporary, needs optimization)

/**
 * Partial section info per symbol
 *
 * Summary information for a symbol's sections, used for partial slot info.
 */
typedef struct PartialSectionInfoPerSymbol {
    uint32_t wci;                     //!< WQE control information
    uint16_t num_messages;            //!< Number of messages in symbol
    uint64_t ptp_ts;                  //!< PTP timestamp
    uint64_t ts;                      //!< System timestamp
    uint16_t num_packets;             //!< Number of packets
    uint16_t cumulative_packets;      //!< Cumulative packet count
} PartialSectionInfoPerSymbol_t;

/**
 * Modulation compression parameters per message per symbol
 */
typedef struct ModCompPartialSectionInfoPerMessagePerSymbol {
    uint8_t prb_size_upl[TMP_MAX_MESSAGES_PER_SYMBOL];      //!< PRB size per message
    uint8_t mod_comp_enabled[TMP_MAX_MESSAGES_PER_SYMBOL];  //!< Modulation compression enabled flag per message
} ModCompPartialSectionInfoPerMessagePerSymbol_t;

/**
 * Partial section info per message per symbol
 *
 * Detailed section information for each message within a symbol.
 */
typedef struct PartialSectionInfoPerMessagePerSymbol {
    uint16_t num_prbu[TMP_MAX_MESSAGES_PER_SYMBOL];         //!< Number of PRBs per message
    uint16_t start_prbu[TMP_MAX_MESSAGES_PER_SYMBOL];       //!< Starting PRB per message
    uint16_t rb[TMP_MAX_MESSAGES_PER_SYMBOL];               //!< Resource block indicator per message
    uint16_t section_id[TMP_MAX_MESSAGES_PER_SYMBOL];       //!< Section ID per message
    uint16_t num_packets[TMP_MAX_MESSAGES_PER_SYMBOL];      //!< Packet count per message
    uint16_t num_bytes[TMP_MAX_MESSAGES_PER_SYMBOL];        //!< Byte count per message
    uint16_t flow_index_info[TMP_MAX_MESSAGES_PER_SYMBOL];  //!< Flow index (eAxC ID) per message
    ModCompPartialSectionInfoPerMessagePerSymbol_t *mod_comp_params; //!< Modulation compression params (nullptr if disabled)
} PartialSectionInfoPerMessagePerSymbol_t;

/**
 * Partial flow info per slot
 *
 * Tracks flow-level packet counts per slot and symbol.
 */
typedef struct PartialFlowInfoPerSlot{
    cuda::std::array<int,kMaxFlows> flow_eaxcid;                                 //!< eAxC ID per flow
    cuda::std::array<int,kMaxFlows> flow_packet_count;                            //!< Total packet count per flow
    cuda::std::array<cuda::std::array<int, 14>, kMaxFlows> sym_flow_packet_count; //!< Per-symbol packet count per flow
    cuda::std::array<cuda::std::array<int, 14>, kMaxFlows> cumulative_sym_flow_packet_count; //!< Cumulative per-symbol count per flow
    uint32_t num_flows;                                                           //!< Number of flows
}PartialFlowInfoPerSlot_t;

/**
 * Partial U-plane slot information
 *
 * Lightweight slot info optimized for GPU access.
 * Contains timing, section, and flow information without full symbol details.
 */
typedef struct PartialUplaneSlotInfo {
    uint32_t frame_8b_subframe_4b_slot_6b;                       //!< Packed frame/subframe/slot ID
    uint32_t qp_clock_id;                                        //!< QP clock ID (common for all symbols)
    uint32_t ttl_pkts;                                           //!< Total packets in slot
    uint32_t total_num_flows;                                    //!< Total number of flows
    uint16_t syms_with_packets;                                  //!< Symbols containing packets
    uint16_t last_sym_with_packets;                              //!< Last symbol with packets
    PartialSectionInfoPerSymbol_t section_info[kPeerSymbolsInfo]; //!< Per-symbol section info
    PartialSectionInfoPerMessagePerSymbol_t message_info[kPeerSymbolsInfo]; //!< Per-symbol message info
    PartialFlowInfoPerSlot_t flowInfo_slot;                      //!< Flow info for slot
} PartialUplaneSlotInfo_t;

/**
 * Full U-plane slot information
 *
 * Complete slot info containing both GPU and CPU symbol information.
 */
typedef struct UplaneSlotInfo {
    UplaneSymbolInfoGpu_t gpu_symbol_info[kPeerSymbolsInfo];           //!< GPU-side symbol info array
    UplaneSymbolInfo_t symbol_info[kPeerSymbolsInfo];                  //!< CPU-side symbol info array
    uint32_t busy;                                                      //!< Busy flag (slot in use)
    cuda::std::array<int32_t, kPeerSymbolsInfo> last_wqebb_ctrl_idx;   //!< Last WQEBB control index per symbol
} UplaneSlotInfo_t;

struct FlowPtrInfo {
    uint8_t *gpu_pkt_addr;			//!< GPU memory address of the buffer
    uint8_t *cpu_pkt_addr;			//!< CPU memory address of the buffer    
    uint32_t pkt_buff_mkey;        //!< mkey value of the buffer
    uint32_t cpu_comms_pkt_buff_mkey; //!< mkey value of the CPU comms buffer
    uint32_t max_pkt_sz;           //!< maximum packet size
    uint32_t hdr_stride;           //!< header stride
};

struct TxRequestUplaneGpuComm
{
    Txq*        txq;
    size_t      prb_size_upl;
    size_t      prbs_per_pkt_upl;
    uint8_t     frame_id;
    uint16_t    subframe_id;
    uint16_t    slot_id;
    UplaneSlotInfo_t* d_slot_info;
    UplaneSlotInfoHost_t* h_up_slot_info_;
    PartialUplaneSlotInfo_t* partial_slot_info;
    uint32_t* flow_d_info;
    uint32_t* flow_sym_d_info;
    uint32_t* block_count;
    FlowPtrInfo* flow_hdr_size_info;
    uint32_t* flow_d_ecpri_seq_id;
    uint32_t* flow_d_hdr_template_info;
    uint16_t  max_num_prb_per_symbol;
};

typedef struct docaGpuParams
{
    struct doca_gpu *gpu;
}docaGpuParams_t;

struct doca_order_sem_info
{
    uint32_t pkts;
};

enum workload_mode {
	WORKLOAD_PERSISTENT = 0,
	WORKLOAD_SINGLE,
};

enum pipeline_mode {
	PIPELINE_INFERENCE_NO = 0,
	PIPELINE_INFERENCE_HTTP,
};

enum receive_mode {
	RECEIVE_CPU = 0,
	RECEIVE_GPU_DEDICATED,
	RECEIVE_GPU_PROXY,
};

class Fronthaul {
public:
    Fronthaul(FronthaulInfo const* info);
    ~Fronthaul();

    FronthaulInfo const& get_info() const;
    void                 update_metrics() const;
    bool                 pdump_enabled() const;
    bool                 fh_stats_dump_enabled() const;

    /* NICs */
    std::vector<Nic*> const& nics() const;
    uint16_t                 add_nic(Nic* nic);
    void                     remove_nic(Nic* nic);

    /* Accurate send scheduling */
    int32_t  get_timestamp_offset() const;
    uint64_t get_timestamp_mask_() const;

    UniqueGpuMap& gpus();

    /* Rivermax specific functions */
    bool    rmax_enabled() const;
    RivermaxPrx* rmax_get() const;
    int     rmax_init_nic(Nic* nic);

    docaGpuParams_t* get_docaGpuParams();
    void           doca_gpu_argp_start();
    void           doca_register_gpu_params(void);

protected:
    FronthaulInfo                info_;
    docaGpuParams_t                docaParams_;
    std::vector<Nic*>            nics_;
    std::unique_ptr<PdumpClient> pdump_client_;
    UniqueGpuMap                 gpus_;
    std::unique_ptr<RivermaxPrx> rmaxh_;
    std::unique_ptr<FHStatsDump> fh_stats_dump_;

    int32_t  timestamp_offset_{0};
    uint64_t timestamp_mask_{0};

    int rte_eal_init_wrapper(int argc, char **argv);
    void eal_init();
    void doca_gpu_setup();
    void setup_accurate_send_scheduling();
    void tune_virtual_memory();
    void validate_input();
    void setup_gpus();
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_INSTANCE_HPP__
