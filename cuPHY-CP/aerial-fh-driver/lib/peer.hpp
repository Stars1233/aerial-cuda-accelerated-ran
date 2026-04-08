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

#ifndef AERIAL_FH_PEER_HPP__
#define AERIAL_FH_PEER_HPP__

#include <unordered_map>
#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "dpdk.hpp"
#include "metrics.hpp"
#include "utils.hpp"
#include "queue.hpp"
#include "aerial-fh-driver/fh_mutex.hpp"

#ifndef TAG_PEER_HPP
#define TAG_PEER_HPP (NVLOG_TAG_BASE_FH_DRIVER + 1) // "FH.PEER"
#endif

namespace aerial_fh
{
class Fronthaul;
class Nic;
class Flow;
struct TxRequestUplane;
struct TxRequestCplane;
struct TxRequestUplaneGpuComm;
struct UplaneSlotInfo;
typedef struct UplaneSlotInfo UplaneSlotInfo_t;

struct UplaneSymbolInfoGpu;
typedef struct UplaneSymbolInfoGpu UplaneSymbolInfoGpu_t;

//!< Unique pointer vector for flow management
using FlowsUnique = std::vector<std::unique_ptr<Flow>>;

/**
 * Beamforming weights C-plane chaining information
 *
 * Contains information for chaining multiple mbufs to send beamforming weights
 * in C-plane messages for massive MIMO configurations.
 */
struct bfw_cplane_chain_info {
    void* bfw_buffer = nullptr;                      //!< BFW buffer pointer
    size_t bfw_buffer_size = 0;                      //!< BFW buffer size in bytes
    size_t bfw_padding_size = 0;                     //!< Padding size in bytes
    struct rte_mbuf* header_mbuf = nullptr;          //!< Header mbuf pointer
    struct rte_mbuf** chain_mbufs = nullptr;         //!< Array of chained mbuf pointers
    size_t chained_mbufs = 0;                        //!< Number of chained mbufs
    int ap_idx = 0;                                  //!< Antenna port index
    int num_bundles_per_ap = 0;                      //!< Number of bundles per antenna port
    BfwCplaneChainingMode bfw_chain_mode = BfwCplaneChainingMode::NO_CHAINING;  //!< BFW chaining mode
};

/**
 * Stack-allocated array of mbuf pointers for C-plane TX
 *
 * Provides efficient fixed-size array on stack for batch C-plane packet transmission.
 * Avoids heap allocation overhead for better performance.
 */
class MbufArray {
private:
    static constexpr size_t kMaxMbufsPerArray = kTxPktBurstCplane;  //!< Maximum mbufs per array
    rte_mbuf* mbufs_[kMaxMbufsPerArray];  //!< Fixed-size stack array of mbuf pointers
    size_t size_;  //!< Current number of mbufs in array

public:
    /**
     * Default constructor - initializes empty array
     * mbufs_ elements are only accessed via operator[]/data() within [0, size_), so uninit is safe.
     */
    // coverity[uninit_ctor]
    MbufArray() : size_(0) {}

    /**
     * Destructor - no heap allocation to clean up
     */
    ~MbufArray() = default;

    // Non-copyable
    MbufArray(const MbufArray&) = delete;
    MbufArray& operator=(const MbufArray&) = delete;

    /**
     * Move constructor
     * @param[in] other Array to move from
     */
    MbufArray(MbufArray&& other) noexcept
        : size_(other.size_) {
        std::copy(other.mbufs_, other.mbufs_ + other.size_, mbufs_);
        other.size_ = 0;
    }

    /**
     * Clear array (set size to 0)
     */
    void clear() { size_ = 0; }

    /**
     * Get current size
     * @return Number of mbufs in array
     */
    size_t size() const { return size_; }

    /**
     * Get maximum capacity
     * @return Maximum number of mbufs that can be stored
     */
    size_t capacity() const { return kMaxMbufsPerArray; }

    /**
     * Const array access operator
     * @param[in] index Array index
     * @return Const mbuf pointer at index
     */
    rte_mbuf* operator[](size_t index) const {
        if(index >= size_) {
            NVLOGF_FMT(TAG_PEER_HPP, AERIAL_ORAN_FH_EVENT, "MbufArray index out of bounds: {} >= {}", index, size_);
        }
        return mbufs_[index];
    }

    /**
     * Non-const array access operator
     * @param[in] index Array index
     * @return Reference to mbuf pointer at index
     */
    rte_mbuf*& operator[](size_t index) {
        if(index >= size_) {
            NVLOGF_FMT(TAG_PEER_HPP, AERIAL_ORAN_FH_EVENT, "MbufArray index out of bounds: {} >= {}", index, size_);
        }
        return mbufs_[index];
    }

    /**
     * Add mbuf to end of array
     * @param[in] mbuf Mbuf pointer to add
     */
    void push_back(rte_mbuf* mbuf) {
        if(size_ >= kMaxMbufsPerArray) {
            NVLOGF_FMT(TAG_PEER_HPP, AERIAL_ORAN_FH_EVENT, "MbufArray size exceeds capacity: {} >= {}", size_, kMaxMbufsPerArray);
            return;
        }
        mbufs_[size_++] = mbuf;
    }

    /**
     * Get underlying array pointer
     * @return Pointer to mbuf array
     */
    rte_mbuf** data() { return mbufs_; }

    /**
     * Get const underlying array pointer
     * @return Const pointer to mbuf array
     */
    [[nodiscard]] const rte_mbuf* const* data() const { return mbufs_; }

private:
    /**
     * Validate array state
     */
    void check_valid() const {
        if(size_ > kMaxMbufsPerArray) {
            NVLOGF_FMT(TAG_PEER_HPP, AERIAL_ORAN_FH_EVENT, "MbufArray size exceeds capacity: {} > {}", size_, kMaxMbufsPerArray);
        }
    }
};

/**
 * C-plane TX queue information
 */
struct sendCPlaneTxqInfo
{
    Ns* last_cplane_tx_ts_;    //!< Last C-plane TX timestamp
    MbufArray* mbufs_;         //!< Array of mbufs to send
    bool is_bfw_send_req;      //!< Flag indicating BFW send request
};

/**
 * C-plane packet count information
 */
struct cplaneCountInfo
{
    size_t num_packets;             //!< Total number of packets
    size_t num_bfw_mbufs;           //!< Number of BFW mbufs
    size_t num_bfw_padding_mbufs;   //!< Number of BFW padding mbufs
};

/**
 * C-plane packet preparation information
 */
struct cplanePrepareInfo
{
    size_t created_pkts;    //!< Number of packets created
    size_t chained_mbufs;   //!< Number of chained mbufs
};

/**
 * ORAN Fronthaul Peer Management
 *
 * Manages communication with a single ORAN fronthaul peer (typically an RU).
 * Handles:
 * - C-plane and U-plane packet transmission/reception
 * - RX flow steering rules for packet routing
 * - TX/RX queue management per peer
 * - Massive MIMO beamforming weight transmission
 * - GPU-accelerated packet processing
 * - Multi-section U-plane support
 * - Performance metrics tracking
 */
class Peer {
public:
    /**
     * Constructor
     * @param[in] nic Associated NIC instance
     * @param[in] info Peer configuration information
     * @param[in] eAxC_list_ul Uplink eAxC ID list
     * @param[in] eAxC_list_srs SRS eAxC ID list
     * @param[in] eAxC_list_dl Downlink eAxC ID list
     */
    Peer(Nic* nic, PeerInfo const* info,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs,std::vector<uint16_t>& eAxC_list_dl);

    /**
     * Destructor - cleanup peer resources
     */
    ~Peer();

    /**
     * Get associated NIC
     * @return Pointer to NIC instance
     */
    Nic*            get_nic() const;

    /**
     * Get associated fronthaul instance
     * @return Pointer to fronthaul instance
     */
    Fronthaul*      get_fronthaul() const;

    /**
     * Get C-plane TX queue
     * @return Pointer to C-plane TXQ
     */
    Txq*            get_cplane_txq();

    /**
     * Get next available U-plane TX queue (round-robin)
     * @return Pointer to next U-plane TXQ
     */
    Txq*            get_next_uplane_txq();

    /**
     * Get next available GPU-accelerated U-plane TX queue (round-robin)
     * @return Pointer to next GPU U-plane TXQ
     */
    Txq*            get_next_uplane_txq_gpu();

    /**
     * Get RX queue for regular traffic
     * @return Pointer to RXQ
     */
    Rxq*            get_rxq();

    /**
     * Get RX queue for SRS traffic
     * @return Pointer to SRS RXQ
     */
    Rxq*            get_rxq_srs();

    /**
     * Get peer configuration
     * @return Reference to peer info
     */
    PeerInfo const& get_info() const;

    /**
     * Get peer ID
     * @return Peer identifier
     */
    PeerId          get_id() const;

    /**
     * Get eAxC ID to index mapping for uplink
     * @return Reference to eAxC map
     */
    std::unordered_map<u_int16_t, u_int16_t>& get_eaxcid_idx_mp();

    /**
     * Get eAxC ID to index mapping for downlink
     * @return Reference to downlink eAxC map
     */
    std::unordered_map<u_int16_t, u_int16_t>& get_dlu_eaxcid_idx_mp();

    /**
     * Count number of U-plane packets needed for messages
     * @param[in] info Array of U-plane message info
     * @param[in] num_msgs Number of messages
     * @return Total packet count
     */
    size_t          count_uplane_packets(UPlaneMsgSendInfo const* info, size_t num_msgs);

    /**
     * Get all U-plane TX queues
     * @param[out] txqs Array to store TXQ pointers
     * @param[out] num_txqs Number of TXQs
     */
    void            get_uplane_txqs(Txq** txqs, size_t* num_txqs) const;

    /**
     * Get all GPU-accelerated U-plane TX queues
     * @param[out] txqs Array to store TXQ pointers
     * @param[out] num_txqs Number of TXQs
     */
    void            get_uplane_txqs_gpu(Txq** txqs, size_t* num_txqs) const;

    /**
     * Get peer MAC address as string
     * @return MAC address string
     */
    std::string     get_mac_address();

    /**
     * Update maximum PRBs per symbol
     * @param[in] max_num_prbs_per_symbol Maximum PRB count
     */
    void            update_max_num_prbs_per_symbol(uint16_t max_num_prbs_per_symbol);

    /**
     * Update compression configuration
     * @param[in] dl_comp_meth Downlink compression method
     * @param[in] dl_bit_width Downlink bit width
     */
    void            update(UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width);

    /**
     * Update peer MAC address and eAxC lists
     * @param[in] dst_mac_addr Destination MAC address
     * @param[in] eAxC_list_ul Uplink eAxC list
     * @param[in] eAxC_list_srs SRS eAxC list
     */
    void            update(MacAddr dst_mac_addr,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs);

    /**
     * Update peer MAC address, VLAN, and eAxC lists
     * @param[in] dst_mac_addr Destination MAC address
     * @param[in] vlan_tci VLAN TCI value
     * @param[in] eAxC_list_ul Uplink eAxC list
     * @param[in] eAxC_list_srs SRS eAxC list
     */
    void            update(MacAddr dst_mac_addr, uint16_t vlan_tci,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs);

    // ========== TX API ==========

    /**
     * Prepare C-plane messages for transmission
     * @param[in] info Array of C-plane message info
     * @param[in] num_msgs Number of messages
     * @param[out] tx_request Allocated TX request handle
     */
    void   prepare_cplane(CPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestCplane** tx_request);

    /**
     * Enqueue C-plane TX request descriptor
     * @param[in] tx_request TX request to enqueue
     */
    void   enqueue_cplane_tx_request_descriptor(TxRequestCplane* tx_request);

    /**
     * Send C-plane messages immediately
     * @param[in] info Array of C-plane message info
     * @param[in] num_msgs Number of messages
     * @return Number of packets successfully sent
     */
    size_t send_cplane(CPlaneMsgSendInfo const* info, size_t num_msgs);

    /**
     * Send C-plane messages for massive MIMO (with BFW)
     * @param[in] info Array of C-plane message info
     * @param[in] num_msgs Number of messages
     * @return Number of packets successfully sent
     */
    size_t send_cplane_mmimo(CPlaneMsgSendInfo const* info, size_t num_msgs);

    /**
     * Prepare U-plane messages for transmission
     * @param[in] info Array of U-plane message info
     * @param[in] num_msgs Number of messages
     * @param[in] notification TX completion callback
     * @param[out] tx_request Allocated TX request handle
     * @param[in] txq_index TX queue index to use
     */
    void   prepare_uplane(UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxRequestUplane** tx_request, int txq_index);

    /**
     * Prepare single U-plane message
     * @param[in] info U-plane message info
     * @param[out] header_mbufs Array for header mbufs
     * @param[out] iq_data_mbufs Array for IQ data mbufs
     * @param[in] txq_index TX queue index
     * @return Number of packets prepared
     */
    size_t prepare_uplane_message(const UPlaneMsgSendInfo& info, rte_mbuf** header_mbufs, rte_mbuf** iq_data_mbufs, int txq_index);

    /**
     * Send prepared U-plane TX request
     * @param[in] tx_request TX request to send
     * @param[in] txq TX queue to use
     * @return Number of packets successfully sent
     */
    size_t send_uplane(TxRequestUplane* tx_request, Txq* txq);

    /**
     * Poll for TX completion notifications
     */
    void   poll_tx_complete();

    /**
     * Get flow pointer information for GPU comm
     * @return Pointer to flow info array
     */
    FlowPtrInfo*   get_flow_ptr_info();

    /**
     * Get header template information for GPU comm
     * @return Pointer to header template array
     */
    uint32_t*   get_hdr_template_info();

    // ========== Multi-section U-plane API v3 (without chained mbufs) ==========


    /**
     * Prepare single multi-section U-plane message (single mbuf version)
     * @param[in] info Multi-section message info
     * @param[out] header_mbufs Array for header mbufs
     * @param[in] txq_index TX queue index
     * @return Number of packets prepared
     */
    size_t prepare_uplane_message_v3(const UPlaneMsgMultiSectionSendInfo& info, rte_mbuf** header_mbufs, int txq_index);

    // ========== TX Request Management ==========

    /**
     * Pre-allocate mbufs for TX request
     * @param[in,out] tx_request TX request to allocate mbufs for
     * @param[in] num_mbufs Number of mbufs to allocate
     */
    void   preallocate_mbufs(TxRequestUplane** tx_request, int num_mbufs);

    /**
     * Prepare U-plane with pre-allocated TX request
     * @param[in] info Multi-section message info
     * @param[in] notification TX completion callback
     * @param[in,out] tx_request Pre-allocated TX request
     * @param[in] txq_index TX queue index
     */
    void   prepare_uplane_with_preallocated_tx_request(UPlaneMsgMultiSectionSendInfo const* info, UPlaneTxCompleteNotification notification, TxRequestUplane** tx_request, int txq_index);

    /**
     * Send U-plane without freeing TX request (for reuse)
     * @param[in] tx_request TX request to send
     * @param[in] txq TX queue to use
     * @param[out] timing Optional timing information
     * @return Number of packets successfully sent
     */
    size_t send_uplane_without_freeing_tx_request(TxRequestUplane* tx_request, Txq* txq, TxqSendTiming* timing = nullptr);

    /**
     * Allocate TX request from pool
     * @param[out] tx_request Allocated TX request handle
     */
    void   alloc_tx_request(TxRequestUplane** tx_request);

    /**
     * Free TX request back to pool
     * @param[in] tx_request TX request to free
     */
    void   free_tx_request(TxRequestUplane* tx_request);

    /**
     * Free pre-allocated mbufs in TX request
     * @param[in,out] tx_request TX request with pre-allocated mbufs
     */
    void   free_preallocated_mbufs(TxRequestUplane** tx_request);

    // ========== Packet Counting ==========

    /**
     * Count U-plane packets needed for multi-section message
     * @param[in] info Multi-section message info
     * @return Total packet count
     */
    size_t prepare_uplane_count_packets(UPlaneMsgMultiSectionSendInfo const* info);

    /**
     * Count C-plane packets needed
     * @param[in] info Array of C-plane message info
     * @param[in] num_msgs Number of messages
     * @return Total packet count
     */
    size_t prepare_cplane_count_packets(CPlaneMsgSendInfo const* info, size_t num_msgs);

    // ========== RX API ==========

    /**
     * Receive packets from peer
     * @param[out] info Array to store received message info
     * @param[in,out] num_msgs Input: max messages, Output: received messages
     * @param[in] srs true for SRS RXQ, false for regular RXQ
     */
    void receive(MsgReceiveInfo* info, size_t* num_msgs, bool srs);

    /**
     * Receive packets with timeout
     * @param[out] info Array to store received message info
     * @param[in,out] num_msgs Input: max messages, Output: received messages
     * @param[in] timeout Timeout in nanoseconds
     */
    void receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout);

    /**
     * Update RX metrics
     * @param[in] rx_packets Number of packets received
     * @param[in] rx_bytes Number of bytes received
     */
    void update_rx_metrics(size_t rx_packets, size_t rx_bytes);

    /**
     * Update TX metrics
     * @param[in] tx_packets Number of packets transmitted
     * @param[in] tx_bytes Number of bytes transmitted
     */
    void update_tx_metrics(size_t tx_packets, size_t tx_bytes);

    /**
     * Set total number of flows
     * @param[in] val Flow count
     */
    void setTotalNumFlows(uint32_t val) {total_num_flows_=val;};

    /**
     * Get total number of flows
     * @return Flow count
     */
    uint32_t getTotalNumFlows(){return total_num_flows_;};

    // ========== GPU-Initiated Communication ==========

    /**
     * Create U-plane slot list for GPU comm
     */
    void     gpu_comm_create_up_slot_list();

    /**
     * Get next U-plane slot index for GPU comm
     * @return Slot index
     */
    uint32_t gpu_comm_get_next_up_slot_idx();

    /**
     * Create C-plane sections cache for GPU comm
     */
    void     create_cplane_sections_cache();

    /**
     * Get next C-plane sections info list index
     * @return List index
     */
    uint32_t get_next_cplane_sections_info_list_idx();

    /**
     * Update TX metrics from GPU comm TX request
     * @param[in] tx_request GPU comm TX request
     */
    void     gpu_comm_update_tx_metrics(TxRequestUplaneGpuComm* tx_request);

    /**
     * Prepare U-plane for GPU-initiated communication
     * @param[in] info Array of U-plane message info
     * @param[in] num_msgs Number of messages
     * @param[out] tx_request Allocated GPU comm TX request
     * @param[in] cell_start_time Cell start time
     * @param[in] symbols_duration Symbol duration
     * @param[in] commViaCpu true for CPU-assisted comm
     */
    void     gpu_comm_prepare_uplane(UPlaneMsgSendInfo const* info, size_t num_msgs,
                                    TxRequestUplaneGpuComm** tx_request, std::chrono::nanoseconds cell_start_time, std::chrono::nanoseconds symbols_duration,bool commViaCpu);


    /**
     * Create DOCA GPU semaphore for synchronization
     */
    void doca_gpu_sem_create();

    // ========== Memory Footprint Getters ==========

    /**
     * Get GPU pinned memory size
     * @return Size in bytes
     */
    size_t getGpuPinnedSize() const;

    /**
     * Get GPU regular memory size
     * @return Size in bytes
     */
    size_t getGpuRegularSize() const;

    /**
     * Get CPU pinned memory size
     * @return Size in bytes
     */
    size_t getCpuPinnedSize() const;

    /**
     * Get CPU regular memory size
     * @return Size in bytes
     */
    size_t getCpuRegularSize() const;
protected:
    Nic*              nic_;                             //!< Associated NIC instance
    PeerInfo          info_;                            //!< Peer configuration
    size_t            prb_size_upl_;                    //!< PRB size for uplink
    size_t            prbs_per_pkt_upl_;                //!< PRBs per packet for uplink
    Txq*              txq_dl_cplane_{};                 //!< Downlink C-plane TX queue
    Txq*              txq_ul_cplane_{};                 //!< Uplink C-plane TX queue
    Txq*              txq_dl_bfw_cplane_{};             //!< Downlink BFW C-plane TX queue
    Txq*              txq_ul_bfw_cplane_{};             //!< Uplink BFW C-plane TX queue
    Iterator<Txq*>    txqs_uplane_;                     //!< U-plane TX queue iterator
    Iterator<Txq*>    txqs_uplane_gpu_;                 //!< GPU U-plane TX queue iterator
    Rxq*              rxq_{};                           //!< Regular RX queue
    Rxq*              rxqSrs_{};                        //!< SRS RX queue
    PeerMetrics       metrics_;                         //!< Performance metrics
    RxFlowRulesUnique rx_flow_rules_;                   //!< RX flow steering rules
    UplaneSlotInfo_t* d_up_slot_info_;                  //!< U-plane slot info (GPU device memory)
    UplaneSlotInfoHost_t* h_up_slot_info_;              //!< U-plane slot info (host memory)
    PartialUplaneSlotInfo_t* partial_up_slot_info_;     //!< Partial U-plane slot info (host pinned)
    uint32_t* d_flow_pkt_hdr_index_;                    //!< Flow packet header index (GPU)
    uint32_t* d_flow_sym_pkt_hdr_index_;                //!< Flow symbol packet header index (GPU)
    uint32_t *d_block_count_;                           //!< Block count (GPU)
    uint32_t* d_ecpri_seq_id_;                          //!< eCPRI sequence ID (GPU, uint32_t for atomicInc)
    FlowPtrInfo* h_flow_hdr_size_info_;                 //!< Flow header size info (host pinned)
    uint32_t* d_hdr_template_;                          //!< Header template (GPU)
    std::unordered_map<u_int16_t, u_int16_t> eaxcid_idx_mp;    //!< eAxC ID to index map (uplink)
    std::unordered_map<u_int16_t, u_int16_t> dlu_eaxcid_idx_mp;  //!< eAxC ID to index map (downlink)

    slot_command_api::cplane_sections_info_t* cplane_sections_info_list_;  //!< C-plane sections cache
    std::atomic<uint32_t>                     cplane_sections_info_list_cnt_;  //!< C-plane sections count

    UplaneSymbolInfoGpu_t*           up_symbol_info_gpu_;      //!< U-plane symbol info for GPU
    uint32_t                         up_slot_info_cnt_;        //!< U-plane slot info counter
    aerial_fh::FHMutex               mtx_{};                   //!< Mutex for thread-safe access
    Ns                               last_dl_cplane_tx_ts_{0}; //!< Last DL C-plane TX timestamp
    Ns                               last_ul_cplane_tx_ts_{0}; //!< Last UL C-plane TX timestamp
    Ns                               last_dl_bfw_cplane_tx_ts_{0};  //!< Last DL BFW C-plane TX timestamp
    Ns                               last_ul_bfw_cplane_tx_ts_{0};  //!< Last UL BFW C-plane TX timestamp
    TxRequestUplaneGpuComm*          up_tx_request_;           //!< GPU comm U-plane TX request

    void     create_rx_rules(const std::vector<uint16_t>& eAxC_list_ul,const std::vector<uint16_t>& eAxC_list_srs, const std::vector<uint16_t>& eAxC_list_dl);
    void     create_rx_rule_for_cplane(const std::vector<uint16_t>& eAxC_list_ul,const std::vector<uint16_t>& eAxC_list_srs, const std::vector<uint16_t>& eAxC_list_dl);
    void     create_rx_rule(const std::vector<uint16_t>& eAxC_list_ul,const std::vector<uint16_t>& eAxC_list_srs);
    void     create_rx_rule_with_cpu_mirroring();
    void     create_rx_rule_for_dl_uplane(const std::vector<uint16_t>& eAxC_list_dl);

    size_t   count_cplane_packets(CPlaneMsgSendInfo const* infos, size_t num_msgs);
    cplaneCountInfo count_cplane_packets_mmimo(CPlaneMsgSendInfo const* infos, size_t num_msgs, size_t max_num_packets);
    cplaneCountInfo count_cplane_packets_mmimo_dl(CPlaneMsgSendInfo const* infos, size_t num_msgs);
    size_t   count_cplane_packets_mmimo_ul(CPlaneMsgSendInfo const* infos, size_t num_msgs);

    uint16_t prepare_cplane_message(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs);
    void prepare_cplane_message_mmimo(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, rte_mbuf** chain_mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw, cplanePrepareInfo& cplane_prepare_info);
    void prepare_cplane_message_mmimo_dl(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, rte_mbuf** chain_mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw, cplanePrepareInfo& cplane_prepare_info);
    uint16_t prepare_cplane_message_mmimo_ul(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw);
    uint16_t prepare_cplane_message_mmimo_no_se(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw);
    void     adjust_src_mac_addr();
    void     request_nic_resources();
    void     free_nic_resources();
    void     validate_input();
    void     send_cplane_packets(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts, Txq* txq, Txq* bfw_txq, bool divide_budget);
    virtual void send_cplane_packets_dl(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts);
    virtual void send_cplane_packets_ul(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts);
    virtual void send_cplane_enqueue_nic(Txq *txq, rte_mbuf* mbufs[], size_t num_packets); 
    uint32_t total_num_flows_{0};  //!< Total number of flows for this peer

    /**
     * Count U-plane packets for multi-section message
     * @param[in] info Multi-section message info
     * @param[in] chained_mbuf true if using chained mbufs
     * @return Pair of (header_count, data_count)
     */
    std::pair<size_t, size_t> count_uplane_packets(UPlaneMsgMultiSectionSendInfo const* info, bool chained_mbuf);

    // Memory footprint tracking
    size_t cpu_regular_size;    //!< CPU regular memory size in bytes
    size_t cpu_pinned_size;     //!< CPU pinned memory size in bytes
    size_t gpu_regular_size;    //!< GPU regular memory size in bytes
    size_t gpu_pinned_size;     //!< GPU pinned memory size in bytes
private:
    void     count_cplane_packets_mmimo_ext11(
        uint16_t section_hdr_size,
        uint16_t pkt_section_info_room,
        uint16_t& total_section_info_size,
        size_t& section_num_packets,
        size_t& section_bfw_mbufs,
        size_t& section_padding_mbufs,
        int& start_prbc,
        int& max_num_prbc,
        int& num_bundles,
        CPlaneSectionExtInfo* ext11_ptr,
        const uint8_t direction,
        bool& prev_section_dyn_bfw
    );
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_PEER_HPP__
