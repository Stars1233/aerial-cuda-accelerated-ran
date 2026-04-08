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

#ifndef AERIAL_FH_NIC_HPP__
#define AERIAL_FH_NIC_HPP__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"
#include "fronthaul.hpp"
#include "metrics.hpp"
#include "queue_manager.hpp"
#include "utils.hpp"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <queue>
#include "aerial-fh-driver/fh_mutex.hpp"

#include <doca_dev.h>

namespace aerial_fh
{
class Fronthaul;
class GpuMempool;
class Peer;
class GpuComm;

//!< Unique pointer type for GPU memory pools
using GpuMempoolUnique = std::unique_ptr<GpuMempool>;

#define TX_PP_ONLY  //!< TX packet pacing only mode
#define MAX_NUM_XSTATS 200  //!< Maximum number of extended statistics

/**
 * Fronthaul extended statistics structure
 */
struct fh_xstats_t
{
    char                                   names[MAX_NUM_XSTATS][RTE_ETH_XSTATS_NAME_SIZE];  //!< Statistic names
    uint64_t                               ids[MAX_NUM_XSTATS];       //!< Statistic IDs
    uint64_t                               values[MAX_NUM_XSTATS];    //!< Current values
    size_t                                 num_xstats;                //!< Number of statistics
    std::unordered_map<uint64_t, uint64_t> prev_values;              //!< Previous values for delta calculation
};

/**
 * Network Interface Card (NIC) management
 *
 * Manages DPDK/DOCA network interface including:
 * - TX/RX queue setup and management
 * - Memory pools (CPU, GPU, pinned)
 * - Hardware flow steering
 * - Extended statistics collection
 * - GPU-accelerated packet processing (DOCA GPUNetIO)
 */
class Nic {
public:
    /**
     * Constructor
     * @param[in] fhi Fronthaul instance
     * @param[in] NicInfo NIC configuration
     */
    Nic(Fronthaul* fhi, NicInfo const* NicInfo);

    /**
     * Destructor - cleanup NIC resources
     */
    ~Nic();

    /**
     * Start the NIC (begin packet processing)
     */
    void           start();

    /**
     * Get associated fronthaul instance
     * @return Pointer to fronthaul
     */
    Fronthaul*     get_fronthaul() const;

    /**
     * Get NIC device name
     * @return Device name string
     */
    std::string    get_name() const;

    /**
     * Get network interface name
     * @return Interface name string (e.g., "eth0")
     */
    std::string    get_if_name() const;

    /**
     * Get DPDK port ID
     * @return Port identifier
     */
    uint16_t       get_port_id() const;

    /**
     * Get associated CUDA device ID
     * @return GPU device ID
     */
    [[nodiscard]] GpuId get_cuda_device() const;

    /**
     * Get queue pair clock ID
     * @return Clock ID for timestamping
     */
    uint32_t       get_qp_clock_id() const;

    /**
     * Get queue pair clock ID in big-endian format
     * @return Clock ID (big-endian)
     */
    uint32_t       get_qp_clock_id_be() const;

    /**
     * Get Maximum Transmission Unit
     * @return MTU size in bytes
     */
    uint16_t       get_mtu() const;

    /**
     * Get next available flow index
     * @return Flow index for flow steering
     */
    uint16_t       get_nxt_flow_idx();

    /**
     * Free (return) a flow index
     * @param[in] flow_idx Flow index to free
     */
    void           free_flow_idx(uint16_t flow_idx);

    /**
     * Get CPU mbuf memory pool
     * @return Pointer to DPDK mempool
     */
    rte_mempool*   get_cpu_mbuf_pool() const;

    /**
     * Get CPU TX mbuf memory pool
     * @return Pointer to DPDK mempool for TX
     */
    rte_mempool*   get_cpu_tx_mbuf_pool() const;

    /**
     * Get RX mbuf memory pool
     * @param[in] hostPinned true for host-pinned memory, false for regular
     * @return Pointer to DPDK mempool
     */
    rte_mempool*   get_rx_mbuf_pool(bool hostPinned) const;

    /**
     * Get TX request memory pool for U-plane
     * @return Pointer to DPDK mempool
     */
    rte_mempool*   get_tx_request_pool() const;

    /**
     * Get TX request memory pool for C-plane
     * @return Pointer to DPDK mempool
     */
    rte_mempool*   get_tx_request_cplane_pool() const;

    /**
     * Update NIC performance metrics
     */
    void           update_metrics();

    /**
     * Check if packet dump (PCAP) is enabled
     * @return true if pdump enabled, false otherwise
     */
    bool           pdump_enabled() const;

    /**
     * Print NIC statistics to log
     */
    void           print_stats() const;

    /**
     * Print extended NIC statistics to log
     */
    void           print_extended_stats() const;

    /**
     * Reset NIC statistics counters
     */
    void           reset_stats() const;

    /**
     * Get NIC configuration information
     * @return Reference to NIC info structure
     */
    NicInfo const& get_info() const;

    /**
     * Get queue manager for this NIC
     * @return Reference to queue manager
     */
    QueueManager&  get_queue_manager();

    /**
     * Get PCAP RX queue
     * @return Pointer to PCAP RX queue
     */
    RxqPcap*       get_pcap_rxq() const;

    /**
     * Get NIC MAC address
     * @return MAC address string
     */
    std::string    get_mac_address();

    /**
     * Get socket handle
     * @return Reference to socket handle
     */
    socket_handle& get_socket();

    /**
     * Set socket handle
     * @param[in] sockh Pointer to socket handle
     */
    void           set_socket(socket_handle* sockh);

    /**
     * Create per-queue CPU RX mbuf pool
     * @param[in,out] cpu_mbuf_pool_queue_ Mempool unique pointer to initialize
     * @param[in] id Queue ID
     */
    void           create_cpu_rx_mbuf_per_queue_pool(MempoolUnique& cpu_mbuf_pool_queue_,uint16_t id);

    /**
     * Initialize fronthaul extended statistics collection
     */
    void fh_extended_stats_init();

    /**
     * Retrieve current extended statistics
     * @return 0 on success, negative on error
     */
    int  fh_extended_stats_retrieval();

    /**
     * Get transmitted bytes on physical layer
     * @return Number of bytes transmitted
     */
    uint64_t     get_tx_bytes_phy();

    /**
     * Get GPU memory region handle for packet headers
     * @return Memory region handle
     */
    MemRegHandle get_pkt_hdr_gpu_mr();

    /**
     * Get aggregated packet header GPU memory pointer
     * @return Pointer to GPU memory
     */
    uint8_t*     get_aggr_pkt_hdr_gpu();

    /**
     * Get GPU communication manager
     * @return Pointer to GPU comm instance
     */
    GpuComm* get_gpu_comm();

    /**
     * GPU-initiated send for U-plane packets
     * @param[in] pTxRequestGpuPercell TX request structure for cell
     * @param[in] prb_info PRB information
     * @return 0 on success, negative on error
     */
    int      gpu_comm_send_uplane(TxRequestGpuPercell* pTxRequestGpuPercell, PreparePRBInfo& prb_info);

    /**
     * Set trigger timestamp for GPU-initiated communication
     * @param[in] slot_idx Slot index
     * @param[in] trigger_ts Trigger timestamp in nanoseconds
     * @return 0 on success, negative on error
     */
    int      gpu_comm_set_trigger_ts(uint32_t slot_idx, uint64_t trigger_ts);

    /**
     * Ring CPU doorbell to trigger packet processing
     * @param[in] pTxRequestGpuPercell TX request structure for cell
     * @param[in] prb_info PRB information
     * @param[in] packet_timing_info Packet timing info
     * @return 0 on success, negative on error
     */
    int      ring_cpu_doorbell(TxRequestGpuPercell* pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info);

    /**
     * Trigger CQE tracer callback for GPU communication
     * @param[in] pTxRequestGpuPercell TX request structure for cell
     * @return 0 on success, negative on error
     */
    int      gpu_comm_trigger_cqe_tracer_cb(TxRequestGpuPercell* pTxRequestGpuPercell);

    /**
     * Initialize TX queues for GPU-initiated communication
     */
    void     gpu_comm_init_tx_queues();

    /**
     * Log maximum packet processing delays
     */
    void log_max_delays();

    /**
     * Check if NIC is ConnectX-6
     * @return true if CX6, false otherwise
     */
    bool is_cx6();

    /**
     * Get packet buffer size (rounded)
     * @return Buffer size in bytes
     */
    size_t get_buf_size() { return packet_size_rnd_local_; }

    /**
     * Setup flow communication buffer
     */
    void set_flow_comm_buf();

    /**
     * Get flow communication buffer
     * @return Pointer to DOCA TX buffer
     */
    struct doca_tx_buf * get_flow_comm_buf();

    /**
     * Get DOCA device handle
     * @return Pointer to DOCA device
     */
    struct doca_dev * get_doca_dev();

protected:
    Fronthaul*               fhi_;                         //!< Associated fronthaul instance
    NicInfo                  info_;                        //!< NIC configuration
    uint32_t                 qp_clock_id_;                 //!< Queue pair clock ID
    uint32_t                 qp_clock_id_be_;              //!< Queue pair clock ID (big-endian)
    uint16_t                 port_id_{(uint8_t)-1};        //!< DPDK port ID
    std::queue<uint16_t>     flow_idx_q_;                  //!< Available flow index queue
    aerial_fh::FHMutex       flow_idx_q_lock_;             //!< Mutex for flow index queue
    NicMetrics               metrics_;                     //!< NIC performance metrics
    std::string              driver_name_{};               //!< NIC driver name
    std::string              if_name_{};                   //!< Network interface name
    socket_handle            sockh_;                       //!< Socket handle
    std::unique_ptr<GpuComm> gcomm_;                       //!< GPU communication manager
    bool                     cx6;                          //!< ConnectX-6 flag
    fh_xstats_t              fh_xstats_{};                 //!< Extended statistics
    MemRegHandle             pkt_hdr_gpu_mr_   = MemRegHandle();  //!< GPU packet header memory region
    uint8_t*                 aggr_pkt_hdr_gpu_ = nullptr;  //!< Aggregated packet headers (GPU)
    size_t                   packet_size_rnd_local_;       //!< Rounded packet size (local cache)

    // NIC resources
    QueueManager     queue_manager_;                                 //!< TX/RX queue manager
    MempoolUnique    cpu_mbuf_pool_{nullptr, &rte_mempool_free};     //!< CPU mbuf memory pool
    MempoolUnique    cpu_tx_mbuf_pool_{nullptr, &rte_mempool_free};  //!< CPU TX mbuf memory pool
    GpuMempoolUnique gpu_mbuf_pool_;                                 //!< GPU mbuf memory pool
    GpuMempoolUnique cpu_pinned_mbuf_pool_;                          //!< CPU-pinned mbuf memory pool
    MempoolUnique    tx_request_pool_{nullptr, &rte_mempool_free};   //!< TX request pool (U-plane)
    MempoolUnique    tx_request_cplane_pool_{nullptr, &rte_mempool_free};  //!< TX request pool (C-plane)
    struct doca_dev *ddev_;           //!< DOCA device handle
    struct doca_tx_buf flow_tx_buf;   //!< Flow TX buffer
    size_t packet_size_rnd;           //!< Rounded packet size
    int num_packets;                  //!< Number of packets

    // Initialization and configuration methods
    void doca_probe_device();                      //!< Probe for DOCA device
    void validate_input();                         //!< Validate NIC configuration
    void print_rx_offloads(uint64_t offloads);     //!< Print RX offload capabilities
    void print_tx_offloads(uint64_t offloads);     //!< Print TX offload capabilities
    void configure();                              //!< Configure NIC port
    void set_mtu();                                //!< Set Maximum Transmission Unit
    void setup_tx_queues();                        //!< Setup TX queues
    void setup_tx_queues_gpu();                    //!< Setup GPU-accelerated TX queues
    void setup_rx_queues();                        //!< Setup RX queues
    void create_cpu_mbuf_pool();                   //!< Create CPU mbuf memory pool
    void create_gpu_mbuf_pool();                   //!< Create GPU mbuf memory pool
    void create_cpu_pinned_mbuf_pool();            //!< Create CPU-pinned mbuf memory pool
    void create_tx_request_uplane_pool();          //!< Create TX request pool for U-plane
    void create_tx_request_cplane_pool();          //!< Create TX request pool for C-plane
    void check_physical_link_status() const;       //!< Check link status
    void add_device();                             //!< Add device to DPDK
    void remove_device();                          //!< Remove device from DPDK
    void set_port_id();                            //!< Set DPDK port ID
    void set_qp_clock_id();                        //!< Set queue pair clock ID
    void validate_driver();                        //!< Validate NIC driver
    void restrict_ingress_traffic();               //!< Restrict ingress traffic (default drop)
    void disable_ethernet_flow_control();          //!< Disable Ethernet flow control
    void set_pcie_max_read_request_size();         //!< Set PCIe max read request size
    void enable_weight_arbiter();                  //!< Enable weighted round-robin arbiter
    void disable_weight_arbiter();                 //!< Disable weighted round-robin arbiter
    void warm_up_txqs();                           //!< Warm up TX queues (pre-allocate)
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_NIC_HPP__
