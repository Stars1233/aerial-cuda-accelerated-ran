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

#ifndef AERIAL_FH_DEFAULTS_HPP__
#define AERIAL_FH_DEFAULTS_HPP__

#include "aerial-fh-driver/api.hpp"
#include "app_config.hpp"
#include "dpdk.hpp"

namespace aerial_fh
{
// Memory pool and burst sizes
constexpr size_t   kGpuMbufPoolSz          = (1 << 18);   //!< GPU mbuf pool size (262144 packets)
constexpr uint16_t kMbufPoolDroomSzAlign   = 128;         //!< Mbuf pool data room size alignment in bytes
constexpr size_t   kTxPktBurstUplane       = (1 << 11);   //!< Max U-plane packets per TX burst (2048)
constexpr size_t   kTxPktBurstCplane       = (1 << 10);   //!< Max C-plane packets per TX burst (1024)
constexpr size_t   kRxPktBurst             = (1 << 11);   //!< Max packets per RX burst (2048)
constexpr uint64_t kPdumpWorkerSpinDelayUs = 1;           //!< Pdump worker idle delay (microseconds)
constexpr size_t   kRxqPcapSize            = (1UL << 14); //!< RXQ PCAP buffer size (16384 packets)
constexpr size_t   kMaxTxqUplanePerPeer    = (1 << 6);    //!< Max U-plane TX queues per peer (64)
constexpr uint64_t kFHStatsPollingDelayUs  = 500;         //!< FH stats polling interval (microseconds)
constexpr uint64_t kNicTputPollingInterval = 1000000000;  //!< NIC throughput polling interval (1 second in ns)

// TX timeouts
constexpr uint64_t kTxBurstTimeout = 100000000;  //!< TX burst timeout (100ms in ns)

// Section handling
constexpr uint64_t kMaxSplitCount = 32;  //!< Maximum C-plane section split count

// GPU page management
constexpr uint8_t  kNvGpuPageShift = 16;                   //!< GPU page size shift (16 = 64KB pages)
constexpr uint32_t kNvGpuPageSize  = (1UL << kNvGpuPageShift); //!< GPU page size (65536 bytes)

// Accurate TX scheduling
constexpr uint64_t kMinAccuTxSchedResNs = 500;    //!< Min accurate TX scheduling resolution (ns)
constexpr uint64_t kMaxAccuTxSchedResNs = 10000;  //!< Max accurate TX scheduling resolution (ns)

// Metrics keys
constexpr char   kMetricPeerKey[] = "cell";  //!< Metric key for peer/cell
constexpr char   kMetricNicKey[]  = "nic";   //!< Metric key for NIC
constexpr size_t kMetricNicXCount = 7;       //!< NIC extended stats count

// Driver names
constexpr char kPciBusName[]       = "pci";            //!< PCI bus name
constexpr char kAuxBusName[]       = "auxiliary";      //!< Auxiliary bus name
constexpr char kMlxPciDriverName[] = "mlx5_pci";       //!< Mellanox PCI driver name
constexpr char kMlxAuxDriverName[] = "mlx5_auxiliary"; //!< Mellanox auxiliary driver name

// Flow limits
constexpr size_t   pageSizeAlign   = 128;                     //!< Page size alignment in bytes
constexpr int      kMaxPktsFlow    = 2048;                    //!< Max packets per flow
constexpr int      kMaxFlows       = MAX_DL_EAXCIDS;          //!< Max flows (multi-cell MIMO support)
static_assert(PARTIAL_UPLANE_MAX_FLOWS == kMaxFlows, "partial_uplane_slot_info must match defaults");

// Section limits
constexpr int kMaxULUPSections = 20;  //!< Max uplink U-plane sections

// GPU comm host page constants
constexpr uint64_t GPUCOMM_HOST_PAGE_BITS = 12;                              //!< Host page bits (12 = 4KB pages)
constexpr uint64_t GPUCOMM_HOST_PAGE_SIZE = (1ULL<<GPUCOMM_HOST_PAGE_BITS); //!< Host page size (4096 bytes)
constexpr uint64_t GPUCOMM_HOST_PAGE_OFF = (GPUCOMM_HOST_PAGE_SIZE-1);      //!< Host page offset mask
constexpr uint64_t GPUCOMM_HOST_PAGE_MASK = (~(GPUCOMM_HOST_PAGE_OFF));     //!< Host page address mask

// Slot and symbol configuration
constexpr uint32_t kPeerSlotsInfo = 16;                      //!< Number of slots per peer
constexpr uint32_t kPeerSymbolsInfo = 14;                    //!< Number of OFDM symbols per slot
static_assert(PARTIAL_UPLANE_SYMBOLS_PER_SLOT == kPeerSymbolsInfo, "partial_uplane_slot_info must match defaults");
constexpr uint32_t THREAD_PER_PACKET_PRB_PHASE = 4;          //!< Threads per packet PRB phase
constexpr uint32_t THREAD_PER_PACKET_COPY = 8;               //!< Threads per packet copy operation
constexpr uint32_t kThreadSymbol = 16;                       //!< Threads per symbol
constexpr uint32_t kPeerSymbolPrbBox = 16 * 32;              //!< PRB boxes per symbol (16 AP × 32 boxes)

/**
 * GPU-comm peer count for DOCA TX buffer sizing (init path).
 *
 * cell_group_num from YAML, rounded up to the next power of 2 (DOCA ring
 * requirement), capped at API_MAX_NUM_CELLS. Used at Nic/Flow init and for
 * hot-path bound checks. Per-slot stack arrays in GpuComm must stay sized to
 * API_MAX_NUM_CELLS — never use this value as a VLA bound.
 */
[[nodiscard]] inline uint32_t getGpuCommSendPeers()
{
    const uint32_t cellGroupNum = AppConfig::getInstance().getCellGroupNum();
    if(cellGroupNum == 0)
    {
        // Called before setCellGroupNum(); fall back so init does not size a 0-length ring.
        return API_MAX_NUM_CELLS;
    }

    const uint32_t peers =
        (rte_is_power_of_2(cellGroupNum) == 0) ? rte_align32pow2(cellGroupNum) : cellGroupNum;
    return (peers > API_MAX_NUM_CELLS) ? API_MAX_NUM_CELLS : peers;
}

// PRB split info size
constexpr uint32_t kPrbSplitInfo = kPeerSlotsInfo * kPeerSymbolsInfo * 32 * 273;  //!< Total PRB split info entries

// Compression constants
constexpr int      BFP_NO_COMPRESSION  = 16;  //!< BFP no compression (16-bit samples)
constexpr int      FIXED_POINT_16_BITS = 17;  //!< Fixed point 16-bit mode indicator

// DOCA queue and buffer limits
constexpr uint32_t QUEUE_DESC = 8192;        //!< Queue descriptor count
constexpr uint32_t MAX_PKT_SIZE = 8192;      //!< Maximum packet size in bytes
constexpr uint32_t MAX_PKT_NUM = 16384;      //!< Maximum number of packets
constexpr uint32_t MAX_SEM_ITEMS = 4096;     //!< Maximum semaphore items

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_DEFAULTS_HPP__
