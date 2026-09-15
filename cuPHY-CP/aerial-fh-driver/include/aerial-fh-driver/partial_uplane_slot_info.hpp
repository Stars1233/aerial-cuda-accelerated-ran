/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef AERIAL_FH_DRIVER_PARTIAL_UPLANE_SLOT_INFO_HPP__
#define AERIAL_FH_DRIVER_PARTIAL_UPLANE_SLOT_INFO_HPP__

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda/std/array>

/**
 * @file partial_uplane_slot_info.hpp
 * @brief Public definitions for Partial U-plane slot info types.
 *
 * Use this header when you need to fill or inspect PartialUplaneSlotInfo_t
 * (e.g. fapi_to_cplane_direct) without including internal fronthaul headers.
 * Layout must match lib/fronthaul.hpp.
 */

/** Number of OFDM symbols per slot (must match defaults.hpp kPeerSymbolsInfo) */
inline constexpr uint32_t PARTIAL_UPLANE_SYMBOLS_PER_SLOT = 14;

/** Maximum messages per symbol */
inline constexpr uint32_t TMP_MAX_MESSAGES_PER_SYMBOL = 240;

/** Max flows (must match defaults.hpp kMaxFlows / MAX_DL_EAXCIDS) */
inline constexpr int PARTIAL_UPLANE_MAX_FLOWS = 32;

/**
 * Partial section info per symbol
 */
struct PartialSectionInfoPerSymbol {
    uint32_t wci{};                     //!< WQE control information
    uint16_t num_messages{};            //!< Number of messages in symbol
    uint64_t ptp_ts{};                  //!< PTP timestamp
    uint64_t ts{};                      //!< System timestamp
    uint16_t num_packets{};             //!< Number of packets
    uint16_t cumulative_packets{};      //!< Cumulative packet count
};

using PartialSectionInfoPerSymbol_t = PartialSectionInfoPerSymbol;

/**
 * Modulation compression parameters per message per symbol
 */
struct ModCompPartialSectionInfoPerMessagePerSymbol {
    uint8_t prb_size_upl[TMP_MAX_MESSAGES_PER_SYMBOL]{};      //!< PRB size per message
    uint8_t mod_comp_enabled[TMP_MAX_MESSAGES_PER_SYMBOL]{};  //!< Modulation compression enabled flag per message
};

using ModCompPartialSectionInfoPerMessagePerSymbol_t = ModCompPartialSectionInfoPerMessagePerSymbol;

/**
 * Partial section info per message per symbol
 */
struct PartialSectionInfoPerMessagePerSymbol {
    uint16_t num_prbu[TMP_MAX_MESSAGES_PER_SYMBOL]{};         //!< Number of PRBs per message
    uint16_t start_prbu[TMP_MAX_MESSAGES_PER_SYMBOL]{};       //!< Starting PRB per message
    uint16_t rb[TMP_MAX_MESSAGES_PER_SYMBOL]{};               //!< Resource block indicator per message
    uint16_t section_id[TMP_MAX_MESSAGES_PER_SYMBOL]{};       //!< Section ID per message
    uint16_t num_packets[TMP_MAX_MESSAGES_PER_SYMBOL]{};      //!< Packet count per message
    uint16_t num_bytes[TMP_MAX_MESSAGES_PER_SYMBOL]{};        //!< Byte count per message
    uint16_t flow_index_info[TMP_MAX_MESSAGES_PER_SYMBOL]{};  //!< Flow index (eAxC ID) per message
    ModCompPartialSectionInfoPerMessagePerSymbol_t *mod_comp_params{}; //!< Modulation compression params (nullptr if disabled)
};

using PartialSectionInfoPerMessagePerSymbol_t = PartialSectionInfoPerMessagePerSymbol;

/**
 * Partial flow info per slot
 */
struct PartialFlowInfoPerSlot {
    cuda::std::array<int, PARTIAL_UPLANE_MAX_FLOWS> flow_eaxcid;                                 //!< eAxC ID per flow
    cuda::std::array<int, PARTIAL_UPLANE_MAX_FLOWS> flow_packet_count;                            //!< Total packet count per flow
    cuda::std::array<cuda::std::array<int, PARTIAL_UPLANE_SYMBOLS_PER_SLOT>, PARTIAL_UPLANE_MAX_FLOWS> sym_flow_packet_count;   //!< Per-symbol packet count per flow
    cuda::std::array<cuda::std::array<int, PARTIAL_UPLANE_SYMBOLS_PER_SLOT>, PARTIAL_UPLANE_MAX_FLOWS> cumulative_sym_flow_packet_count; //!< Cumulative per-symbol count per flow
    uint32_t num_flows;                                                             //!< Number of flows
};

using PartialFlowInfoPerSlot_t = PartialFlowInfoPerSlot;

/**
 * Partial U-plane slot information
 *
 * Lightweight slot info optimized for GPU access.
 * Contains timing, section, and flow information without full symbol details.
 */
struct PartialUplaneSlotInfo {
    uint32_t frame_8b_subframe_4b_slot_6b;                       //!< Packed frame/subframe/slot ID
    uint32_t qp_clock_id;                                        //!< QP clock ID (common for all symbols)
    uint32_t ttl_pkts;                                           //!< Total packets in slot
    uint32_t total_num_flows;                                    //!< Total number of flows
    uint16_t syms_with_packets;                                  //!< Symbols containing packets
    uint16_t last_sym_with_packets;                               //!< Last symbol with packets
    PartialSectionInfoPerSymbol_t section_info[PARTIAL_UPLANE_SYMBOLS_PER_SLOT]; //!< Per-symbol section info
    PartialSectionInfoPerMessagePerSymbol_t message_info[PARTIAL_UPLANE_SYMBOLS_PER_SLOT]; //!< Per-symbol message info
    PartialFlowInfoPerSlot_t flowInfo_slot;                      //!< Flow info for slot
};

using PartialUplaneSlotInfo_t = PartialUplaneSlotInfo;

/**
 * Parameters extracted from the Peer needed to convert C-plane slot data into
 * PartialUplaneSlotInfo_t.  Self-contained POD struct (fixed-size arrays, no
 * spans) so it can live on the stack with no external backing storage.
 */
struct UplaneConversionParams final {
    std::size_t  prb_size_upl{};       //!< get_prb_size(iq_width, comp_method), 0 for mod comp
    std::size_t  prbs_per_pkt_upl{};   //!< (mtu - ORAN_IQ_HDR_SZ) / prb_size_upl, 0 for mod comp
    std::size_t  oran_hdr_size{};      //!< ORAN_IQ_HDR_SZ (per-packet header overhead)
    std::uint16_t mtu{};               //!< NIC MTU (needed for mod-comp per-section prb_size calc)
    std::uint16_t bandwidth_prbs{};    //!< Cell bandwidth in PRBs (for numPrbc=0 resolution)

    bool          comm_via_cpu{};
    std::uint32_t total_num_flows{};   //!< Distinct DL U-plane eAxCs (dlu_eaxcid_idx_mp.size())
    std::uint16_t num_ports{};         //!< Number of valid entries in dl_eaxcid_list / dl_flow_index_map
    std::array<std::uint16_t, PARTIAL_UPLANE_MAX_FLOWS> dl_eaxcid_list{};    //!< DL (PDSCH) eAxC IDs per port index
    std::array<std::uint16_t, PARTIAL_UPLANE_MAX_FLOWS> dl_flow_index_map{}; //!< dlu_eaxcid_idx_mp[eaxcid] per port (DL flow map)

    std::uint64_t cell_start_time_ns{};
    std::uint64_t symbol_duration_ns{};
    std::uint32_t qp_clock_id{};
    std::array<std::uint64_t, PARTIAL_UPLANE_SYMBOLS_PER_SLOT> ts{}; //!< DOCA timestamps per symbol
};

#endif // AERIAL_FH_DRIVER_PARTIAL_UPLANE_SLOT_INFO_HPP__
