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

#include "nv_fapi_message_storage.hpp"
#include "nv_fapi_tti_pdu_counts.hpp"
#include "nvlog.hpp"
#include "scf_5g_fapi.h"
#include <utility>

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 6)

namespace nv
{

/**
 * @brief Construct slot storage with `capacity_per_type` slots per message-type lane.
 */
FapiSlotMessageStorage::FapiSlotMessageStorage(std::size_t capacity_per_type)
{
    reserve_per_type(capacity_per_type);
}

/**
 * @brief Allocate (or grow) the single flat buffer: NUM_TYPES lanes × capacity each.
 *
 * All lane pointers remain valid after this call; the per-type counts_ are not reset.
 * PDU-count sidecars are fixed-capacity arrays (no heap). Call only at ring
 * construction time, not on the hot path.
 */
void FapiSlotMessageStorage::reserve_per_type(std::size_t capacity_per_type)
{
    // Sidecars are fixed-capacity arrays sized to k_fapi_lane_capacity_max; clamp
    // so store_message's write_idx can never index past them, even in release
    // builds where the assert is stripped. A capacity above the max is an
    // unsupported cell count; the excess lanes are then rejected by store_message.
    assert(capacity_per_type <= k_fapi_lane_capacity_max);
    if (capacity_per_type > k_fapi_lane_capacity_max)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "reserve_per_type: capacity {} exceeds max {}; clamping",
                   capacity_per_type,
                   k_fapi_lane_capacity_max);
        capacity_per_type = k_fapi_lane_capacity_max;
    }
    capacity_per_type_ = capacity_per_type;
    msgs_.resize(NUM_TYPES * capacity_per_type_);
}

bool FapiSlotMessageStorage::dl_tti_has_pdsch_pdu(const uint16_t index) const
{
    if (index >= count(MsgType::DL_TTI))
    {
        return false;
    }
    return dl_tti_pdu_counts_[index].by_type[DL_TTI_NPDUS_IDX_PDSCH] > 0;
}

NvDlTtiPduCounts FapiSlotMessageStorage::dl_tti_pdu_counts(uint16_t index) const noexcept
{
    if (index >= count(MsgType::DL_TTI))
    {
        return {};
    }
    return dl_tti_pdu_counts_[index];
}

NvUlTtiPduCounts FapiSlotMessageStorage::ul_tti_pdu_counts(uint16_t index) const noexcept
{
    if (index >= count(MsgType::UL_TTI))
    {
        return {};
    }
    return ul_tti_pdu_counts_[index];
}

/**
 * @brief Store a slot-scoped message into the appropriate lane.
 *
 * Hot path: one switch, one bounds check, one 40-byte write, one count
 * increment. For DL_TTI / UL_TTI also fills the fixed-capacity PDU-count
 * sidecar (no heap). No vector header updates, no capacity/size pair reads.
 */
StoreResult FapiSlotMessageStorage::store_message(const phy_mac_msg_desc& msg)
{
    std::size_t type_idx = 0;
    switch (msg.msg_id)
    {
        case SCF_FAPI_DL_TTI_REQUEST:      type_idx = to_underlying(MsgType::DL_TTI);  break;
        case SCF_FAPI_UL_TTI_REQUEST:      type_idx = to_underlying(MsgType::UL_TTI);  break;
        case SCF_FAPI_UL_DCI_REQUEST:      type_idx = to_underlying(MsgType::UL_DCI);  break;
        case SCF_FAPI_TX_DATA_REQUEST:     type_idx = to_underlying(MsgType::TX_DATA); break;
        case SCF_FAPI_DL_BFW_CVI_REQUEST:  type_idx = to_underlying(MsgType::DL_BFW); break;
        case SCF_FAPI_UL_BFW_CVI_REQUEST:  type_idx = to_underlying(MsgType::UL_BFW); break;
        // Control messages handled on ingress; not replayed from slot storage.
        case SCF_FAPI_SLOT_INDICATION:
        case SCF_FAPI_ERROR_INDICATION:
        // EOM signal handled on ingress (updates fapi_eom_rcvd_bitmap); not a payload.
        case SCF_FAPI_SLOT_RESPONSE:
            return StoreResult::ControlSkip;
        default:
            NVLOGW_FMT(TAG, "Ignored unknown slot msg_id=0x{:02X} cell_id={}", msg.msg_id, msg.cell_id);
            return StoreResult::Rejected;
    }

    const uint16_t write_idx = counts_[type_idx].load(std::memory_order_relaxed);
    if (write_idx >= capacity_per_type_)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "store_message: lane overflow msg_id=0x{:02X} cell_id={} count={} capacity={}",
                   msg.msg_id,
                   msg.cell_id,
                   write_idx,
                   capacity_per_type_);
        return StoreResult::Rejected;
    }
    msgs_[type_idx * capacity_per_type_ + write_idx] = msg;
    if (type_idx == to_underlying(MsgType::DL_TTI))
    {
        dl_tti_pdu_counts_[write_idx] = make_dl_tti_pdu_counts_from_msg(msg);
    }
    else if (type_idx == to_underlying(MsgType::UL_TTI))
    {
        ul_tti_pdu_counts_[write_idx] = make_ul_tti_pdu_counts_from_msg(msg);
    }
    // Release so a consumer that acquire-loads the count (e.g. the release paths)
    // observes the descriptor (and any sidecar) written above before it sees the
    // incremented count.
    counts_[type_idx].store(static_cast<uint16_t>(write_idx + 1), std::memory_order_release);
    return StoreResult::Stored;
}

/**
 * @brief Clear stored messages without releasing transport buffers.
 *
 * Resets per-lane counts to zero; the flat buffer allocation and capacity are
 * unchanged so the next slot can write immediately. Sidecar arrays are not
 * zeroed — entries are overwritten on the next store for that index.
 */
void FapiSlotMessageStorage::clear()
{
    for (auto& c : counts_)
    {
        c.store(0, std::memory_order_relaxed);
    }
    tracked_slot_.reset();
    ready_ = false;
}

void FapiSlotMessageStorage::reset_for_slot(uint32_t slot_u32)
{
    clear();
    tracked_slot_ = slot_u32;
}

/**
 * @brief Map msg_id to the matching optional slot, or nullptr if unsupported.
 */
std::optional<phy_mac_msg_desc>* FapiNonSlotMessageStorage::find_slot(uint8_t msg_id)
{
    switch (msg_id)
    {
        case SCF_FAPI_CONFIG_REQUEST: return &config_request_;
        case SCF_FAPI_START_REQUEST:  return &start_request_;
        case SCF_FAPI_STOP_REQUEST:   return &stop_request_;
        default:                      return nullptr;
    }
}

/**
 * @brief Clear stored non-slot messages without releasing buffers.
 */
void FapiNonSlotMessageStorage::clear()
{
    config_request_.reset();
    start_request_.reset();
    stop_request_.reset();
}

} // namespace nv
