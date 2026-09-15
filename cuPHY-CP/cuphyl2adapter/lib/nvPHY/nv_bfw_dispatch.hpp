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

#ifndef NV_BFW_DISPATCH_HPP
#define NV_BFW_DISPATCH_HPP

#include "nv_fapi_message_storage.hpp"   // FapiSlotMessageStorage, phy_mac_msg_desc
#include "nvlog.hpp"                     // NVLOGD_FMT, NVLOGW_FMT
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <span>

namespace nv
{

/**
 * Dispatch stored BFW lane messages to per-cell handlers with bounds checking.
 *
 * Unified template shared by DL and UL BFW dispatch; the caller supplies a
 * span over the lane-specific message array.
 *
 * @tparam Fn                Callable void(int cell_id, const phy_mac_msg_desc& msg).
 * @param msgs               Span over the BFW lane messages to dispatch.
 * @param num_phy_instances  Upper bound for valid cell_id (exclusive).
 * @param per_cell           Per-cell callback invoked for each in-range message.
 * @param slot_u32           Packed SFN/slot for logging.
 * @param ring_idx           Ring index for logging.
 * @param tag                Log prefix string (e.g. "dispatch_dl_bfw" or "dispatch_ul_bfw").
 */
template<typename Fn>
void dispatch_bfw_messages(std::span<const phy_mac_msg_desc> msgs,
                           std::size_t num_phy_instances,
                           Fn&&        per_cell,
                           uint32_t    slot_u32,
                           uint32_t    ring_idx,
                           const char* tag)
{
    static_assert(std::is_invocable_v<Fn, int, const phy_mac_msg_desc&>);
 
    NVLOGD_FMT((NVLOG_TAG_BASE_L2_ADAPTER + 6),
               "{}: slot=0x{:08X} ring_idx={} bfw_msgs={}",
               tag, slot_u32, ring_idx, msgs.size());
 
    for (std::size_t i = 0; i < msgs.size(); ++i) {
        const phy_mac_msg_desc& msg = msgs[i];
        if (msg.cell_id < 0 ||
            static_cast<std::size_t>(msg.cell_id) >= num_phy_instances)
        {
            NVLOGW_FMT((NVLOG_TAG_BASE_L2_ADAPTER + 6),
                       "{}: invalid cell_id={} index={} "
                       "msg_id=0x{:02X} num_phy_instances={}",
                       tag, msg.cell_id, i, msg.msg_id, num_phy_instances);
            continue;
        }
        per_cell(msg.cell_id, msg);
    }
}

} // namespace nv

#endif // NV_BFW_DISPATCH_HPP
