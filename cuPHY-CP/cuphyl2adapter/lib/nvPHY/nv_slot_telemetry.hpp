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

/**
 * @file nv_slot_telemetry.hpp
 * @brief Per-ring L2A telemetry snapshot + ingestion-time stamp helper.
 */

#ifndef NV_SLOT_TELEMETRY_HPP
#define NV_SLOT_TELEMETRY_HPP

#include <chrono>
#include <cstdint>

#include "nv_phy_mac_transport.hpp"   // phy_mac_msg_desc, sfn_slot_t

namespace nv {

struct SlotTelemetrySnapshot final
{
    sfn_slot_t               ss{};
    std::chrono::nanoseconds current_tick{};       ///< from current_tick_list_[slot%10]
    std::chrono::nanoseconds l1_slot_ind_tick{};   ///< from l1_slot_ind_tick_[slot%10]
    std::chrono::nanoseconds last_fapi_msg_tick{};
    std::chrono::nanoseconds l2a_start_tick{};
    std::chrono::nanoseconds l2a_end_tick{};       ///< written by publish_slot_command
    uint32_t                 slot_interval{0};     ///< get_fapi_latency snapshot
    bool                     is_ul_slot{false};
    bool                     is_dl_slot{false};
    bool                     is_csirs_slot{false};
};

/**
 * @brief Stamp ingestion-time fields on @p snap for one payload msg.
 *
 * Sets @c last_fapi_msg_tick unconditionally; @c is_dl_slot / @c is_ul_slot
 * per msg_id; @c is_csirs_slot when @p has_csirs is true (caller uses the
 * store-time sidecar); gates @c l2a_start_tick on the first slot-typed msg.
 *
 * @param[in,out] snap      Per-ring snapshot to mutate.
 * @param[in]     now       Caller's clock read.
 * @param[in]     msg_id    FAPI msg_id.
 * @param[in]     has_csirs True when this DL_TTI carries CSI-RS (sidecar).
 */
void stamp_ring_telemetry(SlotTelemetrySnapshot&     snap,
                          std::chrono::nanoseconds   now,
                          int32_t                    msg_id,
                          bool                       has_csirs = false);

} // namespace nv

#endif // NV_SLOT_TELEMETRY_HPP
