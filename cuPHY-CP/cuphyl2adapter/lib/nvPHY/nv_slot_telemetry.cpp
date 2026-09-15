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
 * @file nv_slot_telemetry.cpp
 * @brief Definition of nv::stamp_ring_telemetry.
 */

#include "nv_slot_telemetry.hpp"

#include "scf_5g_fapi.h" // SCF_FAPI_*_REQUEST

namespace nv {

void stamp_ring_telemetry(SlotTelemetrySnapshot&   snap,
                          std::chrono::nanoseconds now,
                          int32_t                  msg_id,
                          bool                     has_csirs)
{
    snap.last_fapi_msg_tick = now;

    switch (msg_id)
    {
    case SCF_FAPI_DL_TTI_REQUEST:
        snap.is_dl_slot = true;
        if (has_csirs)
        {
            snap.is_csirs_slot = true;
        }
        break;
    case SCF_FAPI_UL_DCI_REQUEST:
        snap.is_dl_slot = true;
        break;
    case SCF_FAPI_UL_TTI_REQUEST:
        snap.is_ul_slot = true;
        break;
    case SCF_FAPI_DL_BFW_CVI_REQUEST:
        snap.is_dl_slot = true;
        break;
    case SCF_FAPI_UL_BFW_CVI_REQUEST:
        snap.is_ul_slot = true;
        break;
    default:
        // TX_DATA + unknown: only last_fapi_msg_tick (stamped above); no gate.
        return;
    }

    if (snap.l2a_start_tick.count() == 0)
    {
        snap.l2a_start_tick = now;
    }
}

} // namespace nv
