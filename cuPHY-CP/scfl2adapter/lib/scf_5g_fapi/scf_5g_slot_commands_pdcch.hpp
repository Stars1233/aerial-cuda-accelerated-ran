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

#pragma once

#include <cstddef>

#include "slot_command/slot_command.hpp"
#include "scf_5g_fapi.h"
#include "nv_phy_fapi_msg_common.hpp"
#include "scf_5g_fapi_slot_types.hpp"
#include "nv_phy_limit_errors.hpp"


namespace scf_5g_fapi {

    using slot_command_api::cell_group_command;
    using slot_command_api::cell_sub_command;
    using slot_command_api::dci_param_list;
    using slot_command_api::pm_group;
    using slot_command_api::slot_indication;

#ifdef ENABLE_L2_SLT_RSP
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode,
        int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
        pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled, nv::pdcch_limit_error_t* pdcch_error);
#else
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode,
        int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
        pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
#endif

    void update_pdcch_sym_prb_info_for_pdcch_only_validation(
        cell_sub_command& cell_cmd,
        cuphyPdcchCoresetDynPrm_t& coreset,
        dci_param_list& dci,
        uint16_t bandwidth,
        pm_group* pm_grp,
        scf_fapi_pdcch_pdu_t& msg,
        std::size_t* fapiDciOffsets,
        nv::phy_config_option& config_option,
        nv::slot_detail_t* slot_detail,
        bool mmimo_enabled,
        int32_t cell_index);
}
