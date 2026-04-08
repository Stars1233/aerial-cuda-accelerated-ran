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

#include "scf_5g_fapi.hpp"
#include "nv_phy_fapi_msg_common.hpp"
#include "scf_5g_fapi_phy.hpp"
#include "slot_command/slot_command.hpp"
using namespace slot_command_api;
#include "scf_5g_fapi.h"
#include "scf_5g_slot_commands_common.hpp"
#include "scf_5g_slot_commands_pdsch_csirs.hpp"
#include "scf_5g_slot_commands_ssb_pbch.hpp"
#include "scf_5g_slot_commands_pdcch.hpp"
#include "scf_5g_fapi_utils.hpp"

// #include <unordered_map>

namespace scf_5g_fapi
{
    using pm_weight_map_t = std::unordered_map<uint32_t, pm_weights_t>;
    using static_digBeam_weight_map_t = std::unordered_map<uint16_t, digBeam_t>;
    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_pusch_pdu_t& msg, int32_t cell_index, slot_indication & slotinfo, int staticPuschSlotNum, uint8_t lbrm, bool bf_enabled, uint16_t cell_stat_prm_idx, float dtx_threshold, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool mmimo_enabled, nv::slot_detail_t* slot_detail, uint16_t ul_bandwidth);
    int update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_srs_pdu_t& msg, int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t cell_params, uint16_t cell_stat_prm_idx, bool bf_enabled, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, int mutiple_srs_ind_allowed, nv::phy_mac_transport& transport, bool is_last_srs_pdu, bool is_last_non_prach_pdu, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
    bool update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_pdsch_pdu_t& cmd, uint8_t testMode, slot_indication& slot, int32_t cell_index, pm_weight_map_t& pm_map, bool pm_enabled, bool bf_enabled, uint16_t num_dl_prb, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool mmimo_enabled, nv::slot_detail_t* slot_detail);
    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, const scf_fapi_csi_rsi_pdu_t& msg, slot_indication & slotinfo, int32_t cell_index, cuphyCellStatPrm_t cell_params,nv::phy_config_option& config_option, pm_weight_map_t& pm_map, uint32_t csirs_offset, bool pdsch_exist, uint16_t cell_stat_prm_idx, bool mmimo_enabled, nv::slot_detail_t* slot_detail);
    void update_cell_command(cell_group_command* cell_group_cmd, cell_sub_command& cell_cmd, void* buffer, bool ssb, uint32_t cell_index,int buffLoc,  nv::slot_detail_t* slot_detail);
    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, slot_indication & slotinfo, const scf_fapi_pucch_pdu_t& pdu, int32_t cell_index,const nv::pucch_dtx_t_list& dtx_thresholds,uint16_t cell_stat_prm_idx, nv::phy_config_option& config_option, nv::slot_detail_t* slot_detail, bool mmimo_enabled, uint16_t ul_bandwidth, uint16_t pucch_hopping_id);
    void update_cell_command(cell_group_command* grp, cell_sub_command& cell_cmd, slot_indication & slotinfo, const scf_fapi_prach_pdu_t& req, nv::phy_config& cell_params, nv::prach_addln_config_t& addln_config, int32_t cell_index, bool bf_enabled, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_dl_bfw_group_config_t& msg, int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t cell_params, bfw_coeff_mem_info_t *bfwCoeff_mem_info,bfw_type bfwType, nv::slot_detail_t* slot_detail, uint32_t &droppedBFWPdu);
    
    // SRS slot finalization function - populates PRB parameters based on accumulated SRS state
    void finalize_srs_slot(cell_sub_command& cell_cmd, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, uint8_t nSrsSym, uint8_t srsStartSym, srs_params *srs_params, bool bf_enabled, enum ru_type ru, nv::slot_detail_t* slot_detail, int32_t cell_index, bool last_non_prach_pdu);

#ifndef ENABLE_L2_SLT_RSP
    void reset_cell_command(cell_sub_command& command, slot_command_api::slot_indication& slot_ind, int32_t cell_index, bool cell_group, cell_group_command* cmd);
#endif
    void reset_cell_command(cell_sub_command& command, slot_command_api::slot_indication& slot_ind, int32_t cell_index, bool cell_group);

}
