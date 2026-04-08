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

#include "slot_command/csirs_lookup.hpp"
#include "scf_5g_fapi.h"
#include "nv_phy_fapi_msg_common.hpp"
#include "nvlog_fmt.hpp"
#include "scf_5g_slot_commands_common.hpp"
#include "nv_phy_config_option.hpp"
#include "scf_5g_fapi_utils.hpp"
#include "scf_5g_slot_commands_pdsch_mod_comp.hpp"
#include "scf_5g_fh_callback_context.hpp"

using namespace slot_command_api;

namespace scf_5g_fapi {
    void update_non_overlapping_csirs(cell_sub_command& cell_cmd);
    /**
     * Update the cell command for PDSCH
     * @param cell_grp_cmd The cell_group_command data structure
     * @param cell_sub_cmd The cell_sub_command data structure
     * @param cmd The scf_fapi_pdsch_pdu_t data structure
     * @param testMode The test mode
     * @param slot The slot information, including SFN, slot, and tick
     * @param cell_index The cell index
     * @param pm_map The PM map
     * @return true if PDSCH was accepted, false if rejected (e.g. check_bf_pc_params failed). */
    bool update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_pdsch_pdu_t& cmd, uint8_t testMode, slot_indication& slot, int32_t cell_index, pm_weight_map_t& pm_map, bool pm_enabled, bool bf_enabled, uint16_t num_dl_prb, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool mmimo_enabled, nv::slot_detail_t* slot_detail);

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, const scf_fapi_csi_rsi_pdu_t& msg, slot_indication & slotinfo, int32_t cell_index, cuphyCellStatPrm_t cell_params,nv::phy_config_option& config_option, pm_weight_map_t& pm_map, uint32_t csirs_offset, bool pdsch_exist, uint16_t cell_stat_prm_idx, bool mmimo_enabled, nv::slot_detail_t* slot_detail);
#ifdef ENABLE_L2_SLT_RSP
    void reset_pdsch_cw_offset();
#endif
    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch(const pm_weight_map_t& pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail);
    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs_mmimo(const pm_weight_map_t& pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, bool csirs_compact_mode);
    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs_nonmmimo(const pm_weight_map_t& pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, bool csirs_compact_mode);
    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs(const pm_weight_map_t& pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, bool csirs_compact_mode);
    void update_static_bf_wt_csi_rs(int32_t cell_index, tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, uint16_t beam_id, bool modcomp_enabled);

    template <bool config_options_precoding_enabled, bool config_options_bf_enabled>
    void update_fh_params_csirs_remap(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail, const CsirsFhParamsView& csirs_fh_params, bool& csirs_compact_mode);
    void update_beam_list_csirs(beamid_array_t& array, size_t& array_size, tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, int32_t cell_idx, bool mmimo_enabled, bool modcomp_enabled);
    template <bool mplane_configured_ru_type, bool config_options_precoding_enabled, bool config_options_bf_enabled>
    void fh_callback(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);


    struct PdschFhContext {
        struct NewGroupPolicy {
            static inline void execute_impl(const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail);
        };

        struct NonGroupPolicy{
            static inline void execute_impl(const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail);
        };
    };

    struct PdschCsirsFhContext {
    };

    /// @brief Process consecutive bits in a mask and call a function with the sequence info
    /// @tparam Func 
    /// @tparam ...Args 
    /// @param mask 
    /// @param func 
    /// @param ...args 
    template<typename Func, typename... Args>
    void processConsecutiveBits(uint16_t mask, Func&& func, Args&&... args) {
        if (!mask) return;
        
        uint16_t pos = 0;
        while (mask) {
            // Find first set bit
            uint16_t start = __builtin_ctz(mask);
            // Find length of consecutive 1s starting at start
            uint16_t ones = __builtin_ctz(~(mask >> start));
            
            // Call function with the sequence info
            func(start, ones, std::forward<Args>(args)...);
            
            // Clear the processed bits and continue
            uint16_t sequence_mask = ((1U << ones) - 1) << start;
            mask &= ~sequence_mask;
        }
    }

    template<typename Func, typename... Args>
    void processSetBits(uint16_t mask, Func&& func, Args&&... args) {
        if (!mask) return;
        
        while (mask) {
            // Find position of least significant set bit
            uint16_t pos = __builtin_ctz(mask);
            
            // Call function for this bit position
            func(pos, 1, std::forward<Args>(args)...);
            
            // Clear the least significant bit
            mask &= (mask - 1);
        }
    }
}
