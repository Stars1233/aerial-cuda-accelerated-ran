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

#include "scf_5g_csirs_slot_command_helpers.hpp"

#include "nvlog_fmt.hpp"
#include <aerial/casts/casts.hpp>
#include <algorithm>
#include <cmath>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

namespace scf_5g_fapi {

namespace {

void update_new_csirs_pm(const scf_fapi_csi_rsi_pdu_t& msg_csirs,
                         cuphyCsirsRrcDynPrm_t& csirs_rrc_dyn_params,
                         slot_command_api::pm_group*                prec_group,
                         const pm_weight_map_t&   pm_map,
                         nv::phy_config_option&   config_options,
                         const int32_t            cell_index)
{
    static_assert(sizeof(decltype(msg_csirs.pc_and_bf)) == sizeof(scf_fapi_tx_precoding_beamforming_t),
                  "FAPI CSI-RS pc_and_bf layout mismatch");
    const auto& pdu = aerial::casts::assume_cast_ref<const scf_fapi_tx_precoding_beamforming_t>(msg_csirs.pc_and_bf);
    csirs_rrc_dyn_params.enablePrcdBf = config_options.precoding_enabled;

    auto default_values = [&csirs_rrc_dyn_params]() {
        csirs_rrc_dyn_params.enablePrcdBf = false;
    };

    uint16_t offset = 0;

    // enablePrcdBf is sticky-false: line `enablePrcdBf = enablePrcdBf && (pdu_pmi != 0)`
    // and default_values() can only clear it. Once disabled, no remaining PRG can write
    // pmwPrmIdx or mutate prec_group, so break out instead of spinning the loop.
    for (uint16_t i = 0; i < pdu.num_prgs; i++)
    {
        const uint16_t pdu_pmi   = pdu.pm_idx_and_beam_idx[i + offset];
        const uint32_t cache_pmi = pdu_pmi | static_cast<uint32_t>(cell_index) << 16;

        csirs_rrc_dyn_params.enablePrcdBf = csirs_rrc_dyn_params.enablePrcdBf && (pdu_pmi != 0);
        if (!csirs_rrc_dyn_params.enablePrcdBf) { break; }

        auto pmw_iter = pm_map.find(cache_pmi);
        if (pmw_iter == pm_map.end())
        {
            default_values();
            break;
        }

        if (pmw_iter->second.layers != 1)
        {
            default_values();
            break;
        }

        auto iter = std::find_if(prec_group->csirs_pmw_idx_cache.begin(),
                                 prec_group->csirs_pmw_idx_cache.end(),
                                 [&cache_pmi](const auto& e) { return e.pmwIdx == cache_pmi; });

        if (iter == prec_group->csirs_pmw_idx_cache.end())
        {
            // Use the per-channel counter nPmCsirs (not the cross-channel
            // pm_group::nCacheEntries) so this cache and csirs_list grow in
            // lockstep, both bounded by MAX_CSIRS_OCCASIONS_PER_SLOT *
            // MAX_CELLS_PER_CELL_GROUP. nCacheEntries is shared with SSB/PDCCH
            // caches (each of a different size) and would also race with the
            // SSB/PDCCH parallel aggregation tasks against the same pm_group
            // instance — incorrect for both reasons.
            auto& cache_entry = prec_group->csirs_pmw_idx_cache[prec_group->nPmCsirs];
            cache_entry.pmwIdx = cache_pmi;
            cache_entry.nIndex = prec_group->nPmCsirs;
            auto& val = prec_group->csirs_list[prec_group->nPmCsirs];
            val.nPorts = pmw_iter->second.weights.nPorts;
            csirs_rrc_dyn_params.pmwPrmIdx = prec_group->nPmCsirs;
            std::copy(pmw_iter->second.weights.matrix,
                      pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports),
                      val.matrix);
            prec_group->nPmCsirs++;
        }
        else
        {
            csirs_rrc_dyn_params.pmwPrmIdx = iter->nIndex;
        }
        offset += static_cast<uint16_t>(pdu.dig_bf_interfaces + 1);
    }
}

} // namespace

void fill_csirs_rrc_dyn_from_fapi(const scf_fapi_csi_rsi_pdu_t& msg,
                                  cuphyCsirsRrcDynPrm_t&       dst,
                                  const slot_command_api::slot_indication&       slotinfo,
                                  const int                    static_csi_rs_slot_num)
{
    dst.startRb        = msg.start_rb;
    dst.nRb            = msg.num_of_rbs;
    dst.freqDomain     = msg.freq_domain;
    dst.row            = msg.row;
    dst.symbL0         = msg.sym_l0;
    dst.symbL1         = msg.sym_l1;
    dst.freqDensity    = msg.freq_density;
    dst.scrambId       = msg.scrambling_id;
    dst.idxSlotInFrame = static_cast<uint16_t>((static_csi_rs_slot_num > -1) ? static_csi_rs_slot_num : slotinfo.slot_);
    dst.csiType        = static_cast<cuphyCsiType_t>(msg.csi_type);
    dst.cdmType        = static_cast<cuphyCdmType_t>(msg.cdm_type);
    dst.beta           = std::pow(10.0, (msg.tx_power.power_control_offset_ss - 1) * 3.0 / 20.0);
}

bool should_skip_zp_csirs_without_pdsch(const cuphyCsiType_t csi_type, const bool has_pdsch_pdus)
{
    return (csi_type == cuphyCsiType_t::ZP_CSI_RS) && !has_pdsch_pdus;
}

void apply_csirs_fapi_pdu_to_slot_command(slot_command_api::cell_group_command&           cell_grp_cmd,
                                          slot_command_api::cell_sub_command&             cell_cmd,
                                          const scf_fapi_csi_rsi_pdu_t& msg,
                                          const slot_command_api::slot_indication&        slotinfo,
                                          const int32_t                 cell_index,
                                          nv::phy_config_option&        config_option,
                                          pm_weight_map_t&              pm_map,
                                          const uint32_t                csirs_offset,
                                          const bool                    has_pdsch_pdus,
                                          const uint16_t                cell_stat_prm_idx,
                                          const bool                    mmimo_enabled,
                                          const uint32_t                pdsch_dyn_idx)
{
    const cuphyCsiType_t csi_type = static_cast<cuphyCsiType_t>(msg.csi_type);
    if (should_skip_zp_csirs_without_pdsch(csi_type, has_pdsch_pdus))
    {
        NVLOGD_FMT(TAG,
                   "apply_csirs: ZP CSI-RS & no PDSCH — skip");
        return;
    }

    cuphyCsirsRrcDynPrm_t* pdsch_rrc_dyn_params = nullptr;
    cell_cmd.slot.set_downlink(slotinfo);
    cell_grp_cmd.slot.set_downlink(slotinfo);
    const int staticCsiRsSlotNum    = config_option.staticCsiRsSlotNum;

    // PDSCH-side CSI-RS bookkeeping below populates pCsiRsPrms[] with the FAPI
    // shape (startRb, nRb, sym*, density, ...) plus default enablePrcdBf=false,
    // pmwPrmIdx=0. PDSCH rate-matches around these REs based on RRC scheduling,
    // independent of whether CSI-RS itself transmits this slot. The CSI-RS pipeline
    // registration further down (gated by csi_type != ZP and check_bf_pc_params)
    // only "upgrades" the precoding fields via the mirror at the end of this
    // function. If the BF check fails or the CSI-RS path bails, the safe defaults
    // remain in place — that is intentional, not stale state.
    if (has_pdsch_pdus)
    {
        slot_command_api::pdsch_params* pdsch_params = cell_grp_cmd.get_pdsch_params();
        const std::size_t dyn_idx = static_cast<std::size_t>(pdsch_dyn_idx);
        pdsch_params->cell_grp_info.nCsiRsPrms++;
        pdsch_params->cell_dyn_info[dyn_idx].csiRsPrmsOffset = csirs_offset;
        pdsch_params->cell_dyn_info[dyn_idx].nCsiRsPrms++;
        pdsch_rrc_dyn_params = &pdsch_params->cell_grp_info.pCsiRsPrms[pdsch_params->num_csirs_info++];

        fill_csirs_rrc_dyn_from_fapi(msg, *pdsch_rrc_dyn_params, slotinfo, staticCsiRsSlotNum);
        pdsch_rrc_dyn_params->startRb      = static_cast<uint16_t>(msg.start_rb + msg.bwp.bwp_start);
        pdsch_rrc_dyn_params->enablePrcdBf = false;
        pdsch_rrc_dyn_params->pmwPrmIdx    = 0;

        NVLOGD_FMT(TAG,
                   "apply_csirs: PDSCH csirs_offset updated — "
                   "nCsiRsPrms={} cell_dyn_info[{}].nCsiRsPrms={} csiRsPrmsOffset={} "
                   "CSI-Type={} num_csirs_info={}",
                   pdsch_params->cell_grp_info.nCsiRsPrms,
                   dyn_idx,
                   pdsch_params->cell_dyn_info[dyn_idx].nCsiRsPrms,
                   csirs_offset, +csi_type, pdsch_params->num_csirs_info);
    }

    // TODO(post-merge follow-up): the two `return` paths below silently drop the PDU;
    // the caller `process_aggr_csirs_channel` has no way to detect or count drops
    // beyond the NVLOGE_FMT events. Returning `tl::expected<void, CsiRsError>` would
    // enable per-slot error accounting. Deferred from MR !5076 because the same
    // `void`-with-log-on-failure pattern is used pervasively across the SCF L2
    // channel-update functions (PDSCH, PDCCH, SSB, PUSCH, PUCCH, ...); a uniform
    // conversion ticket across all DL/UL channel handlers is the right scope.
    if (csi_type != cuphyCsiType_t::ZP_CSI_RS)
    {
        if (!check_bf_pc_params(msg.pc_and_bf.num_prgs, msg.pc_and_bf.dig_bf_interfaces, mmimo_enabled))
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "{} line {}: check_bf_pc_params failed: numPRGs={} digBFInterfaces={} mmimo_enabled={}",
                       __FUNCTION__,
                       __LINE__,
                       static_cast<uint16_t>(msg.pc_and_bf.num_prgs),
                       static_cast<uint16_t>(msg.pc_and_bf.dig_bf_interfaces),
                       mmimo_enabled);
            return;
        }

        cell_grp_cmd.create_if(slot_command_api::channel_type::CSI_RS);
        slot_command_api::csirs_params* csirs_params = cell_grp_cmd.csirs.get();
        if (csirs_params == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no csirs command");
            return;
        }

        if (csirs_params->symbolMapArray[cell_index])
        {
            csirs_params->symbolMapArray[cell_index] = 0;
        }

        auto it = std::find(csirs_params->cell_index_list.begin(), csirs_params->cell_index_list.end(), cell_index);
        if (it == csirs_params->cell_index_list.end())
        {
            csirs_params->cell_index_list.push_back(cell_index);
            csirs_params->phy_cell_index_list.push_back(cell_cmd.cell);
            csirs_params->cellInfo[csirs_params->nCells].rrcParamsOffset = csirs_params->nCsirsRrcDynPrm;
            csirs_params->cellInfo[csirs_params->nCells].cellPrmStatIdx = cell_stat_prm_idx;
            csirs_params->nCells++;
            csirs_params->lastCell = std::max(csirs_params->lastCell, static_cast<uint16_t>(cell_index + 1));
            NVLOGD_FMT(TAG,
                       "apply_csirs: new cell — nCells={} lastCell={} rrcParamsOffset={} cell_index={}",
                       csirs_params->nCells, csirs_params->lastCell,
                       csirs_params->cellInfo[csirs_params->nCells - 1].rrcParamsOffset, cell_index);
        }

        const uint32_t cell_info_idx = csirs_params->nCells - 1;

        cuphyCsirsRrcDynPrm_t& csirs_rrc_dyn_params = csirs_params->csirsList[csirs_params->nCsirsRrcDynPrm];
        csirs_params->nCsirsRrcDynPrm++;

        fill_csirs_rrc_dyn_from_fapi(msg, csirs_rrc_dyn_params, slotinfo, staticCsiRsSlotNum);
        update_new_csirs_pm(msg, csirs_rrc_dyn_params, cell_grp_cmd.get_pm_group(), pm_map, config_option, cell_index);
        // Upgrade PDSCH-side defaults with the actual CSI-RS precoding params now
        // that the CSI-RS pipeline is registered. Only reached when csi_type != ZP
        // and BF validation passed. If we never reach this, PDSCH keeps the safe
        // defaults written above (enablePrcdBf=false, pmwPrmIdx=0) and rate-matches
        // around the CSI-RS REs without precoding awareness — see comment above the
        // PDSCH bookkeeping block.
        if (has_pdsch_pdus)
        {
            pdsch_rrc_dyn_params->enablePrcdBf = csirs_rrc_dyn_params.enablePrcdBf;
            pdsch_rrc_dyn_params->pmwPrmIdx    = csirs_rrc_dyn_params.pmwPrmIdx;
        }

        csirs_params->cellInfo[cell_info_idx].nRrcParams++;
        NVLOGD_FMT(TAG,
                   "apply_csirs: csirs_params updated — "
                   "nCsirsRrcDynPrm={} nCells={} cell[{}].rrcParamOffset={} nRrcParams={} cell_index={}",
                   csirs_params->nCsirsRrcDynPrm,
                   csirs_params->nCells,
                   cell_info_idx,
                   csirs_params->cellInfo[cell_info_idx].rrcParamsOffset,
                   csirs_params->cellInfo[cell_info_idx].nRrcParams,
                   cell_index);
    }
}

void apply_csirs_dl_aggr_slot_command_for_pdu(slot_command_api::cell_group_command&            cell_grp_cmd,
                                               slot_command_api::cell_sub_command&              cell_cmd,
                                               const scf_fapi_csi_rsi_pdu_t&  csi_pdu,
                                               const slot_command_api::slot_indication&         slotinfo,
                                               const int32_t                  cell_index,
                                               const bool                     has_pdsch_pdus,
                                               bool&                          captured_csirs_offset,
                                               uint32_t&                      csirs_offset,
                                               nv::phy_config_option&         config_option,
                                               pm_weight_map_t&               pm_map,
                                               const bool                     mmimo_enabled,
                                               const uint32_t                 pdsch_dyn_idx)
{
    if (!captured_csirs_offset)
    {
        if (has_pdsch_pdus)
        {
            csirs_offset = cell_grp_cmd.get_pdsch_params()->cell_grp_info.nCsiRsPrms;
        }
        else
        {
            csirs_offset = 0;
        }
        captured_csirs_offset = true;
        NVLOGD_FMT(TAG,
                   "apply_csirs_aggr: captured csirs_offset={} has_pdsch={} cell_index={} "
                   "nCsiRsPrms_before={}",
                   csirs_offset, has_pdsch_pdus, cell_index,
                   has_pdsch_pdus ? cell_grp_cmd.get_pdsch_params()->cell_grp_info.nCsiRsPrms : 0);
    }

    apply_csirs_fapi_pdu_to_slot_command(cell_grp_cmd,
                                         cell_cmd,
                                         csi_pdu,
                                         slotinfo,
                                         cell_index,
                                         config_option,
                                         pm_map,
                                         csirs_offset,
                                         has_pdsch_pdus,
                                         static_cast<uint16_t>(cell_index),
                                         mmimo_enabled,
                                         pdsch_dyn_idx);
}

// TEMPORARY(csirs-fh-aggr-path): begin -- populate FH params from aggregation worker
void populate_csirs_fh_params_for_cell(
    slot_command_api::fh_prepare_callback_params& fh_params,
    slot_command_api::cell_sub_command&            cell_cmd,
    const int32_t                                 carrier_id,
    const bool                                    has_pdsch,
    const int32_t                                 cuphy_params_cell_idx,
    const bool                                    bf_enabled,
    const bool                                    mmimo_enabled,
    const uint16_t                                num_dl_prb)
{
    auto& csirs_fh = fh_params.csirs_fh_params.at(carrier_id);
    csirs_fh.cell_idx              = carrier_id;
    csirs_fh.cuphy_params_cell_idx = has_pdsch ? cuphy_params_cell_idx : -1;
    csirs_fh.cell_cmd              = &cell_cmd;
    csirs_fh.bf_enabled            = bf_enabled;
    csirs_fh.mmimo_enabled         = mmimo_enabled;
    csirs_fh.num_dl_prb            = num_dl_prb;

    fh_params.is_csirs_cell.at(carrier_id) = 1;
    fh_params.num_csirs_cell++;
}
// TEMPORARY(csirs-fh-aggr-path): end

} // namespace scf_5g_fapi
