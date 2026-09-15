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

#if !defined(SCF_5G_FAPI_PUCCH_PDU_PARSER_COMMON_HPP_INCLUDED_)
#define SCF_5G_FAPI_PUCCH_PDU_PARSER_COMMON_HPP_INCLUDED_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <algorithm>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi_ul_validate.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi
{
/**
 * Append PUCCH Order metadata to a caller-provided slot_info_t.
 */
void append_pucch_order_prbs(cuphyPucchUciPrm_t& uci_info,
                             uint16_t prb_size,
                             const scf_fapi_rx_beamforming_t& pmi_bf_pdu,
                             slot_command_api::slot_info_t& sym_prbs,
                             bool bf_enabled,
                             enum ru_type ru,
                             nv::slot_detail_t* slot_detail,
                             bool mmimo_enabled,
                             int32_t cell_index,
                             uint16_t ul_bandwidth);
} // namespace scf_5g_fapi

namespace scf_5g_fapi::pucch
{

/**
 * Populate PUCCH/cuPHY slot-command params for one FAPI PUCCH PDU.
 *
 * This is the parser-owned equivalent of the legacy PUCCH update wrapper. It
 * deliberately does not append Order metadata; channel-task parsers route that
 * through order_sym_prb_info() so PRACH/PUCCH/SRS share the same scratch/merge
 * model.
 */
[[nodiscard]] inline cuphyPucchUciPrm_t*
populate_slot_command(slot_command_api::cell_group_command& cell_grp_cmd,
                      slot_command_api::cell_sub_command& cell_sub_cmd,
                      slot_command_api::slot_indication& slotinfo,
                      const scf_fapi_pucch_pdu_t& pdu,
                      int32_t cell_index,
                      const nv::pucch_dtx_t_list& dtx_thresholds,
                      uint16_t cell_stat_prm_idx,
                      int32_t static_pucch_slot_num,
                      uint16_t ul_bandwidth,
                      uint16_t pucch_hopping_id) noexcept
{
    static_cast<void>(ul_bandwidth);
    cell_sub_cmd.slot.type = SLOT_UPLINK;
    cell_grp_cmd.slot.type = SLOT_UPLINK;
    cell_sub_cmd.slot.slot_3gpp = slotinfo;
    cell_grp_cmd.slot.slot_3gpp = slotinfo;

    auto* const pucch_grp_params = cell_grp_cmd.get_pucch_params();
    if (pucch_grp_params == nullptr)
    {
        return nullptr;
    }

    bool new_cell = false;
    auto it = std::find(pucch_grp_params->cell_index_list.begin(),
                        pucch_grp_params->cell_index_list.end(),
                        cell_index);
    if (it == pucch_grp_params->cell_index_list.end())
    {
        pucch_grp_params->cell_index_list.push_back(cell_index);
        pucch_grp_params->phy_cell_index_list.push_back(cell_sub_cmd.cell);
        new_cell = true;
    }

    pucch_grp_params->scf_ul_tti_handle_list[pdu.format_type].push_back(pdu.handle);

    auto& params = pucch_grp_params->params[pdu.format_type];
    auto& grp = pucch_grp_params->grp_dyn_pars;

    uint16_t index = UINT16_MAX;
    switch (pdu.format_type)
    {
        case UL_TTI_PUCCH_FORMAT_0:
            index = grp.nF0Ucis;
            break;
        case UL_TTI_PUCCH_FORMAT_1:
            index = grp.nF1Ucis;
            break;
        case UL_TTI_PUCCH_FORMAT_2:
            index = grp.nF2Ucis;
            break;
        case UL_TTI_PUCCH_FORMAT_3:
            index = grp.nF3Ucis;
            break;
        case UL_TTI_PUCCH_FORMAT_4:
            index = grp.nF4Ucis;
            break;
        default:
            return nullptr;
    }

    auto& uci_info = params[index];
    uci_info.uciOutputIdx = index;
    uci_info.formatType = pdu.format_type;
    uci_info.rnti = pdu.rnti;
    uci_info.multiSlotTxIndicator = pdu.multi_slot_tx_indicator;
    uci_info.pi2Bpsk = pdu.pi_2_bpsk;
    uci_info.bwpStart = pdu.bwp.bwp_start;
    uci_info.startPrb = pdu.prb_start;
    uci_info.prbSize = pdu.prb_size;
    uci_info.startSym = pdu.start_symbol_index;
    uci_info.nSym = pdu.num_of_symbols;
    uci_info.freqHopFlag = pdu.freq_hop_flag;
    uci_info.secondHopPrb = pdu.second_hop_prb;
    uci_info.groupHopFlag = pdu.group_hop_flag;
    uci_info.sequenceHopFlag = pdu.seq_hop_flag;
    uci_info.initialCyclicShift = pdu.initial_cyclic_shift;
    uci_info.timeDomainOccIdx = pdu.time_domain_occ_idx;
    uci_info.srFlag = pdu.format_type > 1 ? 0 : pdu.sr_flag;
    uci_info.bitLenHarq = pdu.bit_len_harq;
    uci_info.bitLenCsiPart1 = pdu.bit_len_csi_part_1;
    uci_info.AddDmrsFlag = pdu.add_dmrs_flag;
    uci_info.dataScramblingId = pdu.data_scrambling_id;
    uci_info.DmrsScramblingId = pdu.dmrs_scrambling_id;
    uci_info.uciP1P2Crpd_t.numPart2s = 0;
    uci_info.rankBitOffset = 0;
    uci_info.nRanksBits = 0;
    uci_info.DTXthreshold = dtx_thresholds[uci_info.formatType];
    uci_info.bitLenSr = pdu.sr_flag;

    if (new_cell)
    {
        auto& dyn = pucch_grp_params->dyn_pars[grp.nCells];
        dyn.cellPrmStatIdx = cell_stat_prm_idx;
        dyn.cellPrmDynIdx = grp.nCells;
        dyn.slotNum = (static_pucch_slot_num != -1) ? static_pucch_slot_num : slotinfo.slot_;
        dyn.pucchHoppingId = pucch_hopping_id;
        grp.nCells++;
    }

    uci_info.cellPrmDynIdx = grp.nCells - 1;
    uci_info.cellPrmStatIdx = cell_stat_prm_idx;

    switch (pdu.format_type)
    {
        case UL_TTI_PUCCH_FORMAT_0:
            grp.pF0UciPrms = params.data();
            grp.nF0Ucis++;
            break;
        case UL_TTI_PUCCH_FORMAT_1:
            grp.pF1UciPrms = params.data();
            grp.nF1Ucis++;
            break;
        case UL_TTI_PUCCH_FORMAT_2:
            grp.pF2UciPrms = params.data();
            grp.nF2Ucis++;
            break;
        case UL_TTI_PUCCH_FORMAT_3:
            grp.pF3UciPrms = params.data();
            grp.nF3Ucis++;
            break;
        case UL_TTI_PUCCH_FORMAT_4:
            grp.pF4UciPrms = params.data();
            grp.nF4Ucis++;
            break;
        default:
            break;
    }

    return &uci_info;
}

} // namespace scf_5g_fapi::pucch

namespace scf_5g_fapi::detail
{

// scf_fapi_pucch_pdu_t ends with flexible payload[0] beamforming data. Keep
// parser/helper calls by reference so the trailing bytes remain available.
static_assert(sizeof(scf_fapi_pucch_pdu_t) == offsetof(scf_fapi_pucch_pdu_t, payload),
              "scf_fapi_pucch_pdu_t has trailing payload; do not copy by value");

/**
 * @brief Shared PUCCH PDU dispatch + slot-command population for F0/1 and F2/3/4.
 *
 * Centralizes the carrier_id lookup, L1-limit pre-check, and
 * @c update_cell_command invocation used by both @c PucchPduParser and
 * @c Pucch234PduParser.  Logs and drops the PDU on negative carrier id,
 * L1-limit overflow, or @c update_cell_command exception.
 *
 * @tparam V                  Type satisfying @c UlModuleView.
 * @param[in,out] view        Module view providing per-cell accessors and
 *                            the shared slot-command buffer.
 * @param[in]     local_cell_idx  Zero-based local cell index in the
 *                            slot's cell array (FAPI @c msg.cell_id).
 * @param[in]     sfn         System frame number from the parent UL_TTI.
 * @param[in]     slot        Slot number from the parent UL_TTI.
 * @param[in]     pdu         PUCCH PDU body.  Must remain alive for the
 *                            duration of the call.
 * @param[in]     parser_name Caller name used as a log prefix
 *                            (@c "PucchPduParser" / @c "Pucch234PduParser").
 * @return  @c true if the PDU was processed cleanly OR dropped per L1 limit
 *          (a counted-out drop is not a slot-fatal error); @c false on
 *          carrier_id < 0, null group command, or helper exception.
 *          Return value must be checked.
 */
template<UlModuleView V>
[[nodiscard]] bool parse_pucch_pdu_common(V& view,
                                          uint32_t local_cell_idx,
                                          uint16_t sfn,
                                          uint16_t slot,
                                          const scf_fapi_pucch_pdu_t& pdu,
                                          const char* parser_name) noexcept
{
    slot_command_api::slot_indication slot_ind{sfn, slot, 0u};
    const int32_t carrier_id = view.carrier_id(local_cell_idx);
    if (carrier_id < 0)
    {
        NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                   "{}: carrier_id<0 for local_cell_idx={}; dropping PDU",
                   parser_name, local_cell_idx);
        return false;
    }

    // NOTE(GT-12612-MR2): slot_ind tick_=0 is a placeholder; MR2's ULSlotProcessor
    // will plumb the real slot tick through here once the channel-task path owns
    // tick assignment. Legacy update_cell_command is tick-agnostic today.
    try
    {
#ifdef ENABLE_L2_SLT_RSP
        // Wrapped inside try: get_group_limit_errors() and validate_pucch_pdu_l1_limits()
        // are trivially non-throwing today but are not marked noexcept; keeping them
        // inside the catch keeps parse_pucch_pdu_common's own noexcept contract sound
        // without forcing legacy-side noexcept annotations.
        auto& group_limit_errors = view.get_group_limit_errors();
        if (validate_pucch_pdu_l1_limits(pdu, group_limit_errors.pucch_errors) != VALID_FAPI_PDU)
        {
            NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                       "{}: PUCCH L1 limit exceeded sfn={} slot={} carrier_id={} format={}",
                       parser_name, sfn, slot, carrier_id, static_cast<uint16_t>(pdu.format_type));
            return true;
        }
#endif
        auto& cell_cmd = view.cell_sub_command(local_cell_idx);
        auto* group_cmd = view.group_command();
        if (group_cmd == nullptr)
        {
            NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                       "{}: group_command is null sfn={} slot={} carrier_id={}",
                       parser_name, sfn, slot, carrier_id);
            return false;
        }

        const auto cell_view = view.cell_view(local_cell_idx, slot_ind);
        cell_cmd.cell = view.phy_cell_id(local_cell_idx);
        auto* const uci_info = ::scf_5g_fapi::pucch::populate_slot_command(*group_cmd,
                                                                          cell_cmd,
                                                                          slot_ind,
                                                                          pdu,
                                                                          carrier_id,
                                                                          view.dtx_thresholds(),
                                                                          view.cell_stat_prm_idx(local_cell_idx),
                                                                          view.config_options().staticPucchSlotNum,
                                                                          cell_view.cell_params().nPrbUlBwp,
                                                                          pdu.hopping_id);
        auto* const sym_prbs = view.order_sym_prb_info(local_cell_idx);
        if (uci_info == nullptr || sym_prbs == nullptr)
        {
            NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                       "{}: PUCCH Order metadata destination unavailable sfn={} slot={} carrier_id={}",
                       parser_name, sfn, slot, carrier_id);
            return false;
        }
        ::scf_5g_fapi::append_pucch_order_prbs(*uci_info,
                                               pdu.prb_size,
                                               *reinterpret_cast<const scf_fapi_rx_beamforming_t*>(&pdu.payload[0]),
                                               *sym_prbs,
                                               view.config_options().bf_enabled,
                                               view.ru_type_for_cell(static_cast<uint32_t>(carrier_id)),
                                               cell_view.slot_detail(),
                                               view.mmimo_enabled(),
                                               carrier_id,
                                               cell_view.cell_params().nPrbUlBwp);
    }
    catch (const std::exception& e)
    {
        NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                   "{}: exception while parsing PUCCH sfn={} slot={} carrier_id={}: {}",
                   parser_name, sfn, slot, carrier_id, e.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                   "{}: unknown exception while parsing PUCCH sfn={} slot={} carrier_id={}",
                   parser_name, sfn, slot, carrier_id);
        return false;
    }

    return true;
}

} // namespace scf_5g_fapi::detail

#endif // SCF_5G_FAPI_PUCCH_PDU_PARSER_COMMON_HPP_INCLUDED_
