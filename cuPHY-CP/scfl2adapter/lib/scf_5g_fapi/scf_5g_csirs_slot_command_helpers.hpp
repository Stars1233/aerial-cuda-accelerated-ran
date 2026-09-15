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

#ifndef SCF_5G_CSIRS_SLOT_COMMAND_HELPERS_HPP
#define SCF_5G_CSIRS_SLOT_COMMAND_HELPERS_HPP

#include "scf_5g_fapi.h"
#include "scf_5g_slot_commands_common.hpp"
#include "cuphy_api.h"
#include "nv_phy_config_option.hpp"
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi {

/**
 * Parallel DL aggregation path (EOM / cuphyl2adapter). Legacy SCF L2 sequential
 * processing is unchanged in scf_5g_slot_commands_pdsch_csirs.cpp — do not merge
 * implementations until the aggregation path is validated.
 *
 * Consumer trace (repo-wide, for deciding whether to memcpy FAPI pc_and_bf into
 * csirs_params->pcAndBf): those fields are only populated in
 * scf_5g_slot_commands_pdsch_csirs.cpp and consumed there (beam-list / FH helpers).
 * No other translation unit reads csirs_params::pcAndBf / numPcBf. cuPHY CSI-RS
 * execution uses cuphyCsirsRrcDynPrm_t / pCsiRsPrms and PM indices from
 * update_new_csirs_pm — not the raw FAPI pc_and_bf memcpy into csirs_params (that
 * copy exists only in scf_5g_slot_commands_pdsch_csirs.cpp for beam-list / FH).
 */

/**
 * @brief Copy FAPI CSI-RS PDU shape into a @c cuphyCsirsRrcDynPrm_t.
 *
 * Maps the resource-grid layout fields (startRb, nRb, freqDomain, row, symL0/L1,
 * freqDensity, scrambId, csiType, cdmType, beta) and the slot-in-frame index
 * from the FAPI message into the cuPHY RRC-dynamic parameter struct.
 * @param[in]  msg                     FAPI CSI-RS PDU.
 * @param[out] dst                     Destination cuPHY RRC-dynamic parameter struct.
 * @param[in]  slotinfo                Current slot indication; used as the default
 *                                     slot-in-frame source.
 * @param[in]  static_csi_rs_slot_num  Override for @c idxSlotInFrame when @c >=0;
 *                                     pass @c -1 to use @c slotinfo.slot_.
 *                                     Used by static-slot test paths.
 */
void fill_csirs_rrc_dyn_from_fapi(const scf_fapi_csi_rsi_pdu_t& msg,
                                  cuphyCsirsRrcDynPrm_t&       dst,
                                  const slot_command_api::slot_indication& slotinfo,
                                  int                        static_csi_rs_slot_num);

/**
 * @brief Whether a CSI-RS PDU should be skipped when no PDSCH is co-scheduled in the slot.
 *
 * Encodes the policy that ZP-CSI-RS PDUs are only meaningful as PDSCH rate-matching
 * markers; without any PDSCH PDU in the same slot they have no consumer in cuPHY.
 * NZP-CSI-RS is always processed regardless of PDSCH presence.
 * @param[in] csi_type        CSI-RS type from the FAPI PDU.
 * @param[in] has_pdsch_pdus  @c true if the same DL_TTI.req carries at least one PDSCH PDU.
 * @return @c true iff @p csi_type is @c ZP_CSI_RS and @p has_pdsch_pdus is @c false.
 */
[[nodiscard]] bool should_skip_zp_csirs_without_pdsch(cuphyCsiType_t csi_type, bool has_pdsch_pdus);

/**
 * Internal CSI-RS FAPI → slot_command update (parallel DL aggregation implementation).
 * PHY_module calls apply_csirs_dl_aggr_slot_command_for_pdu once per CSI-RS PDU.
 * Does not populate csirs_params->pcAndBf (legacy beam-list / FH only; see file comment).
 * @p has_pdsch_pdus replaces the legacy pdsch_exist flag; use DL_TTI
 * nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] > 0 when SCF_FAPI_10_04 is enabled.
 *
 * @param[in,out] cell_grp_cmd          Cell-group slot command being built for the slot.
 * @param[in,out] cell_cmd              Per-cell sub-command for @p cell_index.
 * @param[in]     msg                   FAPI CSI-RS PDU to apply.
 * @param[in]     slotinfo              Current slot indication (SFN/slot/numerology).
 * @param[in]     cell_index            Carrier index within the cell group (0-based).
 * @param[in,out] config_option         Runtime config flags.
 * @param[in,out] pm_map                PM-weight cache shared across the slot.
 * @param[in]     csirs_offset          Per-cell PDSCH-mirror offset.
 * @param[in]     has_pdsch_pdus        @c true if the same DL_TTI.req carries at least
 *                                      one PDSCH PDU.
 * @param[in]     cell_stat_prm_idx     Static cell-parameter index for @p cell_index.
 * @param[in]     mmimo_enabled         L1 mMIMO capability (for example,
 *                                      PHYDriverProxy::l1_mMIMO_enable_info) so
 *                                      check_bf_pc_params uses the same limits as
 *                                      legacy update_cell_command.
 * @param[in]     pdsch_dyn_idx         Index into @c pdsch_params->cell_dyn_info[] for
 *                                      @p cell_index.
 */
void apply_csirs_fapi_pdu_to_slot_command(slot_command_api::cell_group_command& cell_grp_cmd,
                                          slot_command_api::cell_sub_command&   cell_cmd,
                                          const scf_fapi_csi_rsi_pdu_t&         msg,
                                          const slot_command_api::slot_indication& slotinfo,
                                          int32_t                               cell_index,
                                          nv::phy_config_option&                config_option,
                                          pm_weight_map_t&                      pm_map,
                                          uint32_t                              csirs_offset,
                                          bool                                  has_pdsch_pdus,
                                          uint16_t                              cell_stat_prm_idx,
                                          bool                                  mmimo_enabled,
                                          uint32_t                              pdsch_dyn_idx);

/**
 * @brief DL EOM aggregation: CSI-RS-specific slot_command updates for one PDU.
 *
 * Captures the PDSCH-mirror @c csirs_offset on the first CSI-RS PDU of a cell
 * (read from @c pdsch_params->cell_grp_info.nCsiRsPrms when PDSCH is co-scheduled,
 * else 0) and forwards each PDU to @c apply_csirs_fapi_pdu_to_slot_command. Keeps
 * slot_command layout details out of @c PHY_module.
 *
 * @todo (post-merge follow-up): the @c bool @c mmimo_enabled flag below flows
 *       unchanged through @c apply_csirs_fapi_pdu_to_slot_command and
 *       @c check_bf_pc_params. Project rule §4.3 prefers a typed enum
 *       (e.g. @c MmimoMode::{Disabled,Enabled}) so reader at every call site
 *       knows what @c true / @c false mean without context. Deferred from
 *       MR !5076 because the same pattern is used across SSB/PDCCH/PUSCH
 *       channels and the legacy code; a one-shot conversion ticket is the
 *       right scope. Same rationale applies to the @c bool @c mmimo_enabled
 *       parameter on @c apply_csirs_fapi_pdu_to_slot_command above.
 *
 * @par Caller contract — per-cell contiguous processing
 * The caller MUST drain ALL CSI-RS PDUs of one cell (i.e. one DL_TTI.req) before invoking
 * this function for the next cell within the same @p cell_grp_cmd. The implementation
 * relies on this to compute @c cell_info_idx as @c csirs_params->nCells-1 in the
 * "cell already registered" branch (i.e. second+ PDU of the cell currently being drained).
 * Interleaving cells across PDUs (e.g. cell0_pdu0, cell1_pdu0, cell0_pdu1) would write
 * @c nRrcParams to the wrong @c cellInfo[] entry. The current production caller
 * @c PHY_module::process_aggr_csirs_channel honours this contract via its outer
 * for_each_tti_msg / inner for_each_pdu nesting.
 *
 * @param[in,out] cell_grp_cmd            Cell-group slot command being built for the slot.
 * @param[in,out] cell_cmd                Per-cell sub-command for @p cell_index.
 * @param[in]     csi_pdu                 FAPI CSI-RS PDU to apply.
 * @param[in]     slotinfo                Current slot indication (SFN/slot/numerology).
 * @param[in]     cell_index              Carrier index within the cell group (0-based).
 * @param[in]     has_pdsch_pdus          @c true if the same DL_TTI.req carries at least
 *                                        one PDSCH PDU (drives the PDSCH-mirror path).
 * @param[in,out] captured_csirs_offset   Per-cell capture flag. Caller initialises to
 *                                        @c false before processing the first PDU of a
 *                                        cell; first call sets it to @c true and snapshots
 *                                        @p csirs_offset, subsequent calls reuse the snapshot.
 * @param[in,out] csirs_offset            Per-cell PDSCH-mirror offset. Initial value is ignored
 *                                        on the first PDU; set by the first call to the
 *                                        current @c pdsch_params->cell_grp_info.nCsiRsPrms
 *                                        (or 0 when @p has_pdsch_pdus is false), then read on
 *                                        subsequent PDUs to keep all mirrored entries
 *                                        contiguous.
 * @param[in,out] config_option           Runtime config flags (precoding_enabled, static slot
 *                                        override).
 * @param[in,out] pm_map                  PM-weight cache shared across the slot.
 * @param[in]     mmimo_enabled           L1 mMIMO flag (same source as
 *                                        @c apply_csirs_fapi_pdu_to_slot_command).
 * @param[in]     pdsch_dyn_idx           Index into @c pdsch_params->cell_dyn_info[] for
 *                                        @p cell_index (precomputed by the caller from the
 *                                        slot's PDSCH ordinal).
 */
void apply_csirs_dl_aggr_slot_command_for_pdu(slot_command_api::cell_group_command& cell_grp_cmd,
                                              slot_command_api::cell_sub_command&   cell_cmd,
                                              const scf_fapi_csi_rsi_pdu_t&         csi_pdu,
                                              const slot_command_api::slot_indication& slotinfo,
                                              int32_t                               cell_index,
                                              bool                                  has_pdsch_pdus,
                                              bool&                                 captured_csirs_offset,
                                              uint32_t&                             csirs_offset,
                                              nv::phy_config_option&                config_option,
                                              pm_weight_map_t&                      pm_map,
                                              bool                                  mmimo_enabled,
                                              uint32_t                              pdsch_dyn_idx);

/**
 * @brief Populate the per-cell FH callback params for a CSI-RS-carrying cell.
 *
 * Fills @c fh_params.csirs_fh_params[carrier_id] with the metadata the FH
 * prepare callback needs to drive O-RAN CSI-RS sample generation, marks
 * @c fh_params.is_csirs_cell[carrier_id] = 1, and increments
 * @c fh_params.num_csirs_cell.
 *
 * @note TEMPORARY: lives on the parallel DL aggregation path; will be removed
 *       once the FH prepare path consumes CSI-RS info from @c csirs_params
 *       directly. Tracked via the @c TEMPORARY(csirs-fh-aggr-path) markers
 *       in this header and the implementation.
 *
 * @param[in,out] fh_params              Cell-group FH-prepare callback params being built.
 * @param[in,out] cell_cmd               Per-cell sub-command (back-pointer cached on the entry).
 * @param[in]     carrier_id             Carrier index within the cell group; used as the
 *                                       index into @c csirs_fh_params[] and @c is_csirs_cell[].
 * @param[in]     has_pdsch              @c true if the same DL_TTI.req carries at least one
 *                                       PDSCH PDU. Drives @c cuphy_params_cell_idx selection.
 * @param[in]     cuphy_params_cell_idx  Index into the PDSCH dynamic-info array for this cell;
 *                                       written verbatim when @p has_pdsch is @c true, else -1.
 * @param[in]     bf_enabled             Whether DL beamforming is enabled (from the PHY
 *                                       module @c bf_enabled() flag).
 * @param[in]     mmimo_enabled          L1 mMIMO flag.
 * @param[in]     num_dl_prb             Cell's DL BWP size in PRBs (@c nPrbDlBwp).
 */
// TEMPORARY(csirs-fh-aggr-path): begin -- populate FH params from aggregation worker
void populate_csirs_fh_params_for_cell(
    slot_command_api::fh_prepare_callback_params& fh_params,
    slot_command_api::cell_sub_command&            cell_cmd,
    int32_t                                       carrier_id,
    bool                                          has_pdsch,
    int32_t                                       cuphy_params_cell_idx,
    bool                                          bf_enabled,
    bool                                          mmimo_enabled,
    uint16_t                                      num_dl_prb);
// TEMPORARY(csirs-fh-aggr-path): end

} // namespace scf_5g_fapi

#endif // SCF_5G_CSIRS_SLOT_COMMAND_HELPERS_HPP
