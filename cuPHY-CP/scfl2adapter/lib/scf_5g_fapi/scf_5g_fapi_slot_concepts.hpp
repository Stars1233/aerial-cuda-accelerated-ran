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

#if !defined(SCF_5G_FAPI_SLOT_CONCEPTS_HPP_INCLUDED_)
#define SCF_5G_FAPI_SLOT_CONCEPTS_HPP_INCLUDED_

#include <concepts>
#include <cstdint>
#include <utility>

#include "scf_5g_fapi_slot_types.hpp"   // CellView, pm_weight_map_t, module_view_traits
#include "scf_5g_fapi_errors.hpp"       // SlotParseError, SlotParseResult, FapiMessageParseError
#include "scf_5g_fapi_dl_stats.hpp"     // DlPdschStatsBatch
#include "scf_5g_fapi.h"                // scf_fapi_message_id_e, scf_fapi_error_codes_t (used in concept body)
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi
{

// ---------------------------------------------------------------------------
// SRS chest-buffer classification
// ---------------------------------------------------------------------------

// Verdict for a UL SRS PDU's chest-buffer state lookup. Returned by the
// view's classify_srs_chest_buffer() so the parser does not need to
// decode the FAPI handle's chest-buffer index or branch on the raw
// buffer state itself — the view owns that knowledge.
enum class SrsChestBuffVerdict : uint8_t
{
    Accept,            // proceed with parsing
    LookupFailed,      // l1_cv_mem_bank_get_buffer_state returned -1
    AlreadyRequested,  // chest buffer is already in REQUESTED state
};

// ---------------------------------------------------------------------------
// Direction-specific module-view concepts
// ---------------------------------------------------------------------------

/**
 * Subset of ModuleView required by DL slot processing.
 *
 * Derived from auditing update_cell_command call sites in scf_5g_fapi_phy.cpp:
 *   - PDCCH: group_command, cell_sub_command, pm_map, config_options,
 *            staticPdcchSlotNum, mmimo_enabled, get_cell_limit_errors, cell_view
 *   - PDSCH: group_command, cell_sub_command, pm_map, pm_enabled, bf_enabled,
 *            bfw_coeff_mem_info, mmimo_enabled, cell_view
 *   - CSI-RS: group_command, cell_sub_command, pm_map, config_options,
 *             mmimo_enabled, cell_view
 *   - SSB:   group_command, cell_sub_command, pm_map, config_options,
 *            bf_enabled, mmimo_enabled, cell_view
 *
 * PhyModuleView satisfies this concept (static_assert in scf_5g_fapi_message_context.hpp).
 *
 * @tparam V  Type to be checked against the DlModuleView requirements.
 */
template<typename V>
concept DlModuleView =
    requires(const V& v,
             uint32_t cell_idx,
             uint8_t  slot_idx,
             uint16_t cell_id_u16,
             uint32_t cell_id,
             const slot_command_api::slot_indication& slot_ind,
             const DlPdschStatsBatch& dl_pdsch_stats)
    {
        { v.group_command() }                        -> std::same_as<slot_command_api::cell_group_command*>;
        { v.cell_sub_command(cell_idx) }             -> std::same_as<slot_command_api::cell_sub_command&>;
        { v.pm_map() }                               -> std::same_as<pm_weight_map_t&>;
        { v.pm_enabled() }                           -> std::convertible_to<bool>;
        { v.bf_enabled() }                           -> std::convertible_to<bool>;
        { v.bfw_coeff_mem_info(cell_idx, slot_idx) } -> std::same_as<slot_command_api::bfw_coeff_mem_info_t*>;
        { v.mmimo_enabled() }                        -> std::convertible_to<bool>;
        { v.slot_command() }                         -> std::same_as<slot_command_api::slot_command&>;
        // PDCCH/SSB/CSI-RS slot-command paths need config_options.
        { v.config_options() }                       -> std::same_as<nv::phy_config_option&>;
        // PDCCH slot-command path needs static slot number.
        { v.staticPdcchSlotNum() }                   -> std::convertible_to<int>;
        // PDSCH slot-command path needs static slot number (test/debug override; -1 = use msg.slot).
        { v.staticPdschSlotNum() }                   -> std::convertible_to<int>;
        // Per-cell static parameter index assigned during cell creation.
        { v.cell_stat_prm_idx(cell_idx) }            -> std::convertible_to<uint16_t>;
        // Per-cell logical carrier ID (populates pdsch_params::cell_index_list).
        { v.carrier_id(cell_idx) }                   -> std::convertible_to<int32_t>;
        // Per-cell physical cell ID (populates pdsch_params::phy_cell_index_list).
        { v.phy_cell_id(cell_idx) }                  -> std::convertible_to<uint16_t>;
        // SSB/PBCH needs full cell PHY config for SSB case, Lmax, NID, nF, and
        // numerology-derived slot metadata.
        { v.phy_config(cell_id) }                    -> std::same_as<const nv::phy_config&>;
        // PDCCH limit error reporting (DCI count overflow, etc.).
        { v.get_cell_limit_errors(cell_id_u16) }     -> std::same_as<nv::slot_limit_cell_error_t&>;
        // Per-cell DL BWP PRB count (populates pdsch_fh_prepare_params::num_dl_prb).
        { v.num_dl_prb(cell_idx) }                   -> std::convertible_to<uint16_t>;
        // Runtime FH gate: when true, C-plane is created by other worker threads
        // (l1_setup_early_cplane_slot_maps) and the parser must NOT populate fh_params.
        // YAML key: fapi_to_cplane_direct (cuphydriver).
        { v.is_fapi_to_cplane_direct_enabled() }     -> std::convertible_to<bool>;
        // PDSCH DL throughput/slot stats publish: batched by DLSlotProcessor,
        // written to the PHY counters at message-batch boundary.
        { v.publish_dl_pdsch_stats(dl_pdsch_stats) }  -> std::same_as<void>;
        requires CellView<decltype(v.cell_view(cell_id, slot_ind))>;
    };

/**
 * Subset of ModuleView required by UL slot processing.
 *
 * Derived from auditing update_cell_command call sites in scf_5g_fapi_phy.cpp:
 *   - PUSCH:  group_command, cell_sub_command, bf_enabled, mmimo_enabled,
 *             cell_view, cell_stat_prm_idx, carrier_id, ru
 *             (PUSCH-only accessors — staticPuschSlotNum, lbrm,
 *              dtx_thresholds_pusch, bfw_coeff_mem_info — live in the
 *              PuschModuleView refinement, not the base concept)
 *   - PUCCH:  group_command, cell_sub_command, dtx_thresholds, config_options,
 *             mmimo_enabled, cell_view
 *   - PRACH:  group_command, cell_sub_command, bf_enabled, mmimo_enabled, cell_view
 *   - SRS:    group_command, cell_sub_command, bf_enabled, mmimo_enabled,
 *             transport, cell_view
 *
 * PhyModuleView satisfies this concept (static_assert in scf_5g_fapi_message_context.hpp).
 *
 * @tparam V  Type to be checked against the UlModuleView requirements.
 */
template<typename V>
concept UlModuleView =
    requires(const V& v,
             uint32_t cell_idx,
             uint32_t cell_id,
             int      cell_id_int,
             const slot_command_api::slot_indication& slot_ind)
    {
        { v.group_command() }            -> std::same_as<slot_command_api::cell_group_command*>;
        { v.cell_sub_command(cell_idx) } -> std::same_as<slot_command_api::cell_sub_command&>;
        { v.order_sym_prb_info(cell_idx) } -> std::same_as<slot_command_api::slot_info_t*>;
        { v.slot_command() }             -> std::same_as<slot_command_api::slot_command&>;
        { v.mmimo_enabled() }            -> std::convertible_to<bool>;
        { v.bf_enabled() }               -> std::convertible_to<bool>;
        { v.ru_type_for_cell(cell_id) }  -> std::same_as<ru_type>;
        // PUCCH slot-command path needs carrier id, cell stat index, and group L1-limit accounting.
        { v.carrier_id(cell_idx) }       -> std::convertible_to<int32_t>;
        { v.cell_stat_prm_idx(cell_idx) } -> std::convertible_to<uint16_t>;
        { v.get_group_limit_errors() }   -> std::same_as<nv::slot_limit_group_error_t&>;
        // PUCCH slot-command path needs config_options.
        { v.config_options() }           -> std::same_as<nv::phy_config_option&>;
        // PUCCH DTX thresholds (format 0/1 and 2/3/4).
        { v.dtx_thresholds() }           -> std::same_as<const nv::pucch_dtx_t_list&>;
        // SRS needs transport for IPC response path.
        { v.transport(cell_id_int) }     -> std::same_as<nv::phy_mac_transport&>;
        { v.phy_cell_id(cell_idx) }      -> std::convertible_to<uint16_t>;
        { v.get_cell_limit_errors(static_cast<uint16_t>(cell_idx)) } -> std::same_as<nv::slot_limit_cell_error_t&>;
        { v.indication_instances_per_slot(cell_idx) } -> std::convertible_to<uint8_t>;
        { v.send_fapi_error_indication(cell_idx,
                                       std::declval<scf_fapi_message_id_e>(),
                                       std::declval<scf_fapi_error_codes_t>(),
                                       static_cast<uint16_t>(0),
                                       static_cast<uint16_t>(0)) } -> std::same_as<void>;
        { v.srs_enabled() } -> std::convertible_to<bool>;
        { v.ru(cell_idx) } -> std::same_as<ru_type>;
        { v.classify_srs_chest_buffer(cell_idx, std::declval<const scf_fapi_srs_pdu_t&>()) } -> std::same_as<SrsChestBuffVerdict>;
        { v.fapi_to_cplane_direct_enabled() } -> std::convertible_to<bool>;
        requires CellView<decltype(v.cell_view(cell_id, slot_ind))>;
    };

/**
 * Refinement of UlModuleView for the PUSCH parser (cubb-review §12.4, ISP).
 *
 * Mirrors the PrachModuleView pattern: names the PUSCH-specific slice of the
 * module-view surface that PuschPduParser needs, so the parser documents its
 * own contract instead of templating on the broad universal concept.
 *
 * Per cubb-review §12.4 (ISP), PUSCH-only state must not leak into the
 * universal UlModuleView concept used by PUCCH/SRS/PRACH sibling parsers.
 * These accessors are required by no other UL parser, so they live here in the
 * refinement rather than the base concept (PUCCH/SRS/PRACH mocks no longer need
 * to stub them):
 *
 *   - staticPuschSlotNum()      — static PUSCH slot override (-1 = use msg.slot)
 *   - lbrm()                    — limited-buffer rate-matching flag
 *   - dtx_thresholds_pusch()    — PUSCH DTX threshold
 *   - enable_weighted_avg_cfo() — PUSCH weighted-average CFO / LDPC extension gate
 *   - bfw_coeff_mem_info(idx, slot) — per-cell BFW coefficient memory for the
 *                                     PUSCH beamforming path
 *
 * cell_stat_prm_idx(), carrier_id(), and ru() remain on the base UlModuleView
 * (other UL parsers need them); they are restated below so PuschPduParser
 * documents its full contract in one place:
 *
 *   - cell_stat_prm_idx(idx) — per-cell static parameter index
 *   - carrier_id(idx)        — per-cell logical carrier id
 *   - ru(idx)                — per-cell RU type (beamforming path)
 *
 * @c PhyModuleView satisfies this concept (static_assert in
 * @c scf_5g_fapi_message_context.hpp).
 *
 * @tparam V  Type to be checked against the PuschModuleView requirements.
 */
template<typename V>
concept PuschModuleView =
    UlModuleView<V> &&
    requires(const V& v, uint32_t cell_idx, uint8_t slot_idx)
    {
        { v.staticPuschSlotNum() }        -> std::convertible_to<int>;
        { v.lbrm() }                      -> std::convertible_to<uint8_t>;
        { v.dtx_thresholds_pusch() }      -> std::same_as<const float&>;
        { v.cell_stat_prm_idx(cell_idx) } -> std::convertible_to<uint16_t>;
        { v.carrier_id(cell_idx) }        -> std::convertible_to<int32_t>;
        { v.ru(cell_idx) }                -> std::same_as<ru_type>;
        { v.enable_weighted_avg_cfo() }   -> std::convertible_to<bool>;
        { v.bfw_coeff_mem_info(cell_idx, slot_idx) }
                                          -> std::same_as<slot_command_api::bfw_coeff_mem_info_t*>;
    };

/**
 * Refinement of UlModuleView for the new PRACH parser (GT-11843, MR 2b).
 *
 * Per cubb-review §12.4 (ISP), PRACH-only state must not leak into the
 * universal UlModuleView concept used by PUCCH/SRS sibling parsers.
 * PrachModuleView adds the five PRACH-only accessors the new parser needs:
 *
 *   - phy_config(cell_id)         — for prach_config_.{start_ro_index,
 *                                   prach_scs, prach_seq_length, k1,
 *                                   root_sequence}
 *   - prach_addln_config(cell_id) — for n_ra_dur / n_ra_slot / n_ra_rb
 *   - get_cell_limit_errors(pci)  — for L1-limit validation (already on
 *                                   the DL concept; mirrored here)
 *   - is_fapi_to_cplane_direct()  — gates the sym_prb_info FH-metadata fill
 *                                   (skipped when downstream FH callback is
 *                                   bypassed by SKIP_DL_FHCB)
 *
 * @c PhyModuleView satisfies this concept (static_assert in
 * @c scf_5g_fapi_message_context.hpp). Sibling parsers continue to template
 * on the bare UlModuleView and are unaffected by this addition.
 *
 * @tparam V  Type to be checked against the PrachModuleView requirements.
 */
template<typename V>
concept PrachModuleView =
    UlModuleView<V> &&
    requires(const V& v,
             uint32_t cell_id,
             uint16_t phys_cell_id)
    {
        { v.phy_config(cell_id) }         -> std::same_as<const nv::phy_config&>;
        { v.prach_addln_config(cell_id) } -> std::same_as<const nv::prach_addln_config_t&>;
        { v.get_cell_limit_errors(phys_cell_id) }
                                          -> std::same_as<nv::slot_limit_cell_error_t&>;
        { v.is_fapi_to_cplane_direct() }  -> std::convertible_to<bool>;
    };

// ---------------------------------------------------------------------------
// Per-PDU-type parser concept
// ---------------------------------------------------------------------------

/**
 * Concept satisfied by each per-PDU-type final parser.
 *
 * Each parser must:
 *   - expose a static constexpr pdu_type of type Enum
 *   - expose a pdu_t alias for the concrete PDU struct
 *   - provide bool parse(uint16_t sfn, uint16_t slot, const pdu_t&) noexcept
 *
 * @tparam P     Parser type to check.
 * @tparam Enum  PDU-type enum (e.g. scf_fapi_dl_tti_pdu_type_t).
 */
template<typename P, typename Enum>
concept PduParser =
    requires { { P::pdu_type } -> std::same_as<const Enum&>; } &&
    requires(P& p, uint16_t sfn, uint16_t slot, const typename P::pdu_t& pdu)
    {
        { p.parse(sfn, slot, pdu) } -> std::same_as<bool>;
    };

// SlotParseError, SlotParseResult, FapiMessageParseError, MessageBatchParseResult
// are defined in scf_5g_fapi_errors.hpp (included above via scf_5g_fapi_errors.hpp).

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_SLOT_CONCEPTS_HPP_INCLUDED_
