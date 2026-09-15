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

#if !defined(SCF_5G_FAPI_MESSAGE_CONTEXT_HPP_INCLUDED_)
#define SCF_5G_FAPI_MESSAGE_CONTEXT_HPP_INCLUDED_

#include <bit>
#include <cstdint>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_dl_stats.hpp"      // DlPdschStatsBatch
#include "scf_5g_fapi_slot_concepts.hpp" // CellView, SrsChestBuffVerdict, module_view_traits (transitively pulls slot_types)
#include "scf_5g_fapi_slot_types.hpp"    // CellView, pm_weight_map_t, module_view_traits (lightweight)
#include "scf_5g_fapi_phy.hpp"           // scf_5g_fapi::phy (pulls in nv_phy_instance.hpp)
#include "nv_phy_module.hpp"             // nv::PHY_module

namespace scf_5g_fapi
{

class PhyModuleView;
class PhyCellView;

// Specializations of the primary templates defined in scf_5g_fapi_slot_types.hpp.
template <>
struct module_view_traits<PhyModuleView> {
    using cell_view_type = PhyCellView;
};

template <>
struct cell_view_traits<PhyCellView> {
    using module_view_type = PhyModuleView;
};

/**
 * Per-cell PHY carrier view: DL PRB count and TDD slot detail from a @c phy instance.
 *
 * @c num_dl_prb and @c slot_detail are read from the stored @c phy reference
 * (@c get_phy_cell_params().nPrbDlBwp and @c phy::get_slot_detail). When @c cell_id does not
 * match this @c phy instance's carrier id (@c phy::get_carrier_id), @c num_dl_prb is zero
 * and @c slot_detail is null.
 */
class PhyCellView final {
public:
    /**
     * @param[in] p        PHY instance; must outlive this view.
     * @param[in] cell_id  Logical cell id (must match @c p.get_carrier_id() for non-null data).
     * @param[in] slot_ind Slot indication used to resolve TDD slot detail (copied internally).
     */
    PhyCellView(phy& p, uint32_t cell_id, slot_command_api::slot_indication slot_ind)
        : phy_{&p}
        , cell_id_{cell_id}
        , slot_ind_{slot_ind}
    {}

    /** DL BWP PRB count from @c phy.get_phy_cell_params() when @c cell_id matches the carrier. */
    [[nodiscard]] uint16_t num_dl_prb() const noexcept
    {
        if (cell_id_ != static_cast<uint32_t>(phy_->get_carrier_id())) {
            return 0;
        }
        return phy_->get_phy_cell_params().nPrbDlBwp;
    }

    /** TDD slot detail from @c phy::get_slot_detail; null when RU is not SINGLE_SECT_MODE or cell mismatch. */
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept
    {
        if (cell_id_ != static_cast<uint32_t>(phy_->get_carrier_id())) {
            return nullptr;
        }
        return phy_->get_slot_detail(slot_ind_);
    }

    /**
     * UL start symbol for the single-sector C-plane section: @c slot_detail->start_sym_ul,
     * or 0 when slot detail is unavailable.  Encapsulates the slot_detail field read so the
     * UL parser need not pull in the complete nv::slot_detail_ type (mirrors legacy
     * update_fh_params_pusch start-symbol derivation).
     */
    [[nodiscard]] uint8_t ul_start_symbol() const noexcept
    {
        const auto* sd = slot_detail();
        return (sd == nullptr) ? 0u : static_cast<uint8_t>(sd->start_sym_ul);
    }

    /**
     * UL symbol count for the single-sector C-plane section: @c slot_detail->max_ul_symbols,
     * or @p fallback when slot detail is unavailable or reports 0 (mirrors legacy
     * update_fh_params_pusch numSymbols derivation).
     *
     * @param[in] fallback  Symbol count to use when slot detail is null / reports 0.
     */
    [[nodiscard]] uint8_t ul_max_symbols(const uint8_t fallback) const noexcept
    {
        const auto* sd = slot_detail();
        return (sd == nullptr || sd->max_ul_symbols == 0u) ? fallback : sd->max_ul_symbols;
    }

    /** Live cell stat parameters from @c phy::get_phy_cell_params().
     *  Provides nPrbDlBwp, nPrbUlBwp, and all other per-cell fields
     *  used by slot-command paths (PDCCH, PDSCH, PUSCH, PUCCH, CSI-RS, SRS, SSB). */
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept
    {
        return phy_->get_phy_cell_params();
    }

    void publish_dl_pdsch_stats(uint64_t bytes, uint32_t slots) const noexcept
    {
        phy_->publish_dl_pdsch_stats(cell_id_, bytes, slots);
    }

private:
    gsl_lite::not_null<phy*>            phy_;   //!< Non-owning; must outlive this view.
    uint32_t                            cell_id_{};
    slot_command_api::slot_indication   slot_ind_{};
};

// CellView concept is defined in scf_5g_fapi_slot_types.hpp (included above).

/**
 * Concept for a view over PHY-module-derived PDSCH parameters.
 *
 * Requires slot-command accessors and a per-cell @ref CellView from @c cell_view.
 * Concrete implementation: @ref PhyModuleView.
 */
template <typename T>
concept ModuleView = requires(const T& view,
                              uint32_t   cell_idx,
                              uint8_t    slot_idx,
                              uint32_t   cell_id,
                              uint16_t   cell_id_u16,
                              int        cell_id_int,
                              const slot_command_api::slot_indication& slot_ind) {
    // --- existing slot-command accessors ---
    { view.cell_group() }                           -> std::convertible_to<bool>;
    { view.group_command() }                        -> std::same_as<slot_command_api::cell_group_command*>;
    { view.cell_sub_command(cell_idx) }             -> std::same_as<slot_command_api::cell_sub_command&>;
    { view.pm_map() }                               -> std::same_as<pm_weight_map_t&>;
    { view.pm_enabled() }                           -> std::convertible_to<bool>;
    { view.bf_enabled() }                           -> std::convertible_to<bool>;
    { view.bfw_coeff_mem_info(cell_idx, slot_idx) } -> std::same_as<bfw_coeff_mem_info_t*>;
    { view.mmimo_enabled() }                        -> std::convertible_to<bool>;
    { view.slot_command() }                         -> std::same_as<slot_command_api::slot_command&>;
    requires CellView<typename module_view_traits<T>::cell_view_type>;
    { view.cell_view(cell_id, slot_ind) }           -> std::same_as<typename module_view_traits<T>::cell_view_type>;
    // --- DL channel processing ---
    // config_options() used by PDCCH, SSB, CSI-RS, PUCCH slot-command paths.
    { view.config_options() }                       -> std::same_as<nv::phy_config_option&>;
    // staticPdcchSlotNum() used by PDCCH update_cell_command.
    { view.staticPdcchSlotNum() }                   -> std::convertible_to<int>;
    // staticPdschSlotNum() used by PDSCH update_cell_command (test/debug override).
    { view.staticPdschSlotNum() }                   -> std::convertible_to<int>;
    // cell_stat_prm_idx() returns the per-cell static parameter index assigned at cell creation.
    { view.cell_stat_prm_idx(cell_idx) }            -> std::convertible_to<uint16_t>;
    // carrier_id() returns the logical carrier index (populates pdsch_params::cell_index_list).
    { view.carrier_id(cell_idx) }                   -> std::convertible_to<int32_t>;
    // phy_cell_id() returns the physical cell ID (populates pdsch_params::phy_cell_index_list).
    { view.phy_cell_id(cell_idx) }                  -> std::convertible_to<uint16_t>;
    // get_cell_limit_errors() used by PDCCH to report per-cell L1 limit violations.
    { view.get_cell_limit_errors(cell_id_u16) }     -> std::same_as<nv::slot_limit_cell_error_t&>;
    // --- UL channel processing ---
    // staticPuschSlotNum() used by PUSCH update_cell_command.
    { view.staticPuschSlotNum() }                   -> std::convertible_to<int>;
    // lbrm() used by PUSCH update_cell_command.
    { view.lbrm() }                                 -> std::convertible_to<uint8_t>;
    // dtx_thresholds() used by PUCCH update_cell_command (format 0/1 and 2/3/4).
    { view.dtx_thresholds() }                       -> std::same_as<const nv::pucch_dtx_t_list&>;
    // dtx_thresholds_pusch() used by PUSCH update_cell_command.
    { view.dtx_thresholds_pusch() }                 -> std::same_as<const float&>;
    // get_group_limit_errors() used by PUCCH L1 limit validation.
    { view.get_group_limit_errors() }               -> std::same_as<nv::slot_limit_group_error_t&>;
    // enable_weighted_avg_cfo() gates PUSCH weighted-average CFO / LDPC extension parsing.
    { view.enable_weighted_avg_cfo() }              -> std::convertible_to<bool>;
    // transport() used by SRS update_cell_command for IPC transport access.
    { view.transport(cell_id_int) }                 -> std::same_as<nv::phy_mac_transport&>;
};

/**
 * Non-owning view over a PHY_module plus @c phy for per-cell PRB and slot detail.
 *
 * Forwards slot-command accessors to the held @c PHY_module (including @c slot_command for the
 * active slot index). @c cell_view uses the stored @c phy reference for @c num_dl_prb and @c slot_detail.
 *
 * @note The referenced @c PHY_module and @c phy must outlive this object.
 */
class PhyModuleView final {
public:
    using OrderSymPrbInfoFn = slot_command_api::slot_info_t* (*)(void*, uint32_t) noexcept;

    /**
     * @param[in] mod           PHY module instance (must outlive this object).
     * @param[in] mmimo_enabled Whether MU-MIMO dynamic beamforming is active.
     * @param[in] srs_enabled   Whether SRS processing is enabled (cached at slot boundary;
     *                          do not re-query per PDU).
     * @param[in] slot_cmd_idx  Slot-command ring index captured at EOM for this slot.
     *                          Must match the index used when the slot command was set up;
     *                          do NOT pass the live @c current_slot_cmd_index (already
     *                          advanced to N+1 before worker tasks execute).
     */
    explicit PhyModuleView(nv::PHY_module& mod,
                           bool mmimo_enabled,
                           bool srs_enabled,
                           uint32_t slot_cmd_idx,
                           OrderSymPrbInfoFn order_sym_prb_info_fn = nullptr,
                           void* order_sym_prb_info_ctx = nullptr) noexcept
        : mod_{&mod}
        , mmimo_enabled_{mmimo_enabled}
        , srs_enabled_{srs_enabled}
        , enable_weighted_avg_cfo_{nv::PHYDriverProxy::getInstance().l1_get_enable_weighted_average_cfo() != 0u}
        , slot_cmd_idx_{slot_cmd_idx}
        , order_sym_prb_info_fn_{order_sym_prb_info_fn}
        , order_sym_prb_info_ctx_{order_sym_prb_info_ctx}
    {}

    /** Whether a cell group is active this slot. @return True when a cell group is present. */
    [[nodiscard]] bool cell_group() const noexcept { return mod_->cell_group(); }
    /**
     * Active cell-group command pointer for the captured slot.
     *
     * Indexes @c slot_command_array directly with @c slot_cmd_idx_ rather than
     * delegating to @c PHY_module::group_command() — the live
     * @c current_slot_cmd_index has already been advanced to N+1 before worker tasks run.
     *
     * @return Pointer to the cell-group command. Return value must be checked.
     */
    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &(mod_->slot_command_array.at(slot_cmd_idx_).cell_groups);
    }

    /**
     * Cell sub-command for a given cell index in the captured slot.
     *
     * @param[in] cell_index Logical cell index.
     * @return Reference to the cell sub-command.
     */
    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_index) const
    {
        return mod_->slot_command_array.at(slot_cmd_idx_).cells.at(cell_index);
    }

    /**
     * UL Order metadata destination for channel-task parsers.
     *
     * Normal/legacy users receive the slot command's real @c sym_prb_info. The
     * store-replay channel-task path can provide a per-channel scratch resolver
     * so parallel UL channel tasks do not concurrently mutate the same
     * @c slot_info_t; the scratch is merged into the real slot command once all
     * tasks for the slot have completed.
     */
    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t cell_index) const noexcept
    {
        if (order_sym_prb_info_fn_ != nullptr)
        {
            if (auto* scratch = order_sym_prb_info_fn_(order_sym_prb_info_ctx_, cell_index);
                scratch != nullptr)
            {
                return scratch;
            }
        }
        return mod_->slot_command_array[slot_cmd_idx_].cells[cell_index].sym_prb_info();
    }

    /** Static PM weight lookup map. @return Reference to the shared PM weight map. */
    [[nodiscard]] pm_weight_map_t& pm_map() const noexcept { return nv::PHY_module::pm_map(); }
    /** Whether precoding-matrix processing is active. @return True when PM is enabled. */
    [[nodiscard]] bool pm_enabled() const noexcept { return mod_->pm_enabled(); }
    /** Whether beamforming is active. @return True when BF is enabled. */
    [[nodiscard]] bool bf_enabled() const noexcept { return mod_->bf_enabled(); }

    /**
     * BFW coefficient memory info for a given cell and previous slot.
     *
     * @param[in] cell_index Logical cell index.
     * @param[in] slot_index Previous slot index into the BFW ring buffer.
     * @return Pointer to the BFW coefficient ring-buffer entry.
     */
    [[nodiscard]] bfw_coeff_mem_info_t* bfw_coeff_mem_info(uint32_t cell_index, uint8_t slot_index) const noexcept
    {
        return mod_->get_bfw_coeff_buff_info(cell_index, slot_index);
    }

    /** Whether MU-MIMO dynamic beamforming is active. @return True when mMIMO is active. */
    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }

    /**
     * Active slot command for the captured slot index.
     *
     * Indexes @c slot_command_array directly with @c slot_cmd_idx_ — the live
     * @c current_slot_cmd_index is already N+1 when worker tasks execute.
     *
     * @return Reference to the slot command entry for this slot.
     */
    [[nodiscard]] slot_command_api::slot_command& slot_command() const
    {
        return mod_->slot_command_array.at(slot_cmd_idx_);
    }

    /**
     * Per-cell PHY view for DL PRB count, slot detail, and cell stat params.
     *
     * @param[in] cell_id   Logical cell id (must match @c phy.get_carrier_id() for carrier data).
     * @param[in] slot_ind  Slot indication for resolving TDD slot detail.
     */
    [[nodiscard]] PhyCellView cell_view(uint32_t cell_id, const slot_command_api::slot_indication& slot_ind) const
    {
        auto& phy_inst = mod_->PHY_instances().at(cell_id);
        return PhyCellView{ static_cast<scf_5g_fapi::phy&>(phy_inst.get()), cell_id, slot_ind};
    }

    // --- DL channel processing accessors ---

    /** PHY config options (PDCCH/SSB/CSI-RS/PUCCH slot-command paths). */
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept
    {
        return mod_->config_options();
    }

    /** Static PDCCH slot number used by PDCCH update_cell_command. */
    [[nodiscard]] int staticPdcchSlotNum() const noexcept
    {
        return mod_->staticPdcchSlotNum();
    }

    /** Static PDSCH slot number used by PDSCH update_cell_command (test/debug override; -1 = use msg.slot). */
    [[nodiscard]] int staticPdschSlotNum() const noexcept
    {
        return mod_->staticPdschSlotNum();
    }

    /**
     * Per-cell static parameter index assigned during cell creation.
     *
     * Retrieves the index from the @c phy instance for the given cell.
     *
     * @param[in] cell_id  Logical cell id.
     * @return             Static parameter index for the cell.
     *                     Return value must be checked.
     */
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t cell_id) const noexcept
    {
        const auto* phy = guarded_phy(cell_id);
        if (phy == nullptr) [[unlikely]] { return UINT16_MAX; }
        return static_cast<uint16_t>(phy->get_cell_stat_prm_idx());
    }

    /**
     * Logical carrier ID for a given cell (maps to phy::get_carrier_id()).
     *
     * Used to populate @c pdsch_params::cell_index_list during per-cell setup.
     *
     * @param[in] cell_idx  Logical cell index.
     * @return              Carrier index for the cell. Return value must be checked.
     */
    [[nodiscard]] int32_t carrier_id(uint32_t cell_idx) const noexcept
    {
        const auto* phy = guarded_phy(cell_idx);
        if (phy == nullptr) [[unlikely]] { return -1; }
        return phy->get_carrier_id();
    }

    /**
     * Physical cell ID for a given cell (maps to phy::get_phy_cell_params().phyCellId).
     *
     * Used to populate @c pdsch_params::phy_cell_index_list during per-cell setup.
     *
     * @param[in] cell_idx  Logical cell index.
     * @return              Physical cell identifier. Return value must be checked.
     */
    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_idx) const noexcept
    {
        const auto* phy = guarded_phy(cell_idx);
        if (phy == nullptr) [[unlikely]] { return UINT16_MAX; }
        return phy->get_phy_cell_params().phyCellId;
    }

    /**
     * Per-cell L1 limit error state for PDCCH (e.g. DCI count overflow).
     * @param[in] cell_id  Logical cell id.
     */
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t cell_id) const
    {
        return mod_->get_cell_limit_errors(cell_id);
    }

    /**
     * DL BWP PRB count for a given cell (maps to phy::get_phy_cell_params().nPrbDlBwp).
     *
     * Used to populate @c pdsch_fh_prepare_params::num_dl_prb during per-PDU FH-params fill.
     * Uses the same per-cell PHY-instance lookup pattern as @c carrier_id() — avoids needing
     * a @c slot_indication to construct a @c PhyCellView on the hot path.
     *
     * @param[in] cell_idx  Logical cell index.
     * @return              DL BWP PRB count for the cell. Return value must be checked.
     */
    [[nodiscard]] uint16_t num_dl_prb(uint32_t cell_idx) const noexcept
    {
        const auto* phy = guarded_phy(cell_idx);
        if (phy == nullptr) [[unlikely]] { return 0; }
        return phy->get_phy_cell_params().nPrbDlBwp;
    }

    /**
     * Runtime FH gate: whether @c fapi_to_cplane_direct mode is enabled.
     *
     * When true, C-plane is created by other worker threads via
     * @c l1_setup_early_cplane_slot_maps() and the PDSCH parser must NOT populate
     * @c cell_group_command::fh_params. When false (current production default),
     * the parser populates @c fh_params for the FH callback to consume via
     * @c LegacyCellGroupFhContext.
     *
     * Delegates to @c l1_is_fapi_to_cplane_direct(driver) on the cuphydriver
     * singleton; returns false when the driver is unavailable.
     *
     * @return  True when direct mode is active; false otherwise.
     *          Return value must be checked.
     */
    [[nodiscard]] bool is_fapi_to_cplane_direct_enabled() const noexcept
    {
        auto* proxy = nv::PHYDriverProxy::getInstancePtr();
        if (proxy == nullptr) [[unlikely]] { return false; }
        const auto driver = proxy->get_driver();
        if (driver == nullptr) [[unlikely]] { return false; }
        return l1_is_fapi_to_cplane_direct(driver);
    }

    void publish_dl_pdsch_stats(const DlPdschStatsBatch& batch) const noexcept
    {
        auto& phy_instances = mod_->PHY_instances();
        auto active_mask = batch.active_mask();
        while (active_mask != 0U) {
            const auto cell_id = static_cast<std::uint32_t>(
                std::countr_zero(active_mask));
            if (cell_id < phy_instances.size()) [[likely]] {
                const auto& delta = batch.delta(cell_id);
                stats_cell_view(cell_id).publish_dl_pdsch_stats(delta.bytes, delta.slots);
            }
            active_mask &= (active_mask - 1U);
        }
    }

    // --- UL channel processing accessors ---

    /** Static PUSCH slot number used by PUSCH update_cell_command. */
    [[nodiscard]] int staticPuschSlotNum() const noexcept
    {
        return mod_->staticPuschSlotNum();
    }

    /** LBRM flag used by PUSCH update_cell_command. */
    [[nodiscard]] uint8_t lbrm() const noexcept
    {
        return mod_->lbrm();
    }

    /**
     * @brief Group-level L1 limit counters used by PUCCH validation.
     *
     * Called from @c parse_pucch_pdu_common (which is @c noexcept) outside
     * the @c try block, so this accessor must also be @c noexcept.
     *
     * @return  Mutable reference to the slot's group-level L1 limit counters.
     *          Return value must be checked.
     */
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return mod_->get_group_limit_errors();
    }

    /** PUCCH DTX thresholds (format 0/1 and 2/3/4). */
    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds() const noexcept
    {
        return mod_->dtx_thresholds();
    }

    /** PUSCH DTX threshold. */
    [[nodiscard]] const float& dtx_thresholds_pusch() const noexcept
    {
        return mod_->dtx_thresholds_pusch();
    }

    /**
     * Whether PUSCH weighted-average CFO estimation (and the associated LDPC
     * extension fields) is enabled.
     *
     * Mirrors the legacy gate l1_get_enable_weighted_average_cfo() on the
     * PHYDriverProxy singleton (see update_cell_command PUSCH path).  The flag
     * is constant within a slot, so it is queried once at view construction and
     * cached (mirrors srs_enabled_/mmimo_enabled_) rather than hitting the
     * singleton per PUSCH PDU on the slot hot path.
     *
     * @return True when the PUSCH extension section is present and should be parsed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool enable_weighted_avg_cfo() const noexcept
    {
        return enable_weighted_avg_cfo_;
    }

    /**
     * IPC transport for a given cell (used by SRS update_cell_command).
     * @param[in] cell_id  Logical cell id.
     */
    [[nodiscard]] nv::phy_mac_transport& transport(int cell_id) const
    {
        return mod_->transport(cell_id);
    }

    // --- PRACH-only accessors (PrachModuleView concept, MR 2b of GT-11843) ---
    //
    // These accessors expand the universal UlModuleView into the narrower
    // PrachModuleView concept (cubb-review §12.4 ISP — sibling parsers do not
    // see PRACH-only state). The new PRACH parser introduced in MR 3a-MR 4 is
    // templated on PrachModuleView and reads PHY config / additional config /
    // RU type / limit errors / is_fapi_to_cplane_direct through these accessors
    // instead of calling PHYDriverProxy::getInstance() on the hot path.

    /**
     * PRACH PHY config (read-only) for a cell.
     *
     * @param[in] cell_id  Logical cell id.
     * @return  Const reference to the cell's @c nv::phy_config (incl. prach_config_).
     *          Return value must be checked.
     * @throws  std::out_of_range when @p cell_id is not a valid PHY instance index.
     */
    [[nodiscard]] const nv::phy_config& phy_config(uint32_t cell_id) const
    {
        return to_scf_5g_fapi(cell_id).get_phy_config();
    }

    /**
     * PRACH additional config (read-only) for a cell.
     *
     * @param[in] cell_id  Logical cell id.
     * @return  Const reference to the cell's @c nv::prach_addln_config_t.
     *          Return value must be checked.
     * @throws  std::out_of_range when @p cell_id is not a valid PHY instance index.
     */
    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(uint32_t cell_id) const
    {
        return to_scf_5g_fapi(cell_id).get_prach_addln_config();
    }

    /**
     * RU type for a cell.
     *
     * Hoists the @c PHYDriverProxy::getInstance().getMPlaneConfig(cell_id).ru
     * lookup out of the per-occasion PRACH loop into a single per-PDU read.
     *
     * @param[in] cell_id  Logical cell id (== carrier_id for single-carrier configs).
     * @return  RU type for the cell; @c OTHER_MODE on out-of-range @p cell_id (noexcept).
     */
    [[nodiscard]] ru_type ru_type_for_cell(uint32_t cell_id) const noexcept
    {
        const auto* phy = guarded_phy(cell_id);
        if (phy == nullptr) [[unlikely]] { return OTHER_MODE; }
        return phy->get_ru_type();
    }

    /**
     * FAPI-to-C-plane direct mode flag.
     *
     * True when the framework C-plane service is consuming PartialUplaneSlotInfo
     * directly (FH callback bypassed via EnqueueSkipMask::SKIP_DL_FHCB at
     * nv_phy_module.cpp:1217-1225). The new PRACH parser uses this to skip
     * populating @c sym_prb_info when the FH path won't consume it — saves
     * ~10 stores × N occasions per PRACH PDU per cell in direct deployments.
     *
     * @return  true in direct mode; false in legacy/FH-callback mode.
     *          Return value must be checked.
     */
    [[nodiscard]] bool is_fapi_to_cplane_direct() const noexcept
    {
        auto driver = nv::PHYDriverProxy::getInstance().get_driver();
        return driver != nullptr && l1_is_fapi_to_cplane_direct(driver);
    }

    // --- SRS / generic UlModuleView accessors (MR 1 of GT-12419) ---

    /**
     * Configured indication instance mode for a given FAPI indication type.
     * @param[in] indication_idx  Index such as nv::SRS_DATA_IND_IDX.
     */
    [[nodiscard]] uint8_t indication_instances_per_slot([[maybe_unused]] const uint32_t indication_idx) const noexcept
    {
#ifdef SCF_FAPI_10_04
        if (indication_idx >= nv::MAX_IND_INDEX)
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "FapiMessageContext::indication_instances_per_slot: indication_idx={} "
                       "out of range (MAX_IND_INDEX={}) — returning 0",
                       indication_idx, nv::MAX_IND_INDEX);
            return 0;
        }
        return mod_->get_phy_config().indication_instances_per_slot[indication_idx];
#else
        return 0;
#endif
    }

    void send_fapi_error_indication(const uint32_t cell_id,
                                    const scf_fapi_message_id_e msg_id,
                                    const scf_fapi_error_codes_t error_code,
                                    const uint16_t sfn,
                                    const uint16_t slot) const
    {
        if (cell_id >= mod_->PHY_instances().size())
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "FapiMessageContext::send_fapi_error_indication: cell_id={} "
                       "out of range (PHY_instances size={}) — dropping error indication "
                       "(msg_id={}, error_code={}, sfn={}, slot={})",
                       cell_id, mod_->PHY_instances().size(),
                       static_cast<int>(msg_id), static_cast<int>(error_code), sfn, slot);
            return;
        }
        auto& phy_inst = mod_->PHY_instances()[cell_id];
        phy_inst.get().send_fapi_error_indication(msg_id, error_code, sfn, slot);
    }

    [[nodiscard]] bool srs_enabled() const noexcept { return srs_enabled_; }

    [[nodiscard]] ru_type ru(uint32_t cell_id) const noexcept
    {
        try {
            return nv::PHYDriverProxy::getInstance()
                       .getMPlaneConfig(static_cast<int>(cell_id)).ru;
        } catch (const std::exception& ex) {
            NVLOGW_FMT(detail::k_tag,
                       "FapiMessageContext::ru: getMPlaneConfig(cell_id={}) failed — "
                       "returning OTHER_MODE: {}",
                       cell_id, ex.what());
            return OTHER_MODE;
        }
    }

    // Owns the FAPI-handle index decode + the chest-buffer state lookup
    // for an SRS PDU. The parser acts on the verdict; it does not need
    // to know the layout of pdu.handle or the meaning of the underlying
    // srsChestBuffState enum.
    [[nodiscard]] SrsChestBuffVerdict
    classify_srs_chest_buffer(const uint32_t cell_id, const scf_fapi_srs_pdu_t& pdu) const
    {
        const auto buffer_idx = static_cast<uint16_t>((pdu.handle >> 8) & 0xFFFF);
        slot_command_api::srsChestBuffState state{slot_command_api::SRS_CHEST_BUFF_NONE};
        if (nv::PHYDriverProxy::getInstance().l1_cv_mem_bank_get_buffer_state(
                cell_id, buffer_idx, &state) == -1)
        {
            return SrsChestBuffVerdict::LookupFailed;
        }
        if (state == slot_command_api::SRS_CHEST_BUFF_REQUESTED)
        {
            return SrsChestBuffVerdict::AlreadyRequested;
        }
        return SrsChestBuffVerdict::Accept;
    }

    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept
    {
        auto* driver = nv::PHYDriverProxy::getInstance().get_driver();
        return driver && l1_is_fapi_to_cplane_direct(driver);
    }

private:
    /** True when @p cell_id is a valid index into @c PHY_instances(). */
    [[nodiscard]] bool valid_cell_id(uint32_t cell_id) const noexcept
    {
        return cell_id < mod_->PHY_instances().size();
    }

    /**
     * Bounds-guarded PHY lookup shared by every per-cell accessor.
     *
     * Owns @c valid_cell_id + @c PHY_instances()[cell_id] + @c static_cast so
     * sentinel-return accessors cannot forget the guard (A-1 / B-1).
     *
     * @param[in] cell_id  Logical cell id / index into @c PHY_instances().
     * @return  Pointer to the @c scf_5g_fapi::phy instance, or @c nullptr when
     *          @p cell_id is out of range.
     */
    [[nodiscard]] scf_5g_fapi::phy* guarded_phy(uint32_t cell_id) const noexcept
    {
        if (!valid_cell_id(cell_id)) [[unlikely]] { return nullptr; }
        return &static_cast<scf_5g_fapi::phy&>(mod_->PHY_instances()[cell_id].get());
    }

    /**
     * Encapsulate the @c PHY_instances().at(cell_id) + @c static_cast pattern for
     * PRACH accessors that must return a reference (cannot use a sentinel).
     * Matches @c cell_view()'s exception-based bounds contract — @c std::optional /
     * @c expected are intentionally not used here.
     *
     * @param[in] cell_id  Logical cell id.
     * @return  Const reference to the scf_5g_fapi::phy instance for @p cell_id.
     * @throws  std::out_of_range when @p cell_id is not a valid PHY instance index.
     */
    [[nodiscard]] const scf_5g_fapi::phy& to_scf_5g_fapi(uint32_t cell_id) const
    {
        return static_cast<const scf_5g_fapi::phy&>(mod_->PHY_instances().at(cell_id).get());
    }

    /**
     * Stats-only cell view. Uses @c .at() for the same loud out-of-range failure
     * mode as @c cell_view() (caller @c publish_dl_pdsch_stats already bounds-checks).
     *
     * @param[in] cell_id  Logical cell id.
     * @throws  std::out_of_range when @p cell_id is not a valid PHY instance index.
     */
    [[nodiscard]] PhyCellView stats_cell_view(uint32_t cell_id) const
    {
        auto& phy_inst = mod_->PHY_instances().at(cell_id);
        return PhyCellView{static_cast<scf_5g_fapi::phy&>(phy_inst.get()),
                           cell_id,
                           slot_command_api::slot_indication{}};
    }

    gsl_lite::not_null<nv::PHY_module*> mod_;  //!< Non-owning pointer to the PHY module.
    bool            mmimo_enabled_{};  //!< MU-MIMO flag from PHYDriverProxy.
    bool            srs_enabled_{};    //!< SRS-enable flag from PHYDriverProxy; cached at slot boundary.
    bool            enable_weighted_avg_cfo_{};  //!< PUSCH weighted-avg-CFO / LDPC-extension gate; cached from PHYDriverProxy at slot boundary (constant within a slot, queried once per view).
    uint32_t        slot_cmd_idx_{};   //!< Slot-command ring index captured at EOM for this slot.
    OrderSymPrbInfoFn order_sym_prb_info_fn_{};
    void*             order_sym_prb_info_ctx_{};
};

static_assert(CellView<PhyCellView>);
static_assert(ModuleView<PhyModuleView>);
static_assert(PrachModuleView<PhyModuleView>,
              "PhyModuleView must satisfy PrachModuleView (GT-11843)");
static_assert(PuschModuleView<PhyModuleView>,
              "PhyModuleView must satisfy PuschModuleView");
static_assert(UlModuleView<PhyModuleView>);
static_assert(std::is_same_v<typename module_view_traits<PhyModuleView>::cell_view_type, PhyCellView>);
static_assert(std::is_same_v<typename cell_view_traits<PhyCellView>::module_view_type, PhyModuleView>);

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_MESSAGE_CONTEXT_HPP_INCLUDED_
