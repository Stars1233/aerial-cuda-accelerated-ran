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

#ifndef SCF_5G_FH_CALLBACK_CONTEXT_H
#define SCF_5G_FH_CALLBACK_CONTEXT_H

#include <array>
#include <concepts>
#include <cstdint>
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi {

class PdschFhParamsView final {
public:
    /**
     * @brief Construct a PDSCH FH parameter view.
     * @param params Non-owning reference to PDSCH FH prepare parameters.
     */
    explicit PdschFhParamsView(slot_command_api::pdsch_fh_prepare_params& params) noexcept
        : params_(&params)
    {}

    /** @brief Get UE group parameters. @return Reference to UE group parameters. */
    [[nodiscard]] cuphyPdschUeGrpPrm_t& grp() const { return *params_->grp; }
    /** @brief Get UE parameters. @return Reference to UE parameters. */
    [[nodiscard]] cuphyPdschUePrm_t& ue() const { return *params_->ue; }
    /** @brief Get symbol PRB info pointer. @return Pointer to slot PRB info. */
    [[nodiscard]] slot_command_api::slot_info_* sym_prb_info() const { return params_->cell_cmd->sym_prb_info(); }
    /** @brief Get slot indication for this PDSCH entry. @return Reference to slot indication. */
    [[nodiscard]] slot_command_api::slot_indication& slot_3gpp() const { return params_->cell_cmd->slot.slot_3gpp; }
    /** @brief Get precoding/beamforming view in SCF format. @return Reference to SCF TX precoding/beamforming object. */
    [[nodiscard]] scf_fapi_tx_precoding_beamforming_t& pc_bf() const
    {
        return *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(params_->pc_bf);
    }
    /** @brief Get BFW coefficient memory info. @return Pointer to coefficient memory info, may be null. */
    [[nodiscard]] bfw_coeff_mem_info_t* bfw_coeff_mem_info() const { return params_->bfwCoeff_mem_info; }
    /** @brief Get per-cell BFW index for this UE group. @return BFW index value. */
    [[nodiscard]] uint32_t ue_grp_bfw_index_per_cell() const { return params_->ue_grp_bfw_index_per_cell; }
    /** @brief Get UE group index. @return UE group index. */
    [[nodiscard]] uint16_t ue_grp_index() const { return params_->ue_grp_index; }
    /** @brief Get DL PRB count. @return Number of DL PRBs. */
    [[nodiscard]] uint16_t num_dl_prb() const { return params_->num_dl_prb; }
    /** @brief Get logical cell index. @return Cell index. */
    [[nodiscard]] uint16_t cell_index() const { return params_->cell_index; }
    /** @brief Check whether this entry starts a new UE group. @return True if this is a new group entry. */
    [[nodiscard]] bool is_new_grp() const { return params_->is_new_grp; }
    /** @brief Check BF enable flag. @return True when BF is enabled. */
    [[nodiscard]] bool bf_enabled() const { return params_->bf_enabled; }
    /** @brief Check PM enable flag. @return True when PM is enabled. */
    [[nodiscard]] bool pm_enabled() const { return params_->pm_enabled; }
    /** @brief Check mMIMO enable flag. @return True when mMIMO is enabled. */
    [[nodiscard]] bool mmimo_enabled() const { return params_->mmimo_enabled; }
    /** @brief Check CSI-RS compact mode flag. @return True when compact mode is enabled. */
    [[nodiscard]] bool csirs_compact_mode() const { return params_->csirs_compact_mode; }
    /**
     * @brief Set CSI-RS compact mode flag.
     * @param value New compact mode value.
     */
    void set_csirs_compact_mode(const bool value) { params_->csirs_compact_mode = value; }

private:
    slot_command_api::pdsch_fh_prepare_params* params_;
};

class CsirsFhParamsView final {
public:
    /**
     * @brief Construct a CSI-RS FH parameter view.
     * @param params Non-owning reference to CSI-RS FH prepare parameters.
     */
    explicit CsirsFhParamsView(slot_command_api::csirs_fh_prepare_params& params) noexcept
        : params_(&params)
    {}

    /** @brief Get symbol PRB info pointer. @return Pointer to slot PRB info. */
    [[nodiscard]] slot_command_api::slot_info_* sym_prb_info() const { return params_->cell_cmd->sym_prb_info(); }
    /** @brief Get slot indication for this CSI-RS entry. @return Reference to slot indication. */
    [[nodiscard]] slot_command_api::slot_indication& slot_3gpp() const { return params_->cell_cmd->slot.slot_3gpp; }
    /** @brief Get CSI-RS cell index. @return Cell index for CSI-RS FH params. */
    [[nodiscard]] uint32_t cell_idx() const { return params_->cell_idx; }
    /** @brief Get cuPHY params cell index. @return cuPHY cell index, or sentinel value set by caller. */
    [[nodiscard]] int32_t cuphy_params_cell_idx() const { return params_->cuphy_params_cell_idx; }
    /** @brief Get DL PRB count. @return Number of DL PRBs. */
    [[nodiscard]] uint16_t num_dl_prb() const { return params_->num_dl_prb; }
    /** @brief Check BF enable flag. @return True when BF is enabled. */
    [[nodiscard]] bool bf_enabled() const { return params_->bf_enabled; }
    /** @brief Check mMIMO enable flag. @return True when mMIMO is enabled. */
    [[nodiscard]] bool mmimo_enabled() const { return params_->mmimo_enabled; }

private:
    slot_command_api::csirs_fh_prepare_params* params_;
};

/**
 * @brief Concept for PDSCH FH parameter access views.
 * @tparam T Candidate view type.
 *
 * Requires access to PDSCH UE/group data, PRB/slot context, beamforming data,
 * and mMIMO enable state needed by FH PDSCH processing.
 */
template <typename T>
concept FhPdschParamsAccessor = requires(const T& view) {
    { view.grp() } -> std::same_as<cuphyPdschUeGrpPrm_t&>;
    { view.ue() } -> std::same_as<cuphyPdschUePrm_t&>;
    { view.sym_prb_info() } -> std::same_as<slot_command_api::slot_info_*>;
    { view.slot_3gpp() } -> std::same_as<slot_command_api::slot_indication&>;
    { view.pc_bf() } -> std::same_as<scf_fapi_tx_precoding_beamforming_t&>;
    { view.mmimo_enabled() } -> std::convertible_to<bool>;
};

/**
 * @brief Concept for CSI-RS FH parameter access views.
 * @tparam T Candidate view type.
 *
 * Requires access to CSI-RS slot/PRB context and core per-cell CSI-RS FH fields.
 */
template <typename T>
concept FhCsirsParamsAccessor = requires(const T& view) {
    { view.sym_prb_info() } -> std::same_as<slot_command_api::slot_info_*>;
    { view.slot_3gpp() } -> std::same_as<slot_command_api::slot_indication&>;
    { view.cell_idx() } -> std::convertible_to<uint32_t>;
    { view.num_dl_prb() } -> std::convertible_to<uint16_t>;
};

/**
 * @brief Concept for cell-scoped FH context views.
 * @tparam T Candidate context view type.
 *
 * Defines the minimum API required by the FH callback flow to read PDSCH/CSI-RS
 * command data, FH param ranges, slot identity, and aggregate counters.
 */
template <typename T>
concept FhCellView = requires(T& view, const T& cview, uint32_t idx) {
    { view.get_pdsch_params() } -> std::same_as<slot_command_api::pdsch_params*>;
    { view.get_csirs_params() } -> std::same_as<slot_command_api::csirs_params*>;
    { view.get_pm_group() } -> std::same_as<slot_command_api::pm_group*>;
    { cview.is_csirs_cell() } -> std::convertible_to<bool>;
    { cview.num_pdsch_fh_params() } -> std::convertible_to<int>;
    { view.csirs_fh_params() } -> std::same_as<slot_command_api::csirs_fh_prepare_params&>;
    { cview.start_index_pdsch_fh_params() } -> std::convertible_to<uint32_t>;
    { cview.csirs_symbol_map() } -> std::convertible_to<uint16_t>;
    { cview.slot_sfn() } -> std::convertible_to<uint16_t>;
    { cview.slot_slot() } -> std::convertible_to<uint8_t>;
    { cview.num_csirs_cells() } -> std::convertible_to<uint32_t>;
    { cview.total_num_pdsch_pdus() } -> std::convertible_to<uint32_t>;
};

class CellGroupFhView final {
public:
    /**
     * @brief Construct a bound view over a cell group command.
     * @pre @p grp_cmd must outlive this view instance.
     * @post This object stores a non-owning pointer to @p grp_cmd.
     */
    explicit CellGroupFhView(slot_command_api::cell_group_command& grp_cmd, const uint8_t cell_index = 0) noexcept
        : cmd_(&grp_cmd), cell_index_(cell_index)
    {}

    /**
     * @brief Construct a bound view from a raw cell group command pointer.
     * @pre @p grp_cmd must be non-null and must outlive this view instance.
     * @post This object stores the same non-owning pointer passed by the caller.
     */
    explicit CellGroupFhView(slot_command_api::cell_group_command* grp_cmd, const uint8_t cell_index = 0) noexcept
        : cmd_(grp_cmd), cell_index_(cell_index)
    {}

    /**
     * @brief Bind this view to a specific cell index.
     * @param cell_index Cell index to bind.
     */
    void bind_cell(const uint8_t cell_index) noexcept
    {
        cell_index_ = cell_index;
    }

    /** @brief Get PDSCH aggregate params pointer. @return Pointer to PDSCH params. */
    [[nodiscard]] slot_command_api::pdsch_params* get_pdsch_params()
    {
        return cmd_->get_pdsch_params();
    }

    /** @brief Get CSI-RS aggregate params pointer. @return Pointer to CSI-RS params, or null if absent. */
    [[nodiscard]] slot_command_api::csirs_params* get_csirs_params() const
    {
        return cmd_->csirs.get();
    }

    /** @brief Get PM group pointer. @return Pointer to PM group data. */
    [[nodiscard]] slot_command_api::pm_group* get_pm_group()
    {
        return cmd_->get_pm_group();
    }

    /** @brief Check if bound cell is marked as CSI-RS cell. @return True if CSI-RS cell. */
    [[nodiscard]] bool is_csirs_cell() const
    {
        return cmd_->fh_params.is_csirs_cell[cell_index_];
    }

    /** @brief Get number of PDSCH FH entries for bound cell. @return PDSCH FH count. */
    [[nodiscard]] int num_pdsch_fh_params() const
    {
        return cmd_->fh_params.num_pdsch_fh_params[cell_index_];
    }

    /** @brief Get mutable CSI-RS FH params for bound cell. @return Reference to CSI-RS FH params. */
    [[nodiscard]] slot_command_api::csirs_fh_prepare_params& csirs_fh_params()
    {
        return cmd_->fh_params.csirs_fh_params[cell_index_];
    }

    /** @brief Get CSI-RS FH params view for bound cell. @return CSI-RS FH params view. */
    [[nodiscard]] CsirsFhParamsView csirs_fh_params_view() const
    {
        return CsirsFhParamsView(cmd_->fh_params.csirs_fh_params[cell_index_]);
    }

    /** @brief Get start index of PDSCH FH params for bound cell. @return Start index into PDSCH FH array. */
    [[nodiscard]] uint32_t start_index_pdsch_fh_params() const
    {
        return cmd_->fh_params.start_index_pdsch_fh_params[cell_index_];
    }

    /**
     * @brief Get PDSCH FH params view by global index.
     * @param index Global PDSCH FH index.
     * @return PDSCH FH params view.
     */
    [[nodiscard]] PdschFhParamsView pdsch_fh_params_view(const uint32_t index) const
    {
        return PdschFhParamsView(cmd_->fh_params.pdsch_fh_params[index]);
    }

    /** @brief Get bound-cell CSI-RS symbol map. @return CSI-RS symbol map for bound cell, or 0 when CSI-RS is absent. */
    [[nodiscard]] uint16_t csirs_symbol_map() const
    {
        auto* csirs = cmd_->csirs.get();
        return csirs ? csirs->symbolMapArray[cell_index_] : 0;
    }

    /** @brief Get slot SFN. @return SFN from slot indication. */
    [[nodiscard]] uint16_t slot_sfn() const
    {
        return cmd_->slot.slot_3gpp.sfn_;
    }

    /** @brief Get slot number. @return Slot from slot indication. */
    [[nodiscard]] uint8_t slot_slot() const
    {
        return cmd_->slot.slot_3gpp.slot_;
    }

    /** @brief Get total number of CSI-RS cells in FH params. @return CSI-RS cell count. */
    [[nodiscard]] uint32_t num_csirs_cells() const
    {
        return cmd_->fh_params.num_csirs_cell;
    }

    /** @brief Get total number of PDSCH PDUs in FH params. @return PDSCH PDU count. */
    [[nodiscard]] uint32_t total_num_pdsch_pdus() const
    {
        return cmd_->fh_params.total_num_pdsch_pdus;
    }

    /** @brief Access underlying cell group command. @return Raw non-owning pointer to cell group command. */
    [[nodiscard]] slot_command_api::cell_group_command* raw_group_cmd()
    {
        return cmd_;
    }

    // Backward-compatible methods for existing FH pipeline code.
    /** @brief Check whether CSI-RS params are present. @return True when CSI-RS params pointer is non-null. */
    [[nodiscard]] bool has_csirs_params() const
    {
        return get_csirs_params() != nullptr;
    }

    /** @brief Get CSI-RS params pointer. @return Pointer to CSI-RS params, may be null. */
    [[nodiscard]] slot_command_api::csirs_params* csirs_params_ptr() const
    {
        return cmd_->csirs.get();
    }

    /**
     * @brief Get CSI-RS symbol map for a specific cell.
     * @param cell_index Cell index.
     * @return Symbol map value.
     */
    [[nodiscard]] uint16_t csirs_symbol_map(const uint8_t cell_index) const
    {
        return cmd_->csirs->symbolMapArray[cell_index];
    }

    /**
     * @brief Get mutable reference to CSI-RS symbol map entry.
     * @param cell_index Cell index.
     * @return Mutable reference to symbol map entry.
     */
    [[nodiscard]] uint16_t& csirs_symbol_map_ref(const uint8_t cell_index)
    {
        return cmd_->csirs->symbolMapArray[cell_index];
    }

    /**
     * @brief Get CSI-RS remap value.
     * @param cell_index Cell index.
     * @param remap_index Remap array index.
     * @return Remap value.
     */
    [[nodiscard]] uint16_t csirs_remap(const uint8_t cell_index, const uint32_t remap_index) const
    {
        return cmd_->csirs->reMap[cell_index][remap_index];
    }

    /**
     * @brief Get pointer to CSI-RS remap row for a cell.
     * @param cell_index Cell index.
     * @return Pointer to remap row data.
     */
    [[nodiscard]] uint16_t* csirs_remap_row(const uint8_t cell_index) const
    {
        return cmd_->csirs->reMap[cell_index];
    }

    /** @brief Get CSI-RS cell index list. @return Reference to CSI-RS cell index vector. */
    [[nodiscard]] const std::vector<int32_t>& csirs_cell_index_list() const
    {
        return cmd_->csirs->cell_index_list;
    }

    /**
     * @brief Get number of CSI-RS RRC params for a CSI-RS cell info entry.
     * @param cell_info_index Index in CSI-RS cellInfo array.
     * @return Number of RRC params.
     */
    [[nodiscard]] uint32_t csirs_cell_num_rrc_params(const uint32_t cell_info_index) const
    {
        return cmd_->csirs->cellInfo[cell_info_index].nRrcParams;
    }

    /**
     * @brief Get CSI-RS RRC params offset for a CSI-RS cell info entry.
     * @param cell_info_index Index in CSI-RS cellInfo array.
     * @return Offset into CSI-RS RRC params array.
     */
    [[nodiscard]] uint32_t csirs_cell_rrc_params_offset(const uint32_t cell_info_index) const
    {
        return cmd_->csirs->cellInfo[cell_info_index].rrcParamsOffset;
    }

    /** @brief Get CSI-RS dynamic params pointer. @return Pointer to CSI-RS dynamic params array. */
    [[nodiscard]] const cuphyCsirsRrcDynPrm_t* csirs_rrc_dyn_params() const
    {
        return cmd_->csirs->csirsDynPrms.pRrcDynPrm;
    }

    /**
     * @brief Get CSI-RS precoding/beamforming entry by index.
     * @param pc_bf_index Index in CSI-RS pcAndBf array.
     * @return Reference to TX precoding/beamforming entry.
     */
    [[nodiscard]] tx_precoding_beamforming_t& csirs_pc_and_bf(const uint32_t pc_bf_index) 
    {
        return cmd_->csirs->pcAndBf[pc_bf_index];
    }

    /** @brief Get PDSCH cell-group dynamic params. @return Reference to PDSCH cell-group info. */
    [[nodiscard]] const cuphyPdschCellGrpDynPrm_t& pdsch_cell_grp_info() const
    {
        return cmd_->get_pdsch_params()->cell_grp_info;
    }

    /**
     * @brief Get PDSCH UE DMRS info by UE group index.
     * @param ue_grp_index UE group index.
     * @return Reference to UE-group DMRS parameters.
     */
    [[nodiscard]] const cuphyPdschDmrsPrm_t& pdsch_ue_dmrs_info(const uint32_t ue_grp_index) const
    {
        return cmd_->get_pdsch_params()->ue_dmrs_info[ue_grp_index];
    }

    /**
     * @brief Get QAM modulation order for a codeword.
     * @param cw_global_index Global codeword index.
     * @return QAM modulation order.
     */
    [[nodiscard]] uint8_t pdsch_ue_cw_qam_mod_order(const uint32_t cw_global_index) const
    {
        return cmd_->get_pdsch_params()->ue_cw_info[cw_global_index].qamModOrder;
    }

    /**
     * @brief Get number of RA type-0 info entries for UE group.
     * @param ue_grp_index UE group index.
     * @return Number of RA type-0 entries.
     */
    [[nodiscard]] uint32_t pdsch_num_ra_type0_info(const uint32_t ue_grp_index) const
    {
        return cmd_->get_pdsch_params()->num_ra_type0_info[ue_grp_index];
    }

    /**
     * @brief Get RA type-0 info entry.
     * @param info_index RA info index.
     * @param ue_grp_index UE group index.
     * @return Reference to RA type-0 info entry.
     */
    [[nodiscard]] const slot_command_api::ra_type0_info_t_& pdsch_ra_type0_info(const uint32_t info_index, const uint32_t ue_grp_index) const
    {
        return cmd_->get_pdsch_params()->ra_type0_info[info_index][ue_grp_index];
    }

    /** @brief Get PDSCH-owned CSI-RS dynamic params pointer. @return Pointer to CSI-RS dynamic params in PDSCH group info. */
    [[nodiscard]] const cuphyCsirsRrcDynPrm_t* pdsch_csirs_rrc_dyn_params() const
    {
        return cmd_->get_pdsch_params()->cell_grp_info.pCsiRsPrms;
    }

    /**
     * @brief Get number of CSI-RS params for a cuPHY cell index.
     * @param cuphy_cell_index cuPHY cell index.
     * @return Number of CSI-RS params.
     */
    [[nodiscard]] uint32_t pdsch_cell_num_csirs_params(const int32_t cuphy_cell_index) const
    {
        return cmd_->get_pdsch_params()->cell_grp_info.pCellPrms[cuphy_cell_index].nCsiRsPrms;
    }

    /**
     * @brief Get CSI-RS params offset for a cuPHY cell index.
     * @param cuphy_cell_index cuPHY cell index.
     * @return CSI-RS params offset.
     */
    [[nodiscard]] uint32_t pdsch_cell_csirs_params_offset(const int32_t cuphy_cell_index) const
    {
        return cmd_->get_pdsch_params()->cell_dyn_info[cuphy_cell_index].csiRsPrmsOffset;
    }

    /**
     * @brief Check FH CSI-RS-cell flag for a cell.
     * @param cell_index Cell index.
     * @return True if FH marks the cell as CSI-RS.
     */
    [[nodiscard]] bool fh_is_csirs_cell(const uint8_t cell_index) const
    {
        return cmd_->fh_params.is_csirs_cell[cell_index];
    }

    /**
     * @brief Get FH PDSCH FH-params count for a cell.
     * @param cell_index Cell index.
     * @return Number of PDSCH FH params for the cell.
     */
    [[nodiscard]] uint8_t fh_num_pdsch_fh_params(const uint8_t cell_index) const
    {
        return cmd_->fh_params.num_pdsch_fh_params[cell_index];
    }

    /**
     * @brief Get FH start index for PDSCH FH params of a cell.
     * @param cell_index Cell index.
     * @return Start index in FH PDSCH params array.
     */
    [[nodiscard]] uint16_t fh_start_index_pdsch_fh_params(const uint8_t cell_index) const
    {
        return cmd_->fh_params.start_index_pdsch_fh_params[cell_index];
    }

    /** @brief Get FH number of CSI-RS cells. @return CSI-RS cell count. */
    [[nodiscard]] uint32_t fh_num_csirs_cells() const
    {
        return cmd_->fh_params.num_csirs_cell;
    }

    /** @brief Get FH number of PDSCH PDUs. @return PDSCH PDU count. */
    [[nodiscard]] uint32_t fh_num_pdsch_pdus() const
    {
        return cmd_->fh_params.total_num_pdsch_pdus;
    }

    /** @brief Get FH slot indication. @return Reference to slot indication. */
    [[nodiscard]] const slot_command_api::slot_indication& fh_slot_indication() const
    {
        return cmd_->slot.slot_3gpp;
    }

    /** @brief Get PM group pointer. @return Pointer to PM group. */
    [[nodiscard]] slot_command_api::pm_group* pm_group_ptr() const
    {
        return cmd_->get_pm_group();
    }

private:
    slot_command_api::cell_group_command* cmd_;
    uint8_t cell_index_;
};

static_assert(FhCellView<CellGroupFhView>);
static_assert(FhPdschParamsAccessor<PdschFhParamsView>);
static_assert(FhCsirsParamsAccessor<CsirsFhParamsView>);

class SingleCellFhView final {
public:
    /**
     * @brief Construct a single-cell FH view.
     * @param cell_cmd Non-owning pointer to cell sub-command.
     * @param pdsch Non-owning pointer to PDSCH params.
     * @param csirs Non-owning pointer to CSI-RS params.
     * @param pm_grp Non-owning pointer to PM group.
     * @param slot Slot indication value copied into this view.
     */
    SingleCellFhView(slot_command_api::cell_sub_command* cell_cmd,
        slot_command_api::pdsch_params* pdsch,
        slot_command_api::csirs_params* csirs,
        slot_command_api::pm_group* pm_grp,
        const slot_command_api::slot_indication slot) noexcept
        : cell_cmd_(cell_cmd)
        , pdsch_(pdsch)
        , csirs_(csirs)
        , pm_grp_(pm_grp)
        , slot_(slot)
    {}

    /** @brief Set CSI-RS-cell flag. @param value Flag value. */
    void set_is_csirs_cell(const bool value) { is_csirs_cell_ = value; }
    /** @brief Set number of PDSCH FH params. @param value PDSCH FH count. */
    void set_num_pdsch_fh_params(const int value) { num_pdsch_fh_params_ = value; }
    /** @brief Set PDSCH FH params start index. @param value Start index. */
    void set_start_index_pdsch_fh_params(const uint32_t value) { start_index_ = value; }
    /** @brief Set CSI-RS symbol map. @param value Symbol map value. */
    void set_csirs_symbol_map(const uint16_t value) { csirs_symbol_map_ = value; }
    /** @brief Get mutable CSI-RS FH params storage. @return Reference to local CSI-RS FH params. */
    slot_command_api::csirs_fh_prepare_params& mutable_csirs_fh_params() { return csirs_fh_params_; }
    /**
     * @brief Get mutable PDSCH FH params storage by index.
     * @param index PDSCH FH index.
     * @return Reference to local PDSCH FH params entry.
     */
    slot_command_api::pdsch_fh_prepare_params& mutable_pdsch_fh_param(const uint32_t index) { return pdsch_fh_params_[index]; }

    /** @brief Get PDSCH params pointer. @return Pointer to PDSCH params. */
    [[nodiscard]] slot_command_api::pdsch_params* get_pdsch_params() { return pdsch_; }
    /** @brief Get CSI-RS params pointer. @return Pointer to CSI-RS params. */
    [[nodiscard]] slot_command_api::csirs_params* get_csirs_params() { return csirs_; }
    /** @brief Get PM group pointer. @return Pointer to PM group. */
    [[nodiscard]] slot_command_api::pm_group* get_pm_group() { return pm_grp_; }
    /** @brief Check CSI-RS-cell flag. @return True if this view is CSI-RS cell. */
    [[nodiscard]] bool is_csirs_cell() const { return is_csirs_cell_; }
    /** @brief Get number of PDSCH FH params. @return PDSCH FH count. */
    [[nodiscard]] int num_pdsch_fh_params() const { return num_pdsch_fh_params_; }
    /** @brief Get CSI-RS FH params. @return Reference to local CSI-RS FH params. */
    [[nodiscard]] slot_command_api::csirs_fh_prepare_params& csirs_fh_params() { return csirs_fh_params_; }
    /** @brief Get start index for PDSCH FH params. @return Start index. */
    [[nodiscard]] uint32_t start_index_pdsch_fh_params() const { return start_index_; }
    /** @brief Get CSI-RS symbol map. @return Symbol map value. */
    [[nodiscard]] uint16_t csirs_symbol_map() const { return csirs_symbol_map_; }
    /** @brief Get slot SFN. @return SFN value. */
    [[nodiscard]] uint16_t slot_sfn() const { return slot_.sfn_; }
    /** @brief Get slot number. @return Slot value. */
    [[nodiscard]] uint8_t slot_slot() const { return slot_.slot_; }
    /** @brief Get CSI-RS cell count represented by this view. @return 1 when CSI-RS cell, otherwise 0. */
    [[nodiscard]] uint32_t num_csirs_cells() const { return is_csirs_cell_ ? 1U : 0U; }
    /** @brief Get total PDSCH PDU count represented by this view. @return PDSCH PDU count. */
    [[nodiscard]] uint32_t total_num_pdsch_pdus() const { return static_cast<uint32_t>(num_pdsch_fh_params_); }

private:
    slot_command_api::cell_sub_command* cell_cmd_;
    slot_command_api::pdsch_params* pdsch_;
    slot_command_api::csirs_params* csirs_;
    slot_command_api::pm_group* pm_grp_;
    slot_command_api::slot_indication slot_;
    std::array<slot_command_api::pdsch_fh_prepare_params, MAX_PDSCH_UE_PER_TTI> pdsch_fh_params_{};
    slot_command_api::csirs_fh_prepare_params csirs_fh_params_{};
    uint32_t start_index_{};
    int num_pdsch_fh_params_{};
    bool is_csirs_cell_{};
    uint16_t csirs_symbol_map_{};
};

static_assert(FhCellView<SingleCellFhView>);

using IFhCallbackContext = CellGroupFhView;
using LegacyCellGroupFhContext = IFhCallbackContext;

} // namespace scf_5g_fapi

#endif /* SCF_5G_FH_CALLBACK_CONTEXT_H */
