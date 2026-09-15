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

#ifndef CUPHY_CP_TESTS_UL_MODULE_VIEW_MOCK_HPP_
#define CUPHY_CP_TESTS_UL_MODULE_VIEW_MOCK_HPP_

/**
 * @file ul_module_view_mock.hpp
 * @brief Minimal CellView / UlModuleView test doubles for the UL parser path.
 *
 * Satisfies @c scf_5g_fapi::CellView and @c scf_5g_fapi::UlModuleView so that
 * @c ULSlotProcessor can be exercised in isolation (no PHY module, no GPU) by
 * any uplink channel equivalence test or benchmark (PUSCH, PUCCH, PRACH, SRS).
 *
 * GTest-free on purpose: shared between the functional test and the benchmark.
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>  // std::abort

#include "scf_5g_slot_commands.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_config_option.hpp"

namespace cuphy_cp::tests
{

/// Static cell view returning fixed UL bandwidth and no per-slot detail.
struct MockCellView final
{
    cuphyCellStatPrm_t stat_prm_{};

    /// 273 PRB == 100 MHz @ 30 kHz SCS; the carrier bandwidth in PRB, matching
    /// the ul_bandwidth the legacy populator is configured with.
    static constexpr uint16_t kBandwidthPrb = 273u;

    /// Default antenna count for both nRxAnt and nTxAnt (prior LBRM maxLayers = 4).
    static constexpr uint16_t kDefaultAntCount = 4;

    /**
     * Construct a mock cell view with default antenna counts.
     *
     * Initializes @c stat_prm_.nRxAnt and @c stat_prm_.nTxAnt to
     * @ref kDefaultAntCount so LBRM maxLayers matches the prior hardcoded
     * value of 4.
     */
    MockCellView() noexcept
    {
        stat_prm_.nRxAnt = kDefaultAntCount;
        stat_prm_.nTxAnt = kDefaultAntCount;
    }

    /// @brief Carrier bandwidth in PRB (CellView concept accessor; fixed mock value).
    /// @return Bandwidth in PRB (@ref kBandwidthPrb).
    [[nodiscard]] uint16_t                   num_dl_prb()  const noexcept { return kBandwidthPrb; }
    /// @brief Per-slot TDD detail; the mock has none.
    /// @return Always nullptr (drives the full-slot/symbol-0 fallback path).
    [[nodiscard]] nv::slot_detail_t*         slot_detail() const noexcept { return nullptr; }
    /// @brief Static cell parameters backing the parse.
    /// @return Reference to the owned (zero-initialized) cuphyCellStatPrm_t.
    [[nodiscard]] const cuphyCellStatPrm_t&  cell_params() const noexcept { return stat_prm_; }

    /// @brief Single-sector UL start symbol (used by the SINGLE_SECT FH path).
    /// @return Always 0, mirroring the real CellView when slot_detail() is null.
    [[nodiscard]] uint8_t ul_start_symbol() const noexcept { return 0u; }
    /// @brief Single-sector UL symbol count (used by the SINGLE_SECT FH path).
    /// @param[in] fallback Symbol count to use when no slot detail is available.
    /// @return @p fallback, mirroring the real CellView when slot_detail() is null.
    [[nodiscard]] uint8_t ul_max_symbols(uint8_t fallback) const noexcept { return fallback; }
};

static_assert(scf_5g_fapi::CellView<MockCellView>, "MockCellView must satisfy CellView");

/**
 * @brief UlModuleView test double wired to mirror the legacy populator inputs.
 *
 * Each field maps to one argument of @c scf_5g_fapi::update_cell_command so the
 * new parser path and the legacy path observe identical inputs. The owned
 * @c slot_command receives the parsed result.
 */
struct MockUlModuleView final
{
    int       static_pusch_slot_{-1};
    int32_t   carrier_id_val_{0};
    uint16_t  phy_cell_id_val_{0};
    bool      enable_weighted_avg_cfo_{false};
    uint8_t   lbrm_{0};
    bool      mmimo_enabled_{false};
    bool      bf_enabled_{false};
    ::ru_type ru_type_{OTHER_MODE};
    slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info_{nullptr};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable nv::phy_config_option              config_opt_{};
    mutable nv::pucch_dtx_t_list               dtx_list_{};
    mutable float                              dtx_pusch_{0.0f};
    mutable nv::slot_limit_cell_error_t        limit_errors_{};
    mutable nv::slot_limit_group_error_t       group_limit_errors_{};
    mutable nv::phy_config                     phy_config_obj_{};
    mutable nv::prach_addln_config_t           prach_addln_config_obj_{};
    MockCellView                               cell_view_{};

    /// @brief Active cell-group command for the slot (receives the parsed result).
    /// @return Pointer to the owned slot_command's cell_groups (never null).
    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }
    /// @brief Cell sub-command for a cell index (mock returns one shared instance).
    /// @param[in] cell_index Logical cell index (ignored; single mock cell).
    /// @return Reference to the owned cell_sub_command.
    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_index) const noexcept
    {
        static_cast<void>(cell_index);
        return cell_sub_cmd_;
    }
    /// @brief UL Order metadata destination.
    /// @param[in] cell_index Logical cell index (ignored; single mock cell).
    /// @return Pointer to the owned cell_sub_command slot_info scratch.
    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t cell_index) const noexcept
    {
        static_cast<void>(cell_index);
        return cell_sub_cmd_.sym_prb_info();
    }
    /// @brief The owned slot command under test.
    /// @return Reference to the owned slot_command.
    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept { return slot_cmd_; }

    /// @brief Whether mMIMO is enabled for this slot.
    /// @return The configured mmimo_enabled_ flag.
    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }
    /// @brief Whether beamforming is enabled for this slot.
    /// @return The configured bf_enabled_ flag.
    [[nodiscard]] bool bf_enabled()    const noexcept { return bf_enabled_; }

    /// @brief PHY config options observed by the parser.
    /// @return Reference to the owned phy_config_option.
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    /// @brief Static-PUSCH slot number override.
    /// @return The configured value (-1 disables the static-PUSCH path).
    [[nodiscard]] int     staticPuschSlotNum() const noexcept { return static_pusch_slot_; }
    /// @brief LBRM enable byte.
    /// @return The configured lbrm_ value.
    [[nodiscard]] uint8_t lbrm()               const noexcept { return lbrm_; }

    /// @brief PUCCH DTX thresholds list.
    /// @return Reference to the owned (empty) dtx threshold list.
    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds()       const noexcept { return dtx_list_; }
    /// @brief PUSCH DTX threshold.
    /// @return Reference to the owned PUSCH DTX threshold.
    [[nodiscard]] const float&                dtx_thresholds_pusch() const noexcept { return dtx_pusch_; }
    /// @brief Whether weighted-average CFO is enabled.
    /// @return The configured enable_weighted_avg_cfo_ flag.
    [[nodiscard]] bool enable_weighted_avg_cfo() const noexcept { return enable_weighted_avg_cfo_; }

    /// @brief Transport accessor required by the UlModuleView concept; unreachable
    ///        on the PUSCH parse path.
    /// @param[in] idx Transport index (ignored; the stub never returns).
    /// @note Not [[nodiscard]]: the function is [[noreturn]] in effect (always
    ///       std::abort()), so there is no value to discard. The concept only
    ///       requires the signature; PUCCH/PRACH callers that need a real
    ///       transport must inject one.
    nv::phy_mac_transport& transport(int idx) const noexcept
    {
        static_cast<void>(idx);
        // Fail loudly rather than return a reference to never-constructed storage
        // (UB). phy_mac_transport has no default ctor and its real ctors open NVIPC,
        // so a constructed stub is neither possible nor desirable here.
        std::abort();
    }

    /// @brief Index into the static cell-parameter table.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @return Always 0 (single mock cell).
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t cell_index) const noexcept
    {
        static_cast<void>(cell_index);
        return 0u;
    }
    /// @brief Logical carrier id for a cell index.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @return The configured carrier_id_val_.
    [[nodiscard]] int32_t  carrier_id(uint32_t cell_index)        const noexcept
    {
        static_cast<void>(cell_index);
        return carrier_id_val_;
    }
    /// @brief PHY cell id for a cell index.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @return The configured phy_cell_id_val_ (the physical cell id the new
    ///         parser writes into phy_cell_index_list; mirrors the legacy
    ///         populator's cell_sub_cmd.cell source).
    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_index)       const noexcept
    {
        static_cast<void>(cell_index);
        return phy_cell_id_val_;
    }

    /// @brief Per-cell error accumulator.
    /// @param[in] cell_stat_prm_idx Static cell-parameter index (ignored).
    /// @return Reference to the owned (shared) error accumulator.
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t cell_stat_prm_idx) const noexcept
    {
        static_cast<void>(cell_stat_prm_idx);
        return limit_errors_;
    }
    /// @brief Group-level L1-limit error accumulator (required by UlModuleView; PUCCH-only).
    /// @return Reference to the owned (shared) group error accumulator.
    /// @note PUSCH parsing does not exercise this accessor; presence-only to satisfy the concept.
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }
    /// @brief Number of indication instances configured per slot.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @return Always 0.
    [[nodiscard]] uint8_t indication_instances_per_slot(uint32_t cell_index) const noexcept
    {
        static_cast<void>(cell_index);
        return 0u;
    }

    /// @brief FAPI error-indication sink (no-op in the mock).
    /// @param[in] cell_index  Logical cell index (ignored).
    /// @param[in] message_id  FAPI message id (ignored).
    /// @param[in] error_code  FAPI error code (ignored).
    /// @param[in] sfn         System frame number (ignored).
    /// @param[in] slot        Slot number (ignored).
    void send_fapi_error_indication(uint32_t               cell_index,
                                    scf_fapi_message_id_e  message_id,
                                    scf_fapi_error_codes_t error_code,
                                    uint16_t               sfn,
                                    uint16_t               slot) const noexcept
    {
        static_cast<void>(cell_index);
        static_cast<void>(message_id);
        static_cast<void>(error_code);
        static_cast<void>(sfn);
        static_cast<void>(slot);
    }

    /// @brief Whether SRS is enabled for this slot.
    /// @return Always false (PUSCH-focused mock).
    [[nodiscard]] bool srs_enabled() const noexcept { return false; }

    /// @brief Classify an SRS channel-estimate buffer.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @param[in] pdu        SRS PDU (ignored).
    /// @return Always Accept.
    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t cell_index, const scf_fapi_srs_pdu_t& pdu) const noexcept
    {
        static_cast<void>(cell_index);
        static_cast<void>(pdu);
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }

    /// @brief Per-cell BFW coefficient memory info.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @param[in] slot       Slot number (ignored).
    /// @return The configured bfw_coeff_mem_info_ pointer (may be null).
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t cell_index, uint8_t slot) const noexcept
    {
        static_cast<void>(cell_index);
        static_cast<void>(slot);
        return bfw_coeff_mem_info_;
    }

    /// @brief Runtime FH gate (C-plane created directly by other worker threads).
    /// @return Always false; this mock exercises the parser's FH-fill path.
    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept { return false; }

    // PrachModuleView refinement stubs — required to satisfy the concept via
    // ULSlotProcessor's PrachPduParser; never invoked by the PUSCH test path.

    /// @brief Per-cell PHY config (PRACH parser dependency).
    /// @param[in] cell_id Logical cell id (ignored).
    /// @return Reference to the owned phy_config stub.
    [[nodiscard]] const nv::phy_config& phy_config(uint32_t cell_id) const noexcept
    {
        static_cast<void>(cell_id);
        return phy_config_obj_;
    }
    /// @brief Per-cell additional PRACH config (PRACH parser dependency).
    /// @param[in] cell_id Logical cell id (ignored).
    /// @return Reference to the owned prach_addln_config stub.
    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(uint32_t cell_id) const noexcept
    {
        static_cast<void>(cell_id);
        return prach_addln_config_obj_;
    }
    /// @brief Per-cell RU type resolved from M-plane config (PRACH parser dependency).
    /// @param[in] cell_id Logical cell id (ignored).
    /// @return The configured ru_type_.
    [[nodiscard]] ::ru_type ru_type_for_cell(uint32_t cell_id) const noexcept
    {
        static_cast<void>(cell_id);
        return ru_type_;
    }
    /// @brief PRACH FH-direct gate.
    /// @return Always false; this mock exercises the parser's FH-fill path.
    [[nodiscard]] bool is_fapi_to_cplane_direct() const noexcept { return false; }

    /// @brief RU mode for a cell index.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @return The configured ru_type_.
    [[nodiscard]] ::ru_type ru(uint32_t cell_index) const noexcept
    {
        static_cast<void>(cell_index);
        return ru_type_;
    }

    /// @brief Cell view for a cell index / slot indication.
    /// @param[in] cell_index Logical cell index (ignored).
    /// @param[in] slot_ind   Slot indication (ignored).
    /// @return A copy of the owned MockCellView.
    [[nodiscard]] MockCellView cell_view(uint32_t                                cell_index,
                                         const slot_command_api::slot_indication& slot_ind) const noexcept
    {
        static_cast<void>(cell_index);
        static_cast<void>(slot_ind);
        return cell_view_;
    }
};

static_assert(scf_5g_fapi::UlModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy UlModuleView");
static_assert(scf_5g_fapi::PuschModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy PuschModuleView");

} // namespace cuphy_cp::tests

#endif // CUPHY_CP_TESTS_UL_MODULE_VIEW_MOCK_HPP_
