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

/**
 * @file test_prach_module_view.cpp
 * @brief Concept-satisfaction tests for the new PrachModuleView (MR 2b).
 *
 * MR 2b of the PRACH parser stack (GT-11843). Verifies:
 *
 *   1. The narrower @c scf_5g_fapi::PrachModuleView concept compiles and is
 *      satisfied by a minimal in-test mock (proves the concept's requirements
 *      are reachable without pulling in the full @c PhyModuleView dependency
 *      graph).
 *
 *   2. The mock returns settable per-cell values for each of the five
 *      PRACH-only accessors — a regression guard against future ISP leaks
 *      where one of the methods accidentally hardcodes a default.
 *
 * MR 3a/3b/4 reuse @c MockPrachModuleView via inheritance from the
 * @c MockUlModuleView introduced in MR 1's test_prach_pdu_parser.cpp.
 * In this MR (a parallel sibling of MR 1) we define a self-contained
 * mock so the concept can be exercised independently.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_fapi_msg_common.hpp"
#include "nv_phy_limit_errors.hpp"
#include "scf_5g_fapi.h"

namespace
{

// MockCellView — same shape as MR 1's; satisfies scf_5g_fapi::CellView.
struct MockCellView final
{
    uint16_t           bwp_size_{59};
    cuphyCellStatPrm_t stat_prm_{};

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return bwp_size_; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }
};

static_assert(scf_5g_fapi::CellView<MockCellView>,
              "MockCellView must satisfy CellView");

// MockPrachModuleView — satisfies scf_5g_fapi::PrachModuleView (which itself
// refines UlModuleView).
struct MockPrachModuleView final
{
    // UlModuleView base members
    bool    bf_enabled_      {false};
    bool    mmimo_enabled_   {false};
    int     static_pusch_slot_{-1};
    uint8_t lbrm_            {0};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable nv::phy_config_option              config_opt_{};
    mutable nv::pucch_dtx_t_list               dtx_{};
    mutable float                              dtx_pusch_{0.0f};
    // See the matching comment in test_prach_pdu_parser.cpp (MR 1): the
    // universal UlModuleView concept's transport() requirement is an SRS-only
    // ISP leak; this null pointer satisfies the signature without forcing a
    // real NVIPC interface. MR 4 switches PrachPduParser to PrachModuleView
    // (this concept) — at that point this member can be dropped from any
    // PRACH-specific mock that no longer needs to satisfy UlModuleView's
    // transport() requirement.
    mutable nv::phy_mac_transport*             transport_ptr_{nullptr};
    MockCellView                               cell_view_obj_{};

    // PRACH-only state (new in MR 2b)
    mutable nv::phy_config                     phy_config_obj_{};
    mutable nv::prach_addln_config_t           addln_config_obj_{};
    ru_type                                    ru_type_value_{SINGLE_SECT_MODE};
    mutable nv::slot_limit_cell_error_t        limit_errors_{};
    mutable nv::slot_limit_group_error_t       group_limit_errors_{};
    bool                                       is_fapi_to_cplane_direct_value_{false};

    // ---- UlModuleView interface ------------------------------------------
    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }
    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }
    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t) const noexcept
    {
        return cell_sub_cmd_.sym_prb_info();
    }
    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept { return slot_cmd_; }
    [[nodiscard]] bool bf_enabled()    const noexcept { return bf_enabled_; }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int     staticPuschSlotNum() const noexcept { return static_pusch_slot_; }
    [[nodiscard]] uint8_t lbrm()               const noexcept { return lbrm_; }
    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds()       const noexcept { return dtx_; }
    [[nodiscard]] const float&                dtx_thresholds_pusch() const noexcept { return dtx_pusch_; }
    [[nodiscard]] nv::phy_mac_transport& transport(int) const noexcept { return *transport_ptr_; }
    [[nodiscard]] MockCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_obj_;
    }

    // ---- PrachModuleView additional accessors (new in MR 2b) -------------
    [[nodiscard]] const nv::phy_config& phy_config(uint32_t) const noexcept
    {
        return phy_config_obj_;
    }
    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(uint32_t) const noexcept
    {
        return addln_config_obj_;
    }
    [[nodiscard]] ru_type ru_type_for_cell(uint32_t) const noexcept { return ru_type_value_; }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    [[nodiscard]] bool is_fapi_to_cplane_direct() const noexcept { return is_fapi_to_cplane_direct_value_; }

    // ---- SRS-side UlModuleView stubs (expanded by MR1 of GT-12419) ----------
    // PrachPduParser never invokes these; present so the UlModuleView concept
    // static_assert below holds.
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t  carrier_id(uint32_t)        const noexcept { return 0; }
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t)       const noexcept { return 0u; }
    [[nodiscard]] uint8_t  indication_instances_per_slot(uint32_t) const noexcept { return 0u; }

    void send_fapi_error_indication(uint32_t,
                                    scf_fapi_message_id_e,
                                    scf_fapi_error_codes_t,
                                    uint16_t,
                                    uint16_t) const noexcept {}

    [[nodiscard]] bool    srs_enabled()        const noexcept { return false; }
    [[nodiscard]] ru_type ru(uint32_t cell_id) const noexcept { return ru_type_for_cell(cell_id); }

    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t, const scf_fapi_srs_pdu_t&) const noexcept
    {
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }

    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept
    {
        return is_fapi_to_cplane_direct();
    }
};

// Compile-time concept satisfaction — this is the primary "test" of MR 2b.
static_assert(scf_5g_fapi::UlModuleView<MockPrachModuleView>,
              "MockPrachModuleView must satisfy UlModuleView (PrachModuleView refines it)");
static_assert(scf_5g_fapi::PrachModuleView<MockPrachModuleView>,
              "MockPrachModuleView must satisfy PrachModuleView");

} // namespace

// ---------------------------------------------------------------------------
// Behavioural sanity tests — each accessor returns its configured value.
//
// These are not "real" tests of behaviour; they catch the regression where a
// future change accidentally hardcodes a return value (e.g. always returning
// false from is_fapi_to_cplane_direct() regardless of the mock's flag).
// ---------------------------------------------------------------------------

TEST(MockPrachModuleView, FapiToCplaneDirectReflectsConfiguredFlag)
{
    MockPrachModuleView v;
    EXPECT_FALSE(v.is_fapi_to_cplane_direct());      // default

    v.is_fapi_to_cplane_direct_value_ = true;
    EXPECT_TRUE(v.is_fapi_to_cplane_direct());
}

TEST(MockPrachModuleView, RuTypeReflectsConfiguredValue)
{
    MockPrachModuleView v;
    EXPECT_EQ(v.ru_type_for_cell(/*cell_id=*/0u), SINGLE_SECT_MODE);

    v.ru_type_value_ = OTHER_MODE;
    EXPECT_EQ(v.ru_type_for_cell(/*cell_id=*/0u), OTHER_MODE);
}

TEST(MockPrachModuleView, PhyConfigAndAddlnConfigReturnRefsToOwnedState)
{
    MockPrachModuleView v;
    // Mutate via the public member, observe via the const accessor —
    // proves the accessor returns a reference into our owned state, not a copy.
    constexpr uint8_t k_n_ra_dur = 5u;
    v.addln_config_obj_.n_ra_dur = k_n_ra_dur;
    EXPECT_EQ(v.prach_addln_config(/*cell_id=*/0u).n_ra_dur, k_n_ra_dur);

    constexpr uint8_t k_prach_scs = 1u;
    v.phy_config_obj_.prach_config_.prach_scs = k_prach_scs;
    EXPECT_EQ(v.phy_config(/*cell_id=*/0u).prach_config_.prach_scs, k_prach_scs);
}

TEST(MockPrachModuleView, GetCellLimitErrorsReturnsAliasToOwnedState)
{
    MockPrachModuleView v;
    constexpr uint16_t k_pci = 42u;
    auto& errs = v.get_cell_limit_errors(k_pci);
    EXPECT_EQ(&errs, &v.limit_errors_)
        << "Accessor must alias the owned slot_limit_cell_error_t, not return a copy";
}
