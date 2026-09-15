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
 * @file test_prach_pdu_parser.cpp
 * @brief Unit tests for PrachPduParser and the PRACH slot-builder pipeline.
 *
 * Contents:
 *   - MockCellView          — satisfies scf_5g_fapi::CellView
 *   - MockUlModuleView      — satisfies scf_5g_fapi::UlModuleView
 *   - PrachPduParams +
 *     make_prach_pdu()      — table-driven test parameter struct with
 *                             sensible defaults
 *   - Test groups exercising the parser stub, the compute_prach_params pure
 *     core (guards, field plumbing, multi-occasion symbol math, format LUT,
 *     mmimo gate), the always-on imperative shell (rach_params), and the
 *     PRACH order-metadata shell (sym_prb_info).
 *
 * Design notes:
 *   - All mocks live in an anonymous namespace.
 *   - Test inputs are raw integers; NamedType wrapping is confined to the
 *     DOP API boundary (PrachOccasionEntry), not test scope.
 *   - Naming follows the local test-file convention (snake_case for members),
 *     matching test_pdsch_pdu_parser.cpp.
 */

#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "scf_5g_fapi_prach_pdu_parser.hpp"
#include "scf_5g_fapi_prach_slot_builder.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"        // nv::slot_limit_cell_error_t (mock field)
#include "scf_5g_fapi.h"

namespace
{

// ---------------------------------------------------------------------------
// MockCellView — satisfies scf_5g_fapi::CellView
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MockUlModuleView — satisfies scf_5g_fapi::UlModuleView
//
// Concept requires: group_command(), cell_sub_command(cell_idx), slot_command(),
//                   mmimo_enabled(), bf_enabled(), config_options(),
//                   staticPuschSlotNum(), lbrm(), dtx_thresholds(),
//                   dtx_thresholds_pusch(), transport(cell_id_int),
//                   cell_view(cell_id, slot_ind).
// ---------------------------------------------------------------------------

struct MockUlModuleView final
{
    // Configurable flags — defaults are deliberately minimal so each test can
    // toggle only what it exercises.
    bool    bf_enabled_      {false};
    bool    mmimo_enabled_   {false};
    int     static_pusch_slot_{-1};
    uint8_t lbrm_            {0};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable nv::phy_config_option              config_opt_{};
    // dtx_ / dtx_pusch_ are returned by const-ref — no `mutable` needed.
    nv::pucch_dtx_t_list                       dtx_{};
    float                                      dtx_pusch_{0.0f};
    mutable nv::slot_limit_group_error_t       group_limit_errors_{};
    // ISP workaround: the universal UlModuleView concept in
    // scf_5g_fapi_slot_concepts.hpp:102-127 requires transport(int) returning
    // nv::phy_mac_transport& — but only the SRS parser actually calls it.
    // PRACH/PUSCH/PUCCH don't. Since today's PrachPduParser is declared
    // `template<UlModuleView V>`, the mock must satisfy the full concept just to
    // let us instantiate the parser for the stub-sanity test.
    //
    // The transport() accessor below ADD_FAILUREs + std::aborts if called —
    // dereferencing a null pointer to satisfy the return type is UB (LTO can
    // miscompile around it). A narrower PrachModuleView concept (without
    // transport()) exists; once the parser switches to that concept, this
    // accessor can be deleted entirely.
    MockCellView                               cell_view_{};

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

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] bool bf_enabled()    const noexcept { return bf_enabled_; }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }

    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }

    [[nodiscard]] int     staticPuschSlotNum() const noexcept { return static_pusch_slot_; }
    [[nodiscard]] uint8_t lbrm()               const noexcept { return lbrm_; }
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t  carrier_id(uint32_t)        const noexcept { return 0; }

    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds()       const noexcept { return dtx_; }
    [[nodiscard]] const float&                dtx_thresholds_pusch() const noexcept { return dtx_pusch_; }
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }

    [[nodiscard]] nv::phy_mac_transport& transport(int) const noexcept
    {
        // *nullptr would be UB even when unused (LTO/-O3 can miscompile the
        // surrounding code). Hard-fail instead so any future SRS-path test
        // that wrongly relies on the universal UlModuleView path gets a
        // clear error rather than a silent miscompile.
        ADD_FAILURE() << "PRACH MockUlModuleView::transport() must not be called "
                         "(SRS-only accessor; PRACH parser never invokes it)";
        std::abort(); // [[noreturn]] — satisfies the non-void return type
    }

    [[nodiscard]] MockCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }

    // ---- UlModuleView concept members added by MR1 -----------------------
    // Same ISP workaround as transport() above: the expanded UlModuleView
    // concept (scf_5g_fapi_slot_concepts.hpp) now requires SRS-side accessors,
    // but PrachPduParser never invokes them. Stub with safe defaults so the
    // concept check passes; the narrower PrachModuleView concept (planned for
    // MR 2b) will let these go away entirely.

    [[nodiscard]] uint16_t phy_cell_id(uint32_t)       const noexcept { return 0u; }
    [[nodiscard]] uint8_t  indication_instances_per_slot(uint32_t) const noexcept { return 0u; }

    void send_fapi_error_indication(uint32_t,
                                    scf_fapi_message_id_e,
                                    scf_fapi_error_codes_t,
                                    uint16_t,
                                    uint16_t) const noexcept {}

    [[nodiscard]] bool    srs_enabled() const noexcept { return false; }
    [[nodiscard]] ru_type ru(uint32_t)  const noexcept { return OTHER_MODE; }

    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t, const scf_fapi_srs_pdu_t&) const noexcept
    {
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }

    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept { return false; }

    // ---- PrachModuleView refinement --------------------------------------
    // PrachPduParser is templated on the refined PrachModuleView concept.
    // To keep the existing Group A–E tests working with the local
    // MockUlModuleView, we add the 5 PrachModuleView-specific accessors
    // here. Each returns a configurable owned member so tests can wire up
    // the parser-under-test against well-defined inputs without standing up
    // the separate MockPrachModuleView in test_prach_module_view.cpp.

    nv::phy_config                            phy_config_obj_{};
    nv::prach_addln_config_t                  addln_config_obj_{};
    // mutable required: get_cell_limit_errors returns a non-const reference
    // from a const view per the PrachModuleView concept.
    mutable nv::slot_limit_cell_error_t       cell_limit_errors_{};
    ru_type                                   ru_type_value_{OTHER_MODE};
    bool                                      is_fapi_to_cplane_direct_value_{false};
    // Records the cell_id observed by the most recent module-view lookup,
    // so tests can prove setup_cell()'s value reaches parse()'s lookups.
    mutable uint32_t                          last_lookup_cell_id_{UINT32_MAX};

    [[nodiscard]] const nv::phy_config& phy_config(const uint32_t cell_id) const noexcept
    {
        last_lookup_cell_id_ = cell_id;
        return phy_config_obj_;
    }

    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(const uint32_t cell_id) const noexcept
    {
        last_lookup_cell_id_ = cell_id;
        return addln_config_obj_;
    }

    [[nodiscard]] ru_type ru_type_for_cell(const uint32_t cell_id) const noexcept
    {
        last_lookup_cell_id_ = cell_id;
        return ru_type_value_;
    }

    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(const uint16_t) const noexcept
    {
        return cell_limit_errors_;
    }

    [[nodiscard]] bool is_fapi_to_cplane_direct() const noexcept
    {
        return is_fapi_to_cplane_direct_value_;
    }
};

static_assert(scf_5g_fapi::UlModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy UlModuleView");
static_assert(scf_5g_fapi::PrachModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy PrachModuleView");

// ---------------------------------------------------------------------------
// PrachPduParams — table-driven test parameter struct
//
// Defaults: single-occasion, format 0, no beamforming. Override only the
// fields under test. Cell-config knobs live alongside the PDU fields so a
// single PrachPduParams instance fully describes a test case.
// ---------------------------------------------------------------------------

/**
 * Test parameter container for one PRACH PDU and its cell-config knobs.
 *
 * Used across all test groups; defined here once so the scaffolding is
 * complete and downstream MRs only need to add cases, not infrastructure.
 */
struct PrachPduParams
{
    // FAPI PDU fields (mirrors scf_fapi_prach_pdu_t)
    uint16_t phys_cell_id        {1};   //!< [0:1007]
    uint8_t  num_prach_ocas      {1};   //!< [1:7] time-domain occasions
    uint8_t  prach_format        {0};   //!< [0:13]
    uint8_t  num_ra              {0};   //!< [0:7] frequency-domain index
    uint8_t  prach_start_symbol  {0};   //!< [0:13]
    uint16_t num_cs              {0};   //!< [0:419]
    uint8_t  dig_bf_interfaces   {0};   //!< Beam-forming digital interfaces

    // Cell-config knobs (used to populate the cell view's phy_config /
    // prach_addln_config when tests exercise the real compute pipeline).
    uint16_t start_ro_index      {0};
    uint8_t  prach_scs           {1};
    uint8_t  prach_seq_length    {1};
    uint16_t k1                  {0};
    uint8_t  n_ra_dur            {1};
    uint8_t  n_ra_slot           {0};
    uint16_t n_ra_rb             {6};
};

/**
 * Serialise a PrachPduParams into a fully-initialised scf_fapi_prach_pdu_t.
 *
 * Cell-config fields of PrachPduParams are intentionally NOT consumed here —
 * they belong to the cell view, not the PDU. Use them when populating
 * MockPrachModuleView::phy_config_ / prach_addln_config_.
 *
 * @param[in]  p  Test parameters.
 * @return  Fully-initialised PRACH PDU.
 *          Return value must be checked (no error mode today; included for
 *          forward compatibility).
 */
[[nodiscard]] scf_fapi_prach_pdu_t make_prach_pdu(const PrachPduParams& p) noexcept
{
    // Designated initializers for the top-level aggregate (CLAUDE.md §1.11).
    // The single nested-struct field (beam_index.dig_bf_interfaces) is set
    // after the aggregate brace-init to avoid coupling this test to the
    // declaration order of scf_fapi_rx_beamforming_t's members.
    scf_fapi_prach_pdu_t pdu{
        .phys_cell_id       = p.phys_cell_id,
        .num_prach_ocas     = p.num_prach_ocas,
        .prach_format       = p.prach_format,
        .num_ra             = p.num_ra,
        .prach_start_symbol = p.prach_start_symbol,
        .num_cs             = p.num_cs,
    };
    pdu.beam_index.dig_bf_interfaces = p.dig_bf_interfaces;
    return pdu;
}

} // namespace

// Forward declarations for the cell-config helpers defined later in the file's
// anonymous namespace — Group A tests need to populate the mock view's
// addln/phy_config before driving parse() so the cell-config knobs on
// PrachPduParams actually reach compute_prach_params.
namespace
{
[[nodiscard]] inline nv::phy_config           make_phy_config(const PrachPduParams& p) noexcept;
[[nodiscard]] inline nv::prach_addln_config_t make_addln(const PrachPduParams& p) noexcept;
} // namespace

// ---------------------------------------------------------------------------
// Group A — End-to-end PrachPduParser::parse
//
// Drives the real compute -> apply_rach -> apply_fh/order metadata pipeline. The
// cases below verify the parser successfully drives the pipeline for the
// canonical happy path, the fast-out path, and the guard-failure path.
// ---------------------------------------------------------------------------

TEST(PrachPduParser, HappyPathBumpsOccasionAndTagsUplink)
{
    MockUlModuleView v;
    v.is_fapi_to_cplane_direct_value_ = true;
    scf_5g_fapi::PrachPduParser parser{v};

    const PrachPduParams p{};
    const auto pdu = make_prach_pdu(p);

    EXPECT_TRUE(parser.parse(/*sfn=*/12u, /*slot=*/4u, pdu));

    auto* const params = v.slot_cmd_.cell_groups.get_prach_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->nOccasion, 1u);

    // Both cell.slot and group.slot transitioned to (SLOT_UPLINK, slot_ind).
    EXPECT_EQ(v.slot_cmd_.cell_groups.slot.type, slot_command_api::SLOT_UPLINK);
    EXPECT_EQ(v.slot_cmd_.cell_groups.slot.slot_3gpp.sfn_,  12u);
    EXPECT_EQ(v.slot_cmd_.cell_groups.slot.slot_3gpp.slot_, 4u);
    EXPECT_EQ(v.cell_sub_cmd_.slot.type, slot_command_api::SLOT_UPLINK);
}

TEST(PrachPduParser, NumPrachOcas0IsFastOutWithoutMutation)
{
    MockUlModuleView v;
    scf_5g_fapi::PrachPduParser parser{v};

    PrachPduParams p{};
    p.num_prach_ocas = 0u;
    const auto pdu = make_prach_pdu(p);

    EXPECT_TRUE(parser.parse(/*sfn=*/0u, /*slot=*/0u, pdu))
        << "Empty PDU must be a no-op true, matching legacy on_prach_pdu_info";

    auto* const params = v.slot_cmd_.cell_groups.get_prach_params();
    if (params != nullptr) {
        EXPECT_EQ(params->nOccasion, 0u)
            << "Fast-out must not advance nOccasion";
    }
}

TEST(PrachPduParser, GuardFailureReturnsFalseWithoutMutation)
{
    MockUlModuleView v;
    scf_5g_fapi::PrachPduParser parser{v};

    PrachPduParams p{};
    p.phys_cell_id =
        static_cast<uint16_t>(scf_5g_fapi::prach::k_max_phys_cell_id + 1u);
    const auto pdu = make_prach_pdu(p);

    EXPECT_FALSE(parser.parse(/*sfn=*/0u, /*slot=*/0u, pdu));

    auto* const params = v.slot_cmd_.cell_groups.get_prach_params();
    if (params != nullptr) {
        EXPECT_EQ(params->nOccasion, 0u);
    }
}

TEST(PrachPduParser, DirectModeFromViewPopulatesOrderMetadata)
{
    MockUlModuleView v;
    v.is_fapi_to_cplane_direct_value_ = true;
    scf_5g_fapi::PrachPduParser parser{v};

    PrachPduParams p{};
    p.num_prach_ocas = 2u;
    const auto pdu = make_prach_pdu(p);

    EXPECT_TRUE(parser.parse(0u, 0u, pdu));

    EXPECT_EQ(v.cell_sub_cmd_.sym_prb_info()->prbs_size, 2u)
        << "Direct mode must preserve PRACH sym_prb_info for the UL Order kernel";
}

TEST(PrachPduParser, LegacyModeFromViewPopulatesFh)
{
    MockUlModuleView v;
    v.is_fapi_to_cplane_direct_value_ = false;
    scf_5g_fapi::PrachPduParser parser{v};

    PrachPduParams p{};
    p.num_prach_ocas = 2u;
    p.n_ra_dur       = 2u;
    p.n_ra_rb        = 6u;
    // Cell-config knobs live on the view, not the PDU — populate before parse.
    v.phy_config_obj_   = make_phy_config(p);
    v.addln_config_obj_ = make_addln(p);
    const auto pdu = make_prach_pdu(p);

    EXPECT_TRUE(parser.parse(0u, 0u, pdu));

    EXPECT_EQ(v.cell_sub_cmd_.sym_prb_info()->prbs_size, 2u)
        << "Legacy mode must append one sym_prb entry per occasion";
}

TEST(PrachPduParser, SetupCellBindsLogicalCarrierIndexToLookups)
{
    // Regression guard: parse() must resolve module-view lookups via the
    // logical carrier index stashed by setup_cell(), NOT via pdu.phys_cell_id
    // (which is the 3GPP PCI 0-1007 and is unsuitable for array indexing).
    MockUlModuleView v;
    v.is_fapi_to_cplane_direct_value_ = true;
    scf_5g_fapi::PrachPduParser parser{v};

    constexpr uint32_t k_logical_cell_id = 2u;
    parser.setup_cell(k_logical_cell_id);

    PrachPduParams p{};
    p.phys_cell_id = 137u;                       // PCI distinct from logical idx
    v.phy_config_obj_   = make_phy_config(p);
    v.addln_config_obj_ = make_addln(p);
    const auto pdu = make_prach_pdu(p);

    EXPECT_TRUE(parser.parse(/*sfn=*/0u, /*slot=*/0u, pdu));
    EXPECT_EQ(v.last_lookup_cell_id_, k_logical_cell_id)
        << "module-view lookups must use the setup_cell value, not pdu.phys_cell_id";
}

// ---------------------------------------------------------------------------
// L1-limit validation is deferred (TODO in parser body).
// When re-enabled, add a test here that exercises the path:
//
//   * Construct a PDU that triggers validate_prach_pdu_l1_limits → INVALID_FAPI_PDU
//   * EXPECT_FALSE(parser.parse(...))
//   * EXPECT_NE(v.cell_limit_errors_.error_mask & SCF_FAPI_PRACH_L1_LIMIT_EXCEEDED, 0u)
//   * Assert prach_params untouched
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers — build the cell-config inputs to compute_prach_params
//
// These translate the PrachPduParams cell-config knobs (start_ro_index,
// prach_scs, prach_seq_length, k1, n_ra_dur, n_ra_slot, n_ra_rb) into
// fully-initialised nv::phy_config / nv::prach_addln_config_t structs.
// ---------------------------------------------------------------------------

namespace
{

/**
 * @brief  Build an @c nv::phy_config matching the test's PrachPduParams cell-config knobs.
 *
 * Populates @c cell_config_.phy_cell_id and the @c prach_config_ scalars that
 * @c compute_prach_params reads. The runtime-indexed @c root_sequence[num_ra]
 * write is guarded against out-of-range @c num_ra so guard tests (which
 * intentionally set @c num_ra past the bound) don't trigger UB.
 *
 * @param[in] p  Test parameters; @c phys_cell_id / @c start_ro_index /
 *               @c prach_scs / @c prach_seq_length / @c num_ra / @c k1 are read.
 * @return  Fully-populated @c nv::phy_config.
 */
[[nodiscard]] nv::phy_config make_phy_config(const PrachPduParams& p) noexcept
{
    // nv::phy_config is non-aggregate — designated init unavailable, so fall
    // back to value-init followed by per-field assignment.
    nv::phy_config cfg{};
    cfg.cell_config_.phy_cell_id           = p.phys_cell_id;
    cfg.prach_config_.start_ro_index       = p.start_ro_index;
    cfg.prach_config_.prach_scs            = p.prach_scs;
    cfg.prach_config_.prach_seq_length     = p.prach_seq_length;
    // freqOffset for the configured frequency-domain slot. Test cases that
    // care about freq_offset set k1 explicitly; default is 0. Guard against
    // OutOfRangeNumRa guard tests that intentionally set num_ra past the
    // root_sequence[] bound — the parser rejects the PDU before reading the
    // freqOffset, but the helper still runs at setup so the write must be
    // skipped to avoid UB.
    if (p.num_ra < nv::NV_MAX_PRACH_FD_OCCASION_NUM) {
        cfg.prach_config_.root_sequence[p.num_ra].freqOffset = p.k1;
    }
    return cfg;
}

/**
 * @brief  Build an @c nv::prach_addln_config_t mirroring the test's n_ra_* knobs.
 *
 * Designated init in declaration order (n_ra_slot, n_ra_dur, n_ra_rb) per the
 * struct's layout in @c nv_phy_fapi_msg_common.hpp.
 *
 * @param[in] p  Test parameters; @c n_ra_slot / @c n_ra_dur / @c n_ra_rb are read.
 * @return  Fully-populated @c nv::prach_addln_config_t.
 */
[[nodiscard]] nv::prach_addln_config_t make_addln(const PrachPduParams& p) noexcept
{
    // PrachPduParams::n_ra_rb is uint16_t; nv::prach_addln_config_t::n_ra_rb
    // is uint8_t. Guard against silent truncation if a future test passes
    // a wider value. assert is debug-only and never throws — noexcept stays
    // valid.
    assert(p.n_ra_rb <= std::numeric_limits<uint8_t>::max()
           && "make_addln: PrachPduParams::n_ra_rb exceeds uint8_t — narrow before passing");
    return nv::prach_addln_config_t{
        .n_ra_slot = p.n_ra_slot,
        .n_ra_dur  = p.n_ra_dur,
        .n_ra_rb   = static_cast<uint8_t>(p.n_ra_rb),
    };
}

} // namespace

// ---------------------------------------------------------------------------
// Group B — Validation guards
//
// Table of out-of-range / unsupported inputs and the PrachError each must
// produce. compute_prach_params returns tl::unexpected{...}; we assert the
// specific enumerator so a guard that gets silently disabled becomes loud.
// ---------------------------------------------------------------------------

namespace
{
struct GuardCase final
{
    const char*                             name           {};
    PrachPduParams                          params         {};
    scf_5g_fapi::prach::PrachError          expected_error {};
};

constexpr std::array<GuardCase, 7> k_guard_cases{{
    {"ZeroNumPrachOcas",
     {.num_prach_ocas = 0u},
     scf_5g_fapi::prach::PrachError::ZeroNumPrachOcas},

    {"OutOfRangeNumPrachOcas",
     {.num_prach_ocas =
          static_cast<uint8_t>(scf_5g_fapi::prach::k_max_num_prach_ocas + 1u)},
     scf_5g_fapi::prach::PrachError::OutOfRangeNumPrachOcas},

    {"OutOfRangePhysCellId",
     {.phys_cell_id =
          static_cast<uint16_t>(scf_5g_fapi::prach::k_max_phys_cell_id + 1u)},
     scf_5g_fapi::prach::PrachError::OutOfRangePhysCellId},

    {"OutOfRangePrachFormat",
     {.prach_format =
          static_cast<uint8_t>(scf_5g_fapi::prach::k_max_prach_format + 1u)},
     scf_5g_fapi::prach::PrachError::OutOfRangePrachFormat},

    {"OutOfRangeNumRa",
     {.num_ra =
          static_cast<uint8_t>(scf_5g_fapi::prach::k_max_num_ra + 1u)},
     scf_5g_fapi::prach::PrachError::OutOfRangeNumRa},

    {"OutOfRangePrachStartSymbol_AtBoundary",
     {.prach_start_symbol = scf_5g_fapi::prach::k_ofdm_symbols_per_slot},
     scf_5g_fapi::prach::PrachError::OutOfRangePrachStartSymbol},

    {"OutOfRangePrachStartSymbol_FarPastBoundary",
     {.prach_start_symbol = 200u},
     scf_5g_fapi::prach::PrachError::OutOfRangePrachStartSymbol},
}};
} // namespace

TEST(ComputePrachParamsGuards, ReturnsExpectedPrachError)
{
    for (const auto& tc : k_guard_cases)
    {
        SCOPED_TRACE(tc.name);
        const auto pdu   = make_prach_pdu(tc.params);
        const auto phy   = make_phy_config(tc.params);
        const auto addln = make_addln(tc.params);

        const auto result =
            scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln,
                                                     /*mmimo_enabled=*/false);
        ASSERT_FALSE(result.has_value())
            << "Guard must reject the PDU before populating the result";
        EXPECT_EQ(result.error(), tc.expected_error);
    }
}

// ---------------------------------------------------------------------------
// Group C — Happy-path field plumbing
//
// One canonical PDU; assert every PrachComputed field carries the value the
// legacy update_cell_command(PRACH) overload at scf_5g_slot_commands.cpp:550
// would have written. This is the field-by-field parity check.
// ---------------------------------------------------------------------------

TEST(ComputePrachParamsHappyPath, CanonicalPduYieldsExpectedFields)
{
    const PrachPduParams p{
        .phys_cell_id       = 42u,
        .num_prach_ocas     = 2u,
        .prach_format       = 0u,
        .num_ra             = 3u,
        .prach_start_symbol = 4u,
        .start_ro_index     = 10u,
        .prach_scs          = 1u,           // long format → PRACH_LONG_FORMAT_FFT
        .prach_seq_length   = 0u,           // 0 → long format
        .k1                 = 7,
        .n_ra_dur           = 2u,
        .n_ra_slot          = 0u,
        .n_ra_rb            = 6u,
    };

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto result = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln,
                                                                 /*mmimo_enabled=*/false);
    ASSERT_TRUE(result.has_value());
    const auto& c = *result;

    EXPECT_EQ(c.num_occasions,     2u);
    EXPECT_EQ(c.freq_index,        3u);                          // pdu.num_ra
    EXPECT_EQ(c.phy_cell_id,       42u);                         // pdu.phys_cell_id
    EXPECT_EQ(c.occa_prm_stat_idx, 10u + 3u);                    // start_ro_index + num_ra
    EXPECT_EQ(c.freq_offset,       7);                           // root_sequence[num_ra].freqOffset
    EXPECT_EQ(c.mu,                1u);                          // prach_scs
    EXPECT_EQ(c.nfft,              nv::PRACH_LONG_FORMAT_FFT);   // prach_seq_length != 1
    EXPECT_EQ(c.n_ra_dur,          2u);
    EXPECT_EQ(c.n_ra_rb,           6u);
    EXPECT_EQ(c.n_uplink_streams,  0u);                          // mmimo_enabled == false
    // Filter index for format 0 is 1 per the LUT.
    EXPECT_EQ(c.filter_indices[0], 1u);
    EXPECT_EQ(c.filter_indices[1], 1u);
    // start_symbol[i] = (l0 + n_ra_dur*i + 14*n_ra_slot) % 14
    EXPECT_EQ(c.start_symbols[0],  4u);                          // (4 + 0 + 0) % 14
    EXPECT_EQ(c.start_symbols[1],  6u);                          // (4 + 2 + 0) % 14
}

TEST(ComputePrachParamsHappyPath, ShortFormatSelectsShortFft)
{
    const PrachPduParams p{.prach_seq_length = 1u};              // 1 → short format

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto result = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->nfft, nv::PRACH_SHORT_FORMAT_FFT);
}

// ---------------------------------------------------------------------------
// Group D — Multi-occasion symbol math + format LUT
//
// Parametric over (num_prach_ocas, format). Asserts:
//   - start_symbols[i] follows (l0 + n_ra_dur*i + 14*n_ra_slot) % 14
//   - filter_indices[i] == k_prach_format_to_filter_index[format] for every i
// ---------------------------------------------------------------------------

TEST(ComputePrachParamsMultiOccasion, StartSymbolsFollowFormula)
{
    // Exercises the full FAPI 10.04 range [1:7]; locks in the wider valid
    // range so the prior 4-cap (when k_max_num_prach_ocas == 4) cannot
    // silently regress if the bound is ever lowered again.
    static constexpr std::array<uint8_t, 6> k_occasion_counts{1u, 2u, 4u, 5u, 6u, 7u};
    static constexpr uint8_t                k_l0           = 3u;
    static constexpr uint8_t                k_n_ra_dur     = 3u;
    static constexpr uint8_t                k_n_ra_slot    = 1u;

    for (const auto n_ocas : k_occasion_counts)
    {
        SCOPED_TRACE("num_prach_ocas=" + std::to_string(n_ocas));

        const PrachPduParams p{
            .num_prach_ocas     = n_ocas,
            .prach_start_symbol = k_l0,
            .n_ra_dur           = k_n_ra_dur,
            .n_ra_slot          = k_n_ra_slot,
        };

        const auto pdu   = make_prach_pdu(p);
        const auto phy   = make_phy_config(p);
        const auto addln = make_addln(p);

        const auto result =
            scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
        ASSERT_TRUE(result.has_value());

        for (uint8_t i = 0; i < n_ocas; ++i)
        {
            const uint32_t expected =
                (static_cast<uint32_t>(k_l0)
                 + static_cast<uint32_t>(k_n_ra_dur) * i
                 + static_cast<uint32_t>(scf_5g_fapi::prach::k_ofdm_symbols_per_slot)
                       * static_cast<uint32_t>(k_n_ra_slot))
                % scf_5g_fapi::prach::k_ofdm_symbols_per_slot;
            EXPECT_EQ(result->start_symbols[i], static_cast<uint8_t>(expected))
                << "occasion i=" << static_cast<int>(i);
        }
    }
}

TEST(ComputePrachParamsFormatLut, EveryFormatYieldsLutEntry)
{
    using scf_5g_fapi::prach::k_prach_format_to_filter_index;
    using scf_5g_fapi::prach::k_max_prach_format;

    for (uint8_t fmt = 0; fmt <= k_max_prach_format; ++fmt)
    {
        SCOPED_TRACE("prach_format=" + std::to_string(fmt));

        const PrachPduParams p{
            .num_prach_ocas = 1u,
            .prach_format   = fmt,
        };

        const auto pdu   = make_prach_pdu(p);
        const auto phy   = make_phy_config(p);
        const auto addln = make_addln(p);

        const auto result =
            scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->filter_indices[0], k_prach_format_to_filter_index[fmt]);
    }
}

// ---------------------------------------------------------------------------
// Group E (partial) — MMIMO / digital BF gate
//
// n_uplink_streams is set only when mmimo_enabled AND dig_bf_interfaces != 0.
// Other PrachComputed fields are unaffected by the gate.
// ---------------------------------------------------------------------------

TEST(ComputePrachParamsMmimoGate, UplinkStreamsSetOnlyWhenMmimoAndDigBf)
{
    const PrachPduParams p_off{.dig_bf_interfaces = 0u};   // dig_bf == 0
    const PrachPduParams p_on {.dig_bf_interfaces = 4u};

    const auto pdu_off   = make_prach_pdu(p_off);
    const auto pdu_on    = make_prach_pdu(p_on);
    const auto phy       = make_phy_config(p_off);
    const auto addln     = make_addln(p_off);

    // Case 1: mmimo=false, dig_bf=0  → 0
    {
        const auto r = scf_5g_fapi::prach::compute_prach_params(pdu_off, phy, addln, false);
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r->n_uplink_streams, 0u);
    }
    // Case 2: mmimo=true, dig_bf=0   → 0
    {
        const auto r = scf_5g_fapi::prach::compute_prach_params(pdu_off, phy, addln, true);
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r->n_uplink_streams, 0u);
    }
    // Case 3: mmimo=false, dig_bf=4  → 0 (gate requires mmimo)
    {
        const auto r = scf_5g_fapi::prach::compute_prach_params(pdu_on, phy, addln, false);
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r->n_uplink_streams, 0u);
    }
    // Case 4: mmimo=true, dig_bf=4   → 4 (gate open)
    {
        const auto r = scf_5g_fapi::prach::compute_prach_params(pdu_on, phy, addln, true);
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(r->n_uplink_streams, 4u);
    }
}

// ---------------------------------------------------------------------------
// apply_prach_to_slot_command + populate_slot_command
//
// End-to-end shell tests: validated PrachComputed in, slot_command state out.
// Verifies the DOP-based shell honours the cubb-review §12.1 rules:
//   * cell.slot and group.slot transition to (SLOT_UPLINK, slot_ind)
//   * rach[i] / freqIndex[i] / startSymbols[i] / cell_index_list /
//     phy_cell_index_list advance together (nOccasion++ owns the invariant)
//   * mu / nfft set as a pair
// sym_prb_info is left untouched — that's the FH shell's domain.
// ---------------------------------------------------------------------------

namespace
{
[[nodiscard]] inline slot_command_api::slot_indication make_slot_ind(
    const uint16_t sfn, const uint16_t slot) noexcept
{
    return slot_command_api::slot_indication{sfn, slot, /*tick=*/uint64_t{0}};
}
} // namespace

TEST(ApplyPrachToSlotCommand, HappyPathPopulatesRachAndSlotInfo)
{
    const PrachPduParams p{
        .phys_cell_id       = 7u,
        .num_prach_ocas     = 2u,
        .prach_format       = 0u,
        .num_ra             = 1u,
        .prach_start_symbol = 4u,
        .start_ro_index     = 100u,
        .prach_scs          = 1u,
        .prach_seq_length   = 1u,           // short format
        .k1                 = 13,
        .n_ra_dur           = 2u,
        .n_ra_rb            = 6u,
    };

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto computed = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
    ASSERT_TRUE(computed.has_value());

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(/*sfn=*/123u, /*slot=*/4u);

    const bool ok = scf_5g_fapi::prach::apply_prach_to_slot_command(
        *computed, group, cell, slot_ind, /*cell_index=*/5);
    ASSERT_TRUE(ok);

    // Slot type + 3GPP indication propagated atomically (dop::set_uplink).
    EXPECT_EQ(cell.slot.type,        slot_command_api::SLOT_UPLINK);
    EXPECT_EQ(group.slot.type,       slot_command_api::SLOT_UPLINK);
    EXPECT_EQ(cell.slot.slot_3gpp.sfn_,  slot_ind.sfn_);
    EXPECT_EQ(group.slot.slot_3gpp.sfn_, slot_ind.sfn_);

    // prach_params advanced by exactly one occasion (the rach[] entry is
    // per-PDU, not per-FH-occasion — matches legacy update_cell_command).
    auto* const params = group.get_prach_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->nOccasion, 1u);
    EXPECT_EQ(params->freqIndex[0],            computed->freq_index);
    EXPECT_EQ(params->startSymbols[0],         computed->start_symbols[0]);
    EXPECT_EQ(params->rach[0].occaPrmStatIdx,  computed->occa_prm_stat_idx);
    EXPECT_EQ(params->rach[0].occaPrmDynIdx,   0u);
    EXPECT_EQ(params->rach[0].force_thr0,      0.0f);
    EXPECT_EQ(params->mu,                       computed->mu);
    EXPECT_EQ(params->nfft,                     computed->nfft);
    EXPECT_EQ(params->cell_index_list.size(),  1u);
    EXPECT_EQ(params->cell_index_list[0],      5);
    EXPECT_EQ(params->phy_cell_index_list[0],  static_cast<int32_t>(computed->phy_cell_id));
}

TEST(ApplyPrachToSlotCommand, MmimoPropagatesUplinkStreams)
{
    const PrachPduParams p{.dig_bf_interfaces = 3u};

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto computed = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln,
                                                                   /*mmimo_enabled=*/true);
    ASSERT_TRUE(computed.has_value());

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    ASSERT_TRUE(scf_5g_fapi::prach::apply_prach_to_slot_command(
        *computed, group, cell, slot_ind, /*cell_index=*/0));

    auto* const params = group.get_prach_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->rach[0].nUplinkStreams, 3u);
}

TEST(ApplyPrachToSlotCommand, ReturnsFalseAtCapacity)
{
    // Pre-fill prach_params to MAX_PRACH_OCCASIONS_PER_SLOT so add_occasion refuses.
    const PrachPduParams p{};
    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto computed = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
    ASSERT_TRUE(computed.has_value());

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    auto* const params = group.get_prach_params();
    ASSERT_NE(params, nullptr);
    params->nOccasion = slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT;

    EXPECT_FALSE(scf_5g_fapi::prach::apply_prach_to_slot_command(
        *computed, group, cell, slot_ind, /*cell_index=*/0));
    EXPECT_EQ(params->nOccasion,
              static_cast<uint16_t>(slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT));

    // "false = no mutation" contract: capacity refusal must leave cell.slot
    // and group.slot untouched (NOT tagged SLOT_UPLINK). Regression guard for
    // the partial-side-effect bug fixed in apply_prach_to_slot_command.
    EXPECT_EQ(cell.slot.type,  slot_command_api::SLOT_NONE);
    EXPECT_EQ(group.slot.type, slot_command_api::SLOT_NONE);
}

// ---------------------------------------------------------------------------
// Group F (orchestrator) — populate_slot_command exercises compute → apply
// ---------------------------------------------------------------------------

TEST(PopulateSlotCommand, HappyPathProducesOneOccasion)
{
    const PrachPduParams p{
        .phys_cell_id       = 42u,
        .num_prach_ocas     = 1u,
        .num_ra             = 0u,
        .prach_start_symbol = 2u,
    };

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = false,
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = true,
    };

    ASSERT_TRUE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));

    auto* const params = group.get_prach_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->nOccasion, 1u);
}

TEST(PopulateSlotCommand, GuardFailurePropagatesAsFalse)
{
    // Invalid PDU: phys_cell_id out of range → compute returns PrachError,
    // populate_slot_command must return false without mutating prach_params.
    const PrachPduParams p{
        .phys_cell_id =
            static_cast<uint16_t>(scf_5g_fapi::prach::k_max_phys_cell_id + 1u),
    };

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = false,
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = true,
    };

    EXPECT_FALSE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));

    auto* const params = group.get_prach_params();
    if (params != nullptr) {
        EXPECT_EQ(params->nOccasion, 0u)
            << "Failed validation must not mutate prach_params";
    }
}

// ---------------------------------------------------------------------------
// MR 3c — apply_prach_fh_to_sym_prb_info + direct C-plane order metadata
//
// Group F-direct  : ctx.fapi_to_cplane_direct = true  → sym_prb_info appended
//                                                       once per occasion
// Group F-legacy  : ctx.fapi_to_cplane_direct = false → same sym_prb_info fill
// Group F-fh-fields: direct call to apply_prach_fh_to_sym_prb_info verifies
//                    per-occasion common fields (freqOffset, numSymbols,
//                    direction, filterIndex) plus the bf_enabled and
//                    mmimo portMask gates.
// ---------------------------------------------------------------------------

TEST(PopulateSlotCommand, DirectModePopulatesSymPrbInfoForOrderKernel)
{
    PrachPduParams p{};
    p.num_prach_ocas = 2u;

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = false,
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = true,
    };

    ASSERT_TRUE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));

    // rach_params advanced (always-on shell ran)
    auto* const params = group.get_prach_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->nOccasion, 1u);

    // Direct C-plane creates packets from FAPI, but cuphydriver's UL Order
    // kernel still derives PRACH reorder work from sym_prb_info.
    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    EXPECT_EQ(sym_prbs->prbs_size, 2u)
        << "Direct mode must keep per-occasion sym_prb_info for PRACH Order";
    EXPECT_FALSE(sym_prbs->symbols[0][slot_command_api::channel_type::PRACH].empty());
}

TEST(PopulateSlotCommand, LegacyModePopulatesSymPrbInfoPerOccasion)
{
    PrachPduParams p{};
    p.num_prach_ocas     = 3u;
    p.prach_format       = 0u;
    p.num_ra             = 2u;
    p.prach_start_symbol = 1u;
    p.n_ra_dur           = 2u;
    p.n_ra_rb            = 6u;
    p.k1                 = 42;

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = false,
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = false,
    };

    ASSERT_TRUE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));

    // One sym_prb entry per FH occasion (3 here)
    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    EXPECT_EQ(sym_prbs->prbs_size, 3u);

    // Spot-check common fields propagated from PrachComputed
    EXPECT_EQ(sym_prbs->prbs[0].common.numSymbols, 2u);
    EXPECT_EQ(sym_prbs->prbs[0].common.freqOffset, 42);
    EXPECT_EQ(sym_prbs->prbs[0].common.direction,
              slot_command_api::fh_dir_t::FH_DIR_UL);
    EXPECT_EQ(sym_prbs->prbs[0].common.filterIndex, 1u);  // format 0 → 1
}

TEST(PopulateSlotCommand, LegacyModeFhCapacityOverflowReturnsFalse)
{
    // Pre-fill sym_prb_info to MAX_PRB_INFO so the FH-shell's per-occasion
    // dop::full() check trips on the first iteration, returning false from
    // apply_prach_fh_to_sym_prb_info. populate_slot_command must propagate
    // the false through to its caller.
    const PrachPduParams p{.num_prach_ocas = 1u};

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    sym_prbs->prbs_size = MAX_PRB_INFO;

    const scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = false,
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = false,    // legacy mode → FH-shell runs
    };

    EXPECT_FALSE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));
    // FH-shell refuses without writing past the cap.
    EXPECT_EQ(sym_prbs->prbs_size,
              static_cast<std::size_t>(MAX_PRB_INFO));
}

TEST(PopulateSlotCommand, LegacyModeBfEnabledPopulatesSymPrbInfoPerOccasion)
{
    // Sibling of LegacyModePopulatesSymPrbInfoPerOccasion with bf_enabled=true.
    // The bf-side write goes through update_beam_list, which is stubbed to a
    // no-op in tests/test_stubs.cpp (the production impl pulls in cuphy/CUDA
    // which the unit-test target avoids). The observable signal therefore is
    // that the bf branch doesn't crash and still produces one sym_prb entry
    // per occasion with the common-field values intact — the same contract
    // the bf=false test asserts.
    const PrachPduParams p{
        .num_prach_ocas     = 2u,
        .num_ra             = 1u,
        .prach_start_symbol = 2u,
        .k1                 = 11,
        .n_ra_dur           = 2u,
        .n_ra_rb            = 6u,
    };

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    slot_command_api::cell_group_command group{};
    slot_command_api::cell_sub_command   cell{};
    const auto slot_ind = make_slot_ind(0u, 0u);

    const scf_5g_fapi::prach::BuildContext ctx{
        .group                  = group,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = nullptr,
        .cell_index             = 0,
        .ru                     = OTHER_MODE,
        .bf_enabled             = true,     // exercise the bf branch
        .mmimo_enabled          = false,
        .fapi_to_cplane_direct  = false,
    };

    ASSERT_TRUE(scf_5g_fapi::prach::populate_slot_command(ctx, pdu));

    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    EXPECT_EQ(sym_prbs->prbs_size, 2u);

    EXPECT_EQ(sym_prbs->prbs[0].common.numSymbols, 2u);
    EXPECT_EQ(sym_prbs->prbs[0].common.freqOffset, 11);
    EXPECT_EQ(sym_prbs->prbs[0].common.filterIndex, 1u);
}

TEST(ApplyPrachFhToSymPrbInfo, MmimoSetsPortMaskWhenDigBfNonZero)
{
    PrachPduParams p{};
    p.num_prach_ocas    = 1u;
    p.dig_bf_interfaces = 4u;  // → portMask = (1<<4) - 1 = 0x0F

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto computed = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln,
                                                                   /*mmimo_enabled=*/true);
    ASSERT_TRUE(computed.has_value());

    slot_command_api::cell_sub_command cell{};
    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    ASSERT_TRUE(scf_5g_fapi::prach::apply_prach_fh_to_sym_prb_info(
        *computed, *sym_prbs, pdu.beam_index,
        /*cell_index=*/0, OTHER_MODE,
        /*bf_enabled=*/false, /*mmimo_enabled=*/true));

    ASSERT_EQ(sym_prbs->prbs_size, 1u);
    EXPECT_EQ(sym_prbs->prbs[0].common.portMask, 0x0Fu);
}

TEST(ApplyPrachFhToSymPrbInfo, NoBfNoMmimoLeavesPortMaskZero)
{
    PrachPduParams p{};
    p.num_prach_ocas = 1u;

    const auto pdu   = make_prach_pdu(p);
    const auto phy   = make_phy_config(p);
    const auto addln = make_addln(p);

    const auto computed = scf_5g_fapi::prach::compute_prach_params(pdu, phy, addln, false);
    ASSERT_TRUE(computed.has_value());

    slot_command_api::cell_sub_command cell{};
    auto* const sym_prbs = cell.sym_prb_info();
    ASSERT_NE(sym_prbs, nullptr);
    ASSERT_TRUE(scf_5g_fapi::prach::apply_prach_fh_to_sym_prb_info(
        *computed, *sym_prbs, pdu.beam_index,
        /*cell_index=*/0, OTHER_MODE,
        /*bf_enabled=*/false, /*mmimo_enabled=*/false));

    ASSERT_EQ(sym_prbs->prbs_size, 1u);
    EXPECT_EQ(sym_prbs->prbs[0].common.portMask, 0u);
    EXPECT_EQ(sym_prbs->prbs[0].beams_array_size, 0u);
}
