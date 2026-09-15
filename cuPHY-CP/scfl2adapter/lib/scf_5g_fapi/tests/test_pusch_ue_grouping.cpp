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
 * @file test_pusch_ue_grouping.cpp
 * @brief Unit tests for DefaultPuschUeGroupingPolicy and PuschPduParser::setup_cell.
 *
 * Covers:
 *   Group UGP — UE grouping policy: matches() and init_new_group() behaviour
 *   Group SC  — setup_cell: cell dynamic param init, duplicate detection,
 *               null-params, capacity-guard and carrier-bounds error paths
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "aerial/casts/casts.hpp"  // aerial::casts::assume_cast
#include "scf_5g_fapi_pusch_ue_grouping.hpp"
#include "scf_5g_fapi_pusch_pdu_parser.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"
#include "scf_5g_fapi_lbrm.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

// ---------------------------------------------------------------------------
// Mock CellView (satisfies scf_5g_fapi::CellView)
// ---------------------------------------------------------------------------

namespace {

struct MockCellView final
{
    cuphyCellStatPrm_t stat_prm_{};

    // Single-sector TDD slot-detail stand-ins (see ul_*_symbol() below). Default
    // 0 -> the parser's fallback path (full slot @ symbol 0), matching the prior
    // hardcoded behavior so non-SINGLE_SECT tests are unaffected.
    uint8_t ul_max_symbols_{0};
    uint8_t ul_start_symbol_{0};

    MockCellView() noexcept
    {
        // Default matches prior hardcoded LBRM maxLayers (=4).
        stat_prm_.nRxAnt = scf_5g_fapi::lbrm::k_max_layers;
        stat_prm_.nTxAnt = scf_5g_fapi::lbrm::k_max_layers;
    }

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return 59u; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }

    // Mirror PhyCellView's slot_detail-derived accessors used by the PUSCH parser.
    [[nodiscard]] uint8_t ul_start_symbol() const noexcept { return ul_start_symbol_; }
    [[nodiscard]] uint8_t ul_max_symbols(uint8_t fallback) const noexcept
    {
        return (ul_max_symbols_ == 0u) ? fallback : ul_max_symbols_;
    }
};

static_assert(scf_5g_fapi::CellView<MockCellView>,
              "MockCellView must satisfy CellView");

// ---------------------------------------------------------------------------
// Mock UlModuleView (satisfies scf_5g_fapi::UlModuleView)
// ---------------------------------------------------------------------------

struct MockUlModuleView final
{
    int  static_pusch_slot_{-1};
    int32_t carrier_id_val_{0};
    uint16_t phy_cell_id_val_{0};
    bool enable_weighted_avg_cfo_{false};
    uint8_t lbrm_{0};

    // Per-cell overrides keyed by logical cell id.  When a key is absent the scalar
    // fallbacks (carrier_id_val_ / 0) are returned, so single-cell tests that only
    // set carrier_id_val_ keep working regardless of the cell id passed.  Multi-cell
    // tests populate these to prove setup_cell keys source lookups off msg_cell_id.
    std::map<uint32_t, int32_t>  carrier_by_cell_{};
    std::map<uint32_t, uint16_t> stat_prm_by_cell_{};

    // Beamforming / fronthaul configuration (Phase 3).
    bool      mmimo_enabled_{false};
    bool      bf_enabled_{false};
    bool      fapi_to_cplane_direct_enabled_{false};
    ::ru_type ru_type_{OTHER_MODE};
    slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info_{nullptr};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    // Distinct from cell_sub_cmd_.sym_prb_info(): in fapi_to_cplane_direct mode the
    // parser redirects the FH PRB/symbol fill to this UL Order scratch, so the two
    // destinations must be separate objects for a test to tell them apart.
    // Heap-allocated because slot_info_t is large (3 x MAX_PRB_INFO arrays).
    mutable std::unique_ptr<slot_command_api::slot_info_t> order_sym_prbs_{
        std::make_unique<slot_command_api::slot_info_t>()};
    mutable nv::phy_config_option              config_opt_{};
    mutable nv::pucch_dtx_t_list              dtx_list_{};
    mutable float                              dtx_pusch_{0.0f};
    mutable nv::slot_limit_cell_error_t        limit_errors_{};
    mutable nv::slot_limit_group_error_t       group_limit_errors_{};
    MockCellView                               cell_view_{};

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
        return order_sym_prbs_.get();
    }

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }
    [[nodiscard]] bool bf_enabled()    const noexcept { return bf_enabled_; }

    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPuschSlotNum() const noexcept { return static_pusch_slot_; }
    [[nodiscard]] uint8_t lbrm() const noexcept { return lbrm_; }

    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds() const noexcept { return dtx_list_; }
    [[nodiscard]] const float& dtx_thresholds_pusch() const noexcept { return dtx_pusch_; }
    [[nodiscard]] bool enable_weighted_avg_cfo() const noexcept { return enable_weighted_avg_cfo_; }

    [[nodiscard]] nv::phy_mac_transport& transport(int) const noexcept
    {
        // The UL concept requires transport() (used by the SRS path); PUSCH tests
        // never call it.  nv::phy_mac_transport has no default ctor, so there is
        // no valid object to return.  Reaching here is a test bug — fail-fast via
        // std::abort() (it is [[noreturn]], so no dangling reference is handed
        // back) rather than returning a reference to never-constructed storage
        // (UB).  Mirrors the equivalence-harness mock.
        ADD_FAILURE() << "MockUlModuleView::transport() unexpectedly called";
        std::abort();
    }

    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t cell_id) const noexcept
    {
        const auto it = stat_prm_by_cell_.find(cell_id);
        return (it != stat_prm_by_cell_.end()) ? it->second : 0u;
    }
    [[nodiscard]] int32_t carrier_id(uint32_t cell_id) const noexcept
    {
        const auto it = carrier_by_cell_.find(cell_id);
        return (it != carrier_by_cell_.end()) ? it->second : carrier_id_val_;
    }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t) const noexcept { return phy_cell_id_val_; }

    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    // Required by the UlModuleView concept (extended by PUCCH MR1 for group-level
    // L1-limit counters). PUSCH does not exercise this accessor; presence-only.
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }
    [[nodiscard]] uint8_t indication_instances_per_slot(uint32_t) const noexcept { return 0u; }

    void send_fapi_error_indication(uint32_t,
                                    scf_fapi_message_id_e,
                                    scf_fapi_error_codes_t,
                                    uint16_t,
                                    uint16_t) const noexcept {}

    [[nodiscard]] bool    srs_enabled() const noexcept { return false; }
    [[nodiscard]] ru_type ru(uint32_t)  const noexcept { return ru_type_; }
    [[nodiscard]] ru_type ru_type_for_cell(uint32_t) const noexcept { return ru_type_; }

    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t, const scf_fapi_srs_pdu_t&) const noexcept
    {
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }

    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept { return bfw_coeff_mem_info_; }

    /**
     * @brief Return whether FAPI-to-CPlane direct mode is enabled.
     * @return true when direct mode is enabled, false otherwise; the caller must
     *         check the result. Configurable via @c fapi_to_cplane_direct_enabled_;
     *         defaults to false so tests exercise the parser's FH-fill path.
     */
    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept
    {
        return fapi_to_cplane_direct_enabled_;
    }

    [[nodiscard]] MockCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }
};

static_assert(scf_5g_fapi::UlModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy UlModuleView");
static_assert(scf_5g_fapi::PuschModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy PuschModuleView");

// ---------------------------------------------------------------------------
// Helper: build a minimal scf_fapi_pusch_pdu_t
// ---------------------------------------------------------------------------

scf_fapi_pusch_pdu_t make_pusch_pdu(uint8_t start_sym, uint8_t num_sym,
                                     uint16_t bwp_start, uint16_t rb_start,
                                     uint16_t rb_size,
                                     uint16_t dmrs_sym_pos = 0x0004)
{
    scf_fapi_pusch_pdu_t pdu{};
    pdu.start_symbol_index = start_sym;
    pdu.num_of_symbols     = num_sym;
    pdu.bwp.bwp_start      = bwp_start;
    pdu.rb_start            = rb_start;
    pdu.rb_size             = rb_size;
    pdu.ul_dmrs_sym_pos    = dmrs_sym_pos;
    return pdu;
}

// ---------------------------------------------------------------------------
// Custom test policies
// ---------------------------------------------------------------------------

struct AlwaysSameGroupPolicy final
{
    [[nodiscard]] static bool matches(
        const cuphyPuschUeGrpPrm_t&,
        const scf_fapi_pusch_pdu_t&) noexcept
    {
        return true;
    }

    static void init_new_group(
        cuphyPuschUeGrpPrm_t&,
        [[maybe_unused]] slot_command_api::pusch_params&,
        [[maybe_unused]] std::size_t,
        const scf_fapi_pusch_pdu_t&) noexcept
    {}
};

struct AlwaysNewGroupPolicy final
{
    [[nodiscard]] static bool matches(
        const cuphyPuschUeGrpPrm_t&,
        const scf_fapi_pusch_pdu_t&) noexcept
    {
        return false;
    }

    static void init_new_group(
        cuphyPuschUeGrpPrm_t& grp,
        [[maybe_unused]] slot_command_api::pusch_params&,
        [[maybe_unused]] std::size_t,
        const scf_fapi_pusch_pdu_t& pdu) noexcept
    {
        grp.puschStartSym = pdu.start_symbol_index;
        grp.nPuschSym     = pdu.num_of_symbols;
        grp.startPrb      = static_cast<uint16_t>(pdu.bwp.bwp_start + pdu.rb_start);
        grp.nPrb          = pdu.rb_size;
    }
};

static_assert(scf_5g_fapi::PuschUeGroupingPolicy<AlwaysSameGroupPolicy>);
static_assert(scf_5g_fapi::PuschUeGroupingPolicy<AlwaysNewGroupPolicy>);

// ---------------------------------------------------------------------------
// Helper to get pusch_params from a mock view
// ---------------------------------------------------------------------------

slot_command_api::pusch_params* get_pusch_params(const MockUlModuleView& view)
{
    return view.slot_cmd_.cell_groups.get_pusch_params();
}

// ---------------------------------------------------------------------------
// PUSCH PDU binary buffer builder (for parse() round-trip tests)
// ---------------------------------------------------------------------------

/**
 * One CSI-Part2 report description used to serialise the CSI-Part2 section.
 * paramOffsets/paramSizes carry numPart1Params entries each.
 */
struct Csip2PartDesc
{
    std::vector<uint16_t> param_offsets{};
    std::vector<uint8_t>  param_sizes{};
    uint16_t              part2_size_map_index{0};
};

/**
 * Parameters for one serialised PUSCH PDU.  Defaults produce a minimal,
 * transform-precoding-disabled PDU with no optional payload sections, suitable
 * for UE-grouping and DMRS tests.  Set has_data/has_uci/etc. to append the
 * corresponding payload sections in SCF wire order.
 */
struct PuschPduParams
{
    uint16_t pdu_bitmap         {0};
    uint16_t rnti               {0x1234};
    uint32_t handle             {0xCAFE};
    uint16_t bwp_size           {59};
    uint16_t bwp_start          {0};
    uint8_t  transform_precoding{1};      //!< 1 = disabled (no DFT-s-OFDM / fallback).
    uint16_t data_scrambling_id {42};
    uint8_t  num_of_layers      {1};
    uint8_t  mcs_table          {0};
    uint8_t  mcs_index          {5};
    uint16_t target_code_rate   {193};
    uint8_t  qam_mod_order      {2};
    uint16_t ul_dmrs_sym_pos    {0x0004};
    uint16_t ul_dmrs_scrambling_id {100};
    uint32_t pusch_identity     {7};
    uint8_t  scid               {0};
    uint8_t  num_dmrs_cdm_groups_no_data {1};
    uint16_t dmrs_ports         {1};
    uint16_t rb_start           {0};
    uint16_t rb_size            {50};
    uint8_t  start_symbol_index {2};
    uint8_t  num_of_symbols     {12};

    // Optional puschData (bit 0)
    bool     has_data           {false};
    uint8_t  rv_index           {0};
    uint8_t  harq_process_id    {0};
    uint8_t  new_data_indicator {1};
    uint32_t tb_size            {1000};

    // Optional puschUci (bit 1)
    bool     has_uci            {false};
    uint16_t harq_ack_bit_length{0};
    uint16_t csi_part_1_bit_length{0};
    bool     csi_part2_signaled {false};  //!< 10.04: sets flag_csi_part2 = 0xFFFF.
    uint8_t  alpha_scaling      {0};
    uint8_t  beta_offset_harq_ack{0};
    uint8_t  beta_offset_csi_1  {0};
    uint8_t  beta_offset_csi_2  {0};

    // Optional dftsOfdm (bit 3, only when transform_precoding == 0)
    uint8_t  low_papr_group_number   {0};
    uint16_t low_papr_sequence_number{0};

    // Beamforming (always present)
    uint16_t num_prgs           {1};
    uint16_t prg_size           {1};
    uint8_t  dig_bf_interfaces  {0};

    // Maintenance (10.04, always present)
    uint8_t  group_or_sequence_hopping{0};

    // CSI-Part2 section (10.04, present when csi_part2_signaled && has_uci)
    std::vector<Csip2PartDesc> csip2_parts{};

    // Extension (read only when the view enables weighted-average CFO)
    uint8_t  ext_n_iterations        {0};
    uint8_t  ext_ldpc_early_termination{0};
    uint8_t  ext_fo_forget_coeff     {0};
};

/** Append @p n bytes from @p src to @p buf. */
void append_bytes(std::vector<uint8_t>& buf, const void* src, std::size_t n)
{
    const auto* p = static_cast<const uint8_t*>(src);
    buf.insert(buf.end(), p, p + n);
}

/**
 * Serialise a PUSCH PDU body (starting at the fixed scf_fapi_pusch_pdu_t
 * header) in SCF wire order: header, [data], [uci], [dftsOfdm], beamforming,
 * [maintenance], [csip2], extension.
 *
 * @param[in] p  PDU parameters to serialise.
 * @return Byte buffer holding the serialised PDU. Return value must be checked.
 */
[[nodiscard]] std::vector<uint8_t> build_pusch_pdu_body(const PuschPduParams& p)
{
    std::vector<uint8_t> buf;

    scf_fapi_pusch_pdu_t hdr{};
    hdr.pdu_bitmap          = p.pdu_bitmap;
    hdr.rnti                = p.rnti;
    hdr.handle              = p.handle;
    hdr.bwp.bwp_size        = p.bwp_size;
    hdr.bwp.bwp_start       = p.bwp_start;
    hdr.target_code_rate    = p.target_code_rate;
    hdr.qam_mod_order       = p.qam_mod_order;
    hdr.mcs_index           = p.mcs_index;
    hdr.mcs_table           = p.mcs_table;
    hdr.transform_precoding = p.transform_precoding;
    hdr.data_scrambling_id  = p.data_scrambling_id;
    hdr.num_of_layers       = p.num_of_layers;
    hdr.ul_dmrs_sym_pos     = p.ul_dmrs_sym_pos;
    hdr.ul_dmrs_scrambling_id = p.ul_dmrs_scrambling_id;
    hdr.pusch_identity      = p.pusch_identity;
    hdr.scid                = p.scid;
    hdr.num_dmrs_cdm_groups_no_data = p.num_dmrs_cdm_groups_no_data;
    hdr.dmrs_ports          = p.dmrs_ports;
    hdr.resource_alloc      = 1u;
    hdr.rb_start            = p.rb_start;
    hdr.rb_size             = p.rb_size;
    hdr.start_symbol_index  = p.start_symbol_index;
    hdr.num_of_symbols      = p.num_of_symbols;
    append_bytes(buf, &hdr, sizeof(hdr));

    if (p.has_data)
    {
        scf_fapi_pusch_data_t data{};
        data.rv_index           = p.rv_index;
        data.harq_process_id    = p.harq_process_id;
        data.new_data_indicator = p.new_data_indicator;
        data.tb_size            = p.tb_size;
        append_bytes(buf, &data, sizeof(data));
    }

    if (p.has_uci)
    {
        scf_fapi_pusch_uci_t uci{};
        uci.harq_ack_bit_length   = p.harq_ack_bit_length;
        uci.csi_part_1_bit_length = p.csi_part_1_bit_length;
        uci.flag_csi_part2 = p.csi_part2_signaled
            ? std::numeric_limits<uint16_t>::max() : 0u;
        uci.alpha_scaling        = p.alpha_scaling;
        uci.beta_offset_harq_ack = p.beta_offset_harq_ack;
        uci.beta_offset_csi_1    = p.beta_offset_csi_1;
        uci.beta_offset_csi_2    = p.beta_offset_csi_2;
        append_bytes(buf, &uci, sizeof(uci));
    }

    if (p.transform_precoding == 0u && (p.pdu_bitmap & 0x8u) != 0u)
    {
        scf_fapi_pusch_dftsofdm_t dft{};
        dft.lowPaprGroupNumber    = p.low_papr_group_number;
        dft.lowPaprSequenceNumber = p.low_papr_sequence_number;
        append_bytes(buf, &dft, sizeof(dft));
    }

    scf_fapi_rx_beamforming_t bf{};
    bf.num_prgs          = p.num_prgs;
    bf.prg_size          = p.prg_size;
    bf.dig_bf_interfaces = p.dig_bf_interfaces;
    append_bytes(buf, &bf, sizeof(bf));
    // beam_idx[] holds numPRGs * digBFInterfaces entries (SCF FAPI Table 3-53).
    const std::size_t num_beam_entries =
        static_cast<std::size_t>(p.num_prgs) *
        static_cast<std::size_t>(p.dig_bf_interfaces);
    for (std::size_t i = 0u; i < num_beam_entries; ++i)
    {
        const uint16_t beam = 0u;
        append_bytes(buf, &beam, sizeof(beam));
    }

    scf_fapi_pusch_maintenance_t maint{};
    maint.groupOrSequenceHopping = p.group_or_sequence_hopping;
    append_bytes(buf, &maint, sizeof(maint));

    if (p.csi_part2_signaled && p.has_uci)
    {
        scf_uci_csip2_info_t info{};
        info.numPart2s = static_cast<uint16_t>(p.csip2_parts.size());
        append_bytes(buf, &info, sizeof(info));

        for (const auto& part : p.csip2_parts)
        {
            scf_uci_csip2_part_t part_hdr{};
            part_hdr.priority       = 0u;
            part_hdr.numPart1Params = static_cast<uint8_t>(part.param_offsets.size());
            append_bytes(buf, &part_hdr, sizeof(part_hdr));
            for (const uint16_t off : part.param_offsets) { append_bytes(buf, &off, sizeof(off)); }
            for (const uint8_t  sz  : part.param_sizes)   { append_bytes(buf, &sz,  sizeof(sz));  }

            scf_uci_csip2_part_scope_t scope{};
            scope.part2SizeMapIndex = part.part2_size_map_index;
            append_bytes(buf, &scope, sizeof(scope));
        }
    }

    scf_fapi_pusch_extension_t ext{};
    ext.n_iterations          = p.ext_n_iterations;
    ext.ldpc_early_termination = p.ext_ldpc_early_termination;
    ext.fo_forget_coeff       = p.ext_fo_forget_coeff;
    append_bytes(buf, &ext, sizeof(ext));

    return buf;
}

/** View a built body buffer as a PUSCH PDU (byte-buffer pun via assume_cast). */
[[nodiscard]] const scf_fapi_pusch_pdu_t& as_pdu(const std::vector<uint8_t>& body)
{
    return *aerial::casts::assume_cast<scf_fapi_pusch_pdu_t>(body.data());
}

} // anonymous namespace

// ===========================================================================
// Group UGP — DefaultPuschUeGroupingPolicy::matches()
// ===========================================================================

// matches() depends only on input data, so the six cases below are a single
// data-parameterised suite (UGP1 same-alloc match, UGP2-5 each single-field
// mismatch, UGP6 bwp_start contribution).  See §13.4.
struct MatchesCase
{
    const char* name;
    uint8_t     grp_start_sym;
    uint8_t     grp_num_sym;
    uint16_t    grp_start_prb;
    uint16_t    grp_n_prb;
    uint8_t     pdu_start_sym;
    uint8_t     pdu_num_sym;
    uint16_t    pdu_bwp_start;
    uint16_t    pdu_rb_start;
    uint16_t    pdu_n_prb;
    bool        expected;
};

class PuschMatchesTest : public ::testing::TestWithParam<MatchesCase> {};

TEST_P(PuschMatchesTest, MatchesContract)
{
    const auto& c = GetParam();

    cuphyPuschUeGrpPrm_t grp{};
    grp.puschStartSym = c.grp_start_sym;
    grp.nPuschSym     = c.grp_num_sym;
    grp.startPrb      = c.grp_start_prb;
    grp.nPrb          = c.grp_n_prb;

    const auto pdu = make_pusch_pdu(c.pdu_start_sym, c.pdu_num_sym,
                                    c.pdu_bwp_start, c.pdu_rb_start, c.pdu_n_prb);

    EXPECT_EQ(scf_5g_fapi::DefaultPuschUeGroupingPolicy::matches(grp, pdu), c.expected);
}

INSTANTIATE_TEST_SUITE_P(
    UGP, PuschMatchesTest,
    ::testing::Values(
        //           grp(sym,nsym,prb,nprb)   pdu(sym,nsym,bwp,rb,nprb)  expect
        MatchesCase{"SameAllocation",      2, 12, 10,  50,  2, 12,  0, 10,  50, true},
        MatchesCase{"DifferentStartSym",   2, 12, 10,  50,  3, 12,  0, 10,  50, false},
        MatchesCase{"DifferentNumSym",     2, 12, 10,  50,  2, 10,  0, 10,  50, false},
        MatchesCase{"DifferentStartPrb",   2, 12, 10,  50,  2, 12,  5, 10,  50, false},
        MatchesCase{"DifferentNPrb",       2, 12, 10,  50,  2, 12,  0, 10,  40, false},
        MatchesCase{"BwpStartContributes", 0, 14, 25, 100,  0, 14, 20,  5, 100, true}),
    [](const ::testing::TestParamInfo<MatchesCase>& info) {
        return std::string(info.param.name);
    });

// ===========================================================================
// Group UGP — DefaultPuschUeGroupingPolicy::init_new_group()
// ===========================================================================

TEST(PuschUeGrouping, UGP7_InitNewGroup_PopulatesAllFields)
{
    cuphyPuschUeGrpPrm_t grp{};
    slot_command_api::pusch_params params{};

    auto pdu = make_pusch_pdu(3, 11, 10, 5, 80, 0x0C01);

    scf_5g_fapi::DefaultPuschUeGroupingPolicy::init_new_group(grp, params, 0, pdu);

    EXPECT_EQ(grp.puschStartSym, 3u);
    EXPECT_EQ(grp.nPuschSym, 11u);
    EXPECT_EQ(grp.startPrb, 15u);  // bwp_start(10) + rb_start(5)
    EXPECT_EQ(grp.nPrb, 80u);
    EXPECT_EQ(grp.dmrsSymLocBmsk, 0x0C01u);
    EXPECT_EQ(grp.rssiSymLocBmsk, 0x0C01u);
}

// ===========================================================================
// Group UGP — Custom policies (concept conformance + isolation testing)
// ===========================================================================

TEST(PuschUeGrouping, UGP8_AlwaysSameGroupPolicy_AlwaysMatches)
{
    cuphyPuschUeGrpPrm_t grp{};
    grp.puschStartSym = 0;

    auto pdu = make_pusch_pdu(5, 7, 20, 30, 40);  // completely different

    EXPECT_TRUE(AlwaysSameGroupPolicy::matches(grp, pdu));
}

TEST(PuschUeGrouping, UGP9_AlwaysNewGroupPolicy_NeverMatches)
{
    cuphyPuschUeGrpPrm_t grp{};
    grp.puschStartSym = 2;
    grp.nPuschSym     = 12;
    grp.startPrb      = 10;
    grp.nPrb          = 50;

    auto pdu = make_pusch_pdu(2, 12, 0, 10, 50);  // identical to group

    EXPECT_FALSE(AlwaysNewGroupPolicy::matches(grp, pdu));
}

// ===========================================================================
// Group SC — PuschPduParser::setup_cell()
// ===========================================================================

// Shared fixture for setup_cell() tests: each case constructs the same
// MockUlModuleView, a parser bound to it, and a default UL_TTI request.
// Members are declared in init order so parser_ binds to the constructed view_.
class PuschSetupCellTest : public ::testing::Test
{
protected:
    MockUlModuleView                              view_{};
    scf_5g_fapi::PuschPduParser<MockUlModuleView> parser_{view_};
    scf_fapi_ul_tti_req_t                         req_{};
};

TEST_F(PuschSetupCellTest, SC1_BasicSetupCell_PopulatesCellDynInfo)
{
    view_.static_pusch_slot_ = -1;  // use req.slot
    view_.phy_cell_id_val_   = 42;  // distinct PHY cell id (≠ carrier 0)
    req_.slot = 7;
    req_.sfn  = 100;

    parser_.setup_cell(req_, 0u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nCells, 1u);
    EXPECT_EQ(params->cell_dyn_info[0].slotNum, 7u);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmDynIdx, 0u);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmStatIdx, 0u);
    EXPECT_EQ(params->cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list[0], 0);
    // phy_cell_index_list is sourced from phy_cell_id() (the physical cell id),
    // distinct from cell_index_list (which holds the logical carrier id).
    EXPECT_EQ(params->phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params->phy_cell_index_list[0], 42);
}

TEST_F(PuschSetupCellTest, SC2_StaticSlotOverride)
{
    view_.static_pusch_slot_ = 3;  // override
    req_.slot = 9;                 // should be ignored

    parser_.setup_cell(req_, 0u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_dyn_info[0].slotNum, 3u);
}

TEST_F(PuschSetupCellTest, SC3_DuplicateCarrierDetection)
{
    view_.carrier_id_val_ = 5;
    req_.slot = 0;

    parser_.setup_cell(req_, 0u);
    parser_.setup_cell(req_, 0u);  // duplicate — should be skipped

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nCells, 1u);
    EXPECT_EQ(params->cell_index_list.size(), 1u);
}

TEST_F(PuschSetupCellTest, SC4_NullPuschParams_NoOp)
{
    // Force get_pusch_params() to return nullptr.  pusch_params is eagerly
    // allocated by the cell_group_command ctor and create_if() never re-creates
    // it, so releasing the holder makes get_pusch_params() return null.
    view_.slot_cmd_.cell_groups.pusch.reset();
    req_.slot = 4;

    // Must be a safe no-op (early return on null params) — no crash, no throw.
    parser_.setup_cell(req_, 0u);

    EXPECT_EQ(get_pusch_params(view_), nullptr);
}

TEST_F(PuschSetupCellTest, SC5_CapacityGuard_StopsAtMax)
{
    req_.slot = 0;

    // Register MAX_CELLS_PER_CELL_GROUP distinct cells (carriers 0..MAX-1, all
    // within bounds of cell_ue_group_idx_start[]).
    for (int c = 0; c < slot_command_api::MAX_CELLS_PER_CELL_GROUP; ++c)
    {
        view_.carrier_id_val_ = c;
        parser_.setup_cell(req_, 0u);
    }

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(static_cast<int>(params->cell_grp_info.nCells),
              slot_command_api::MAX_CELLS_PER_CELL_GROUP);

    // One more cell must be rejected by the capacity guard: cell_idx == nCells
    // == MAX trips the guard, which returns before reading carrier_id_val_ (so
    // the out-of-range carrier is never used as an array index).
    view_.carrier_id_val_ = slot_command_api::MAX_CELLS_PER_CELL_GROUP;
    parser_.setup_cell(req_, 0u);

    EXPECT_EQ(static_cast<int>(params->cell_grp_info.nCells),
              slot_command_api::MAX_CELLS_PER_CELL_GROUP);
}

TEST_F(PuschSetupCellTest, SC6_OutOfRangeCarrierBeforeCapacity_Rejected)
{
    req_.slot = 0;

    // carrier == MAX_CELLS_PER_CELL_GROUP is out of bounds for
    // cell_ue_group_idx_start[] (valid indices 0..MAX-1) while nCells is still 0,
    // so the carrier bounds check — not the capacity guard — must reject it
    // before it is used as an array index.
    view_.carrier_id_val_ = slot_command_api::MAX_CELLS_PER_CELL_GROUP;
    parser_.setup_cell(req_, 0u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nCells, 0u);
    EXPECT_TRUE(params->cell_index_list.empty());
}

TEST_F(PuschSetupCellTest, SC7_NegativeCarrierBeforeCapacity_Rejected)
{
    req_.slot = 0;

    // A negative carrier_id (signed int32_t from L2) must be rejected by the
    // bounds check before being used as an array index.
    view_.carrier_id_val_ = -1;
    parser_.setup_cell(req_, 0u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nCells, 0u);
    EXPECT_TRUE(params->cell_index_list.empty());
}

// A-2 regression: setup_cell must key its source lookups (carrier_id /
// cell_stat_prm_idx) off the UL_TTI message cell id, not the running nCells
// count.  Otherwise a slot whose lower-id cells carry no PUSCH registers the
// PUSCH-bearing cell under the wrong carrier.

TEST_F(PuschSetupCellTest, SC8_SourceLookupKeyedByMsgCellId_NotProcessingOrder)
{
    req_.slot = 0;
    // Distinct carrier / static-param index per logical cell id (carriers must be
    // < MAX_CELLS_PER_CELL_GROUP == 48 to pass the carrier bounds check).
    view_.carrier_by_cell_  = {{0u, 30}, {1u, 31}, {2u, 32}};
    view_.stat_prm_by_cell_ = {{0u, static_cast<uint16_t>(10)},
                               {1u, static_cast<uint16_t>(11)},
                               {2u, static_cast<uint16_t>(12)}};

    // Cells 0 and 1 carry no PUSCH this slot, so only cell 2's setup_cell runs
    // first — nCells is still 0 at this point.
    parser_.setup_cell(req_, 2u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_grp_info.nCells, 1u);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    // Registered under cell 2's carrier (32), NOT carrier_id(nCells==0) == 30.
    EXPECT_EQ(params->cell_index_list[0], 32);
    // Static-param idx sourced from cell 2 (12), not cell 0 (10).
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmStatIdx, 12u);
    // Destination slot is still the running count (0).
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmDynIdx, 0u);
}

TEST_F(PuschSetupCellTest, SC9_MultiCell_NonSequentialIds_MapEachToOwnCarrier)
{
    req_.slot = 0;
    view_.carrier_by_cell_  = {{0u, 30}, {1u, 31}, {2u, 32}};
    view_.stat_prm_by_cell_ = {{0u, static_cast<uint16_t>(10)},
                               {1u, static_cast<uint16_t>(11)},
                               {2u, static_cast<uint16_t>(12)}};

    // Cell 0 has no PUSCH; cells 1 and 2 do, registered in message order.
    parser_.setup_cell(req_, 1u);
    parser_.setup_cell(req_, 2u);

    auto* params = get_pusch_params(view_);
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_grp_info.nCells, 2u);
    ASSERT_EQ(params->cell_index_list.size(), 2u);
    // Each cell registered under its own carrier at sequential destination slots.
    EXPECT_EQ(params->cell_index_list[0], 31);
    EXPECT_EQ(params->cell_index_list[1], 32);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmStatIdx, 11u);
    EXPECT_EQ(params->cell_dyn_info[1].cellPrmStatIdx, 12u);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmDynIdx, 0u);
    EXPECT_EQ(params->cell_dyn_info[1].cellPrmDynIdx, 1u);
}

// ===========================================================================
// PuschParseTest fixture — parse() round-trip tests
// ===========================================================================

namespace {

class PuschParseTest : public ::testing::Test
{
protected:
    MockUlModuleView view_{};

    // One persistent parser shared across setup_cell()/parse(), mirroring
    // production: ULSlotProcessor holds a single PuschPduParser and calls
    // setup_cell() then parse() on that same instance, so per-message cached
    // state (e.g. fh_enabled_for_msg_) survives from setup to parse.  view_ is
    // declared first so it outlives the not_null<V*> captured here.
    scf_5g_fapi::PuschPduParser<MockUlModuleView> parser_{view_};

    /** Register one cell via setup_cell() using the view's current carrier id. */
    void setup_cell(uint16_t slot = 0u, uint32_t cell_id = 0u)
    {
        scf_fapi_ul_tti_req_t req{};
        req.slot = slot;
        parser_.setup_cell(req, cell_id);
    }

    /** Parse one PDU body with the default grouping policy. */
    [[nodiscard]] bool parse(const std::vector<uint8_t>& body)
    {
        return parser_.parse(0u, 0u, as_pdu(body));
    }

    [[nodiscard]] slot_command_api::pusch_params* params()
    {
        return get_pusch_params(view_);
    }
};

/** Build a default-allocation body with optionally overridden resource fields. */
[[nodiscard]] std::vector<uint8_t> body_with_alloc(uint8_t start_sym, uint8_t num_sym,
                                                   uint16_t rb_start, uint16_t rb_size,
                                                   uint16_t rnti = 0x1u)
{
    PuschPduParams p{};
    p.rnti               = rnti;
    p.start_symbol_index = start_sym;
    p.num_of_symbols     = num_sym;
    p.rb_start           = rb_start;
    p.rb_size            = rb_size;
    return build_pusch_pdu_body(p);
}

} // namespace

// ---------------------------------------------------------------------------
// Capacity boundary
// ---------------------------------------------------------------------------

TEST_F(PuschParseTest, Capacity_DropsPduWhenUesAtMax)
{
    setup_cell();

    // Identical allocation -> all UEs land in one group; fills nUes to MAX.
    const auto body = body_with_alloc(2u, 12u, 0u, 50u);
    for (int i = 0; i < slot_command_api::MAX_PUSCH_UE_PER_TTI; ++i)
    {
        EXPECT_TRUE(parse(body)) << "UE " << i << " should be accepted";
    }

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUes), slot_command_api::MAX_PUSCH_UE_PER_TTI);

    // One more must be dropped by the nUes capacity guard.
    EXPECT_FALSE(parse(body));
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUes), slot_command_api::MAX_PUSCH_UE_PER_TTI);
}

TEST_F(PuschParseTest, Capacity_DropsPduWhenGroupsAtMax)
{
    setup_cell();

    // AlwaysNewGroupPolicy makes every PDU its own group, so nUeGrps tracks nUes
    // 1:1.  Because MAX_PUSCH_UE_GROUPS == MAX_PUSCH_UE_PER_TTI and parse() checks
    // the slot-global nUes cap before setup_ue_group(), the new-group path
    // saturates at MAX_PUSCH_UE_GROUPS groups (== MAX_PUSCH_UE_PER_TTI UEs).  The
    // dedicated per-cell group-count guard in setup_ue_group() is defense-in-depth
    // (unreachable while the two caps are equal): the nUes guard is what drops the
    // overflowing PDU here.
    const auto body = body_with_alloc(2u, 12u, 0u, 50u);
    scf_5g_fapi::PuschPduParser<MockUlModuleView, AlwaysNewGroupPolicy> parser{view_};

    for (int i = 0; i < slot_command_api::MAX_PUSCH_UE_GROUPS; ++i)
    {
        EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(body))) << "group " << i << " should be accepted";
    }

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUeGrps), slot_command_api::MAX_PUSCH_UE_GROUPS);
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUes),    slot_command_api::MAX_PUSCH_UE_PER_TTI);

    // One more PDU must be dropped, and neither counter advances past the cap.
    EXPECT_FALSE(parser.parse(0u, 0u, as_pdu(body)));
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUeGrps), slot_command_api::MAX_PUSCH_UE_GROUPS);
    EXPECT_EQ(static_cast<int>(p->cell_grp_info.nUes),    slot_command_api::MAX_PUSCH_UE_PER_TTI);
}

// ---------------------------------------------------------------------------
// UE grouping (UG-*) — integration through parse()
// ---------------------------------------------------------------------------

TEST_F(PuschParseTest, UG1_SameAllocation_SharesGroup)
{
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u, 0x1u)));
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u, 0x2u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 1u);
    EXPECT_EQ(p->cell_grp_info.nUes,    2u);
    EXPECT_EQ(p->ue_grp_info[0].nUes,   2u);
    EXPECT_EQ(p->ue_grp_info[0].pUePrmIdxs[0], 0u);
    EXPECT_EQ(p->ue_grp_info[0].pUePrmIdxs[1], 1u);
}

TEST_F(PuschParseTest, UG2_DifferentStartSym_NewGroup)
{
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));
    EXPECT_TRUE(parse(body_with_alloc(3u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 2u);
    EXPECT_EQ(p->ue_grp_info[0].nUes,   1u);
    EXPECT_EQ(p->ue_grp_info[1].nUes,   1u);
}

TEST_F(PuschParseTest, UG3_DifferentNumSym_NewGroup)
{
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));
    EXPECT_TRUE(parse(body_with_alloc(2u, 10u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 2u);
}

TEST_F(PuschParseTest, UG4_DifferentStartPrb_NewGroup)
{
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 20u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 2u);
}

TEST_F(PuschParseTest, UG7_NoSetupCell_ParseFails)
{
    // setup_cell intentionally not called -> nCells == 0.
    EXPECT_FALSE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUes,    0u);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 0u);
}

TEST_F(PuschParseTest, UG8_AlwaysSameGroupPolicy_SingleGroup)
{
    setup_cell();

    // Different allocations, but the custom policy forces a single group.
    const auto b1 = body_with_alloc(2u, 12u, 10u, 50u);
    const auto b2 = body_with_alloc(5u,  7u, 20u, 40u);

    scf_5g_fapi::PuschPduParser<MockUlModuleView, AlwaysSameGroupPolicy> parser{view_};
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(b1)));
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(b2)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 1u);
    EXPECT_EQ(p->cell_grp_info.nUes,    2u);
    EXPECT_EQ(p->ue_grp_info[0].nUes,   2u);
}

TEST_F(PuschParseTest, UG9_AlwaysNewGroupPolicy_GroupPerUe)
{
    setup_cell();

    // Identical allocations, but the custom policy forces a new group each time.
    const auto body = body_with_alloc(2u, 12u, 10u, 50u);

    scf_5g_fapi::PuschPduParser<MockUlModuleView, AlwaysNewGroupPolicy> parser{view_};
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(body)));
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(body)));
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(body)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 3u);
    EXPECT_EQ(p->cell_grp_info.nUes,    3u);
}

TEST_F(PuschParseTest, UG10_MultiCell_GroupsIsolatedPerCell)
{
    // Cell 0: two distinct allocations -> groups 0 and 1.
    view_.carrier_id_val_ = 0;
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));
    EXPECT_TRUE(parse(body_with_alloc(3u, 12u, 10u, 50u)));

    // Cell 1: allocation identical to cell 0's first UE must still create a new
    // group because group search is scoped to the cell.
    view_.carrier_id_val_ = 1;
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nCells,  2u);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 3u);
    EXPECT_EQ(p->cell_ue_group_idx_start[0], 0u);
    EXPECT_EQ(p->cell_ue_group_idx_start[1], 2u);
    // The cell-1 UE is the third UE and owns the third group.
    EXPECT_EQ(p->ue_info[2].ueGrpIdx, 2u);
}

TEST_F(PuschParseTest, UG11_NewGroup_CellPointerLinkedToOwningCell)
{
    view_.carrier_id_val_ = 0;
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    view_.carrier_id_val_ = 1;
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(p->cell_grp_info.nUeGrps, 2u);
    // Group 0 -> cell 0 (cellPrmDynIdx 0); group 1 -> cell 1 (cellPrmDynIdx 1).
    ASSERT_NE(p->ue_grp_info[0].pCellPrm, nullptr);
    ASSERT_NE(p->ue_grp_info[1].pCellPrm, nullptr);
    EXPECT_EQ(p->ue_grp_info[0].pCellPrm->cellPrmDynIdx, 0u);
    EXPECT_EQ(p->ue_grp_info[1].pCellPrm->cellPrmDynIdx, 1u);
}

TEST_F(PuschParseTest, UG12_NewGroup_DmrsPointerMatchesGroupIndex)
{
    setup_cell();
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    ASSERT_EQ(p->cell_grp_info.nUeGrps, 1u);
    EXPECT_EQ(p->ue_grp_info[0].pDmrsDynPrm, &p->ue_dmrs_info[0]);
}

// ---------------------------------------------------------------------------
// UE scalar field population + LBRM
// ---------------------------------------------------------------------------

TEST_F(PuschParseTest, UeFields_PopulatedFromPdu)
{
    view_.lbrm_ = 1u;  // enable LBRM parameter derivation
    setup_cell();

    PuschPduParams p{};
    p.rnti               = 0xABCDu;
    p.pusch_identity     = 513u;
    p.scid               = 1u;
    p.dmrs_ports         = 0x000Fu;
    p.mcs_table          = 1u;   // 256-QAM -> maxQm = 8
    p.mcs_index          = 20u;
    p.data_scrambling_id = 99u;
    p.num_of_layers      = 4u;
    p.target_code_rate   = 948u;
    p.qam_mod_order      = 8u;
    p.bwp_size           = 59u;  // -> n_PRB_LBRM = 66

    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    ASSERT_EQ(prm->cell_grp_info.nUes, 1u);

    const auto& ue = prm->ue_info[0];
    EXPECT_EQ(ue.rnti,          0xABCDu);
    EXPECT_EQ(ue.puschIdentity, 513u);
    EXPECT_EQ(ue.scid,          1u);
    EXPECT_EQ(ue.dmrsPortBmsk,  0x000Fu);
    EXPECT_EQ(ue.mcsTableIndex, 1u);
    EXPECT_EQ(ue.mcsIndex,      20u);
    EXPECT_EQ(ue.dataScramId,   99u);
    EXPECT_EQ(ue.nUeLayers,     4u);
    EXPECT_EQ(ue.targetCodeRate,948u);
    EXPECT_EQ(ue.qamModOrder,   8u);
    EXPECT_EQ(ue.pduBitmap,     0u);  // no optional sections set

    // LBRM (38.212 §5.4.2.1)
    EXPECT_EQ(ue.i_lbrm,     1u);
    EXPECT_EQ(ue.maxLayers,  4u);
    EXPECT_EQ(ue.maxQm,      8u);   // 256-QAM table
    EXPECT_EQ(ue.n_PRB_LBRM, 66u);  // BWP 59 -> bucket 33-66

    // Extension defaults
    EXPECT_FLOAT_EQ(ue.foForgetCoeff,         0.0f);
    EXPECT_EQ(ue.ldpcEarlyTerminationPerUe,   0u);
    EXPECT_EQ(ue.ldpcMaxNumItrPerUe,          10u);

    // Handle list committed in UE order.
    ASSERT_EQ(prm->scf_ul_tti_handle_list.size(), 1u);
    EXPECT_EQ(prm->scf_ul_tti_handle_list[0], p.handle);
}

TEST_F(PuschParseTest, UeFields_LbrmMaxLayersFollowsRxAnt)
{
    view_.lbrm_ = 1u;
    view_.cell_view_.stat_prm_.nRxAnt = 2u;
    setup_cell();

    PuschPduParams p{};
    p.mcs_table = 1u;
    p.bwp_size  = 59u;

    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    ASSERT_EQ(prm->cell_grp_info.nUes, 1u);
    EXPECT_EQ(prm->ue_info[0].maxLayers, 2u);
}

TEST_F(PuschParseTest, UeFields_LbrmDisabled_LeavesLbrmParamsZero)
{
    view_.lbrm_ = 0u;
    setup_cell();

    PuschPduParams p{};
    p.mcs_table = 1u;
    p.bwp_size  = 59u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    EXPECT_EQ(ue.i_lbrm,     0u);
    EXPECT_EQ(ue.maxQm,      0u);
    EXPECT_EQ(ue.n_PRB_LBRM, 0u);
}

// ---------------------------------------------------------------------------
// Variable-length payload walk: Data + UCI
// ---------------------------------------------------------------------------

TEST_F(PuschParseTest, Payload_DataAndUci_Populated)
{
    view_.dtx_pusch_ = 1.5f;
    setup_cell();

    PuschPduParams p{};
    p.pdu_bitmap            = 0x1u | 0x2u;  // data + uci
    p.has_data              = true;
    p.rv_index              = 2u;
    p.harq_process_id       = 7u;
    p.new_data_indicator    = 1u;
    p.tb_size               = 4096u;
    p.has_uci               = true;
    p.harq_ack_bit_length   = 4u;
    p.csi_part_1_bit_length = 8u;
    p.alpha_scaling         = 3u;
    p.beta_offset_harq_ack  = 5u;
    p.beta_offset_csi_1     = 6u;
    p.beta_offset_csi_2     = 7u;

    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];

    // Data section
    EXPECT_EQ(ue.rv,            2u);
    EXPECT_EQ(ue.TBSize,        4096u);
    EXPECT_EQ(ue.ndi,           1u);
    EXPECT_EQ(ue.harqProcessId, 7u);
    EXPECT_EQ(prm->ue_tb_size[0], 4096u);

    // UCI section
    ASSERT_NE(ue.pUciPrms, nullptr);
    EXPECT_EQ(ue.pUciPrms, &prm->uci_info[0]);
    EXPECT_EQ(ue.pUciPrms->nBitsHarq,         4u);
    EXPECT_EQ(ue.pUciPrms->nBitsCsi1,         8u);
    EXPECT_EQ(ue.pUciPrms->alphaScaling,      3u);
    EXPECT_EQ(ue.pUciPrms->betaOffsetHarqAck, 5u);
    EXPECT_EQ(ue.pUciPrms->betaOffsetCsi1,    6u);
    EXPECT_EQ(ue.pUciPrms->betaOffsetCsi2,    7u);
    EXPECT_EQ(ue.pUciPrms->nCsiReports,       1u);
    EXPECT_FLOAT_EQ(ue.pUciPrms->DTXthreshold, 1.5f);

    EXPECT_EQ(ue.pduBitmap, 0x3u);  // 10.04: nRanksBits=255, so bit 5 not set
}

// ---------------------------------------------------------------------------
// DMRS configuration (DMRS-*)
// ---------------------------------------------------------------------------

// DMRS dmrsMaxLen/dmrsAddlnPos derivation depends only on the ul_dmrs_sym_pos
// bitmask, so the cases below are a single data-parameterised suite.  Expected
// values follow TS 38.211 §6.4.1.1 (single vs double-symbol DMRS, additional
// position count from the symbol bitmap popcount/pairing).
struct DmrsLenCase
{
    const char* name;
    uint16_t    sym_pos;
    uint8_t     expected_max_len;
    uint8_t     expected_addln_pos;
};

class PuschDmrsTest : public ::testing::TestWithParam<DmrsLenCase> {};

TEST_P(PuschDmrsTest, MaxLenAndAddlnPos)
{
    const auto& c = GetParam();

    MockUlModuleView local_view{};
    scf_5g_fapi::PuschPduParser<MockUlModuleView> parser{local_view};
    scf_fapi_ul_tti_req_t req{};
    parser.setup_cell(req, 0u);

    PuschPduParams p{};
    p.ul_dmrs_sym_pos              = c.sym_pos;
    p.num_dmrs_cdm_groups_no_data  = 2u;
    p.ul_dmrs_scrambling_id        = 321u;
    const auto body = build_pusch_pdu_body(p);
    EXPECT_TRUE(parser.parse(0u, 0u, as_pdu(body)));

    auto* prm = get_pusch_params(local_view);
    ASSERT_NE(prm, nullptr);
    const auto& dmrs = prm->ue_dmrs_info[0];
    EXPECT_EQ(dmrs.dmrsMaxLen,         c.expected_max_len);
    EXPECT_EQ(dmrs.dmrsAddlnPos,       c.expected_addln_pos);
    EXPECT_EQ(dmrs.nDmrsCdmGrpsNoData, 2u);
    EXPECT_EQ(dmrs.dmrsScrmId,         321u);
}

INSTANTIATE_TEST_SUITE_P(
    DMRS, PuschDmrsTest,
    ::testing::Values(
        //          name                        sym_pos  maxLen  addlnPos
        DmrsLenCase{"SingleSym2_Len1_Addln0",       0x0004u, 1u, 0u},
        DmrsLenCase{"TwoNonAdjacent0_6_Len1_Addln1",0x0041u, 1u, 1u},
        DmrsLenCase{"TwoAdjacent0_1_Len2_Addln0",   0x0003u, 2u, 0u},
        DmrsLenCase{"Double0_1And10_11_Len2_Addln1",0x0C03u, 2u, 1u},
        DmrsLenCase{"EmptyMask_Len1_Addln0_Guarded",0x0000u, 1u, 0u}),
    [](const ::testing::TestParamInfo<DmrsLenCase>& info) {
        return std::string(info.param.name);
    });

TEST_F(PuschParseTest, Dmrs_SameGroupSecondUe_Idempotent)
{
    setup_cell();
    PuschPduParams p{};
    p.ul_dmrs_sym_pos = 0x0003u;  // double-symbol
    const auto body = build_pusch_pdu_body(p);

    EXPECT_TRUE(parse(body));
    EXPECT_TRUE(parse(body));  // same group, second UE re-writes identical DMRS

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    ASSERT_EQ(prm->cell_grp_info.nUeGrps, 1u);
    EXPECT_EQ(prm->ue_dmrs_info[0].dmrsMaxLen, 2u);
}

// ---------------------------------------------------------------------------
// Maintenance / CSI-Part2 / Extension (ME-*) — SCF_FAPI_10_04 tail
// (the parser header requires SCF_FAPI_10_04, so these always compile here).
// ---------------------------------------------------------------------------

TEST_F(PuschParseTest, ME1_Csip2Signaled_ZeroParts_NoReports)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.pdu_bitmap         = 0x2u;  // uci
    p.has_uci            = true;
    p.csi_part2_signaled = true;  // flag_csi_part2 = 0xFFFF
    // csip2_parts left empty -> numPart2s == 0
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    ASSERT_NE(ue.pUciPrms, nullptr);
    EXPECT_EQ(ue.pUciPrms->nCsi2Reports,      0u);
    EXPECT_EQ(ue.pUciPrms->pCalcCsi2SizePrms, nullptr);
}

TEST_F(PuschParseTest, ME2_Csip2Signaled_TwoParts_Populated)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.pdu_bitmap         = 0x2u;
    p.has_uci            = true;
    p.csi_part2_signaled = true;
    p.csip2_parts = {
        Csip2PartDesc{ /*offsets=*/{10u, 20u}, /*sizes=*/{1u, 2u}, /*mapIdx=*/5u},
        Csip2PartDesc{ /*offsets=*/{30u, 40u}, /*sizes=*/{3u, 4u}, /*mapIdx=*/6u},
    };
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    ASSERT_NE(ue.pUciPrms, nullptr);
    EXPECT_EQ(ue.pUciPrms->nCsi2Reports, 2u);
    ASSERT_NE(ue.pUciPrms->pCalcCsi2SizePrms, nullptr);

    const auto& r0 = ue.pUciPrms->pCalcCsi2SizePrms[0];
    EXPECT_EQ(r0.nPart1Prms,    2u);
    EXPECT_EQ(r0.prmOffsets[0], 10u);
    EXPECT_EQ(r0.prmOffsets[1], 20u);
    EXPECT_EQ(r0.prmSizes[0],   1u);
    EXPECT_EQ(r0.prmSizes[1],   2u);
    EXPECT_EQ(r0.csi2sizeMapIdx, 5u);

    const auto& r1 = ue.pUciPrms->pCalcCsi2SizePrms[1];
    EXPECT_EQ(r1.nPart1Prms,    2u);
    EXPECT_EQ(r1.prmOffsets[0], 30u);
    EXPECT_EQ(r1.prmOffsets[1], 40u);
    EXPECT_EQ(r1.prmSizes[0],   3u);
    EXPECT_EQ(r1.prmSizes[1],   4u);
    EXPECT_EQ(r1.csi2sizeMapIdx, 6u);
}

TEST_F(PuschParseTest, ME3_ExtensionPresent_Populated)
{
    view_.enable_weighted_avg_cfo_ = true;
    setup_cell();

    PuschPduParams p{};
    p.transform_precoding       = 1u;  // disable DFT-s-OFDM fallback
    p.ext_fo_forget_coeff       = 50u; // -> 0.5f
    p.ext_ldpc_early_termination = 1u;
    p.ext_n_iterations          = 8u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    EXPECT_FLOAT_EQ(ue.foForgetCoeff,        0.5f);
    EXPECT_EQ(ue.ldpcEarlyTerminationPerUe,  1u);
    EXPECT_EQ(ue.ldpcMaxNumItrPerUe,         8u);
}

TEST_F(PuschParseTest, ME4_ExtensionDisabled_DefaultsPreserved)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.transform_precoding        = 1u;
    p.ext_fo_forget_coeff        = 50u;  // present in buffer but must NOT be read
    p.ext_ldpc_early_termination = 1u;
    p.ext_n_iterations           = 8u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    EXPECT_FLOAT_EQ(ue.foForgetCoeff,       0.0f);
    EXPECT_EQ(ue.ldpcEarlyTerminationPerUe, 0u);
    EXPECT_EQ(ue.ldpcMaxNumItrPerUe,        10u);
}

TEST_F(PuschParseTest, ME5_DftSofdmFallback_FromMaintenance)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.transform_precoding        = 0u;  // transform precoding ON
    p.pdu_bitmap                 = 0u;  // no dftsOfdm section (bit 3 unset)
    p.group_or_sequence_hopping  = 1u;  // comes from maintenance
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    EXPECT_EQ(ue.enableTfPrcd,            1u);
    EXPECT_EQ(ue.groupOrSequenceHopping,  1u);
    EXPECT_EQ(ue.N_symb_slot,             14u);
}

// ---------------------------------------------------------------------------
// Beamforming / fronthaul (BF-*) — setup_beamforming_and_fh + update_fh_params
// ---------------------------------------------------------------------------

// BF-1: explicit digital-BF interfaces set the uplink-stream count and a
// contiguous port mask; the cursor must skip the inline beam indices so the
// PUSCH extension that follows is still parsed correctly.
TEST_F(PuschParseTest, BF1_DigBfInterfaces_StreamsPortMaskAndCursor)
{
    view_.mmimo_enabled_           = true;
    view_.enable_weighted_avg_cfo_ = true;  // read extension -> validates cursor
    setup_cell();

    PuschPduParams p{};
    p.num_prgs            = 1u;
    p.dig_bf_interfaces   = 4u;
    p.ext_fo_forget_coeff = 50u;   // -> 0.5f only if cursor landed correctly
    p.ext_n_iterations    = 8u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    EXPECT_EQ(prm->ue_grp_info[0].nUplinkStreams, 4u);
    EXPECT_FLOAT_EQ(prm->ue_info[0].foForgetCoeff, 0.5f);
    EXPECT_EQ(prm->ue_info[0].ldpcMaxNumItrPerUe,  8u);

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.portMask, (static_cast<uint64_t>(1u) << 4u) - 1u);
}

// FH-DIRECT: in fapi_to_cplane_direct mode the framework builds the O-RAN C-plane
// directly from FAPI, so the parser must not touch the cell command's FH
// sym_prb_info.  The PRB metadata is still needed by the UL Order kernel, so the
// fill is redirected to order_sym_prb_info() instead of being dropped.  mMIMO
// stream accounting on the cuPHY UE-group params runs in both modes.  Contrast
// with BF-1 (same PDU, direct mode off) which records the entry on the cell
// command itself.
TEST_F(PuschParseTest, FhRedirectedToOrderScratchInFapiToCplaneDirectMode)
{
    view_.mmimo_enabled_                  = true;
    view_.fapi_to_cplane_direct_enabled_  = true;
    view_.enable_weighted_avg_cfo_        = true;  // read extension -> validates cursor
    setup_cell();

    PuschPduParams p{};
    p.num_prgs            = 1u;
    p.dig_bf_interfaces   = 4u;
    p.ext_fo_forget_coeff = 50u;   // -> 0.5f only if cursor landed correctly
    p.ext_n_iterations    = 8u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    // Stream accounting and the extension parse still run in direct mode: the
    // cursor must remain byte-aligned past the beam section.
    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    EXPECT_EQ(prm->ue_grp_info[0].nUplinkStreams, 4u);
    EXPECT_FLOAT_EQ(prm->ue_info[0].foForgetCoeff, 0.5f);
    EXPECT_EQ(prm->ue_info[0].ldpcMaxNumItrPerUe,  8u);

    // The cell command's FH buffer is untouched...
    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(sp->prbs_size, 0u);

    // ...and the same entry BF-1 records on the cell command landed on the UL
    // Order scratch instead, with the port mask still derived from dig_bf_interfaces.
    auto* order = view_.order_sym_prb_info(0u);
    ASSERT_NE(order, nullptr);
    ASSERT_EQ(order->prbs_size, 1u);
    EXPECT_EQ(order->prbs[0].common.portMask, (static_cast<uint64_t>(1u) << 4u) - 1u);
}

// BF-1b: a full-width 64-antenna deployment (dig_bf_interfaces == 64) must yield
// an all-ones port mask. Regresses the UB in (1u << 64) - 1u, which on x86 wraps
// the shift count to 0 and silently produced portMask == 0.
TEST_F(PuschParseTest, BF1_DigBfInterfaces64_PortMaskAllOnes)
{
    view_.mmimo_enabled_ = true;
    setup_cell();

    PuschPduParams p{};
    p.num_prgs          = 1u;
    p.dig_bf_interfaces = 64u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.portMask, ~static_cast<uint64_t>(0u));
}

// BF-2: dynamic BFW (dig_bf_interfaces == 0) under mMIMO derives the port mask
// from the DMRS ports (scid 0, nlAbove16 0 -> mask == dmrs_ports).
TEST_F(PuschParseTest, BF2_DynamicBfw_PortMaskFromDmrsPorts)
{
    view_.mmimo_enabled_ = true;
    setup_cell();

    PuschPduParams p{};
    p.dig_bf_interfaces = 0u;
    p.dmrs_ports        = 0x3u;  // ports 0,1
    p.scid              = 0u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.portMask, 0x3u);
}

// BF-3: with mMIMO disabled the uplink-stream count is left untouched and no
// port mask is derived; the cursor still skips the inline beam indices.
TEST_F(PuschParseTest, BF3_MmimoDisabled_NoUplinkStreamUpdate)
{
    view_.mmimo_enabled_           = false;
    view_.enable_weighted_avg_cfo_ = true;
    setup_cell();

    PuschPduParams p{};
    p.num_prgs            = 1u;
    p.dig_bf_interfaces   = 2u;
    p.ext_fo_forget_coeff = 50u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    EXPECT_EQ(prm->ue_grp_info[0].nUplinkStreams, 0u);
    EXPECT_FLOAT_EQ(prm->ue_info[0].foForgetCoeff, 0.5f);

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.portMask, 0u);
}

// BF-4: a second UE joining an existing mMIMO dynamic-BFW group only OR-merges
// its port mask into the existing PRB entry — no new PRB entry is appended.
TEST_F(PuschParseTest, BF4_ExistingGroupDynamicBfw_MergesPortMask)
{
    view_.mmimo_enabled_ = true;
    setup_cell();

    PuschPduParams p1{};
    p1.rnti             = 0x1u;
    p1.dig_bf_interfaces = 0u;
    p1.dmrs_ports       = 0x1u;  // port 0
    EXPECT_TRUE(parse(build_pusch_pdu_body(p1)));

    PuschPduParams p2 = p1;
    p2.rnti       = 0x2u;
    p2.dmrs_ports = 0x2u;        // port 1; same allocation -> same group
    EXPECT_TRUE(parse(build_pusch_pdu_body(p2)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    EXPECT_EQ(prm->cell_grp_info.nUeGrps, 1u);
    EXPECT_EQ(prm->ue_grp_info[0].nUes,   2u);

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(sp->prbs_size, 1u);                  // no new PRB for the 2nd UE
    EXPECT_EQ(sp->prbs[0].common.portMask, 0x3u);  // 0x1 | 0x2
}

// BF-5: a stale BFW buffer (not exactly one slot behind) must NOT attach the
// BFW extension — extType stays 0.
TEST_F(PuschParseTest, BF5_BfwStale_ExtensionSkipped)
{
    view_.mmimo_enabled_ = true;
    view_.bf_enabled_    = true;

    uint8_t header = static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_BUSY);
    slot_command_api::bfw_coeff_mem_info_t mem{};
    mem.header  = &header;
    mem.sfn     = 0u;
    mem.slot    = 0u;   // == current slot (0,0) -> not fresh
    mem.nGnbAnt = 4u;
    view_.bfw_coeff_mem_info_ = &mem;

    setup_cell();

    PuschPduParams p{};
    p.dig_bf_interfaces = 0u;
    p.num_prgs          = 1u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));  // current slot (0,0)

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.extType, 0u);
}

// BF-6: a fresh BFW buffer (exactly one slot behind) attaches the dynamic-BFW
// extension (extType 11) and copies the coefficient buffer descriptors.
TEST_F(PuschParseTest, BF6_BfwFresh_AppliesCoefficients)
{
    view_.mmimo_enabled_ = true;
    view_.bf_enabled_    = true;

    uint8_t header = static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_BUSY);
    uint8_t hbuf   = 0u;
    uint8_t dbuf   = 0u;
    slot_command_api::bfw_coeff_mem_info_t mem{};
    mem.header  = &header;
    mem.sfn     = 0u;
    mem.slot    = 0u;   // one slot behind current (0,1)
    mem.nGnbAnt = 8u;
    mem.buff_addr_chunk_h[0] = &hbuf;
    mem.buff_addr_chunk_d[0] = &dbuf;
    view_.bfw_coeff_mem_info_ = &mem;

    setup_cell();

    PuschPduParams p{};
    p.dig_bf_interfaces = 0u;
    p.num_prgs          = 2u;
    p.prg_size          = 4u;
    const auto body = build_pusch_pdu_body(p);

    scf_5g_fapi::PuschPduParser<MockUlModuleView> parser{view_};
    EXPECT_TRUE(parser.parse(0u, 1u, as_pdu(body)));  // current slot 1, BFW slot 0 -> fresh

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    const auto& prb = sp->prbs[0];
    EXPECT_EQ(prb.common.extType,                   11u);
    EXPECT_EQ(prb.bfwCoeff_buf_info.num_prgs,        2u);
    EXPECT_EQ(prb.bfwCoeff_buf_info.prg_size,        4u);
    EXPECT_EQ(prb.bfwCoeff_buf_info.nGnbAnt,         8u);
    EXPECT_EQ(prb.bfwCoeff_buf_info.header,          &header);
    EXPECT_EQ(prb.bfwCoeff_buf_info.p_buf_bfwCoef_h, &hbuf);
    EXPECT_EQ(prb.bfwCoeff_buf_info.p_buf_bfwCoef_d, &dbuf);
}

// A-3 regression (Kobi, MR-5330): grp.puschStartSym is copied verbatim from the
// wire start_symbol_index.  When it points past the last OFDM symbol of the slot,
// the dynamic-BFW port-mask merge branch must bound-check before indexing the
// fixed-size symbol map, instead of reading out of bounds.  Two UEs share a
// group (so the 2nd UE reaches the merge branch) with an out-of-range start
// symbol; the parser must not read OOB and must skip the merge, leaving the 1st
// UE's port mask intact.
TEST_F(PuschParseTest, MergePortMask_OutOfRangeStartSymbol_SkipsSafely)
{
    view_.mmimo_enabled_ = true;
    setup_cell();

    // Past the last OFDM symbol index (k_n_symb_per_slot - 1 == 13).
    constexpr uint8_t k_oob_start = 20u;

    PuschPduParams p1{};
    p1.rnti               = 0x1u;
    p1.dig_bf_interfaces  = 0u;    // dynamic BFW -> merge branch for the 2nd UE
    p1.dmrs_ports         = 0x1u;  // port 0
    p1.start_symbol_index = k_oob_start;
    p1.num_of_symbols     = 1u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p1)));

    PuschPduParams p2 = p1;
    p2.rnti       = 0x2u;
    p2.dmrs_ports = 0x2u;          // port 1; same allocation -> same group
    EXPECT_TRUE(parse(build_pusch_pdu_body(p2)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    EXPECT_EQ(prm->ue_grp_info[0].nUes, 2u);

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    // Out-of-range start symbol -> merge skipped; only the 1st UE's port survives.
    EXPECT_EQ(sp->prbs[0].common.portMask, 0x1u);
}

// A-1 regression: the published CSI-Part2 counts must reflect what was actually
// written into the fixed-size cuPHY backing arrays (capped), never the raw wire
// value, so a downstream [0, count) walk cannot read past the initialized window.

TEST_F(PuschParseTest, ME6_Csip2_ReportCountClampedToMax)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.pdu_bitmap         = 0x2u;  // uci
    p.has_uci            = true;
    p.csi_part2_signaled = true;

    // Signal one more report than the per-UE backing window can hold.
    constexpr uint16_t k_over = static_cast<uint16_t>(CUPHY_MAX_N_CSI2_REPORTS_PER_UE + 1);
    for (uint16_t k = 0u; k < k_over; ++k)
    {
        p.csip2_parts.push_back(Csip2PartDesc{
            /*offsets=*/{static_cast<uint16_t>(100u + k)},
            /*sizes=*/{1u},
            /*mapIdx=*/k});
    }
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    ASSERT_NE(ue.pUciPrms, nullptr);

    // Published count is capped at capacity, not the raw numPart2s (== 17).
    EXPECT_EQ(ue.pUciPrms->nCsi2Reports, CUPHY_MAX_N_CSI2_REPORTS_PER_UE);
    ASSERT_NE(ue.pUciPrms->pCalcCsi2SizePrms, nullptr);

    // In-range reports are still populated correctly (first and last in-range entry).
    EXPECT_EQ(ue.pUciPrms->pCalcCsi2SizePrms[0].prmOffsets[0], 100u);
    const auto& last = ue.pUciPrms->pCalcCsi2SizePrms[CUPHY_MAX_N_CSI2_REPORTS_PER_UE - 1];
    EXPECT_EQ(last.prmOffsets[0],
              static_cast<uint16_t>(100u + (CUPHY_MAX_N_CSI2_REPORTS_PER_UE - 1)));
    EXPECT_EQ(last.csi2sizeMapIdx,
              static_cast<uint16_t>(CUPHY_MAX_N_CSI2_REPORTS_PER_UE - 1));
}

TEST_F(PuschParseTest, ME7_Csip2_Part1CountClampedToMax)
{
    view_.enable_weighted_avg_cfo_ = false;
    setup_cell();

    PuschPduParams p{};
    p.pdu_bitmap         = 0x2u;
    p.has_uci            = true;
    p.csi_part2_signaled = true;

    // A single report carrying more Part-1 params than prmOffsets[]/prmSizes[] hold.
    Csip2PartDesc part{};
    for (uint8_t k = 0u; k < static_cast<uint8_t>(CUPHY_MAX_N_CSI1_PRMS + 1); ++k)
    {
        part.param_offsets.push_back(static_cast<uint16_t>(11u + k));
        part.param_sizes.push_back(static_cast<uint8_t>(1u + k));
    }
    part.part2_size_map_index = 9u;
    p.csip2_parts = {part};
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    ASSERT_NE(ue.pUciPrms, nullptr);
    EXPECT_EQ(ue.pUciPrms->nCsi2Reports, 1u);

    const auto& r0 = ue.pUciPrms->pCalcCsi2SizePrms[0];
    // Published Part-1 count is capped, not the raw numPart1Params (== 5).
    EXPECT_EQ(r0.nPart1Prms, CUPHY_MAX_N_CSI1_PRMS);
    EXPECT_EQ(r0.prmOffsets[0], 11u);
    EXPECT_EQ(r0.prmOffsets[1], 12u);
    EXPECT_EQ(r0.prmOffsets[2], 13u);
    EXPECT_EQ(r0.prmOffsets[3], 14u);
    EXPECT_EQ(r0.prmSizes[0],   1u);
    EXPECT_EQ(r0.prmSizes[1],   2u);
    EXPECT_EQ(r0.prmSizes[2],   3u);
    EXPECT_EQ(r0.prmSizes[3],   4u);
    EXPECT_EQ(r0.csi2sizeMapIdx, 9u);
}

// A-3 regression: the inline beam_idx[] array holds numPRGs * digBFInterfaces
// entries.  With numPRGs > 1 the beamforming cursor must skip the full array, or
// the maintenance / extension tail is read from the wrong offset.  Setting
// numPRGs = 2 (extra digBFInterfaces uint16_t entries) and checking the
// extension fields verifies the cursor advanced past every beam entry.
TEST_F(PuschParseTest, ME8_Beamforming_MultiPrgCursorAdvance)
{
    view_.enable_weighted_avg_cfo_ = true;
    setup_cell();

    // The extension (weighted-avg CFO / LDPC) is the last section on the wire —
    // after beamforming, maintenance and the optional CSI-Part2 — so reading it
    // correctly proves the numPRGs=2 beamforming cursor advanced past every
    // beam_idx[] entry (a short advance would misparse these fields).
    PuschPduParams p{};
    p.transform_precoding        = 1u;  // disable DFT-s-OFDM fallback
    p.num_prgs                   = 2u;  // > 1: extra beam_idx[] entries inline
    p.dig_bf_interfaces          = 3u;  // 2 * 3 = 6 uint16_t beam entries
    p.ext_fo_forget_coeff        = 50u; // -> 0.5f (extension, after maintenance)
    p.ext_ldpc_early_termination = 1u;
    p.ext_n_iterations           = 8u;
    EXPECT_TRUE(parse(build_pusch_pdu_body(p)));

    auto* prm = params();
    ASSERT_NE(prm, nullptr);
    const auto& ue = prm->ue_info[0];
    EXPECT_FLOAT_EQ(ue.foForgetCoeff,       0.5f);
    EXPECT_EQ(ue.ldpcEarlyTerminationPerUe, 1u);
    EXPECT_EQ(ue.ldpcMaxNumItrPerUe,        8u);
}

// ---------------------------------------------------------------------------
// Group FH-SS — single-sector fronthaul symbol map (B-9)
// ---------------------------------------------------------------------------

// FH-SS-1: in single-sector mode the UL C-plane section's symbol span and start
// come from the cell's TDD slot detail (max_ul_symbols / start_sym_ul), the PRB
// covers the full UL bandwidth (startPrbc == 0), and the PRB index is listed at
// its start symbol only (legacy update_prb_sym_list numSym == 1).
TEST_F(PuschParseTest, FhSingleSect_SlotDetailDrivesSymbolSpanAndStart)
{
    view_.ru_type_                    = SINGLE_SECT_MODE;
    view_.cell_view_.ul_max_symbols_  = 10u;
    view_.cell_view_.ul_start_symbol_ = 3u;
    setup_cell();

    // Per-UE start/num/PRB are intentionally ignored in single-sector.
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);

    const auto& prb = sp->prbs[0];
    EXPECT_EQ(prb.common.numSymbols, 10u);  // from slot_detail->max_ul_symbols
    EXPECT_EQ(prb.common.startPrbc,   0u);  // full UL bandwidth section
    EXPECT_EQ(sp->start_symbol_ul,    3u);  // from slot_detail->start_sym_ul

    // Legacy single-symbol registration: listed at start symbol only.
    const auto& at_start = sp->symbols[3][slot_command_api::channel_type::PUSCH];
    ASSERT_EQ(at_start.size(), 1u);
    EXPECT_EQ(at_start[0], 0u);
    EXPECT_TRUE(sp->symbols[4][slot_command_api::channel_type::PUSCH].empty());
}

// FH-SS-2: when slot detail is unavailable (mock reports 0), single-sector falls
// back to a full slot (numSymbols == 14) starting at symbol 0 — preserving the
// prior hardcoded behavior.
TEST_F(PuschParseTest, FhSingleSect_NoSlotDetailFallsBackToFullSlot)
{
    view_.ru_type_ = SINGLE_SECT_MODE;
    // ul_max_symbols_ / ul_start_symbol_ left at 0 -> fallback path.
    setup_cell();

    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    ASSERT_EQ(sp->prbs_size, 1u);
    EXPECT_EQ(sp->prbs[0].common.numSymbols, 14u);  // k_n_symb_per_slot fallback
    EXPECT_EQ(sp->start_symbol_ul,            0u);

    const auto& at_start = sp->symbols[0][slot_command_api::channel_type::PUSCH];
    ASSERT_EQ(at_start.size(), 1u);
    EXPECT_EQ(at_start[0], 0u);
}

// FH-SS-3: single-sector mode keeps the one-full-bandwidth-entry-per-slot invariant
// even when the entry is registered at a non-zero start symbol (ul_start_symbol > 0).
// The early-out guard probes start_symbol_ul (where append_prb_to_symbols wrote the
// entry), not a hardcoded symbol 0, so a second UE group must NOT append a duplicate
// PRB entry. Regresses the symbol-0 probe that left the guard unable to fire here.
TEST_F(PuschParseTest, FhSingleSect_SecondGroupDoesNotDuplicateEntryAtNonZeroStart)
{
    view_.ru_type_                    = SINGLE_SECT_MODE;
    view_.cell_view_.ul_max_symbols_  = 10u;
    view_.cell_view_.ul_start_symbol_ = 3u;  // entry lands at symbol 3, not 0
    setup_cell();

    // Two distinct allocations -> two UE groups; each new group runs the FH path.
    EXPECT_TRUE(parse(body_with_alloc(2u, 12u, 10u, 50u)));
    EXPECT_TRUE(parse(body_with_alloc(3u, 12u, 10u, 50u)));

    auto* p = params();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->cell_grp_info.nUeGrps, 2u);  // two UE groups were formed

    auto* sp = view_.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(sp, nullptr);
    EXPECT_EQ(sp->prbs_size,       1u);  // second group early-outs: only one entry
    EXPECT_EQ(sp->start_symbol_ul, 3u);
    EXPECT_EQ(sp->symbols[3][slot_command_api::channel_type::PUSCH].size(), 1u);
}
