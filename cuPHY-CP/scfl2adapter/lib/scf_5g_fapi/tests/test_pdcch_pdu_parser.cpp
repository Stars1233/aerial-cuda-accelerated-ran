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
 * @file test_pdcch_pdu_parser.cpp
 * @brief Unit tests for PdcchPduParser.
 *
 * Covers the parser-level aggregation path: a real serialized PDCCH PDU is
 * parsed into slot_command::cell_groups.pdcch without requiring PHY_module or
 * CUDA-backed channel workers.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "scf_5g_fapi_pdcch_pdu_parser.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi::test
{

void reset_pdcch_sym_prb_info_stub_state() noexcept;
[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_call_count() noexcept;
[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_bandwidth() noexcept;
[[nodiscard]] int32_t pdcch_sym_prb_info_stub_cell_index() noexcept;
[[nodiscard]] bool pdcch_sym_prb_info_stub_mmimo_enabled() noexcept;
[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_num_dl_dci() noexcept;
[[nodiscard]] uint8_t pdcch_sym_prb_info_stub_coreset_n_dci() noexcept;
[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_coreset_dci_start_idx() noexcept;
[[nodiscard]] uint16_t pdcch_sym_prb_info_stub_first_dci_rnti() noexcept;
[[nodiscard]] uint32_t pdcch_sym_prb_info_stub_first_dci_payload_bits() noexcept;
[[nodiscard]] std::size_t pdcch_sym_prb_info_stub_first_dci_offset() noexcept;

} // namespace scf_5g_fapi::test

namespace {

struct MockCellView final
{
    uint16_t           bwp_size_{106};
    cuphyCellStatPrm_t stat_prm_{};

    MockCellView()
    {
        stat_prm_.nPrbDlBwp = bwp_size_;
    }

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return bwp_size_; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }
};

static_assert(scf_5g_fapi::CellView<MockCellView>,
              "MockCellView must satisfy CellView");

struct MockDlModuleView final
{
    bool pm_enabled_    {false};
    bool bf_enabled_    {false};
    bool mmimo_enabled_ {true};
    bool fapi_to_cplane_direct_enabled_{false};
    int  static_pdcch_slot_{-1};
    int32_t  carrier_id_{3};
    uint16_t phy_cell_id_{77};
    bool use_per_cell_ids_{false};
    std::array<int32_t, 4>  carrier_ids_{3, 4, 5, 6};
    std::array<uint16_t, 4> phy_cell_ids_{77, 78, 79, 80};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable scf_5g_fapi::pm_weight_map_t       pm_map_{};
    mutable nv::phy_config_option              config_opt_{};
    mutable nv::phy_config                     phy_config_{};
    mutable nv::slot_limit_cell_error_t        limit_errors_{};
    MockCellView                               cell_view_{};

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }

    [[nodiscard]] scf_5g_fapi::pm_weight_map_t& pm_map() const noexcept { return pm_map_; }
    [[nodiscard]] bool pm_enabled() const noexcept { return pm_enabled_; }
    [[nodiscard]] bool bf_enabled() const noexcept { return bf_enabled_; }

    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept { return nullptr; }

    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] const nv::phy_config& phy_config([[maybe_unused]] uint32_t cell_idx) const noexcept
    {
        return phy_config_;
    }
    [[nodiscard]] bool is_fapi_to_cplane_direct_enabled() const noexcept
    {
        return fapi_to_cplane_direct_enabled_;
    }
    [[nodiscard]] int staticPdcchSlotNum() const noexcept { return static_pdcch_slot_; }
    [[nodiscard]] int staticPdschSlotNum() const noexcept { return -1; }

    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t carrier_id(uint32_t cell_idx) const noexcept
    {
        return use_per_cell_ids_ ? carrier_ids_[cell_idx] : carrier_id_;
    }

    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_idx) const noexcept
    {
        return use_per_cell_ids_ ? phy_cell_ids_[cell_idx] : phy_cell_id_;
    }

    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }

    [[nodiscard]] uint16_t num_dl_prb(uint32_t) const noexcept
    {
        return cell_view_.num_dl_prb();
    }

    /**
     * Return a CellView for @p cell_idx with @c phyCellId stamped for the parser.
     *
     * @c PdcchPduParser::parse pushes @c cell_params().phyCellId into
     * @c phy_cell_index_list (not @c ModuleView::phy_cell_id()), so this stamps the
     * per-cell id onto the CellView the parser actually reads. Without that, the
     * zero-initialized @c cuphyCellStatPrm_t leaves @c phy_cell_index_list entries
     * at 0 and the assertions below fail.
     *
     * @param[in] cell_idx  Logical cell index used to resolve @c phy_cell_id().
     * @param[in] slot_ind  Slot indication required by the CellView concept; unused.
     *
     * @return Copy of @c cell_view_ with @c stat_prm_.phyCellId set from
     *         @c phy_cell_id(cell_idx).
     */
    [[nodiscard]] MockCellView cell_view(const uint32_t cell_idx,
        [[maybe_unused]] const slot_command_api::slot_indication& slot_ind) const noexcept
    {
        MockCellView cv = cell_view_;
        cv.stat_prm_.phyCellId = phy_cell_id(cell_idx);
        return cv;
    }

    void publish_dl_pdsch_stats(const scf_5g_fapi::DlPdschStatsBatch&) const noexcept {}
};

static_assert(scf_5g_fapi::DlModuleView<MockDlModuleView>,
              "MockDlModuleView must satisfy DlModuleView");

struct PdcchDciParams
{
    uint16_t rnti              {0x1234};
    uint16_t scrambling_id     {0x0042};
    uint16_t scrambling_rnti   {0x5678};
    uint8_t  cce_index         {1};
    uint8_t  aggregation_level {4};

    uint16_t num_prgs          {1};
    uint16_t prg_size          {48};
    uint8_t  dig_bf_interfaces {0};
    std::vector<uint16_t> pm_idx_and_beam_idx{0u};

    uint8_t beta_pdcch_1_0 {0};
#ifdef SCF_FAPI_10_04
    int8_t power_control_offset_ss_profile_nr {-3};
#else
    uint8_t power_control_offset_ss {4};
#endif

    uint16_t payload_size_bits {20};
    std::vector<uint8_t> payload{0xA5u, 0x5Au, 0xC3u};
};

struct PdcchPduParams
{
    uint16_t bwp_size     {106};
    uint16_t bwp_start    {12};
    uint8_t  scs          {1};
    uint8_t  cyclic_prefix{0};
    uint8_t  start_sym    {2};
    uint8_t  duration_sym {3};
    std::array<uint8_t, 6> freq_domain_resource{
        0x80u, 0x40u, 0x20u, 0x10u, 0x08u, 0x04u};
    uint8_t  cce_reg_mapping_type {1};
    uint8_t  reg_bundle_size      {6};
    uint8_t  interleaver_size     {3};
    uint8_t  coreset_type         {1};
    uint16_t shift_index          {11};
    uint8_t  precoder_granularity {0};
    std::vector<PdcchDciParams> dcis{};
};

template <typename T>
void append_object(std::vector<uint8_t>& buf, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);

    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    buf.insert(buf.end(), bytes, bytes + sizeof(T));
}

void append_u16(std::vector<uint8_t>& buf, uint16_t value)
{
    append_object(buf, value);
}

[[nodiscard]] constexpr uint32_t ceil_div(uint32_t numerator, uint32_t denominator) noexcept
{
    return (numerator + denominator - 1u) / denominator;
}

[[nodiscard]] uint64_t freq_domain_resource(
    const std::array<uint8_t, 6>& freq_domain_resource) noexcept
{
    uint64_t resource = 0u;
    for (std::size_t i = 0u; i < freq_domain_resource.size(); ++i) {
        resource |= static_cast<uint64_t>(freq_domain_resource[i])
            << (56u - static_cast<unsigned>(i * 8u));
    }
    return resource;
}

[[nodiscard]] std::vector<uint8_t> build_pdcch_pdu_body(const PdcchPduParams& p)
{
    std::vector<uint8_t> buf;

    scf_fapi_pdcch_pdu_t pdu{};
    pdu.bwp.bwp_size        = p.bwp_size;
    pdu.bwp.bwp_start       = p.bwp_start;
    pdu.bwp.scs             = p.scs;
    pdu.bwp.cyclic_prefix   = p.cyclic_prefix;
    pdu.start_sym_index     = p.start_sym;
    pdu.duration_sym        = p.duration_sym;
    std::memcpy(pdu.freq_domain_resource, p.freq_domain_resource.data(),
                p.freq_domain_resource.size());
    pdu.cce_reg_mapping_type = p.cce_reg_mapping_type;
    pdu.reg_bundle_size      = p.reg_bundle_size;
    pdu.interleaver_size     = p.interleaver_size;
    pdu.coreset_type         = p.coreset_type;
    pdu.shift_index          = p.shift_index;
    pdu.precoder_granularity = p.precoder_granularity;
    pdu.num_dl_dci           = static_cast<uint16_t>(p.dcis.size());
    append_object(buf, pdu);

    for (const auto& dci_params : p.dcis) {
        scf_fapi_dl_dci_t dci{};
        dci.rnti              = dci_params.rnti;
        dci.scrambling_id     = dci_params.scrambling_id;
        dci.scrambling_rnti   = dci_params.scrambling_rnti;
        dci.cce_index         = dci_params.cce_index;
        dci.aggregation_level = dci_params.aggregation_level;
        append_object(buf, dci);

        scf_fapi_tx_precoding_beamforming_t pc_bf{};
        pc_bf.num_prgs          = dci_params.num_prgs;
        pc_bf.prg_size          = dci_params.prg_size;
        pc_bf.dig_bf_interfaces = dci_params.dig_bf_interfaces;
        append_object(buf, pc_bf);
        for (const auto entry : dci_params.pm_idx_and_beam_idx) {
            append_u16(buf, entry);
        }

        scf_fapi_pdcch_tx_power_info_t power{};
        power.beta_pdcch_1_0 = dci_params.beta_pdcch_1_0;
#ifdef SCF_FAPI_10_04
        power.power_control_offset_ss_profile_nr =
            dci_params.power_control_offset_ss_profile_nr;
#else
        power.power_control_offset_ss = dci_params.power_control_offset_ss;
#endif
        append_object(buf, power);

        scf_fapi_pdcch_dci_payload_t payload_info{};
        payload_info.payload_size_bits = dci_params.payload_size_bits;
        append_object(buf, payload_info);

        const auto payload_bytes = ceil_div(dci_params.payload_size_bits, 8u);
        buf.insert(buf.end(), dci_params.payload.begin(),
                   dci_params.payload.begin() + payload_bytes);
    }

    return buf;
}

struct FapiDlPdcchMsg
{
    std::vector<uint8_t> buf;
    nv::phy_mac_msg_desc desc{};

    FapiDlPdcchMsg(uint16_t sfn, uint16_t slot,
                   const std::vector<uint8_t>& pdcch_body,
                   uint16_t cell_id)
        : FapiDlPdcchMsg{sfn, slot, std::vector<std::vector<uint8_t>>{pdcch_body}, cell_id}
    {}

    FapiDlPdcchMsg(uint16_t sfn, uint16_t slot,
                   const std::vector<std::vector<uint8_t>>& pdcch_bodies,
                   uint16_t cell_id)
    {
        const std::size_t hdr_sz     = sizeof(scf_fapi_header_t);
        const std::size_t req_sz     = sizeof(scf_fapi_dl_tti_req_t);
        const std::size_t gen_hdr_sz = sizeof(scf_fapi_generic_pdu_info_t);
        std::size_t total = hdr_sz + req_sz;
        for (const auto& pdcch_body : pdcch_bodies) {
            total += gen_hdr_sz + pdcch_body.size();
        }

        buf.resize(total, 0u);

        auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(buf.data() + hdr_sz);
        req->sfn      = sfn;
        req->slot     = slot;
        req->num_pdus = static_cast<uint16_t>(pdcch_bodies.size());
#ifdef SCF_FAPI_10_04
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDCCH] = req->num_pdus;
        for (const auto& pdcch_body : pdcch_bodies) {
            const auto* pdcch =
                aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(pdcch_body.data());
            req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_DlDCIs] += pdcch->num_dl_dci;
        }
#endif

        auto* cur = buf.data() + hdr_sz + req_sz;
        for (const auto& pdcch_body : pdcch_bodies) {
            auto* gen = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(cur);
            gen->pdu_type = static_cast<uint16_t>(DL_TTI_PDU_TYPE_PDCCH);
            gen->pdu_size = static_cast<uint16_t>(gen_hdr_sz + pdcch_body.size());

            std::memcpy(cur + gen_hdr_sz, pdcch_body.data(), pdcch_body.size());
            cur += gen_hdr_sz + pdcch_body.size();
        }

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = cell_id;
    }
};

[[nodiscard]] float expected_beta(const PdcchDciParams& dci) noexcept
{
#ifdef SCF_FAPI_10_04
    return static_cast<float>(
        std::pow(10.0, dci.power_control_offset_ss_profile_nr / 20.0));
#else
    return static_cast<float>(
        std::pow(10.0, (dci.power_control_offset_ss - 1) * 3.0 / 20.0));
#endif
}

void expect_payload_prefix(const slot_command_api::dci_payload_t& actual,
                           const std::vector<uint8_t>& expected,
                           uint16_t payload_size_bits)
{
    const auto payload_bytes = ceil_div(payload_size_bits, 8u);
    ASSERT_GE(expected.size(), payload_bytes);
    for (uint32_t i = 0u; i < payload_bytes; ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "payload byte " << i;
    }
}

} // namespace

// ===========================================================================
// Purpose:
//   Verify the real PDCCH parser-level aggregation path.
//
// Flow mirrors the PDSCH parser UT style:
//   1. Build a serialized FAPI PDCCH PDU body.
//   2. Call setup_cell() on the real PdcchPduParser.
//   3. Parse the PDU through the real parser.
//   4. Assert the populated slot_command::cell_groups.pdcch aggregate.
//
// Coverage:
//   - setup_cell() cell_index_list / phy_cell_index_list bookkeeping
//   - coreset count and coreset dynamic fields
//   - DCI count and DCI dynamic fields
//   - beta_qam / beta_dmrs conversion from FAPI power-control fields
//   - DCI payload byte copy into csets_group.payloads
//
// This intentionally keeps temporary sym_prb_info disabled; that path needs a
// separate focused UT/integration test because it calls the fronthaul bridge.
// ===========================================================================
TEST(PdcchPduParser, Parse_PopulatesCellGroupsPdcch)
{
    // k_sfn / k_slot are arbitrary valid values (k_slot is verified to propagate
    // to coreset.slot_number). k_cell_index is intentionally non-zero so the test
    // exercises per-cell index bookkeeping (carrier_id / phy_cell_id mapping and
    // slotBufferIdx) and guards against code that assumes cell index 0.
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.use_per_cell_ids_ = true;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchPduParams input{};
    PdcchDciParams first_dci{};
    first_dci.rnti              = 0x1111u;
    first_dci.scrambling_id     = 0x002Au;
    first_dci.scrambling_rnti   = 0x2222u;
    first_dci.cce_index         = 3u;
    first_dci.aggregation_level = 8u;
    first_dci.payload_size_bits = 20u;
    first_dci.payload           = {0xA5u, 0x5Au, 0xC3u};

    PdcchDciParams second_dci{};
    second_dci.rnti              = 0x3333u;
    second_dci.scrambling_id     = 0x0044u;
    second_dci.scrambling_rnti   = 0x5555u;
    second_dci.cce_index         = 9u;
    second_dci.aggregation_level = 4u;
#ifdef SCF_FAPI_10_04
    second_dci.power_control_offset_ss_profile_nr = 6;
#else
    second_dci.power_control_offset_ss = 2u;
#endif
    second_dci.payload_size_bits = 16u;
    second_dci.payload           = {0x33u, 0xCCu};

    input.dcis = {first_dci, second_dci};
    const auto body = build_pdcch_pdu_body(input);
    const auto& pdu = *aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(body.data());

    uint8_t test_mode = 0u;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
    test_mode = 1u;
#endif

    scf_5g_fapi::PdcchPduParser<MockDlModuleView> parser{view};
    parser.setup_cell(static_cast<uint32_t>(k_cell_index), test_mode, false);

    const auto* setup_params = view.slot_cmd_.cell_groups.pdcch.get();
    ASSERT_NE(setup_params, nullptr);
    ASSERT_TRUE(setup_params->cell_index_list.empty());
    ASSERT_TRUE(setup_params->phy_cell_index_list.empty());

    ASSERT_TRUE(parser.parse(k_sfn, k_slot, pdu));

    const auto* params = view.slot_cmd_.cell_groups.pdcch.get();
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    ASSERT_EQ(params->phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list[0], view.carrier_id(k_cell_index));
    EXPECT_EQ(params->phy_cell_index_list[0], view.phy_cell_id(k_cell_index));

    const auto& group = params->csets_group;
    ASSERT_EQ(group.nCoresets, 1u);
    ASSERT_EQ(group.nDcis, input.dcis.size());

    const auto& coreset = group.csets[0];
    EXPECT_EQ(coreset.n_f, static_cast<uint32_t>(view.cell_view_.bwp_size_) * 12u);
    EXPECT_EQ(coreset.slot_number, k_slot);
    EXPECT_EQ(coreset.start_rb, input.bwp_start);
    EXPECT_EQ(coreset.start_sym, input.start_sym);
    EXPECT_EQ(coreset.n_sym, input.duration_sym);
    EXPECT_EQ(coreset.bundle_size, input.reg_bundle_size);
    EXPECT_EQ(coreset.interleaver_size, input.interleaver_size);
    EXPECT_EQ(coreset.shift_index, input.shift_index);
    EXPECT_EQ(coreset.interleaved, input.cce_reg_mapping_type);
    EXPECT_EQ(coreset.freq_domain_resource,
              freq_domain_resource(input.freq_domain_resource));
    EXPECT_EQ(coreset.coreset_type, input.coreset_type);
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
    EXPECT_EQ(coreset.testModel, test_mode);
#else
    EXPECT_EQ(coreset.testModel, 0u);
#endif
    EXPECT_EQ(coreset.nDci, static_cast<uint8_t>(input.dcis.size()));
    EXPECT_EQ(coreset.dciStartIdx, 0u);
    EXPECT_EQ(coreset.slotBufferIdx, static_cast<uint32_t>(view.carrier_id(k_cell_index)));

    // Table-driven: input.dcis is the table of expected cases; assert each
    // parsed DCI against its corresponding input row. SCOPED_TRACE reports the
    // failing index, and adding a DCI to input.dcis is covered automatically.
    for (std::size_t i = 0u; i < input.dcis.size(); ++i) {
        SCOPED_TRACE(testing::Message() << "dci index " << i);
        const auto& expected = input.dcis[i];
        const auto& actual   = group.dcis[i];
        EXPECT_EQ(actual.rntiCrc, expected.rnti);
        EXPECT_EQ(actual.rntiBits, expected.scrambling_rnti);
        EXPECT_EQ(actual.dmrs_id, expected.scrambling_id);
        EXPECT_EQ(actual.aggr_level, expected.aggregation_level);
        EXPECT_EQ(actual.cce_index, expected.cce_index);
        EXPECT_EQ(actual.Npayload, expected.payload_size_bits);
        EXPECT_FALSE(actual.enablePrcdBf);
        EXPECT_EQ(actual.pmwPrmIdx, 0u);
        EXPECT_NEAR(actual.beta_qam, expected_beta(expected), 1.0e-6f);
        EXPECT_FLOAT_EQ(actual.beta_qam, actual.beta_dmrs);
        expect_payload_prefix(group.payloads[i], expected.payload,
                              expected.payload_size_bits);
    }
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 0u);
}

TEST(DLSlotProcessor, Process_TwoCellsEightPdcchPdusEach_PopulatesAggregate)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cells = 2u;
    static constexpr uint16_t k_pdus_per_cell = 8u;
    static constexpr uint16_t k_total_pdus = k_cells * k_pdus_per_cell;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.use_per_cell_ids_ = true;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    auto make_pdu_bodies = [](uint16_t cell_idx) {
        std::vector<std::vector<uint8_t>> bodies;
        bodies.reserve(k_pdus_per_cell);

        for (uint16_t pdu_idx = 0u; pdu_idx < k_pdus_per_cell; ++pdu_idx) {
            PdcchDciParams dci{};
            dci.rnti = static_cast<uint16_t>(0x1000u + (cell_idx * 0x100u) + pdu_idx);
            dci.scrambling_id = static_cast<uint16_t>(0x20u + pdu_idx);
            dci.scrambling_rnti = static_cast<uint16_t>(0x2000u + (cell_idx * 0x100u) + pdu_idx);
            dci.cce_index = static_cast<uint8_t>(pdu_idx);
            dci.aggregation_level = 1u;
            dci.payload_size_bits = 8u;
            dci.payload = {static_cast<uint8_t>(0xA0u + (cell_idx * 0x10u) + pdu_idx)};

            PdcchPduParams pdu{};
            pdu.bwp_start = static_cast<uint16_t>(12u + pdu_idx);
            pdu.start_sym = static_cast<uint8_t>(pdu_idx % 4u);
            pdu.duration_sym = 1u;
            pdu.dcis = {dci};

            bodies.push_back(build_pdcch_pdu_body(pdu));
        }

        return bodies;
    };

    FapiDlPdcchMsg cell0_msg{k_sfn, k_slot, make_pdu_bodies(0u), 0u};
    FapiDlPdcchMsg cell1_msg{k_sfn, k_slot, make_pdu_bodies(1u), 1u};
    const std::array<nv::phy_mac_msg_desc, k_cells> msgs{
        cell0_msg.desc,
        cell1_msg.desc,
    };

    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDCCH>(std::span{msgs});

    ASSERT_TRUE(result.has_value());

    const auto* params = view.slot_cmd_.cell_groups.pdcch.get();
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_index_list.size(), k_total_pdus);
    ASSERT_EQ(params->phy_cell_index_list.size(), k_total_pdus);

    const auto& group = params->csets_group;
    ASSERT_EQ(group.nCoresets, k_total_pdus);
    ASSERT_EQ(group.nDcis, k_total_pdus);

    for (uint16_t cell_idx = 0u; cell_idx < k_cells; ++cell_idx) {
        for (uint16_t pdu_idx = 0u; pdu_idx < k_pdus_per_cell; ++pdu_idx) {
            const uint16_t flat_idx =
                static_cast<uint16_t>((cell_idx * k_pdus_per_cell) + pdu_idx);
            const auto expected_carrier = view.carrier_id(cell_idx);
            const auto expected_phy_cell = view.phy_cell_id(cell_idx);

            const auto& coreset = group.csets[flat_idx];
            EXPECT_EQ(params->cell_index_list[flat_idx], expected_carrier);
            EXPECT_EQ(params->phy_cell_index_list[flat_idx], expected_phy_cell);
            EXPECT_EQ(coreset.n_f, static_cast<uint32_t>(view.cell_view_.bwp_size_) * 12u);
            EXPECT_EQ(coreset.slot_number, k_slot);
            EXPECT_EQ(coreset.start_rb, static_cast<uint16_t>(12u + pdu_idx));
            EXPECT_EQ(coreset.start_sym, static_cast<uint8_t>(pdu_idx % 4u));
            EXPECT_EQ(coreset.n_sym, 1u);
            EXPECT_EQ(coreset.nDci, 1u);
            EXPECT_EQ(coreset.dciStartIdx, flat_idx);
            EXPECT_EQ(coreset.slotBufferIdx, static_cast<uint32_t>(expected_carrier));

            const auto& dci = group.dcis[flat_idx];
            EXPECT_EQ(dci.rntiCrc,
                      static_cast<uint16_t>(0x1000u + (cell_idx * 0x100u) + pdu_idx));
            EXPECT_EQ(dci.rntiBits,
                      static_cast<uint16_t>(0x2000u + (cell_idx * 0x100u) + pdu_idx));
            EXPECT_EQ(dci.dmrs_id, static_cast<uint16_t>(0x20u + pdu_idx));
            EXPECT_EQ(dci.cce_index, pdu_idx);
            EXPECT_EQ(dci.aggr_level, 1u);
            EXPECT_EQ(dci.Npayload, 8u);
            EXPECT_EQ(group.payloads[flat_idx][0],
                      static_cast<uint8_t>(0xA0u + (cell_idx * 0x10u) + pdu_idx));
        }
    }

    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(),
              k_total_pdus);
}

TEST(DLSlotProcessor, Process_PdcchActiveSkipsMessagesWithoutPdcchForCellSetup)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_empty_cell = 0u;
    static constexpr uint16_t k_pdcch_cell = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.use_per_cell_ids_ = true;

    PdcchDciParams dci{};
    dci.payload_size_bits = 8u;
    dci.payload = {0x5Au};

    PdcchPduParams pdu{};
    pdu.dcis = {dci};

    FapiDlPdcchMsg empty_msg{
        k_sfn, k_slot, std::vector<std::vector<uint8_t>>{}, k_empty_cell};
    FapiDlPdcchMsg pdcch_msg{
        k_sfn, k_slot, std::vector<std::vector<uint8_t>>{build_pdcch_pdu_body(pdu)}, k_pdcch_cell};
    const std::array<nv::phy_mac_msg_desc, 2u> msgs{
        empty_msg.desc,
        pdcch_msg.desc,
    };

    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDCCH>(std::span{msgs});

    ASSERT_TRUE(result.has_value());

    const auto* params = view.slot_cmd_.cell_groups.pdcch.get();
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    ASSERT_EQ(params->phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list.back(), view.carrier_id(k_pdcch_cell));
    EXPECT_EQ(params->phy_cell_index_list.back(), view.phy_cell_id(k_pdcch_cell));
    EXPECT_EQ(params->csets_group.nCoresets, 1u);
    EXPECT_EQ(params->csets_group.csets[0].slotBufferIdx,
              static_cast<uint32_t>(view.carrier_id(k_pdcch_cell)));
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 1u);
}

// ===========================================================================
// Purpose:
//   Verify the temporary PDCCH-only sym_prb_info bridge is invoked from the
//   real parser when its temporary sym_prb_info gate is enabled.
//
// Flow:
//   1. Build a serialized FAPI PDCCH PDU body.
//   2. Enable the parser's temporary sym_prb_info gate.
//   3. Parse the PDU through the real parser.
//   4. Assert the test stub saw one bridge call with parsed PDCCH state.
//
// Coverage:
//   - temporary_sym_prb_info_enabled_ gates the bridge call
//   - parsed coreset/DCI state is available before the bridge is invoked
//   - bridge receives bandwidth, cell index, mMIMO flag, DCI count, DCI
//     payload size, and FAPI DCI offset
//
// This does not validate final fronthaul PRB section contents; that belongs in
// a heavier integration test around the production fronthaul helper.
// ===========================================================================
TEST(PdcchPduParser, Parse_TemporarySymPrbInfoEnabled_CallsBridgeWithParsedState)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.use_per_cell_ids_ = true;
    view.carrier_id_ = 5;
    view.mmimo_enabled_ = false;
    view.cell_view_.bwp_size_ = 79u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchDciParams dci{
        .rnti              = 0x4444u,
        .payload_size_bits = 24u,
        .payload           = {0x12u, 0x34u, 0x56u},
    };

    PdcchPduParams input{
        .bwp_size = view.cell_view_.bwp_size_,
        .dcis     = {dci},
    };

    const auto body = build_pdcch_pdu_body(input);
    const auto& pdu = *aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(body.data());

    scf_5g_fapi::PdcchPduParser<MockDlModuleView> parser{view};
    parser.setup_cell(static_cast<uint32_t>(k_cell_index), 0u, true);

    const auto* setup_params = view.slot_cmd_.cell_groups.pdcch.get();
    ASSERT_NE(setup_params, nullptr);
    ASSERT_TRUE(setup_params->cell_index_list.empty());
    ASSERT_TRUE(setup_params->phy_cell_index_list.empty());

    ASSERT_TRUE(parser.parse(k_sfn, k_slot, pdu));

    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_bandwidth(),
              view.cell_view_.bwp_size_);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_cell_index(),
              view.carrier_id(k_cell_index));
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_mmimo_enabled(),
              view.mmimo_enabled_);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_num_dl_dci(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_coreset_n_dci(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_coreset_dci_start_idx(), 0u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_first_dci_rnti(), dci.rnti);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_first_dci_payload_bits(),
              dci.payload_size_bits);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_first_dci_offset(), 0u);
}

// ===========================================================================
// Purpose:
//   Verify DLSlotProcessor automatically enables the temporary sym_prb_info
//   bridge when the actual DL_TTI payload contains only PDCCH PDUs.
//
// Flow:
//   1. Build a complete FAPI DL_TTI.request containing one PDCCH generic PDU.
//   2. Run DLSlotProcessor with only PDCCH active.
//   3. Assert processing succeeds and the bridge stub is invoked once.
//
// Coverage:
//   - dl_tti_pdcch_presence() classifies the message as pdcch_only
//   - DLSlotProcessor forwards that gate to PdcchPduParser
//   - the PDCCH-only validation bridge is reached through the real processor
//
// This is still a lightweight unit test: it verifies the processor/parser
// control flow, while the production fronthaul helper remains stubbed.
// ===========================================================================
TEST(DLSlotProcessor, Process_PdcchOnly_EnablesTemporarySymPrbInfoBridge)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.carrier_id_ = 6;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchPduParams input{};
    PdcchDciParams dci{};
    dci.rnti = 0x7777u;
    dci.payload_size_bits = 8u;
    dci.payload = {0xEFu};
    input.dcis = {dci};

    FapiDlPdcchMsg msg{k_sfn, k_slot, build_pdcch_pdu_body(input), k_cell_index};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_cell_index(),
              view.carrier_id_);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_num_dl_dci(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_first_dci_rnti(), dci.rnti);
}

TEST(DLSlotProcessor, Process_PdcchOnlyDirectCplane_DisablesTemporarySymPrbInfoBridge)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.fapi_to_cplane_direct_enabled_ = true;
    view.carrier_id_ = 6;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchPduParams input{};
    PdcchDciParams dci{};
    dci.rnti = 0x7777u;
    dci.payload_size_bits = 8u;
    dci.payload = {0xEFu};
    input.dcis = {dci};

    FapiDlPdcchMsg msg{k_sfn, k_slot, build_pdcch_pdu_body(input), k_cell_index};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 0u);
}

TEST(PdcchPduParser, Parse_UlDciNonDirectCplane_EnablesTemporarySymPrbInfoBridge)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.fapi_to_cplane_direct_enabled_ = false;
    view.carrier_id_ = 6;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchPduParams input{};
    PdcchDciParams dci{};
    dci.rnti = 0x8888u;
    dci.payload_size_bits = 8u;
    dci.payload = {0xABu};
    input.dcis = {dci};

    const auto body = build_pdcch_pdu_body(input);
    const auto& pdu = *aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(body.data());

    scf_5g_fapi::PdcchPduParser<MockDlModuleView> parser{view};
    parser.setup_cell(k_cell_index, 0u, !view.fapi_to_cplane_direct_enabled_);

    ASSERT_TRUE(parser.parse(k_sfn, k_slot, pdu));
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 1u);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_cell_index(),
              view.carrier_id_);
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_first_dci_rnti(), dci.rnti);
}

TEST(PdcchPduParser, Parse_UlDciDirectCplane_DisablesTemporarySymPrbInfoBridge)
{
    static constexpr uint16_t k_sfn = 42u;
    static constexpr uint16_t k_slot = 7u;
    static constexpr uint16_t k_cell_index = 1u;

    scf_5g_fapi::test::reset_pdcch_sym_prb_info_stub_state();

    MockDlModuleView view{};
    view.fapi_to_cplane_direct_enabled_ = true;
    view.carrier_id_ = 6;
    view.cell_view_.bwp_size_ = 106u;
    view.cell_view_.stat_prm_.nPrbDlBwp = view.cell_view_.bwp_size_;

    PdcchPduParams input{};
    PdcchDciParams dci{};
    dci.rnti = 0x9999u;
    dci.payload_size_bits = 8u;
    dci.payload = {0xCDu};
    input.dcis = {dci};

    const auto body = build_pdcch_pdu_body(input);
    const auto& pdu = *aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(body.data());

    scf_5g_fapi::PdcchPduParser<MockDlModuleView> parser{view};
    parser.setup_cell(k_cell_index, 0u, !view.fapi_to_cplane_direct_enabled_);

    ASSERT_TRUE(parser.parse(k_sfn, k_slot, pdu));
    EXPECT_EQ(scf_5g_fapi::test::pdcch_sym_prb_info_stub_call_count(), 0u);
}
