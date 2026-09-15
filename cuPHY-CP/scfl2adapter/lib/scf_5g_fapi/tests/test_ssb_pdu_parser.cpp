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

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "scf_5g_fapi_ssb_pdu_parser.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"

namespace
{

struct MockCellView final
{
    cuphyCellStatPrm_t stat_{};

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return 106u; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_; }
};

struct MockDlModuleView final
{
    mutable slot_command_api::slot_command slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable scf_5g_fapi::pm_weight_map_t pm_map_{};
    mutable nv::phy_config_option config_opt_{};
    mutable nv::phy_config phy_config_{};
    mutable nv::slot_limit_cell_error_t limit_errors_{};
    mutable std::array<scf_5g_fapi::DlPdschCellDelta, MAX_CELLS_PER_SLOT> published_stats_{};
    MockCellView cell_view_{};
    bool group_command_null_{false};

    MockDlModuleView()
    {
        cell_sub_cmd_.cell = 0u;
        phy_config_.cell_config_.phy_cell_id = 41u;
        phy_config_.carrier_config_.dl_freq_abs_A = 2110000u;
        phy_config_.carrier_config_.ul_freq_abs_A = 1920000u;
        phy_config_.carrier_config_.dl_grid_size[0] = 106u;
        phy_config_.ssb_config_.sub_c_common = 0u;
    }

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        if (group_command_null_) { return nullptr; }
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }

    [[nodiscard]] scf_5g_fapi::pm_weight_map_t& pm_map() const noexcept { return pm_map_; }
    [[nodiscard]] bool pm_enabled() const noexcept { return false; }
    [[nodiscard]] bool bf_enabled() const noexcept { return false; }
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept
    {
        return nullptr;
    }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return false; }
    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPdcchSlotNum() const noexcept { return -1; }
    [[nodiscard]] int staticPdschSlotNum() const noexcept { return -1; }
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t carrier_id(uint32_t cell_idx) const noexcept
    {
        return static_cast<int32_t>(10u + cell_idx);
    }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_idx) const noexcept
    {
        return static_cast<uint16_t>(41u + cell_idx);
    }
    [[nodiscard]] const nv::phy_config& phy_config(uint32_t) const noexcept
    {
        return phy_config_;
    }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    [[nodiscard]] uint16_t num_dl_prb(uint32_t) const noexcept { return 106u; }
    [[nodiscard]] bool is_fapi_to_cplane_direct_enabled() const noexcept { return false; }
    void publish_dl_pdsch_stats(const scf_5g_fapi::DlPdschStatsBatch&) const noexcept {}

    [[nodiscard]] MockCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }
};

static_assert(scf_5g_fapi::DlModuleView<MockDlModuleView>);

[[nodiscard]] scf_fapi_dl_tti_req_t make_dl_req(uint16_t sfn = 9u,
                                                 uint16_t slot = 7u,
                                                 [[maybe_unused]] uint16_t cell_id = 0u)
{
    scf_fapi_dl_tti_req_t req{};
    req.sfn = sfn;
    req.slot = slot;
    req.num_pdus = 1u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB] = 1u;
#endif
    return req;
}

[[nodiscard]] scf_fapi_ssb_pdu_t make_ssb_pdu(uint8_t block_index = 1u)
{
    scf_fapi_ssb_pdu_t pdu{};
    pdu.phys_cell_id = 41u;
    pdu.beta_pss = 0u;
    pdu.ssb_block_index = block_index;
    pdu.ssb_subcarrier_offset = 5u;
    pdu.ssb_offset_point_a = 2u;
    pdu.bch_payload_flag = 1u;
    pdu.mib_pdu.agg = 0x00ABCDEFu;
    return pdu;
}

struct FapiDlSsbMsg final
{
    std::vector<uint8_t> buf;
    nv::phy_mac_msg_desc desc{};

    FapiDlSsbMsg(uint16_t sfn,
                 uint16_t slot,
                 uint32_t cell_id,
                 const scf_fapi_ssb_pdu_t& ssb_body)
    {
        const auto hdr_sz     = sizeof(scf_fapi_header_t);
        const auto req_sz     = sizeof(scf_fapi_dl_tti_req_t);
        const auto gen_hdr_sz = sizeof(scf_fapi_generic_pdu_info_t);
        const auto total      = hdr_sz + req_sz + gen_hdr_sz + sizeof(ssb_body);

        buf.resize(total, 0u);

        auto* req = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(buf.data() + hdr_sz);
        req->sfn = sfn;
        req->slot = slot;
        req->num_pdus = 1u;
#ifdef SCF_FAPI_10_04
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB] = 1u;
#endif

        auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(
            buf.data() + hdr_sz + req_sz);
        gen->pdu_type = static_cast<uint16_t>(DL_TTI_PDU_TYPE_SSB);
        gen->pdu_size = static_cast<uint16_t>(gen_hdr_sz + sizeof(ssb_body));
        const auto* src = reinterpret_cast<const uint8_t*>(&ssb_body);
        std::copy_n(src, sizeof(ssb_body),
                    buf.data() + hdr_sz + req_sz + gen_hdr_sz);

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = cell_id;
    }
};

void setup_and_parse([[maybe_unused]] MockDlModuleView& view,
                     scf_5g_fapi::SsbPduParser<MockDlModuleView>& parser,
                     const scf_fapi_dl_tti_req_t& req,
                     const scf_fapi_ssb_pdu_t& pdu)
{
    parser.setup_cell(0u);
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, pdu));
}

void setup_and_parse_cell([[maybe_unused]] MockDlModuleView& view,
                          scf_5g_fapi::SsbPduParser<MockDlModuleView>& parser,
                          const scf_fapi_dl_tti_req_t& req,
                          const scf_fapi_ssb_pdu_t& pdu,
                          uint32_t cell_id)
{
    parser.setup_cell(cell_id);
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, pdu));
}

} // namespace

TEST(SsbPduParser, Parse_PopulatesSingleCellAndBlockParams)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();
    const auto pdu = make_ssb_pdu(1u);

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, pdu));

    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 1u);
    EXPECT_EQ(params->nSsbBlocks, 1u);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    ASSERT_EQ(params->phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list[0], view.carrier_id(0u));
    EXPECT_EQ(params->phy_cell_index_list[0], view.phy_cell_id(0u));
    EXPECT_EQ(view.cell_sub_cmd_.cell, view.phy_cell_id(0u));

    const auto& cell = params->pbch_dyn_cell_params[0];
    EXPECT_EQ(cell.NID, 41u);
    EXPECT_EQ(cell.nHF, 1u);
    EXPECT_EQ(cell.Lmax, 4u);
    EXPECT_EQ(cell.SFN, req.sfn);
    EXPECT_EQ(cell.k_SSB, pdu.ssb_subcarrier_offset);
    EXPECT_EQ(cell.nF, 106u * CUPHY_N_TONES_PER_PRB);
    EXPECT_EQ(cell.slotBufferIdx, 0u);

    const auto& block = params->pbch_dyn_block_params[0];
    EXPECT_EQ(block.blockIndex, pdu.ssb_block_index);
    EXPECT_EQ(block.t0, 8u);
    EXPECT_EQ(block.f0, pdu.ssb_subcarrier_offset + (pdu.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB));
    EXPECT_FLOAT_EQ(block.beta_pss, 1.0F);
    EXPECT_FLOAT_EQ(block.beta_sss, 1.0F);
    EXPECT_EQ(block.cell_index, 0u);
    EXPECT_FALSE(block.enablePrcdBf);
    EXPECT_EQ(params->pbch_dyn_mib_data[0], pdu.mib_pdu.agg);

    EXPECT_EQ(view.cell_sub_cmd_.slot.type, slot_command_api::SLOT_DOWNLINK);
    EXPECT_EQ(view.slot_cmd_.cell_groups.slot.type, slot_command_api::SLOT_DOWNLINK);
}

TEST(SsbPduParser, Parse_ReusesCellForMultipleBlocks)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));
    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(2u)));

    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 1u);
    EXPECT_EQ(params->nSsbBlocks, 2u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].cell_index, 0u);
    EXPECT_EQ(params->pbch_dyn_block_params[1].cell_index, 0u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].t0, 2u);
    EXPECT_EQ(params->pbch_dyn_block_params[1].t0, 2u);
}

TEST(SsbPduParser, Parse_BetaPssThreeDb)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();
    auto pdu = make_ssb_pdu(0u);
    pdu.beta_pss = 1u;

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, pdu));

    const auto& block = view.slot_cmd_.cell_groups.pbch->pbch_dyn_block_params[0];
    EXPECT_FLOAT_EQ(block.beta_pss, scf_5g_fapi::detail::k_beta_pss_3db);
}

TEST(SsbPduParser, Parse_AppliesStaticOverrides)
{
    MockDlModuleView view;
    view.config_opt_.enableTickDynamicSfnSlot = 0;
    view.config_opt_.staticSsbPcid = 99;
    view.config_opt_.staticSsbSFN = 123;
    view.config_opt_.staticSsbSlotNum = 17;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));

    const auto& cell = view.slot_cmd_.cell_groups.pbch->pbch_dyn_cell_params[0];
    EXPECT_EQ(cell.NID, 99u);
    EXPECT_EQ(cell.SFN, 123u);
    EXPECT_EQ(cell.nHF, 3u);
}

TEST(SsbPduParser, Parse_L1LimitExceededDropsPdu)
{
#ifdef ENABLE_L2_SLT_RSP
    MockDlModuleView view;
    view.limit_errors_.ssb_pbch_errors.parsed = CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));

    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
    EXPECT_EQ(view.slot_cmd_.cell_groups.pbch->nSsbBlocks, 0u);
    EXPECT_EQ(view.limit_errors_.ssb_pbch_errors.errors, 1u);
#else
    GTEST_SKIP() << "SSB/PBCH L1-limit accounting requires ENABLE_L2_SLT_RSP";
#endif
}

TEST(SsbPduParser, Parse_InvalidBlockIndexDropsPdu)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(4u)));

    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
    EXPECT_EQ(view.slot_cmd_.cell_groups.pbch->nSsbBlocks, 0u);
}

TEST(SsbPduParser, Parse_WithoutSetupCellReturnsFalse)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    EXPECT_FALSE(parser.parse(req.sfn, req.slot, make_ssb_pdu(0u)));
    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
}

TEST(SsbPduParser, Parse_NullGroupCommandReturnsFalse)
{
    MockDlModuleView view;
    view.group_command_null_ = true;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    parser.setup_cell(0u);
    EXPECT_FALSE(parser.parse(req.sfn, req.slot, make_ssb_pdu(0u)));
}

TEST(SsbPduParser, Parse_AccumulatesMultipleCells)
{
    MockDlModuleView view;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse_cell(view, parser, req, make_ssb_pdu(0u), 0u));
    ASSERT_NO_FATAL_FAILURE(setup_and_parse_cell(view, parser, req, make_ssb_pdu(1u), 1u));

    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 2u);
    EXPECT_EQ(params->nSsbBlocks, 2u);
    ASSERT_EQ(params->cell_index_list.size(), 2u);
    ASSERT_EQ(params->phy_cell_index_list.size(), 2u);
    EXPECT_EQ(params->cell_index_list[0], view.carrier_id(0u));
    EXPECT_EQ(params->cell_index_list[1], view.carrier_id(1u));
    EXPECT_EQ(params->phy_cell_index_list[0], view.phy_cell_id(0u));
    EXPECT_EQ(params->phy_cell_index_list[1], view.phy_cell_id(1u));
    EXPECT_EQ(params->pbch_dyn_block_params[0].cell_index, 0u);
    EXPECT_EQ(params->pbch_dyn_block_params[1].cell_index, 1u);
}

TEST(SsbPduParser, Parse_UnsupportedSsbCaseDropsPdu)
{
    MockDlModuleView view;
    view.phy_config_.carrier_config_.dl_freq_abs_A = 7000000u;
    view.phy_config_.carrier_config_.ul_freq_abs_A = 7000000u;
    view.phy_config_.ssb_config_.sub_c_common = 0u;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));

    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
    if (view.slot_cmd_.cell_groups.pbch)
    {
        EXPECT_EQ(view.slot_cmd_.cell_groups.pbch->nSsbBlocks, 0u);
    }
}

TEST(SsbPduParser, Parse_UnsupportedSsbNumerologyDropsPdu)
{
    MockDlModuleView view;
    view.phy_config_.carrier_config_.dl_freq_abs_A = 3500000u;
    view.phy_config_.carrier_config_.ul_freq_abs_A = 3500000u;
    view.phy_config_.ssb_config_.sub_c_common = 2u;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));

    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
    if (view.slot_cmd_.cell_groups.pbch)
    {
        EXPECT_EQ(view.slot_cmd_.cell_groups.pbch->nSsbBlocks, 0u);
    }
}

TEST(SsbPduParser, CalcSsbF0RejectsMu2)
{
    const auto pdu = make_ssb_pdu(0u);
    EXPECT_FALSE(scf_5g_fapi::detail::calc_ssb_f0(pdu, 2u).has_value());
}

TEST(SsbPduParser, Parse_CaseCUsesLmax8Symbols)
{
    MockDlModuleView view;
    view.phy_config_.carrier_config_.dl_freq_abs_A = 3500000u;
    view.phy_config_.carrier_config_.ul_freq_abs_A = 3500000u;
    view.phy_config_.carrier_config_.dl_grid_size[1] = 273u;
    view.phy_config_.ssb_config_.sub_c_common = 1u;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(4u)));

    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->pbch_dyn_cell_params[0].Lmax, 8u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].t0, 2u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].f0,
              (5u + (2u * CUPHY_N_TONES_PER_PRB)) >> 1u);
}

TEST(SsbPduParser, Parse_CaseDUsesLmax64Symbols)
{
    MockDlModuleView view;
    view.phy_config_.carrier_config_.dl_freq_abs_A = 27000000u;
    view.phy_config_.carrier_config_.ul_freq_abs_A = 27000000u;
    view.phy_config_.carrier_config_.dl_grid_size[3] = 132u;
    view.phy_config_.ssb_config_.sub_c_common = 3u;
    scf_5g_fapi::SsbPduParser<MockDlModuleView> parser{view};
    const auto req = make_dl_req();

    ASSERT_NO_FATAL_FAILURE(setup_and_parse(view, parser, req, make_ssb_pdu(0u)));

    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->pbch_dyn_cell_params[0].Lmax, 64u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].t0, 4u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].f0,
              (5u + (2u * CUPHY_N_TONES_PER_PRB)) >> 3u);
}

TEST(DLSlotProcessor, Process_SsbPduPopulatesPbchSlotCommand)
{
    MockDlModuleView view;
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    FapiDlSsbMsg msg{9u, 7u, 0u, make_ssb_pdu(1u)};

    const auto result = processor.process<DL_TTI_PDU_TYPE_SSB>(
        std::span<const nv::phy_mac_msg_desc>{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 1u);
    EXPECT_EQ(params->nSsbBlocks, 1u);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list[0], view.carrier_id(0u));
    EXPECT_EQ(params->pbch_dyn_block_params[0].blockIndex, 1u);
    EXPECT_EQ(params->pbch_dyn_block_params[0].cell_index, 0u);
    EXPECT_EQ(view.cell_sub_cmd_.slot.type, slot_command_api::SLOT_DOWNLINK);
    EXPECT_EQ(view.slot_cmd_.cell_groups.slot.type, slot_command_api::SLOT_DOWNLINK);
}

// ---------------------------------------------------------------------------
// DLSlotProcessor integration coverage for the MR2 wiring (review B-1).
// ---------------------------------------------------------------------------

namespace
{

/// Build a DL_TTI request carrying zero SSB PDUs so process<SSB> exits without
/// touching the parser. Verifies the activation path is a no-op when no SSB is
/// scheduled (cf. real slots where only PDSCH/PDCCH fire).
struct FapiDlNoSsbMsg final
{
    std::vector<uint8_t> buf;
    nv::phy_mac_msg_desc desc{};

    explicit FapiDlNoSsbMsg(uint32_t cell_id = 0u)
    {
        const auto hdr_sz = sizeof(scf_fapi_header_t);
        const auto req_sz = sizeof(scf_fapi_dl_tti_req_t);
        buf.resize(hdr_sz + req_sz, 0u);

        auto* req = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(buf.data() + hdr_sz);
        req->sfn = 9u;
        req->slot = 7u;
        req->num_pdus = 0u;
#ifdef SCF_FAPI_10_04
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB] = 0u;
#endif

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = cell_id;
    }
};

} // namespace

TEST(DLSlotProcessor, Process_NoSsbPdus_LeavesPbchUntouched)
{
    MockDlModuleView view;
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    FapiDlNoSsbMsg msg{0u};

    const auto result = processor.process<DL_TTI_PDU_TYPE_SSB>(
        std::span<const nv::phy_mac_msg_desc>{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 0u);
    EXPECT_EQ(params->nSsbBlocks, 0u);
    EXPECT_TRUE(params->cell_index_list.empty());
}

TEST(DLSlotProcessor, Process_TwoCellsAccumulateBothInPbchParams)
{
    MockDlModuleView view;
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    FapiDlSsbMsg msg_a{9u, 7u, 0u, make_ssb_pdu(0u)};
    FapiDlSsbMsg msg_b{9u, 7u, 1u, make_ssb_pdu(2u)};
    const std::array<nv::phy_mac_msg_desc, 2> msgs{msg_a.desc, msg_b.desc};

    const auto result = processor.process<DL_TTI_PDU_TYPE_SSB>(
        std::span<const nv::phy_mac_msg_desc>{msgs.data(), msgs.size()});

    ASSERT_TRUE(result.has_value());
    const auto* params = view.slot_cmd_.cell_groups.pbch.get();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->ncells, 2u);
    EXPECT_EQ(params->nSsbBlocks, 2u);
    ASSERT_EQ(params->cell_index_list.size(), 2u);
    EXPECT_EQ(params->cell_index_list[0], view.carrier_id(0u));
    EXPECT_EQ(params->cell_index_list[1], view.carrier_id(1u));
    EXPECT_EQ(params->pbch_dyn_block_params[0].cell_index, 0u);
    EXPECT_EQ(params->pbch_dyn_block_params[1].cell_index, 1u);
}

TEST(DLSlotProcessor, Process_NullGroupCommand_PropagatesParserFailure)
{
    MockDlModuleView view;
    view.group_command_null_ = true;
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};
    FapiDlSsbMsg msg{9u, 7u, 0u, make_ssb_pdu(0u)};

    const auto result = processor.process<DL_TTI_PDU_TYPE_SSB>(
        std::span<const nv::phy_mac_msg_desc>{&msg.desc, 1u});

#ifdef SCF_FAPI_10_04
    // 10_04 ON: post-dispatch verify_per_type_counts_typed sees
    // processed=0 != nPDUsOfEachType[SSB]=1 and surfaces PerTypeMismatch.
    ASSERT_FALSE(result.has_value());
#else
    // 10_04 OFF: there is no post-check; SsbPduParser::parse() returning
    // false on a null group_command is silently swallowed by dispatch_pdus_typed
    // (PDU not counted). process<SSB> returns success. Documented coverage
    // gap; the gating that exists today is profile-specific.
    ASSERT_TRUE(result.has_value());
#endif
}
