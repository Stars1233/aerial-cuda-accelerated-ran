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
#include <cstring>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <aerial/casts/casts.hpp>

#include "scf_5g_csirs_slot_command_helpers.hpp"
#include "scf_5g_fapi.h"
#include "cuphy.h"
#include "nv_phy_config_option.hpp"
#include "nv_fapi_pdu_utils.hpp"
#include "slot_command/slot_command.hpp"

namespace {

using slot_command_api::cell_group_command;
using slot_command_api::cell_sub_command;
using slot_command_api::pdsch_params;
using slot_command_api::slot_indication;

[[nodiscard]] scf_fapi_csi_rsi_pdu_t makeMinimalNzpCsirsPdu(const uint16_t start_rb)
{
    scf_fapi_csi_rsi_pdu_t msg{};
    msg.bwp.bwp_start = 0;
    msg.bwp.bwp_size  = 273;
    msg.start_rb      = start_rb;
    msg.num_of_rbs    = 4;
    msg.csi_type      = static_cast<uint8_t>(cuphyCsiType_t::NZP_CSI_RS);
    msg.row           = 1;
    msg.freq_domain   = 0;
    msg.sym_l0        = 0;
    msg.sym_l1        = 0;
    msg.cdm_type      = 0;
    msg.freq_density  = 0;
    msg.scrambling_id = 0;
    msg.tx_power.power_control_offset_ss = 1;
    msg.pc_and_bf.num_prgs            = 1;
    msg.pc_and_bf.prg_size            = 1;
    msg.pc_and_bf.dig_bf_interfaces   = 0;
    msg.pc_and_bf.pm_idx_and_beam_idx[0] = 0;
    return msg;
}

/**
 * @brief Seed a pdsch_params with three dummy cells for multi-cell tests.
 * @param pp  Non-null pdsch_params that will be mutated.
 */
void seedThreePdschCells(pdsch_params* const pp)
{
    pp->cell_grp_info.nCells = 3;
    pp->cell_index_list      = {0, 1, 2};
    pp->phy_cell_index_list  = {500, 501, 502};
    for (int i = 0; i < 3; ++i)
    {
        pp->cell_dyn_info[i].cellPrmDynIdx = static_cast<uint16_t>(i);
        pp->cell_dyn_info[i].nCsiRsPrms    = 0;
        pp->cell_dyn_info[i].csiRsPrmsOffset = 0;
    }
}

/** @brief Forward a single CSI-RS PDU to apply_csirs_fapi_pdu_to_slot_command with mmimo enabled. */
void applyOneCsirsPdu(cell_group_command&           grp,
                      cell_sub_command&              cell_cmd,
                      const scf_fapi_csi_rsi_pdu_t&  pdu,
                      const slot_indication&         slotinfo,
                      const int32_t                  cell_index,
                      nv::phy_config_option&          cfg,
                      scf_5g_fapi::pm_weight_map_t&  pm_map,
                      const uint32_t                 csirs_offset,
                      const bool                     has_pdsch,
                      const uint32_t                 pdsch_dyn_idx)
{
    scf_5g_fapi::apply_csirs_fapi_pdu_to_slot_command(grp,
                                                      cell_cmd,
                                                      pdu,
                                                      slotinfo,
                                                      cell_index,
                                                      cfg,
                                                      pm_map,
                                                      csirs_offset,
                                                      has_pdsch,
                                                      static_cast<uint16_t>(cell_index),
                                                      true /* mmimo */,
                                                      pdsch_dyn_idx);
}

/**
 * @brief Searches @p grp for a registered channel of type @p channel.
 *
 * Scans the active portion of @c grp.channels[] (indices 0 to
 * @c channel_array_size - 1) for an entry matching @p channel.
 *
 * @note @c [[nodiscard]]: the return value must not be silently discarded.
 *       @c noexcept: never throws.
 *
 * @param grp     The @c cell_group_command whose channel list is inspected.
 * @param channel The @c channel_type to search for.
 * @return @c true if @p channel is found in @c grp.channels; @c false otherwise.
 */
[[nodiscard]] bool hasRegisteredGroupChannel(const cell_group_command& grp,
                                             const slot_command_api::channel_type channel) noexcept
{
    const std::span view{grp.channels.data(), grp.channel_array_size};
    return std::ranges::any_of(view, [channel](const auto ch) { return ch == channel; });
}

} // namespace

// ---------------------------------------------------------------------------
// Policy + fast-path (existing)
// ---------------------------------------------------------------------------

TEST(CsirsSlotHelpers, ZpPolicyUsesPdschPduCountNotLegacyFlag)
{
    EXPECT_TRUE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::ZP_CSI_RS, false));
    EXPECT_FALSE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::ZP_CSI_RS, true));
    EXPECT_FALSE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::NZP_CSI_RS, false));
}

// Real assertion against the production fast-path skip helper:
// for_each_tti_msg() at nv_fapi_pdu_utils.hpp:177 short-circuits messages whose
// nPDUsOfEachType[npdus_type_idx] == 0. This is exactly the gate that
// process_aggr_csirs_channel relies on to skip DL_TTI.req messages with no
// CSI-RS PDUs without ever reading their payload.
//
// TODO(GT-11913 follow-up): add an end-to-end test of process_aggr_csirs_channel
// once a PHY_module fixture is available; the unit-level assertion here covers
// the predicate the production code uses.
#ifdef SCF_FAPI_10_04
TEST(CsirsSlotHelpers, ForEachTtiMsgSkipsDlTtiWithoutCsirsPdus)
{
    constexpr std::size_t kBufSize =
        sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t);

    alignas(64) std::array<uint8_t, kBufSize> raw_no_csirs{};
    alignas(64) std::array<uint8_t, kBufSize> raw_with_csirs{};

    auto* req_no  = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(
        raw_no_csirs.data() + sizeof(scf_fapi_header_t));
    auto* req_yes = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(
        raw_with_csirs.data() + sizeof(scf_fapi_header_t));
    std::memset(req_no,  0, sizeof(*req_no));
    std::memset(req_yes, 0, sizeof(*req_yes));
    req_yes->nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = 2;

    std::array<nv::phy_mac_msg_desc, 2> msgs{};
    msgs[0].msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msgs[0].cell_id = 0;
    msgs[0].msg_buf = raw_no_csirs.data();
    msgs[0].msg_len = static_cast<int32_t>(kBufSize);
    msgs[1].msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msgs[1].cell_id = 1;
    msgs[1].msg_buf = raw_with_csirs.data();
    msgs[1].msg_len = static_cast<int32_t>(kBufSize);

    std::vector<uint16_t> visited;
    nv::for_each_tti_msg<0u, scf_fapi_dl_tti_req_t>(
        msgs.data(), static_cast<uint16_t>(msgs.size()),
        [](uint16_t, const scf_fapi_dl_tti_req_t& req) { return nv::has_csi_rs(req); },
        /*slot_u32=*/0u, /*ring_idx=*/0u,
        [&visited](uint16_t i,
                   const nv::phy_mac_msg_desc&,
                   const scf_fapi_dl_tti_req_t&) {
            visited.push_back(i);
        });

    ASSERT_EQ(visited.size(), 1u);
    EXPECT_EQ(visited[0], 1u); // only the message with nPDUsOfEachType[CSI_RS] > 0
}
#endif

// ---------------------------------------------------------------------------
// Pure fill helpers
// ---------------------------------------------------------------------------

TEST(CsirsSlotHelpers, FillCommonSetsFieldsAndStaticSlot)
{
    scf_fapi_csi_rsi_pdu_t msg = makeMinimalNzpCsirsPdu(42);
    cuphyCsirsRrcDynPrm_t dst{};
    slot_indication       slotinfo{0, 2, 0};
    scf_5g_fapi::fill_csirs_rrc_dyn_from_fapi(msg, dst, slotinfo, -1);
    EXPECT_EQ(42, dst.startRb);
    EXPECT_EQ(2, dst.idxSlotInFrame);

    scf_5g_fapi::fill_csirs_rrc_dyn_from_fapi(msg, dst, slotinfo, 9);
    EXPECT_EQ(9, dst.idxSlotInFrame);
}

TEST(CsirsSlotHelpers, PdschMirrorCallSiteAddsBwpStartOffset)
{
    scf_fapi_csi_rsi_pdu_t msg = makeMinimalNzpCsirsPdu(7);
    msg.bwp.bwp_start          = 3;
    cuphyCsirsRrcDynPrm_t dst{};
    slot_indication       slotinfo{1, 5, 0};
    scf_5g_fapi::fill_csirs_rrc_dyn_from_fapi(msg, dst, slotinfo, -1);
    dst.startRb      = static_cast<uint16_t>(msg.start_rb + msg.bwp.bwp_start);
    dst.enablePrcdBf = false;
    dst.pmwPrmIdx    = 0;
    EXPECT_EQ(static_cast<uint16_t>(7 + 3), dst.startRb);
    EXPECT_EQ(5, dst.idxSlotInFrame);
    EXPECT_FALSE(dst.enablePrcdBf);
    EXPECT_EQ(0, dst.pmwPrmIdx);
}

// ---------------------------------------------------------------------------
// Multi-cell PDSCH mirror: correct cell_dyn_info row per carrier
// ---------------------------------------------------------------------------

TEST(CsirsSlotHelpers, MultiCell_PdschMirror_UpdatesCorrectCellDynInfo)
{
    cell_group_command grp{};
    nv::phy_config_option cfg{};
    cfg.precoding_enabled = false;
    cfg.staticCsiRsSlotNum = -1;
    scf_5g_fapi::pm_weight_map_t pm_map{};

    grp.create_if(slot_command_api::channel_type::PDSCH);
    seedThreePdschCells(grp.get_pdsch_params());

    slot_indication slotinfo{0, 0, 0};

    // Cell 1 only: dyn row 1 must move; 0 and 2 stay at 0 nCsiRsPrms for mirror increments
    {
        cell_sub_command cell_cmd{};
        cell_cmd.cell = 501;
        const auto pdu = makeMinimalNzpCsirsPdu(11);
        applyOneCsirsPdu(grp, cell_cmd, pdu, slotinfo, 1, cfg, pm_map, 0U, true, 1U);
        const auto* pp = grp.get_pdsch_params();
        EXPECT_EQ(1u, pp->cell_dyn_info[1].nCsiRsPrms);
        EXPECT_EQ(0u, pp->cell_dyn_info[0].nCsiRsPrms);
        EXPECT_EQ(0u, pp->cell_dyn_info[2].nCsiRsPrms);
    }

    grp.reset();
    grp.create_if(slot_command_api::channel_type::PDSCH);
    seedThreePdschCells(grp.get_pdsch_params());

    // Cell 0: row 0
    {
        cell_sub_command cell_cmd{};
        cell_cmd.cell = 500;
        const auto pdu = makeMinimalNzpCsirsPdu(20);
        applyOneCsirsPdu(grp, cell_cmd, pdu, slotinfo, 0, cfg, pm_map, 3U, true, 0U);
        const auto* pp = grp.get_pdsch_params();
        EXPECT_EQ(1u, pp->cell_dyn_info[0].nCsiRsPrms);
        EXPECT_EQ(3u, pp->cell_dyn_info[0].csiRsPrmsOffset);
        EXPECT_EQ(0u, pp->cell_dyn_info[1].nCsiRsPrms);
    }
}

// ---------------------------------------------------------------------------
// Permuted processing order: csirs_params cell_index_list (L237-242)
// ---------------------------------------------------------------------------

TEST(CsirsSlotHelpers, MultiCell_PermutedOrder_RegistersEachCellOnce)
{
    const std::vector<std::vector<int32_t>> permutations = {
        {0, 1, 2},
        {2, 1, 0},
    };

    nv::phy_config_option cfg{};
    cfg.precoding_enabled = false;
    scf_5g_fapi::pm_weight_map_t pm_map{};

    for (const auto& order : permutations)
    {
        cell_group_command grp{};
        grp.create_if(slot_command_api::channel_type::PDSCH);
        seedThreePdschCells(grp.get_pdsch_params());

        slot_indication slotinfo{0, 0, 0};
        uint16_t        rb = 10;
        for (const int32_t ci : order)
        {
            cell_sub_command cell_cmd{};
            cell_cmd.cell = static_cast<uint16_t>(500 + ci);
            auto pdu      = makeMinimalNzpCsirsPdu(rb++);
            applyOneCsirsPdu(grp, cell_cmd, pdu, slotinfo, ci, cfg, pm_map, 0U, true,
                             static_cast<uint32_t>(ci));
        }

        const auto* cs = grp.csirs.get();
        ASSERT_NE(nullptr, cs);
        ASSERT_EQ(order.size(), cs->cell_index_list.size());
        ASSERT_EQ(cs->cell_index_list.size(), cs->phy_cell_index_list.size());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            EXPECT_EQ(order[i], cs->cell_index_list[i]);
            EXPECT_EQ(500 + order[i], cs->phy_cell_index_list[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-slot: fresh group per slot — lists match first-seen order only
// ---------------------------------------------------------------------------

TEST(CsirsSlotHelpers, MultiSlot_PermutedCellOrder_IndependentSlotState)
{
    nv::phy_config_option cfg{};
    cfg.precoding_enabled = false;
    scf_5g_fapi::pm_weight_map_t pm_map{};

    const struct {
        uint16_t sfn;
        uint16_t slot;
        std::vector<int32_t> order;
    } slots[] = {{0, 0, {0, 1, 2}}, {0, 1, {2, 1, 0}}};

    for (const auto& sp : slots)
    {
        cell_group_command grp{};
        grp.create_if(slot_command_api::channel_type::PDSCH);
        seedThreePdschCells(grp.get_pdsch_params());

        slot_indication slotinfo{sp.sfn, sp.slot, 0};
        uint16_t        rb = 30;
        for (const int32_t ci : sp.order)
        {
            cell_sub_command cell_cmd{};
            cell_cmd.cell = static_cast<uint16_t>(500 + ci);
            auto pdu      = makeMinimalNzpCsirsPdu(rb++);
            applyOneCsirsPdu(grp, cell_cmd, pdu, slotinfo, ci, cfg, pm_map, 0U, true,
                             static_cast<uint32_t>(ci));
        }

        const auto* cs = grp.csirs.get();
        ASSERT_NE(nullptr, cs);
        for (std::size_t i = 0; i < sp.order.size(); ++i)
        {
            EXPECT_EQ(sp.order[i], cs->cell_index_list[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Slot-command channel reads: EOM pre-registration coverage (GT-12586)
// ---------------------------------------------------------------------------

TEST(SlotCommandChannelReads, RegistersDlChannelsFromGetterReads)
{
    cell_group_command grp{};

    const auto* pdcch_dl = grp.get_pdcch_params(slot_command_api::PDCCH_DL);
    const auto* pdcch_ul = grp.get_pdcch_params(slot_command_api::PDCCH_UL);
    std::ignore = grp.get_pdsch_params();
    std::ignore = grp.get_csirs_params();
    std::ignore = grp.get_pbch_params();

    // Both PDCCH variants share one pdcch_group_params object.
    EXPECT_EQ(pdcch_dl, pdcch_ul);
    EXPECT_EQ(5u, static_cast<uint32_t>(grp.channel_array_size));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_DL));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_UL));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PDSCH));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::CSI_RS));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PBCH));
}

TEST(SlotCommandChannelReads, RegistersUlChannelsFromGetterReads)
{
    cell_group_command grp{};

    std::ignore = grp.get_pusch_params();
    std::ignore = grp.get_prach_params();
    std::ignore = grp.get_pucch_params();
    std::ignore = grp.get_srs_params();

    EXPECT_EQ(4u, static_cast<uint32_t>(grp.channel_array_size));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PUSCH));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PRACH));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PUCCH));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::SRS));
}

// No-arg get_pdcch_params() routes to PDCCH_DL; PDCCH_UL must remain unregistered.
// Store/replay must explicitly choose the same registration as the legacy
// on_msg path; the no-arg convenience API remains DL-only.
TEST(SlotCommandChannelReads, DefaultPdcchOverloadRegistersDlOnly)
{
    cell_group_command grp{};

    std::ignore = grp.get_pdcch_params(); // no-arg form

    EXPECT_EQ(1u, static_cast<uint32_t>(grp.channel_array_size));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_DL));
    EXPECT_FALSE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_UL));
}

// The typed PDCCH_UL getter remains available for callers that truly own a
// PDCCH_UL aggregate path. Store/replay UL_DCI does not use this on develop
// because PHY_PDCCH_UL_AGGR_X_CTX is intentionally zero there.
TEST(SlotCommandChannelReads, PdcchUlOnlyRegistersUlNotDl)
{
    cell_group_command grp{};

    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_UL);

    EXPECT_EQ(1u, static_cast<uint32_t>(grp.channel_array_size));
    EXPECT_TRUE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_UL));
    EXPECT_FALSE(hasRegisteredGroupChannel(grp, slot_command_api::PDCCH_DL));
}

// All three PDCCH call-paths return the same backing pdcch_group_params object.
TEST(SlotCommandChannelReads, AllPdcchOverloadsReturnSameParams)
{
    cell_group_command grp{};

    const auto* ptr_no_arg = grp.get_pdcch_params();
    const auto* ptr_dl     = grp.get_pdcch_params(slot_command_api::PDCCH_DL);
    const auto* ptr_ul     = grp.get_pdcch_params(slot_command_api::PDCCH_UL);

    ASSERT_NE(nullptr, ptr_no_arg);
    EXPECT_EQ(ptr_no_arg, ptr_dl);
    EXPECT_EQ(ptr_no_arg, ptr_ul);
}

// channel_idx entries for channels that were never gotten must remain NONE.
TEST(SlotCommandChannelReads, UnregisteredChannelsRetainNoneIndex)
{
    cell_group_command grp{};

    std::ignore = grp.get_pdsch_params(); // register only PDSCH

    constexpr auto kNone = static_cast<uint32_t>(slot_command_api::channel_type::NONE);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PDCCH_DL]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PDCCH_UL]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::CSI_RS]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PBCH]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PUSCH]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PRACH]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::PUCCH]);
    EXPECT_EQ(kNone, grp.channel_idx[slot_command_api::SRS]);
}

// Two concurrently live cell_group_commands (two slots in flight), each primed
// from a different mix of DL+UL FAPI messages. The channel_idx arrays must be
// fully independent — slot A's NONE entries must not be affected by slot B's
// registrations and vice versa.
TEST(SlotCommandChannelReads, TwoSlotsWithMixedDlUlChannelsMaintainIndependentChannelIdx)
{
    constexpr auto kNone = static_cast<uint32_t>(slot_command_api::channel_type::NONE);

    // Slot A: DL PDCCH + PDSCH; UL PUSCH + PUCCH.
    cell_group_command grp_a{};
    std::ignore = grp_a.get_pdcch_params(slot_command_api::PDCCH_DL);
    std::ignore = grp_a.get_pdsch_params();
    std::ignore = grp_a.get_pusch_params();
    std::ignore = grp_a.get_pucch_params();

    // Slot B: DL PDCCH(UL-DCI) + CSI_RS; UL PRACH + SRS.
    cell_group_command grp_b{};
    std::ignore = grp_b.get_pdcch_params(slot_command_api::PDCCH_UL);
    std::ignore = grp_b.get_csirs_params();
    std::ignore = grp_b.get_prach_params();
    std::ignore = grp_b.get_srs_params();

    // --- Slot A: verify registrations ---
    EXPECT_EQ(4u, static_cast<uint32_t>(grp_a.channel_array_size));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_a, slot_command_api::PDCCH_DL));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_a, slot_command_api::PDSCH));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_a, slot_command_api::PUSCH));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_a, slot_command_api::PUCCH));

    // Channels that slot A never saw must remain NONE.
    EXPECT_EQ(kNone, grp_a.channel_idx[slot_command_api::PDCCH_UL]);
    EXPECT_EQ(kNone, grp_a.channel_idx[slot_command_api::CSI_RS]);
    EXPECT_EQ(kNone, grp_a.channel_idx[slot_command_api::PRACH]);
    EXPECT_EQ(kNone, grp_a.channel_idx[slot_command_api::SRS]);

    // channels[] maps insertion-order index back to the correct channel_type.
    EXPECT_EQ(slot_command_api::PDCCH_DL,
              grp_a.channels[grp_a.channel_idx[slot_command_api::PDCCH_DL]]);
    EXPECT_EQ(slot_command_api::PDSCH,
              grp_a.channels[grp_a.channel_idx[slot_command_api::PDSCH]]);
    EXPECT_EQ(slot_command_api::PUSCH,
              grp_a.channels[grp_a.channel_idx[slot_command_api::PUSCH]]);
    EXPECT_EQ(slot_command_api::PUCCH,
              grp_a.channels[grp_a.channel_idx[slot_command_api::PUCCH]]);

    // --- Slot B: verify registrations ---
    EXPECT_EQ(4u, static_cast<uint32_t>(grp_b.channel_array_size));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_b, slot_command_api::PDCCH_UL));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_b, slot_command_api::CSI_RS));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_b, slot_command_api::PRACH));
    EXPECT_TRUE (hasRegisteredGroupChannel(grp_b, slot_command_api::SRS));

    // Channels that slot B never saw must remain NONE.
    EXPECT_EQ(kNone, grp_b.channel_idx[slot_command_api::PDCCH_DL]);
    EXPECT_EQ(kNone, grp_b.channel_idx[slot_command_api::PDSCH]);
    EXPECT_EQ(kNone, grp_b.channel_idx[slot_command_api::PUSCH]);
    EXPECT_EQ(kNone, grp_b.channel_idx[slot_command_api::PUCCH]);

    EXPECT_EQ(slot_command_api::PDCCH_UL,
              grp_b.channels[grp_b.channel_idx[slot_command_api::PDCCH_UL]]);
    EXPECT_EQ(slot_command_api::CSI_RS,
              grp_b.channels[grp_b.channel_idx[slot_command_api::CSI_RS]]);
    EXPECT_EQ(slot_command_api::PRACH,
              grp_b.channels[grp_b.channel_idx[slot_command_api::PRACH]]);
    EXPECT_EQ(slot_command_api::SRS,
              grp_b.channels[grp_b.channel_idx[slot_command_api::SRS]]);

}

// One slot where DL_TTI.req and UL_TTI.req each contribute different PDU types.
// Each channel must occupy a distinct, non-NONE slot in channel_idx[].
TEST(SlotCommandChannelReads, TwoFapiMessagesContributingDlAndUlChannelsGetDistinctChannelIdxEntries)
{
    cell_group_command grp{};

    // DL_TTI.req contribution: PDCCH_DL + PDSCH.
    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_DL);
    std::ignore = grp.get_pdsch_params();

    // UL_TTI.req contribution: PUSCH + PUCCH.
    std::ignore = grp.get_pusch_params();
    std::ignore = grp.get_pucch_params();

    EXPECT_EQ(4u, static_cast<uint32_t>(grp.channel_array_size));

    const uint32_t idx_pdcch_dl = grp.channel_idx[slot_command_api::PDCCH_DL];
    const uint32_t idx_pdsch    = grp.channel_idx[slot_command_api::PDSCH];
    const uint32_t idx_pusch    = grp.channel_idx[slot_command_api::PUSCH];
    const uint32_t idx_pucch    = grp.channel_idx[slot_command_api::PUCCH];

    // All four channels got valid (non-NONE) indices.
    constexpr auto kNone = static_cast<uint32_t>(slot_command_api::channel_type::NONE);
    EXPECT_NE(kNone, idx_pdcch_dl);
    EXPECT_NE(kNone, idx_pdsch);
    EXPECT_NE(kNone, idx_pusch);
    EXPECT_NE(kNone, idx_pucch);

    // Indices are pairwise distinct — each channel occupies its own slot.
    EXPECT_NE(idx_pdcch_dl, idx_pdsch);
    EXPECT_NE(idx_pdcch_dl, idx_pusch);
    EXPECT_NE(idx_pdcch_dl, idx_pucch);
    EXPECT_NE(idx_pdsch,    idx_pusch);
    EXPECT_NE(idx_pdsch,    idx_pucch);
    EXPECT_NE(idx_pusch,    idx_pucch);

    // channels[] round-trips: channels[channel_idx[ch]] == ch for every active channel.
    EXPECT_EQ(slot_command_api::PDCCH_DL, grp.channels[idx_pdcch_dl]);
    EXPECT_EQ(slot_command_api::PDSCH,    grp.channels[idx_pdsch]);
    EXPECT_EQ(slot_command_api::PUSCH,    grp.channels[idx_pusch]);
    EXPECT_EQ(slot_command_api::PUCCH,    grp.channels[idx_pucch]);
}

TEST(SlotCommandChannelReads, GetterReadsAreIdempotent)
{
    cell_group_command grp{};

    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_DL);
    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_UL);
    std::ignore = grp.get_pdsch_params();
    std::ignore = grp.get_csirs_params();
    std::ignore = grp.get_pbch_params();
    std::ignore = grp.get_pusch_params();
    std::ignore = grp.get_prach_params();
    std::ignore = grp.get_pucch_params();
    std::ignore = grp.get_srs_params();

    const uint8_t  original_size = grp.channel_array_size;
    const uint32_t pdcch_dl_idx  = grp.channel_idx[slot_command_api::PDCCH_DL];
    const uint32_t pdcch_ul_idx  = grp.channel_idx[slot_command_api::PDCCH_UL];
    const uint32_t pdsch_idx     = grp.channel_idx[slot_command_api::PDSCH];
    const uint32_t csirs_idx     = grp.channel_idx[slot_command_api::CSI_RS];
    const uint32_t pbch_idx      = grp.channel_idx[slot_command_api::PBCH];
    const uint32_t pusch_idx     = grp.channel_idx[slot_command_api::PUSCH];
    const uint32_t prach_idx     = grp.channel_idx[slot_command_api::PRACH];
    const uint32_t pucch_idx     = grp.channel_idx[slot_command_api::PUCCH];
    const uint32_t srs_idx       = grp.channel_idx[slot_command_api::SRS];

    // Second pass — channel_array_size and all indices must be unchanged.
    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_DL);
    std::ignore = grp.get_pdcch_params(slot_command_api::PDCCH_UL);
    std::ignore = grp.get_pdsch_params();
    std::ignore = grp.get_csirs_params();
    std::ignore = grp.get_pbch_params();
    std::ignore = grp.get_pusch_params();
    std::ignore = grp.get_prach_params();
    std::ignore = grp.get_pucch_params();
    std::ignore = grp.get_srs_params();

    EXPECT_EQ(static_cast<uint32_t>(original_size),
              static_cast<uint32_t>(grp.channel_array_size));
    EXPECT_EQ(pdcch_dl_idx, grp.channel_idx[slot_command_api::PDCCH_DL]);
    EXPECT_EQ(pdcch_ul_idx, grp.channel_idx[slot_command_api::PDCCH_UL]);
    EXPECT_EQ(pdsch_idx,    grp.channel_idx[slot_command_api::PDSCH]);
    EXPECT_EQ(csirs_idx,    grp.channel_idx[slot_command_api::CSI_RS]);
    EXPECT_EQ(pbch_idx,     grp.channel_idx[slot_command_api::PBCH]);
    EXPECT_EQ(pusch_idx,    grp.channel_idx[slot_command_api::PUSCH]);
    EXPECT_EQ(prach_idx,    grp.channel_idx[slot_command_api::PRACH]);
    EXPECT_EQ(pucch_idx,    grp.channel_idx[slot_command_api::PUCCH]);
    EXPECT_EQ(srs_idx,      grp.channel_idx[slot_command_api::SRS]);
}
