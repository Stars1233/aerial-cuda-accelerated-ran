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
 * @file test_prach_dop.cpp
 * @brief Unit tests for the PRACH Tell-Don't-Ask helpers — MR 2a.
 *
 * Exercises every free function in scf_5g_fapi::prach::dop so the header is
 * fully type-checked and the invariants (nOccasion++, slot.type+slot_3gpp
 * pairing, prbs_size++, prb_info common-field initialisation) are verified.
 *
 * Test suites:
 *   PrachParamsAddOccasion — full(), single + multi add, capacity overflow
 *   PrachParamsSetGlobalConfig — mu/nfft pair, fluent return
 *   SlotInfoSetUplink — type + slot_3gpp set together
 *   SlotInfoTAddPrb — full(), single + multi add, capacity overflow,
 *                     returned pointer aliases the stored entry
 *   PrbInfoMakePrachPrb — common-field initialisation, direction = UL
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "scf_5g_fapi_prach_dop.hpp"

namespace
{

using scf_5g_fapi::prach::dop::OccasionEntry;
using scf_5g_fapi::prach::dop::add_occasion;
using scf_5g_fapi::prach::dop::add_prb;
using scf_5g_fapi::prach::dop::full;
using scf_5g_fapi::prach::dop::make_prach_prb;
using scf_5g_fapi::prach::dop::set_global_config;
using scf_5g_fapi::prach::dop::set_uplink;

namespace detail
{
// Test constants — small canonical values for the OccasionEntry fields.
constexpr int32_t  k_cell_index        = 7;
constexpr uint16_t k_phy_cell_id       = 123;
constexpr uint8_t  k_freq_index        = 2;
constexpr uint8_t  k_start_symbol      = 4;
constexpr uint16_t k_occa_prm_stat_idx = 42;
constexpr float    k_force_thr0        = 0.5f;
constexpr uint16_t k_n_uplink_streams  = 2;
} // namespace detail

// Helper: canonical OccasionEntry with the constants above.
[[nodiscard]] inline OccasionEntry make_canonical_entry() noexcept
{
    return OccasionEntry{
        .cell_index        = detail::k_cell_index,
        .force_thr0        = detail::k_force_thr0,
        .phy_cell_id       = detail::k_phy_cell_id,
        .occa_prm_stat_idx = detail::k_occa_prm_stat_idx,
        .n_uplink_streams  = detail::k_n_uplink_streams,
        .freq_index        = detail::k_freq_index,
        .start_symbol      = detail::k_start_symbol,
    };
}

} // namespace

// ---------------------------------------------------------------------------
// prach_params::add_occasion + full()
// ---------------------------------------------------------------------------

TEST(PrachParamsAddOccasion, EmptyParamsAreNotFull)
{
    slot_command_api::prach_params params{};
    EXPECT_FALSE(full(params));
    EXPECT_EQ(params.nOccasion, 0u);
}

TEST(PrachParamsAddOccasion, SingleAddSetsAllFieldsAndIncrementsCounter)
{
    slot_command_api::prach_params params{};
    const auto e = make_canonical_entry();

    EXPECT_TRUE(add_occasion(params, e));
    EXPECT_EQ(params.nOccasion, 1u);

    // Per-occasion arrays at index 0.
    EXPECT_EQ(params.freqIndex[0],          detail::k_freq_index);
    EXPECT_EQ(params.startSymbols[0],       detail::k_start_symbol);
    EXPECT_EQ(params.rach[0].occaPrmStatIdx, detail::k_occa_prm_stat_idx);
    EXPECT_EQ(params.rach[0].occaPrmDynIdx,  0u);          // matches occasion index
    EXPECT_FLOAT_EQ(params.rach[0].force_thr0, detail::k_force_thr0);
    EXPECT_EQ(params.rach[0].nUplinkStreams, detail::k_n_uplink_streams);

    // Cell index lists.
    ASSERT_EQ(params.cell_index_list.size(),     1u);
    ASSERT_EQ(params.phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params.cell_index_list[0],     detail::k_cell_index);
    EXPECT_EQ(params.phy_cell_index_list[0], static_cast<int32_t>(detail::k_phy_cell_id));
}

TEST(PrachParamsAddOccasion, MultipleAddsIncrementOccaDynIdx)
{
    slot_command_api::prach_params params{};

    // Add three distinct entries; verify occaPrmDynIdx matches insertion index.
    for (uint8_t i = 0; i < 3u; ++i)
    {
        OccasionEntry e{
            .cell_index        = static_cast<int32_t>(i),
            .phy_cell_id       = static_cast<uint16_t>(100u + i),
            .occa_prm_stat_idx = static_cast<uint16_t>(50u + i),
            .freq_index        = i,
            .start_symbol      = static_cast<uint8_t>(2u * i),
        };
        EXPECT_TRUE(add_occasion(params, e)) << "add_occasion[" << i << "] failed";
    }

    EXPECT_EQ(params.nOccasion, 3u);
    for (uint8_t i = 0; i < 3u; ++i)
    {
        SCOPED_TRACE(testing::Message{} << "occasion=" << static_cast<int>(i));
        EXPECT_EQ(params.freqIndex[i],           i);
        EXPECT_EQ(params.startSymbols[i],        2u * i);
        EXPECT_EQ(params.rach[i].occaPrmStatIdx, 50u + i);
        EXPECT_EQ(params.rach[i].occaPrmDynIdx,  i);
    }
}

TEST(PrachParamsAddOccasion, AddBeyondCapacityReturnsFalseAndPreservesCounter)
{
    slot_command_api::prach_params params{};

    // Fill to capacity.
    for (int i = 0; i < slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT; ++i)
    {
        EXPECT_TRUE(add_occasion(params, make_canonical_entry()))
            << "fill iteration " << i << " failed";
    }
    EXPECT_TRUE(full(params));
    EXPECT_EQ(params.nOccasion, slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT);

    // One more must fail without bumping nOccasion.
    EXPECT_FALSE(add_occasion(params, make_canonical_entry()));
    EXPECT_EQ(params.nOccasion, slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT);
}

// ---------------------------------------------------------------------------
// prach_params::set_global_config
// ---------------------------------------------------------------------------

TEST(PrachParamsSetGlobalConfig, SetsMuAndNfftAndReturnsRef)
{
    slot_command_api::prach_params params{};
    constexpr uint8_t  k_mu   = 1u;
    constexpr uint32_t k_nfft = 2048u;

    auto& returned = set_global_config(params, k_mu, k_nfft);
    EXPECT_EQ(params.mu,   k_mu);
    EXPECT_EQ(params.nfft, k_nfft);
    EXPECT_EQ(&returned, &params) << "must return the same reference for fluent chaining";
}

// ---------------------------------------------------------------------------
// slot_info::set_uplink
// ---------------------------------------------------------------------------

TEST(SlotInfoSetUplink, SetsTypeAndSlot3gppTogether)
{
    slot_command_api::slot_info slot{};
    constexpr uint16_t k_sfn  = 100u;
    constexpr uint16_t k_slot = 7u;
    constexpr uint64_t k_tick = 1234567u;
    const slot_command_api::slot_indication ind{k_sfn, k_slot, k_tick};

    set_uplink(slot, ind);

    EXPECT_EQ(slot.type, slot_command_api::SLOT_UPLINK);
    EXPECT_EQ(slot.slot_3gpp.sfn_,  k_sfn);
    EXPECT_EQ(slot.slot_3gpp.slot_, k_slot);
    EXPECT_EQ(slot.slot_3gpp.tick_, k_tick);
}

// ---------------------------------------------------------------------------
// slot_info_t::add_prb + full()
// ---------------------------------------------------------------------------

TEST(SlotInfoTAddPrb, EmptyIsNotFull)
{
    slot_command_api::slot_info_t sym_prbs{};
    EXPECT_FALSE(full(sym_prbs));
    EXPECT_EQ(sym_prbs.prbs_size, 0u);
}

TEST(SlotInfoTAddPrb, SingleAddReturnsAliasAndIncrementsCounter)
{
    // slot_info_t is large (~1 MB) — heap-allocate to keep stack frame sane.
    auto sym_prbs = std::make_unique<slot_command_api::slot_info_t>();
    const auto src = make_prach_prb(/*startPrb=*/0u, /*numPrb=*/6u,
                                     /*freqOffset=*/12, /*numSymbols=*/2u,
                                     /*filterIndex=*/3u);

    auto& out = add_prb(*sym_prbs, src);
    EXPECT_EQ(sym_prbs->prbs_size, 1u);
    EXPECT_EQ(&out, &sym_prbs->prbs[0]) << "returned reference must alias the stored entry";

    EXPECT_EQ(out.common.freqOffset,  12);
    EXPECT_EQ(out.common.numSymbols,  2u);
    EXPECT_EQ(out.common.filterIndex, 3u);
    EXPECT_EQ(out.common.direction,   slot_command_api::fh_dir_t::FH_DIR_UL);
}

TEST(SlotInfoTAddPrb, MultipleAddsAdvanceIndex)
{
    auto sym_prbs = std::make_unique<slot_command_api::slot_info_t>();

    for (uint8_t i = 0; i < 4u; ++i)
    {
        const auto src = make_prach_prb(/*startPrb=*/0u,
                                         /*numPrb=*/static_cast<uint16_t>(6u + i),
                                         /*freqOffset=*/static_cast<int32_t>(i),
                                         /*numSymbols=*/1u,
                                         /*filterIndex=*/i);
        auto& out = add_prb(*sym_prbs, src);
        EXPECT_EQ(&out, &sym_prbs->prbs[i]) << "iteration " << static_cast<int>(i);
    }
    EXPECT_EQ(sym_prbs->prbs_size, 4u);
    EXPECT_EQ(sym_prbs->prbs[3].common.filterIndex, 3u);
}

// add_prb now requires !full(sym_prbs) as a precondition (enforced by
// gsl_Expects). Calling it when full() is a contract violation, not a
// returnable error — gsl is configured with
// `-Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS`; throwing from a noexcept function
// terminates, so the death test matches the abort/terminate signature.
TEST(SlotInfoTAddPrbDeathTest, AddWhenFullViolatesPrecondition)
{
    auto sym_prbs = std::make_unique<slot_command_api::slot_info_t>();
    sym_prbs->prbs_size = MAX_PRB_INFO;     // simulate full state without filling
    ASSERT_TRUE(full(*sym_prbs));

    const auto src = make_prach_prb(0u, 6u, 0, 1u, 1u);
    EXPECT_DEATH(static_cast<void>(add_prb(*sym_prbs, src)), ".*")
        << "add_prb when full() must trigger gsl_Expects precondition failure";
}

// ---------------------------------------------------------------------------
// prb_info_t::make_prach_prb factory
// ---------------------------------------------------------------------------

TEST(PrbInfoMakePrachPrb, PopulatesCommonFieldsAndSetsUlDirection)
{
    constexpr uint16_t k_start_prb    = 0u;
    constexpr uint16_t k_num_prb      = 6u;
    constexpr int32_t  k_freq_offset  = -3;
    constexpr uint8_t  k_num_symbols  = 4u;
    constexpr uint8_t  k_filter_index = 2u;

    const auto p = make_prach_prb(k_start_prb, k_num_prb, k_freq_offset,
                                   k_num_symbols, k_filter_index);

    EXPECT_EQ(p.common.freqOffset,  k_freq_offset);
    EXPECT_EQ(p.common.numSymbols,  k_num_symbols);
    EXPECT_EQ(p.common.filterIndex, k_filter_index);
    EXPECT_EQ(p.common.direction,   slot_command_api::fh_dir_t::FH_DIR_UL);
    // startPrb/numPrb live on common (set by the prb_info_t_(startPrb, numPrb) ctor).
    EXPECT_EQ(p.common.startPrbc,   k_start_prb);
    EXPECT_EQ(p.common.numPrbc,     k_num_prb);
}
