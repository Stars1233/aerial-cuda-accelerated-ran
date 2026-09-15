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
 * @file
 * @brief Table-driven tests for SSB precoding-matrix cache indexing.
 *
 * Exercises resolve_ssb_pmw_slot(), the pure core extracted from
 * update_pm_weights_ssb_cuphy(). The caller itself is not unit-testable: it ends with
 * nv::PHYDriverProxy::getInstance(), which dereferences a static unique_ptr that stays
 * null until the driver calls make() -- so invoking it without a live driver segfaults.
 *
 * pm_group carries three differently-sized PMW caches (SSB / PDCCH / CSI-RS) and
 * historically indexed all of them with one shared counter, nCacheEntries. SSB now uses
 * its own nPmPbch. DoesNotAdvanceSharedCacheCounter is the regression guard.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "scf_5g_slot_commands_ssb_pbch.hpp"

namespace
{

using slot_command_api::pm_group;
using slot_command_api::pm_weights_t;

constexpr uint32_t CELL_INDEX = 0U;
constexpr uint16_t NUM_PORTS  = 4U;

// Arbitrary but distinct and non-zero PMI values for cache-key tests. The cache
// keys on equality only, so no 3GPP significance is intended. Zero is avoided
// because the caller treats pdu_pmi == 0 as "precoding disabled" before it ever
// reaches resolve_ssb_pmw_slot().
constexpr uint16_t PMI_A = 7U;
constexpr uint16_t PMI_B = 9U;
constexpr uint16_t PMI_C = 3U;
constexpr uint16_t PMI_D = 5U;
constexpr uint16_t PMI_E = 8U;

/// Bit position the cell index occupies in the cache key; mirrors the production
/// expression in update_pm_weights_ssb_cuphy().
constexpr uint32_t CELL_INDEX_SHIFT = 16U;

/// Cache key exactly as update_pm_weights_ssb_cuphy() forms it:
/// pmi | cell_index << CELL_INDEX_SHIFT.
[[nodiscard]] uint32_t make_cache_pmi(const uint16_t pmi, const uint32_t cell_index = CELL_INDEX) noexcept
{
    return static_cast<uint32_t>(pmi) | (cell_index << CELL_INDEX_SHIFT);
}

[[nodiscard]] pm_weights_t make_weights(const uint16_t layers = 1U)
{
    pm_weights_t w{};
    w.layers         = layers;
    w.ports          = NUM_PORTS;
    w.weights.nPorts = NUM_PORTS;
    return w;
}

/// pm_map entry for @p pmi. layers must be 1 or the lookup reports "no slot".
void add_pm_entry(scf_5g_fapi::pm_weight_map_t& map, const uint16_t pmi, const uint16_t layers = 1U)
{
    map[make_cache_pmi(pmi)] = make_weights(layers);
}

struct Fixture final
{
    pm_group                     pm{true, 1U};
    scf_5g_fapi::pm_weight_map_t pm_map{};

    [[nodiscard]] std::optional<uint32_t> resolve(const uint16_t pmi)
    {
        return scf_5g_fapi::resolve_ssb_pmw_slot(pm, pm_map, make_cache_pmi(pmi));
    }
};

// ---------------------------------------------------------------------------
// Table-driven: distinct PMIs take sequential slots; repeats reuse them.
// ---------------------------------------------------------------------------

struct CacheCase final
{
    std::string_view      name;
    std::vector<uint16_t> pmis;           ///< one lookup per entry, in order
    std::vector<uint32_t> expected_slots; ///< expected slot per lookup
    uint16_t              expected_n_pm_pbch;
};

class SsbPmCacheTest : public ::testing::TestWithParam<CacheCase>
{
};

TEST_P(SsbPmCacheTest, AssignsAndReusesCacheSlots)
{
    const auto& tc = GetParam();
    Fixture fx;
    for (const auto pmi : tc.pmis) { add_pm_entry(fx.pm_map, pmi); }

    std::vector<uint32_t> got;
    got.reserve(tc.pmis.size());
    for (const auto pmi : tc.pmis)
    {
        const auto slot = fx.resolve(pmi);
        ASSERT_TRUE(slot.has_value()) << "case: " << tc.name << " pmi " << pmi;
        got.push_back(*slot);
    }

    EXPECT_EQ(got, tc.expected_slots) << "case: " << tc.name;
    EXPECT_EQ(fx.pm.nPmPbch, tc.expected_n_pm_pbch) << "case: " << tc.name;

    // Cache and ssb_list stay in lockstep: every live slot describes its own entry.
    for (uint16_t i = 0; i < fx.pm.nPmPbch; ++i)
    {
        EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[i].nIndex, i) << "case: " << tc.name << " slot " << i;
        EXPECT_NE(fx.pm.ssb_pmw_idx_cache[i].pmwIdx, UINT32_MAX) << "case: " << tc.name;
        EXPECT_EQ(fx.pm.ssb_list[i].nPorts, NUM_PORTS) << "case: " << tc.name;
    }
    // Nothing beyond nPmPbch may have been written. Skip when full — there is no
    // sentinel slot past the last live entry (nPmPbch == size()).
    ASSERT_LT(static_cast<std::size_t>(fx.pm.nPmPbch), fx.pm.ssb_pmw_idx_cache.size())
        << "case: " << tc.name << " cache is full; no sentinel slot available";
    EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[fx.pm.nPmPbch].pmwIdx, UINT32_MAX) << "case: " << tc.name;
}

INSTANTIATE_TEST_SUITE_P(
    SsbPmWeights,
    SsbPmCacheTest,
    ::testing::Values(
        CacheCase{"single", {PMI_A}, {0U}, 1U},
        CacheCase{"two_distinct", {PMI_A, PMI_B}, {0U, 1U}, 2U},
        CacheCase{"repeat_reuses_slot", {PMI_A, PMI_A}, {0U, 0U}, 1U},
        CacheCase{"interleaved_repeats", {PMI_A, PMI_B, PMI_A, PMI_B}, {0U, 1U, 0U, 1U}, 2U},
        CacheCase{"three_distinct", {PMI_C, PMI_D, PMI_E}, {0U, 1U, 2U}, 3U}),
    [](const ::testing::TestParamInfo<CacheCase>& i) { return std::string{i.param.name}; });

// ---------------------------------------------------------------------------
// Regression guard for the shared-counter fix.
// ---------------------------------------------------------------------------

TEST(SsbPmWeights, DoesNotAdvanceSharedCacheCounter)
{
    Fixture fx;
    add_pm_entry(fx.pm_map, PMI_A);
    add_pm_entry(fx.pm_map, PMI_B);

    ASSERT_EQ(fx.pm.nCacheEntries, 0U);
    ASSERT_TRUE(fx.resolve(PMI_A).has_value());
    ASSERT_TRUE(fx.resolve(PMI_B).has_value());

    EXPECT_EQ(fx.pm.nPmPbch, 2U);
    // SSB must index its own cache with nPmPbch and leave the cross-channel counter
    // alone -- it is shared with the differently-sized PDCCH/CSI-RS caches.
    EXPECT_EQ(fx.pm.nCacheEntries, 0U)
        << "SSB advanced pm_group::nCacheEntries; it must use nPmPbch";
}

// ---------------------------------------------------------------------------
// Paths that must not consume a cache slot.
// ---------------------------------------------------------------------------

TEST(SsbPmWeights, MissingPmMapEntryConsumesNoSlot)
{
    Fixture fx;                       // pm_map deliberately empty
    EXPECT_FALSE(fx.resolve(PMI_A).has_value());
    EXPECT_EQ(fx.pm.nPmPbch, 0U);
    EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[0].pmwIdx, UINT32_MAX);
}

TEST(SsbPmWeights, MultiLayerEntryConsumesNoSlot)
{
    Fixture fx;
    add_pm_entry(fx.pm_map, PMI_A, /*layers=*/2U);   // only single-layer PMWs are cached
    EXPECT_FALSE(fx.resolve(PMI_A).has_value());
    EXPECT_EQ(fx.pm.nPmPbch, 0U);
    EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[0].pmwIdx, UINT32_MAX);
}

TEST(SsbPmWeights, OverSizedPortsConsumesNoSlot)
{
    // ports > MAX_DL_PORTS would overflow cuphyPmWOneLayer_t::matrix[MAX_DL_PORTS] in the
    // std::copy into ssb_list. The resolver must reject the entry, not corrupt memory.
    Fixture fx;
    pm_weights_t w = make_weights();                                  // layers == 1
    w.ports          = static_cast<uint16_t>(MAX_DL_PORTS + 1);
    w.weights.nPorts = static_cast<uint8_t>(MAX_DL_PORTS + 1);
    fx.pm_map[make_cache_pmi(PMI_A)] = w;

    EXPECT_FALSE(fx.resolve(PMI_A).has_value());
    EXPECT_EQ(fx.pm.nPmPbch, 0U);
    EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[0].pmwIdx, UINT32_MAX);
}

TEST(SsbPmWeights, DistinctCellsDoNotAliasSamePmi)
{
    // cache_pmi folds cell_index into the high half, so the same PMI on two cells
    // must occupy two slots.
    Fixture fx;
    const uint32_t pmi_cell0 = make_cache_pmi(PMI_A, 0U);
    const uint32_t pmi_cell1 = make_cache_pmi(PMI_A, 1U);
    fx.pm_map[pmi_cell0] = make_weights();
    fx.pm_map[pmi_cell1] = make_weights();

    const auto a = scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, pmi_cell0);
    const auto b = scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, pmi_cell1);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_NE(*a, *b);
    EXPECT_EQ(fx.pm.nPmPbch, 2U);
}

TEST(SsbPmWeights, FullCacheReportsNoSlotInsteadOfTerminating)
{
    // resolve_ssb_pmw_slot() is noexcept, so an unchecked .at() overflow would call
    // std::terminate rather than throw. Fill the cache exactly, then prove the next
    // insert reports "no slot" and leaves the counter alone.
    Fixture fx;
    const auto capacity = static_cast<uint32_t>(fx.pm.ssb_pmw_idx_cache.size());
    ASSERT_GT(capacity, 0U);

    for (uint32_t i = 0; i < capacity; ++i)
    {
        const uint32_t key = make_cache_pmi(static_cast<uint16_t>(i + 1U));
        fx.pm_map[key] = make_weights();
        const auto slot = scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, key);
        ASSERT_TRUE(slot.has_value()) << "unexpected miss at " << i;
        ASSERT_EQ(*slot, i);
    }
    ASSERT_EQ(fx.pm.nPmPbch, capacity);

    // One past capacity: a distinct PMI, so the find_if cannot hit.
    const uint32_t overflow_key = make_cache_pmi(static_cast<uint16_t>(capacity + 1U));
    fx.pm_map[overflow_key] = make_weights();
    EXPECT_FALSE(scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, overflow_key).has_value());
    EXPECT_EQ(fx.pm.nPmPbch, capacity) << "counter advanced past the array bound";

    // A previously cached PMI must still resolve once the cache is full.
    const auto hit = scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, make_cache_pmi(1U));
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(*hit, 0U);
}

TEST(SsbPmWeights, CacheFullFlagNamesTheRealFailureCause)
{
    // The caller used to re-derive "cache full" from nPmPbch >= capacity, so once the
    // cache filled, an unconfigured PMI was reported as a full cache. Only a genuine
    // capacity failure may raise the flag.
    Fixture fx;

    // Unconfigured PMI while the cache still has room.
    bool cache_full = false;
    EXPECT_FALSE(scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, make_cache_pmi(PMI_A),
                                                   &cache_full)
                     .has_value());
    EXPECT_FALSE(cache_full);

    const auto capacity = static_cast<uint32_t>(fx.pm.ssb_pmw_idx_cache.size());
    for (uint32_t i = 0; i < capacity; ++i)
    {
        const uint32_t key = make_cache_pmi(static_cast<uint16_t>(i + 1U));
        fx.pm_map[key]     = make_weights();
        ASSERT_TRUE(scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, key).has_value());
    }

    // Configured PMI that cannot be appended: the cache really is the reason.
    const uint32_t overflow_key = make_cache_pmi(static_cast<uint16_t>(capacity + 1U));
    fx.pm_map[overflow_key]     = make_weights();
    cache_full                  = false;
    EXPECT_FALSE(scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, overflow_key, &cache_full)
                     .has_value());
    EXPECT_TRUE(cache_full);

    // Unconfigured PMI with the cache full: still not a capacity failure.
    const uint32_t absent_key = make_cache_pmi(static_cast<uint16_t>(capacity + 2U));
    cache_full                = false;
    EXPECT_FALSE(scf_5g_fapi::resolve_ssb_pmw_slot(fx.pm, fx.pm_map, absent_key, &cache_full)
                     .has_value());
    EXPECT_FALSE(cache_full) << "unconfigured PMI reported as a full cache";
}

TEST(SsbPmWeights, StaleEntryBeyondLiveRangeIsNotAHit)
{
    // The lookup must scan only [0, nPmPbch). Plant an entry past the live range -- as a
    // missed reset() would leave -- and prove it is neither returned as a hit nor allowed
    // to shadow a fresh append. Without the bound, find_if over the whole array returns
    // the planted nIndex and this test fails.
    Fixture fx;
    const uint32_t key = make_cache_pmi(PMI_A);
    add_pm_entry(fx.pm_map, PMI_A);

    ASSERT_EQ(fx.pm.nPmPbch, 0U);
    constexpr std::size_t STALE_SLOT = 5U;  // any index past the live prefix
    fx.pm.ssb_pmw_idx_cache[STALE_SLOT].pmwIdx = key;
    fx.pm.ssb_pmw_idx_cache[STALE_SLOT].nIndex = static_cast<uint32_t>(STALE_SLOT);

    const auto slot = fx.resolve(PMI_A);
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(*slot, 0U) << "stale entry beyond nPmPbch was treated as a cache hit";
    EXPECT_EQ(fx.pm.nPmPbch, 1U);
}

TEST(SsbPmWeights, ResetClearsPerChannelCounter)
{
    Fixture fx;
    add_pm_entry(fx.pm_map, PMI_A);
    ASSERT_TRUE(fx.resolve(PMI_A).has_value());
    ASSERT_EQ(fx.pm.nPmPbch, 1U);

    fx.pm.reset();
    EXPECT_EQ(fx.pm.nPmPbch, 0U);
    EXPECT_EQ(fx.pm.ssb_pmw_idx_cache[0].pmwIdx, UINT32_MAX);
}

} // namespace
