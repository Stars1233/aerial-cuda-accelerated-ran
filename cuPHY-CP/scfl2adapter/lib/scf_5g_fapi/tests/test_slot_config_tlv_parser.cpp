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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

#include "aerial/casts/casts.hpp"
#include "cuphy.h"
#include "nv_phy_fapi_msg_common.hpp"
#include "scf_5g_fapi_slot_config_tlv_parser.hpp"

namespace {

constexpr auto kSymbolsPerSlot = static_cast<std::size_t>(OFDM_SYMBOLS_PER_SLOT);

// TDD periodicity indices consumed by nv::get_duration (see nv_phy_utils.hpp).
constexpr std::uint8_t kPeriod625us = 1U; // 0.625 ms -> 0 slots at mu 0
constexpr std::uint8_t kPeriod5ms = 6U;   // 5 ms
constexpr std::uint8_t kPeriod10ms = 7U;  // 10 ms

/**
 * Builds a flat SLOT_CONFIG TLV payload: OFDM_SYMBOLS_PER_SLOT symbol bytes per slot.
 */
class SlotConfigTlvBuilder final
{
public:
    SlotConfigTlvBuilder& add_downlink_slot()
    {
        return add_slot(nv::SlotConfig::DL_SLOT, kSymbolsPerSlot, nv::SlotConfig::DL_SLOT);
    }

    SlotConfigTlvBuilder& add_uplink_slot()
    {
        return add_slot(nv::SlotConfig::UL_SLOT, kSymbolsPerSlot, nv::SlotConfig::UL_SLOT);
    }

    // Half downlink, half uplink -> neither symbol count reaches a full slot -> SLOT_SPECIAL.
    SlotConfigTlvBuilder& add_special_slot()
    {
        return add_slot(nv::SlotConfig::DL_SLOT, kSymbolsPerSlot / 2U, nv::SlotConfig::UL_SLOT);
    }

    [[nodiscard]] scf_fapi_tl_t* build()
    {
        const auto storageBytes = sizeof(scf_fapi_tl_t) + payload_.size();
        const auto storageWords =
            (storageBytes + sizeof(std::max_align_t) - 1U) / sizeof(std::max_align_t);
        storage_.assign(storageWords, std::max_align_t{});
        auto* tlv = aerial::casts::assume_cast<scf_fapi_tl_t>(storage_.data());
        tlv->tag = CONFIG_TLV_SLOT_CONFIG;
        tlv->length = static_cast<decltype(tlv->length)>(payload_.size());
        std::copy(payload_.begin(), payload_.end(), tlv->val);
        return tlv;
    }

private:
    // Fill a slot with `firstCount` bytes of `first`, the remainder with `rest`.
    SlotConfigTlvBuilder& add_slot(const uint8_t first, const std::size_t firstCount,
                                   const uint8_t rest)
    {
        for (std::size_t sym = 0U; sym < kSymbolsPerSlot; ++sym) {
            payload_.push_back(sym < firstCount ? first : rest);
        }
        return *this;
    }

    std::vector<uint8_t> payload_; //!< Serialized symbol bytes.
    // max_align_t guarantees storage suitable for scf_fapi_tl_t before assume_cast.
    std::vector<std::max_align_t> storage_; //!< Backing storage for the built scf_fapi_tl_t.
};

// Destination buffer pre-filled with a recognizable sentinel so "left untouched"
// is verifiable after a rejection.
class SlotDetailBuffer final
{
public:
    SlotDetailBuffer()
        : entries_(nv::NV_MAX_TDD_PERIODICITY)
    {
        for (auto& entry : entries_) {
            entry.type = kSentinelType;
            entry.max_dl_symbols = kSentinelU8;
            entry.max_ul_symbols = kSentinelU8;
            entry.start_sym_dl = kSentinelI8;
            entry.start_sym_ul = kSentinelI8;
        }
    }

    [[nodiscard]] std::span<nv::slot_detail_t> span() { return entries_; }
    [[nodiscard]] const nv::slot_detail_t& at(const std::size_t i) const { return entries_.at(i); }

    [[nodiscard]] bool is_sentinel(const std::size_t i) const
    {
        const auto& e = entries_.at(i);
        return e.type == kSentinelType && e.max_dl_symbols == kSentinelU8 &&
               e.max_ul_symbols == kSentinelU8 && e.start_sym_dl == kSentinelI8 &&
               e.start_sym_ul == kSentinelI8;
    }

    static constexpr auto kSentinelType = static_cast<nv::slot_type>(0x7F);
    static constexpr std::uint8_t kSentinelU8 = 0xEEU;
    static constexpr std::int8_t kSentinelI8 = 0x7F;

private:
    std::vector<nv::slot_detail_t> entries_;
};

} // namespace

TEST(SlotConfigTlvParser, ClassifiesSlotTypes)
{
    // mu = 0, 5 ms period -> 5 slots.
    SlotConfigTlvBuilder builder;
    builder.add_downlink_slot().add_uplink_slot().add_special_slot().add_downlink_slot().add_uplink_slot();

    SlotDetailBuffer dest;
    const auto result = scf_5g_fapi::parse_slot_config_tlv(*builder.build(), 0U, kPeriod5ms, dest.span());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5U);
    EXPECT_EQ(dest.at(0).type, nv::slot_type::SLOT_DOWNLINK);
    EXPECT_EQ(dest.at(0).max_dl_symbols, OFDM_SYMBOLS_PER_SLOT);
    EXPECT_EQ(dest.at(0).start_sym_dl, 0);
    EXPECT_EQ(dest.at(1).type, nv::slot_type::SLOT_UPLINK);
    EXPECT_EQ(dest.at(1).start_sym_ul, 0);
    EXPECT_EQ(dest.at(2).type, nv::slot_type::SLOT_SPECIAL);
    EXPECT_EQ(dest.at(2).max_dl_symbols, OFDM_SYMBOLS_PER_SLOT / 2);
    EXPECT_EQ(dest.at(2).max_ul_symbols, OFDM_SYMBOLS_PER_SLOT - OFDM_SYMBOLS_PER_SLOT / 2);
    EXPECT_TRUE(dest.is_sentinel(5)); // first entry past the written range is untouched
}

TEST(SlotConfigTlvParser, ReturnsZeroForSubMillisecondPeriod)
{
    // mu = 0, 0.625 ms period -> (0.625 truncated) 0 slots; empty payload is valid.
    SlotConfigTlvBuilder builder;

    SlotDetailBuffer dest;
    const auto result = scf_5g_fapi::parse_slot_config_tlv(*builder.build(), 0U, kPeriod625us, dest.span());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0U);
    EXPECT_TRUE(dest.is_sentinel(0));
}

TEST(SlotConfigTlvParser, FillsToCapacityBoundary)
{
    // mu = 3 (8 slots/ms), 10 ms period -> 80 slots == NV_MAX_TDD_PERIODICITY.
    SlotConfigTlvBuilder builder;
    for (std::size_t i = 0U; i < nv::NV_MAX_TDD_PERIODICITY; ++i) {
        builder.add_downlink_slot();
    }

    SlotDetailBuffer dest;
    const auto result = scf_5g_fapi::parse_slot_config_tlv(*builder.build(), 3U, kPeriod10ms, dest.span());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), nv::NV_MAX_TDD_PERIODICITY);
    EXPECT_EQ(dest.at(nv::NV_MAX_TDD_PERIODICITY - 1).type, nv::slot_type::SLOT_DOWNLINK);
}

TEST(SlotConfigTlvParser, AcceptsMaximumNumerologyWithShortPeriod)
{
    // mu = 4 (16 slots/ms), 0.625 ms period -> 10 slots.
    SlotConfigTlvBuilder builder;
    for (std::size_t i = 0U; i < 10U; ++i) {
        builder.add_downlink_slot();
    }

    SlotDetailBuffer dest;
    const auto result = scf_5g_fapi::parse_slot_config_tlv(*builder.build(), 4U, kPeriod625us, dest.span());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 10U);
    EXPECT_TRUE(dest.is_sentinel(10));
}

TEST(SlotConfigTlvParser, RejectsCountOnePastDestinationCapacity)
{
    // mu = 4, 5 ms period -> 80 slots; expose a 79-entry destination.
    SlotConfigTlvBuilder builder;
    for (std::size_t i = 0U; i < nv::NV_MAX_TDD_PERIODICITY; ++i) {
        builder.add_downlink_slot();
    }

    SlotDetailBuffer dest;
    const auto result = scf_5g_fapi::parse_slot_config_tlv(
        *builder.build(), 4U, kPeriod5ms, dest.span().first(nv::NV_MAX_TDD_PERIODICITY - 1U));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), scf_5g_fapi::SlotConfigError::EntriesExceedCapacity);
    EXPECT_TRUE(dest.is_sentinel(0));
}

// Each rejection path: build `slots` downlink slots, parse with the given mu/period and either a
// valid or a null destination span, and assert the expected error with the buffer left untouched.
struct RejectCase final
{
    const char* name = nullptr;
    std::uint8_t mu = 0U;
    std::uint8_t period = 0U;
    std::size_t slots = 0U;
    bool null_dest = false;
    scf_5g_fapi::SlotConfigError expected = scf_5g_fapi::SlotConfigError::NullDestination;
};

class SlotConfigTlvParserReject : public ::testing::TestWithParam<RejectCase>
{
};

TEST_P(SlotConfigTlvParserReject, ReturnsExpectedError)
{
    const RejectCase& tc = GetParam();
    SlotConfigTlvBuilder builder;
    for (std::size_t i = 0U; i < tc.slots; ++i) {
        builder.add_downlink_slot();
    }

    SlotDetailBuffer dest;
    const auto target = tc.null_dest ? std::span<nv::slot_detail_t>{} : dest.span();
    const auto result = scf_5g_fapi::parse_slot_config_tlv(*builder.build(), tc.mu, tc.period, target);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), tc.expected);
    if (!tc.null_dest) {
        EXPECT_TRUE(dest.is_sentinel(0)); // rejection leaves the destination untouched
    }
}

INSTANTIATE_TEST_SUITE_P(
    SlotConfigTlvParser, SlotConfigTlvParserReject,
    ::testing::Values(
        // mu = 4 (16 slots/ms) x 10 ms = 160 > 80 — the reported exploit.
        RejectCase{"CountExceedsCapacity", 4U, kPeriod10ms, 1U, false, scf_5g_fapi::SlotConfigError::EntriesExceedCapacity},
        RejectCase{"InvalidNumerology", 5U, kPeriod5ms, 1U, false, scf_5g_fapi::SlotConfigError::InvalidNumerology},
        // mu = 0, 5 ms needs 5 slots' worth of bytes; supply only 3.
        RejectCase{"TruncatedPayload", 0U, kPeriod5ms, 3U, false, scf_5g_fapi::SlotConfigError::PayloadTooSmall},
        RejectCase{"NullDestination", 0U, kPeriod5ms, 1U, true, scf_5g_fapi::SlotConfigError::NullDestination}),
    [](const ::testing::TestParamInfo<RejectCase>& info) { return info.param.name; });
