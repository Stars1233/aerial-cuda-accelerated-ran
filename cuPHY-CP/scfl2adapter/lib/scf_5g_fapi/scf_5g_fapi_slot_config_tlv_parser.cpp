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

#include "scf_5g_fapi_slot_config_tlv_parser.hpp"

#include "cuphy.h"
#include "nvlog.hpp"
#include "nv_phy_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace scf_5g_fapi
{

namespace
{
constexpr std::uint16_t kLogTag = NVLOG_TAG_BASE_SCF_L2_ADAPTER + 3; // "SCF.PHY"
constexpr std::uint8_t kMaxNumerology = 4U; // 3GPP TS 38.211, section 4.2
constexpr std::uint32_t kMicrosecondsPerMs = 1000U;
constexpr std::uint32_t kMaxTddPeriodUs = 10000U;
constexpr std::size_t kSymbolsPerSlot = static_cast<std::size_t>(OFDM_SYMBOLS_PER_SLOT);

static_assert((std::uint64_t{1U} << kMaxNumerology) * kMaxTddPeriodUs <=
                  std::numeric_limits<std::uint32_t>::max(),
              "slot-count computation must not overflow uint32_t");
static_assert(((std::uint64_t{1U} << kMaxNumerology) * kMaxTddPeriodUs) / kMicrosecondsPerMs <=
                  std::numeric_limits<std::uint16_t>::max(),
              "derived slot count must fit the uint16_t return type");
} // namespace

tl::expected<std::uint16_t, SlotConfigError> parse_slot_config_tlv(
    const scf_fapi_tl_t& tlv,
    const std::uint8_t subCarrierCommon,
    const std::uint8_t tddPeriodNum,
    const std::span<nv::slot_detail_t> dest) noexcept
{
    if (dest.data() == nullptr) [[unlikely]] {
        return tl::make_unexpected(SlotConfigError::NullDestination);
    }
    if (subCarrierCommon > kMaxNumerology) [[unlikely]] {
        return tl::make_unexpected(SlotConfigError::InvalidNumerology);
    }

    const auto periodUs = static_cast<std::uint32_t>(nv::get_duration(tddPeriodNum).count());
    const std::uint32_t validEntries = ((1U << subCarrierCommon) * periodUs) / kMicrosecondsPerMs;

    if (validEntries > dest.size()) [[unlikely]] {
        return tl::make_unexpected(SlotConfigError::EntriesExceedCapacity);
    }

    const std::size_t requiredBytes = static_cast<std::size_t>(validEntries) * kSymbolsPerSlot;
    if (static_cast<std::size_t>(tlv.length) < requiredBytes) [[unlikely]] {
        return tl::make_unexpected(SlotConfigError::PayloadTooSmall);
    }

    const std::uint8_t* symbols = tlv.val;
    for (std::uint32_t i = 0U; i < validEntries; ++i, symbols += kSymbolsPerSlot) {
        std::uint8_t dlSymbols = 0U;
        std::uint8_t ulSymbols = 0U;
        std::int8_t startDl = -1;
        std::int8_t startUl = -1;
        for (std::size_t j = 0U; j < kSymbolsPerSlot; ++j) {
            const std::uint8_t symbol = symbols[j];
            if (symbol == nv::SlotConfig::DL_SLOT) {
                ++dlSymbols;
                startDl = (startDl < 0) ? static_cast<std::int8_t>(j) : startDl;
            } else if (symbol == nv::SlotConfig::UL_SLOT) {
                ++ulSymbols;
                startUl = (startUl < 0) ? static_cast<std::int8_t>(j) : startUl;
            }
        }

        nv::slot_type type = nv::slot_type::SLOT_NONE;
        if (dlSymbols < OFDM_SYMBOLS_PER_SLOT && ulSymbols < OFDM_SYMBOLS_PER_SLOT) {
            type = nv::slot_type::SLOT_SPECIAL;
        } else if (dlSymbols == OFDM_SYMBOLS_PER_SLOT) {
            type = nv::slot_type::SLOT_DOWNLINK;
        } else if (ulSymbols == OFDM_SYMBOLS_PER_SLOT) {
            type = nv::slot_type::SLOT_UPLINK;
        }

        auto& slot_cfg = dest[i];
        slot_cfg = nv::slot_detail_t{type, dlSymbols, ulSymbols, startDl, startUl};
        NVLOGD_FMT(kLogTag,
                   "{}: Slot type = {}, DL symbols = {}, UL symbols = {} Start DL = {} Start UL = {}",
                   __FUNCTION__,
                   +slot_cfg.type,
                   slot_cfg.max_dl_symbols,
                   slot_cfg.max_ul_symbols,
                   slot_cfg.start_sym_dl,
                   slot_cfg.start_sym_ul);
    }

    return static_cast<std::uint16_t>(validEntries);
}

} // namespace scf_5g_fapi
