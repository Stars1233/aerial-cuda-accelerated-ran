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

#ifdef SCF_FAPI_10_04

#include "scf_5g_fapi_uci_config_parser.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace scf_5g_fapi
{
namespace
{

struct uci_config_map_metadata final
{
    cuphyCsi2MapPrm_t params{}; //!< Validated destination map parameters.
    std::size_t payloadOffset{}; //!< Offset of this map's payload data in the TLV value.
};

} // namespace

uci_config_parse_result parse_uci_config_tlv(const scf_fapi_tl_t& tlv,
                                             uint16_t* destMapBuf,
                                             cuphyCsi2MapPrm_t* destMapParamBuf) noexcept
{
    constexpr auto kMaxMapEntries = static_cast<std::size_t>(CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL);
    constexpr auto kMaxMapCount = static_cast<std::size_t>(CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL);
    static_assert(kMaxMapEntries <= (std::numeric_limits<std::size_t>::max() / sizeof(uint16_t)),
                  "kMaxMapEntries * sizeof(uint16_t) must not overflow size_t");
    static_assert(kMaxMapEntries <= std::numeric_limits<uint16_t>::max(),
                  "CSI-2 map size must fit in cuphyCsi2MapPrm_t::csi2MapSize");
    static_assert(kMaxMapEntries <= std::numeric_limits<uint32_t>::max(),
                  "CSI-2 map offset must fit in cuphyCsi2MapPrm_t::csi2MapStartIdx");

    if (destMapBuf == nullptr || destMapParamBuf == nullptr) {
        return {false, "destination buffer is null", 0U};
    }

    const auto* const payload = tlv.val;
    const auto payloadLen = static_cast<std::size_t>(tlv.length);
    std::size_t offset = 0U;

    const auto has_bytes = [payloadLen](const std::size_t currentOffset, const std::size_t bytes) noexcept {
        return currentOffset <= payloadLen && bytes <= (payloadLen - currentOffset);
    };

    const auto read_u16 = [&payload, &offset, &has_bytes](uint16_t& value) noexcept {
        if (!has_bytes(offset, sizeof(value))) {
            return false;
        }
        std::memcpy(&value, payload + offset, sizeof(value));
        offset += sizeof(value);
        return true;
    };

    uint16_t numUci2Maps = 0U;
    if (!read_u16(numUci2Maps)) {
        return {false, "payload is shorter than numUci2Maps", 0U};
    }

    if (numUci2Maps > kMaxMapCount) {
        return {false, "numUci2Maps exceeds destination capacity", 0U};
    }

    std::array<uci_config_map_metadata, kMaxMapCount> maps{};
    std::size_t mapOffset = 0U;

    for (uint16_t mapIdx = 0U; mapIdx < numUci2Maps; ++mapIdx) {
        if (!has_bytes(offset, sizeof(uint8_t))) {
            return {false, "payload is truncated before numPart1Params", 0U};
        }

        const uint8_t numPart1Params = payload[offset];
        offset += sizeof(uint8_t);

        if (!has_bytes(offset, numPart1Params)) {
            return {false, "payload is truncated before Part1 parameter sizes", 0U};
        }

        uint32_t sigma = 0U;
        for (uint8_t paramIdx = 0U; paramIdx < numPart1Params; ++paramIdx) {
            sigma += payload[offset + paramIdx];
        }
        offset += numPart1Params;

        if (sigma >= std::numeric_limits<std::size_t>::digits) {
            return {false, "sigma shift exceeds size_t width", 0U};
        }

        const std::size_t uciMapSize = std::size_t{1U} << sigma;
        if (uciMapSize > kMaxMapEntries || uciMapSize > (kMaxMapEntries - mapOffset)) {
            return {false, "UCI map entries exceed destination capacity", 0U};
        }

        const std::size_t mapBytes = uciMapSize * sizeof(uint16_t);
        if (!has_bytes(offset, mapBytes)) {
            return {false, "payload is truncated before UCI map data", 0U};
        }

        maps[mapIdx].params.csi2MapSize = static_cast<uint16_t>(uciMapSize);
        maps[mapIdx].params.csi2MapStartIdx = static_cast<uint32_t>(mapOffset);
        maps[mapIdx].payloadOffset = offset;
        mapOffset += uciMapSize;
        offset += mapBytes;
    }

    if (offset != payloadLen) {
        return {false, "payload has trailing bytes after UCI map data", 0U};
    }

    for (uint16_t mapIdx = 0U; mapIdx < numUci2Maps; ++mapIdx) {
        const auto& map = maps[mapIdx];
        const auto uciMapSize = static_cast<std::size_t>(map.params.csi2MapSize);
        const std::size_t mapBytes = uciMapSize * sizeof(uint16_t);
        std::memcpy(destMapBuf + map.params.csi2MapStartIdx, payload + map.payloadOffset, mapBytes);
        destMapParamBuf[mapIdx] = map.params;
    }

    return {true, nullptr, numUci2Maps};
}

} // namespace scf_5g_fapi

#endif // SCF_FAPI_10_04
