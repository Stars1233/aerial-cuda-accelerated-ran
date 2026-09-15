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
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_uci_config_parser.hpp"

namespace {

/**
 * Helper for constructing UCI CONFIG TLV payloads in parser tests.
 */
class UciConfigTlvBuilder final
{
public:
    /**
     * Append one payload byte.
     *
     * @param[in] value Byte value to append.
     */
    UciConfigTlvBuilder& append(const uint8_t value)
    {
        payload_.push_back(value);
        return *this;
    }

    /**
     * Append one 16-bit payload value using explicit little-endian byte order.
     *
     * @param[in] value 16-bit value to append.
     */
    UciConfigTlvBuilder& append(const uint16_t value)
    {
        payload_.push_back(static_cast<uint8_t>(value & 0xFFU));
        payload_.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
        return *this;
    }

    /**
     * Append CSI-2 map entries.
     *
     * @param[in] values Map entries to append.
     */
    UciConfigTlvBuilder& append_map(const std::initializer_list<uint16_t> values)
    {
        for (const uint16_t value : values) {
            append(value);
        }
        return *this;
    }

    /**
     * Build a TLV object backed by the builder storage.
     *
     * @return Pointer to the constructed scf_fapi_tl_t storage. This [[nodiscard]]
     *     return value must be checked by the caller.
     */
    [[nodiscard]] scf_fapi_tl_t* build()
    {
        storage_.assign(sizeof(scf_fapi_tl_t) + payload_.size(), uint8_t{0});
        auto* tlv = aerial::casts::assume_cast<scf_fapi_tl_t>(storage_.data());
        tlv->tag = CONFIG_TLV_UCI_CONFIG;
        tlv->length = static_cast<decltype(tlv->length)>(payload_.size());
        std::copy(payload_.begin(), payload_.end(), tlv->val);
        return tlv;
    }

private:
    std::vector<uint8_t> payload_; //!< Serialized TLV payload bytes.
    std::vector<uint8_t> storage_; //!< Backing storage for the built scf_fapi_tl_t.
};

struct ParseBuffers final
{
    std::vector<uint16_t> maps =
        std::vector<uint16_t>(CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL,
                              uint16_t{0xCAFE}); //!< Destination map buffer with sentinel values.
    std::vector<cuphyCsi2MapPrm_t> params =
        std::vector<cuphyCsi2MapPrm_t>(CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL,
                                       cuphyCsi2MapPrm_t{0xFFFF,
                                                         0xFFFFFFFFU}); //!< Destination parameter sentinels.
};

} // namespace

TEST(UciConfigParser, ParsesSingleMap)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{1U}) // numUci2Maps
        .append(uint8_t{2U})     // numPart1Params
        .append(uint8_t{1U})
        .append(uint8_t{1U}) // sigma = 2, map size = 4
        .append_map({0x10U, 0x20U, 0x30U, 0x40U});

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.nCsi2Maps, 1U);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 4U);
    EXPECT_EQ(buffers.params[0].csi2MapStartIdx, 0U);
    EXPECT_EQ(buffers.maps[0], 0x10U);
    EXPECT_EQ(buffers.maps[1], 0x20U);
    EXPECT_EQ(buffers.maps[2], 0x30U);
    EXPECT_EQ(buffers.maps[3], 0x40U);
    EXPECT_EQ(buffers.maps[4], 0xCAFEU);
}

TEST(UciConfigParser, ParsesMultipleMapsWithOffsets)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{2U}) // numUci2Maps
        .append(uint8_t{1U})
        .append(uint8_t{1U}) // sigma = 1, map size = 2
        .append_map({0x01U, 0x02U})
        .append(uint8_t{2U})
        .append(uint8_t{1U})
        .append(uint8_t{1U}) // sigma = 2, map size = 4
        .append_map({0x03U, 0x04U, 0x05U, 0x06U});

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.nCsi2Maps, 2U);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 2U);
    EXPECT_EQ(buffers.params[0].csi2MapStartIdx, 0U);
    EXPECT_EQ(buffers.params[1].csi2MapSize, 4U);
    EXPECT_EQ(buffers.params[1].csi2MapStartIdx, 2U);
    EXPECT_EQ(buffers.maps[5], 0x06U);
    EXPECT_EQ(buffers.maps[6], 0xCAFEU);
}

TEST(UciConfigParser, AcceptsZeroMaps)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{0U}); // numUci2Maps

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, ParsesExactBoundaryMapCount)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL});

    for (uint16_t mapIdx = 0U; mapIdx < CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL; ++mapIdx) {
        builder.append(uint8_t{1U}) // numPart1Params
            .append(uint8_t{0U})    // sigma = 0, map size = 1
            .append_map({mapIdx});
    }

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.nCsi2Maps, CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL);
    for (uint16_t mapIdx = 0U; mapIdx < CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL; ++mapIdx) {
        EXPECT_EQ(buffers.params[mapIdx].csi2MapSize, 1U);
        EXPECT_EQ(buffers.params[mapIdx].csi2MapStartIdx, mapIdx);
        EXPECT_EQ(buffers.maps[mapIdx], mapIdx);
    }
    EXPECT_EQ(buffers.maps[CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL], 0xCAFEU);
}

TEST(UciConfigParser, RejectsMapCountPastDestinationCapacity)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{static_cast<uint16_t>(CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL + 1U)});

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsEmptyPayload)
{
    UciConfigTlvBuilder builder;

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsNullDestinationBuffers)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{0U});

    ParseBuffers buffers;
    const auto nullMapResult = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), nullptr, buffers.params.data());
    EXPECT_FALSE(nullMapResult.ok);
    EXPECT_EQ(nullMapResult.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);

    const auto nullParamResult = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), nullptr);
    EXPECT_FALSE(nullParamResult.ok);
    EXPECT_EQ(nullParamResult.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
}

TEST(UciConfigParser, RejectsMapSizePastDestinationCapacity)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{1U})
        .append(uint8_t{1U})
        .append(uint8_t{13U}); // 8192 entries, larger than the 4096-entry destination

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsSigmaAtShiftWidth)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{1U})
        .append(uint8_t{1U})
        .append(static_cast<uint8_t>(std::numeric_limits<std::size_t>::digits));

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsTruncatedPartSizeList)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{1U})
        .append(uint8_t{3U})
        .append(uint8_t{1U}); // two Part1 size bytes are missing

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsTrailingPayloadBytes)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{0U})
        .append(uint8_t{0xEEU});

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}

TEST(UciConfigParser, RejectsTruncatedMapData)
{
    UciConfigTlvBuilder builder;
    builder.append(uint16_t{1U})
        .append(uint8_t{1U})
        .append(uint8_t{2U}) // sigma = 2, requires 4 uint16_t map entries
        .append_map({0x01U, 0x02U});

    ParseBuffers buffers;
    const auto result = scf_5g_fapi::parse_uci_config_tlv(*builder.build(), buffers.maps.data(), buffers.params.data());

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.nCsi2Maps, 0U);
    EXPECT_EQ(buffers.maps[0], 0xCAFEU);
    EXPECT_EQ(buffers.params[0].csi2MapSize, 0xFFFFU);
}
