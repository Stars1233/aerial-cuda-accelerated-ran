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

// Unit tests for scf_5g_fapi::decode_reconfig_tlvs — the shared reconfig TLV
// decoder used by both the serial (on_config_request) and offload
// (on_config_request_offload) paths. Feeds a synthetic CONFIG.request TLV
// buffer and asserts the decoded nv::cell_update_config fields, with no PHY /
// driver / GPU state involved.

#include <gtest/gtest.h>

#include "scf_5g_fapi_config_decode.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "aerial/casts/casts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

using scf_5g_fapi::decode_reconfig_tlvs;

namespace {

// Builds CONFIG.request TLVs into a fixed buffer using scf_fapi_tl_t::Set<>,
// which lays out tag/length/value and 4-byte-aligns each entry exactly as the
// decoder's TLV walk expects.
class ConfigRequestBuilder
{
public:
    scf_fapi_config_request_msg_t& msg()
    {
        auto* m = aerial::casts::assume_cast<scf_fapi_config_request_msg_t>(buf_.data());
        m->msg_body.num_tlvs = num_tlvs_;
        return *m;
    }

    template <typename T>
    ConfigRequestBuilder& add(uint16_t tag, uint32_t value)
    {
        // Bound the write to the fixed buffer; Set<>() advances by the 4-byte-aligned
        // TLV size, so require that much room before touching cursor_/num_tlvs_.
        const std::size_t write_size = sizeof(scf_fapi_tl_t) + ((sizeof(T) + 3) / 4) * 4;
        const std::size_t remaining  = static_cast<std::size_t>(buf_.data() + buf_.size() - cursor_);
        if(write_size > remaining)
        {
            throw std::length_error("ConfigRequestBuilder: TLV exceeds fixed buffer");
        }
        auto* tl = aerial::casts::assume_cast<scf_fapi_tl_t>(cursor_);
        cursor_ = static_cast<uint8_t*>(tl->Set<T>(tag, value));
        ++num_tlvs_;
        return *this;
    }

private:
    alignas(8) std::array<uint8_t, 1024> buf_{};
    // TLVs start at msg_body.tlvs[0]; offset past msg_hdr + num_tlvs byte.
    uint8_t* cursor_ = &aerial::casts::assume_cast<scf_fapi_config_request_msg_t>(buf_.data())->msg_body.tlvs[0];
    uint8_t  num_tlvs_ = 0;
};

} // namespace

// A representative reconfig request: bandwidths, phy cell id, PRACH occasion
// config, one root-sequence entry, and a DBT-PDU presence flag.
TEST(ConfigTlvDecode, DecodesReconfigSubsetAndDbtFlag)
{
    ConfigRequestBuilder b;
    b.add<uint16_t>(CONFIG_TLV_DL_BANDWIDTH,           273)
     .add<uint16_t>(CONFIG_TLV_UL_BANDWIDTH,           106)  // distinct from DL so a DL<->UL swap is caught
     .add<uint16_t>(CONFIG_TLV_PHY_CELL_ID,            42)
     .add<uint8_t>(CONFIG_TLV_NUM_PRACH_FD_OCCASIONS,  4)
     .add<uint16_t>(CONFIG_TLV_PRACH_ROOT_SEQ_INDEX,   100)  // -> root_sequence[0]
     .add<uint8_t>(CONFIG_TLV_NUM_ROOT_SEQ,            8)
     .add<uint16_t>(CONFIG_TLV_K1,                     2)
     .add<uint8_t>(CONFIG_TLV_PRACH_ZERO_CORR_CONF,    11)
     .add<uint8_t>(CONFIG_TLV_PRACH_CONFIG_INDEX,      160)
     .add<uint8_t>(CONFIG_TLV_RESTRICTED_SET_CONFIG,   1)
     .add<uint8_t>(CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU, 0);

    nv::cell_update_config out{};
    const bool dbt_present = decode_reconfig_tlvs(b.msg(), out);

    EXPECT_EQ(out.carrier_config_.dl_bandwidth, 273);
    EXPECT_EQ(out.carrier_config_.ul_bandwidth, 106);
    EXPECT_EQ(out.cell_config_.phy_cell_id,     42);
    EXPECT_EQ(out.prach_config_.num_prach_fd_occasions, 4);
    EXPECT_EQ(out.prach_config_.prach_conf_index,       160);
    EXPECT_EQ(out.prach_config_.restricted_set_config,  1);
    EXPECT_EQ(out.prach_config_.root_sequence[0].seq_index,            100);
    EXPECT_EQ(out.prach_config_.root_sequence[0].number_root_sequence, 8);
    EXPECT_EQ(out.prach_config_.root_sequence[0].k1,                   2);
    EXPECT_EQ(out.prach_config_.root_sequence[0].zero_conf,            11);
    EXPECT_TRUE(dbt_present);
}

// Absent DBT-PDU TLV -> returns false.
TEST(ConfigTlvDecode, NoDbtTlvLeavesFlagFalse)
{
    ConfigRequestBuilder b;
    b.add<uint16_t>(CONFIG_TLV_PHY_CELL_ID, 7);

    nv::cell_update_config out{};
    const bool dbt_present = decode_reconfig_tlvs(b.msg(), out);

    EXPECT_EQ(out.cell_config_.phy_cell_id, 7);
    EXPECT_FALSE(dbt_present);
}

// Two root-sequence groups: PRACH_ROOT_SEQ_INDEX advances the fd-occasion index
// so the second group lands in root_sequence[1].
TEST(ConfigTlvDecode, MultipleRootSequenceGroupsIndexIndependently)
{
    ConfigRequestBuilder b;
    b.add<uint16_t>(CONFIG_TLV_PRACH_ROOT_SEQ_INDEX, 10)  // -> root_sequence[0]
     .add<uint8_t>(CONFIG_TLV_NUM_ROOT_SEQ,          1)
     .add<uint16_t>(CONFIG_TLV_PRACH_ROOT_SEQ_INDEX, 20)  // -> root_sequence[1]
     .add<uint8_t>(CONFIG_TLV_NUM_ROOT_SEQ,          2);

    nv::cell_update_config out{};
    EXPECT_FALSE(decode_reconfig_tlvs(b.msg(), out));  // no DBT TLV in this request

    EXPECT_EQ(out.prach_config_.root_sequence[0].seq_index,            10);
    EXPECT_EQ(out.prach_config_.root_sequence[0].number_root_sequence, 1);
    EXPECT_EQ(out.prach_config_.root_sequence[1].seq_index,            20);
    EXPECT_EQ(out.prach_config_.root_sequence[1].number_root_sequence, 2);
}
