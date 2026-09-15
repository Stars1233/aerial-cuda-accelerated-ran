/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "aerial_fh_rmax_tests.hpp"

using namespace aerial_fh;

static FronthaulHandle fhi{};
static NicHandle       nic{};
static PeerHandle      peer_rx_rmax_1{};
static PeerHandle      peer_rx_rmax_2{};
static StreamRxHandle  stream_rx_1{};
static StreamRxHandle  stream_rx_2{};
static FlowHandle      cplane_flow{};
static FlowHandle      uplane_flow{};
static FlowHandle      uplane_flow_rx_mode_rmax{};


static void user_cb(void* addr, void* opaque) {}

TEST(RmaxTest, Open)
{
    FronthaulInfo fronthaul_info{2, 500, 1, false, false, "unit_tests", {0}, true};
    fronthaul_info.max_dl_antenna_ports = static_cast<uint16_t>(kMaxFlows);
    EXPECT_EQ(0, open(&fronthaul_info, &fhi));
    EXPECT_LE(0, rte_mbuf_dynflag_lookup(RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME, nullptr));
    EXPECT_LE(0, rte_mbuf_dynfield_lookup(RTE_MBUF_DYNFIELD_TIMESTAMP_NAME, nullptr));
    EXPECT_EQ(2, rte_get_main_lcore());
}

TEST(RmaxTest, AddNicRemoveNic)
{
    NicInfo nic_info{get_nic_name(), UT_MTU, 1 << 16, 32, 12, 10, 256, 256, 0};
    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic));
}

TEST(RmaxTest, AddPeerError)
{
    static PeerHandle      peer_rx_rmax_err{};

    PeerInfo peer_info{
        41,
        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66},
        {0x22, 0x04, 0x9b, 0x9e, 0x27, 0xa3},
        {16, UserDataCompressionMethod::NO_COMPRESSION},
        1,
        RxApiMode::TXONLY,
        true,
    };
    
    EXPECT_NE(0, add_peer(nic, &peer_info, &peer_rx_rmax_err));
}


TEST(RmaxTest, AddPeer)
{
    PeerInfo peer_info_1{
        41,
        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66},
        {0x22, 0x04, 0x9b, 0x9e, 0x27, 0xa3},
        {16, UserDataCompressionMethod::NO_COMPRESSION},
        1,
        RxApiMode::TXONLY,
        true,
    };
    
    PeerInfo peer_info_2{
        42,
        {0x13, 0x23, 0x33, 0x43, 0x53, 0x63},
        {0x23, 0x03, 0x93, 0x93, 0x23, 0xa3},
        {16, UserDataCompressionMethod::NO_COMPRESSION},
        1,
        RxApiMode::TXONLY,
        true,
    };

    for(int i = 0; i < 20; i++)
    {
        EXPECT_EQ(0, add_peer(nic, &peer_info_1, &peer_rx_rmax_1));
        EXPECT_EQ(0, add_peer(nic, &peer_info_2, &peer_rx_rmax_2));
        EXPECT_EQ(0, remove_peer(peer_rx_rmax_1));
        EXPECT_EQ(0, remove_peer(peer_rx_rmax_2));
    }
    EXPECT_EQ(0, add_peer(nic, &peer_info_1, &peer_rx_rmax_1));
    EXPECT_EQ(0, add_peer(nic, &peer_info_2, &peer_rx_rmax_2));
}

TEST(RmaxTest, AddStream)
{
    StreamRxInfo stream_rx_info_1{
        2,
        {4, 5, 6, 7},
        1024,
        3822 // 273 x 14
    };

     StreamRxInfo stream_rx_info_2{
        2,
        {6, 7, 8, 9},
        1024,
        3822 // 273 x 14
    };

    for(int i = 0; i < 20; i++)
    {
        EXPECT_EQ(0, add_stream_rx(peer_rx_rmax_1, &stream_rx_info_1, &stream_rx_1));
        EXPECT_EQ(0, add_stream_rx(peer_rx_rmax_2, &stream_rx_info_2, &stream_rx_2));
        EXPECT_EQ(0, remove_stream_rx(stream_rx_1));
        EXPECT_EQ(0, remove_stream_rx(stream_rx_2));
    }

    EXPECT_EQ(0, add_stream_rx(peer_rx_rmax_1, &stream_rx_info_1, &stream_rx_1));
    EXPECT_EQ(0, add_stream_rx(peer_rx_rmax_2, &stream_rx_info_2, &stream_rx_2));
}

TEST(RmaxTest, GetStreamSlots)
{
    std::vector<StreamRxSlotInfo> slot_info_1;
    std::vector<StreamRxSlotInfo> slot_info_2;
    #define SLOT_PER_STREAM 4

    EXPECT_EQ(0, get_stream_rx_slots(stream_rx_1, slot_info_1));
    EXPECT_EQ(SLOT_PER_STREAM, slot_info_1.size());

    for (auto s : slot_info_1) {
        EXPECT_NE(nullptr, s.addr);
        EXPECT_NE(0, s.flow_stride);
        EXPECT_NE(0, s.eAxC_num);
    }

    EXPECT_EQ(0, get_stream_rx_slots(stream_rx_2, slot_info_2));
    EXPECT_EQ(SLOT_PER_STREAM, slot_info_2.size());

    for (auto s : slot_info_2) {
        EXPECT_NE(nullptr, s.addr);
        EXPECT_NE(0, s.flow_stride);
        EXPECT_NE(0, s.eAxC_num);
    }
}

TEST(RmaxTest, GetRxBytes)
{
    uint64_t rx_bytes = 0;

    for(int i = 0; i < 50; i++) {
        EXPECT_EQ(0, get_stream_rx_bytes(stream_rx_1, &rx_bytes));
        EXPECT_EQ(0, rx_bytes);
        EXPECT_EQ(0, get_stream_rx_bytes(stream_rx_2, &rx_bytes));
        EXPECT_EQ(0, rx_bytes);
    }
}

TEST(RmaxTest, Close)
{
    EXPECT_EQ(0, remove_stream_rx(stream_rx_1));
    EXPECT_EQ(0, remove_stream_rx(stream_rx_2));

    EXPECT_EQ(0, remove_peer(peer_rx_rmax_1));
    EXPECT_EQ(0, remove_peer(peer_rx_rmax_2));

    EXPECT_EQ(0, close(fhi));
}
