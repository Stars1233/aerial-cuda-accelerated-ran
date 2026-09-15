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

#include "aerial_fh_tests.hpp"

using namespace aerial_fh;

static FronthaulHandle fhi{};
static NicHandle       nic{};
static PeerHandle      peer_rx_peer{};
static PeerHandle      peer_rx_flow{};
static PeerHandle      peer_rx_hybrid{};
static PeerHandle      peer_no_cplane_tx{};
static FlowHandle      cplane_flow{};
static FlowHandle      uplane_flow{};
static FlowHandle      uplane_flow_rx_mode_peer{};
static FlowHandle      uplane_flow_rx_mode_hybrid{};
static FlowHandle      cplane_flow_rx_mode_hybrid{};

static void user_cb(void* addr, void* opaque) {}

TEST(UnitTest, Iterator)
{
    Iterator<int>        iter;
    std::unique_ptr<int> u_ptr{std::make_unique<int>(3)};
    iter.add(3);
    auto a = iter.next();
    EXPECT_EQ(a, 3);
    auto b = iter.next();
    EXPECT_EQ(b, 3);

    Iterator<int> iter2;
    iter2.add(1);
    iter2.add(2);
    a = iter2.next();
    EXPECT_EQ(a, 1);
    a = iter2.next();
    EXPECT_EQ(a, 2);
    a = iter2.next();
    EXPECT_EQ(a, 1);
}

TEST(UnitTest, PrbSize)
{
    EXPECT_EQ(3, get_prb_size(1, UserDataCompressionMethod::NO_COMPRESSION));
    EXPECT_EQ(43, get_prb_size(14, UserDataCompressionMethod::BLOCK_FLOATING_POINT));
    EXPECT_EQ(41, get_prb_size(13, UserDataCompressionMethod::MOD_COMPR_SELECTIVE_RE_SENDING));
    EXPECT_EQ(48, get_prb_size(16, UserDataCompressionMethod::NO_COMPRESSION));
    EXPECT_THROW(get_prb_size(0, UserDataCompressionMethod::NO_COMPRESSION), FronthaulException);
    EXPECT_THROW(get_prb_size(17, UserDataCompressionMethod::NO_COMPRESSION), FronthaulException);
}

TEST(UnitTest, EcpriSequenceIdGenerator)
{
    SequenceIdGenerator generator;

    for(uint16_t i = 0; i < 260; i++)
    {
        ASSERT_EQ(static_cast<uint8_t>(i), generator.next());
    }
}

TEST(UnitTest, CPlaneCommonHdrSize)
{
    EXPECT_EQ(12, get_cmsg_common_hdr_size(0));
    EXPECT_EQ(8, get_cmsg_common_hdr_size(1));
    EXPECT_EQ(0, get_cmsg_common_hdr_size(2));
    EXPECT_EQ(12, get_cmsg_common_hdr_size(3));
    EXPECT_EQ(0, get_cmsg_common_hdr_size(4));
    EXPECT_EQ(8, get_cmsg_common_hdr_size(5));
}

TEST(UnitTest, CPlaneSectionSize)
{
    EXPECT_EQ(8, get_cmsg_section_size(0));
    EXPECT_EQ(8, get_cmsg_section_size(1));
    EXPECT_EQ(0, get_cmsg_section_size(2));
    EXPECT_EQ(12, get_cmsg_section_size(3));
    EXPECT_EQ(0, get_cmsg_section_size(4));
    EXPECT_EQ(8, get_cmsg_section_size(5));
}

TEST(UnitTest, Bitfield)
{
    struct oran_u_section_uncompressed section = {};
    size_t                             test    = 0;

    section.symInc = 1;
    test           = section.symInc;
    EXPECT_EQ(1, test);
    section.startPrbu = 37;
    test              = section.startPrbu;
    EXPECT_EQ(37, test);
    section.numPrbu = 255;
    test            = section.numPrbu;
    EXPECT_EQ(255, test);
    section.numPrbu = 0;
    test            = section.numPrbu;
    EXPECT_EQ(0, test);
}

TEST(ApiTest, OpenInvalidDpdkThread)
{
    FronthaulHandle fhibad{};
    FronthaulInfo   fronthaul_info{1 << 12, 3000, 7, false, false, "unit_tests"};
    EXPECT_NE(0, open(&fronthaul_info, &fhibad));
}

TEST(ApiTest, Open)
{
    FronthaulInfo fronthaul_info{2, 500, 1, false, false, "unit_tests", {0}};
    fronthaul_info.max_dl_antenna_ports = static_cast<uint16_t>(kMaxFlows);
    EXPECT_EQ(0, open(&fronthaul_info, &fhi));
    EXPECT_LE(0, rte_mbuf_dynflag_lookup(RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME, nullptr));
    EXPECT_LE(0, rte_mbuf_dynfield_lookup(RTE_MBUF_DYNFIELD_TIMESTAMP_NAME, nullptr));
    EXPECT_EQ(2, rte_get_main_lcore());
}

TEST(ApiTest, OpenTwice)
{
    FronthaulInfo fronthaul_info{0, 500, 1, false, true, "unit_tests"};
    EXPECT_NE(0, open(&fronthaul_info, &fhi));
}

TEST(ApiTest, OpenWithInvalidAccuTxSchedClockRes)
{
    FronthaulInfo fronthaul_info_bad{2, 499, 0, true, true, "unit_tests"};
    EXPECT_NE(0, open(&fronthaul_info_bad, &fhi));
    fronthaul_info_bad.accu_tx_sched_res_ns = 10001;
    EXPECT_NE(0, open(&fronthaul_info_bad, &fhi));
}

TEST(ApiTest, AddNicWhichDontExist)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{"non_existing_nic", UT_MTU, 1 << 14, 32, 4, 4, 256, 256, 0};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
}

/* Test fails on BF2
TEST(ApiTest, AddNicWithWrongMtu)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), 1, 1 << 14, 32, 4, 4, 256, 256, -1};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}
*/

TEST(ApiTest, AddNicWithNoCpuMbufs)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), 500, 0, 32, 4, 4, 256, 256, 0};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithNoTxRequest)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), UT_MTU, 1 << 16, 0, 1, 1, 1024, 4096, 0};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithNoTxq)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), 9000, 1 << 17, 16, 0, 0, 2, 8192, 8192, 0}; // 1 << 14 problematic
    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithGpuTxqOnly)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), 9000, 1 << 17, 16, 0, 1, 2, 8192, 8192, 0};
    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithNoRxq)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), UT_MTU, 1 << 17, 16, 1, 1, 1, 512, 512, 0};
    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithEmptyTxq)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), UT_MTU, 1 << 10, 16, 1, 1, 1, 0, 256, -1};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWithEmptyRxq)
{
    NicHandle nic_bad{};
    NicInfo   nic_info{get_nic_name(), UT_MTU, 1 << 10, 16, 4, 4, 4, 256, 0, 0};
    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic_bad));
    EXPECT_EQ(0, remove_nic(nic_bad));
}

TEST(ApiTest, AddNicWrongGpu)
{
    NicInfo nic_info{get_nic_name(), UT_MTU, 1 << 16, 32, 12, 12, 10, 256, 256, -1};

    EXPECT_NE(0, add_nic(fhi, &nic_info, &nic));
    EXPECT_EQ(0, remove_nic(nic));
}

TEST(ApiTest, AddNicGpuOnly)
{
    NicInfo nic_info{get_nic_name(), UT_MTU, 1 << 16, 32, 0, 12, 10, 256, 256, 0};

    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic));
    EXPECT_EQ(0, remove_nic(nic));
}

TEST(ApiTest, AddNicRemoveNic)
{
    NicInfo nic_info{get_nic_name(), UT_MTU, 1 << 16, 32, 12, 12, 10, 256, 256, 0};

    for(int i = 0; i < 5; i++)
    {
        EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic));
        EXPECT_EQ(0, remove_nic(nic));
    }
    EXPECT_EQ(0, add_nic(fhi, &nic_info, &nic));
}

TEST(ApiTest, AddPeerWithCustomSrcMac)
{
    PeerInfo peer_info{
        41,
        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66},
        {0x22, 0x04, 0x9b, 0x9e, 0x27, 0xa3},
        1,
        {16, UserDataCompressionMethod::NO_COMPRESSION},
        1, 1,
        RxApiMode::FLOW,
        true,
    };
    for(int i = 0; i < 20; i++)
    {
        EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_rx_flow));
        EXPECT_EQ(0, remove_peer(peer_rx_flow));
    }
    EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_rx_flow));
}

TEST(ApiTest, AddPeerWithHybridMode)
{
    PeerInfo peer_info{
        3,
        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66},
        {0x22, 0x04, 0x9b, 0x9e, 0x27, 0xa3},
        1,
        {9, UserDataCompressionMethod::NO_COMPRESSION},
        1, 1,
        RxApiMode::HYBRID,
        true,
    };
    for(int i = 0; i < 2; i++)
    {
        EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_rx_hybrid));
        EXPECT_EQ(0, remove_peer(peer_rx_hybrid));
    }
    EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_rx_hybrid));
}

TEST(ApiTest, AddPeerWithTxOnlyMode)
{
    PeerHandle peer_local;
    PeerInfo   peer_info{
        666,
        {0x11, 0x22, 0x33, 0x44, 0x55, 0x66},
        {0x22, 0x04, 0x9b, 0x9e, 0x27, 0xa3},
        1,
        {1, UserDataCompressionMethod::NO_COMPRESSION},
        1, 1,
        RxApiMode::TXONLY,
        true,
    };

    EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_local));
    EXPECT_EQ(0, remove_peer(peer_local));
}

TEST(ApiTest, AddPeerWithDefaultSrcMac)
{
    PeerInfo peer_info{
        41,
        {0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
        {0x0c, 0x42, 0xa1, 0xd1, 0xd1, 0xc0},
        1,
        {14, UserDataCompressionMethod::U_LAW},
        2, 2,
        RxApiMode::PEER,
        true,
    };

    ASSERT_EQ(0, add_peer(nic, &peer_info, &peer_rx_peer));

    auto peer_handle    = static_cast<Peer*>(peer_rx_peer);
    auto peer_info_init = peer_handle->get_info();
    auto peer_mac       = peer_info_init.src_mac_addr.bytes;

    Nic* nic_handle  = static_cast<Nic*>(nic);
    auto nic_port_id = nic_handle->get_port_id();

    rte_ether_addr dev_mac;
    ASSERT_EQ(0, rte_eth_macaddr_get(nic_port_id, &dev_mac));
    EXPECT_EQ(0, memcmp(dev_mac.addr_bytes, peer_mac, 6));

    EXPECT_EQ(0, remove_peer(peer_rx_peer));
    EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_rx_peer));
}

TEST(ApiTest, AddPeerWithNoUplaneTxqs)
{
    PeerHandle peer_bad{};
    PeerInfo   peer_info{
        41,
        {0x11, 0x23, 0x33, 0x74, 0x55, 0x66},
        {0x27, 0x14, 0x9b, 0x5e, 0x27, 0xa3},
        1,
        {1, UserDataCompressionMethod::U_LAW},
        0, 0,
        RxApiMode::FLOW,
        true,
    };
    EXPECT_NE(0, add_peer(nic, &peer_info, &peer_bad));
    EXPECT_EQ(0, remove_peer(peer_bad));
}

TEST(ApiTest, AddPeerWithTooManyUplaneTxqs)
{
    PeerHandle peer_bad{};
    PeerInfo   peer_info{
        4,
        {0x11, 0x23, 0x33, 0x74, 0x55, 0x66},
        {0x27, 0x14, 0x9b, 0x5e, 0x29, 0xa3},
        1,
        {4, UserDataCompressionMethod::U_LAW},
        30, 1,
        RxApiMode::FLOW,
        true,
    };
    EXPECT_NE(0, add_peer(nic, &peer_info, &peer_bad));
    EXPECT_EQ(0, remove_peer(peer_bad));
}

TEST(ApiTest, AddCplaneFlowWithRxApiPeer)
{
    FlowInfo flow_info{8, FlowType::CPLANE};
    for(int i = 0; i < 13; i++)
    {
        EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &cplane_flow));
        EXPECT_EQ(0, remove_flow(cplane_flow));
    }
    EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &cplane_flow));
}

TEST(ApiTest, AddUplaneFlowWithRxApiFlow)
{
    FlowInfo flow_info{1, FlowType::UPLANE};
    EXPECT_EQ(0, add_flow(peer_rx_flow, &flow_info, &uplane_flow));
    for(int i = 0; i < 2; i++)
    {
        EXPECT_EQ(0, remove_flow(uplane_flow));
        EXPECT_EQ(0, add_flow(peer_rx_flow, &flow_info, &uplane_flow));
    }
}

TEST(ApiTest, AddUplaneFlowWithRxApiPeer)
{
    FlowInfo flow_info{7, FlowType::UPLANE};
    EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &uplane_flow_rx_mode_peer));
    for(int i = 0; i < 3; i++)
    {
        EXPECT_EQ(0, remove_flow(uplane_flow_rx_mode_peer));
        EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &uplane_flow_rx_mode_peer));
    }
}

TEST(ApiTest, AddUplaneFlowWithRxApiHybrid)
{
    FlowInfo flow_info{75, FlowType::UPLANE};
    EXPECT_EQ(0, add_flow(peer_rx_hybrid, &flow_info, &uplane_flow_rx_mode_hybrid));
    for(int i = 0; i < 3; i++)
    {
        EXPECT_EQ(0, remove_flow(uplane_flow_rx_mode_hybrid));
        EXPECT_EQ(0, add_flow(peer_rx_hybrid, &flow_info, &uplane_flow_rx_mode_hybrid));
    }
}

TEST(ApiTest, AddCplaneFlowWithRxApiHybrid)
{
    FlowInfo flow_info{2312, FlowType::CPLANE};
    for(int i = 0; i < 4; i++)
    {
        EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &cplane_flow_rx_mode_hybrid));
        EXPECT_EQ(0, remove_flow(cplane_flow_rx_mode_hybrid));
    }
    EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info, &cplane_flow_rx_mode_hybrid));
}

static Ns now_ns()
{
    timespec t;
    if(unlikely(clock_gettime(CLOCK_REALTIME, &t) != 0))
    {
        THROW_FH(EIO, StringBuilder() << "Could not access timer");
    }
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

TEST(ApiTest, TxqUplaneGpuCommRequest)
{
    UPlaneMsgSendInfo uplane_msg_info[4];
    std::vector<TxRequestGpuCommHandle> tx_handle_v;
    TxRequestGpuCommHandle output_handle;
    auto prb_size = 48;
    auto symbol_size = prb_size * ORAN_MAX_PRB_X_SLOT;
    uint32_t * send_wait_flag;

    FlowInfo flow_info_1{7, FlowType::UPLANE, 2};
    FlowInfo flow_info_2{8, FlowType::UPLANE, 2};
    FlowHandle flow_handle_1{};
    FlowHandle flow_handle_2{};

    EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info_1, &flow_handle_1));
    EXPECT_EQ(0, add_flow(peer_rx_peer, &flow_info_2, &flow_handle_2));

    MemRegHandle memreg{};
    MemRegInfo   memreg_info{nullptr, symbol_size * ORAN_ALL_SYMBOLS, kNvGpuPageSize};
    uint16_t gpu_id = 0;
    memreg_info.addr = rte_gpu_mem_alloc(gpu_id, memreg_info.len, kNvGpuPageSize);
    ASSERT_NE(nullptr, memreg_info.addr);
    EXPECT_EQ(0, register_memory(fhi, &memreg_info, &memreg));

    for (int idx = 0; idx < 4; idx++)
    {
        memset(&uplane_msg_info[idx], 0, sizeof(uplane_msg_info[idx]));

        auto& radio_app_hdr          = uplane_msg_info[idx].radio_app_hdr;
        radio_app_hdr.dataDirection  = DIRECTION_DOWNLINK;
        radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
        radio_app_hdr.filterIndex    = ORAN_DEF_FILTER_INDEX;
        radio_app_hdr.frameId    = 1;
        radio_app_hdr.subframeId = 3;
        radio_app_hdr.slotId     = 0;
        radio_app_hdr.symbolId   = idx;

        auto& section_hdr   = uplane_msg_info[idx].section_info;
        section_hdr.rb      = ORAN_RB_ALL;
        section_hdr.sym_inc = ORAN_SYMCINC_NO;

        if(idx % 2 == 0)
            uplane_msg_info[idx].flow = flow_handle_1;
        else
            uplane_msg_info[idx].flow = flow_handle_2;

        uplane_msg_info[idx].tx_window.tx_window_start = now_ns() + (3000000 * (idx+1)); //50ms per symbol

        uplane_msg_info[idx].section_info.section_id     = 4;
        uplane_msg_info[idx].section_info.start_prbu     = 0;
        uplane_msg_info[idx].section_info.num_prbu       = 273;
        uplane_msg_info[idx].section_info.iq_data_buffer = static_cast<uint8_t*>(memreg_info.addr) + (idx * symbol_size);
    }
    
    cudaMallocHost((void**)&send_wait_flag, sizeof(uint32_t));
    *send_wait_flag = 0;
    PreparePRBInfo prb_info;

    EXPECT_EQ(0, prepare_uplane_gpu_comm(peer_rx_peer, uplane_msg_info, 4, &output_handle, 0, 0));
    EXPECT_EQ(0, set_gpu_request_cell_idx(output_handle, 0));
    tx_handle_v.push_back(output_handle);
    EXPECT_EQ(0, send_uplane_gpu_comm(nic, tx_handle_v, prb_info));

    // Unblock packet send
    sleep(1);
    *send_wait_flag = 1;
    // Wait packet transmission
    sleep(1);
    cudaFreeHost(send_wait_flag);
    EXPECT_EQ(0, remove_flow(flow_handle_1));
    EXPECT_EQ(0, remove_flow(flow_handle_2));
    EXPECT_EQ(0, unregister_memory(memreg));
}

TEST(ApiTest, RingBuffer)
{
    RingBufferHandle ring{};
    RingBufferInfo   ring_info0{"", 3, AERIAL_SOCKET_ID_ANY, false, false};
    RingBufferInfo   ring_info1{"empty_ring", 0, AERIAL_SOCKET_ID_ANY, false, false};
    RingBufferInfo   ring_info2{"ring", 3, AERIAL_SOCKET_ID_ANY, true, true};
    RingBufferInfo   ring_info3{"ring", 3, AERIAL_SOCKET_ID_ANY, true, true};

    int  a     = 1;
    int* a_ptr = &a;
    int  b[4]{2, 3, 4, 5};
    int* b_ptr[4]{};

    for(int i = 0; i < 4; i++)
    {
        b_ptr[i] = &b[i];
    }

    EXPECT_EQ(0, ring_create(fhi, &ring_info0, &ring));
    EXPECT_NE(0, ring_create(fhi, &ring_info0, &ring));
    EXPECT_EQ(0, ring_destroy(ring));
    EXPECT_EQ(0, ring_create(fhi, &ring_info1, &ring));
    EXPECT_NE(0, ring_create(fhi, &ring_info1, &ring));
    EXPECT_NE(0, ring_create(fhi, &ring_info2, &ring));
    ASSERT_EQ(0, ring_create(fhi, &ring_info3, &ring));
    ASSERT_NE(nullptr, ring);

    EXPECT_TRUE(ring_empty(ring));
    EXPECT_EQ(3, ring_free_count(ring));
    EXPECT_EQ(0, ring_enqueue(ring, (void*)a_ptr));
    EXPECT_FALSE(ring_empty(ring));
    EXPECT_FALSE(ring_full(ring));
    a_ptr = nullptr;
    EXPECT_EQ(0, ring_dequeue(ring, reinterpret_cast<void**>(&a_ptr)));
    EXPECT_EQ(1, *a_ptr);
    EXPECT_NE(0, ring_dequeue(ring, reinterpret_cast<void**>(&a_ptr)));

    EXPECT_EQ(0, ring_enqueue_bulk(ring, reinterpret_cast<void* const*>(&b_ptr[0]), 4));
    EXPECT_EQ(3, ring_enqueue_burst(ring, reinterpret_cast<void* const*>(&b_ptr[0]), 3));
    EXPECT_TRUE(ring_full(ring));
    EXPECT_EQ(3, ring_dequeue_burst(ring, reinterpret_cast<void**>(&b_ptr[0]), 4));
    EXPECT_EQ(0, ring_dequeue_bulk(ring, reinterpret_cast<void**>(&b_ptr[0]), 3));
    EXPECT_EQ(2, *b_ptr[0]);
    EXPECT_EQ(4, *b_ptr[2]);

    RingBufferInfo ring_info4{"sp_mc", 1000, 0, false, true};
    RingBufferInfo ring_info5{"mp_sc", 1, 0, true, false};
    RingBufferInfo ring_info6{"sp_sc", 17, 0, false, false};
    RingBufferInfo ring_info7{"this_name_will_definitely_be_too_long", 17, 0, false, false};

    EXPECT_EQ(0, ring_create(fhi, &ring_info4, &ring));
    EXPECT_EQ(1023, ring_free_count(ring));
    EXPECT_EQ(0, ring_create(fhi, &ring_info5, &ring));
    EXPECT_EQ(0, ring_create(fhi, &ring_info6, &ring));
    EXPECT_NE(0, ring_create(fhi, &ring_info7, &ring));
    EXPECT_EQ(0, ring_destroy(ring));
}

TEST(ApiTest, MemReg)
{
    MemRegHandle memreg{};
    MemRegInfo   memreg_info{nullptr, 8, kNvGpuPageSize};
    uint16_t     gpu_id = 0;
    memreg_info.addr    = rte_gpu_mem_alloc(gpu_id, memreg_info.len, kNvGpuPageSize);
    ASSERT_NE(nullptr, memreg_info.addr);
    EXPECT_EQ(0, register_memory(fhi, &memreg_info, &memreg));
    EXPECT_NE(nullptr, rte_mem_virt2memseg(memreg_info.addr, nullptr));
    EXPECT_EQ(0, unregister_memory(memreg));
    rte_gpu_mem_free(gpu_id, memreg_info.addr);

    auto page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    MemRegHandle memreg_cpu{};
    MemRegInfo   memreg_info_cpu{nullptr, 8, static_cast<size_t>(page_size)};
    memreg_info_cpu.addr = aligned_alloc(memreg_info_cpu.page_sz, memreg_info_cpu.len);
    EXPECT_EQ(0, register_memory(fhi, &memreg_info_cpu, &memreg_cpu));
    EXPECT_EQ(0, unregister_memory(memreg_cpu));
    free(memreg_info_cpu.addr);
}

TEST(ApiTest, PeerTxqUplaneRequest)
{
    constexpr size_t                        kTxqRequestCount = 32;
    std::array<TxqHandle, kTxqRequestCount> txqs;
    size_t                                  num_txqs = kTxqRequestCount;
    EXPECT_EQ(0, get_uplane_txqs(peer_rx_peer, txqs.data(), &num_txqs));
    ASSERT_EQ(2, num_txqs);

    Txq* txq_handle = static_cast<Txq*>(txqs[0]);
    auto txq_id     = txq_handle->get_id();
    EXPECT_EQ(5, txq_id);

    UPlaneTxCompleteNotification notification{};
    EXPECT_EQ(0, prepare_and_send_uplane(peer_rx_peer, nullptr, 0, notification, txqs[0]));
}

TEST(ApiTest, HugePageAlloc)
{
    auto ptr = allocate_memory(1, 0);
    EXPECT_NE(nullptr, ptr);
    EXPECT_EQ(0, free_memory(ptr));

    ptr = allocate_memory(1 << 10, 64);
    EXPECT_NE(nullptr, ptr);
    EXPECT_EQ(0, free_memory(ptr));

    ptr = allocate_memory(1 << 14, 1 << 14);
    EXPECT_NE(nullptr, ptr);
    EXPECT_EQ(0, free_memory(ptr));
}

TEST(ApiTest, NicMetricsUpdate)
{
    EXPECT_EQ(0, update_metrics(fhi));
}

TEST(ApiTest, PrintAndResetNicStats)
{
    EXPECT_EQ(0, print_stats(nic));
    EXPECT_EQ(0, reset_stats(nic));
    EXPECT_EQ(0, print_stats(nic, true));
}

TEST(ApiTest, PollTxComplete)
{
    EXPECT_EQ(0, poll_tx_complete(peer_rx_peer));
    EXPECT_EQ(0, poll_tx_complete(peer_rx_flow));
}

TEST(ApiTest, SendCplaneZeroMsgs)
{
    EXPECT_EQ(0, send_cplane(peer_rx_peer, nullptr, 0));
    EXPECT_EQ(0, send_cplane(peer_rx_flow, nullptr, 0));
}

TEST(ApiTest, SendUplaneZeroMsgs)
{
    TxRequestHandle              tx_request{};
    UPlaneTxCompleteNotification notification{user_cb, nullptr};

    EXPECT_EQ(0, prepare_uplane(peer_rx_peer, nullptr, 0, notification, &tx_request));
    EXPECT_EQ(0, send_uplane(tx_request));
    EXPECT_EQ(0, prepare_uplane(peer_rx_flow, nullptr, 0, notification, &tx_request));
    EXPECT_EQ(0, send_uplane(tx_request));
}

TEST(ApiTest, FreeRxMessages)
{
    constexpr size_t kNumMsgs = 5;
    MsgReceiveInfo   info[kNumMsgs]{};

    struct rte_mempool* mp = rte_pktmbuf_pool_create("free_uplane_test_pool", kNumMsgs, 0, 0, rte_align32pow2(UT_MTU), 0);
    ASSERT_NE(nullptr, mp);

    for(size_t i = 0; i < kNumMsgs; i++)
    {
        void* mbuf = rte_pktmbuf_alloc(mp);
        EXPECT_NE(nullptr, mbuf);
        info[i].opaque = mbuf;
    }

    EXPECT_EQ(0, free_rx_messages(&info[0], kNumMsgs));
}

TEST(ApiTest, FreeZeroRxMessages)
{
    EXPECT_EQ(0, free_rx_messages(nullptr, 0));
}

TEST(ApiTest, Receive)
{
    constexpr size_t kNumMsgs = 2;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    EXPECT_EQ(0, receive(peer_rx_peer, &info[0], &num_msgs));
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveZeroMsgs)
{
    size_t num_msgs = 0;
    EXPECT_EQ(0, receive(peer_rx_peer, nullptr, &num_msgs));
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, UpdatePeer)
{
    MacAddr dst_mac_addr{{0x55, 0x66, 0x65, 0x43, 0x21, 0x00}};
    EXPECT_EQ(0, update_peer(peer_rx_peer, dst_mac_addr));
}

TEST(ApiTest, UpdateFlow)
{
    FlowInfo flow_info{214, FlowType::UPLANE, 0x3333};
    EXPECT_EQ(0, update_flow(uplane_flow, &flow_info));
}

TEST(ApiTest, ReceiveWrongRxMode)
{
    constexpr size_t kNumMsgs = 2;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;
    EXPECT_NE(0, receive(peer_rx_flow, &info[0], &num_msgs));
}

TEST(ApiTest, ReceiveUntil)
{
    constexpr size_t kNumMsgs = 1024;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    Ns current_ns       = get_ns();
    Ns delay_ns         = 1000 * 1000;
    Ns receive_until_ns = current_ns + delay_ns;
    EXPECT_EQ(0, receive_until(peer_rx_peer, &info[0], &num_msgs, receive_until_ns));
    Ns time_spent_receiving_ns = get_ns() - current_ns;
    EXPECT_GT(time_spent_receiving_ns, delay_ns);
    EXPECT_LT(time_spent_receiving_ns, delay_ns + 4000);
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveUntilWrongRxMode)
{
    constexpr size_t kNumMsgs = 1024;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    Ns current_ns       = get_ns();
    Ns delay_ns         = 1000 * 1000;
    Ns receive_until_ns = current_ns + delay_ns;
    EXPECT_NE(0, receive_until(peer_rx_flow, &info[0], &num_msgs, receive_until_ns));
}

TEST(ApiTest, ReceiveUntilPast)
{
    constexpr size_t kNumMsgs = 10;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    Ns current_ns = get_ns();
    EXPECT_EQ(0, receive_until(peer_rx_peer, &info[0], &num_msgs, current_ns));
    Ns time_spent_receiving_ns = get_ns() - current_ns;
    EXPECT_LT(time_spent_receiving_ns, 5000);
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveFlow)
{
    constexpr size_t kNumMsgs = 100;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    EXPECT_EQ(0, receive_flow(uplane_flow, &info[0], &num_msgs));
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveFlowZeroMsgs)
{
    size_t num_msgs = 0;
    EXPECT_EQ(0, receive_flow(uplane_flow, nullptr, &num_msgs));
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveFlowWrongRxMode)
{
    constexpr size_t kNumMsgs = 100;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;
    EXPECT_NE(0, receive_flow(uplane_flow_rx_mode_peer, &info[0], &num_msgs));
}

TEST(ApiTest, ReceiveFlowUntil)
{
    constexpr size_t kNumMsgs = 2;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    Ns current_ns       = get_ns();
    Ns delay_ns         = 5 * 1000 * 1000;
    Ns receive_until_ns = current_ns + delay_ns;
    EXPECT_EQ(0, receive_flow_until(uplane_flow, &info[0], &num_msgs, receive_until_ns));
    Ns time_spent_receiving_ns = get_ns() - current_ns;
    EXPECT_GT(time_spent_receiving_ns, delay_ns);
    EXPECT_LT(time_spent_receiving_ns, delay_ns + 4000);
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, ReceiveFlowUntilWrongRxMode)
{
    constexpr size_t kNumMsgs = 743;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    Ns current_ns       = get_ns();
    Ns delay_ns         = 5 * 1000 * 1000;
    Ns receive_until_ns = current_ns + delay_ns;
    EXPECT_NE(0, receive_flow_until(uplane_flow_rx_mode_peer, &info[0], &num_msgs, receive_until_ns));
}

TEST(ApiTest, ReceiveRxModeHybrid)
{
    constexpr size_t kNumMsgs = 12;
    MsgReceiveInfo   info[kNumMsgs]{};
    size_t           num_msgs = kNumMsgs;

    EXPECT_EQ(0, receive_flow(uplane_flow_rx_mode_hybrid, &info[0], &num_msgs));
    EXPECT_EQ(0, num_msgs);

    num_msgs = kNumMsgs;
    EXPECT_NE(0, receive_flow(cplane_flow_rx_mode_hybrid, &info[0], &num_msgs));

    num_msgs = kNumMsgs;
    EXPECT_EQ(0, receive(peer_rx_hybrid, &info[0], &num_msgs));
    EXPECT_EQ(0, num_msgs);
}

TEST(ApiTest, AddPeerWithNoCplane)
{
    PeerInfo peer_info{
        41,
        {0x11, 0x33, 0x33, 0x33, 0x55, 0x66},
        {0x11, 0x14, 0x7e, 0x5e, 0x27, 0xa3},
        1,
        {1, UserDataCompressionMethod::U_LAW},
        1, 1,
        RxApiMode::FLOW,
        false,
    };

    for(int i = 0; i < 3; i++)
    {
        EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_no_cplane_tx));
        EXPECT_EQ(0, remove_peer(peer_no_cplane_tx));
    }

    EXPECT_EQ(0, add_peer(nic, &peer_info, &peer_no_cplane_tx));
}

TEST(ApiTest, SendCplaneNoTxq)
{
    EXPECT_EQ(0, send_cplane(peer_no_cplane_tx, nullptr, 4));
}

TEST(ApiTest, Close)
{
    EXPECT_EQ(0, close(fhi));
}
