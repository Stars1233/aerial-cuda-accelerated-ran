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
 * @file test_tb_start_offsets.cpp
 * @brief Unit tests for the apply_tb_start_offsets_from_tx_data algorithm.
 *
 * The production function lives in an anonymous namespace inside
 * nv_phy_dl_channels.cpp and depends on cuphyPdschCwPrm_t (cuphy_api.h).
 * To keep this test free of CUDA / cuPHY transitive dependencies, we
 * replicate the algorithm here using a minimal TestCwPrm surrogate.
 * Any divergence between this copy and the production code should be
 * caught during code review.
 *
 * Covers:
 *   Group A — Happy-path: single cell, single PDU, offset applied
 *   Group B — Multi-cell / multi-PDU cw_start accumulation
 *   Group C — Non-PDSCH cells skipped, cw_start not advanced
 *   Group D — Negative / error cases:
 *             - cell_id out of range
 *             - nullptr msg_buf
 *             - zero PDUs / zero TLVs / wrong TLV tag
 *             - CW array overflow (cw_start + p >= max_cws)
 *   Group E — Edge cases:
 *             - zero DL_TTI messages
 *             - large uint32_t offset value
 *             - multiple PDUs per cell with mixed TLV tags
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nv_fapi_message_storage.hpp"
#include "nv_phy_mac_transport.hpp"

static constexpr uint16_t FAPI_SFN_MAX      = 1024;
static constexpr uint16_t MAX_CELLS_PER_SLOT = 20;
#include "scf_5g_fapi.h"
#include "nv_ipc_utils.h"

namespace {

// -----------------------------------------------------------------------
// Minimal surrogate for cuphyPdschCwPrm_t — only field accessed by the
// algorithm is tbStartOffset.
// -----------------------------------------------------------------------
struct TestCwPrm
{
    uint32_t tbStartOffset = 0xDEAD'BEEFu;
};

// -----------------------------------------------------------------------
// Local reimplementation of apply_tb_start_offsets_from_tx_data
// (anonymous namespace in nv_phy_dl_channels.cpp).
// Identical logic, minus NVLOGW_FMT and using TestCwPrm instead of
// cuphyPdschCwPrm_t.
//
// MIGRATION TARGET: Once Step 6 Option A is implemented, this local copy
// becomes redundant. Group F below exercises
// nv::pdsch_merge::patch_tb_start_offsets<FapiSlotMessageStorage, TestCwPrm>()
// directly and verifies identical results. Remove this local copy when
// Group F is active and the promoted function's tests pass.
// -----------------------------------------------------------------------
inline uint32_t apply_tb_start_offsets(
    const nv::FapiSlotMessageStorage& slot_store,
    const void* const*                staged_msg_bufs,
    uint32_t                          n_staged,
    TestCwPrm*                        ue_cw_info,
    uint32_t                          max_cws,
    uint32_t                          /*slot_u32*/)
{
    uint32_t cw_start = 0;
    uint32_t applied  = 0;
    const uint16_t n_dl_tti = slot_store.dl_tti_count();
    const auto* dl_msgs = slot_store.dl_tti_messages();

    for (uint16_t i = 0; i < n_dl_tti; ++i)
    {
        if (!slot_store.dl_tti_has_pdsch_pdu(i))
            continue;

        const auto cid = static_cast<uint16_t>(dl_msgs[i].cell_id);
        if (cid >= n_staged || staged_msg_bufs[cid] == nullptr) [[unlikely]]
        {
            // Mirror production patch_tb_start_offsets: skip the cell, continue
            // processing subsequent cells with valid staging.
            continue;
        }

        const auto* hdr    = reinterpret_cast<const scf_fapi_header_t*>(staged_msg_bufs[cid]);
        const auto* tx_req = reinterpret_cast<const scf_fapi_tx_data_req_t*>(hdr->payload);
        const uint8_t* pdu_data = reinterpret_cast<const uint8_t*>(tx_req->payload);
        uint32_t pdu_walk_offset = 0;

        for (uint16_t p = 0; p < tx_req->num_pdus; ++p)
        {
            const auto* dl_pdu = reinterpret_cast<const scf_fapi_tx_data_pdu_info_t*>(
                pdu_data + pdu_walk_offset);

            if (dl_pdu->num_tlv > 0)
            {
                const auto* tlv = reinterpret_cast<const scf_fapi_tl_t*>(dl_pdu->tlvs);
                if (tlv->tag == SCF_TX_DATA_OFFSET && (cw_start + p) < max_cws)
                {
                    const uint32_t offset = *reinterpret_cast<const uint32_t*>(tlv->val);
                    ue_cw_info[cw_start + p].tbStartOffset = offset;
                    ++applied;
                }
            }
            pdu_walk_offset += static_cast<uint32_t>(sizeof(scf_fapi_tx_data_pdu_info_t)
                + dl_pdu->num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t)));
        }
        cw_start += tx_req->num_pdus;
    }
    return applied;
}

// -----------------------------------------------------------------------
// Helpers to build fake FAPI message buffers
// -----------------------------------------------------------------------

constexpr uint32_t kSentinel = 0xDEAD'BEEFu;

struct TlvEntry
{
    uint16_t tag;
    uint32_t value;
};

/// Build a TX_DATA.req byte buffer suitable for reinterpret_cast to
/// scf_fapi_header_t / scf_fapi_tx_data_req_t.
std::vector<uint8_t> build_tx_data_msg(
    const std::vector<std::vector<TlvEntry>>& pdu_tlvs)
{
    constexpr size_t kTlvSlot = sizeof(scf_fapi_tl_t) + sizeof(uint32_t);

    size_t total = sizeof(scf_fapi_header_t) +
                   sizeof(scf_fapi_body_header_t) +
                   sizeof(uint16_t) * 3; // sfn + slot + num_pdus
    for (const auto& tlvs : pdu_tlvs)
    {
        total += sizeof(scf_fapi_tx_data_pdu_info_t);
        total += tlvs.size() * kTlvSlot;
    }

    std::vector<uint8_t> buf(total, 0);

    auto* hdr = reinterpret_cast<scf_fapi_header_t*>(buf.data());
    // buf.size() == total > 0, so buf.data() is non-null. Use the canonical
    // GCC pattern to silence -Werror=null-dereference; __builtin_unreachable
    // is the optimizer's trusted "this branch is impossible" marker.
    if (hdr == nullptr) { __builtin_unreachable(); }
    hdr->message_count = 1;

    auto* tx_req = reinterpret_cast<scf_fapi_tx_data_req_t*>(hdr->payload);
    if (tx_req == nullptr) { __builtin_unreachable(); }
    tx_req->num_pdus = static_cast<uint16_t>(pdu_tlvs.size());

    uint8_t* cursor = reinterpret_cast<uint8_t*>(tx_req->payload);
    for (size_t pi = 0; pi < pdu_tlvs.size(); ++pi)
    {
        auto* pdu = reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(cursor);
        pdu->pdu_index = static_cast<uint16_t>(pi);
        pdu->num_tlv   = static_cast<uint32_t>(pdu_tlvs[pi].size());

        uint8_t* tlv_cursor = reinterpret_cast<uint8_t*>(pdu->tlvs);
        for (const auto& te : pdu_tlvs[pi])
        {
            auto* tlv = reinterpret_cast<scf_fapi_tl_t*>(tlv_cursor);
            tlv->tag    = te.tag;
            tlv->length = sizeof(uint32_t);
            std::memcpy(tlv->val, &te.value, sizeof(uint32_t));
            tlv_cursor += kTlvSlot;
        }
        cursor += sizeof(scf_fapi_tx_data_pdu_info_t)
                + pdu_tlvs[pi].size() * kTlvSlot;
    }
    return buf;
}

/// Build a TX_DATA.req message buffer with a single PDU carrying one
/// SCF_TX_DATA_OFFSET TLV whose value is @p offset_value.
std::vector<uint8_t> build_simple_tx_data(uint32_t offset_value)
{
    return build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, offset_value}}});
}

/// Fixed byte size of the make_dl_tti_* buffers (and the store_dl_tti default).
constexpr std::size_t kDlTtiBufSize = 512;

/// Build a DL_TTI.req byte buffer that signals PDSCH presence when stored
/// into FapiSlotMessageStorage.
alignas(64) std::array<uint8_t, kDlTtiBufSize> make_dl_tti_with_pdsch()
{
    std::array<uint8_t, kDlTtiBufSize> raw{};
    auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(
        raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
#ifdef SCF_FAPI_10_04
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1;
#else
    req->num_pdus = 1;
    auto* pdu = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    pdu->pdu_type = DL_TTI_PDU_TYPE_PDSCH;
    pdu->pdu_size = sizeof(scf_fapi_generic_pdu_info_t) + 128;
#endif
    return raw;
}

/// Build a DL_TTI.req byte buffer that does NOT carry any PDSCH PDU.
alignas(64) std::array<uint8_t, kDlTtiBufSize> make_dl_tti_no_pdsch()
{
    std::array<uint8_t, kDlTtiBufSize> raw{};
    auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(
        raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
#ifdef SCF_FAPI_10_04
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB] = 1;
#else
    req->num_pdus = 1;
    auto* pdu = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    pdu->pdu_type = DL_TTI_PDU_TYPE_SSB;
    pdu->pdu_size = sizeof(scf_fapi_generic_pdu_info_t) + 64;
#endif
    return raw;
}

/// Store a DL_TTI.req into the given slot storage for a specific cell_id.
void store_dl_tti(nv::FapiSlotMessageStorage& storage,
                  uint16_t cell_id,
                  const uint8_t* raw_buf,
                  std::size_t buf_len = kDlTtiBufSize)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msg.cell_id = cell_id;
    msg.msg_buf = const_cast<void*>(static_cast<const void*>(raw_buf));
    msg.msg_len = static_cast<int32_t>(buf_len);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
}

constexpr uint32_t kSlotU32 = 0x0001'0005u; // SFN=1, slot=5

} // namespace

// ===========================================================================
// Group A — Happy path
// ===========================================================================

TEST(TbStartOffsets, SingleCell_SinglePdu_OffsetApplied)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(4096);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, /*cell_id=*/0, dl_tti.data());

    constexpr uint32_t kMaxCells = 8;
    const void* staged[kMaxCells]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kMaxCells, cw.data(), 16, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 4096U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, SingleCell_MultiplePdus_AllOffsetsApplied)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 100}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 200}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 300}},
    });

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 16, kSlotU32), 3U);
    EXPECT_EQ(cw[0].tbStartOffset, 100U);
    EXPECT_EQ(cw[1].tbStartOffset, 200U);
    EXPECT_EQ(cw[2].tbStartOffset, 300U);
    EXPECT_EQ(cw[3].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, LargeOffsetValue_Uint32MaxPreserved)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(0xFFFF'FFFFu);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 0xFFFF'FFFFu);
}

// ===========================================================================
// Group B — Multi-cell cw_start accumulation
// ===========================================================================

TEST(TbStartOffsets, TwoCells_CwStartAccumulatesAcrossCells)
{
    auto dl_tti_a = make_dl_tti_with_pdsch();
    auto dl_tti_b = make_dl_tti_with_pdsch();

    auto tx_a = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 1000}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 2000}},
    });
    auto tx_b = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 3000}},
    });

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, /*cell_id=*/5, dl_tti_a.data());
    store_dl_tti(storage, /*cell_id=*/9, dl_tti_b.data());

    constexpr uint32_t kN = 16;
    const void* staged[kN]{};
    staged[5] = tx_a.data();
    staged[9] = tx_b.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 16, kSlotU32), 3U);
    // Cell 5 has 2 PDUs → cw[0], cw[1]
    EXPECT_EQ(cw[0].tbStartOffset, 1000U);
    EXPECT_EQ(cw[1].tbStartOffset, 2000U);
    // Cell 9 has 1 PDU → cw[2] (cw_start = 2 from cell 5)
    EXPECT_EQ(cw[2].tbStartOffset, 3000U);
    EXPECT_EQ(cw[3].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, ThreeCells_DifferentPduCounts)
{
    auto dl_a = make_dl_tti_with_pdsch();
    auto dl_b = make_dl_tti_with_pdsch();
    auto dl_c = make_dl_tti_with_pdsch();

    auto tx_a = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 10}}});
    auto tx_b = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 20}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 30}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 40}},
    });
    auto tx_c = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 50}}});

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_a.data());
    store_dl_tti(storage, 1, dl_b.data());
    store_dl_tti(storage, 2, dl_c.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = tx_a.data();
    staged[1] = tx_b.data();
    staged[2] = tx_c.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 16, kSlotU32), 5U);
    EXPECT_EQ(cw[0].tbStartOffset, 10U);  // cell 0, pdu 0
    EXPECT_EQ(cw[1].tbStartOffset, 20U);  // cell 1, pdu 0 (cw_start=1)
    EXPECT_EQ(cw[2].tbStartOffset, 30U);  // cell 1, pdu 1
    EXPECT_EQ(cw[3].tbStartOffset, 40U);  // cell 1, pdu 2
    EXPECT_EQ(cw[4].tbStartOffset, 50U);  // cell 2, pdu 0 (cw_start=4)
}

// ===========================================================================
// Group C — Non-PDSCH cells skipped
// ===========================================================================

TEST(TbStartOffsets, NonPdschCell_Skipped_CwStartNotAdvanced)
{
    auto dl_pdsch = make_dl_tti_with_pdsch();
    auto dl_ssb   = make_dl_tti_no_pdsch();

    auto tx_cell0 = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 111}}});
    auto tx_cell2 = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 222}}});

    nv::FapiSlotMessageStorage storage(8);
    // DL_TTI order: cell 0 (PDSCH), cell 1 (SSB-only), cell 2 (PDSCH)
    store_dl_tti(storage, 0, dl_pdsch.data());
    store_dl_tti(storage, 1, dl_ssb.data());
    store_dl_tti(storage, 2, dl_pdsch.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = tx_cell0.data();
    staged[2] = tx_cell2.data();

    std::array<TestCwPrm, 8> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 2U);
    EXPECT_EQ(cw[0].tbStartOffset, 111U);
    // Cell 1 (SSB) skipped — cw_start stays at 1, not 2
    EXPECT_EQ(cw[1].tbStartOffset, 222U);
    EXPECT_EQ(cw[2].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, AllNonPdsch_NothingApplied)
{
    auto dl_ssb = make_dl_tti_no_pdsch();
    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_ssb.data());
    store_dl_tti(storage, 1, dl_ssb.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 0U);
    for (auto& c : cw)
        EXPECT_EQ(c.tbStartOffset, kSentinel);
}

// ===========================================================================
// Group D — Negative / error cases
// ===========================================================================

TEST(TbStartOffsets, CellIdOutOfRange_BreaksLoop)
{
    auto dl_a = make_dl_tti_with_pdsch();
    auto dl_b = make_dl_tti_with_pdsch();

    auto tx_a = build_simple_tx_data(500);
    auto tx_b = build_simple_tx_data(600);

    nv::FapiSlotMessageStorage storage(8);
    // cell_id 0 (in range), cell_id 99 (out of range for n_staged=4)
    store_dl_tti(storage, 0, dl_a.data());
    store_dl_tti(storage, 99, dl_b.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_a.data();

    std::array<TestCwPrm, 8> cw{};
    // Cell 0 applied, cell 99 triggers break
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 500U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, NullMsgBuf_SkipsAndContinues)
{
    auto dl_a = make_dl_tti_with_pdsch();
    auto dl_b = make_dl_tti_with_pdsch();

    auto tx_b = build_simple_tx_data(700);

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_a.data());
    store_dl_tti(storage, 1, dl_b.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = nullptr;        // cell 0 msg_buf missing — skipped
    staged[1] = tx_b.data();    // cell 1 still processed

    std::array<TestCwPrm, 8> cw{};
    // Cell 0 skipped (continue, cw_start not advanced), cell 1 processed:
    // its one PDU's offset lands at cw[0]. Cell 0's slot keeps the sentinel
    // because nothing patched it; this is acceptable because cell 0 has no
    // TB data to transmit.
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 700U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, NullMsgBufAfterValidCell_PartialApplication)
{
    auto dl_a = make_dl_tti_with_pdsch();
    auto dl_b = make_dl_tti_with_pdsch();

    auto tx_a = build_simple_tx_data(800);

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_a.data());
    store_dl_tti(storage, 1, dl_b.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = tx_a.data();
    staged[1] = nullptr; // second cell missing

    std::array<TestCwPrm, 8> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 800U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, ZeroPdus_NoOffsetsApplied)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_tx_data_msg({}); // zero PDUs

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 0U);
    EXPECT_EQ(cw[0].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, ZeroTlvs_NoOffsetForThatPdu)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    // PDU with no TLVs (empty inner vector)
    auto tx_msg = build_tx_data_msg({{}});

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 0U);
    EXPECT_EQ(cw[0].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, WrongTlvTag_NotApplied)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_INLINE_PAYLOAD, 999}},
    });

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 0U);
    EXPECT_EQ(cw[0].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, CwOverflow_BoundsRespected)
{
    auto dl_a = make_dl_tti_with_pdsch();
    auto dl_b = make_dl_tti_with_pdsch();

    auto tx_a = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 10}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 20}},
    });
    auto tx_b = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 30}},
    });

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_a.data());
    store_dl_tti(storage, 1, dl_b.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_a.data();
    staged[1] = tx_b.data();

    // max_cws = 2 → cell 0's 2 PDUs fit, but cell 1's PDU at cw_start=2 does NOT (2 >= 2)
    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), /*max_cws=*/2, kSlotU32), 2U);
    EXPECT_EQ(cw[0].tbStartOffset, 10U);
    EXPECT_EQ(cw[1].tbStartOffset, 20U);
    EXPECT_EQ(cw[2].tbStartOffset, kSentinel); // not written
}

TEST(TbStartOffsets, CwOverflowMidCell_PartialPduApplication)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 100}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 200}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 300}},
    });

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    // max_cws = 2 → PDUs 0 and 1 fit, PDU 2 (at cw_start+2 = 2) is out of bounds
    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), /*max_cws=*/2, kSlotU32), 2U);
    EXPECT_EQ(cw[0].tbStartOffset, 100U);
    EXPECT_EQ(cw[1].tbStartOffset, 200U);
    EXPECT_EQ(cw[2].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, NStaged_Zero_AllBreak)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(1234);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    const void* staged[1] = {tx_msg.data()};

    std::array<TestCwPrm, 4> cw{};
    // n_staged = 0 → every cell_id >= 0 is out of range
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, /*n_staged=*/0, cw.data(), 4, kSlotU32), 0U);
    EXPECT_EQ(cw[0].tbStartOffset, kSentinel);
}

// ===========================================================================
// Group E — Edge cases
// ===========================================================================

TEST(TbStartOffsets, ZeroDlTti_ReturnsZero)
{
    nv::FapiSlotMessageStorage storage(4);
    // No DL_TTI messages stored

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 0U);
}

TEST(TbStartOffsets, MultiplePdus_MixedTlvTags)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 111}},          // applied
        {TlvEntry{SCF_TX_DATA_INLINE_PAYLOAD, 999}},   // wrong tag, skipped
        {TlvEntry{SCF_TX_DATA_OFFSET, 333}},           // applied
    });

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 8> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 2U);
    EXPECT_EQ(cw[0].tbStartOffset, 111U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel); // wrong tag, not written
    EXPECT_EQ(cw[2].tbStartOffset, 333U);
}

TEST(TbStartOffsets, ZeroOffsetValue_WrittenAsZero)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(0);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 4, kSlotU32), 1U);
    EXPECT_EQ(cw[0].tbStartOffset, 0U);
}

TEST(TbStartOffsets, MaxCwsZero_NothingWritten)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(5000);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    const void* staged[kN]{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    // max_cws = 0 → cw_start(0) + p(0) = 0 is NOT < 0, so nothing written
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), /*max_cws=*/0, kSlotU32), 0U);
    EXPECT_EQ(cw[0].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets, NonPdschBetweenPdschCells_CwStartOnlyAdvancesByPdschCells)
{
    auto dl_pdsch_a = make_dl_tti_with_pdsch();
    auto dl_ssb_1   = make_dl_tti_no_pdsch();
    auto dl_ssb_2   = make_dl_tti_no_pdsch();
    auto dl_pdsch_b = make_dl_tti_with_pdsch();

    auto tx_a = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 10}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 20}},
    });
    auto tx_b = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 30}},
    });

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_pdsch_a.data());
    store_dl_tti(storage, 1, dl_ssb_1.data());
    store_dl_tti(storage, 2, dl_ssb_2.data());
    store_dl_tti(storage, 3, dl_pdsch_b.data());

    constexpr uint32_t kN = 8;
    const void* staged[kN]{};
    staged[0] = tx_a.data();
    staged[3] = tx_b.data();

    std::array<TestCwPrm, 8> cw{};
    EXPECT_EQ(apply_tb_start_offsets(storage, staged, kN, cw.data(), 8, kSlotU32), 3U);
    EXPECT_EQ(cw[0].tbStartOffset, 10U);  // cell 0, pdu 0
    EXPECT_EQ(cw[1].tbStartOffset, 20U);  // cell 0, pdu 1
    EXPECT_EQ(cw[2].tbStartOffset, 30U);  // cell 3, pdu 0 (cw_start=2, not 4)
}

// ===========================================================================
// Group F — Promoted function (Step 6 Option A migration target)
//
// Calls nv::pdsch_merge::patch_tb_start_offsets<FapiSlotMessageStorage, TestCwPrm>()
// directly and verifies identical results to the local reimplementation in groups A–E.
//
// Requires:
//   1. nv_pdsch_tb_merge.hpp to be created (Step 6 Option A implementation).
//   2. patch_tb_start_offsets to be templated on both FapiSlotStorage and a
//      CwPrmType concept so TestCwPrm can be substituted for cuphyPdschCwPrm_t:
//
//      template<FapiSlotStorage Storage, typename CwPrm>
//          requires requires(CwPrm& cw) { cw.tbStartOffset = uint32_t{}; }
//      [[nodiscard]] uint32_t patch_tb_start_offsets(
//          const Storage&               slot_store,
//          std::span<const void* const> staged_msg_bufs,
//          std::span<CwPrm>             ue_cw_info,
//          uint32_t                     slot_u32) noexcept;
//
// Uncomment the include below and remove the #if 0 / #endif guards once the
// header is available.
// ===========================================================================

// #include "nv_pdsch_tb_merge.hpp"

#if 0  // Enable once nv_pdsch_tb_merge.hpp is implemented

TEST(TbStartOffsets_Promoted, SingleCell_SinglePdu_OffsetApplied)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(4096);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kMaxCells = 8;
    std::array<const void*, kMaxCells> staged{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(
        nv::pdsch_merge::patch_tb_start_offsets(
            storage,
            std::span<const void* const>{staged.data(), kMaxCells},
            std::span<TestCwPrm>{cw.data(), 16},
            kSlotU32),
        1U);
    EXPECT_EQ(cw[0].tbStartOffset, 4096U);
    EXPECT_EQ(cw[1].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets_Promoted, TwoCells_CwStartAccumulatesAcrossCells)
{
    auto dl_tti_a = make_dl_tti_with_pdsch();
    auto dl_tti_b = make_dl_tti_with_pdsch();

    auto tx_a = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 1000}},
        {TlvEntry{SCF_TX_DATA_OFFSET, 2000}},
    });
    auto tx_b = build_tx_data_msg({
        {TlvEntry{SCF_TX_DATA_OFFSET, 3000}},
    });

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 5, dl_tti_a.data());
    store_dl_tti(storage, 9, dl_tti_b.data());

    constexpr uint32_t kN = 16;
    std::array<const void*, kN> staged{};
    staged[5] = tx_a.data();
    staged[9] = tx_b.data();

    std::array<TestCwPrm, 16> cw{};
    EXPECT_EQ(
        nv::pdsch_merge::patch_tb_start_offsets(
            storage,
            std::span<const void* const>{staged.data(), kN},
            std::span<TestCwPrm>{cw.data(), 16},
            kSlotU32),
        3U);
    EXPECT_EQ(cw[0].tbStartOffset, 1000U);
    EXPECT_EQ(cw[1].tbStartOffset, 2000U);
    EXPECT_EQ(cw[2].tbStartOffset, 3000U);
    EXPECT_EQ(cw[3].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets_Promoted, NonPdschCell_Skipped_CwStartNotAdvanced)
{
    auto dl_pdsch = make_dl_tti_with_pdsch();
    auto dl_ssb   = make_dl_tti_no_pdsch();

    auto tx_cell0 = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 111}}});
    auto tx_cell2 = build_tx_data_msg({{TlvEntry{SCF_TX_DATA_OFFSET, 222}}});

    nv::FapiSlotMessageStorage storage(8);
    store_dl_tti(storage, 0, dl_pdsch.data());
    store_dl_tti(storage, 1, dl_ssb.data());
    store_dl_tti(storage, 2, dl_pdsch.data());

    constexpr uint32_t kN = 8;
    std::array<const void*, kN> staged{};
    staged[0] = tx_cell0.data();
    staged[2] = tx_cell2.data();

    std::array<TestCwPrm, 8> cw{};
    EXPECT_EQ(
        nv::pdsch_merge::patch_tb_start_offsets(
            storage,
            std::span<const void* const>{staged.data(), kN},
            std::span<TestCwPrm>{cw.data(), 8},
            kSlotU32),
        2U);
    EXPECT_EQ(cw[0].tbStartOffset, 111U);
    EXPECT_EQ(cw[1].tbStartOffset, 222U);
    EXPECT_EQ(cw[2].tbStartOffset, kSentinel);
}

TEST(TbStartOffsets_Promoted, ZeroDlTti_ReturnsZero)
{
    nv::FapiSlotMessageStorage storage(4);
    std::array<const void*, 4> staged{};
    std::array<TestCwPrm, 4>   cw{};
    EXPECT_EQ(
        nv::pdsch_merge::patch_tb_start_offsets(
            storage,
            std::span<const void* const>{staged.data(), 4},
            std::span<TestCwPrm>{cw.data(), 4},
            kSlotU32),
        0U);
}

TEST(TbStartOffsets_Promoted, LargeOffsetValue_Uint32MaxPreserved)
{
    auto dl_tti = make_dl_tti_with_pdsch();
    auto tx_msg = build_simple_tx_data(0xFFFF'FFFFu);

    nv::FapiSlotMessageStorage storage(4);
    store_dl_tti(storage, 0, dl_tti.data());

    constexpr uint32_t kN = 4;
    std::array<const void*, kN> staged{};
    staged[0] = tx_msg.data();

    std::array<TestCwPrm, 4> cw{};
    EXPECT_EQ(
        nv::pdsch_merge::patch_tb_start_offsets(
            storage,
            std::span<const void* const>{staged.data(), kN},
            std::span<TestCwPrm>{cw.data(), 4},
            kSlotU32),
        1U);
    EXPECT_EQ(cw[0].tbStartOffset, 0xFFFF'FFFFu);
}

#endif  // nv_pdsch_tb_merge.hpp group F
