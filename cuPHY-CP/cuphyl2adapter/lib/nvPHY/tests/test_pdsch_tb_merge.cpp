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
 * @file test_pdsch_tb_merge.cpp
 * @brief Unit tests for nv::pdsch_merge::merge_and_patch<FapiSlotStorage>() and
 *        nv::pdsch_merge::patch_tb_start_offsets<FapiSlotStorage>() (Step 6 Option A).
 *
 * Uses a lightweight mock satisfying the FapiSlotStorage concept — no dependency on
 * FapiSlotMessageStorage, PHY_module, CUDA, or cuPHY. cuphyPdschParams_t is approximated
 * by TestPdschParams (only the fields accessed by merge_and_patch are populated).
 *
 * Covers:
 *   Group MergeOnly_NoPatch      — empty msg_span: ordinal map only; pBufferType set; no offset patch
 *   Group MergeAndPatch_Full     — all cells PDSCH + TX_DATA msg_buf: ue_tb_assigned + offsets applied
 *   Group MergeStats_AllZero     — no PDSCH PDUs → all MergeStats counters zero
 *   Group Overflow               — PDSCH ordinals > MAX_PDSCH_UE_PER_TTI: overflow_count correct, no OOB write
 *   Group PBufferType            — pBufferType set unconditionally after merge, before patch
 *   Group NoStagedMsgBufs_NoPatch — staged_msg_bufs span empty → tb_offsets_applied == 0, no crash
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <tuple>
#include <vector>

#include "nv_pdsch_tb_merge.hpp"

namespace
{

// ---------------------------------------------------------------------------
// Minimal surrogate for cuphyPdschCwPrm_t — only tbStartOffset is accessed.
// ---------------------------------------------------------------------------
struct TestCwPrm
{
    uint32_t tbStartOffset{0xDEAD'BEEFu};
};

// ---------------------------------------------------------------------------
// Minimal surrogate for cuphyPdschParams_t — only fields touched by
// merge_and_patch are present: ue_tb_ptr[], ue_cw_info[], tb_data.pBufferType.
// tb_data is nested to mirror cuphyPdschDataIn_t inside the real pdsch_params.
// ---------------------------------------------------------------------------
enum class TestBufferType : uint8_t
{
    CPU_BUFFER = 0,
    GPU_BUFFER = 1
};

constexpr uint32_t kMaxUePerTti = 16U;
constexpr uint32_t kMaxCwPerTti = 32U;

struct TestTbData
{
    TestBufferType pBufferType{TestBufferType::CPU_BUFFER};
};

struct TestPdschParams
{
    std::array<uint8_t*, kMaxUePerTti>  ue_tb_ptr{};
    std::array<TestCwPrm, kMaxCwPerTti> ue_cw_info{};
    TestTbData                          tb_data{};
};

// ---------------------------------------------------------------------------
// FapiSlotStorage concept mock.
// Satisfies nv::pdsch_merge::FapiSlotStorage without linking FapiSlotMessageStorage.
// ---------------------------------------------------------------------------
struct MockSlotStorage
{
    struct Entry
    {
        bool     has_pdsch{};
        uint16_t cell_id{};
    };

    std::vector<Entry> entries{};

    [[nodiscard]] uint16_t dl_tti_count() const noexcept
    {
        return static_cast<uint16_t>(entries.size());
    }

    [[nodiscard]] bool dl_tti_has_pdsch_pdu(uint16_t i) const noexcept
    {
        return i < entries.size() && entries[static_cast<std::size_t>(i)].has_pdsch;
    }

    [[nodiscard]] uint16_t dl_tti_cell_id(uint16_t i) const noexcept
    {
        return (i < entries.size()) ? entries[static_cast<std::size_t>(i)].cell_id : 0U;
    }
};

// ---------------------------------------------------------------------------
// TX_DATA msg_buf builder (mirrors test_tb_start_offsets.cpp helpers).
// ---------------------------------------------------------------------------
constexpr std::size_t kTlvSlot = sizeof(scf_fapi_tl_t) + sizeof(uint32_t);

/**
 * @brief Build a TX_DATA.req byte buffer carrying one PDU per supplied offset.
 *
 * Each PDU holds a single SCF_TX_DATA_OFFSET TLV whose value is the matching
 * entry in @p offsets. @c msg_hdr.length is set to the SCF body length counted
 * from immediately after the body header (sfn + slot + num_pdus + the PDU
 * array) so patch_tb_start_offsets() walks every PDU; an undersized length
 * would drop trailing PDUs from the walk.
 *
 * @param[in] offsets  One tbStartOffset value per PDU to emit, in order.
 *
 * @return Byte buffer sized exactly to hold the header, body header, fixed
 *         prefix, and @c offsets.size() PDUs. Return value must be checked.
 */
std::vector<uint8_t> build_tx_data_pdus(std::initializer_list<uint32_t> offsets)
{
    const std::size_t n        = offsets.size();
    const std::size_t pdu_span = n * (sizeof(scf_fapi_tx_data_pdu_info_t) + kTlvSlot);
    const std::size_t total    = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t)
                              + sizeof(uint16_t) * 3 // sfn + slot + num_pdus
                              + pdu_span;

    std::vector<uint8_t> buf(total, 0);
    // total is always >= the header sizes, so buf is never empty; the explicit
    // guard tells GCC's -Wnull-dereference analysis that data() cannot return
    // nullptr (vector::data() may return null only for an empty vector).
    if (buf.empty()) [[unlikely]] { return buf; }
    auto*                hdr = reinterpret_cast<scf_fapi_header_t*>(buf.data());
    hdr->message_count       = 1;

    auto* tx_req           = reinterpret_cast<scf_fapi_tx_data_req_t*>(hdr->payload);
    tx_req->num_pdus       = static_cast<uint16_t>(n);
    tx_req->msg_hdr.length = static_cast<uint32_t>(sizeof(uint16_t) * 3 + pdu_span);

    auto*    cursor = reinterpret_cast<uint8_t*>(tx_req->payload);
    uint16_t idx    = 0;
    for(uint32_t offset_value : offsets)
    {
        auto* pdu      = reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(cursor);
        pdu->pdu_index = idx++;
        pdu->num_tlv   = 1;

        auto* tlv   = reinterpret_cast<scf_fapi_tl_t*>(pdu->tlvs);
        tlv->tag    = SCF_TX_DATA_OFFSET;
        tlv->length = sizeof(uint32_t);
        std::memcpy(tlv->val, &offset_value, sizeof(uint32_t));

        cursor += sizeof(scf_fapi_tx_data_pdu_info_t) + kTlvSlot;
    }

    return buf;
}

/**
 * @brief Convenience wrapper building a single-PDU TX_DATA.req buffer.
 *
 * @param[in] offset_value  tbStartOffset carried by the lone SCF_TX_DATA_OFFSET TLV.
 *
 * @return Single-PDU TX_DATA.req byte buffer (see build_tx_data_pdus()).
 */
std::vector<uint8_t> build_tx_data_one_pdu(uint32_t offset_value)
{
    return build_tx_data_pdus({offset_value});
}

constexpr uint32_t kSlotU32  = 0x0001'0005u; // SFN=1, slot=5
constexpr uint32_t kSentinel = 0xDEAD'BEEFu;

} // namespace

// ===========================================================================
// Group MergeOnly_NoPatch
// ===========================================================================

TEST(PdschTbMerge_MergeOnly_NoPatch, EmptyMsgSpan_OrdinalMapOnly)
{
    alignas(8) std::uint8_t gpu0[4]{};
    alignas(8) std::uint8_t gpu1[4]{};

    MockSlotStorage storage;
    storage.entries = {{true, 0U}, {true, 1U}};

    std::array<uint8_t*, 20> staged_gpu{};
    staged_gpu[0] = gpu0;
    staged_gpu[1] = gpu1;

    TestPdschParams                    pdsch{};
    const auto                         gpu_span = std::span<uint8_t* const>{staged_gpu.data(), 20};
    const std::span<const void* const> empty_msg{};

    const auto stats = nv::pdsch_merge::merge_and_patch(
        storage, gpu_span, empty_msg, pdsch, kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 2U);
    EXPECT_EQ(stats.tb_offsets_applied, 0U);
    EXPECT_EQ(stats.overflow_count, 0U);
    EXPECT_EQ(pdsch.ue_tb_ptr[0], gpu0);
    EXPECT_EQ(pdsch.ue_tb_ptr[1], gpu1);
    EXPECT_EQ(pdsch.tb_data.pBufferType, TestBufferType::GPU_BUFFER);
    // Offsets not touched — remain at sentinel.
    EXPECT_EQ(pdsch.ue_cw_info[0].tbStartOffset, kSentinel);
}

// ===========================================================================
// Group MergeAndPatch_Full
// ===========================================================================

TEST(PdschTbMerge_MergeAndPatch_Full, TwoCells_PtrsAndOffsetsApplied)
{
    alignas(8) std::uint8_t gpu0[4]{};
    alignas(8) std::uint8_t gpu1[4]{};

    auto tx0 = build_tx_data_one_pdu(1024U);
    auto tx1 = build_tx_data_one_pdu(2048U);

    MockSlotStorage storage;
    storage.entries = {{true, 0U}, {true, 1U}};

    std::array<uint8_t*, 20>    staged_gpu{};
    std::array<const void*, 20> staged_msg{};
    staged_gpu[0] = gpu0;
    staged_gpu[1] = gpu1;
    staged_msg[0] = tx0.data();
    staged_msg[1] = tx1.data();

    TestPdschParams pdsch{};
    const auto      gpu_span = std::span<uint8_t* const>{staged_gpu.data(), 20};
    const auto      msg_span = std::span<const void* const>{staged_msg.data(), 20};

    const auto stats = nv::pdsch_merge::merge_and_patch(
        storage, gpu_span, msg_span, pdsch, kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 2U);
    EXPECT_EQ(stats.tb_offsets_applied, 2U);
    EXPECT_EQ(stats.overflow_count, 0U);
    EXPECT_EQ(pdsch.ue_tb_ptr[0], gpu0);
    EXPECT_EQ(pdsch.ue_tb_ptr[1], gpu1);
    EXPECT_EQ(pdsch.ue_cw_info[0].tbStartOffset, 1024U);
    EXPECT_EQ(pdsch.ue_cw_info[1].tbStartOffset, 2048U);
    EXPECT_EQ(pdsch.tb_data.pBufferType, TestBufferType::GPU_BUFFER);
}

// One cell, two TX_DATA PDUs in the same msg_buf. This exercises the
// multi-PDU walk so that an imprecise msg_hdr.length (e.g. short by the body
// header) would drop the second PDU's offset — the single-PDU case above
// cannot catch that because the walk stops after the first PDU regardless.
TEST(PdschTbMerge_MergeAndPatch_Full, OneCellTwoPdus_BothOffsetsApplied)
{
    alignas(8) std::uint8_t gpu0[4]{};

    auto tx0 = build_tx_data_pdus({1024U, 4096U});

    MockSlotStorage storage;
    storage.entries = {{true, 0U}};

    std::array<uint8_t*, 20>    staged_gpu{};
    std::array<const void*, 20> staged_msg{};
    staged_gpu[0] = gpu0;
    staged_msg[0] = tx0.data();

    TestPdschParams pdsch{};
    const auto      stats = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{staged_gpu.data(), 20},
        std::span<const void* const>{staged_msg.data(), 20},
        pdsch,
        kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 1U);
    EXPECT_EQ(stats.tb_offsets_applied, 2U);
    EXPECT_EQ(stats.overflow_count, 0U);
    EXPECT_EQ(pdsch.ue_tb_ptr[0], gpu0);
    // Both PDUs in the single cell map to consecutive codewords.
    EXPECT_EQ(pdsch.ue_cw_info[0].tbStartOffset, 1024U);
    EXPECT_EQ(pdsch.ue_cw_info[1].tbStartOffset, 4096U);
    EXPECT_EQ(pdsch.tb_data.pBufferType, TestBufferType::GPU_BUFFER);
}

// ===========================================================================
// Group MergeStats_AllZero — parameterized
// ===========================================================================

using MergeEntries = std::vector<MockSlotStorage::Entry>;

class PdschTbMerge_MergeStats_AllZero_P
    : public testing::TestWithParam<MergeEntries> {};

TEST_P(PdschTbMerge_MergeStats_AllZero_P, AllCountersZero)
{
    MockSlotStorage storage;
    storage.entries = GetParam();

    TestPdschParams pdsch{};
    const auto      stats = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{},
        std::span<const void* const>{},
        pdsch,
        kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 0U);
    EXPECT_EQ(stats.tb_offsets_applied, 0U);
    EXPECT_EQ(stats.overflow_count, 0U);
    // pBufferType is still set — merge_and_patch sets it unconditionally.
    EXPECT_EQ(pdsch.tb_data.pBufferType, TestBufferType::GPU_BUFFER);
}

INSTANTIATE_TEST_SUITE_P(AllZero, PdschTbMerge_MergeStats_AllZero_P, testing::Values(MergeEntries{},                        // zero DL_TTI entries
                                                                                     MergeEntries{{false, 0U}, {false, 1U}} // two entries, neither has PDSCH
                                                                                     ));

// ===========================================================================
// Group Overflow
// ===========================================================================

TEST(PdschTbMerge_Overflow, PdschOrdinalsBeyondMax_OverflowCountCorrect)
{
    // kMaxUePerTti = 16. Stage 20 PDSCH cells — 4 must overflow.
    constexpr uint32_t kStagedCells = 20U;

    alignas(8) std::array<std::uint8_t[4], kStagedCells> bufs{};
    std::array<uint8_t*, 20>                             staged_gpu{};
    for(uint32_t i = 0; i < kStagedCells; ++i)
    {
        staged_gpu[i] = bufs[i];
    }

    MockSlotStorage storage;
    for(uint32_t i = 0; i < kStagedCells; ++i)
    {
        storage.entries.push_back({true, static_cast<uint16_t>(i)});
    }

    TestPdschParams pdsch{};
    const auto      stats = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{staged_gpu.data(), kStagedCells},
        std::span<const void* const>{},
        pdsch,
        kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, kMaxUePerTti);
    EXPECT_EQ(stats.overflow_count, kStagedCells - kMaxUePerTti);
    // The first kMaxUePerTti rows must be filled; no out-of-bounds write occurred.
    for(uint32_t i = 0; i < kMaxUePerTti; ++i)
    {
        EXPECT_EQ(pdsch.ue_tb_ptr[i], bufs[i]) << "ordinal=" << i;
    }
}

// ===========================================================================
// Group PBufferType
// ===========================================================================

TEST(PdschTbMerge_PBufferType, SetUnconditionallyAfterMerge)
{
    // No staged pointers, no PDSCH PDUs — pBufferType must still be set.
    MockSlotStorage storage;
    TestPdschParams pdsch{};
    pdsch.tb_data.pBufferType = TestBufferType::CPU_BUFFER;

    std::ignore = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{},
        std::span<const void* const>{},
        pdsch,
        kSlotU32);

    EXPECT_EQ(pdsch.tb_data.pBufferType, TestBufferType::GPU_BUFFER);
}

// ===========================================================================
// Group NoStagedMsgBufs_NoPatch
// ===========================================================================

TEST(PdschTbMerge_NoStagedMsgBufs_NoPatch, EmptyMsgSpan_ZeroOffsets_NoCrash)
{
    alignas(8) std::uint8_t gpu0[4]{};

    MockSlotStorage storage;
    storage.entries = {{true, 0U}};

    std::array<uint8_t*, 20> staged_gpu{};
    staged_gpu[0] = gpu0;

    TestPdschParams pdsch{};
    // Deliberately pass empty msg_span — must not crash, tb_offsets_applied must be 0.
    const auto stats = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{staged_gpu.data(), 20},
        std::span<const void* const>{},
        pdsch,
        kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 1U);
    EXPECT_EQ(stats.tb_offsets_applied, 0U);
    EXPECT_EQ(pdsch.ue_tb_ptr[0], gpu0);
    EXPECT_EQ(pdsch.ue_cw_info[0].tbStartOffset, kSentinel);
}

TEST(PdschTbMerge_NoStagedMsgBufs_NoPatch, NonPdschCells_ZeroOffsets)
{
    MockSlotStorage storage;
    storage.entries = {{false, 0U}, {false, 1U}};

    std::array<uint8_t*, 20>    staged_gpu{};
    std::array<const void*, 20> staged_msg{};

    TestPdschParams pdsch{};
    const auto      stats = nv::pdsch_merge::merge_and_patch(
        storage,
        std::span<uint8_t* const>{staged_gpu.data(), 20},
        std::span<const void* const>{staged_msg.data(), 20},
        pdsch,
        kSlotU32);

    EXPECT_EQ(stats.ue_tb_assigned, 0U);
    EXPECT_EQ(stats.tb_offsets_applied, 0U);
    for(uint32_t i = 0; i < 4U; ++i)
    {
        EXPECT_EQ(pdsch.ue_cw_info[i].tbStartOffset, kSentinel) << "cw=" << i;
    }
}
