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
#include <array>
#include <cstdint>
#include <vector>

#include "nv_tx_data_h2d_helpers.hpp"

namespace
{

/// Pack sfn/slot into u32 using the sfn_slot_t union — endianness-agnostic.
uint32_t pack_sfn_slot(uint16_t sfn, uint16_t slot) noexcept
{
    sfn_slot_t ss;
    ss.u16.sfn  = sfn;
    ss.u16.slot = slot;
    return ss.u32;
}

constexpr std::size_t kTestMaxCells = 256;
using StagedArray = std::array<std::uint8_t*, kTestMaxCells>;

StagedArray make_staged_array(
    const std::vector<std::uint16_t>& cell_ids,
    const std::vector<std::uint8_t*>& ptrs)
{
    StagedArray arr{};
    for (std::size_t i = 0; i < cell_ids.size(); ++i)
    {
        arr[cell_ids[i]] = ptrs[i];
    }
    return arr;
}

} // namespace

TEST(TxDataH2dHelpers, slot_in_frame_from_slot_u32)
{
    const uint32_t packed = pack_sfn_slot(3, 7);
    EXPECT_EQ(nv::tx_data_h2d::slot_in_frame_from_slot_u32(packed), 7U);
}

TEST(TxDataH2dHelpers, tb_gpu_buffer_index_wraps)
{
    const uint32_t packed = pack_sfn_slot(0, 25);
    EXPECT_EQ(nv::tx_data_h2d::tb_gpu_buffer_index(packed),
              static_cast<uint8_t>(25U % nv::tx_data_h2d::kPdschTbGpuBufferRing));
}

TEST(TxDataH2dHelpers, dl_tti_expects_tx_data)
{
    EXPECT_FALSE(nv::tx_data_h2d::dl_tti_expects_tx_data(0));
    EXPECT_TRUE(nv::tx_data_h2d::dl_tti_expects_tx_data(1));
}

// --- merge_ue_tb_ptr_ordinal (same as process_aggr_pdsch_channel split-phase merge) ---

TEST(TxDataH2dMergeOrdinal, AllPermutationsK3_UeTbPtrMatchesKthPdschDlTtiInStorageOrder)
{
    // Distinct mock GPU buffers per logical cell — merge must not depend on numeric cell_id order.
    alignas(8) std::uint8_t buf_a[4]{};
    alignas(8) std::uint8_t buf_b[4]{};
    alignas(8) std::uint8_t buf_c[4]{};

    const std::vector<std::uint16_t> cell_ids{57, 12, 203};
    const std::vector<std::uint8_t*>   ptrs{buf_a, buf_b, buf_c};

    std::array<std::size_t, 3> perm{0, 1, 2};
    do
    {
        std::array<std::uint8_t*, 32> ue{};
        StagedArray staged = make_staged_array(cell_ids, ptrs);
        const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
            3,
            [&](std::uint16_t /*i*/) { return true; },
            [&](std::uint16_t i) { return cell_ids[perm[static_cast<std::size_t>(i)]]; },
            staged.data(),
            staged.size(),
            ue.data(),
            32U);
        EXPECT_EQ(mr.assigned, 3U);
        EXPECT_EQ(mr.overflow_count, 0U);
        for (std::size_t k = 0; k < 3; ++k)
        {
            EXPECT_EQ(ue[k], ptrs[perm[k]])
                << "perm=" << perm[0] << perm[1] << perm[2] << " k=" << k;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TxDataH2dMergeOrdinal, AllPermutationsK4_UeTbPtrMatchesKthPdschDlTtiInStorageOrder)
{
    alignas(8) std::uint8_t b0[1]{};
    alignas(8) std::uint8_t b1[1]{};
    alignas(8) std::uint8_t b2[1]{};
    alignas(8) std::uint8_t b3[1]{};

    const std::vector<std::uint16_t> cell_ids{1, 2, 3, 4};
    const std::vector<std::uint8_t*>   ptrs{b0, b1, b2, b3};

    std::array<std::size_t, 4> perm{0, 1, 2, 3};
    do
    {
        std::array<std::uint8_t*, 64> ue{};
        StagedArray staged = make_staged_array(cell_ids, ptrs);
        const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
            4,
            [&](std::uint16_t /*i*/) { return true; },
            [&](std::uint16_t i) { return cell_ids[perm[static_cast<std::size_t>(i)]]; },
            staged.data(),
            staged.size(),
            ue.data(),
            64U);
        EXPECT_EQ(mr.assigned, 4U);
        EXPECT_EQ(mr.overflow_count, 0U);
        for (std::size_t k = 0; k < 4; ++k)
        {
            EXPECT_EQ(ue[k], ptrs[perm[k]]);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(TxDataH2dMergeOrdinal, SsbOnlyDlTtiSkipped_DoesNotConsumeOrdinalRow)
{
    // Indices 0..4: only 0,2,4 have PDSCH — ordinals map to those three cells in order.
    const std::vector<std::uint16_t> cell_ids{10, 99, 20, 88, 30};
    alignas(8) std::uint8_t b0[1]{};
    alignas(8) std::uint8_t b1[1]{};
    alignas(8) std::uint8_t b2[1]{};
    const std::vector<std::uint8_t*> ptrs{b0, b1, b2};
    StagedArray staged = make_staged_array(
        std::vector<std::uint16_t>{10, 20, 30},
        ptrs);

    std::array<std::uint8_t*, 16> ue{};
    std::array<bool, 5>         has_pdsch{true, false, true, false, true};

    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        5,
        [&](std::uint16_t i) { return has_pdsch[static_cast<std::size_t>(i)]; },
        [&](std::uint16_t i) { return cell_ids[static_cast<std::size_t>(i)]; },
        staged.data(),
        staged.size(),
        ue.data(),
        16U);
    EXPECT_EQ(mr.assigned, 3U);
    EXPECT_EQ(ue[0], b0);
    EXPECT_EQ(ue[1], b1);
    EXPECT_EQ(ue[2], b2);
}

TEST(TxDataH2dMergeOrdinal, MissingStagedKey_LeavesRowUnset)
{
    alignas(8) std::uint8_t b0[1]{};
    alignas(8) std::uint8_t b1[1]{};

    StagedArray staged{};
    staged[7]  = b0;
    staged[99] = b1;

    std::array<std::uint8_t*, 16> ue{};
    ue.fill(nullptr);

    const std::vector<std::uint16_t> cells{7, 42, 99}; // middle cell has no TX_DATA staging
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        3,
        [&](std::uint16_t /*i*/) { return true; },
        [&](std::uint16_t i) { return cells[static_cast<std::size_t>(i)]; },
        staged.data(),
        staged.size(),
        ue.data(),
        16U);
    EXPECT_EQ(mr.assigned, 2U);
    EXPECT_EQ(ue[0], b0);
    EXPECT_EQ(ue[1], nullptr); // not written — no staged entry for cell 42
    EXPECT_EQ(ue[2], b1);
}

TEST(TxDataH2dMergeOrdinal, OverflowWhenOrdinalExceedsMaxRows)
{
    alignas(8) std::uint8_t b[4]{};
    StagedArray staged{};
    staged[1] = b;
    staged[2] = b;
    staged[3] = b;

    std::array<std::uint8_t*, 4> ue{};
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        3,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return static_cast<std::uint16_t>(i + 1); },
        staged.data(),
        staged.size(),
        ue.data(),
        2U);
    EXPECT_EQ(mr.assigned, 2U);
    EXPECT_EQ(mr.overflow_count, 1U);
}

TEST(TxDataH2dMergeOrdinal, NullStagedPointer_NotCountedAsAssigned)
{
    alignas(8) std::uint8_t b[1]{};
    StagedArray staged{};
    staged[1] = nullptr;
    staged[2] = b;

    std::array<std::uint8_t*, 8> ue{};
    ue.fill(nullptr);
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        2,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return static_cast<std::uint16_t>(i + 1); },
        staged.data(),
        staged.size(),
        ue.data(),
        8U);
    EXPECT_EQ(mr.assigned, 1U);
    EXPECT_EQ(ue[0], nullptr);
    EXPECT_EQ(ue[1], b);
}

// --- A-3 edge-case tests (mismatch / failure path corner conditions) ---

TEST(TxDataH2dMergeOrdinal, CellIdExceedsArraySize_SkippedSafely)
{
    alignas(8) std::uint8_t b0[1]{};
    StagedArray staged{};
    staged[5] = b0;

    std::array<std::uint8_t*, 8> ue{};
    ue.fill(nullptr);

    // cell_id 300 exceeds kTestMaxCells (256) — must be skipped without OOB access.
    // cell_id 5 is within bounds and has a staged pointer.
    const std::vector<std::uint16_t> cells{300, 5};
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        2,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return cells[static_cast<std::size_t>(i)]; },
        staged.data(),
        staged.size(),
        ue.data(),
        8U);
    EXPECT_EQ(mr.assigned, 1U);
    EXPECT_EQ(mr.overflow_count, 0U);
    EXPECT_EQ(ue[0], nullptr); // cell 300 out-of-bounds → row left unset
    EXPECT_EQ(ue[1], b0);     // cell 5 in-bounds → assigned
}

TEST(TxDataH2dMergeOrdinal, AllStagedEntriesNull_AssignsNothing)
{
    StagedArray staged{};

    std::array<std::uint8_t*, 8> ue{};
    ue.fill(nullptr);

    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        3,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return i; },
        staged.data(),
        staged.size(),
        ue.data(),
        8U);
    EXPECT_EQ(mr.assigned, 0U);
    EXPECT_EQ(mr.overflow_count, 0U);
    for (std::size_t k = 0; k < 3; ++k)
    {
        EXPECT_EQ(ue[k], nullptr);
    }
}

TEST(TxDataH2dMergeOrdinal, ZeroDlTti_EmptyResult)
{
    StagedArray staged{};
    std::array<std::uint8_t*, 4> ue{};
    ue.fill(nullptr);

    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        0,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return i; },
        staged.data(),
        staged.size(),
        ue.data(),
        4U);
    EXPECT_EQ(mr.assigned, 0U);
    EXPECT_EQ(mr.overflow_count, 0U);
}

TEST(TxDataH2dMergeOrdinal, SmallNStaged_OutOfBoundsCellIdsSkipped)
{
    constexpr std::size_t kSmallSize = 4;
    alignas(8) std::uint8_t b0[1]{};
    alignas(8) std::uint8_t b1[1]{};

    std::array<std::uint8_t*, kSmallSize> staged{};
    staged[1] = b0;
    staged[3] = b1;

    std::array<std::uint8_t*, 8> ue{};
    ue.fill(nullptr);

    // cell_ids: 1 (in-bounds), 5 (out of bounds for n_staged=4), 3 (in-bounds), 10 (OOB)
    const std::vector<std::uint16_t> cells{1, 5, 3, 10};
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        4,
        [&](std::uint16_t) { return true; },
        [&](std::uint16_t i) { return cells[static_cast<std::size_t>(i)]; },
        staged.data(),
        kSmallSize,
        ue.data(),
        8U);
    EXPECT_EQ(mr.assigned, 2U);
    EXPECT_EQ(mr.overflow_count, 0U);
    EXPECT_EQ(ue[0], b0);     // cell 1 → in-bounds, staged
    EXPECT_EQ(ue[1], nullptr); // cell 5 → OOB, skipped
    EXPECT_EQ(ue[2], b1);     // cell 3 → in-bounds, staged
    EXPECT_EQ(ue[3], nullptr); // cell 10 → OOB, skipped
}
