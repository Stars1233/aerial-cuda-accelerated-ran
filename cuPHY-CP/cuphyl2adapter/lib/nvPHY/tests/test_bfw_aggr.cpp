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
 * @file test_bfw_aggr.cpp
 * @brief Unit tests for DL/UL BFW message lanes: storage and dispatch.
 *
 * Layer 1 - Storage: verifies that FapiSlotMessageStorage correctly stores,
 *           counts, and preserves arrival order of DL/UL_BFW_CVI_REQUEST
 *           messages.
 *
 * Layer 2 - Dispatch: verifies that nv::dispatch_{dl,ul}_bfw_messages
 *           correctly iterates the stored BFW lane, rejects invalid cell_ids
 *           (negative or out-of-range), and invokes the per-cell callback only
 *           for valid entries with the correct cell_id and message reference.
 *
 * These tests cover lane storage and dispatch helpers only. DL/UL BFW EOM
 * scheduling is serialized by PHY_module::enqueue_channel_tasks() and is not
 * validated by this test file.
 *
 * Direction symmetry is captured by a small DirectionTraits struct so each
 * mirror-image test case can be expressed once and parameterized over
 * Direction = {DL, UL} via TEST_P.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <vector>

#include "nv_fapi_message_storage.hpp"
#include "nv_dl_aggr_tasks.hpp"
#include "nv_ul_aggr_tasks.hpp"
#include "nv_bfw_debug_cleanup.hpp"
#include "scf_5g_fapi.h"


#if defined(ENABLE_20C)
static constexpr std::size_t kMaxCells = 20;
#else
static constexpr std::size_t kMaxCells = 40;
#endif

namespace {

// ---------------------------------------------------------------------------
// Per-direction message factories
// ---------------------------------------------------------------------------

[[maybe_unused]] nv::phy_mac_msg_desc make_dl_bfw_msg(uint16_t cell_id)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_DL_BFW_CVI_REQUEST;
    msg.cell_id = cell_id;
    return msg;
}

[[maybe_unused]] nv::phy_mac_msg_desc make_ul_bfw_msg(uint16_t cell_id)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_BFW_CVI_REQUEST;
    msg.cell_id = cell_id;
    return msg;
}

// ---------------------------------------------------------------------------
// DirectionTraits
//
// Bundles the four per-direction operations so every mirror-image test case
// can be written once and run for both DL and UL via INSTANTIATE_TEST_SUITE_P.
// ---------------------------------------------------------------------------

using PerCellCallback =
    std::function<void(int cell_id, const nv::phy_mac_msg_desc& msg)>;

using DispatchFn =
    std::function<void(const nv::FapiSlotMessageStorage& storage,
                       std::size_t                       num_phy_instances,
                       PerCellCallback                   per_cell,
                       uint32_t                          slot_u32,
                       uint32_t                          ring_idx)>;

struct DirectionTraits {
    const char*                                                            name;
    uint8_t                                                                lane_msg_id;
    std::function<nv::phy_mac_msg_desc(uint16_t)>                          make_msg;
    std::function<std::size_t(const nv::FapiSlotMessageStorage&)>          count;
    std::function<const nv::phy_mac_msg_desc*(const nv::FapiSlotMessageStorage&)> messages;
    DispatchFn                                                             dispatch;
};

[[maybe_unused]] inline DirectionTraits dl_traits()
{
    return {
        "DL",
        SCF_FAPI_DL_BFW_CVI_REQUEST,
        &make_dl_bfw_msg,
        [](const nv::FapiSlotMessageStorage& s) { return s.dl_bfw_count(); },
        [](const nv::FapiSlotMessageStorage& s) { return s.dl_bfw_messages(); },
        [](const nv::FapiSlotMessageStorage& storage,
           std::size_t                       num_phy_instances,
           PerCellCallback                   per_cell,
           uint32_t                          slot_u32,
           uint32_t                          ring_idx) {
            nv::dispatch_dl_bfw_messages(storage, num_phy_instances,
                                         std::move(per_cell),
                                         slot_u32, ring_idx);
        },
    };
}

[[maybe_unused]] inline DirectionTraits ul_traits()
{
    return {
        "UL",
        SCF_FAPI_UL_BFW_CVI_REQUEST,
        &make_ul_bfw_msg,
        [](const nv::FapiSlotMessageStorage& s) { return s.ul_bfw_count(); },
        [](const nv::FapiSlotMessageStorage& s) { return s.ul_bfw_messages(); },
        [](const nv::FapiSlotMessageStorage& storage,
           std::size_t                       num_phy_instances,
           PerCellCallback                   per_cell,
           uint32_t                          slot_u32,
           uint32_t                          ring_idx) {
            nv::dispatch_ul_bfw_messages(storage, num_phy_instances,
                                         std::move(per_cell),
                                         slot_u32, ring_idx);
        },
    };
}

} // namespace

// ===========================================================================
// BfwDispatchTest fixture - parameterized over Direction = {DL, UL}
//
// Each TEST_P body runs once per direction (gtest expands the suite to e.g.
// `Directions/BfwDispatchTest.EmptyLaneNoCallback/DL` and `.../UL`).
// Use SCOPED_TRACE(T.name) inside each body so any sub-assertion failure
// reports which direction was running.
// ===========================================================================

class BfwDispatchTest : public ::testing::TestWithParam<DirectionTraits> {};

INSTANTIATE_TEST_SUITE_P(
    Directions,
    BfwDispatchTest,
    ::testing::Values(dl_traits(), ul_traits()),
    [](const ::testing::TestParamInfo<DirectionTraits>& info) {
        return std::string(info.param.name);
    });

// ---------------------------------------------------------------------------
// Empty-lane invariant: dispatch on empty storage must not invoke the
// per-cell callback. Mirrors {Dl,Ul}BfwDispatch.EmptyLaneNoCallback.
// ---------------------------------------------------------------------------
TEST_P(BfwDispatchTest, EmptyLaneNoCallback)
{
    const auto& T = GetParam();
    SCOPED_TRACE(T.name);

    nv::FapiSlotMessageStorage storage(kMaxCells);

    int callback_count = 0;
    T.dispatch(storage, kMaxCells,
        [&callback_count](int /*cell_id*/, const nv::phy_mac_msg_desc& /*msg*/) {
            ++callback_count;
        },
        0x0000'0000, 0);

    EXPECT_EQ(callback_count, 0);
}

// ---------------------------------------------------------------------------
// Single valid cell: dispatch invokes the per-cell callback exactly once,
// with cell_id=0, the lane[0] address, and the expected msg_id.
// Mirrors {Dl,Ul}BfwDispatch.SingleValidCellDispatchesOnceWithCorrectRef.
// ---------------------------------------------------------------------------
TEST_P(BfwDispatchTest, SingleValidCellDispatchesOnceWithCorrectRef)
{
    const auto& T = GetParam();
    SCOPED_TRACE(T.name);

    nv::FapiSlotMessageStorage storage(kMaxCells);

    ASSERT_EQ(storage.store_message(T.make_msg(0)), nv::StoreResult::Stored);
    ASSERT_EQ(T.count(storage), 1u);

    const nv::phy_mac_msg_desc* lane0 = T.messages(storage);

    std::vector<std::pair<int, const nv::phy_mac_msg_desc*>> invocations;
    T.dispatch(storage, kMaxCells,
        [&invocations](int cell_id, const nv::phy_mac_msg_desc& msg) {
            invocations.emplace_back(cell_id, &msg);
        },
        0x0001'0005, 0);

    ASSERT_EQ(invocations.size(), 1u);
    EXPECT_EQ(invocations[0].first, 0);
    EXPECT_EQ(invocations[0].second, &lane0[0]);
    EXPECT_EQ(invocations[0].second->msg_id, T.lane_msg_id);
    EXPECT_EQ(invocations[0].second->cell_id, 0);
}

// ---------------------------------------------------------------------------
// Routing / filtering by cell_id (table-driven over a vector of cases).
//
// Replaces five previously-separate per-direction tests by capturing the
// only thing they vary in: the input cell_ids, the num_phy_instances upper
// bound, the expected dispatched order, and the expected preserved-lane
// order. Mirrors:
//   - {Dl,Ul}BfwDispatch.TwoCellsPreservesArrivalOrder
//   - {Dl,Ul}BfwDispatch.NegativeCellIdSkippedThenValidDispatched
//   - {Dl,Ul}BfwDispatch.CellIdOutOfRangeSkippedThenValidDispatched
//   - {Dl,Ul}BfwDispatch.AllInvalidCellIdsNoCallback
//   - {Dl,Ul}BfwDispatch.SixValidCellsAllDispatchedInOrder
// ---------------------------------------------------------------------------
TEST_P(BfwDispatchTest, RoutingByCellId)
{
    const auto& T = GetParam();
    SCOPED_TRACE(T.name);

    struct RoutingCase {
        const char*          name;
        std::vector<int16_t> stored_cell_ids;       // can include -1 or out-of-range
        std::size_t          num_phy_instances;     // upper bound for valid cell_id
        std::vector<int>     expected_dispatched;   // cell_ids in dispatch order
        std::vector<int16_t> expected_lane_order;   // preserved arrival order in lane
        uint32_t             slot_u32;              // for log/trace context
    };

    const std::vector<RoutingCase> cases = {
        {"TwoCellsPreservesArrivalOrder",
         {0, 1},             kMaxCells, {0, 1},
         {0, 1},             0x0002'0003u},
        {"NegativeCellIdSkippedThenValidDispatched",
         {-1, 0},            kMaxCells, {0},
         {-1, 0},            0x0003'0007u},
        {"CellIdOutOfRangeSkippedThenValidDispatched",
         {0, 2},             /*num_phy=*/2, {0},
         {0, 2},             0x0004'0001u},
        {"AllInvalidCellIdsNoCallback",
         {-1, 99},           /*num_phy=*/2, {},
         {-1, 99},           0x0005'0002u},
        {"SixValidCellsAllDispatchedInOrder",
         {0, 1, 2, 3, 4, 5}, kMaxCells, {0, 1, 2, 3, 4, 5},
         {0, 1, 2, 3, 4, 5}, 0x0006'0000u},
    };

    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);

        nv::FapiSlotMessageStorage storage(kMaxCells);

        for (const auto cid : c.stored_cell_ids) {
            nv::phy_mac_msg_desc msg = T.make_msg(0);
            msg.cell_id              = cid;
            ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
        }
        ASSERT_EQ(T.count(storage), c.stored_cell_ids.size());

        const nv::phy_mac_msg_desc* lane = T.messages(storage);

        std::vector<int> dispatched;
        T.dispatch(storage, c.num_phy_instances,
            [&dispatched](int cell_id, const nv::phy_mac_msg_desc& /*msg*/) {
                dispatched.push_back(cell_id);
            },
            c.slot_u32, 0);

        EXPECT_EQ(dispatched, c.expected_dispatched);

        ASSERT_EQ(c.expected_lane_order.size(), c.stored_cell_ids.size());
        for (std::size_t i = 0; i < c.expected_lane_order.size(); ++i) {
            EXPECT_EQ(lane[i].cell_id, c.expected_lane_order[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Stress: store kMaxCells valid messages, dispatch all of them, and verify
// every cell_id, msg_id, and pointer identity. Kept as its own TEST_P
// because the all-fields, every-entry verification doesn't fold cleanly
// into the RoutingByCellId table above.
// Mirrors {Dl,Ul}BfwDispatch.MaxCellsAllFieldsAndPointerIdentity.
// ---------------------------------------------------------------------------
TEST_P(BfwDispatchTest, MaxCellsAllFieldsAndPointerIdentity)
{
    const auto& T = GetParam();
    SCOPED_TRACE(T.name);

    nv::FapiSlotMessageStorage storage(kMaxCells);

    for (uint16_t c = 0; c < kMaxCells; ++c) {
        ASSERT_EQ(storage.store_message(T.make_msg(c)), nv::StoreResult::Stored);
    }
    ASSERT_EQ(T.count(storage), static_cast<std::size_t>(kMaxCells));

    const nv::phy_mac_msg_desc* lane = T.messages(storage);

    std::vector<std::pair<int, const nv::phy_mac_msg_desc*>> invocations;
    T.dispatch(storage, kMaxCells,
        [&invocations](int cell_id, const nv::phy_mac_msg_desc& msg) {
            invocations.emplace_back(cell_id, &msg);
        },
        0x0007'0000, 0);

    ASSERT_EQ(invocations.size(), kMaxCells);
    for (std::size_t i = 0; i < kMaxCells; ++i) {
        EXPECT_EQ(invocations[i].first, static_cast<int>(i));
        EXPECT_EQ(invocations[i].second, &lane[i]);
        EXPECT_EQ(invocations[i].second->msg_id, T.lane_msg_id);
        EXPECT_EQ(invocations[i].second->cell_id, static_cast<int>(i));
    }
}

TEST(BfwDebugCleanup, FreesBusyHeaderOnly)
{
    uint8_t header = static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_BUSY);
    slot_command_api::bfw_coeff_mem_info_t info{};
    info.header = &header;

    EXPECT_TRUE(nv::detail::debug_free_busy_bfw_coeff_header(&info));
    EXPECT_EQ(header, static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_FREE));
}

TEST(BfwDebugCleanup, LeavesFreeHeaderUnchanged)
{
    uint8_t header = static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_FREE);
    slot_command_api::bfw_coeff_mem_info_t info{};
    info.header = &header;

    EXPECT_FALSE(nv::detail::debug_free_busy_bfw_coeff_header(&info));
    EXPECT_EQ(header, static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_FREE));
}

TEST(BfwDebugCleanup, NullInputsAreNoOp)
{
    slot_command_api::bfw_coeff_mem_info_t info{};

    EXPECT_FALSE(nv::detail::debug_free_busy_bfw_coeff_header(nullptr));
    EXPECT_FALSE(nv::detail::debug_free_busy_bfw_coeff_header(&info));
}
