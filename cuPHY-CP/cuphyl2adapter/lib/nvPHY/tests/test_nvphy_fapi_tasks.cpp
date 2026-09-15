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
 * @file test_nvphy_fapi_tasks.cpp
 * @brief Unit tests for the slot task framework introduced in bhas/dev/fapi_tasks.
 *
 * Covers:
 *   Group A — slot_index_from_u32() (constexpr pure function)
 *   Group B — SlotTaskPool layout and CplaneBatchAccum reset lifecycle
 *   Group C — task_work_fn_cplane_batch dispatch correctness
 *   Group D — DL EOM channel aggr task dispatch (pdsch/csirs/pdcch/ssb/dlbfw)
 *   Group D2— DL aggr staged-pointer propagation & guard logic
 *   Group E — UL EOM channel aggr task dispatch (pusch/prach/pucch/srs/ulbfw)
 *   Group F — store_eom_bitmap accumulation and EOM-trigger logic
 *   Group G — Deferred TX_DATA release (release_lane / tx_data_deferred_ lifecycle)
 *
 * All tests are independent of PHY_module instantiation — they use local
 * SlotTaskPool instances and lambda-backed dispatch tables.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <thread>

#include "nv_slot_task_pool.hpp"      // slot_index_from_u32, SlotTaskPool, SLOT_STORAGE_DEPTH, SFN_SLOT_INVALID
#include "nv_cplane_batch_tasks.hpp" // CplaneBatchTaskArg, CplaneBatchAccum, task_work_fn_cplane_batch
#include "nv_dl_aggr_tasks.hpp"      // DlAggrTaskArg, DlChannelDispatch, task_work_fn_aggr_*
#include "nv_ul_aggr_tasks.hpp"      // UlAggrTaskArg, UlChannelDispatch, task_work_fn_aggr_*

using namespace nv;

// ---------------------------------------------------------------------------
// Test stub: l2a_cpu_tracing_mode is defined in nv_phy_driver_proxy.cpp (part
// of nvphy) which we intentionally do not link.  Provide a minimal definition
// that returns DISABLED so the task work functions compile and link without
// the full PHYDriverProxy / cuphydriver dependency chain.
// ---------------------------------------------------------------------------
#include "task_instrumentation/task_instrumentation_v3.hpp"

namespace nv::detail
{
TracingMode l2a_cpu_tracing_mode() noexcept { return TracingMode::DISABLED; }
} // namespace nv::detail

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Pack sfn and slot into slot_u32 format (sfn in upper 16, slot in lower 16).
constexpr uint32_t make_slot_u32(uint16_t sfn, uint16_t slot)
{
    return (static_cast<uint32_t>(sfn) << 16U) | static_cast<uint32_t>(slot);
}

/// Numerology µ=1 → 20 slots/frame — used throughout as the standard test value.
constexpr uint32_t SPF_MU1 = 20;

} // namespace

// ===========================================================================
// Group A — slot_index_from_u32
// ===========================================================================

TEST(SlotIndex, SlotIndexFromU32)
{
    struct TestCase final {
        std::string_view description;
        uint32_t         slot_u32;
        uint32_t         spf;
        uint32_t         expected;
    };
    static constexpr std::array CASES = {
        TestCase{"zero_sfn_zero_slot",    make_slot_u32(0,    0),  SPF_MU1,     0U},
        TestCase{"nonzero_sfn_zero_slot", make_slot_u32(5,    0),  SPF_MU1,   100U},
        TestCase{"zero_sfn_nonzero_slot", make_slot_u32(0,    7),  SPF_MU1,     7U},
        TestCase{"general_case",          make_slot_u32(2,    7),  SPF_MU1,    47U},
        TestCase{"invalid_returns_zero",  SFN_SLOT_INVALID,        SPF_MU1,     0U},
        TestCase{"max_sfn_value",         make_slot_u32(1023, 19), SPF_MU1, 20479U},
    };
    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        EXPECT_EQ(slot_index_from_u32(tc.slot_u32, tc.spf), tc.expected);
    }
}

TEST(SlotIndex, RingIdxModDepth)
{
    struct TestCase final {
        std::string_view description;
        uint16_t         sfn;
        uint16_t         slot;
        uint32_t         expected_ring_idx;
    };
    static constexpr std::array CASES = {
        TestCase{"slot0_ring0",       0, 0, 0U},
        TestCase{"slot1_ring1",       0, 1, 1U},
        TestCase{"slot2_ring2",       0, 2, 2U},
        TestCase{"slot3_wraps_ring0", 0, 3, 0U},
    };
    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        const uint32_t idx = static_cast<uint32_t>(
            slot_index_from_u32(make_slot_u32(tc.sfn, tc.slot), SPF_MU1) % SLOT_STORAGE_DEPTH);
        EXPECT_EQ(idx, tc.expected_ring_idx);
    }
}

// ===========================================================================
// Group B — SlotTaskPool and CplaneBatchAccum
// ===========================================================================

TEST(SlotTaskPool, DefaultInitAllZero)
{
    nv::SlotTaskPool pool{};
    for (std::size_t i = 0; i < SLOT_STORAGE_DEPTH; ++i)
    {
        EXPECT_EQ(pool.tasks_in_flight[i].load(), 0);
        EXPECT_EQ(pool.dlc_accumulated_bitmap[i], 0U);
        EXPECT_EQ(pool.ulc_accumulated_bitmap[i], 0U);
        EXPECT_EQ(pool.store_eom_bitmap[i], 0U);
        EXPECT_EQ(pool.dlc_batch_accum[i].n_pending, 0);
        EXPECT_EQ(pool.dlc_batch_accum[i].batch_count, 0);
        EXPECT_EQ(pool.ulc_batch_accum[i].n_pending, 0);
        EXPECT_EQ(pool.ulc_batch_accum[i].batch_count, 0);
    }
}

TEST(SlotTaskPool, SentinelValue)
{
    EXPECT_EQ(nv::SlotTaskPool::SLOT_TASK_SENTINEL, 0x4000'0000);
}

TEST(SlotTaskPool, StorageDepthIs3)
{
    EXPECT_EQ(SLOT_STORAGE_DEPTH, std::size_t{3});
}

TEST(CplaneBatchAccum, DefaultZero)
{
    nv::CplaneBatchAccum a{};
    EXPECT_EQ(a.n_pending, 0);
    EXPECT_EQ(a.batch_count, 0);
}

TEST(CplaneBatchAccum, ResetClearsAll)
{
    nv::CplaneBatchAccum a{};
    a.n_pending  = 5;
    a.batch_count = 2;
    a.reset();
    EXPECT_EQ(a.n_pending, 0);
    EXPECT_EQ(a.batch_count, 0);
}

TEST(CplaneBatchAccum, ResetIsIdempotent)
{
    nv::CplaneBatchAccum a{};
    a.n_pending  = 3;
    a.batch_count = 1;
    a.reset();
    a.reset();
    EXPECT_EQ(a.n_pending, 0);
    EXPECT_EQ(a.batch_count, 0);
}

// ===========================================================================
// Group C — task_work_fn_cplane_batch
// ===========================================================================

namespace {

/// Build a minimal CplaneBatchTaskArg suitable for dispatch testing.
/// process_cell and on_complete are supplied by the caller.
nv::CplaneBatchTaskArg make_cplane_arg(
    uint32_t ring_idx,
    uint32_t slot_u32,
    const uint8_t* cell_ids,
    uint8_t n_cells,
    nv::cplane_process_cell_fn_t process_cell,
    nv::task_complete_fn_t       on_complete,
    void*                        slot_map = nullptr,
    nv::post_batch_fn_t          post_batch = nullptr,
    uint8_t                      batch_id = 0,
    uint8_t                      total_batches = 1,
    nv::CplaneBatchDirection     direction = nv::CplaneBatchDirection::DOWNLINK,
    std::size_t                  transaction_id = 0,
    nv::cplane_on_batch_error_fn_t on_batch_error = nullptr)
{
    nv::CplaneBatchTaskArg arg{};
    arg.phy_module   = nullptr;  // never dereferenced; callbacks own all logic
    arg.ring_idx     = ring_idx;
    arg.slot_u32     = slot_u32;
    arg.n_cells      = n_cells;
    arg.process_cell = process_cell;
    arg.on_complete  = on_complete;
    arg.slot_map     = slot_map;
    arg.post_batch   = post_batch;
    arg.on_batch_error = on_batch_error;
    arg.batch_id     = batch_id;
    arg.total_batches = total_batches;
    arg.direction    = direction;
    arg.transaction_id = transaction_id;
    for (uint8_t i = 0; i < n_cells; ++i)
    {
        arg.cell_ids[i] = cell_ids[i];
    }
    return arg;
}

} // namespace

// cplane_process_cell_fn_t / task_complete_fn_t are raw function pointers;
// capturing lambdas cannot be assigned to them.  s_current bridges the gap:
// SetUp() points it at `this` so non-capturing lambdas can reach per-test
// fixture members; TearDown() nulls it to catch any use-after-test access.
// GTest constructs a fresh fixture instance for every TEST_F, so all member
// variables are default-initialised per test with no manual reset needed
// (except inside the table-driven loop where counts are reset each iteration).
class CplaneBatchTask : public testing::Test {
public:
    int                   process_count     = 0;
    int                   complete_count    = 0;
    uint32_t              complete_ring_idx = 0;
    uint32_t              complete_slot_u32 = 0;
    uint32_t              seen_cell         = 0xFF;
    std::vector<uint32_t> seen_cells;
    uint32_t              seen_ring_idx     = 0xFFFF;
    int                   post_batch_count  = 0;
    void*                 seen_slot_map     = nullptr;
    uint8_t               seen_batch_id     = 0;
    uint8_t               seen_total_batches = 0;
    uint8_t               seen_n_cells      = 0;
    nv::CplaneBatchDirection seen_direction = nv::CplaneBatchDirection::DOWNLINK;
    int                   on_batch_error_count = 0;
    void*                 seen_err_slot_map = nullptr;
    std::vector<uint32_t> seen_err_cell_ids;
    int                   seen_err_count    = 0;

    static CplaneBatchTask* s_current;

    void SetUp()    override { s_current = this; }
    void TearDown() override { s_current = nullptr; }
};
CplaneBatchTask* CplaneBatchTask::s_current = nullptr;

// Table-driven: process and complete dispatch counts and return value for
// varying cell counts, using the SCOPED_TRACE pattern from DlAggrTask/UlAggrTask.
TEST_F(CplaneBatchTask, DispatchCountsAndReturn)
{
    struct TestCase final {
        std::string_view description;
        uint8_t          n_cells;
        int              expected_process;
    };
    static constexpr std::array CASES = {
        TestCase{"zero_cells_skips_process",                                  0,                              0},
        TestCase{"single_cell",                                               1,                              1},
        TestCase{"three_cells",                                               3,                              3},
        TestCase{"max_cells", static_cast<uint8_t>(nv::CPLANE_BATCH_CELL_LIMIT),
                                                  static_cast<int>(nv::CPLANE_BATCH_CELL_LIMIT)},
    };

    // Sequential cell IDs; only the first n_cells of each case are used.
    static constexpr auto s_cells = []() {
        std::array<uint8_t, nv::CPLANE_BATCH_CELL_LIMIT> a{};
        for (uint8_t i = 0; i < nv::CPLANE_BATCH_CELL_LIMIT; ++i) { a[i] = i; }
        return a;
    }();

    auto process  = [](nv::PHY_module*, uint32_t, uint32_t, std::size_t) -> int { ++s_current->process_count; return 0; };
    auto complete = [](nv::PHY_module*, uint32_t, uint32_t) { ++s_current->complete_count; };

    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        process_count = 0;
        complete_count = 0;
        auto arg = make_cplane_arg(0, 0,
                                   tc.n_cells > 0 ? s_cells.data() : nullptr,
                                   tc.n_cells, process, complete);
        EXPECT_EQ(task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0), 0);
        EXPECT_EQ(process_count,  tc.expected_process);
        EXPECT_EQ(complete_count, 1);
    }
}

// Verifies that process_cell is called in cell_ids[] order and receives ring_idx.
TEST_F(CplaneBatchTask, ProcessCellOrderAndRingIdx)
{
    const uint8_t cells[] = {5, 12, 23};
    auto process = [](nv::PHY_module*, uint32_t ring_idx, uint32_t cell_id, std::size_t) -> int {
        s_current->seen_ring_idx = ring_idx;
        s_current->seen_cells.push_back(cell_id);
        return 0;
    };
    auto complete = [](nv::PHY_module*, uint32_t, uint32_t) {};

    auto arg = make_cplane_arg(2, make_slot_u32(1, 4), cells, 3, process, complete);
    EXPECT_EQ(task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0), 0);

    ASSERT_EQ(seen_cells.size(), 3U);
    EXPECT_EQ(seen_cells[0],  5U);
    EXPECT_EQ(seen_cells[1], 12U);
    EXPECT_EQ(seen_cells[2], 23U);
    EXPECT_EQ(seen_ring_idx, 2U);
}

// Verifies that on_complete is called exactly once with the correct ring_idx
// and slot_u32 arguments.
TEST_F(CplaneBatchTask, CompleteCallbackReceivesCorrectArgs)
{
    const uint8_t cells[] = {1, 2};
    auto process  = [](nv::PHY_module*, uint32_t, uint32_t, std::size_t) -> int { return 0; };
    auto complete = [](nv::PHY_module*, uint32_t ring_idx, uint32_t slot_u32) {
        ++s_current->complete_count;
        s_current->complete_ring_idx = ring_idx;
        s_current->complete_slot_u32 = slot_u32;
    };

    const uint32_t expected_slot = make_slot_u32(3, 7);
    auto arg = make_cplane_arg(1, expected_slot, cells, 2, process, complete);
    (void)task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0);

    EXPECT_EQ(complete_count,    1);
    EXPECT_EQ(complete_ring_idx, 1U);
    EXPECT_EQ(complete_slot_u32, expected_slot);
}

TEST_F(CplaneBatchTask, PostBatchCallbackReceivesMetadata)
{
    const uint8_t cells[] = {1, 5, 9};
    int slot_map_marker = 0x1234;

    auto process = [](nv::PHY_module*, uint32_t, uint32_t, std::size_t) -> int { return 0; };
    auto complete = [](nv::PHY_module*, uint32_t, uint32_t) {};
    auto post_batch = [](void* slot_map, uint8_t batch_id, uint8_t total_batches, uint8_t n_cells_in_batch, nv::CplaneBatchDirection direction) {
        ++s_current->post_batch_count;
        s_current->seen_slot_map = slot_map;
        s_current->seen_batch_id = batch_id;
        s_current->seen_total_batches = total_batches;
        s_current->seen_n_cells = n_cells_in_batch;
        s_current->seen_direction = direction;
    };

    auto arg = make_cplane_arg(0, make_slot_u32(2, 3), cells, 3,
                               process, complete,
                               &slot_map_marker, post_batch, 2, 4, nv::CplaneBatchDirection::UPLINK);
    EXPECT_EQ(task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0), 0);

    EXPECT_EQ(post_batch_count, 1);
    EXPECT_EQ(seen_slot_map, &slot_map_marker);
    EXPECT_EQ(seen_batch_id, 2U);
    EXPECT_EQ(seen_total_batches, 4U);
    EXPECT_EQ(seen_n_cells, 3U);
    EXPECT_EQ(seen_direction, nv::CplaneBatchDirection::UPLINK);
}

// Verifies that on_batch_error is NOT invoked when every process_cell returns 0.
TEST_F(CplaneBatchTask, OnBatchErrorNotCalledOnAllSuccess)
{
    const uint8_t cells[] = {7, 11, 19};
    int slot_map_marker = 0x4242;

    auto process = [](nv::PHY_module*, uint32_t, uint32_t, std::size_t) -> int { return 0; };
    auto complete = [](nv::PHY_module*, uint32_t, uint32_t) {};
    auto on_batch_error = [](void*, std::span<const uint32_t>) {
        ++s_current->on_batch_error_count;
    };

    auto arg = make_cplane_arg(0, make_slot_u32(2, 3), cells, 3,
                               process, complete,
                               &slot_map_marker, nullptr, 0, 1,
                               nv::CplaneBatchDirection::UPLINK, 0, on_batch_error);
    EXPECT_EQ(task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0), 0);

    EXPECT_EQ(on_batch_error_count, 0);
}

// Verifies that on_batch_error is invoked exactly once with the failing cell IDs
// (in batch order) when a subset of process_cell invocations return non-zero.
TEST_F(CplaneBatchTask, OnBatchErrorCalledWithFailingCellIds)
{
    const uint8_t cells[] = {3, 8, 14, 21};
    int slot_map_marker = 0xABCD;

    // Fail cells 8 and 21 (indices 1 and 3); succeed for 3 and 14.
    auto process = [](nv::PHY_module*, uint32_t, uint32_t cell_id, std::size_t) -> int {
        return (cell_id == 8u || cell_id == 21u) ? 2 /* SEND_*_CPLANE_TIMING_ERROR */ : 0;
    };
    auto complete = [](nv::PHY_module*, uint32_t, uint32_t) {};
    auto on_batch_error = [](void* slot_map, std::span<const uint32_t> err_cell_ids) {
        ++s_current->on_batch_error_count;
        s_current->seen_err_slot_map = slot_map;
        s_current->seen_err_count    = static_cast<int>(err_cell_ids.size());
        for (const auto cid : err_cell_ids) {
            s_current->seen_err_cell_ids.push_back(cid);
        }
    };

    auto arg = make_cplane_arg(0, make_slot_u32(2, 3), cells, 4,
                               process, complete,
                               &slot_map_marker, nullptr, 0, 1,
                               nv::CplaneBatchDirection::UPLINK, 0, on_batch_error);
    EXPECT_EQ(task_work_fn_cplane_batch(nullptr, &arg, 0, 0, 0), 0);

    EXPECT_EQ(on_batch_error_count, 1);
    EXPECT_EQ(seen_err_slot_map,    &slot_map_marker);
    EXPECT_EQ(seen_err_count,       2);
    ASSERT_EQ(seen_err_cell_ids.size(), 2U);
    EXPECT_EQ(seen_err_cell_ids[0],  8U);
    EXPECT_EQ(seen_err_cell_ids[1], 21U);
}

// ===========================================================================
// Group D — each task_work_fn_aggr_<dl_channel> dispatches to the matching
// DlChannelDispatch::process_<channel> callback and invokes on_complete once.
// Covers PDSCH, CSI-RS, PDCCH, SSB, DL-BFW.
// ===========================================================================

namespace {

/// Builds a DlAggrTaskArg with a fully specified DlChannelDispatch.
/// phy_module is set to nullptr — callbacks must not dereference it.
nv::DlAggrTaskArg make_dl_aggr_arg(const nv::DlChannelDispatch& dispatch,
                                   uint32_t ring_idx = 1,
                                   uint32_t slot_u32 = 0x000A'0005u)
{
    nv::DlAggrTaskArg arg{};
    arg.phy_module    = nullptr;
    arg.dispatch      = &dispatch;
    arg.slot_u32      = slot_u32;
    arg.ring_idx      = ring_idx;
    arg.active_ch_mask = 0;
    arg.slot_cmd_idx  = 0;
    arg.slot_type_mask = nv::SlotTypeMask::DL;
    return arg;
}

/// Minimal complete_fn that simply increments a counter.
struct CallTracker
{
    int process_called{0};
    int complete_called{0};

    void reset() { process_called = 0; complete_called = 0; }
};

} // namespace

TEST(DlAggrTask, TableDriven)
{
    using DlProcessField = nv::dl_channel_fn_t nv::DlChannelDispatch::*;
    using WorkFn         = int(*)(Worker*, void*, int, int, int);

    struct TestCase final {
        std::string_view description;
        WorkFn           work_fn;
        DlProcessField   active_field;
    };
    static constexpr std::array CASES = {
        TestCase{"pdsch", task_work_fn_aggr_pdsch, &nv::DlChannelDispatch::process_pdsch},
        TestCase{"csirs", task_work_fn_aggr_csirs, &nv::DlChannelDispatch::process_csirs},
        TestCase{"pdcch", task_work_fn_aggr_pdcch, &nv::DlChannelDispatch::process_pdcch},
        TestCase{"ssb",   task_work_fn_aggr_ssb,   &nv::DlChannelDispatch::process_ssb},
        TestCase{"dlbfw", task_work_fn_aggr_dlbfw, &nv::DlChannelDispatch::process_dlbfw},
    };

    static CallTracker tracker;
    static constexpr auto noop_dl = [](nv::PHY_module*, const nv::DlAggrTaskArg&) {};

    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        tracker.reset();
        nv::DlChannelDispatch dispatch{};
        dispatch.process_pdsch = noop_dl;
        dispatch.process_csirs = noop_dl;
        dispatch.process_pdcch = noop_dl;
        dispatch.process_ssb   = noop_dl;
        dispatch.process_dlbfw = noop_dl;
        dispatch.on_complete   = [](nv::PHY_module*, uint32_t, uint32_t) { tracker.complete_called++; };
        dispatch.*tc.active_field = [](nv::PHY_module*, const nv::DlAggrTaskArg&) { tracker.process_called++; };

        auto arg = make_dl_aggr_arg(dispatch);
        EXPECT_EQ(tc.work_fn(nullptr, &arg, 0, 0, 0), 0);
        EXPECT_EQ(tracker.process_called, 1);
        EXPECT_EQ(tracker.complete_called, 1);
    }
}

// ===========================================================================
// Group D2 — DL aggr staged-pointer propagation & guard logic
//
// Verifies that DlAggrTaskArg correctly carries split-phase TX_DATA
// fields through the dispatch path.
// ===========================================================================

TEST(DlAggrTask, StagedFieldsDefaultToNull)
{
    nv::DlAggrTaskArg arg{};
    EXPECT_EQ(arg.staged_tb_ptrs, nullptr);
    EXPECT_EQ(arg.n_staged_tb_ptrs, 0U);
    EXPECT_EQ(arg.staged_tx_msg_bufs, nullptr);
}

TEST(DlAggrTask, StagedPointersPropagatedToPdschDispatch)
{
    static const nv::DlAggrTaskArg* captured = nullptr;

    nv::DlChannelDispatch dispatch{};
    constexpr auto noop_dl = [](nv::PHY_module*, const nv::DlAggrTaskArg&) {};
    dispatch.process_pdsch = [](nv::PHY_module*, const nv::DlAggrTaskArg& a) { captured = &a; };
    dispatch.process_csirs = noop_dl;
    dispatch.process_pdcch = noop_dl;
    dispatch.process_ssb   = noop_dl;
    dispatch.process_dlbfw = noop_dl;
    dispatch.on_complete   = [](nv::PHY_module*, uint32_t, uint32_t) {};

    alignas(8) uint8_t dummy_buf[4]{};
    uint8_t* staged_ptrs[2] = {dummy_buf, nullptr};
    const void* msg_bufs[2] = {dummy_buf, nullptr};

    auto arg = make_dl_aggr_arg(dispatch);
    arg.staged_tb_ptrs     = staged_ptrs;
    arg.n_staged_tb_ptrs   = 2;
    arg.staged_tx_msg_bufs = msg_bufs;

    captured = nullptr;
    EXPECT_EQ(task_work_fn_aggr_pdsch(nullptr, &arg, 0, 0, 0), 0);
    ASSERT_NE(captured, nullptr);
    EXPECT_EQ(captured->staged_tb_ptrs, staged_ptrs);
    EXPECT_EQ(captured->n_staged_tb_ptrs, 2U);
    EXPECT_EQ(captured->staged_tx_msg_bufs, msg_bufs);
}

TEST(DlAggrTask, NullStagedPtrs_StillDispatches)
{
    static int pdsch_calls = 0;
    pdsch_calls = 0;

    nv::DlChannelDispatch dispatch{};
    constexpr auto noop_dl = [](nv::PHY_module*, const nv::DlAggrTaskArg&) {};
    dispatch.process_pdsch = [](nv::PHY_module*, const nv::DlAggrTaskArg& a) {
        ++pdsch_calls;
        EXPECT_EQ(a.staged_tb_ptrs, nullptr);
        EXPECT_EQ(a.staged_tx_msg_bufs, nullptr);
    };
    dispatch.process_csirs = noop_dl;
    dispatch.process_pdcch = noop_dl;
    dispatch.process_ssb   = noop_dl;
    dispatch.process_dlbfw = noop_dl;
    dispatch.on_complete   = [](nv::PHY_module*, uint32_t, uint32_t) {};

    auto arg = make_dl_aggr_arg(dispatch);
    // staged_tb_ptrs and staged_tx_msg_bufs default to nullptr
    EXPECT_EQ(task_work_fn_aggr_pdsch(nullptr, &arg, 0, 0, 0), 0);
    EXPECT_EQ(pdsch_calls, 1);
}

TEST(DlAggrTask, StagedMsgBufsNullWhileTbPtrsSet_ValidCombination)
{
    static const nv::DlAggrTaskArg* captured = nullptr;

    nv::DlChannelDispatch dispatch{};
    constexpr auto noop_dl = [](nv::PHY_module*, const nv::DlAggrTaskArg&) {};
    dispatch.process_pdsch = [](nv::PHY_module*, const nv::DlAggrTaskArg& a) { captured = &a; };
    dispatch.process_csirs = noop_dl;
    dispatch.process_pdcch = noop_dl;
    dispatch.process_ssb   = noop_dl;
    dispatch.process_dlbfw = noop_dl;
    dispatch.on_complete   = [](nv::PHY_module*, uint32_t, uint32_t) {};

    alignas(8) uint8_t dummy_buf[4]{};
    uint8_t* staged_ptrs[1] = {dummy_buf};

    auto arg = make_dl_aggr_arg(dispatch);
    arg.staged_tb_ptrs     = staged_ptrs;
    arg.n_staged_tb_ptrs   = 1;
    arg.staged_tx_msg_bufs = nullptr; // tbStartOffset override skipped

    captured = nullptr;
    EXPECT_EQ(task_work_fn_aggr_pdsch(nullptr, &arg, 0, 0, 0), 0);
    ASSERT_NE(captured, nullptr);
    EXPECT_NE(captured->staged_tb_ptrs, nullptr);
    EXPECT_EQ(captured->staged_tx_msg_bufs, nullptr);
}

// ===========================================================================
// Group E — UL channel aggregation worker-wrapper dispatch.
// UL_BFW is aggregated by a dedicated worker task (task_aggr_ulbfw), pushed by
// PHY_module::enqueue_channel_tasks() when a UL_BFW lane is present;
// process_aggr_ulbfw_channel() populates slot_command_array from the worker
// thread. The serial on_msg replay path has been removed. The ulbfw case here
// verifies the low-level worker wrapper dispatch.
// ===========================================================================

namespace {

nv::UlAggrTaskArg make_ul_aggr_arg(const nv::UlChannelDispatch& dispatch,
                                   uint32_t ring_idx = 0,
                                   uint32_t slot_u32 = 0x0005'0003u)
{
    nv::UlAggrTaskArg arg{};
    arg.phy_module        = nullptr;
    arg.dispatch          = &dispatch;
    arg.slot_u32          = slot_u32;
    arg.ring_idx          = ring_idx;
    arg.active_ul_ch_mask = 0;
    arg.slot_cmd_idx      = 0;
    arg.slot_type_mask    = nv::SlotTypeMask::UL;
    return arg;
}

} // namespace

TEST(UlAggrTask, TableDriven)
{
    using UlProcessField = nv::ul_channel_fn_t nv::UlChannelDispatch::*;
    using WorkFn         = int(*)(Worker*, void*, int, int, int);

    struct TestCase final {
        std::string_view description;
        WorkFn           work_fn;
        UlProcessField   active_field;
    };
    static constexpr std::array CASES = {
        TestCase{"pusch", task_work_fn_aggr_pusch, &nv::UlChannelDispatch::process_pusch},
        TestCase{"prach", task_work_fn_aggr_prach, &nv::UlChannelDispatch::process_prach},
        TestCase{"pucch", task_work_fn_aggr_pucch, &nv::UlChannelDispatch::process_pucch},
        TestCase{"srs",   task_work_fn_aggr_srs,   &nv::UlChannelDispatch::process_srs},
        TestCase{"ulbfw", task_work_fn_aggr_ulbfw, &nv::UlChannelDispatch::process_ulbfw},
    };

    static CallTracker tracker;
    static constexpr auto noop_ul = [](nv::PHY_module*, const nv::UlAggrTaskArg&) {};

    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        tracker.reset();
        nv::UlChannelDispatch dispatch{};
        dispatch.process_pusch = noop_ul;
        dispatch.process_prach = noop_ul;
        dispatch.process_pucch = noop_ul;
        dispatch.process_srs   = noop_ul;
        dispatch.process_ulbfw = noop_ul;
        dispatch.on_complete   = [](nv::PHY_module*, uint32_t, uint32_t) { tracker.complete_called++; };
        dispatch.*tc.active_field = [](nv::PHY_module*, const nv::UlAggrTaskArg&) { tracker.process_called++; };

        auto arg = make_ul_aggr_arg(dispatch);
        EXPECT_EQ(tc.work_fn(nullptr, &arg, 0, 0, 0), 0);
        EXPECT_EQ(tracker.process_called, 1);
        EXPECT_EQ(tracker.complete_called, 1);
    }
}

TEST(UlAggrTask, SrsDispatchCarriesSlotMetadata)
{
    struct SrsTracker final
    {
        int process_called{0};
        int complete_called{0};
        uint32_t seen_ring_idx{0xFFFFu};
        uint32_t seen_slot_u32{0u};
        uint32_t seen_slot_cmd_idx{0xFFFFu};
        nv::SlotTypeMask seen_slot_type_mask{nv::SlotTypeMask::NONE};
        uint32_t complete_ring_idx{0xFFFFu};
        uint32_t complete_slot_u32{0u};
    };

    // `dispatch.process_srs` and `dispatch.on_complete` are raw function
    // pointers (`ul_channel_fn_t` / `task_complete_fn_t` in nv_ul_aggr_tasks.hpp).
    // Only captureless lambdas decay to a function pointer, so the lambdas
    // below must reference `tracker` by name — which requires `tracker` to
    // have static storage. The `= {}` reset is therefore necessary for
    // re-entry safety under --gtest_repeat (NSDMI runs only at static init).
    static SrsTracker tracker;
    tracker = {};

    nv::UlChannelDispatch dispatch{};
    static constexpr auto noop_ul = [](nv::PHY_module*, const nv::UlAggrTaskArg&) {};
    dispatch.process_pusch = noop_ul;
    dispatch.process_prach = noop_ul;
    dispatch.process_pucch = noop_ul;
    dispatch.process_ulbfw = noop_ul;
    dispatch.process_srs = [](nv::PHY_module*, const nv::UlAggrTaskArg& arg) {
        ++tracker.process_called;
        tracker.seen_ring_idx = arg.ring_idx;
        tracker.seen_slot_u32 = arg.slot_u32;
        tracker.seen_slot_cmd_idx = arg.slot_cmd_idx;
        tracker.seen_slot_type_mask = arg.slot_type_mask;
    };
    dispatch.on_complete = [](nv::PHY_module*, uint32_t ring_idx, uint32_t slot_u32) {
        ++tracker.complete_called;
        tracker.complete_ring_idx = ring_idx;
        tracker.complete_slot_u32 = slot_u32;
    };

    auto arg = make_ul_aggr_arg(dispatch, 2u, make_slot_u32(9u, 13u));
    arg.slot_cmd_idx = 17u;
    arg.slot_type_mask = nv::SlotTypeMask::UL;

    EXPECT_EQ(task_work_fn_aggr_srs(nullptr, &arg, 0, 0, 0), 0);
    EXPECT_EQ(tracker.process_called, 1);
    EXPECT_EQ(tracker.complete_called, 1);
    EXPECT_EQ(tracker.seen_ring_idx, 2u);
    EXPECT_EQ(tracker.seen_slot_u32, make_slot_u32(9u, 13u));
    EXPECT_EQ(tracker.seen_slot_cmd_idx, 17u);
    EXPECT_EQ(tracker.seen_slot_type_mask, nv::SlotTypeMask::UL);
    EXPECT_EQ(tracker.complete_ring_idx, 2u);
    EXPECT_EQ(tracker.complete_slot_u32, make_slot_u32(9u, 13u));
}

// ===========================================================================
// Group F — store_eom_bitmap accumulation and EOM-trigger logic
// ===========================================================================

// These tests exercise the bitmap accumulation logic directly on SlotTaskPool,
// mirroring the in-production pattern:
//   task_pool_.store_eom_bitmap[ring_idx] |= (uint64_t{1} << cell_id)
//   eom_trigger = (bitmap & active) == active  (active != 0 guard)

TEST(StoreEomBitmap, SingleCellSetsOneBit)
{
    nv::SlotTaskPool pool{};
    const uint32_t ring_idx = 2;
    const uint32_t cell_id  = 5;

    pool.store_eom_bitmap[ring_idx] |= (uint64_t{1} << cell_id);

    EXPECT_EQ(pool.store_eom_bitmap[ring_idx], uint64_t{1} << cell_id);
    // No other ring slots affected
    EXPECT_EQ(pool.store_eom_bitmap[0], 0U);
    EXPECT_EQ(pool.store_eom_bitmap[1], 0U);
}

TEST(StoreEomBitmap, MultipleCellsAccumulate)
{
    nv::SlotTaskPool pool{};
    const uint32_t ring_idx = 0;

    pool.store_eom_bitmap[ring_idx] |= uint64_t{1} << 0;
    EXPECT_EQ(pool.store_eom_bitmap[ring_idx], 0x1U);

    pool.store_eom_bitmap[ring_idx] |= uint64_t{1} << 1;
    EXPECT_EQ(pool.store_eom_bitmap[ring_idx], 0x3U);

    pool.store_eom_bitmap[ring_idx] |= uint64_t{1} << 2;
    EXPECT_EQ(pool.store_eom_bitmap[ring_idx], 0x7U);
}

TEST(StoreEomBitmap, EomTrigger)
{
    struct TestCase final {
        std::string_view description;
        uint32_t         n_cells_set;
        uint32_t         n_active_cells;
        bool             expected_eom;
    };
    static constexpr std::array CASES = {
        TestCase{"all_59_cells_fires_eom",   59, 59, true},
        TestCase{"58_of_59_no_spurious_eom", 58, 59, false},
        TestCase{"single_cell_active_fires",  1,  1, true},
        TestCase{"zero_active_never_fires",   0,  0, false},
    };
    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        nv::SlotTaskPool pool{};
        const uint64_t active = tc.n_active_cells == 0
            ? uint64_t{0}
            : tc.n_active_cells == 64 ? ~uint64_t{0} : (uint64_t{1} << tc.n_active_cells) - 1;
        for (uint32_t cell = 0; cell < tc.n_cells_set; ++cell)
        {
            pool.store_eom_bitmap[0] |= uint64_t{1} << cell;
        }
        const bool eom = active && ((pool.store_eom_bitmap[0] & active) == active);
        EXPECT_EQ(eom, tc.expected_eom);
    }
}

TEST(StoreEomBitmap, ResetClearsBitmap)
{
    nv::SlotTaskPool pool{};
    const uint32_t ring_idx = 2;

    for (uint32_t cell = 0; cell < 20; ++cell)
    {
        pool.store_eom_bitmap[ring_idx] |= uint64_t{1} << cell;
    }
    EXPECT_NE(pool.store_eom_bitmap[ring_idx], 0U);

    // Inline reset as done in recv_msg_to_store before run_slot_boundary_reset
    pool.store_eom_bitmap[ring_idx] = 0;

    EXPECT_EQ(pool.store_eom_bitmap[ring_idx], 0U);
}

// ===========================================================================
// Group G — Deferred TX_DATA release (release_lane / tx_data_deferred_ lifecycle)
// ===========================================================================

#include "nv_fapi_message_storage.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_tx_data_ring_buffer.hpp"
#include "scf_5g_fapi.h"
#include "tx_data_ownership_tracker.hpp"

namespace {

// CountingReleaseTransport / DeletingReleaseTransport are shared with
// test_tx_data_ownership.cpp; they live in tx_data_ownership_tracker.hpp to avoid
// duplicate definitions drifting out of sync.
using nv::test::CountingReleaseTransport;
using nv::test::DeletingReleaseTransport;

nv::phy_mac_msg_desc make_task_msg(uint8_t msg_id, uint16_t cell_id)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = msg_id;
    msg.cell_id = cell_id;
    return msg;
}

} // namespace

// G1. release_stored_slot_messages skips TX_DATA when deferred flag is true
// (simulated via per-lane release pattern).
TEST(DeferredTxDataRelease, PerLaneReleaseSkipsTxDataWhenDeferred)
{
    nv::FapiSlotMessageStorage store(16);
    store.reset_for_slot(0x000A'0005u);

    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_DL_TTI_REQUEST, 0)),    nv::StoreResult::Stored);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_UL_TTI_REQUEST, 0)),    nv::StoreResult::Stored);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_UL_DCI_REQUEST, 0)),    nv::StoreResult::Stored);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_TX_DATA_REQUEST, 0)),   nv::StoreResult::Stored);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_DL_BFW_CVI_REQUEST, 0)),nv::StoreResult::Stored);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_UL_BFW_CVI_REQUEST, 0)),nv::StoreResult::Stored);

    using MT = nv::FapiSlotMessageStorage::MsgType;
    nv::NullFapiTransport tw;

    // Simulate release_stored_slot_messages with tx_data_deferred_ = true:
    // release all lanes except TX_DATA
    store.release_lane(MT::DL_TTI,  tw);
    store.release_lane(MT::UL_TTI,  tw);
    store.release_lane(MT::UL_DCI,  tw);
    store.release_lane(MT::DL_BFW,  tw);
    store.release_lane(MT::UL_BFW,  tw);

    EXPECT_EQ(store.dl_tti_count(),  0u);
    EXPECT_EQ(store.ul_tti_count(),  0u);
    EXPECT_EQ(store.ul_dci_count(),  0u);
    EXPECT_EQ(store.dl_bfw_count(),  0u);
    EXPECT_EQ(store.ul_bfw_count(),  0u);
    EXPECT_EQ(store.tx_data_count(), 1u);  // TX_DATA still intact

    // Simulate cuphydriver callback: release_deferred_tx_data
    store.release_lane(MT::TX_DATA, tw);
    EXPECT_EQ(store.tx_data_count(), 0u);
}

// G2. tx_data_deferred_ flag lifecycle: set, skip, callback clears.
TEST(DeferredTxDataRelease, DeferredFlagLifecycle)
{
    std::array<bool, SLOT_STORAGE_DEPTH> tx_data_deferred{};
    const uint32_t ring_idx = 1;

    // Initially false
    EXPECT_FALSE(tx_data_deferred[ring_idx]);

    // PDSCH + split-phase active -> set
    tx_data_deferred[ring_idx] = true;
    EXPECT_TRUE(tx_data_deferred[ring_idx]);

    // Callback fires -> clear
    tx_data_deferred[ring_idx] = false;
    EXPECT_FALSE(tx_data_deferred[ring_idx]);
}

// G3. Strict callback gate: second release_lane after atomic claim is a no-op.
TEST(DeferredTxDataRelease, DuplicateReleaseLaneIsNoOpAfterFirstClaim)
{
    nv::FapiSlotMessageStorage store(16);
    store.reset_for_slot(0x000B'0003u);

    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_TX_DATA_REQUEST, 0)), nv::StoreResult::Stored);
    EXPECT_EQ(store.tx_data_count(), 1u);

    DeletingReleaseTransport tw{};

    store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
    EXPECT_EQ(store.tx_data_count(), 0u);
    EXPECT_EQ(tw.delete_count.load(), 1);

    store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
    EXPECT_EQ(store.tx_data_count(), 0u);
    EXPECT_EQ(tw.delete_count.load(), 1);
}

// G4. Deferred + inline race using CountingReleaseTransport (TSan-friendly).
TEST(DeferredTxDataRelease, ConcurrentInlineAndCallbackReleaseOnce)
{
    nv::FapiSlotMessageStorage store(16);
    store.reset_for_slot(0x000C'0004u);
    EXPECT_EQ(store.store_message(make_task_msg(SCF_FAPI_TX_DATA_REQUEST, 0)), nv::StoreResult::Stored);

    nv::TxDataRingBuffer ring{};
    ring.set_deferred(0U, true);

    CountingReleaseTransport tw{};
    std::atomic_bool          callback_done{false};

    std::jthread callback_thread([&ring, &store, &tw, &callback_done] {
        if(ring.is_deferred(0U))
        {
            store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
            ring.set_deferred(0U, false);
            ring.reset(0U);
        }
        callback_done.store(true, std::memory_order_release);
    });

    std::jthread inline_thread([&ring, &store, &tw, &callback_done] {
        while(!callback_done.load(std::memory_order_acquire))
        {
            if(!ring.is_deferred(0U))
            {
                store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
                break;
            }
        }
    });

    // Join before asserting so release_count reflects the completed race, not a
    // premature read. release_lane claims the lane atomically, so exactly one of the
    // two threads frees the buffers; the deferred callback always wins the claim
    // (deferred starts true, only the callback clears it), so the release count is
    // exactly 1 — EXPECT_EQ, not EXPECT_LE, which would also pass on a silently-
    // missing release.
    callback_thread.join();
    inline_thread.join();

    EXPECT_EQ(tw.release_count.load(), 1);
    EXPECT_EQ(store.tx_data_count(), 0u);
}
