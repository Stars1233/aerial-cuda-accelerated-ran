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
 * @file test_nvphy_non_slot_dispatch.cpp
 * @brief Full-path tests for the low-priority non-slot dispatch substrate.
 */

#include <gtest/gtest.h>

#include "memtrace.h"
#include "nv_phy_non_slot_dispatch.hpp"
#include "scf_5g_fapi.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <thread>

namespace {

using namespace std::chrono_literals;

struct MessageRecord final {
    std::atomic<int32_t> msg_id{-1};
    std::atomic<int32_t> cell_id{-1};
};

struct FakeRuntimeState final {
    static constexpr std::size_t kMaxMessageRecords = nv::NonSlotLpExecutor::kQueueCapacity + 4;

    std::size_t cell_count{2};
    std::atomic<std::size_t> process_count{0};
    std::atomic<int32_t> last_processed_msg_id{-1};
    std::atomic<int32_t> last_processed_cell_id{-1};
    std::atomic<std::size_t> next_record{0};
    std::atomic<std::size_t> next_processed_record{0};
    std::atomic<std::size_t> release_count{0};
    std::array<MessageRecord, kMaxMessageRecords> released{};
    std::array<MessageRecord, kMaxMessageRecords> processed{};
};

[[nodiscard]] std::size_t fake_cell_count(void* context) noexcept
{
    const auto* state = static_cast<const FakeRuntimeState*>(context);
    return state != nullptr ? state->cell_count : 0;
}

void fake_rx_release(void* context, nv::phy_mac_msg_desc& msg) noexcept
{
    auto* state = static_cast<FakeRuntimeState*>(context);
    if (state == nullptr)
    {
        return;
    }

    const auto index = state->next_record.fetch_add(std::size_t{1}, std::memory_order_acq_rel);
    if (index < state->released.size())
    {
        state->released[index].msg_id.store(msg.msg_id, std::memory_order_release);
        state->released[index].cell_id.store(msg.cell_id, std::memory_order_release);
    }
    state->release_count.fetch_add(std::size_t{1}, std::memory_order_release);
}

void fake_process_item(void* context, std::uint16_t cell_id, nv::phy_mac_msg_desc& msg) noexcept
{
    auto* state = static_cast<FakeRuntimeState*>(context);
    if (state == nullptr)
    {
        return;
    }

    state->last_processed_msg_id.store(msg.msg_id, std::memory_order_release);
    state->last_processed_cell_id.store(static_cast<int32_t>(cell_id), std::memory_order_release);
    const auto index = state->next_processed_record.fetch_add(std::size_t{1}, std::memory_order_acq_rel);
    if (index < state->processed.size())
    {
        state->processed[index].msg_id.store(msg.msg_id, std::memory_order_release);
        state->processed[index].cell_id.store(static_cast<int32_t>(cell_id), std::memory_order_release);
    }
    state->process_count.fetch_add(std::size_t{1}, std::memory_order_release);
}

[[nodiscard]] nv::NonSlotLpRuntime make_runtime(FakeRuntimeState& state) noexcept
{
    return nv::NonSlotLpRuntime{&state, fake_cell_count, fake_process_item, fake_rx_release};
}

[[nodiscard]] nv::NonSlotLpThreadConfig make_config()
{
    nv::NonSlotLpThreadConfig config{};
    config.worker_cfg.name = "nonslot_lp";
    config.worker_cfg.cpu_affinity = -1;
    config.worker_cfg.sched_priority = 0;
    return config;
}

[[nodiscard]] nv::phy_mac_msg_desc make_msg(int32_t msg_id, int32_t cell_id, void* msg_buf)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id = msg_id;
    msg.cell_id = cell_id;
    msg.msg_len = 16;
    msg.msg_buf = msg_buf;
    return msg;
}

[[nodiscard]] bool wait_for_releases(const FakeRuntimeState& state,
                                     std::size_t expected,
                                     std::chrono::milliseconds timeout = 500ms)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (state.release_count.load(std::memory_order_acquire) >= expected)
        {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return state.release_count.load(std::memory_order_acquire) >= expected;
}

} // namespace

TEST(NonSlotLpExecutor, MoveContractRunningWorkerViolatesIsRunningPrecondition)
{
    // move_nonslot_lp_worker_from() requires !is_running(). Verify that the
    // state transitions correctly so the precondition is satisfiable before move.
    FakeRuntimeState state{};
    state.cell_count = 1;
    nv::NonSlotLpExecutor executor(make_runtime(state), make_config());

    EXPECT_FALSE(executor.is_running()); // safe to move: precondition satisfied

    executor.start();
    EXPECT_TRUE(executor.is_running()); // would violate !is_running() precondition

    executor.stop();
    EXPECT_FALSE(executor.is_running()); // safe to move again
}

TEST(NonSlotLpExecutor, MoveContractQueuedWorkViolatesDepthPrecondition)
{
    // move_nonslot_lp_worker_from() requires depth()==0. Verify that enqueuing
    // without a running worker raises depth() and that stop() restores depth()==0. 
    FakeRuntimeState state{};
    state.cell_count = 1;
    nv::NonSlotLpExecutor executor(make_runtime(state), make_config());

    EXPECT_EQ(executor.depth(), std::size_t{0}); // safe to move: precondition satisfied

    std::array<std::byte, 16> buf{};
    auto msg = make_msg(SCF_FAPI_START_REQUEST, 0, buf.data());
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    const bool enqueued = executor.enqueue(msg);
    memtrace_set_config(0);
    ASSERT_TRUE(enqueued);
    EXPECT_GT(executor.depth(), std::size_t{0}); // would violate depth()==0 precondition

    executor.stop(); // drains the queue via rx_release
    EXPECT_EQ(executor.depth(), std::size_t{0}); // safe to move again
}

TEST(NonSlotLpExecutor, TableDrivenFullPathAdmissionWorkerReleaseAndReject)
{
    struct TestCase final {
        std::string_view description{};
        int32_t msg_id{};
        int32_t cell_id{};
        std::size_t cell_count{};
        bool expect_accepted{};
        bool expect_worker_release{};
    };

    static constexpr std::array CASES = {
        TestCase{"start_valid_cell", SCF_FAPI_START_REQUEST, 1, std::size_t{2}, true, true},
        TestCase{"stop_valid_cell", SCF_FAPI_STOP_REQUEST, 1, std::size_t{2}, true, true},
        TestCase{"config_valid_cell", SCF_FAPI_CONFIG_REQUEST, 0, std::size_t{2}, true, true},
        TestCase{"unsupported_msg_id", 0x7f, 0, std::size_t{2}, false, false},
        TestCase{"negative_cell_id", SCF_FAPI_START_REQUEST, -1, std::size_t{2}, false, false},
        TestCase{"out_of_range_cell_id", SCF_FAPI_START_REQUEST, 2, std::size_t{2}, false, false},
    };

    for (const auto& tc : CASES)
    {
        SCOPED_TRACE(tc.description);
        FakeRuntimeState state{};
        state.cell_count = tc.cell_count;
        nv::NonSlotLpExecutor executor(make_runtime(state), make_config());
        executor.start();
        executor.start();

        std::array<std::byte, 16> msg_storage{};
        auto msg = make_msg(tc.msg_id, tc.cell_id, msg_storage.data());

        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
        const bool accepted = executor.enqueue(msg);
        memtrace_set_config(0);
        EXPECT_EQ(accepted, tc.expect_accepted);
        if (tc.expect_accepted)
        {
            EXPECT_EQ(msg.msg_buf, nullptr);
            if (tc.expect_worker_release)
            {
                ASSERT_TRUE(wait_for_releases(state, std::size_t{1}));
                EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{1});
                EXPECT_EQ(state.released[0].msg_id.load(std::memory_order_acquire), tc.msg_id);
                EXPECT_EQ(state.released[0].cell_id.load(std::memory_order_acquire), tc.cell_id);
                EXPECT_EQ(state.process_count.load(std::memory_order_acquire), std::size_t{1});
                EXPECT_EQ(state.last_processed_msg_id.load(std::memory_order_acquire), tc.msg_id);
                EXPECT_EQ(state.last_processed_cell_id.load(std::memory_order_acquire), tc.cell_id);
            }
            else
            {
                EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{0});
            }
        }
        else
        {
            EXPECT_NE(msg.msg_buf, nullptr);
            EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{0});
        }

        executor.stop();
        executor.stop();
        EXPECT_EQ(executor.depth(), std::size_t{0});
    }
}

TEST(NonSlotLpExecutor, ConfigRequestRunsOnWorkerBeforeFollowingStart)
{
    FakeRuntimeState state{};
    state.cell_count = 1;
    nv::NonSlotLpExecutor executor(make_runtime(state), make_config());
    executor.start();

    std::array<std::byte, 16> config_storage{};
    std::array<std::byte, 16> start_storage{};
    auto config_msg = make_msg(SCF_FAPI_CONFIG_REQUEST, 0, config_storage.data());
    auto start_msg = make_msg(SCF_FAPI_START_REQUEST, 0, start_storage.data());

    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    const bool config_accepted = executor.enqueue(config_msg);
    const bool start_accepted = executor.enqueue(start_msg);
    memtrace_set_config(0);

    ASSERT_TRUE(config_accepted);
    ASSERT_TRUE(start_accepted);
    EXPECT_EQ(config_msg.msg_buf, nullptr);
    EXPECT_EQ(start_msg.msg_buf, nullptr);

    ASSERT_TRUE(wait_for_releases(state, std::size_t{2}));
    EXPECT_EQ(state.process_count.load(std::memory_order_acquire), std::size_t{2});

    EXPECT_EQ(state.processed[0].msg_id.load(std::memory_order_acquire), SCF_FAPI_CONFIG_REQUEST);
    EXPECT_EQ(state.processed[0].cell_id.load(std::memory_order_acquire), 0);
    EXPECT_EQ(state.processed[1].msg_id.load(std::memory_order_acquire), SCF_FAPI_START_REQUEST);
    EXPECT_EQ(state.processed[1].cell_id.load(std::memory_order_acquire), 0);

    EXPECT_EQ(state.released[0].msg_id.load(std::memory_order_acquire), SCF_FAPI_CONFIG_REQUEST);
    EXPECT_EQ(state.released[1].msg_id.load(std::memory_order_acquire), SCF_FAPI_START_REQUEST);

    executor.stop();
}

TEST(NonSlotLpExecutor, DuplicateConfigRetryIsReleasedWhileOriginalPending)
{
    FakeRuntimeState state{};
    state.cell_count = 1;
    nv::NonSlotLpExecutor executor(make_runtime(state), make_config());

    std::array<std::byte, 16> first_storage{};
    std::array<std::byte, 16> retry_storage{};
    auto first_config = make_msg(SCF_FAPI_CONFIG_REQUEST, 0, first_storage.data());
    auto retry_config = make_msg(SCF_FAPI_CONFIG_REQUEST, 0, retry_storage.data());

    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    const bool first_accepted = executor.enqueue(first_config);
    const bool retry_released = executor.enqueue(retry_config);
    memtrace_set_config(0);

    ASSERT_TRUE(first_accepted);
    ASSERT_TRUE(retry_released);
    EXPECT_EQ(first_config.msg_buf, nullptr);
    EXPECT_EQ(retry_config.msg_buf, nullptr);
    EXPECT_EQ(executor.depth(), std::size_t{1});
    EXPECT_EQ(state.process_count.load(std::memory_order_acquire), std::size_t{0});
    EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{1});
    EXPECT_EQ(state.released[0].msg_id.load(std::memory_order_acquire), SCF_FAPI_CONFIG_REQUEST);

    executor.stop();
    EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{2});

    std::array<std::byte, 16> next_storage{};
    auto next_config = make_msg(SCF_FAPI_CONFIG_REQUEST, 0, next_storage.data());
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    const bool next_accepted = executor.enqueue(next_config);
    memtrace_set_config(0);

    EXPECT_TRUE(next_accepted);
    EXPECT_EQ(next_config.msg_buf, nullptr);
    EXPECT_EQ(executor.depth(), std::size_t{1});

    executor.stop();
}

TEST(NonSlotLpExecutor, OverflowRejectsProducerOwnedDescriptorAndStopDrainsAcceptedWork)
{
    FakeRuntimeState state{};
    state.cell_count = 1;
    nv::NonSlotLpExecutor executor(make_runtime(state), make_config());
    std::array<std::array<std::byte, 16>, nv::NonSlotLpExecutor::kQueueCapacity + 1> msg_storage{};

    for (std::size_t i = 0; i < nv::NonSlotLpExecutor::kQueueCapacity; ++i)
    {
        auto msg = make_msg(SCF_FAPI_START_REQUEST, 0, msg_storage[i].data());
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
        const bool ok = executor.enqueue(msg);
        memtrace_set_config(0);
        ASSERT_TRUE(ok) << "enqueue index " << i;
        EXPECT_EQ(msg.msg_buf, nullptr);
    }

    EXPECT_EQ(executor.depth(), nv::NonSlotLpExecutor::kQueueCapacity);
    auto overflow_msg = make_msg(SCF_FAPI_START_REQUEST,
                                 0,
                                 msg_storage[nv::NonSlotLpExecutor::kQueueCapacity].data());
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    const bool overflow_rejected = executor.enqueue(overflow_msg);
    memtrace_set_config(0);
    EXPECT_FALSE(overflow_rejected);
    EXPECT_NE(overflow_msg.msg_buf, nullptr);
    EXPECT_EQ(state.release_count.load(std::memory_order_acquire), std::size_t{0});

    executor.stop();
    EXPECT_EQ(state.release_count.load(std::memory_order_acquire), nv::NonSlotLpExecutor::kQueueCapacity);
    EXPECT_EQ(executor.depth(), std::size_t{0});

    executor.start();
    executor.stop();
    EXPECT_EQ(state.release_count.load(std::memory_order_acquire), nv::NonSlotLpExecutor::kQueueCapacity);
}

// Routing membership: exactly CONFIG/START/STOP are owned by the nonslot_lp worker. Guards
// against the dispatch set silently changing -- a serial fallback branch reappearing, or a
// message being added to / dropped from the worker path. is_parallel_nonslot_msg() in
// nv_phy_slot_dispatch.cpp delegates to is_nonslot_lp_message(), so this also pins the
// routing gate, not just the executor-side filter.
TEST(NonSlotLpRouting, OnlyConfigStartStopAreWorkerOwned)
{
    EXPECT_TRUE(nv::is_nonslot_lp_message(SCF_FAPI_CONFIG_REQUEST));
    EXPECT_TRUE(nv::is_nonslot_lp_message(SCF_FAPI_START_REQUEST));
    EXPECT_TRUE(nv::is_nonslot_lp_message(SCF_FAPI_STOP_REQUEST));

    // Immediate (PARAM, CV mem-bank), slot, and control-lane messages must never route to the worker.
    EXPECT_FALSE(nv::is_nonslot_lp_message(SCF_FAPI_PARAM_REQUEST));
    EXPECT_FALSE(nv::is_nonslot_lp_message(CV_MEM_BANK_CONFIG_REQUEST));
    EXPECT_FALSE(nv::is_nonslot_lp_message(SCF_FAPI_DL_TTI_REQUEST));
    EXPECT_FALSE(nv::is_nonslot_lp_message(SCF_FAPI_SLOT_INDICATION));
    EXPECT_FALSE(nv::is_nonslot_lp_message(SCF_FAPI_ERROR_INDICATION));
}
