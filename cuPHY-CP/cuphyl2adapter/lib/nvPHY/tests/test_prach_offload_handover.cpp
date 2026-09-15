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

// Unit tests for the GPU-free parts of the offload PRACH reconfiguration feature:
//   - the handover state machine: arming, the SLOT.IND-driven incremental swap
//     drain, commit, bounded-wait timeout, and the cancel-vs-commit guards,
//     using a scripted drain callback that stands in for the aggregator idle scan;
//   - the staging-error -> reconfigure-outcome mapping (toReconfigOutcome), the
//     one consumer of PrachStageError.
// The staging/create fan-out itself is CUDA-entangled (cuphyCreatePrachRx and live
// PhyPrachAggr state) and is not unit-testable without a GPU.

#include <gtest/gtest.h>

#include "prach_offload_handover.hpp"
#include "prach_offload_reconfig.hpp"

#include <chrono>
#include <cstdint>
#include <thread>

namespace {

// Builds a drain callback that simulates a SLOT.IND tick where exactly the
// aggregators in `idle_mask` are idle: each pending-and-idle aggregator is
// swapped (its pending bit cleared and the count decremented), mirroring
// PrachOffloadReconfig::tryCommit's real drain.
auto drainIdle(std::uint64_t idle_mask)
{
    return [idle_mask](std::uint64_t& pending_mask, std::int32_t& pending_count) {
        for(int i = 0; i < 64; ++i)
        {
            const std::uint64_t b = std::uint64_t{1} << i;
            if((pending_mask & b) && (idle_mask & b))
            {
                pending_mask &= ~b;
                --pending_count;
            }
        }
    };
}

} // namespace

// All aggregators busy on the first tick, partially idle on the next, fully
// idle last: the handover must drain incrementally and commit only when the
// count reaches zero — never on a partial tick.
TEST(PrachOffloadHandover, IncrementalDrainCommitsOnlyWhenAllSwapped)
{
    PrachOffloadHandover h;
    h.arm(4);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 4);

    // Tick 1: no aggregator idle -> nothing swaps, still Armed.
    h.tryCommit(drainIdle(0b0000ULL));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 4);

    // Tick 2: aggregators 0 and 2 idle -> two swap, still Armed.
    h.tryCommit(drainIdle(0b0101ULL));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 2);

    // Tick 3: all idle -> last two swap, commit fires.
    h.tryCommit(drainIdle(0b1111ULL));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.pendingCount(), 0);

    // Worker's wait observes the commit immediately.
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);
}

// No tick ever drains the swap: the bounded wait must time out, report
// failure, and leave the handover Cancelled so later ticks do nothing.
TEST(PrachOffloadHandover, BoundedWaitTimesOutAndCancels)
{
    PrachOffloadHandover h;
    h.arm(2);

    EXPECT_EQ(h.wait(std::chrono::milliseconds{20}), PrachOffloadHandover::WaitResult::TimedOut);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Cancelled);

    // A late tick after the timeout must not swap or resurrect the handover.
    bool drain_called = false;
    h.tryCommit([&](std::uint64_t&, std::int32_t&) { drain_called = true; });
    EXPECT_FALSE(drain_called);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Cancelled);
}

// An async cancel that races a completed commit must NOT downgrade Committed
// -> Cancelled (the B-5 guard), otherwise the worker would wrongly take the
// failure path after the handles were already swapped.
TEST(PrachOffloadHandover, CancelDoesNotDowngradeCommitted)
{
    PrachOffloadHandover h;
    h.arm(1);
    h.tryCommit(drainIdle(~std::uint64_t{0}));
    ASSERT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);

    h.cancel();  // guarded: no-op because phase != Armed
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);
}

// Cancel before any swap must prevent the commit entirely: a subsequent tick
// must not swap, and the worker's wait reports failure.
TEST(PrachOffloadHandover, CancelBeforeCommitPreventsSwap)
{
    PrachOffloadHandover h;
    h.arm(2);
    h.cancel();
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Cancelled);

    bool drain_called = false;
    h.tryCommit([&](std::uint64_t&, std::int32_t&) { drain_called = true; });
    EXPECT_FALSE(drain_called);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Cancelled);
}

// A fresh (un-armed) handover ignores ticks entirely (the slot-path fast filter).
TEST(PrachOffloadHandover, TryCommitIsNoOpWhenNotArmed)
{
    PrachOffloadHandover h;
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Idle);

    bool drain_called = false;
    h.tryCommit([&](std::uint64_t&, std::int32_t&) { drain_called = true; });
    EXPECT_FALSE(drain_called);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Idle);
}

// End-to-end across threads: a worker thread arms and blocks in wait() while
// a separate "slot" thread drives ticks until the swap drains. The worker
// must wake with success. Deterministic (no sleeps): the slot loops until
// !armed().
TEST(PrachOffloadHandover, WorkerBlocksUntilSlotThreadDrivenCommit)
{
    PrachOffloadHandover h;
    h.arm(3);

    auto worker_rc = PrachOffloadHandover::WaitResult::TimedOut;
    std::thread worker([&] { worker_rc = h.wait(std::chrono::seconds{5}); });

    // Slot thread: tick until the handover commits. First ticks expose only
    // some idle aggregators; eventually all are idle and the commit fires.
    int tick = 0;
    while(h.armed())
    {
        const std::uint64_t idle = (tick++ < 3) ? 0b001ULL : 0b111ULL;
        h.tryCommit(drainIdle(idle));
    }

    worker.join();
    EXPECT_EQ(worker_rc, PrachOffloadHandover::WaitResult::Committed);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
}

// Re-arming after a completed handover resets the pending state for the next
// reconfiguration (one PRACH reconfig at a time on the single nonslot_lp FIFO).
TEST(PrachOffloadHandover, ReArmResetsPendingState)
{
    PrachOffloadHandover h;
    h.arm(2);
    h.tryCommit(drainIdle(~std::uint64_t{0}));
    ASSERT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);

    h.arm(5);
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 5);

    h.tryCommit(drainIdle(~std::uint64_t{0}));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.pendingCount(), 0);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);
}

// Boundary aggregator counts. arm(0): nothing to swap, so the first tick commits
// immediately. arm(64): the pending mask saturates the full 64 bits and still
// drains to a clean commit.
TEST(PrachOffloadHandover, ArmBoundaryCountsCommitCleanly)
{
    PrachOffloadHandover h;

    h.arm(0);
    EXPECT_EQ(h.pendingCount(), 0);
    h.tryCommit(drainIdle(~std::uint64_t{0}));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);

    h.arm(64);
    EXPECT_EQ(h.pendingCount(), 64);
    h.tryCommit(drainIdle(~std::uint64_t{0}));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.pendingCount(), 0);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);
}

// Mid-run active-cell reconfig: a cell added/reconfigured while other cells are
// actively running data shares the PRACH aggregators, some of which are busy. The
// handover must swap each aggregator only once it goes idle (never interrupting an
// in-flight PRACH on a running cell) and must commit only after the last busy
// aggregator drains -- mirroring the legacy per-slot num_new_prach_handles swap,
// driven here by the SLOT.IND hook. Models the "add cell mid-run, then PRACH HO"
// scenario at the (GPU-free) handover level.
TEST(PrachOffloadHandover, MidRunReconfigSwapsAggregatorsOnlyWhenIdle)
{
    PrachOffloadHandover h;
    h.arm(4);  // shared PRACH aggregators rebuilt for the reconfig
    EXPECT_EQ(h.pendingCount(), 4);

    // Aggregators 1 and 3 are busy serving running cells; 0 and 2 are idle.
    h.tryCommit(drainIdle(0b0101ULL));  // only idle 0 and 2 swap
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 2);

    // Next slot: aggregator 1 frees, 3 still busy -> only 1 swaps; not yet committed.
    h.tryCommit(drainIdle(0b0111ULL));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Armed);
    EXPECT_EQ(h.pendingCount(), 1);

    // The last busy aggregator (3) goes idle -> final swap, handover commits and the
    // blocked worker observes success.
    h.tryCommit(drainIdle(0b1111ULL));
    EXPECT_EQ(h.phase(), PrachOffloadHandover::Phase::Committed);
    EXPECT_EQ(h.pendingCount(), 0);
    EXPECT_EQ(h.wait(std::chrono::milliseconds{0}), PrachOffloadHandover::WaitResult::Committed);
}

// toReconfigOutcome() is the sole consumer of PrachStageError: reconfigure() maps the
// staging cause to a caller-facing PrachReconfigOutcome through it. Lock the mapping so a
// future edit cannot silently swap the causes or route one to a non-failure outcome.

// Compile-time contract (toReconfigOutcome is constexpr).
static_assert(toReconfigOutcome(PrachStageError::StageRejected) == PrachReconfigOutcome::StageFailed);
static_assert(toReconfigOutcome(PrachStageError::TempCreateFailed) == PrachReconfigOutcome::CreateFailed);

TEST(PrachReconfigOutcomeMapping, MapsEachCauseToItsFailureOutcome)
{
    // Each cause maps to its documented outcome.
    EXPECT_EQ(toReconfigOutcome(PrachStageError::StageRejected), PrachReconfigOutcome::StageFailed);
    EXPECT_EQ(toReconfigOutcome(PrachStageError::TempCreateFailed), PrachReconfigOutcome::CreateFailed);

    // Distinct causes never collapse to the same outcome (guards a copy-paste swap).
    EXPECT_NE(toReconfigOutcome(PrachStageError::StageRejected),
              toReconfigOutcome(PrachStageError::TempCreateFailed));

    // A staging failure is never mapped to a success/non-failure outcome.
    for(const auto err : {PrachStageError::StageRejected, PrachStageError::TempCreateFailed})
    {
        const auto outcome = toReconfigOutcome(err);
        EXPECT_NE(outcome, PrachReconfigOutcome::Committed);
        EXPECT_NE(outcome, PrachReconfigOutcome::Busy);
    }
}
