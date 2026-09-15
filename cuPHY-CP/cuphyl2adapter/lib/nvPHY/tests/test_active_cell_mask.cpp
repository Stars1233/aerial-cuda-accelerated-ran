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

#include "active_cell_mask.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>

namespace {

using nv::ActiveCellMask;

constexpr std::uint64_t bit(std::uint16_t c)
{
    return std::uint64_t{1} << c;
}

// ---------------------------------------------------------------------------
// Basic state
// ---------------------------------------------------------------------------

// Fresh mask: nothing staged, produced, or committed.
TEST(ActiveCellMask, DefaultIsEmpty)
{
    ActiveCellMask m;
    EXPECT_EQ(m.staged(), 0ULL);
    EXPECT_EQ(m.produced(), 0ULL);
    EXPECT_EQ(m.committed(), 0ULL);
    EXPECT_EQ(m.committedCount(), 0U);
}

// START stages only: not produced, not committed.
TEST(ActiveCellMask, StartStagesOnly)
{
    ActiveCellMask m;
    m.stageStarted(3);
    EXPECT_EQ(m.staged(), bit(3));
    EXPECT_EQ(m.produced(), 0ULL);
    EXPECT_EQ(m.committed(), 0ULL);
}

// ---------------------------------------------------------------------------
// The defer: a staged-but-not-yet-producing cell is NOT committed
// ---------------------------------------------------------------------------

// A staged cell that has not produced is NOT committed at the boundary -- this is the slot on
// which the established cells must not wait for it.
TEST(ActiveCellMask, StagedButNotProducedIsNotCommitted)
{
    ActiveCellMask m;
    m.stageStarted(3);
    EXPECT_EQ(m.commitStaged(), 0ULL); // committed = staged & produced = {3} & {} = {}
    EXPECT_EQ(m.committed(), 0ULL);
}

// markProduced records production but does not commit; commit happens at the next boundary.
TEST(ActiveCellMask, MarkProducedDoesNotCommitUntilBoundary)
{
    ActiveCellMask m;
    m.stageStarted(3);
    m.markProduced(3);
    EXPECT_EQ(m.produced(), bit(3));
    EXPECT_EQ(m.committed(), 0ULL);      // not committed until commitStaged()
    EXPECT_EQ(m.commitStaged(), bit(3)); // now committed
    EXPECT_EQ(m.committed(), bit(3));
}

// The full defer-by-one-slot join sequence (cell 0 running, cell 1 joins):
//   T_join   : cell 1 staged, has produced nothing -> commitStaged keeps it out (cell 0 alone)
//   T_join   : cell 1 produces its first slot      -> markProduced
//   T_join+1 : commitStaged -> cell 1 now joins the EOM expectation
TEST(ActiveCellMask, DeferredJoinSequence)
{
    ActiveCellMask m;
    m.stageStarted(0);
    m.markProduced(0);
    ASSERT_EQ(m.commitStaged(), bit(0)); // cell 0 established

    m.stageStarted(1);                          // cell 1 START (not yet producing)
    EXPECT_EQ(m.commitStaged(), bit(0));        // T_join: only cell 0 committed (cell 1 deferred)
    EXPECT_EQ(m.committed(), bit(0));

    m.markProduced(1);                          // cell 1 produces its first slot
    EXPECT_EQ(m.commitStaged(), bit(0) | bit(1)); // T_join+1: cell 1 joins
    EXPECT_EQ(m.committed(), bit(0) | bit(1));
}

// markProduced is gated on STAGED: a message for a cell that is not currently staged does not
// set produced (so it cannot pre-arm a later restart's commit).
TEST(ActiveCellMask, MarkProducedIgnoredWhenNotStaged)
{
    ActiveCellMask m;
    m.markProduced(5); // never staged
    EXPECT_EQ(m.produced(), 0ULL);
    EXPECT_EQ(m.commitStaged(), 0ULL);
}

// Two cells staged; only the one that has produced commits at the boundary, the other joins
// once it produces.
TEST(ActiveCellMask, OnlyProducedCellCommits)
{
    ActiveCellMask m;
    m.stageStarted(1);
    m.stageStarted(2);
    m.markProduced(2); // only cell 2 has produced
    EXPECT_EQ(m.commitStaged(), bit(2));
    EXPECT_EQ(m.committed(), bit(2));

    m.markProduced(1); // cell 1 produces later
    EXPECT_EQ(m.commitStaged(), bit(1) | bit(2));
}

// ---------------------------------------------------------------------------
// STOP / removal
// ---------------------------------------------------------------------------

// STOP clears STAGED and PRODUCED; the boundary then drops the cell from COMMITTED.
TEST(ActiveCellMask, StopClearsStagedAndProducedAndDropsAtBoundary)
{
    ActiveCellMask m;
    m.stageStarted(1);
    m.markProduced(1);
    m.stageStarted(2);
    m.markProduced(2);
    (void)m.commitStaged();
    ASSERT_EQ(m.committed(), bit(1) | bit(2));

    m.stageStopped(2);
    EXPECT_EQ(m.staged(), bit(1));               // staged cleared
    EXPECT_EQ(m.produced(), bit(1));             // produced cleared too
    EXPECT_EQ(m.committed(), bit(1) | bit(2));   // committed not dropped until the boundary
    EXPECT_EQ(m.commitStaged(), bit(1));         // boundary drops cell 2
    EXPECT_EQ(m.committed(), bit(1));
}

// Restart after STOP must re-prove production (STOP cleared PRODUCED).
TEST(ActiveCellMask, RestartMustReProveProduction)
{
    ActiveCellMask m;
    m.stageStarted(3);
    m.markProduced(3);
    (void)m.commitStaged();
    m.stageStopped(3);
    (void)m.commitStaged();
    ASSERT_EQ(m.committed(), 0ULL);
    ASSERT_EQ(m.produced(), 0ULL);

    m.stageStarted(3);                   // restart
    EXPECT_EQ(m.commitStaged(), 0ULL);   // not committed: has not re-produced
    m.markProduced(3);
    EXPECT_EQ(m.commitStaged(), bit(3)); // re-produced -> committed
}

// A straggler message arriving AFTER a STOP must not pre-arm a future restart's commit.
TEST(ActiveCellMask, StragglerAfterStopDoesNotPreCommitRestart)
{
    ActiveCellMask m;
    m.stageStarted(4);
    m.markProduced(4);
    (void)m.commitStaged();
    m.stageStopped(4);   // clears staged + produced
    m.markProduced(4);   // straggler for a now-unstaged cell -> ignored
    EXPECT_EQ(m.produced(), 0ULL);

    m.stageStarted(4);                 // restart
    EXPECT_EQ(m.commitStaged(), 0ULL); // must re-produce; straggler did not pre-set it
}

// START then STOP before producing: never committed.
TEST(ActiveCellMask, StartThenStopBeforeProducing)
{
    ActiveCellMask m;
    m.stageStarted(2);
    m.stageStopped(2);
    EXPECT_EQ(m.staged(), 0ULL);
    EXPECT_EQ(m.commitStaged(), 0ULL);
}

// ---------------------------------------------------------------------------
// commitStaged properties
// ---------------------------------------------------------------------------

// commitStaged() == staged & produced: a staged-only cell is excluded.
TEST(ActiveCellMask, CommitStagedIsStagedAndProduced)
{
    ActiveCellMask m;
    m.stageStarted(1);
    m.markProduced(1);
    m.stageStarted(4); // staged, not produced
    EXPECT_EQ(m.commitStaged(), bit(1));
    EXPECT_EQ(m.committed(), bit(1));
}

// Idempotent: committing the same staged&produced set again is unchanged.
TEST(ActiveCellMask, CommitStagedIsIdempotent)
{
    ActiveCellMask m;
    m.stageStarted(7);
    m.markProduced(7);
    EXPECT_EQ(m.commitStaged(), bit(7));
    EXPECT_EQ(m.commitStaged(), bit(7));
    EXPECT_EQ(m.committed(), bit(7));
}

// Invariant: committed is always a subset of (staged & produced) after a boundary.
TEST(ActiveCellMask, CommittedSubsetInvariant)
{
    ActiveCellMask m;
    m.stageStarted(1);
    m.markProduced(1);
    m.stageStarted(2);
    m.markProduced(2);
    m.stageStarted(3); // staged only -- must be excluded
    const auto committed = m.commitStaged();
    EXPECT_EQ(committed & ~(m.staged() & m.produced()), 0ULL);
    EXPECT_EQ(committed, bit(1) | bit(2));
}

// ---------------------------------------------------------------------------
// Legacy serial path (activate/deactivate immediate)
// ---------------------------------------------------------------------------

// Legacy serial START stages, produces, and commits in one step; survives a later boundary.
TEST(ActiveCellMask, ActivateImmediateCommitsAllThreeAndSurvives)
{
    ActiveCellMask m;
    EXPECT_EQ(m.activateImmediate(4), bit(4));
    EXPECT_EQ(m.staged(), bit(4));
    EXPECT_EQ(m.produced(), bit(4)); // produced set so commitStaged retains it
    EXPECT_EQ(m.committed(), bit(4));

    EXPECT_EQ(m.commitStaged(), bit(4)); // staged & produced -> retained
    EXPECT_EQ(m.committed(), bit(4));

    EXPECT_EQ(m.activateImmediate(4), bit(4)); // idempotent
    EXPECT_EQ(m.committedCount(), 1U);
}

// Legacy serial STOP clears staged, produced, and committed at once.
TEST(ActiveCellMask, DeactivateImmediateClearsAllThree)
{
    ActiveCellMask m;
    (void)m.activateImmediate(6);
    ASSERT_EQ(m.committed(), bit(6));

    EXPECT_EQ(m.deactivateImmediate(6), 0ULL);
    EXPECT_EQ(m.staged(), 0ULL);
    EXPECT_EQ(m.produced(), 0ULL);
    EXPECT_EQ(m.committed(), 0ULL);
}

// ---------------------------------------------------------------------------
// Range + value constructor
// ---------------------------------------------------------------------------

// Out-of-range cell ids (>= 64) are ignored by stage/produce; 0 and 63 are valid.
TEST(ActiveCellMask, OutOfRangeCellIgnored)
{
    ActiveCellMask m;
    m.stageStarted(0);
    m.markProduced(0);
    m.stageStarted(63);
    m.markProduced(63);
    m.stageStarted(64);  // ignored
    m.markProduced(64);  // ignored
    m.stageStarted(999); // ignored
    EXPECT_EQ(m.staged(), bit(0) | bit(63));
    EXPECT_EQ(m.produced(), bit(0) | bit(63));
    EXPECT_EQ(m.commitStaged(), bit(0) | bit(63));
}

// Value constructor round-trips all three masks (used to move-rebuild an owning object): a
// mid-join snapshot where one cell is staged-only.
TEST(ActiveCellMask, ValueConstructorRoundTrips)
{
    // staged {1,5,6}, produced {1,5}, committed {1}
    ActiveCellMask m(bit(1) | bit(5) | bit(6), bit(1) | bit(5), bit(1));
    EXPECT_EQ(m.staged(), bit(1) | bit(5) | bit(6));
    EXPECT_EQ(m.produced(), bit(1) | bit(5));
    EXPECT_EQ(m.committed(), bit(1));
    // next boundary commits staged & produced = {1,5}; cell 6 (staged-only) stays out
    EXPECT_EQ(m.commitStaged(), bit(1) | bit(5));
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

// stageStarted/stageStopped/markProduced (nonslot_lp + msg threads) race commitStaged (msg
// thread, per SLOT.IND). Invariants: (a) a started-produced-never-stopped cell stays committed
// regardless of churn elsewhere; (b) after a final barrier, committed == staged & produced
// (convergence) and committed is a subset of (staged & produced). All access is atomic, so it
// must also run clean under TSan. No sleeping; bounded by iteration count.
TEST(ActiveCellMask, ConcurrentStageProduceCommitConverges)
{
    static constexpr int kIters = 100000;
    ActiveCellMask m;

    m.stageStarted(0); // cell 0: stable -- started, produced, never stopped
    m.markProduced(0);

    std::jthread churner([&] {
        for(int i = 0; i < kIters; ++i)
        {
            m.stageStarted(1); // churned cell -- races the committer
            m.markProduced(1);
            m.stageStopped(1);
        }
    });
    std::jthread committer([&] {
        for(int i = 0; i < kIters; ++i)
        {
            (void)m.commitStaged(); // simulates the per-SLOT.ind boundary commit
        }
    });

    churner.join();
    committer.join();

    m.stageStarted(1); // settle staged & produced to a known value
    m.markProduced(1);
    const auto committed = m.commitStaged();                       // final barrier, no writer
    EXPECT_NE(committed & bit(0), 0ULL);                            // (a) stable cell retained
    EXPECT_EQ(m.committed(), m.staged() & m.produced());           // (b) converges to staged & produced
    EXPECT_EQ(m.committed() & ~(m.staged() & m.produced()), 0ULL); // (b) subset invariant
}

} // namespace
