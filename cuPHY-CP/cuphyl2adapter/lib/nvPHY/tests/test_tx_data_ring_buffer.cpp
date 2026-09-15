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
 * @file test_tx_data_ring_buffer.cpp
 * @brief Unit tests for TxDataRingBuffer (Phase 1 Option A) and its Phase 2
 *        extensions: arm(), is_valid(), gpu_ptrs(), msg_bufs().
 *
 * Zero dependencies on PHY_module, PHYDriverProxy, CUDA, or cuPHY. All tests
 * exercise the ring buffer class directly through its public interface.
 *
 * Covers:
 *   Group Stage       — stage() increments cell_count, stores pointers at correct index
 *   Group Arm_Mismatch — arm() returns false on count mismatch or zero expected
 *   Group Arm_Match    — arm() returns true when staged == expected > 0
 *   Group IsValid      — is_valid() tracks arm/reset lifecycle
 *   Group Accessors    — gpu_ptrs()/msg_bufs() return nullptr when slot not armed
 *   Group Reset        — reset() zeroes all fields including pointer arrays
 *   Group RingIsolation — staging into slot 0 does not affect slot 1
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

// MAX_CELLS_PER_SLOT and SLOT_STORAGE_DEPTH are defined via cmake
// target_compile_definitions (mirrors the production build). The test uses
// the same values so TxDataSlotState has the same layout.
#include "nv_tx_data_ring_buffer.hpp"

namespace
{

// Distinct dummy pointers used to verify store / retrieve round-trips.
alignas(8) std::uint8_t g_gpu_buf_0[4]{};
alignas(8) std::uint8_t g_gpu_buf_1[4]{};

alignas(8) std::uint8_t g_fapi_buf_0[4]{};
alignas(8) std::uint8_t g_fapi_buf_1[4]{};

} // namespace

// ===========================================================================
// Group Stage
// ===========================================================================

TEST(TxDataRingBuffer_Stage, HappyPath_CellCountIncrements)
{
    nv::TxDataRingBuffer ring{};
    EXPECT_EQ(ring.staged_count(0U), 0U);

    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    EXPECT_EQ(ring.staged_count(0U), 1U);

    ring.stage(0U, 1U, g_gpu_buf_1, g_fapi_buf_1);
    EXPECT_EQ(ring.staged_count(0U), 2U);
}

TEST(TxDataRingBuffer_Stage, GpuPtrStoredAtCellIndex)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 3U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(0U, 7U, g_gpu_buf_1, g_fapi_buf_1);

    // Arm so accessors return the data pointer instead of nullptr.
    const bool ok = ring.arm(0U, /*expected_pdsch=*/2U);
    ASSERT_TRUE(ok);

    // Verify the stored pointers are accessible at the right cell indices.
    const auto* gpu = ring.gpu_ptrs(0U);
    ASSERT_NE(gpu, nullptr);
    EXPECT_EQ(gpu[3], g_gpu_buf_0);
    EXPECT_EQ(gpu[7], g_gpu_buf_1);
    EXPECT_EQ(gpu[0], nullptr);
}

TEST(TxDataRingBuffer_Stage, MsgBufStoredAtCellIndex)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 5U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(0U, 9U, g_gpu_buf_1, g_fapi_buf_1);

    const bool ok = ring.arm(0U, 2U);
    ASSERT_TRUE(ok);

    const auto* msg = ring.msg_bufs(0U);
    ASSERT_NE(msg, nullptr);
    EXPECT_EQ(msg[5], static_cast<const void*>(g_fapi_buf_0));
    EXPECT_EQ(msg[9], static_cast<const void*>(g_fapi_buf_1));
    EXPECT_EQ(msg[0], nullptr);
}

// ===========================================================================
// Group Arm_Mismatch — parameterized (3 of 4 cases)
// ===========================================================================

struct ArmMismatchParam
{
    uint8_t staged;
    uint8_t expected_count;
};

class TxDataRingBuffer_Arm_Mismatch_P
    : public testing::TestWithParam<ArmMismatchParam> {};

TEST_P(TxDataRingBuffer_Arm_Mismatch_P, ReturnsFalse)
{
    nv::TxDataRingBuffer ring{};
    for(uint8_t i = 0; i < GetParam().staged; ++i)
    {
        ring.stage(0U, static_cast<uint32_t>(i), g_gpu_buf_0, g_fapi_buf_0);
    }
    EXPECT_FALSE(ring.arm(0U, GetParam().expected_count));
}

INSTANTIATE_TEST_SUITE_P(ArmMismatch, TxDataRingBuffer_Arm_Mismatch_P, testing::Values(ArmMismatchParam{1, 0}, // zero expected — always false
                                                                                       ArmMismatchParam{1, 3}, // staged < expected
                                                                                       ArmMismatchParam{3, 1}  // staged > expected
                                                                                       ));

// 4th case: side-effects after failed arm() must also be verified.
TEST(TxDataRingBuffer_Arm_Mismatch, MismatchClearsSlot)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    EXPECT_FALSE(ring.arm(0U, 2U)); // mismatch — arm() calls slot.reset() internally
    EXPECT_EQ(ring.staged_count(0U), 0U);
    EXPECT_FALSE(ring.is_valid(0U));
}

// ===========================================================================
// Group Arm_Match
// ===========================================================================

TEST(TxDataRingBuffer_Arm_Match, ExactCount_ReturnsTrue)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(0U, 1U, g_gpu_buf_1, g_fapi_buf_1);
    EXPECT_TRUE(ring.arm(0U, 2U));
}

TEST(TxDataRingBuffer_Arm_Match, SetsExpectedCells)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    EXPECT_TRUE(ring.is_valid(0U));
}

TEST(TxDataRingBuffer_Arm_Match, SingleCell)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 4U, g_gpu_buf_0, g_fapi_buf_0);
    EXPECT_TRUE(ring.arm(0U, 1U));
    EXPECT_NE(ring.gpu_ptrs(0U), nullptr);
    EXPECT_NE(ring.msg_bufs(0U), nullptr);
}

// ===========================================================================
// Group IsValid
// ===========================================================================

TEST(TxDataRingBuffer_IsValid, FalseAfterConstruction)
{
    nv::TxDataRingBuffer ring{};
    EXPECT_FALSE(ring.is_valid(0U));
    EXPECT_FALSE(ring.is_valid(1U));
}

TEST(TxDataRingBuffer_IsValid, TrueAfterSuccessfulArm)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    EXPECT_TRUE(ring.is_valid(0U));
}

TEST(TxDataRingBuffer_IsValid, FalseAfterReset)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    ring.reset(0U);
    EXPECT_FALSE(ring.is_valid(0U));
}

TEST(TxDataRingBuffer_IsValid, FalseAfterFailedArm)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    EXPECT_FALSE(ring.arm(0U, 0U)); // expected=0 → always false
    EXPECT_FALSE(ring.is_valid(0U));
}

// ===========================================================================
// Group Accessors
// ===========================================================================

TEST(TxDataRingBuffer_Accessors, GpuPtrs_NullWhenNotArmed)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    // Not armed yet — accessors must return nullptr.
    EXPECT_EQ(ring.gpu_ptrs(0U), nullptr);
}

TEST(TxDataRingBuffer_Accessors, MsgBufs_NullWhenNotArmed)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    EXPECT_EQ(ring.msg_bufs(0U), nullptr);
}

TEST(TxDataRingBuffer_Accessors, GpuPtrs_NullAfterReset)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    ring.reset(0U);
    EXPECT_EQ(ring.gpu_ptrs(0U), nullptr);
}

TEST(TxDataRingBuffer_Accessors, GpuPtrs_NonNullAfterArm)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    EXPECT_NE(ring.gpu_ptrs(0U), nullptr);
    EXPECT_NE(ring.msg_bufs(0U), nullptr);
}

// ===========================================================================
// Group Reset
// ===========================================================================

TEST(TxDataRingBuffer_Reset, ClearsStagingPreservesDeferred)
{
    // reset() is staging-only: it clears cell_count / valid / pointers but must
    // NOT touch `deferred`. That is the fix for the observed double-free — the
    // H2D-completion callback calls reset() after releasing, and if that flipped
    // `deferred` back to false the message thread's inline path would re-release
    // the same NVIPC buffer. Ownership is cleared explicitly via set_deferred().
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(0U, 1U, g_gpu_buf_1, g_fapi_buf_1);
    ASSERT_TRUE(ring.arm(0U, 2U));
    ring.set_deferred(0U, true);

    ring.reset(0U);

    EXPECT_EQ(ring.staged_count(0U), 0U);
    EXPECT_FALSE(ring.is_valid(0U));
    EXPECT_TRUE(ring.is_deferred(0U)); // preserved: reset() is staging-only
    EXPECT_EQ(ring.gpu_ptrs(0U), nullptr);
    EXPECT_EQ(ring.msg_bufs(0U), nullptr);
}

TEST(TxDataRingBuffer_Reset, StageAfterResetWorks)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    ring.reset(0U);

    // Re-stage and re-arm for the same slot (next slot reuse).
    ring.stage(0U, 2U, g_gpu_buf_1, g_fapi_buf_1);
    EXPECT_TRUE(ring.arm(0U, 1U));
    EXPECT_TRUE(ring.is_valid(0U));
}

// ===========================================================================
// Group Deferred
// ===========================================================================

TEST(TxDataRingBuffer_Deferred, FalseAfterConstruction)
{
    nv::TxDataRingBuffer ring{};
    EXPECT_FALSE(ring.is_deferred(0U));
}

TEST(TxDataRingBuffer_Deferred, SetAndGet)
{
    nv::TxDataRingBuffer ring{};
    ring.set_deferred(0U, true);
    EXPECT_TRUE(ring.is_deferred(0U));
    ring.set_deferred(0U, false);
    EXPECT_FALSE(ring.is_deferred(0U));
}

// ===========================================================================
// Group RingIsolation
// ===========================================================================

TEST(TxDataRingBuffer_RingIsolation, StagingSlot0DoesNotAffectSlot1)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(0U, 1U, g_gpu_buf_1, g_fapi_buf_1);

    // Slot 1 must be completely untouched.
    EXPECT_EQ(ring.staged_count(1U), 0U);
    EXPECT_FALSE(ring.is_valid(1U));
    EXPECT_EQ(ring.gpu_ptrs(1U), nullptr);
    EXPECT_EQ(ring.msg_bufs(1U), nullptr);
}

TEST(TxDataRingBuffer_RingIsolation, ResetSlot0DoesNotAffectSlot1)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(1U, 0U, g_gpu_buf_1, g_fapi_buf_1);

    ASSERT_TRUE(ring.arm(1U, 1U));
    ring.reset(0U); // reset slot 0

    // Slot 1 must still be armed and valid.
    EXPECT_TRUE(ring.is_valid(1U));
    EXPECT_NE(ring.gpu_ptrs(1U), nullptr);
    EXPECT_NE(ring.msg_bufs(1U), nullptr);
}

TEST(TxDataRingBuffer_RingIsolation, ArmSlot1DoesNotArmSlot0)
{
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.stage(1U, 0U, g_gpu_buf_1, g_fapi_buf_1);

    ASSERT_TRUE(ring.arm(1U, 1U));

    // Slot 0 was staged but not armed — must remain invalid.
    EXPECT_FALSE(ring.is_valid(0U));
    EXPECT_EQ(ring.gpu_ptrs(0U), nullptr);
}

// ===========================================================================
// Group LeakRegression — P0: TX_DATA IPC buffer leak on enqueue failure
//
// Bug context (nv_phy_slot_dispatch.cpp:submit_slot_command):
//   When @c l1_enqueue_phy_work returns non-zero, the cuphydriver completion
//   callback never fires (nothing was enqueued), so a slot left deferred would
//   never release its staged TX_DATA FAPI msg_buf. Result: an IPC buffer leak
//   that grows by one entry per enqueue failure until the pool is exhausted.
//
// Fix: the abandon paths (@c reset_txdata_h2d_state_for_ring on enqueue failure
// and ring collision) explicitly hand release back to the inline message-thread
// path via set_deferred(false). Because reset() is now staging-only (it must NOT
// clear `deferred` — see the double-free note below), the clear is explicit.
//
// These tests pin the primitive contracts the fix depends on. The matching
// end-to-end coverage belongs in the higher-level PHY_module test suite — see
// tests/test_nvphy_fapi_tasks.cpp for the submit_slot_command-flow harness.
// ===========================================================================

TEST(TxDataRingBuffer_LeakRegression, SetDeferredFalseHandsReleaseToInline)
{
    // The abandon path clears ownership explicitly with set_deferred(false) so
    // the inline message-thread release frees the lane (no leak).
    nv::TxDataRingBuffer ring{};
    ring.set_deferred(0U, true);
    ASSERT_TRUE(ring.is_deferred(0U));

    ring.set_deferred(0U, false);

    EXPECT_FALSE(ring.is_deferred(0U));
}

TEST(TxDataRingBuffer_LeakRegression, AbandonSequenceClearsStagingAndOwnership)
{
    // Models reset_txdata_h2d_state_for_ring on a fully staged + armed + deferred
    // slot (the enqueue-failure / collision state): set_deferred(false) then
    // reset() leaves the slot released for the next ring rotation and owned by
    // the inline path.
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ASSERT_TRUE(ring.arm(0U, 1U));
    ring.set_deferred(0U, true);

    ASSERT_TRUE(ring.is_valid(0U));
    ASSERT_TRUE(ring.is_deferred(0U));

    ring.set_deferred(0U, false); // abandon path hands release to inline
    ring.reset(0U);               // staging-only

    EXPECT_FALSE(ring.is_valid(0U));
    EXPECT_FALSE(ring.is_deferred(0U));
    EXPECT_EQ(ring.staged_count(0U), 0U);
    EXPECT_EQ(ring.gpu_ptrs(0U), nullptr);
    EXPECT_EQ(ring.msg_bufs(0U), nullptr);
}

TEST(TxDataRingBuffer_LeakRegression, FailedArmResetsStagingKeepsDeferred)
{
    // arm() internally calls reset() on a staged-count mismatch. reset() is
    // staging-only, so `deferred` is unchanged here; the call site prevents a
    // "deferred but not armed" zombie by setting deferred = is_valid() AFTER arm
    // (launch_tx_data_h2d), which resolves to false on a failed arm.
    nv::TxDataRingBuffer ring{};
    ring.stage(0U, 0U, g_gpu_buf_0, g_fapi_buf_0);
    ring.set_deferred(0U, true);
    ASSERT_TRUE(ring.is_deferred(0U));

    // Expected 2 cells but only 1 staged → arm() fails and resets staging.
    EXPECT_FALSE(ring.arm(0U, 2U));
    EXPECT_FALSE(ring.is_valid(0U));

    // Emulate the call-site ownership resolution after arm().
    ring.set_deferred(0U, ring.is_valid(0U));
    EXPECT_FALSE(ring.is_deferred(0U));
}

// ===========================================================================
// Group DeferredRelease — single-owner TX_DATA release (double-free fix).
//
// Bug context (nv_phy_slot_dispatch.cpp): the TX_DATA IPC buffer is released by
// two agents on two threads — the message thread (release_stored_slot_messages,
// inline) and the cuphydriver H2D-completion callback (release_deferred_tx_data).
// They raced because the callback's reset() flipped the plain `deferred` bool
// back to false; the message thread then read deferred==false and freed the
// SAME buffer again → NVIPC "already in queue" double-free (observed at SFN
// 549.16 / 74.3 / 58.18, different buffer indices).
//
// Fix: `deferred` is atomic and set once per slot at arm time; it routes exactly
// one owner (true => callback frees, inline skips; false => inline frees). reset()
// is staging-only so the callback can no longer flip the gate.
// ===========================================================================

TEST(TxDataRingBuffer_DeferredRelease, CallbackCompletionClearsDeferredAfterRelease)
{
    // Callback path: release_lane (not modeled here) then disarm deferred, then staging reset.
    constexpr uint32_t kRing = 2U;
    nv::TxDataRingBuffer ring{};
    ring.set_deferred(kRing, true);
    ASSERT_TRUE(ring.is_deferred(kRing));

    ring.set_deferred(kRing, false); // callback disarm after release_lane
    ring.reset(kRing);               // staging-only

    EXPECT_FALSE(ring.is_deferred(kRing));
}

TEST(TxDataRingBuffer_DeferredRelease, NonDeferredSlotReleasedInline)
{
    // Non-deferred slot (enqueue failed / non-direct mode): inline owns release.
    constexpr uint32_t kRing = 0U;
    nv::TxDataRingBuffer ring{};
    ring.set_deferred(kRing, false);

    EXPECT_FALSE(ring.is_deferred(kRing)); // inline releases; callback skips
}

TEST(TxDataRingBuffer_DeferredRelease, ReArmSwitchesOwnerForNextSlot)
{
    // The ring entry is reused across slots; each slot re-arms ownership. A slot
    // that was deferred (callback-owned) completes with disarm + staging reset,
    // then the next slot at this ring index can be inline-owned.
    constexpr uint32_t kRing = 0U;
    nv::TxDataRingBuffer ring{};

    ring.set_deferred(kRing, true);
    ring.set_deferred(kRing, false); // callback disarm
    ring.reset(kRing);               // staging-only
    EXPECT_FALSE(ring.is_deferred(kRing));

    ring.set_deferred(kRing, false); // next slot: inline-owned
    EXPECT_FALSE(ring.is_deferred(kRing));
}

TEST(TxDataRingBuffer_DeferredRelease, ConcurrentDeferredFlagAccessStress)
{
    nv::TxDataRingBuffer ring{};
    ring.set_deferred(0U, true);

    std::atomic_bool stop{false};
    std::atomic_int  reads{0};
    // Barrier so the writer cannot finish (and set stop) before both readers are
    // in their loop: readers announce themselves via readers_ready, the writer
    // waits for both, then releases them with start. Without this the writer's
    // 1000 stores can complete before either reader runs, leaving reads == 0.
    std::atomic_bool start{false};
    std::atomic_int  readers_ready{0};

    const auto reader = [&ring, &stop, &reads, &start, &readers_ready] {
        readers_ready.fetch_add(1, std::memory_order_acq_rel);
        while(!start.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        // do/while guarantees at least one read after start, so reads >= 2 holds
        // regardless of how quickly the writer reaches stop.
        do
        {
            (void)ring.is_deferred(0U);
            reads.fetch_add(1, std::memory_order_relaxed);
        } while(!stop.load(std::memory_order_acquire));
    };

    const auto writer = [&ring, &stop, &start, &readers_ready] {
        while(readers_ready.load(std::memory_order_acquire) < 2)
        {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        for(int i = 0; i < 1000; ++i)
        {
            ring.set_deferred(0U, (i % 2) == 0);
        }
        stop.store(true, std::memory_order_release);
    };

    {
        std::jthread t1(reader);
        std::jthread t2(reader);
        std::jthread t3(writer);
    } // all threads join here before the assertion below

    EXPECT_GT(reads.load(), 0);
}
