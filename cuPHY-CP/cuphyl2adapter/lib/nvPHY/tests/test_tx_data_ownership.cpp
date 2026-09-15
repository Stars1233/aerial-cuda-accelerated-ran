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

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "nv_fapi_message_storage.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"
#include "tx_data_ownership_tracker.hpp"

namespace
{

/**
 * @brief Build a minimal TX_DATA FAPI message descriptor for storage tests.
 *
 * @param[in] cell_id  Cell id stamped on the descriptor (default 0).
 * @return Descriptor with msg_id = SCF_FAPI_TX_DATA_REQUEST and the given cell_id.
 */
nv::phy_mac_msg_desc make_tx_data_msg(uint16_t cell_id = 0)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_TX_DATA_REQUEST;
    msg.cell_id = cell_id;
    return msg;
}

// CountingReleaseTransport / DeletingReleaseTransport are shared with
// test_nvphy_fapi_tasks.cpp; they live in tx_data_ownership_tracker.hpp to avoid
// duplicate definitions drifting out of sync.
using nv::test::CountingReleaseTransport;
using nv::test::DeletingReleaseTransport;

class TxDataOwnershipTest : public ::testing::Test
{
protected:
    static constexpr uint32_t kRing = 1U;

    void SetUp() override
    {
        // TxDataRingBuffer / FapiSlotMessageStorage hold atomics (non-assignable),
        // so rely on fresh per-test construction and only (re)bind the slot here.
        tracker_.reset(kRing);
        store_.reset_for_slot(0x0001'0002u);
        ASSERT_EQ(store_.store_message(make_tx_data_msg()), nv::StoreResult::Stored);
    }

    nv::test::TxDataOwnershipTracker tracker_{};
    nv::TxDataRingBuffer             ring_{};
    nv::FapiSlotMessageStorage       store_{8};
};

} // namespace

TEST_F(TxDataOwnershipTest, ReplayBuild_SplitPhaseArmXorDirectArm)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    tracker_.assert_arm_contract(/*replay_build=*/true, /*direct_mode=*/false, kRing);

    tracker_.reset(kRing);
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/true, /*direct_mode=*/true, /*ret=*/0, /*phy_list_size=*/1);
    tracker_.assert_arm_contract(/*replay_build=*/true, /*direct_mode=*/true, kRing);
}

TEST_F(TxDataOwnershipTest, NoReplayBuild_ArmOnlyInsideLaunchTxDataH2d)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    tracker_.assert_arm_contract(/*replay_build=*/false, /*direct_mode=*/false, kRing);

    tracker_.reset(kRing);
    ring_.set_deferred(kRing, false);
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/false, /*direct_mode=*/true, /*ret=*/0, /*phy_list_size=*/1);
    EXPECT_EQ(tracker_.stats(kRing).arm_count, 0);
    EXPECT_FALSE(ring_.is_deferred(kRing));
}

TEST_F(TxDataOwnershipTest, NoReplayBuild_EnqueueChannelTasksNeverArms)
{
    // Models a GT-12162-style adjacent arm in enqueue_channel_tasks — the tracker
    // must flag it as a forbidden production arm site.
    nv::test::try_arm_enqueue_channel_tasks(tracker_, ring_, kRing, true);
    EXPECT_EQ(tracker_.stats(kRing).arm_site, nv::test::ArmSite::EnqueueChannelTasks);
    EXPECT_TRUE(tracker_.arm_site_is_forbidden(kRing));
}

TEST_F(TxDataOwnershipTest, NeverBothArmSitesInOneSlot)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/true, /*direct_mode=*/true, /*ret=*/0, /*phy_list_size=*/1);
    // Arming both sites for one slot is a contract violation.
    EXPECT_TRUE(tracker_.has_violation());
    EXPECT_EQ(tracker_.stats(kRing).arm_count, 2);
}

TEST_F(TxDataOwnershipTest, AbandonPath_DisarmOnce_InlineReleaseOnce_NoCallback)
{
    // Abandon path (enqueue failure): launch never armed, so no prior arm here.
    nv::test::abandon_tx_data_h2d_state(tracker_, ring_, kRing);

    CountingReleaseTransport tw{};
    nv::test::inline_release_tx_data_if_not_deferred(ring_, store_, tw, kRing, &tracker_);
    EXPECT_EQ(tw.release_count.load(), 1);
    EXPECT_EQ(tracker_.stats(kRing).callback_count, 0);
    tracker_.assert_slot_lifecycle_clean(kRing);
}

TEST_F(TxDataOwnershipTest, SecondCallbackIsViolation)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    CountingReleaseTransport tw{};

    EXPECT_EQ(nv::test::callback_release_deferred_strict(tracker_, ring_, store_, tw, kRing),
              nv::test::CallbackGateResult::Released);
    EXPECT_EQ(nv::test::callback_release_deferred_strict(tracker_, ring_, store_, tw, kRing),
              nv::test::CallbackGateResult::Violation);
    EXPECT_EQ(tracker_.stats(kRing).callback_count, 2);
    EXPECT_TRUE(tracker_.has_violation());
}

TEST_F(TxDataOwnershipTest, CallbackWithoutArmIsViolation)
{
    CountingReleaseTransport tw{};
    EXPECT_EQ(nv::test::callback_release_deferred_strict(tracker_, ring_, store_, tw, kRing),
              nv::test::CallbackGateResult::Violation);
    EXPECT_EQ(tracker_.stats(kRing).callback_count, 1);
    EXPECT_TRUE(tracker_.has_violation());
}

TEST_F(TxDataOwnershipTest, DoubleArmIsViolation)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    tracker_.record_arm(kRing, nv::test::ArmSite::LaunchH2D);
    EXPECT_TRUE(tracker_.has_violation());
}

TEST_F(TxDataOwnershipTest, DirectMode_ArmWithoutEnqueueSuccessIsForbidden)
{
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/true, /*direct_mode=*/true, /*ret=*/-1, /*phy_list_size=*/1);
    EXPECT_EQ(tracker_.stats(kRing).arm_count, 0);
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/true, /*direct_mode=*/true, /*ret=*/0, /*phy_list_size=*/0);
    EXPECT_EQ(tracker_.stats(kRing).arm_count, 0);
}

TEST_F(TxDataOwnershipTest, NoReplay_ProcessPhyCommandsArmIsForbidden)
{
    nv::test::try_arm_direct_enqueue_success(
        tracker_, ring_, kRing, /*replay_build=*/false, /*direct_mode=*/true, /*ret=*/0, /*phy_list_size=*/2);
    EXPECT_EQ(tracker_.stats(kRing).arm_count, 0);
    EXPECT_FALSE(ring_.is_deferred(kRing));
}

TEST_F(TxDataOwnershipTest, NoReplay_EnqueueChannelTasksArmIsForbidden)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    tracker_.reset(kRing);
    nv::test::try_arm_enqueue_channel_tasks(tracker_, ring_, kRing, true);
    EXPECT_EQ(tracker_.stats(kRing).arm_site, nv::test::ArmSite::EnqueueChannelTasks);
}

TEST_F(TxDataOwnershipTest, AbandonThenCallbackIsViolation)
{
    nv::test::try_arm_launch_h2d_success(tracker_, ring_, kRing, true);
    nv::test::abandon_tx_data_h2d_state(tracker_, ring_, kRing);

    CountingReleaseTransport tw{};
    EXPECT_EQ(nv::test::callback_release_deferred_strict(tracker_, ring_, store_, tw, kRing),
              nv::test::CallbackGateResult::Violation);
}

TEST(DeferredTxDataRelease, ConcurrentCallbackAndInlineReleaseExactlyOnce)
{
    constexpr uint32_t kRing = 0U;
    nv::FapiSlotMessageStorage store(8);
    store.reset_for_slot(0x0002'0003u);
    ASSERT_EQ(store.store_message(make_tx_data_msg()), nv::StoreResult::Stored);

    nv::TxDataRingBuffer ring{};
    ring.set_deferred(kRing, true);

    CountingReleaseTransport tw{};
    std::atomic_int       inline_attempts{0};

    const auto inline_fn = [&ring, kRing, &store, &tw, &inline_attempts] {
        if(ring.is_deferred(kRing))
        {
            inline_attempts.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
    };

    const auto callback_fn = [&ring, kRing, &store, &tw] {
        if(!ring.is_deferred(kRing))
        {
            return;
        }
        store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
        ring.set_deferred(kRing, false);
        ring.reset(kRing);
    };

    std::jthread t1(inline_fn);
    std::jthread t2(callback_fn);

    // Join before asserting: release_count / store / ring state below are written
    // by the two threads, so reading them before join would both race (the
    // non-atomic store/ring reads vs. the callback thread's writes) and observe a
    // half-finished interleaving. After join, exactly-once must hold regardless of
    // which thread won — release_lane claims the lane atomically and the deferred
    // callback always performs the single release.
    t1.join();
    t2.join();

    EXPECT_EQ(tw.release_count.load(), 1);
    EXPECT_EQ(store.tx_data_count(), 0u);
    EXPECT_FALSE(ring.is_deferred(kRing));
    // Single-shot inline path: it checks is_deferred once and either records one
    // attempt (saw deferred) or performs a no-op release (saw the cleared flag),
    // so the attempt count is bounded by 1.
    EXPECT_LE(inline_attempts.load(), 1);
}

TEST(DeferredTxDataRelease, InlineSkipsWhileDeferred_CallbackReleasesOnce)
{
    constexpr uint32_t kRing = 2U;
    nv::FapiSlotMessageStorage store(8);
    store.reset_for_slot(0x0003'0004u);
    ASSERT_EQ(store.store_message(make_tx_data_msg()), nv::StoreResult::Stored);

    nv::TxDataRingBuffer             ring{};
    nv::test::TxDataOwnershipTracker tracker{};
    ring.set_deferred(kRing, true);

    CountingReleaseTransport tw{};
    nv::test::inline_release_tx_data_if_not_deferred(ring, store, tw, kRing);
    EXPECT_EQ(tw.release_count.load(), 0);
    EXPECT_EQ(store.tx_data_count(), 1u);

    EXPECT_EQ(nv::test::callback_release_deferred_strict(tracker, ring, store, tw, kRing),
              nv::test::CallbackGateResult::Released);
    EXPECT_EQ(store.tx_data_count(), 0u);
    tracker.assert_slot_lifecycle_clean(kRing);
}

TEST(DeferredTxDataRelease, DuplicateReleaseLaneIsNoOpAfterFirstClaim)
{
    nv::FapiSlotMessageStorage store(8);
    store.reset_for_slot(0x0004'0005u);
    ASSERT_EQ(store.store_message(make_tx_data_msg()), nv::StoreResult::Stored);

    DeletingReleaseTransport tw{};
    store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
    store.release_lane(nv::FapiSlotMessageStorage::MsgType::TX_DATA, tw);
    EXPECT_EQ(tw.delete_count.load(), 1);
    EXPECT_EQ(store.tx_data_count(), 0u);
}

TEST(DeferredTxDataRelease, AbandonPathClearsDeferredAndReleasesInline)
{
    nv::test::TxDataOwnershipTracker tracker{};
    constexpr uint32_t               kRing = 1U;
    nv::TxDataRingBuffer             ring{};
    nv::FapiSlotMessageStorage       store(8);
    store.reset_for_slot(0x0005'0006u);
    ASSERT_EQ(store.store_message(make_tx_data_msg()), nv::StoreResult::Stored);

    ring.set_deferred(kRing, true);
    nv::test::abandon_tx_data_h2d_state(tracker, ring, kRing);

    CountingReleaseTransport tw{};
    nv::test::inline_release_tx_data_if_not_deferred(ring, store, tw, kRing, &tracker);
    EXPECT_EQ(tw.release_count.load(), 1);
    EXPECT_EQ(tracker.stats(kRing).callback_count, 0);
}

TEST(DeferredTxDataRelease, MultithreadedInlineVsSingleCallbackStress)
{
    constexpr int kIterations = 200;

    for(int iter = 0; iter < kIterations; ++iter)
    {
        nv::FapiSlotMessageStorage store(8);
        store.reset_for_slot(0x0010'0000u + static_cast<uint32_t>(iter));
        ASSERT_EQ(store.store_message(make_tx_data_msg()), nv::StoreResult::Stored);

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
        // premature read. release_lane claims the lane atomically, so exactly one of
        // the two threads frees the buffers; the deferred callback always wins the
        // claim (deferred starts true, only the callback clears it), so the release
        // count is exactly 1 — EXPECT_EQ, not EXPECT_LE, which would also pass on a
        // silently-missing release.
        callback_thread.join();
        inline_thread.join();

        EXPECT_EQ(tw.release_count.load(), 1);
        EXPECT_EQ(store.tx_data_count(), 0u);
    }
}
