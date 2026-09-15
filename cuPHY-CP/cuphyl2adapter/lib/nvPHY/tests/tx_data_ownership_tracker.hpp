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

#ifndef TX_DATA_OWNERSHIP_TRACKER_HPP
#define TX_DATA_OWNERSHIP_TRACKER_HPP

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "nv_fapi_message_storage.hpp"
#include "nv_tx_data_ring_buffer.hpp"

namespace nv::test
{

/** @brief Test transport that counts rx_release invocations (TSan-safe). */
struct CountingReleaseTransport final
{
    std::atomic_int release_count{0};

    /**
     * @brief Record one release (increments release_count).
     * @param[in] msg  Released descriptor; unused (the mock only counts).
     */
    void rx_release([[maybe_unused]] nv::phy_mac_msg_desc& msg) noexcept
    {
        release_count.fetch_add(1, std::memory_order_relaxed);
    }
};

/** @brief Test transport that counts releases and nulls message buffers (ASan helper). */
struct DeletingReleaseTransport final
{
    std::atomic_int delete_count{0};

    /**
     * @brief Record one release and null the descriptor's buffers.
     * @param[in,out] msg  Descriptor whose msg_buf/data_buf are cleared to catch
     *                     use-after-release under ASan.
     */
    void rx_release(nv::phy_mac_msg_desc& msg) noexcept
    {
        delete_count.fetch_add(1, std::memory_order_relaxed);
        msg.msg_buf  = nullptr;
        msg.data_buf = nullptr;
    }
};

enum class ArmSite : uint8_t
{
    None,
    LaunchH2D,
    DirectEnqueue,
    EnqueueChannelTasks,
};

struct SlotLifecycleStats final
{
    int     arm_count{0};            //!< Number of deferred arms recorded for this slot.
    ArmSite arm_site{ArmSite::None}; //!< Which production arm site fired (at most one).
    int     callback_count{0};       //!< Deferred-release callback invocations.
    int     disarm_count{0};        //!< Times deferred was cleared after callback release.
    int     abandon_count{0};       //!< Abandon-path disarms without callback.
    int     inline_release_count{0}; //!< Inline release_lane calls when not deferred.
};

/**
 * @brief Test-only helper that records TX_DATA ownership transitions per ring slot.
 *
 * Used by sanitizer tests to assert exactly-once arm / callback / disarm semantics
 * without instantiating PHY_module.
 *
 * @warning Not thread-safe. All record_*, reset, and stats calls must occur on a single
 *          thread. The concurrent stress tests deliberately exercise only
 *          TxDataRingBuffer / FapiSlotMessageStorage / the transport mock across
 *          threads and never touch this tracker from a spawned thread. If you add
 *          tracker calls to a multithreaded test, convert the SlotLifecycleStats
 *          fields and violation_ to std::atomic (or guard with a mutex) first.
 */
class TxDataOwnershipTracker final
{
public:
    /**
     * @brief Clear all recorded stats for one ring slot.
     * @param[in] ring_idx  Ring slot index to reset.
     */
    void reset(uint32_t ring_idx)
    {
        slots_[ring_idx] = {};
    }

    /**
     * @brief Record a deferred arm and flag a violation on a second arm or a
     *        conflicting arm site for the same slot.
     * @param[in] ring_idx  Ring slot index being armed.
     * @param[in] site      Production arm site that fired.
     */
    void record_arm(uint32_t ring_idx, ArmSite site)
    {
        auto& s = slots_[ring_idx];
        ++s.arm_count;
        if(s.arm_site == ArmSite::None)
        {
            s.arm_site = site;
        }
        else if(s.arm_site != site)
        {
            violation_ = true;
        }
        if(s.arm_count > 1)
        {
            violation_ = true;
        }
    }

    /** @brief Record a deferred-release callback invocation. @param[in] ring_idx Ring slot index. */
    void record_callback(uint32_t ring_idx) { ++slots_[ring_idx].callback_count; }

    /** @brief Record a disarm (deferred cleared after callback release). @param[in] ring_idx Ring slot index. */
    void record_disarm(uint32_t ring_idx) { ++slots_[ring_idx].disarm_count; }

    /** @brief Record an abandon-path disarm (no callback). @param[in] ring_idx Ring slot index. */
    void record_abandon(uint32_t ring_idx) { ++slots_[ring_idx].abandon_count; }

    /** @brief Record an inline release_lane call on a non-deferred slot. @param[in] ring_idx Ring slot index. */
    void record_inline_release(uint32_t ring_idx) { ++slots_[ring_idx].inline_release_count; }

    /** @brief Read the recorded stats for a ring slot. @param[in] ring_idx Ring slot index. @return Const reference to the slot's stats. */
    [[nodiscard]] const SlotLifecycleStats& stats(uint32_t ring_idx) const { return slots_[ring_idx]; }

    /** @brief Whether any contract violation has been recorded. @return true if a violation was seen. */
    [[nodiscard]] bool has_violation() const { return violation_; }

    // Record a contract breach observed by a caller (e.g. a callback that fired
    // on a slot that was not deferred). Surfaces via has_violation() / the asserts.
    void record_violation() { violation_ = true; }

    /**
     * @brief Assert the slot armed exactly once at the build-appropriate site.
     * @param[in] replay_build  True when the replay build flavor is active.
     * @param[in] direct_mode   True when fapi-to-cplane direct mode is active.
     * @param[in] ring_idx      Ring slot index to check.
     */
    void assert_arm_contract(bool replay_build, bool direct_mode, uint32_t ring_idx) const
    {
        const auto& s = slots_[ring_idx];
        ASSERT_EQ(s.arm_count, 1) << "ring_idx=" << ring_idx;
        if(replay_build && direct_mode)
        {
            EXPECT_EQ(s.arm_site, ArmSite::DirectEnqueue) << "ring_idx=" << ring_idx;
        }
        else
        {
            EXPECT_EQ(s.arm_site, ArmSite::LaunchH2D) << "ring_idx=" << ring_idx;
        }
        EXPECT_NE(s.arm_site, ArmSite::EnqueueChannelTasks) << "ring_idx=" << ring_idx;
        EXPECT_FALSE(has_violation());
    }

    /**
     * @brief Assert a slot's recorded lifecycle is internally consistent and
     *        violation-free (abandon-only, arm→callback→disarm, or balanced).
     * @param[in] ring_idx  Ring slot index to check.
     */
    void assert_slot_lifecycle_clean(uint32_t ring_idx) const
    {
        const auto& s = slots_[ring_idx];
        if(s.abandon_count > 0)
        {
            EXPECT_EQ(s.abandon_count, 1);
            EXPECT_EQ(s.callback_count, 0);
            EXPECT_GE(s.inline_release_count, 1);
            EXPECT_EQ(s.arm_count, 0);
            return;
        }

        if(s.arm_count > 0)
        {
            EXPECT_EQ(s.arm_count, 1);
            EXPECT_EQ(s.callback_count, 1);
            EXPECT_EQ(s.disarm_count, 1);
            EXPECT_EQ(s.abandon_count, 0);
        }
        else
        {
            // No arm and no abandon: any callback that fired must have completed a
            // matching disarm (the Released path). An unbalanced callback — e.g. a
            // callback on a non-deferred slot — is a breach and also trips
            // has_violation() below.
            EXPECT_EQ(s.callback_count, s.disarm_count);
        }
        EXPECT_FALSE(has_violation());
    }

    /**
     * @brief Whether the slot's recorded arm site is one production must never use.
     * @param[in] ring_idx  Ring slot index to check.
     * @return true if the arm site is ArmSite::EnqueueChannelTasks.
     */
    [[nodiscard]] bool arm_site_is_forbidden(uint32_t ring_idx) const
    {
        return slots_[ring_idx].arm_site == ArmSite::EnqueueChannelTasks;
    }

private:
    std::array<SlotLifecycleStats, SLOT_STORAGE_DEPTH> slots_{};
    bool                                               violation_{false};
};

// ---------------------------------------------------------------------------
// Production gate shims (deterministic, no PHY_module / CUDA).
// ---------------------------------------------------------------------------

/**
 * @brief Model the launch-H2D arm site: arm the slot iff the H2D launch succeeded.
 * @param[in,out] tracker   Ownership tracker to record the arm in.
 * @param[in,out] ring      Ring buffer whose deferred flag is set on success.
 * @param[in]     ring_idx  Ring slot index.
 * @param[in]     launch_ok True when l1_launch_tb_h2d succeeded.
 */
inline void try_arm_launch_h2d_success(TxDataOwnershipTracker& tracker,
                                       TxDataRingBuffer&       ring,
                                       uint32_t                ring_idx,
                                       bool                    launch_ok)
{
    if(!launch_ok)
    {
        return;
    }
    ring.set_deferred(ring_idx, true);
    tracker.record_arm(ring_idx, ArmSite::LaunchH2D);
}

/**
 * @brief Model the direct-enqueue arm site: arm only in a replay build, direct
 *        mode, on a successful enqueue with a non-empty PHY list.
 * @param[in,out] tracker        Ownership tracker to record the arm in.
 * @param[in,out] ring           Ring buffer whose deferred flag is set on success.
 * @param[in]     ring_idx       Ring slot index.
 * @param[in]     replay_build   True when the replay build flavor is active.
 * @param[in]     direct_mode    True when fapi-to-cplane direct mode is active.
 * @param[in]     ret            l1_enqueue_phy_work return code (0 == success).
 * @param[in]     phy_list_size  Size of the enqueued PHY list.
 */
inline void try_arm_direct_enqueue_success(TxDataOwnershipTracker& tracker,
                                           TxDataRingBuffer&       ring,
                                           uint32_t                ring_idx,
                                           bool                    replay_build,
                                           bool                    direct_mode,
                                           int                     ret,
                                           int                     phy_list_size)
{
    if(!replay_build || !direct_mode || ret != 0 || phy_list_size <= 0)
    {
        return;
    }
    ring.set_deferred(ring_idx, true);
    tracker.record_arm(ring_idx, ArmSite::DirectEnqueue);
}

/**
 * @brief Model a forbidden arm from enqueue_channel_tasks (GT-12162 regression
 *        shape) so tests can assert the tracker flags it as a forbidden site.
 * @param[in,out] tracker         Ownership tracker to record the arm in.
 * @param[in,out] ring            Ring buffer whose deferred flag is set.
 * @param[in]     ring_idx        Ring slot index.
 * @param[in]     split_phase_ok  True to simulate the (forbidden) arm firing.
 */
inline void try_arm_enqueue_channel_tasks(TxDataOwnershipTracker& tracker,
                                          TxDataRingBuffer&       ring,
                                          uint32_t                ring_idx,
                                          bool                    split_phase_ok)
{
    if(split_phase_ok)
    {
        ring.set_deferred(ring_idx, true);
        tracker.record_arm(ring_idx, ArmSite::EnqueueChannelTasks);
    }
}

/** @brief Outcome of callback_release_deferred_strict(). */
enum class CallbackGateResult : uint8_t
{
    Released,  //!< Slot was deferred; the lane was released and disarmed.
    Violation, //!< Callback fired on a non-deferred slot (recorded as a violation).
};

/**
 * @brief Model the strict deferred-release callback: release + disarm iff the slot
 *        is deferred, otherwise record a violation and release nothing.
 * @tparam Transport        Type satisfying the FapiTransport concept.
 * @param[in,out] tracker   Ownership tracker to record the callback/disarm/violation in.
 * @param[in,out] ring      Ring buffer gating the release (deferred flag).
 * @param[in,out] store     Slot message storage whose TX_DATA lane is released.
 * @param[in,out] transport Transport used to release each stored buffer.
 * @param[in]     ring_idx  Ring slot index.
 * @return CallbackGateResult::Released on the deferred path, else Violation.
 */
template <FapiTransport Transport>
[[nodiscard]] CallbackGateResult callback_release_deferred_strict(TxDataOwnershipTracker& tracker,
                                                           TxDataRingBuffer&       ring,
                                                           FapiSlotMessageStorage& store,
                                                           Transport&              transport,
                                                           uint32_t                ring_idx)
{
    tracker.record_callback(ring_idx);
    if(!ring.is_deferred(ring_idx))
    {
        tracker.record_violation();
        return CallbackGateResult::Violation;
    }

    store.release_lane(FapiSlotMessageStorage::MsgType::TX_DATA, transport);
    ring.set_deferred(ring_idx, false);
    tracker.record_disarm(ring_idx);
    ring.reset(ring_idx);
    return CallbackGateResult::Released;
}

/**
 * @brief Model the inline release path: release the TX_DATA lane only when the
 *        slot is not deferred (ownership was handed back to the inline thread).
 * @tparam Transport        Type satisfying the FapiTransport concept.
 * @param[in,out] ring      Ring buffer gating the release (deferred flag).
 * @param[in,out] store     Slot message storage whose TX_DATA lane is released.
 * @param[in,out] transport Transport used to release each stored buffer.
 * @param[in]     ring_idx  Ring slot index.
 * @param[in,out] tracker   Optional tracker; records the inline release when non-null.
 */
template <FapiTransport Transport>
void inline_release_tx_data_if_not_deferred(TxDataRingBuffer&       ring,
                                                   FapiSlotMessageStorage& store,
                                                   Transport&              transport,
                                                   uint32_t                ring_idx,
                                                   TxDataOwnershipTracker* tracker = nullptr)
{
    if(ring.is_deferred(ring_idx))
    {
        return;
    }
    store.release_lane(FapiSlotMessageStorage::MsgType::TX_DATA, transport);
    if(tracker != nullptr)
    {
        tracker->record_inline_release(ring_idx);
    }
}

/**
 * @brief Model the abandon path (enqueue failure / ring collision): clear the
 *        deferred flag and staging so the inline path owns release, and record it.
 * @param[in,out] tracker   Ownership tracker to record the abandon in.
 * @param[in,out] ring      Ring buffer whose deferred flag and staging are cleared.
 * @param[in]     ring_idx  Ring slot index.
 */
inline void abandon_tx_data_h2d_state(TxDataOwnershipTracker& tracker,
                                      TxDataRingBuffer&       ring,
                                      uint32_t                ring_idx)
{
    ring.set_deferred(ring_idx, false);
    ring.reset(ring_idx);
    tracker.record_abandon(ring_idx);
}

} // namespace nv::test

#endif // TX_DATA_OWNERSHIP_TRACKER_HPP
