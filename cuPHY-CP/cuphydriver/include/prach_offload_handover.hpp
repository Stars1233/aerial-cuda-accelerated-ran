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

#ifndef PRACH_OFFLOAD_HANDOVER_HPP
#define PRACH_OFFLOAD_HANDOVER_HPP

#include "bit_utils.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

/**
 * Lifecycle/concurrency core of the offload PRACH reconfiguration handover.
 *
 * Owns only the phase/generation/pending state and the wait/commit/cancel
 * synchronization — it has no dependency on the driver, cuPHY, or CUDA, so it is
 * unit-testable on its own. Aggregator access (idle check + handle swap) is
 * injected by the caller through the @ref tryCommit drain callback, keeping the
 * PRACH lock and cuPHY handles outside this component.
 *
 * The atomic phase is a lock-free fast filter for the slot hook; the mutex-held
 * phase is authoritative. `pending_swap_mask_`/`pending_count_` mirror the legacy
 * `num_new_prach_handles` countdown (no heap allocation). Lock order at the call
 * site is fixed: this mutex first, then the PRACH aggregator lock.
 */
class PrachOffloadHandover final
{
public:
    /**
     * Phase of the offload active-cell PRACH reconfiguration handover.
     *
     * The `nonslot_lp` worker arms the handover and returns without blocking (legacy
     * fire-and-forget parity); the SLOT.IND slot hook drains an incremental per-aggregator
     * handle swap and publishes Committed when the last aggregator swaps. Cancelled is
     * reached only via cancel() (async reset/shutdown) or a wait() timeout -- and wait() is
     * retained for the unit tests / a future synchronous caller, not the production path.
     */
    enum class Phase
    {
        Idle,       //!< No handover in flight.
        Armed,      //!< Temp PRACH handles created; waiting for the incremental swap to drain.
        Committed,  //!< All aggregators swapped to the new handle; worker may delete old handles and respond OK.
        Cancelled,  //!< Handover aborted (timeout or async reset); worker responds failure.
    };

    /**
     * Outcome of a blocking wait() on the handover.
     *
     * @note The production reconfig path arms and returns (no wait); wait()/WaitResult are
     *       exercised only by the unit tests and kept for a possible future synchronous caller.
     */
    enum class WaitResult
    {
        Committed,  //!< Handover drained and committed; worker proceeds with OK.
        TimedOut,   //!< Bounded wait elapsed before commit; transitioned to Cancelled.
        Cancelled,  //!< Async cancel (reset/shutdown) or re-arm woke the wait before commit.
    };

    /** Constructs an idle handover (no swap in flight). */
    PrachOffloadHandover() = default;

    /** Non-copyable and non-movable: owns a mutex and condition variable. */
    PrachOffloadHandover(const PrachOffloadHandover&)            = delete;
    PrachOffloadHandover& operator=(const PrachOffloadHandover&) = delete;
    PrachOffloadHandover(PrachOffloadHandover&&)                 = delete;
    PrachOffloadHandover& operator=(PrachOffloadHandover&&)      = delete;

    /** Destroys the handover; no outstanding waiters expected. */
    ~PrachOffloadHandover() = default;

    /**
     * Arms a fresh handover for @p n_aggr PRACH aggregators.
     *
     * Marks every aggregator pending and bumps the generation. @p n_aggr must be
     * <= 64 (the caller guards this; larger counts saturate the mask).
     *
     * @param[in] n_aggr Number of PRACH aggregators that must swap.
     */
    void arm(const std::size_t n_aggr)
    {
        std::lock_guard lock(mutex_);
        pending_swap_mask_ = nv::lowBitsMask(n_aggr);
        pending_count_ = static_cast<std::int32_t>(n_aggr);
        phase_         = Phase::Armed;
        phase_atomic_.store(Phase::Armed, std::memory_order_release);
        ++generation_;
    }

    /**
     * Slot-boundary attempt to drain the incremental swap.
     *
     * Lock-free no-op unless a handover is armed. When armed, invokes @p drain
     * under the handover mutex; @p drain swaps each idle pending aggregator and
     * updates the mask/count in place. When the count reaches zero the handover
     * is committed and the blocked worker is notified (outside the lock).
     *
     * @param[in] drain Callable `void(std::uint64_t& pending_mask, std::int32_t& pending_count)`
     *            that performs the aggregator-lock-protected idle scan and swap.
     */
    template <std::invocable<std::uint64_t&, std::int32_t&> Drain>
    void tryCommit(Drain&& drain)
    {
        if(phase_atomic_.load(std::memory_order_acquire) != Phase::Armed)
        {
            return;
        }

        bool committed = false;
        {
            std::lock_guard lock(mutex_);
            if(phase_ == Phase::Armed)
            {
                drain(pending_swap_mask_, pending_count_);
                if(pending_count_ == 0)
                {
                    phase_ = Phase::Committed;
                    phase_atomic_.store(Phase::Committed, std::memory_order_release);
                    committed = true;
                }
            }
        }

        if(committed)
        {
            cv_.notify_all();
        }
    }

    /**
     * Blocks until the armed handover commits/cancels or the timeout elapses.
     *
     * On timeout, transitions to Cancelled so a later slot hook performs no
     * further swaps.
     *
     * @note Not called on the production path (the worker arms and returns); retained for the
     *       unit tests and a possible future synchronous reconfig caller.
     *
     * @param[in] timeout Bounded wait duration.
     * @return WaitResult::Committed if the handover committed; WaitResult::TimedOut
     *         if the bounded wait elapsed first; WaitResult::Cancelled on an async
     *         cancel or re-arm before commit.
     */
    [[nodiscard]] WaitResult wait(const std::chrono::nanoseconds timeout)
    {
        std::unique_lock lock(mutex_);
        const std::uint64_t gen = generation_;
        const bool ok = cv_.wait_for(lock, timeout, [this, gen] {
            return generation_ != gen ||
                   phase_ == Phase::Committed ||
                   phase_ == Phase::Cancelled;
        });

        if(!ok)
        {
            phase_ = Phase::Cancelled;
            phase_atomic_.store(Phase::Cancelled, std::memory_order_release);
            return WaitResult::TimedOut;
        }

        return (phase_ == Phase::Committed) ? WaitResult::Committed : WaitResult::Cancelled;
    }

    /**
     * Cancels an armed handover (async reset/reconnect/shutdown) and wakes the worker.
     *
     * Guarded Armed->Cancelled transition: a no-op when not armed, so it can never
     * downgrade a Committed handover.
     */
    void cancel()
    {
        bool cancelled = false;
        {
            std::lock_guard lock(mutex_);
            if(phase_ != Phase::Armed)
            {
                return;
            }
            phase_ = Phase::Cancelled;
            phase_atomic_.store(Phase::Cancelled, std::memory_order_release);
            cancelled = true;
        }
        if(cancelled)
        {
            cv_.notify_all();
        }
    }

    /**
     * Whether a handover is currently armed (lock-free).
     *
     * @return true if the phase is Armed.
     */
    [[nodiscard]] bool armed() const
    {
        return phase_atomic_.load(std::memory_order_acquire) == Phase::Armed;
    }

    /**
     * Current phase (lock-free snapshot; for inspection/tests).
     *
     * @return the phase as published by the atomic mirror.
     */
    [[nodiscard]] Phase phase() const
    {
        return phase_atomic_.load(std::memory_order_acquire);
    }

    /**
     * Number of aggregators still pending a swap (for tests/logging).
     *
     * @return the pending count under the handover mutex.
     */
    [[nodiscard]] std::int32_t pendingCount() const
    {
        std::lock_guard lock(mutex_);
        return pending_count_;
    }

private:
    // phase_atomic_ is a lock-free fast filter for the slot hook; phase_ (mutex-held)
    // is authoritative. Every release store to phase_atomic_ (arm/tryCommit/wait/cancel)
    // is made under mutex_ right after the matching phase_ write, and pairs with the
    // acquire loads in tryCommit (fast filter), armed(), and phase().
    mutable std::mutex      mutex_;                     //!< Guards phase_/pending_/generation_; first in lock order (before the PRACH aggr lock).
    std::condition_variable cv_;                        //!< Wakes the blocked worker on commit/cancel.
    Phase                   phase_{Phase::Idle};         //!< Authoritative phase (mutex-protected).
    std::atomic<Phase>      phase_atomic_{Phase::Idle};  //!< Lock-free mirror of phase_ for the slot-hook fast filter (see note above).
    std::uint64_t           pending_swap_mask_{};        //!< One bit per aggregator still pending a swap.
    std::int32_t            pending_count_{};             //!< Aggregators still pending (mirrors the mask's popcount).
    std::uint64_t           generation_{};               //!< Bumped on each arm(); lets wait() detect a re-arm.
};

#endif // PRACH_OFFLOAD_HANDOVER_HPP
