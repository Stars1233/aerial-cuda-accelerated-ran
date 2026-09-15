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
 * @file nv_phy_non_slot_dispatch.hpp
 * @brief Declarations for the low-priority non-slot FAPI dispatch substrate.
 *
 * The low-priority (nonslot_lp) dispatch substrate for the store-first FAPI path: it owns
 * the worker thread, bounded SPSC queue, descriptor ownership transfer, and lifecycle
 * hooks that carry CONFIG/START/STOP off the message-processing thread.
 */

#if !defined(NV_PHY_NON_SLOT_DISPATCH_HPP_INCLUDED_)
#define NV_PHY_NON_SLOT_DISPATCH_HPP_INCLUDED_

#include "nv_phy_mac_transport.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>

namespace nv
{

class PHY_module;

/**
 * @brief Thread placement and priority settings for the low-priority worker itself.
 */
struct NonSlotLpWorkerConfig final {
    /// Effective worker thread configuration. Name is forced to "nonslot_lp".
    std::string name{"nonslot_lp"};
    /// CPU core to bind the worker to, or -1 when no affinity is configured.
    int cpu_affinity{-1};
    /// Worker scheduling priority, or 0 when default scheduling is used.
    int sched_priority{0};
};

/**
 * @brief Thread placement and priority settings for the low-priority non-slot worker.
 */
struct NonSlotLpThreadConfig final {
    /// Effective worker thread configuration.
    NonSlotLpWorkerConfig worker_cfg{};
};

/**
 * @brief Runtime callbacks used by @c NonSlotLpExecutor.
 *
 * Production binds these callbacks to @c PHY_module. Tests can bind them to a
 * small fake runtime so the full queue/thread/release path can be validated
 * without constructing the complete PHY stack.
 */
struct NonSlotLpRuntime final {
    /// Opaque caller-owned runtime object passed back to callbacks.
    void* context{nullptr};
    /// Return the number of valid cell ids currently addressable by the executor.
    std::size_t (*cell_count)(void* context) noexcept{nullptr};
    /// Process one descriptor owned by the executor.
    void (*process_item)(void* context, std::uint16_t cell_id, phy_mac_msg_desc& msg){nullptr};
    /// Release one RX descriptor owned by the executor.
    void (*rx_release)(void* context, phy_mac_msg_desc& msg) noexcept{nullptr};
};

/**
 * Single source of truth for which non-slot FAPI messages the nonslot_lp worker owns.
 *
 * Returns true for exactly CONFIG/START/STOP. The dispatch gate in nv_phy_slot_dispatch.cpp
 * (is_parallel_nonslot_msg) delegates here, so the routing gate and the executor's ownership
 * filter cannot diverge. Declared here so the routing set is unit-testable.
 *
 * @param[in] msg_id SCF FAPI message type id to classify.
 * @return true if the message is owned by the nonslot_lp worker (CONFIG/START/STOP), false
 *         otherwise. Return value must be checked.
 */
[[nodiscard]] bool is_nonslot_lp_message(int32_t msg_id) noexcept;

/**
 * @brief Low-priority executor for CONFIG/START/STOP non-slot FAPI work.
 *
 * Provides the execution substrate: a 64-entry bounded SPSC queue, ownership-transfer
 * helper, worker lifecycle, wakeup, and shutdown drain logic. CONFIG/START/STOP
 * messages are routed here, off the message-processing thread, in the store-first path.
 *
 * @thread_safety Exactly one producer is expected: the FAPI message-processing
 * thread. Exactly one consumer is owned by this class: the @c nonslot_lp worker.
 */
class NonSlotLpExecutor final {
public:
    /// Maximum number of queued non-slot descriptors accepted by the executor.
    static constexpr std::size_t kQueueCapacity = 64;
    /// Ring storage slots. One extra slot distinguishes full from empty.
    static constexpr std::size_t kQueueStorageSlots = kQueueCapacity + 1;
    /// Maximum cell ids tracked for duplicate CONFIG suppression.
    static constexpr std::size_t kMaxTrackedConfigCells = 64;
    /// Queue depth at which the executor emits a one-shot saturation warning.
    static constexpr std::size_t kSaturationThreshold = (kQueueCapacity * 3) / 4;
    /// Physical cache-line size used to pad SPSC head/tail indexes to separate lines.
    /// Hard-coded to 64 bytes — correct for all supported aarch64/x86-64 targets —
    /// to avoid the GCC -Werror=interference-size diagnostic raised by
    /// std::hardware_destructive_interference_size (whose value GCC ties to the
    /// prefetch stride, not the physical cache-line size).
    static constexpr std::size_t kCacheLineSize = 64;

    /**
     * @brief Construct an executor bound to a PHY module.
     * @param module Owning module used for transport release and PHY instance validation.
     * @param config Worker placement and diagnostic configuration.
     */
    NonSlotLpExecutor(PHY_module& module, NonSlotLpThreadConfig config);
    /**
     * @brief Construct an executor with explicit runtime callbacks.
     * @param runtime Cell-count and RX-release callbacks.
     * @param config Worker placement and diagnostic configuration.
     */
    NonSlotLpExecutor(NonSlotLpRuntime runtime, NonSlotLpThreadConfig config);
    /**
     * @brief Stop the worker and release queued descriptors before destruction.
     */
    ~NonSlotLpExecutor();

    /// Non-copyable: the executor owns worker state and descriptor ownership.
    NonSlotLpExecutor(const NonSlotLpExecutor&) = delete;
    NonSlotLpExecutor& operator=(const NonSlotLpExecutor&) = delete;
    /// Non-movable: the worker thread, wait primitives, and module binding are stable in place.
    NonSlotLpExecutor(NonSlotLpExecutor&&) = delete;
    NonSlotLpExecutor& operator=(NonSlotLpExecutor&&) = delete;

    /**
     * @brief Start the low-priority worker if it is not already running.
     * @note Safe to call multiple times; later calls are no-ops while running.
     */
    void start();
    /**
     * @brief Stop the worker, wait for it to exit, and release queued descriptors.
     * @note Queued work is released, not processed, during shutdown.
     */
    void stop();
    /**
     * @brief Transfer one CONFIG/START/STOP descriptor into the worker queue.
     * @param smsg RX descriptor owned by the caller on entry.
     * @return true when ownership was transferred; false when caller still owns @p smsg.
     * @note On success @p smsg is reset so only the queued copy owns the RX buffers.
     */
    [[nodiscard]] bool enqueue(phy_mac_msg_desc& smsg);
    /**
     * @brief Current queued descriptor count.
     * @return Approximate SPSC queue depth observed through atomic head/tail indexes.
     */
    [[nodiscard]] std::size_t depth() const noexcept;
    /**
     * @brief Return the effective thread configuration used to construct this executor.
     * @return Immutable configuration snapshot, used when @c PHY_module is move-constructed.
     */
    [[nodiscard]] const NonSlotLpThreadConfig& thread_config() const noexcept;
    /**
     * @brief Return true when the worker thread is alive (started and not yet joined).
     *
     * Used by @c PHY_module::move_nonslot_lp_worker_from() to determine whether an
     * in-flight item may still be executing inside the worker. @c depth()==0 alone is
     * not a sufficient guard because an item already popped from the ring into the
     * worker's local variable is invisible to @c depth().
     */
    [[nodiscard]] bool is_running() const noexcept;

private:
    /**
     * @brief Queue payload owned by @c NonSlotLpExecutor after successful enqueue.
     */
    struct WorkItem final {
        /// Validated PHY cell id associated with @c msg.
        uint16_t cell_id{0};
        /// RX descriptor whose buffers are released by the worker or shutdown drain.
        phy_mac_msg_desc msg{};
    };

    /**
     * @brief Preallocated bounded single-producer/single-consumer ring.
     *
     * Uses release/acquire ordering to publish descriptor ownership from the
     * message-processing thread to the @c nonslot_lp worker. It intentionally
     * avoids mutex-protected std::queue on the receive hot path.
     */
    class SpscQueue final {
    public:
        SpscQueue() = default;
        /// Non-copyable: queue slots and atomic indexes have single producer/consumer ownership.
        SpscQueue(const SpscQueue&) = delete;
        SpscQueue& operator=(const SpscQueue&) = delete;
        /// Non-movable: queue storage and indexes must remain stable while the worker is active.
        SpscQueue(SpscQueue&&) = delete;
        SpscQueue& operator=(SpscQueue&&) = delete;

        /**
         * @brief Push an item when space is available.
         * @return true on success; false when the ring is full.
         */
        [[nodiscard]] bool try_push(const WorkItem& item) noexcept;
        /**
         * @brief Pop the oldest queued item.
         * @return true on success; false when the ring is empty.
         */
        [[nodiscard]] bool try_pop(WorkItem& item) noexcept;
        /**
         * @brief Test whether the queue is empty.
         */
        [[nodiscard]] bool empty() const noexcept;
        /**
         * @brief Current queued item count.
         */
        [[nodiscard]] std::size_t size() const noexcept;

    private:
        /**
         * @brief Advance a ring index with wraparound.
         */
        [[nodiscard]] static constexpr std::size_t next_index(std::size_t index) noexcept
        {
            return (index + 1) % kQueueStorageSlots;
        }

        /// Preallocated descriptor storage; no allocation occurs in enqueue/dequeue.
        std::array<WorkItem, kQueueStorageSlots> entries_{};
        /// Consumer-owned read index, observed by producer for fullness checks.
        /// Cache-line aligned to prevent false sharing with tail_ on the producer core.
        alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};
        /// Producer-owned write index, observed by consumer for emptiness checks.
        /// Cache-line aligned to prevent false sharing with head_ on the consumer core.
        alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};
    };

    /**
     * @brief Worker thread body.
     * @param stop_token Cooperative stop token owned by @c std::jthread.
     */
    void thread_func(std::stop_token stop_token);
    /**
     * @brief Apply worker name, affinity, and priority, then log actual placement.
     */
    void configure_thread();
    /**
     * @brief Handle one dequeued item.
     * @note Handler failures are logged locally; descriptor release is always completed.
     */
    void process_item(WorkItem& item);
    /**
     * @brief Mark a CONFIG as queued/executing for duplicate suppression.
     */
    [[nodiscard]] bool mark_config_inflight(const phy_mac_msg_desc& smsg) noexcept;
    /**
     * @brief Clear the CONFIG duplicate-suppression mark for a finished item.
     */
    void clear_config_inflight(const WorkItem& item) noexcept;
    /**
     * @brief Release a duplicate CONFIG retry already consumed by this executor.
     */
    void release_duplicate_config(phy_mac_msg_desc& smsg) noexcept;
    /**
     * @brief Release an owned RX descriptor and reset the local copy.
     */
    void release_item(WorkItem& item);
    /**
     * @brief Release all queued descriptors without processing them.
     */
    void drain();

    /// Runtime callback bundle bound at construction.
    NonSlotLpRuntime runtime_{};
    /// Effective worker configuration and startup diagnostic context.
    NonSlotLpThreadConfig config_{};
    /// Bounded SPSC descriptor queue.
    SpscQueue queue_{};
    /// Long-lived low-priority worker.
    std::jthread thread_{};
    /// Mutex used only for condition-variable sleep/wakeup, not queue data transfer.
    std::mutex wakeup_mutex_{};
    /// Worker wakeup primitive paired with @c std::stop_token.
    std::condition_variable_any cv_{};
    /// One-shot overflow log guard.
    std::atomic<bool> first_overflow_logged_{false};
    /// One-shot saturation log guard, reset after the queue drains below threshold.
    std::atomic<bool> saturation_logged_{false};
    /// Per-cell CONFIG queued/executing marks used to suppress testmac retries.
    std::array<std::atomic<bool>, kMaxTrackedConfigCells> config_inflight_{};
};

} // namespace nv

#endif
