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
 * @file nv_phy_non_slot_dispatch.cpp
 * @brief Low-priority non-slot FAPI execution substrate for the store-first path.
 *
 * This translation unit is compiled only when ENABLE_FAPI_STORE_REPLAY is ON. It
 * provides the worker, queue, ownership-transfer, wakeup, and shutdown plumbing that
 * carries CONFIG/START/STOP off the message-processing thread.
 */

#include "nv_phy_non_slot_dispatch.hpp"
// TODO: move all #if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST) blocks below
// into nv_phy_non_slot_dispatch_bindings.cpp and remove this seam — see Phase 2 plan.
#if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)
#include "nv_phy_module.hpp"
#endif
#include "scf_5g_fapi.h"

#include "memtrace.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <exception>
#include <gsl-lite/gsl-lite.hpp>
#include <limits>
#include <memory>
#include <sched.h>
#include <system_error>
#include <type_traits>
#include <utility>

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 6) // "L2A.MODULE"

namespace nv
{

namespace {

constexpr const char* NONSLOT_LP_THREAD_NAME = "nonslot_lp";

#if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)
/**
 * @brief Return the current number of PHY instances from the bound module.
 */
[[nodiscard]] std::size_t phy_module_cell_count(void* context) noexcept
{
    auto* module = static_cast<PHY_module*>(context);
    return module != nullptr ? module->PHY_instances().size() : 0;
}

/**
 * @brief Release an RX descriptor through the bound module transport wrapper.
 */
void phy_module_rx_release(void* context, phy_mac_msg_desc& msg) noexcept
{
    auto* module = static_cast<PHY_module*>(context);
    if (module != nullptr)
    {
        module->transport_wrapper().rx_release(msg);
    }
}

void phy_module_process_item(void* context, std::uint16_t cell_id, phy_mac_msg_desc& msg)
{
    auto* module = static_cast<PHY_module*>(context);
    if (module == nullptr || cell_id >= module->PHY_instances().size())
    {
        NVLOGW_FMT(TAG,
                   "{}: invalid runtime binding for msg_id=0x{:02X} cell_id={}; releasing",
                   __func__, msg.msg_id, cell_id);
        return;
    }

    auto& phy_instance = module->PHY_instances()[cell_id].get();
    switch (msg.msg_id)
    {
    case SCF_FAPI_CONFIG_REQUEST:
        phy_instance.on_config_request_offload(cell_id, msg);
        return;
    case SCF_FAPI_START_REQUEST:
        phy_instance.on_cell_start_request_offload(cell_id);
        return;
    case SCF_FAPI_STOP_REQUEST:
        phy_instance.on_cell_stop_request_offload(cell_id);
        return;
    default:
        gsl_Expects(false);
    }
}

/**
 * @brief Build the production runtime callback bundle for a PHY module.
 */
[[nodiscard]] NonSlotLpRuntime make_phy_module_runtime(PHY_module& module) noexcept
{
    return NonSlotLpRuntime{&module, phy_module_cell_count, phy_module_process_item, phy_module_rx_release};
}
#endif

#if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)
template <typename T>
[[nodiscard]] constexpr T safe_left_shift(const T value, const std::size_t shift) noexcept
{
    static_assert(std::is_unsigned_v<T>, "safe_left_shift expects an unsigned integer type");
    return shift < static_cast<std::size_t>(std::numeric_limits<T>::digits) ? static_cast<T>(value << shift) : T{0};
}

[[nodiscard]] uint64_t safe_cell_bit(const uint16_t cell_id) noexcept
{
    return safe_left_shift(uint64_t{1}, cell_id);
}
#endif

} // namespace

// CONFIG/START/STOP are the messages the nonslot_lp worker owns. Single source of truth:
// is_parallel_nonslot_msg() in nv_phy_slot_dispatch.cpp delegates here so the dispatch gate
// and this executor-side filter can never disagree.
bool is_nonslot_lp_message(int32_t msg_id) noexcept
{
    switch (msg_id)
    {
    case SCF_FAPI_CONFIG_REQUEST:
    case SCF_FAPI_START_REQUEST:
    case SCF_FAPI_STOP_REQUEST:
        return true;
    default:
        return false;
    }
}

bool NonSlotLpExecutor::SpscQueue::try_push(const WorkItem& item) noexcept
{
    // RELAXED: producer owns tail_ exclusively; no cross-thread synchronisation needed
    // for the read-own-index pattern.
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto next_tail = next_index(tail);
    // ACQUIRE: pairs with head_.store(release) in try_pop() — ensures the consumer's
    // advance of head_ is visible before we conclude the queue is full.
    if (next_tail == head_.load(std::memory_order_acquire))
    {
        return false;
    }

    entries_[tail] = item;
    // RELEASE: pairs with tail_.load(acquire) in try_pop() — publishes the written
    // entry to the consumer before exposing the updated tail index.
    tail_.store(next_tail, std::memory_order_release);
    return true;
}

bool NonSlotLpExecutor::SpscQueue::try_pop(WorkItem& item) noexcept
{
    // RELAXED: consumer owns head_ exclusively; no cross-thread synchronisation needed
    // for the read-own-index pattern.
    const auto head = head_.load(std::memory_order_relaxed);
    // ACQUIRE: pairs with tail_.store(release) in try_push() — ensures the producer's
    // written entry is visible before we read entries_[head].
    if (head == tail_.load(std::memory_order_acquire))
    {
        return false;
    }

    item = entries_[head];
    entries_[head] = WorkItem{};
    // RELEASE: pairs with head_.load(acquire) in try_push() — publishes the slot as
    // free to the producer before exposing the updated head index.
    head_.store(next_index(head), std::memory_order_release);
    return true;
}

bool NonSlotLpExecutor::SpscQueue::empty() const noexcept
{
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
}

std::size_t NonSlotLpExecutor::SpscQueue::size() const noexcept
{
    const auto head = head_.load(std::memory_order_acquire);
    const auto tail = tail_.load(std::memory_order_acquire);
    if (tail >= head)
    {
        return tail - head;
    }
    return kQueueStorageSlots - head + tail;
}

#if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)
NonSlotLpExecutor::NonSlotLpExecutor(PHY_module& module, NonSlotLpThreadConfig config) :
    NonSlotLpExecutor(make_phy_module_runtime(module), std::move(config))
{
}
#endif

NonSlotLpExecutor::NonSlotLpExecutor(NonSlotLpRuntime runtime, NonSlotLpThreadConfig config) :
    runtime_(runtime),
    config_(std::move(config))
{
}

NonSlotLpExecutor::~NonSlotLpExecutor()
{
    stop();
}

void NonSlotLpExecutor::start()
{
    if (thread_.joinable())
    {
        return;
    }
    // RELEASE: pairs with exchange(acq_rel) in enqueue() — resets one-shot guard before worker restarts.
    first_overflow_logged_.store(false, std::memory_order_release);
    // RELEASE: pairs with exchange(acq_rel) in enqueue() and store(release) in thread_func().
    saturation_logged_.store(false, std::memory_order_release);
    thread_ = std::jthread([this](std::stop_token stop_token) {
        thread_func(stop_token);
    });
    NVLOGI_FMT(TAG,
               "{}: started nonslot_lp worker capacity={} saturation_threshold={}",
               __func__, kQueueCapacity, kSaturationThreshold);
}

void NonSlotLpExecutor::stop()
{
    if (thread_.joinable())
    {
        NVLOGI_FMT(TAG,
                   "{}: stopping nonslot_lp worker queued_depth={}",
                   __func__, queue_.size());
        thread_.request_stop();
        cv_.notify_all();
        thread_.join();
    }
    drain();
}

bool NonSlotLpExecutor::enqueue(phy_mac_msg_desc& smsg)
{
    if (!is_nonslot_lp_message(smsg.msg_id))
    {
        NVLOGW_FMT(TAG,
                   "{}: rejected unsupported msg_id=0x{:02X} cell_id={}",
                   __func__, smsg.msg_id, smsg.cell_id);
        return false;
    }
    if (smsg.cell_id < 0)
    {
        NVLOGW_FMT(TAG,
                   "{}: rejected invalid cell_id={} msg_id=0x{:02X}",
                   __func__, smsg.cell_id, smsg.msg_id);
        return false;
    }
    const auto cell_id = static_cast<std::size_t>(smsg.cell_id);
    const auto cell_count = runtime_.cell_count != nullptr ? runtime_.cell_count(runtime_.context) : 0;
    if (cell_id >= cell_count)
    {
        NVLOGW_FMT(TAG,
                   "{}: rejected out-of-range cell_id={} msg_id=0x{:02X} phy_refs_size={}",
                   __func__, smsg.cell_id, smsg.msg_id, cell_count);
        return false;
    }
    if (!mark_config_inflight(smsg))
    {
        NVLOGW_FMT(TAG,
                   "{}: suppressed duplicate CONFIG retry cell_id={} depth={}",
                   __func__, smsg.cell_id, queue_.size());
        release_duplicate_config(smsg);
        return true;
    }

    WorkItem item{};
    item.cell_id = static_cast<uint16_t>(cell_id);
    item.msg = smsg;

    if (!queue_.try_push(item))
    {
        clear_config_inflight(item);
        const bool first_overflow = !first_overflow_logged_.exchange(true, std::memory_order_acq_rel);
        if (first_overflow)
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "{}: nonslot_lp queue overflow capacity={} msg_id=0x{:02X} cell_id={}",
                       __func__, kQueueCapacity, smsg.msg_id, smsg.cell_id);
        }
        else
        {
            NVLOGW_FMT(TAG,
                       "{}: nonslot_lp queue still full capacity={} msg_id=0x{:02X} cell_id={}",
                       __func__, kQueueCapacity, smsg.msg_id, smsg.cell_id);
        }
        return false;
    }

    smsg.reset();
    const auto queue_depth = queue_.size();
    NVLOGD_FMT(TAG,
               "{}: queued nonslot_lp msg_id=0x{:02X} cell_id={} depth={}",
               __func__, item.msg.msg_id, item.cell_id, queue_depth);
    if (queue_depth >= kSaturationThreshold &&
        !saturation_logged_.exchange(true, std::memory_order_acq_rel))
    {
        NVLOGW_FMT(TAG,
                   "{}: nonslot_lp queue saturation depth={} capacity={} threshold={}",
                   __func__, queue_depth, kQueueCapacity, kSaturationThreshold);
    }
    // Notify under lock to close the lost-wakeup window: the consumer cannot slip
    // a notify between its predicate check (queue empty → false) and its entry into
    // cv_.wait(), because both the predicate recheck and the sleep registration are
    // atomic with respect to the mutex. The performance cost is negligible for the
    // low-frequency CONFIG/START/STOP path.
    {
        std::lock_guard<std::mutex> lk(wakeup_mutex_);
        cv_.notify_one();
    }
    return true;
}

std::size_t NonSlotLpExecutor::depth() const noexcept
{
    return queue_.size();
}

const NonSlotLpThreadConfig& NonSlotLpExecutor::thread_config() const noexcept
{
    return config_;
}

bool NonSlotLpExecutor::is_running() const noexcept
{
    return thread_.joinable();
}

void NonSlotLpExecutor::thread_func(std::stop_token stop_token)
{
    configure_thread();
    nvlog_fmtlog_thread_init(NONSLOT_LP_THREAD_NAME);

    // Arm the memtrace tripwire: any heap allocation in the steady-state loop is
    // a latency bug.  No-op without LD_PRELOAD=<patched libmimalloc.so> + AERIAL_MEMTRACE=1.
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    while (true)
    {
        {
            std::unique_lock lock(wakeup_mutex_);
            cv_.wait(lock, stop_token, [this]() {
                return !queue_.empty();
            });
        }

        if (stop_token.stop_requested())
        {
            break;
        }

        WorkItem item{};
        while (!stop_token.stop_requested() && queue_.try_pop(item))
        {
            process_item(item);
            if (queue_.size() < kSaturationThreshold)
            {
                saturation_logged_.store(false, std::memory_order_release);
            }
        }
    }

    NVLOGI_FMT(TAG,
               "{}: exiting queued_depth={}",
               __func__, queue_.size());
}

void NonSlotLpExecutor::configure_thread()
{
    // std::make_error_code is thread-safe; std::strerror uses a static buffer.
    const auto strerr = [](int err) {
        return std::make_error_code(static_cast<std::errc>(err)).message();
    };

    int status = pthread_setname_np(pthread_self(), NONSLOT_LP_THREAD_NAME);
    if (status != 0)
    {
        NVLOGW_FMT(TAG, "{}: pthread_setname_np failed with status={}", __func__, strerr(status));
    }

    if (config_.worker_cfg.cpu_affinity >= 0)
    {
        const auto cpu_affinity = config_.worker_cfg.cpu_affinity;
        if (cpu_affinity >= CPU_SETSIZE)
        {
            NVLOGW_FMT(TAG,
                       "{}: cpu_affinity={} >= CPU_SETSIZE={}; skipping affinity setup",
                       __func__, cpu_affinity, CPU_SETSIZE);
        }
        else
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(static_cast<std::size_t>(cpu_affinity), &cpuset);
            status = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            if (status != 0)
            {
                NVLOGW_FMT(TAG,
                           "{}: setaffinity_np failed cpu={} status={}",
                           __func__, cpu_affinity, strerr(status));
            }
        }
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: no CPU affinity configured; nonslot_lp is running unpinned", __func__);
    }

    if (config_.worker_cfg.sched_priority > 0)
    {
        sched_param sch{};
        int policy{};
        status = pthread_getschedparam(pthread_self(), &policy, &sch);
        if (status != 0)
        {
            NVLOGW_FMT(TAG, "{}: pthread_getschedparam failed with status={}", __func__, strerr(status));
        }
        else
        {
            sch.sched_priority = config_.worker_cfg.sched_priority;
            status = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
            if (status != 0)
            {
                NVLOGW_FMT(TAG,
                           "{}: setschedparam SCHED_FIFO priority={} failed with status={}",
                           __func__, config_.worker_cfg.sched_priority, strerr(status));
            }
        }
    }

    sched_param actual_sch{};
    int actual_policy{};
    status = pthread_getschedparam(pthread_self(), &actual_policy, &actual_sch);
    if (status != 0)
    {
        NVLOGW_FMT(TAG, "{}: final pthread_getschedparam failed with status={}", __func__, std::strerror(status));
    }

    NVLOGI_FMT(TAG,
               "{}: nonslot_lp started cpu={} configured_cpu={} policy={} priority={} configured_priority={}",
               __func__,
               sched_getcpu(),
               config_.worker_cfg.cpu_affinity,
               actual_policy,
               actual_sch.sched_priority,
               config_.worker_cfg.sched_priority);
}

void NonSlotLpExecutor::process_item(WorkItem& item)
{
    if (runtime_.process_item == nullptr)
    {
        NVLOGW_FMT(TAG,
                   "{}: no process callback for msg_id=0x{:02X} cell_id={}; releasing",
                   __func__, item.msg.msg_id, item.cell_id);
        release_item(item);
        return;
    }

    // Expected CONFIG/START/STOP failures are handled by the handler.
    // This guard only prevents unexpected exceptions from killing nonslot_lp;
    // release_item() below still returns descriptor ownership to NVIPC.
    try
    {
        runtime_.process_item(runtime_.context, item.cell_id, item.msg);
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "{}: exception processing msg_id=0x{:02X} cell_id={}: {} "
                   "(no ERROR.indication sent — see V1 worker-exception contract above)",
                   __func__, item.msg.msg_id, item.cell_id, ex.what());
    }
    catch (...)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "{}: unknown exception processing msg_id=0x{:02X} cell_id={} "
                   "(no ERROR.indication sent — see V1 worker-exception contract above)",
                   __func__, item.msg.msg_id, item.cell_id);
    }

    clear_config_inflight(item);
    release_item(item);
}

bool NonSlotLpExecutor::mark_config_inflight(const phy_mac_msg_desc& smsg) noexcept
{
    if (smsg.msg_id != SCF_FAPI_CONFIG_REQUEST)
    {
        return true;
    }
    const auto cell_id = static_cast<std::size_t>(smsg.cell_id);
    if (cell_id >= config_inflight_.size())
    {
        return true;
    }

    bool expected = false;
    return config_inflight_[cell_id].compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

void NonSlotLpExecutor::clear_config_inflight(const WorkItem& item) noexcept
{
    if (item.msg.msg_id == SCF_FAPI_CONFIG_REQUEST && item.cell_id < config_inflight_.size())
    {
        config_inflight_[item.cell_id].store(false, std::memory_order_release);
    }
}

void NonSlotLpExecutor::release_duplicate_config(phy_mac_msg_desc& smsg) noexcept
{
    if (smsg.msg_buf != nullptr && runtime_.rx_release != nullptr)
    {
        runtime_.rx_release(runtime_.context, smsg);
    }
    smsg.reset();
}

void NonSlotLpExecutor::release_item(WorkItem& item)
{
    if (item.msg.msg_buf != nullptr && runtime_.rx_release != nullptr)
    {
        runtime_.rx_release(runtime_.context, item.msg);
    }
    item.msg.reset();
}

void NonSlotLpExecutor::drain()
{
    std::size_t released = 0;
    WorkItem item{};
    while (queue_.try_pop(item))
    {
        clear_config_inflight(item);
        release_item(item);
        ++released;
    }
    if (released > 0)
    {
        NVLOGW_FMT(TAG,
                   "{}: released {} queued nonslot_lp descriptors without processing",
                   __func__, released);
    }
}

#if !defined(NVPHY_NON_SLOT_DISPATCH_STANDALONE_TEST)
uint64_t NonSlotDispatchState::active_cell_bitmap() const noexcept
{
    return active_cell_mask_.committed();
}

uint32_t NonSlotDispatchState::active_cell_count() const noexcept
{
    return active_cell_mask_.committedCount();
}

uint64_t NonSlotDispatchState::activate_cell_immediate(const uint16_t cell_id) noexcept
{
    if (safe_cell_bit(cell_id) == 0ULL)
    {
        NVLOGW_FMT(TAG, "{}: cell_id={} out of range (>=64); ignoring", __func__, cell_id);
        return active_cell_bitmap();
    }
    return active_cell_mask_.activateImmediate(cell_id);
}

uint64_t NonSlotDispatchState::deactivate_cell_immediate(const uint16_t cell_id) noexcept
{
    if (safe_cell_bit(cell_id) == 0ULL)
    {
        NVLOGW_FMT(TAG, "{}: cell_id={} out of range (>=64); ignoring", __func__, cell_id);
        return active_cell_bitmap();
    }
    return active_cell_mask_.deactivateImmediate(cell_id);
}

void NonSlotDispatchState::stage_cell_started(const uint16_t cell_id) noexcept
{
    if (safe_cell_bit(cell_id) == 0ULL)
    {
        NVLOGW_FMT(TAG, "{}: cell_id={} out of range (>=64); ignoring", __func__, cell_id);
        return;
    }
    active_cell_mask_.stageStarted(cell_id);
}

void NonSlotDispatchState::stage_cell_stopped(const uint16_t cell_id) noexcept
{
    if (safe_cell_bit(cell_id) == 0ULL)
    {
        NVLOGW_FMT(TAG, "{}: cell_id={} out of range (>=64); ignoring", __func__, cell_id);
        return;
    }
    active_cell_mask_.stageStopped(cell_id);
}

// Slot-boundary commit: committed = staged. A staged START joins COMMITTED and a staged-out
// STOP leaves it, both at the SLOT.IND boundary (symmetric). The per-slot snapshot is taken
// from committed() right after, so a cell's membership is fixed for the whole slot.
uint64_t NonSlotDispatchState::commit_staged_active_cell_bitmap() noexcept
{
    return active_cell_mask_.commitStaged();
}

void NonSlotDispatchState::mark_produced(const uint16_t cell_id) noexcept
{
    // markProduced is range- and staged-gated internally; cell_id is already validated by the
    // store path, so forward directly (this runs per stored slot message -- keep it lean).
    active_cell_mask_.markProduced(cell_id);
}

void NonSlotDispatchState::install_worker(PHY_module& owner, NonSlotLpThreadConfig config)
{
    stop_worker();
    executor.emplace(owner, std::move(config));
}

void NonSlotDispatchState::start_worker()
{
    if (executor.has_value())
    {
        executor->start();
    }
}

void NonSlotDispatchState::stop_worker()
{
    if (executor.has_value())
    {
        executor->stop();
    }
}

void NonSlotDispatchState::move_worker_from(PHY_module& owner, NonSlotDispatchState& other)
{
    if (!other.executor.has_value())
    {
        return;
    }

    gsl_Expects(!other.executor->is_running());
    gsl_Expects(other.executor->depth() == 0);

    const auto config = other.executor->thread_config();
    other.executor.reset();
    executor.emplace(owner, config);
}

bool NonSlotDispatchState::enqueue_work(phy_mac_msg_desc& smsg)
{
    gsl_Expects(executor.has_value());
    return executor->enqueue(smsg);
}

uint64_t PHY_module::get_active_cell_bitmap() const noexcept
{
    return nonslot_dispatch_state_.active_cell_bitmap();
}

uint32_t PHY_module::get_active_cell_count() const noexcept
{
    return nonslot_dispatch_state_.active_cell_count();
}

void PHY_module::set_active_cell_bitmap(const uint16_t cell_id)
{
    // Mirror NonSlotDispatchState for legacy PHY_module readers.
    active_cell_bitmap = nonslot_dispatch_state_.activate_cell_immediate(cell_id);
}

void PHY_module::unset_active_cell_bitmap(const uint16_t cell_id)
{
    // Mirror NonSlotDispatchState for legacy PHY_module readers.
    active_cell_bitmap = nonslot_dispatch_state_.deactivate_cell_immediate(cell_id);
}

// Thread ownership of the active-cell rendezvous (ActiveCellMask: committed = staged & produced):
//   stage_cell_started / stage_cell_stopped      -> nonslot_lp worker (phy::on_cell_*_request_offload)
//   mark_cell_produced                           -> msg_processing, per stored slot message
//   commit_staged_active_cell_bitmap             -> msg_processing, once per SLOT.IND
// `staged` and `produced` are disjoint single-writer atomic fields: the worker only writes staged,
// msg_processing only writes produced, and commit reads both. That is why no lock or dispatch-mode
// gate is needed here. Do not add a cross-thread caller to any of these without revisiting the
// single-writer-per-field assumption.
void PHY_module::stage_cell_started(const uint16_t cell_id) noexcept
{
    nonslot_dispatch_state_.stage_cell_started(cell_id);
}

void PHY_module::stage_cell_stopped(const uint16_t cell_id) noexcept
{
    nonslot_dispatch_state_.stage_cell_stopped(cell_id);
}

void PHY_module::commit_staged_active_cell_bitmap() noexcept
{
    const auto committed = nonslot_dispatch_state_.commit_staged_active_cell_bitmap();
    // Mirror NonSlotDispatchState for legacy PHY_module readers.
    active_cell_bitmap = committed;
    num_cells_active = static_cast<uint>(std::popcount(committed));
}

void PHY_module::mark_cell_produced(const uint16_t cell_id) noexcept
{
    // Records production only; commit (and the legacy active_cell_bitmap mirror update) happens
    // at the next SLOT.IND via commit_staged_active_cell_bitmap(). No mirror refresh here.
    nonslot_dispatch_state_.mark_produced(cell_id);
}

void PHY_module::snapshot_active_cell_state(const uint32_t slot_u32) noexcept
{
    const auto ring_idx = ring_idx_from_slot(slot_u32);
    const auto committed = nonslot_dispatch_state_.active_cell_bitmap();
    task_pool_.active_cell_bitmap_snapshot[ring_idx] = committed;
    task_pool_.active_cell_count_snapshot[ring_idx] =
        static_cast<uint8_t>(std::popcount(committed));
}

void PHY_module::start_nonslot_lp_worker()
{
    nonslot_dispatch_state_.start_worker();
}

void PHY_module::stop_nonslot_lp_worker()
{
    nonslot_dispatch_state_.stop_worker();
}

void PHY_module::move_nonslot_lp_worker_from(PHY_module& other)
{
    nonslot_dispatch_state_.move_worker_from(*this, other.nonslot_dispatch_state_);
}

bool PHY_module::enqueue_nonslot_lp_work(phy_mac_msg_desc& smsg)
{
    return nonslot_dispatch_state_.enqueue_work(smsg);
}
#endif

} // namespace nv
