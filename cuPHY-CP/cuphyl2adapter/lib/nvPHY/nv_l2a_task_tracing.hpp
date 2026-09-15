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

#ifndef NV_L2A_TASK_TRACING_HPP
#define NV_L2A_TASK_TRACING_HPP

#include "nv_ipc_utils.h"        // sfn_slot_t
#include "task_instrumentation/task_instrumentation_v3.hpp"   // TracingMode, TaskInstrumentationContext

#include <concepts>
#include <cstdint>

class Worker;
class PMUDeltaSummarizer;
[[nodiscard]] PMUDeltaSummarizer* l1_get_worker_pmu(Worker* worker);

namespace nv
{

namespace detail
{

/**
 * @brief Compile-time interface every L2A aggregation task arg must satisfy
 *        to be usable with @ref makeL2AInstrumentationContext.
 *
 * Currently @c DlAggrTaskArg, @c UlAggrTaskArg and @c CplaneBatchTaskArg all
 * satisfy this duck-typed contract without any explicit tagging. The factory
 * pulls only @c slot_u32 from the arg: SFN/slot are decoded via the
 * @c sfn_slot_t union, and @c slot_u32 itself is reused as the
 * @c TaskInstrumentationContext slot-id (it is unique within the SFN cycle,
 * unlike @c ring_idx which wraps modulo @c SLOT_STORAGE_DEPTH).
 */
template<typename T>
concept L2ATaskArg = requires(const T& ctx) {
    { ctx.slot_u32 } -> std::convertible_to<uint32_t>;
};

/**
 * @brief Look up the active CPU task tracing mode from the driver context.
 *
 * Routed through @c PHYDriverProxy following the cuphyl2adapter convention
 * (see @c is_fapi_to_cplane_direct_enabled in nv_phy_slot_dispatch.cpp).
 * Defined in nv_phy_driver_proxy.cpp so this header stays free of
 * cuphydriver_api.hpp / nv_phy_driver_proxy.hpp transitive includes.
 * Returns @c TracingMode::DISABLED if the proxy is uninitialized.
 */
[[nodiscard]] TracingMode l2a_cpu_tracing_mode() noexcept;

} // namespace detail

/**
 * @brief Build a @c TaskInstrumentationContext for any L2A aggregation task arg.
 *
 * Mirrors the cuphydriver factories in
 * @c task_instrumentation_v3_factories.hpp but operates on the L2A task arg
 * shapes (DlAggrTaskArg / UlAggrTaskArg / CplaneBatchTaskArg) rather than
 * SlotMap pointers. Pulls only @c ctx.slot_u32: SFN/slot are decoded via the
 * @c sfn_slot_t union, and @c slot_u32 itself is reused as the slot-id
 * (it is unique within the SFN cycle, whereas @c ring_idx would wrap modulo
 * @c SLOT_STORAGE_DEPTH and collide across nearby slots). Tracing mode comes
 * from @c l1_get_cpu_task_tracing_mode (configurable via the cuphycontroller
 * YAML @c enable_cpu_task_tracing field) and PMU counters from
 * @c l1_get_worker_pmu (cuphydriver public API, avoids including
 * @c worker.hpp and its heavy transitive dependencies).
 *
 * @tparam TaskArg Any task arg type satisfying @ref detail::L2ATaskArg.
 *                 Currently DlAggrTaskArg, UlAggrTaskArg, CplaneBatchTaskArg.
 * @param[in] ctx     Task arg for the in-progress slot.
 * @param[in] worker  Optional worker, used to attach PMU counter snapshots
 *                    (pass the @c Worker* received by the task work function).
 * @return A fully-populated @c TaskInstrumentationContext ready to drive a
 *         @c TaskInstrumentation scope on the worker thread.
 */
template<detail::L2ATaskArg TaskArg>
[[nodiscard]] inline TaskInstrumentationContext
makeL2AInstrumentationContext(const TaskArg& ctx, Worker* worker = nullptr) noexcept
{
    const sfn_slot_t ss{.u32 = ctx.slot_u32};
    return TaskInstrumentationContext(
        detail::l2a_cpu_tracing_mode(),
        static_cast<uint64_t>(ctx.slot_u32),
        ss.u16.sfn, ss.u16.slot,
        l1_get_worker_pmu(worker));
}

} // namespace nv

#endif // NV_L2A_TASK_TRACING_HPP
