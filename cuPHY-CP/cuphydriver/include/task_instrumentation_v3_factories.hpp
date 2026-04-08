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

#ifndef TASK_INSTRUMENTATION_V3_FACTORIES_H
#define TASK_INSTRUMENTATION_V3_FACTORIES_H

#include "task_instrumentation_v3.hpp"
#include "slot_map_dl.hpp"
#include "slot_map_ul.hpp"
#include "context.hpp"
#include "constant.hpp"
#include "worker.hpp"

/**
 * @file task_instrumentation_v3_factories.hpp
 * @brief Convenience factory functions for creating TaskInstrumentationContext
 *
 * These factories extract the necessary data from SlotMap objects, making
 * migration from v2 simpler. Note that this header DOES have dependencies
 * (SlotMap, PhyDriverCtx), but the core v3 API remains dependency-free.
 *
 * Use this header in production code for convenience.
 * Use the core v3 header directly in unit tests.
 */

/**
 * @brief Create TaskInstrumentationContext from SlotMapDl pointer
 *
 * Extracts tracing_mode, slot_id, sfn, and slot from the SlotMapDl.
 * This is the most convenient way to create a context in production code.
 *
 * @param slot_map Pointer to SlotMapDl (if null, returns disabled context)
 * @return TaskInstrumentationContext ready to use
 *
 * Example:
 *     SlotMapDl* slot_map = (SlotMapDl*)param;
 *     auto ctx = makeInstrumentationContext(slot_map, worker);
 *     TaskInstrumentation ti(ctx, "My Task", 10);
 */
inline TaskInstrumentationContext makeInstrumentationContext(SlotMapDl* slot_map,
                                                             Worker* worker = nullptr) noexcept {
    PMUDeltaSummarizer* pmu = (worker != nullptr) ? worker->getPMU() : nullptr;

    if (slot_map == nullptr) {
        return TaskInstrumentationContext(TracingMode::DISABLED, 0, 0, 0, pmu);
    }

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    if (pdctx == nullptr) {
        return TaskInstrumentationContext(TracingMode::DISABLED, slot_map->getId(),
            slot_map->getSlot3GPP().sfn_, slot_map->getSlot3GPP().slot_, pmu);
    }

    return TaskInstrumentationContext(
        static_cast<TracingMode>(pdctx->enableCPUTaskTracing()),
        slot_map->getId(),
        slot_map->getSlot3GPP().sfn_,
        slot_map->getSlot3GPP().slot_,
        pmu
    );
}

/**
 * @brief Create TaskInstrumentationContext from SlotMapUl pointer
 *
 * Extracts tracing_mode, slot_id, sfn, and slot from the SlotMapUl.
 *
 * @param slot_map Pointer to SlotMapUl (if null, returns disabled context)
 * @return TaskInstrumentationContext ready to use
 */
inline TaskInstrumentationContext makeInstrumentationContext(SlotMapUl* slot_map,
                                                             Worker* worker = nullptr) noexcept {
    PMUDeltaSummarizer* pmu = (worker != nullptr) ? worker->getPMU() : nullptr;

    if (slot_map == nullptr) {
        return TaskInstrumentationContext(TracingMode::DISABLED, 0, 0, 0, pmu);
    }

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    if (pdctx == nullptr) {
        return TaskInstrumentationContext(TracingMode::DISABLED, slot_map->getId(),
            slot_map->getSlot3GPP().sfn_, slot_map->getSlot3GPP().slot_, pmu);
    }

    return TaskInstrumentationContext(
        static_cast<TracingMode>(pdctx->enableCPUTaskTracing()),
        slot_map->getId(),
        slot_map->getSlot3GPP().sfn_,
        slot_map->getSlot3GPP().slot_,
        pmu
    );
}

/**
 * @brief Create TaskInstrumentationContext from void* param (DL variant)
 *
 * This is for compatibility with existing function signatures that take void*.
 * Casts to SlotMapDl* and extracts data.
 *
 * @param param void* pointer to SlotMapDl (if null, returns disabled context)
 * @return TaskInstrumentationContext ready to use
 *
 * Example:
 *     int task_work_function_debug(Worker* worker, void* param, ...) {
 *         auto ctx = makeInstrumentationContextDL(param, worker);
 *         TaskInstrumentation ti(ctx, "Debug Task", 13);
 *     }
 */
inline TaskInstrumentationContext makeInstrumentationContextDL(void* param,
                                                               Worker* worker = nullptr) noexcept {
    if (param == nullptr) {
        PMUDeltaSummarizer* pmu = (worker != nullptr) ? worker->getPMU() : nullptr;
        return TaskInstrumentationContext(TracingMode::DISABLED, 0, 0, 0, pmu);
    }
    return makeInstrumentationContext(static_cast<SlotMapDl*>(param), worker);
}

/**
 * @brief Create TaskInstrumentationContext from void* param (UL variant)
 *
 * This is for compatibility with existing function signatures that take void*.
 * Casts to SlotMapUl* and extracts data.
 *
 * @param param void* pointer to SlotMapUl (if null, returns disabled context)
 * @return TaskInstrumentationContext ready to use
 */
inline TaskInstrumentationContext makeInstrumentationContextUL(void* param,
                                                               Worker* worker = nullptr) noexcept {
    if (param == nullptr) {
        PMUDeltaSummarizer* pmu = (worker != nullptr) ? worker->getPMU() : nullptr;
        return TaskInstrumentationContext(TracingMode::DISABLED, 0, 0, 0, pmu);
    }
    return makeInstrumentationContext(static_cast<SlotMapUl*>(param), worker);
}

#endif // TASK_INSTRUMENTATION_V3_FACTORIES_H
