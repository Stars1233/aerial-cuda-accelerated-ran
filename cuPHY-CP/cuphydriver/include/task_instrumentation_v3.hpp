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

#ifndef TASK_INSTRUMENTATION_V3_H
#define TASK_INSTRUMENTATION_V3_H

#include "perf_metrics/perf_metrics_utils.hpp"  // For timing and fast formatters
#include "task_instrumentation_nested.hpp"  // For ti_subtask_info
#include "pmu_reader.hpp"  // For PMUDeltaSummarizer (lightweight: linux perf + nvlog only)
#include <cstdint>
#include <cstring>

// Type alias for nanoseconds (matches existing t_ns from time.hpp)
using t_ns = perf_metrics::t_ns;

/**
 * @brief Task instrumentation tracing modes
 *
 * Defines the level of detail captured during task execution:
 * - DISABLED: No instrumentation or logging
 * - FULL_TRACING: Log all subtasks with individual durations
 * - START_END_ONLY: Log task start immediately, then end separately (for crash safety)
 */
enum class TracingMode : uint8_t {
    DISABLED       = 0,  ///< No tracing or logging
    FULL_TRACING   = 1,  ///< Full subtask tracing with durations
    START_END_ONLY = 2   ///< Minimal mode: log start/end times only
};

/**
 * @brief Slot context for task instrumentation
 *
 * Lightweight structure containing only the data needed for instrumentation.
 * No dependencies on SlotMap, PhyDriverCtx, or other heavy types.
 */
struct TaskInstrumentationContext {
    TracingMode tracing_mode;       ///< Tracing mode (see TracingMode enum)
    uint64_t slot_id;               ///< Slot map ID for logging
    uint16_t sfn;                   ///< System Frame Number (0-1023)
    uint16_t slot;                  ///< Slot number within frame
    PMUDeltaSummarizer* pmu;        ///< PMU counter provider (nullable)

    /// Default constructor
    TaskInstrumentationContext() noexcept
        : tracing_mode(TracingMode::DISABLED), slot_id(0), sfn(0), slot(0), pmu(nullptr) {}

    /// Constructor with parameters
    TaskInstrumentationContext(TracingMode mode, uint64_t id, uint16_t sfn_val, uint16_t slot_val,
                               PMUDeltaSummarizer* pmu_ptr = nullptr) noexcept
        : tracing_mode(mode), slot_id(id), sfn(sfn_val), slot(slot_val), pmu(pmu_ptr) {}
};

/**
 * @brief Dependency-free task instrumentation
 *
 * New implementation that takes parameters directly instead of extracting
 * from templated SlotMap types. This eliminates massive dependency chains.
 *
 * Key improvements over v2:
 * - No template parameters (no SlotMap dependencies)
 * - No PhyDriverCtx dependency
 * - PMU via context (no Worker dependency)
 * - Same logging output format (preserves post-processing compatibility)
 * - Same stack usage (~2.4KB)
 *
 * Example usage:
 *     TaskInstrumentationContext ctx(TracingMode::FULL_TRACING, 123, 45, 6);
 *     TaskInstrumentation ti(ctx, "My Task", 10);
 *     ti.add("Subtask 1");
 *     ti.add("Subtask 2");
 *     // Automatic logging on scope exit
 */
class TaskInstrumentation {
public:
    /**
     * @brief Constructor - initializes task instrumentation
     *
     * @param ctx Slot context (tracing mode, slot_id, sfn, slot, pmu)
     * @param task_name Name of the task (copied, max 63 chars)
     * @param max_subtasks Maximum number of subtasks to track
     */
    TaskInstrumentation(
        const TaskInstrumentationContext& ctx,
        const char* task_name,
        int max_subtasks
    ) noexcept;

    /**
     * @brief Destructor - automatically logs instrumentation data
     */
    ~TaskInstrumentation() noexcept;

    /**
     * @brief Add subtask checkpoint with current timestamp
     *
     * Formats name and timestamp immediately into output buffer.
     * Uses cycle counter with memory barriers for accurate ordering.
     * No dangling pointers - copies string and formats timestamp in one shot.
     *
     * @param subtask_name Name of the subtask (can be temporary string)
     */
    void add(const char* subtask_name) noexcept;

    /**
     * @brief Append subtask with explicit timestamp
     *
     * Used for nested instrumentation that provides pre-computed absolute times.
     * Formats immediately into output buffer.
     *
     * @param subtask_name Name of the subtask (can be temporary string)
     * @param time Explicit timestamp for the subtask
     */
    void append(const char* subtask_name, t_ns time) noexcept;

    /**
     * @brief Append nested subtask instrumentation data
     *
     * Used for appending instrumentation data from nested function calls.
     * Formats immediately into output buffer.
     * @param info Nested subtask info structure
     */
    void appendList(const ti_subtask_info& info) noexcept;

    // Prevent copying and moving
    TaskInstrumentation(const TaskInstrumentation&) = delete;
    TaskInstrumentation& operator=(const TaskInstrumentation&) = delete;
    TaskInstrumentation(TaskInstrumentation&&) = delete;
    TaskInstrumentation& operator=(TaskInstrumentation&&) = delete;

private:
    // Configuration from context
    TracingMode tracing_mode_;
    uint64_t slot_id_;
    uint16_t sfn_;
    uint16_t slot_;

    // PMU counter provider (nullable)
    PMUDeltaSummarizer* pmu_;

    // Task state
    int max_subtasks_;
    uint32_t cpu_;

    // Stack-based storage arrays
    static constexpr int MAX_NUM_SUBTASKS = 32;
    static constexpr int MAX_TASK_NAME_CHARS = 64;
    static constexpr int MAX_PMU_METRICS_CHARS = 100;
    static constexpr int MAX_NVSLOGI_CHARS = 1024;

    char task_name_[MAX_TASK_NAME_CHARS];

    // Time anchors established at construction for immediate timestamp formatting
    t_ns anchor_ns_;
    perf_metrics::t_raw anchor_raw_;

    // Pre-formatted subtask results buffer and current write offset
    char pmu_metrics_str_[MAX_PMU_METRICS_CHARS];
    char subtask_results_[MAX_NVSLOGI_CHARS];
    uint32_t current_offset_;

    int subtask_count_;
    bool logged_;

    void performLogging() noexcept;
    
    /**
     * @brief Helper to format a subtask entry into subtask_results_
     * 
     * Optimized for the common case (enough space). If buffer runs out,
     * aborts cleanly by null-terminating at entry start.
     * 
     * @param subtask_name Name to copy
     * @param timestamp_ns Absolute timestamp in nanoseconds
     * @return true if successfully formatted, false if out of space
     */
    bool formatSubtask(const char* subtask_name, uint64_t timestamp_ns) noexcept;
};

#endif // TASK_INSTRUMENTATION_V3_H
