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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 17) // "DRV.FUNC"

#include "task_instrumentation_v3.hpp"
#include "perf_metrics/perf_metrics_utils.hpp"
#include "nvlog.hpp"
#include "nvlog_fmt.hpp"
#include <sched.h>

// Fast formatters are now in perf_metrics_utils
// (using those for timestamp conversion)

// ============================================================================
// Helper Functions
// ============================================================================

namespace {
/**
 * @brief Validate and capture synchronized time anchors
 * 
 * Captures nowNs() sandwiched between two monotonicNowRaw() calls.
 * The anchor_raw is set to the average (midpoint) for maximum accuracy.
 * 
 * @param[out] anchor_ns Captured absolute time in nanoseconds
 * @param[out] anchor_raw Captured raw cycle counter (midpoint)
 */
inline void captureSynchronizedAnchorsWithValidation(perf_metrics::t_ns& anchor_ns, 
                                                      perf_metrics::t_raw& anchor_raw) noexcept {
    // Capture raw before and after nowNs() to bound the syscall timing
    const perf_metrics::t_raw raw_before = perf_metrics::monotonicNowRaw();
    anchor_ns = perf_metrics::nowNs();
    const perf_metrics::t_raw raw_after = perf_metrics::monotonicNowRaw();
    
    // Calculate gap between raw readings
    const perf_metrics::t_raw gap_raw = raw_after - raw_before;
    
    // Set anchor_raw to the midpoint (average) for best accuracy
    anchor_raw = raw_before + (gap_raw / 2);
    
    // Validate synchronization - gap should be < 2 microseconds
    const uint64_t gap_ns = perf_metrics::rawToNs(gap_raw);
    const uint64_t threshold_ns = 2000;  // 2 microseconds
    if (gap_ns > threshold_ns) {
        NVLOGW_FMT(TAG, "TaskInstrumentation anchor synchronization gap large: {} ns (> {} ns threshold). "
                        "Timestamps may have reduced accuracy. Consider isolated core or reduce system load.",
                        gap_ns, threshold_ns);
    }
}
} // anonymous namespace

// ============================================================================
// Constructor Implementation
// ============================================================================

TaskInstrumentation::TaskInstrumentation(
    const TaskInstrumentationContext& ctx,
    const char* task_name,
    int max_subtasks
) noexcept
    : tracing_mode_(ctx.tracing_mode)
    , slot_id_(ctx.slot_id)
    , sfn_(ctx.sfn)
    , slot_(ctx.slot)
    , pmu_(ctx.pmu)
    , max_subtasks_(max_subtasks < MAX_NUM_SUBTASKS ? max_subtasks : MAX_NUM_SUBTASKS)
    , cpu_(0)
    , current_offset_(0)
    , subtask_count_(0)
    , logged_(false)
{
    // Copy task name safely (strncpy + ensure null termination)
    if (task_name != nullptr) {
        std::strncpy(task_name_, task_name, MAX_TASK_NAME_CHARS - 1);
        task_name_[MAX_TASK_NAME_CHARS - 1] = '\0';
    } else {
        task_name_[0] = '\0';
    }

    // Capture synchronized time anchors with validation
    captureSynchronizedAnchorsWithValidation(anchor_ns_, anchor_raw_);

    // Mode START_END_ONLY: Log start time immediately (separate line from end time)
    // This ensures we have the start timestamp even if the task crashes
    if (tracing_mode_ == TracingMode::START_END_ONLY) {
        char* p = subtask_results_;
        char* end = subtask_results_ + MAX_NVSLOGI_CHARS - 1;
        *p++ = 's';
        *p++ = ':';
        char* new_p = perf_metrics::fmt_u64toa_safe(perf_metrics::nowNs().count(), p, end);
        if (new_p != nullptr) {
            p = new_p;
        }
        *p = '\0';
        NVLOGI_FMT(TAG, "{{mTI}} <{},{},{},{},{}> {}",
            task_name_,
            sfn_,
            slot_,
            slot_id_,
            0,  // cpu_ not known yet
            subtask_results_);
    }

    if (tracing_mode_ != TracingMode::DISABLED && pmu_ != nullptr) {
        pmu_->recordStart();
    }
}

// ============================================================================
// Destructor Implementation
// ============================================================================

TaskInstrumentation::~TaskInstrumentation() noexcept {
    if (!logged_ && tracing_mode_ != TracingMode::DISABLED) {
        performLogging();
    }
}

// ============================================================================
// Logging Implementation
// ============================================================================

void TaskInstrumentation::performLogging() noexcept {
    logged_ = true;

    // Get CPU affinity - sched_getcpu() is ~10x faster than getcpu()
    int cpu = sched_getcpu();
    cpu_ = (cpu >= 0) ? static_cast<uint32_t>(cpu) : 0xFFFFFFFF;

    if (pmu_ != nullptr) {
        pmu_->recordStop();
        pmu_->formatCounterMetrics(pmu_metrics_str_, MAX_PMU_METRICS_CHARS);
    } else {
        pmu_metrics_str_[0] = '\0';
    }

    if (tracing_mode_ == TracingMode::FULL_TRACING) {
        // Mode FULL_TRACING: Full tracing with all subtasks
        // subtask_results_ is already formatted by add()/append()/appendList()
        // Just ensure null termination
        if (current_offset_ < MAX_NVSLOGI_CHARS) {
            subtask_results_[current_offset_] = '\0';
        } else {
            subtask_results_[MAX_NVSLOGI_CHARS - 1] = '\0';
        }

        // Let fmtlog handle the final formatting
        NVLOGI_FMT(TAG, "{{TI}} <{},{},{},{},{}> <{}> {}",
            task_name_,
            sfn_,
            slot_,
            slot_id_,
            cpu_,
            pmu_metrics_str_,
            subtask_results_);
    }
    else if (tracing_mode_ == TracingMode::START_END_ONLY) {
        // Mode START_END_ONLY: Log end time (start time was logged separately in constructor)
        char* p = subtask_results_;
        char* end = subtask_results_ + MAX_NVSLOGI_CHARS - 1;
        *p++ = 'e';
        *p++ = ':';
        char* new_p = perf_metrics::fmt_u64toa_safe(perf_metrics::nowNs().count(), p, end);
        if (new_p != nullptr) {
            p = new_p;
        }
        *p = '\0';

        NVLOGI_FMT(TAG, "{{mTI}} <{},{},{},{},{}> <{}> {}",
            task_name_,
            sfn_,
            slot_,
            slot_id_,
            cpu_,
            pmu_metrics_str_,
            subtask_results_);
    }
}

// ============================================================================
// Public API Implementations
// ============================================================================

void TaskInstrumentation::add(const char* subtask_name) noexcept {
    if (tracing_mode_ != TracingMode::FULL_TRACING) {
        return;
    }
    
    // Get current time with memory barrier to ensure accurate ordering
    // This is critical - without barriers, timestamps could be reordered relative to work
    const auto current_raw = perf_metrics::monotonicNowRaw();
    const auto elapsed_raw = current_raw - anchor_raw_;
    const auto elapsed_ns = perf_metrics::rawToNs(elapsed_raw);
    const uint64_t abs_time_ns = static_cast<uint64_t>(anchor_ns_.count()) + elapsed_ns;
    
    formatSubtask(subtask_name, abs_time_ns);
}

void TaskInstrumentation::append(const char* subtask_name, t_ns time) noexcept {
    if (tracing_mode_ == TracingMode::FULL_TRACING) {
        formatSubtask(subtask_name, static_cast<uint64_t>(time.count()));
    }
}

void TaskInstrumentation::appendList(const ti_subtask_info& info) noexcept {
    if (tracing_mode_ != TracingMode::FULL_TRACING) {
        return;
    }
    
    for (int ii = 0; ii < info.count && subtask_count_ < max_subtasks_; ++ii) {
        if (!formatSubtask(info.tname[ii], static_cast<uint64_t>(info.time[ii].count()))) {
            break;  // Out of space
        }
    }
}

bool TaskInstrumentation::formatSubtask(const char* subtask_name, uint64_t timestamp_ns) noexcept {
    if (subtask_count_ >= max_subtasks_) {
        return false;
    }
    
    char* start = &subtask_results_[current_offset_];
    char* p = start;
    char* end = subtask_results_ + MAX_NVSLOGI_CHARS - 1;
    
    // Copy name, reserving space for ':' + timestamp (max 20) + ','
    while (*subtask_name && p < end - 22) {
        *p++ = *subtask_name++;
    }
    
    // If name didn't fully fit, abort
    if (*subtask_name != '\0') {
        *start = '\0';
        return false;
    }
    
    *p++ = ':';

    // Format timestamp
    char* new_p = perf_metrics::fmt_u64toa_safe(timestamp_ns, p, end);
    if (new_p == nullptr) {
        *start = '\0';
        return false;
    }
    p = new_p;
    *p++ = ',';
    
    current_offset_ = p - subtask_results_;
    ++subtask_count_;
    return true;
}
