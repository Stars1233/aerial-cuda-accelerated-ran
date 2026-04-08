/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file perf_metrics_utils.hpp
 * @brief Utility functions for performance metrics library
 * 
 * Self-contained timing utilities to avoid heavy dependencies.
 */

#ifndef PERF_METRICS_UTILS_HPP
#define PERF_METRICS_UTILS_HPP

#include <chrono>

namespace perf_metrics {

using t_ns = std::chrono::nanoseconds;
using t_raw = std::uint64_t;  //!< Raw counter ticks (platform-specific)

/**
 * Get current time in nanoseconds since epoch
 * 
 * Equivalent to Time::nowNs() from cuphydriver but without the dependency.
 * Uses std::chrono::system_clock for high-precision timing.
 * 
 * @return Current time as nanoseconds since epoch
 */
t_ns nowNs();

/**
 * Get monotonic time in nanoseconds for performance measurements
 * 
 * Optimized for minimal overhead when only relative time differences are needed.
 * Uses platform-specific high-performance counters when available:
 * - ARM (aarch64): Virtual counter register (cntvct_el0)
 * - x86/x86_64: TSC via RDTSC instruction
 * - Other: std::chrono::steady_clock
 * 
 * @return Monotonic time in nanoseconds
 * 
 * @note This function is optimized for speed, not absolute time accuracy.
 *       Only use for measuring time intervals, not wall-clock time.
 */
t_ns monotonicNowNs();

/**
 * Get raw monotonic counter value (fastest - no conversion)
 * 
 * Returns platform-specific raw counter value without any conversion.
 * Use for accumulating timing data; convert to nanoseconds only when needed.
 * 
 * @return Raw counter ticks (platform-specific units)
 * 
 * @note Must use rawToNs() to convert accumulated ticks to nanoseconds
 */
t_raw monotonicNowRaw();

/**
 * Convert raw counter ticks to nanoseconds
 * 
 * @param[in] raw_ticks Raw counter value from monotonicNowRaw()
 * @return Time in nanoseconds
 */
std::uint64_t rawToNs(t_raw raw_ticks);

/**
 * Convert nanoseconds to raw counter ticks
 * 
 * @param[in] nanoseconds Duration in nanoseconds
 * @return Equivalent duration in raw counter ticks
 */
t_raw nsToRaw(std::uint64_t nanoseconds);

// ============================================================================
// Fast integer to string formatters (5x faster than snprintf)
// ============================================================================

// ============================================================================
// fmt-based integer to string formatters (bounds-checked)
// ============================================================================
// These functions use fmt::format_to with FMT_COMPILE for optimal performance
// with upfront bounds checking. Benchmarks show only ~7% overhead vs unchecked
// versions, making these the best choice for production code.
// ============================================================================

/**
 * @brief Compute number of decimal digits for a uint64_t value at compile time
 *
 * This constexpr function allows generic, type-safe buffer size calculation
 * at compile time based on the maximum value of a type.
 *
 * @param x Value to count digits for
 * @return Number of decimal digits (1-20)
 */
constexpr int digits10_u64(std::uint64_t x) noexcept {
    int d = 1;
    while (x >= 10) {
        x /= 10;
        ++d;
    }
    return d;
}

// Compile-time validation of digits10_u64
static_assert(digits10_u64(0) == 1, "0 has 1 digit");
static_assert(digits10_u64(1) == 1, "1 has 1 digit");
static_assert(digits10_u64(9) == 1, "9 has 1 digit");
static_assert(digits10_u64(10) == 2, "10 has 2 digits");
static_assert(digits10_u64(99) == 2, "99 has 2 digits");
static_assert(digits10_u64(100) == 3, "100 has 3 digits");
static_assert(digits10_u64(999) == 3, "999 has 3 digits");
static_assert(digits10_u64(std::numeric_limits<std::uint16_t>::max()) == 5, "uint16_t max = 5 digits");
static_assert(digits10_u64(std::numeric_limits<std::uint32_t>::max()) == 10, "uint32_t max = 10 digits");
static_assert(digits10_u64(std::numeric_limits<std::uint64_t>::max()) == 20, "uint64_t max = 20 digits");

/**
 * @brief fmt-based uint16_t to string formatter with bounds checking
 *
 * Uses constexpr digits10_u64() to calculate required buffer size (5 digits)
 * at compile time, making the code generic and self-documenting.
 *
 * @param[in] value Value to format (0-65535)
 * @param[out] buffer Output buffer start
 * @param[in] buffer_end Output buffer end (one past last valid byte)
 * @return Pointer to end of written data, or nullptr if insufficient space
 *
 * @note Uses constexpr to compute max digits at compile time (zero runtime cost)
 * @note Does NOT null-terminate the output string
 */
char* fmt_u16toa_safe(std::uint16_t value, char* buffer, char* buffer_end) noexcept;

/**
 * @brief fmt-based uint32_t to string formatter with bounds checking
 *
 * Uses constexpr digits10_u64() to calculate required buffer size (10 digits)
 * at compile time, making the code generic and self-documenting.
 *
 * @param[in] value Value to format (0-4294967295)
 * @param[out] buffer Output buffer start
 * @param[in] buffer_end Output buffer end (one past last valid byte)
 * @return Pointer to end of written data, or nullptr if insufficient space
 *
 * @note Uses constexpr to compute max digits at compile time (zero runtime cost)
 * @note Does NOT null-terminate the output string
 */
char* fmt_u32toa_safe(std::uint32_t value, char* buffer, char* buffer_end) noexcept;

/**
 * @brief fmt-based uint64_t to string formatter with bounds checking
 *
 * Uses constexpr digits10_u64() to calculate required buffer size (20 digits)
 * at compile time, making the code generic and self-documenting.
 *
 * @param[in] value Value to format
 * @param[out] buffer Output buffer start
 * @param[in] buffer_end Output buffer end (one past last valid byte)
 * @return Pointer to end of written data, or nullptr if insufficient space
 *
 * @note Uses constexpr to compute max digits at compile time (zero runtime cost)
 * @note Does NOT null-terminate the output string
 */
char* fmt_u64toa_safe(std::uint64_t value, char* buffer, char* buffer_end) noexcept;

} // namespace perf_metrics

#endif // PERF_METRICS_UTILS_HPP
