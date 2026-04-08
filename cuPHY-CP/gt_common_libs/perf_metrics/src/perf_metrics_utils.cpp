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
 * @file perf_metrics_utils.cpp
 * @brief Implementation of utility functions for performance metrics library
 */

#include "perf_metrics/perf_metrics_utils.hpp"
#include <cstdint>
#include <fmt/format.h>
#include <fmt/compile.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <x86intrin.h>
#include <immintrin.h>
#endif

namespace perf_metrics {

t_ns nowNs() {
    return std::chrono::system_clock::now().time_since_epoch();
}

#if defined(__aarch64__) || defined(__arm64__)
// ARM 64-bit implementation using virtual counter
namespace {
    // Get CPU frequency once at startup
    inline std::uint64_t getArmTimerFrequency() {
        std::uint64_t freq{};
        asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
        return freq;
    }
    
    const std::uint64_t ARM_TIMER_FREQ = getArmTimerFrequency();
    // Optimization: many ARM systems run at exactly 1 GHz, where cycles == nanoseconds
    const bool ARM_TIMER_IS_1GHZ = (ARM_TIMER_FREQ == 1000000000ULL);
}

t_raw monotonicNowRaw() {
    std::uint64_t cycles{};
    // ISB ensures all prior instructions complete before reading timer
    // Memory clobber prevents compiler reordering
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(cycles) :: "memory");
    return cycles;  // Return raw cycles - no conversion!
}

std::uint64_t rawToNs(const t_raw raw_ticks) {
    // Fast path: if frequency is exactly 1 GHz, cycles == nanoseconds
    // This avoids expensive 128-bit division (~10x faster)
    if (ARM_TIMER_IS_1GHZ) [[likely]] {
        return raw_ticks;
    }
    // Slow path: Convert cycles to nanoseconds using 128-bit math
    const __uint128_t ns = (static_cast<__uint128_t>(raw_ticks) * 1000000000ULL) / ARM_TIMER_FREQ;
    return static_cast<std::uint64_t>(ns);
}

t_raw nsToRaw(const std::uint64_t nanoseconds) {
    // Fast path: if frequency is exactly 1 GHz, nanoseconds == cycles
    if (ARM_TIMER_IS_1GHZ) [[likely]] {
        return nanoseconds;
    }
    // Slow path: Convert nanoseconds to cycles using 128-bit math
    const __uint128_t cycles = (static_cast<__uint128_t>(nanoseconds) * ARM_TIMER_FREQ) / 1000000000ULL;
    return static_cast<t_raw>(cycles);
}

t_ns monotonicNowNs() {
    return t_ns(rawToNs(monotonicNowRaw()));
}

#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
// x86/x86_64 implementation using RDTSC
namespace {
    // Estimate TSC frequency by comparing against steady_clock
    inline std::uint64_t estimateTscFrequency() {
        const auto start_tsc = __rdtsc();
        const auto start_time = std::chrono::steady_clock::now();
        
        // Busy wait for ~10ms
        while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(10));
        
        const auto end_tsc = __rdtsc();
        const auto end_time = std::chrono::steady_clock::now();
        
        const std::uint64_t tsc_delta = end_tsc - start_tsc;
        const auto time_delta = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        
        // Calculate frequency in Hz
        return (tsc_delta * 1000000000ULL) / time_delta;
    }
    
    const std::uint64_t TSC_FREQ = estimateTscFrequency();
}

t_raw monotonicNowRaw() {
    // Use LFENCE to serialize execution and prevent out-of-order execution
    // This ensures the timer read reflects actual execution point
    _mm_lfence();
    const std::uint64_t tsc = __rdtsc();
    _mm_lfence();
    return tsc;  // Return raw TSC - no conversion!
}

std::uint64_t rawToNs(const t_raw raw_ticks) {
    // Convert TSC to nanoseconds
    const __uint128_t ns = (static_cast<__uint128_t>(raw_ticks) * 1000000000ULL) / TSC_FREQ;
    return static_cast<std::uint64_t>(ns);
}

t_raw nsToRaw(const std::uint64_t nanoseconds) {
    // Convert nanoseconds to TSC ticks: (ns * frequency) / 1,000,000,000
    const __uint128_t ticks = (static_cast<__uint128_t>(nanoseconds) * TSC_FREQ) / 1000000000ULL;
    return static_cast<t_raw>(ticks);
}

t_ns monotonicNowNs() {
    return t_ns(rawToNs(monotonicNowRaw()));
}

#else
// Fallback implementation using steady_clock
t_raw monotonicNowRaw() {
    // Compiler barrier to prevent reordering
    asm volatile("" ::: "memory");
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    asm volatile("" ::: "memory");
    return ticks;
}

std::uint64_t rawToNs(const t_raw raw_ticks) {
    // Assume steady_clock is already in nanoseconds
    return raw_ticks;
}

t_raw nsToRaw(const std::uint64_t nanoseconds) {
    // Assume steady_clock is already in nanoseconds (1:1 mapping)
    return nanoseconds;
}

t_ns monotonicNowNs() {
    return std::chrono::steady_clock::now().time_since_epoch();
}

#endif

// ============================================================================
// fmt-based integer to string formatters (bounds-checked)
// ============================================================================
// Uses fmt::format_to with FMT_COMPILE for optimal performance (~29ns median)
// with upfront bounds checking. Benchmarks show only ~7% overhead vs unchecked
// versions, making these the best choice for production code.
//
// Performance (50M iterations, isolated CPU core):
//   - fmt_u64toa_safe: 29ns median, 51ns p99.9 (with bounds check)
//   - Unchecked alternative: 27ns median, 46ns p99.9 (unsafe)
//   - Previous custom impl: 32ns median, 67ns p99.9 (with bounds check)
// ============================================================================

char* fmt_u16toa_safe(std::uint16_t value, char* buffer, char* buffer_end) noexcept {
    // Compile-time computation of maximum digits for uint16_t
    constexpr int max_digits = digits10_u64(std::numeric_limits<std::uint16_t>::max());

    // Check if we have enough space (computed at compile time)
    if (buffer + max_digits > buffer_end) {
        return nullptr;
    }
    // Use fmt::format_to with FMT_COMPILE for optimal performance
    return fmt::format_to(buffer, FMT_COMPILE("{}"), value);
}

char* fmt_u32toa_safe(std::uint32_t value, char* buffer, char* buffer_end) noexcept {
    // Compile-time computation of maximum digits for uint32_t
    constexpr int max_digits = digits10_u64(std::numeric_limits<std::uint32_t>::max());

    // Check if we have enough space (computed at compile time)
    if (buffer + max_digits > buffer_end) {
        return nullptr;
    }
    // Use fmt::format_to with FMT_COMPILE for optimal performance
    return fmt::format_to(buffer, FMT_COMPILE("{}"), value);
}

char* fmt_u64toa_safe(std::uint64_t value, char* buffer, char* buffer_end) noexcept {
    // Compile-time computation of maximum digits for uint64_t
    constexpr int max_digits = digits10_u64(std::numeric_limits<std::uint64_t>::max());

    // Check if we have enough space (computed at compile time)
    if (buffer + max_digits > buffer_end) {
        return nullptr;
    }
    // Use fmt::format_to with FMT_COMPILE for optimal performance
    return fmt::format_to(buffer, FMT_COMPILE("{}"), value);
}

} // namespace perf_metrics
