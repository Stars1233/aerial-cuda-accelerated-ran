/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file perf_metrics_accumulator_benchmark.cpp
 * @brief Benchmark test to measure overhead of PerfMetricsAccumulator operations
 */

#include "perf_metrics/perf_metrics_accumulator.hpp"
#include "perf_metrics/perf_metrics_utils.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>

// Use DRV.PERF_METRICS tag for testing (NVLOG_TAG_BASE_CUPHY_DRIVER + 48)
#define TAG_PERF_METRICS 248

/**
 * Calculate statistics from a vector of durations
 */
struct Stats
{
    uint64_t min{};
    uint64_t max{};
    uint64_t avg{};
    uint64_t median{};
    uint64_t p95{};
    uint64_t p99{};
};

Stats calculateStats(std::vector<uint64_t>& durations)
{
    if (durations.empty()) {
        return {};
    }

    std::sort(durations.begin(), durations.end());
    
    Stats stats;
    stats.min = durations.front();
    stats.max = durations.back();
    
    uint64_t sum{};
    for (const auto duration : durations) {
        sum += duration;
    }
    stats.avg = sum / durations.size();
    
    const std::size_t mid = durations.size() / 2;
    stats.median = durations.size() % 2 == 0 
        ? (durations[mid - 1] + durations[mid]) / 2 
        : durations[mid];
    
    const std::size_t p95_idx = static_cast<std::size_t>(durations.size() * 0.95);
    stats.p95 = durations[std::min(p95_idx, durations.size() - 1)];
    
    const std::size_t p99_idx = static_cast<std::size_t>(durations.size() * 0.99);
    stats.p99 = durations[std::min(p99_idx, durations.size() - 1)];
    
    return stats;
}

void printStats(const char* name, const Stats& stats, int iterations = 0)
{
    if (iterations > 0) {
        std::cout << name << " (n=" << iterations << "): "
                  << "min=" << stats.min << "ns, "
                  << "max=" << stats.max << "ns, "
                  << "avg=" << stats.avg << "ns, "
                  << "med=" << stats.median << "ns, "
                  << "p95=" << stats.p95 << "ns, "
                  << "p99=" << stats.p99 << "ns"
                  << std::endl;
    } else {
        std::cout << name << ": "
                  << "min=" << stats.min << "ns, "
                  << "max=" << stats.max << "ns, "
                  << "avg=" << stats.avg << "ns, "
                  << "med=" << stats.median << "ns, "
                  << "p95=" << stats.p95 << "ns, "
                  << "p99=" << stats.p99 << "ns"
                  << std::endl;
    }
}

void benchmark_start_stop_overhead()
{
    std::cout << "\n=== Benchmarking startSection/stopSection Overhead ===" << std::endl;
    
    // Create accumulator with same sections as production (17 sections)
    perf_metrics::PerfMetricsAccumulator pma{
        "all_tx",
        "handle_sect1",
        "handle_sect1_outer",
        "handle_sect1_inner",
        "handle_sect3",
        "sect1_initialize",
        "section_processing",
        "check_if_drop",
        "packet_preparation",
        "packet_send",
        "lock_wait",
        "tx_burst_loop",
        "enqueue_log",
        "packet_stats_update",
        "counter_updates",
        "tput_counters",
        "verify"
    };
    
    constexpr int NUM_ITERATIONS = 10000;
    constexpr int WARMUP_ITERATIONS = 100;
    
    std::vector<uint64_t> start_durations;
    std::vector<uint64_t> stop_durations;
    std::vector<uint64_t> pair_durations;
    
    start_durations.reserve(NUM_ITERATIONS);
    stop_durations.reserve(NUM_ITERATIONS);
    pair_durations.reserve(NUM_ITERATIONS);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        pma.startSection("all_tx");
        pma.stopSection("all_tx");
    }
    pma.reset();
    
    // Benchmark
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Measure startSection
        const auto start_begin = perf_metrics::nowNs();
        pma.startSection("all_tx");
        const auto start_end = perf_metrics::nowNs();
        
        // Measure stopSection
        const auto stop_begin = perf_metrics::nowNs();
        pma.stopSection("all_tx");
        const auto stop_end = perf_metrics::nowNs();
        
        start_durations.push_back(static_cast<uint64_t>((start_end - start_begin).count()));
        stop_durations.push_back(static_cast<uint64_t>((stop_end - stop_begin).count()));
        pair_durations.push_back(static_cast<uint64_t>((stop_end - start_begin).count()));
    }
    
    // Calculate and print statistics
    const Stats start_stats = calculateStats(start_durations);
    printStats("startSection()", start_stats, NUM_ITERATIONS);
    
    const Stats stop_stats = calculateStats(stop_durations);
    printStats("stopSection()", stop_stats, NUM_ITERATIONS);
    
    const Stats pair_stats = calculateStats(pair_durations);
    printStats("start+stop pair", pair_stats, NUM_ITERATIONS);
}

void benchmark_add_section_duration()
{
    std::cout << "\n=== Benchmarking addSectionDuration Overhead ===" << std::endl;
    
    perf_metrics::PerfMetricsAccumulator pma{
        "lock_wait",
        "tx_burst_loop"
    };
    
    constexpr int NUM_ITERATIONS = 10000;
    constexpr int WARMUP_ITERATIONS = 100;
    
    std::vector<uint64_t> add_durations;
    std::vector<uint64_t> double_add_durations;
    
    add_durations.reserve(NUM_ITERATIONS);
    double_add_durations.reserve(NUM_ITERATIONS);
    
    // Warmup
    std::cout << "Warming up with " << WARMUP_ITERATIONS << " iterations..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        pma.addSectionDuration("lock_wait", 1000);
    }
    pma.reset();
    
    // Benchmark single addSectionDuration
    std::cout << "Running " << NUM_ITERATIONS << " iterations..." << std::endl;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::nowNs();
        pma.addSectionDuration("lock_wait", 1000);
        const auto end = perf_metrics::nowNs();
        
        add_durations.push_back(static_cast<uint64_t>((end - start).count()));
    }
    
    // Benchmark double addSectionDuration (like real usage)
    pma.reset();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::nowNs();
        pma.addSectionDuration("lock_wait", 1000);
        pma.addSectionDuration("tx_burst_loop", 2000);
        const auto end = perf_metrics::nowNs();
        
        double_add_durations.push_back(static_cast<uint64_t>((end - start).count()));
    }
    
    // Calculate and print statistics
    std::cout << "\n";
    const Stats add_stats = calculateStats(add_durations);
    printStats("Single addSectionDuration()", add_stats);
    
    std::cout << "\n";
    const Stats double_add_stats = calculateStats(double_add_durations);
    printStats("Two addSectionDuration() calls", double_add_stats);
}

void benchmark_realistic_workload()
{
    std::cout << "\n=== Benchmarking Realistic Workload (112 handle_sect1 calls) ===" << std::endl;
    
    // Simulate sections used within handle_sect1_c_plane (not outer tx_slot sections)
    // Note: tput_counters and verify are called from tx_slot outer loop (8 times), not from handle_sect1
    perf_metrics::PerfMetricsAccumulator pma{
        "handle_sect1_outer",
        "handle_sect1_inner",
        "sect1_initialize",
        "section_processing",
        "check_if_drop",
        "packet_preparation",
        "packet_send",
        "lock_wait",
        "tx_burst_loop",
        "enqueue_log",
        "packet_stats_update",
        "counter_updates"
    };
    
    constexpr int NUM_ITERATIONS = 1000;
    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int CALLS_PER_ITERATION = 112; // Based on real data showing 112 calls
    constexpr int DROPPED_CALLS = 8;  // Calls that skip packet preparation (104 normal + 8 dropped = 112)
    constexpr int NORMAL_CALLS = CALLS_PER_ITERATION - DROPPED_CALLS;
    
    std::vector<uint64_t> total_durations;
    total_durations.reserve(NUM_ITERATIONS);
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        // Simulate 104 normal calls
        for (int call = 0; call < NORMAL_CALLS; ++call) {
            pma.startSection("handle_sect1_outer");
            pma.startSection("handle_sect1_inner");
            pma.startSection("sect1_initialize");
            pma.stopSection("sect1_initialize");
            pma.startSection("section_processing");
            pma.stopSection("section_processing");
            pma.startSection("check_if_drop");
            pma.stopSection("check_if_drop");
            pma.startSection("packet_preparation");
            pma.stopSection("packet_preparation");
            pma.startSection("packet_send");
            pma.stopSection("packet_send");
            pma.addSectionDuration("lock_wait", 1000);
            pma.addSectionDuration("tx_burst_loop", 2000);
            pma.startSection("enqueue_log");
            pma.stopSection("enqueue_log");
            pma.startSection("packet_stats_update");
            pma.stopSection("packet_stats_update");
            pma.startSection("counter_updates");
            pma.stopSection("counter_updates");
            pma.stopSection("handle_sect1_inner");
            pma.stopSection("handle_sect1_outer");
        }
        // Simulate 8 dropped calls (skip packet_preparation through packet_stats_update)
        for (int call = 0; call < DROPPED_CALLS; ++call) {
            pma.startSection("handle_sect1_outer");
            pma.startSection("handle_sect1_inner");
            pma.startSection("sect1_initialize");
            pma.stopSection("sect1_initialize");
            pma.startSection("section_processing");
            pma.stopSection("section_processing");
            pma.startSection("check_if_drop");
            pma.stopSection("check_if_drop");
            // Skip packet_preparation through packet_stats_update
            pma.startSection("counter_updates");
            pma.stopSection("counter_updates");
            pma.stopSection("handle_sect1_inner");
            pma.stopSection("handle_sect1_outer");
        }
        pma.reset();
    }
    
    // Actual benchmark
    std::cout << "Running " << NUM_ITERATIONS << " iterations with " << CALLS_PER_ITERATION << " calls each (104 normal + 8 dropped)..." << std::endl;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto iteration_start = perf_metrics::nowNs();
        
        // Simulate 104 normal calls (not dropped)
        for (int call = 0; call < NORMAL_CALLS; ++call) {
            pma.startSection("handle_sect1_outer");
            pma.startSection("handle_sect1_inner");
            pma.startSection("sect1_initialize");
            pma.stopSection("sect1_initialize");
            pma.startSection("section_processing");
            pma.stopSection("section_processing");
            pma.startSection("check_if_drop");
            pma.stopSection("check_if_drop");
            pma.startSection("packet_preparation");
            pma.stopSection("packet_preparation");
            pma.startSection("packet_send");
            pma.stopSection("packet_send");
            pma.addSectionDuration("lock_wait", 1000);
            pma.addSectionDuration("tx_burst_loop", 2000);
            pma.startSection("enqueue_log");
            pma.stopSection("enqueue_log");
            pma.startSection("packet_stats_update");
            pma.stopSection("packet_stats_update");
            pma.startSection("counter_updates");
            pma.stopSection("counter_updates");
            pma.stopSection("handle_sect1_inner");
            pma.stopSection("handle_sect1_outer");
        }
        
        // Simulate 8 dropped calls (skip packet_preparation through packet_stats_update)
        for (int call = 0; call < DROPPED_CALLS; ++call) {
            pma.startSection("handle_sect1_outer");
            pma.startSection("handle_sect1_inner");
            pma.startSection("sect1_initialize");
            pma.stopSection("sect1_initialize");
            pma.startSection("section_processing");
            pma.stopSection("section_processing");
            pma.startSection("check_if_drop");
            pma.stopSection("check_if_drop");
            // Skip packet_preparation through packet_stats_update (dropped)
            pma.startSection("counter_updates");
            pma.stopSection("counter_updates");
            pma.stopSection("handle_sect1_inner");
            pma.stopSection("handle_sect1_outer");
        }
        
        const auto iteration_end = perf_metrics::nowNs();
        total_durations.push_back(static_cast<uint64_t>((iteration_end - iteration_start).count()));
        pma.reset();
    }
    
    // Calculate statistics
    const Stats total_stats = calculateStats(total_durations);
    
    printStats("Total for 112 sequences", total_stats, NUM_ITERATIONS);
    
    Stats per_call_stats;
    per_call_stats.min = total_stats.min / CALLS_PER_ITERATION;
    per_call_stats.max = total_stats.max / CALLS_PER_ITERATION;
    per_call_stats.avg = total_stats.avg / CALLS_PER_ITERATION;
    per_call_stats.median = total_stats.median / CALLS_PER_ITERATION;
    per_call_stats.p95 = total_stats.p95 / CALLS_PER_ITERATION;
    per_call_stats.p99 = total_stats.p99 / CALLS_PER_ITERATION;
    printStats("Per-call overhead", per_call_stats);
    
    // Normal call: 10 startSection + 10 stopSection + 2 addSectionDuration = 22 operations
    // Dropped call: 6 startSection + 6 stopSection = 12 operations
    // Total: (104 × 22) + (8 × 12) = 2,288 + 96 = 2,384 operations
    constexpr int TOTAL_OPS = (NORMAL_CALLS * 22) + (DROPPED_CALLS * 12);
    Stats per_op_stats;
    per_op_stats.min = total_stats.min / TOTAL_OPS;
    per_op_stats.max = total_stats.max / TOTAL_OPS;
    per_op_stats.avg = total_stats.avg / TOTAL_OPS;
    per_op_stats.median = total_stats.median / TOTAL_OPS;
    per_op_stats.p95 = total_stats.p95 / TOTAL_OPS;
    per_op_stats.p99 = total_stats.p99 / TOTAL_OPS;
    printStats("Per-operation (2384 total ops)", per_op_stats);
    
    std::cout << "Estimated total overhead for 112 handle_sect1 calls: " 
              << total_stats.avg / 1000 << "μs" << std::endl;
}

void benchmark_nowns_overhead()
{
    std::cout << "\n=== Benchmarking Time Function Overhead ===" << std::endl;
    
    constexpr int NUM_ITERATIONS = 100000;
    std::vector<uint64_t> nowns_single_durations;
    std::vector<uint64_t> nowns_double_durations;
    std::vector<uint64_t> monotonic_single_durations;
    std::vector<uint64_t> monotonic_double_durations;
    
    nowns_single_durations.reserve(NUM_ITERATIONS);
    nowns_double_durations.reserve(NUM_ITERATIONS);
    monotonic_single_durations.reserve(NUM_ITERATIONS);
    monotonic_double_durations.reserve(NUM_ITERATIONS);
    
    // Benchmark nowNs() (original)
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::nowNs();
        const auto t1 = perf_metrics::nowNs();
        const auto end = perf_metrics::nowNs();
        
        nowns_single_durations.push_back(static_cast<uint64_t>((t1 - start).count()));
    }
    Stats nowns_single_stats = calculateStats(nowns_single_durations);
    printStats("nowNs() single", nowns_single_stats, NUM_ITERATIONS);
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto outer_start = perf_metrics::nowNs();
        const auto t1 = perf_metrics::nowNs();
        const auto t2 = perf_metrics::nowNs();
        const auto outer_end = perf_metrics::nowNs();
        
        nowns_double_durations.push_back(static_cast<uint64_t>((t2 - t1).count() + (outer_end - outer_start).count()) / 2);
    }
    Stats nowns_double_stats = calculateStats(nowns_double_durations);
    printStats("nowNs() double", nowns_double_stats, NUM_ITERATIONS);
    
    // Benchmark monotonicNowNs() (new optimized)
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::monotonicNowNs();
        const auto t1 = perf_metrics::monotonicNowNs();
        const auto end = perf_metrics::monotonicNowNs();
        
        monotonic_single_durations.push_back(static_cast<uint64_t>((t1 - start).count()));
    }
    Stats monotonic_single_stats = calculateStats(monotonic_single_durations);
    printStats("monotonicNowNs() single", monotonic_single_stats, NUM_ITERATIONS);
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto outer_start = perf_metrics::monotonicNowNs();
        const auto t1 = perf_metrics::monotonicNowNs();
        const auto t2 = perf_metrics::monotonicNowNs();
        const auto outer_end = perf_metrics::monotonicNowNs();
        
        monotonic_double_durations.push_back(static_cast<uint64_t>((t2 - t1).count() + (outer_end - outer_start).count()) / 2);
    }
    Stats monotonic_double_stats = calculateStats(monotonic_double_durations);
    printStats("monotonicNowNs() double", monotonic_double_stats, NUM_ITERATIONS);
    
    // Compare improvement
    std::cout << "Speedup: " 
              << static_cast<double>(nowns_single_stats.avg) / monotonic_single_stats.avg 
              << "x (single), "
              << static_cast<double>(nowns_double_stats.avg) / monotonic_double_stats.avg 
              << "x (double), savings for 4256 reads: " 
              << (nowns_single_stats.avg - monotonic_single_stats.avg) * 4256 / 1000
              << "μs" << std::endl;
}

void benchmark_individual_operations()
{
    std::cout << "\n=== Benchmarking Individual Operations ===" << std::endl;
    
    perf_metrics::PerfMetricsAccumulator pma{"test_section"};
    
    constexpr int NUM_ITERATIONS = 100000;
    std::vector<uint64_t> durations;
    durations.reserve(NUM_ITERATIONS);
    
    // Benchmark startSection alone
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::nowNs();
        pma.startSection("test_section");
        const auto end = perf_metrics::nowNs();
        durations.push_back(static_cast<uint64_t>((end - start).count()));
        pma.stopSection("test_section"); // Clean up but don't measure
    }
    Stats start_stats = calculateStats(durations);
    printStats("startSection()", start_stats, NUM_ITERATIONS);
    
    // Benchmark stopSection alone
    durations.clear();
    pma.reset();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        pma.startSection("test_section"); // Setup but don't measure
        const auto start = perf_metrics::nowNs();
        pma.stopSection("test_section");
        const auto end = perf_metrics::nowNs();
        durations.push_back(static_cast<uint64_t>((end - start).count()));
    }
    Stats stop_stats = calculateStats(durations);
    printStats("stopSection()", stop_stats, NUM_ITERATIONS);
    
    // Benchmark addSectionDuration
    durations.clear();
    pma.reset();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        const auto start = perf_metrics::nowNs();
        pma.addSectionDuration("test_section", 1000);
        const auto end = perf_metrics::nowNs();
        durations.push_back(static_cast<uint64_t>((end - start).count()));
    }
    Stats add_stats = calculateStats(durations);
    printStats("addSectionDuration()", add_stats, NUM_ITERATIONS);
}

int main()
{
    std::cout << "PerfMetricsAccumulator Performance Benchmark" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        benchmark_nowns_overhead();
        benchmark_individual_operations();
        benchmark_start_stop_overhead();
        benchmark_realistic_workload();
        
        std::cout << "\n=== Benchmark Completed Successfully ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

