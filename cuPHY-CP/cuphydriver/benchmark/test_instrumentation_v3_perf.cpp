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

/*
 * Performance test for v3 task instrumentation implementation
 * Tests the new dependency-free API with PercentileTracker for accurate percentile measurements
 */

// Increase fmtlog queue size to handle high-throughput logging without drops
// Default is (1 << 24) = 16MB, we increase to 256MB to handle 100k+ logs
#ifndef FMTLOG_QUEUE_SIZE
#define FMTLOG_QUEUE_SIZE (1 << 28)  // 256 MB
#endif

#include "task_instrumentation_v3.hpp"
#include "perf_metrics/percentile_tracker.hpp"
#include "nvlog.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <sched.h>

// NOTE: NVLOGI_FMT is NOT stubbed out - measuring real logging overhead

// ============================================================================
// Configuration & Command Line Arguments
// ============================================================================

struct BenchmarkConfig {
    int64_t num_iterations = 100000;
    int num_subtasks = 10;
    int64_t warmup_iterations = 1000;
    std::string mode = "full_tracing";
    std::string log_file = "/tmp/test_instrumentation_v3_perf.log";
    int bench_core = 8;   // core for the benchmark (main) thread
    int log_core   = 9;   // core for the bg_fmtlog polling thread
    int pmu_type   = 3;   // PMU_TYPE: 0=disabled, 1=general, 2=topdown(Grace), 3=cache(Grace)
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  -n, --iterations <num>    Number of iterations (default: 100000)\n";
    std::cout << "  -s, --subtasks <num>      Number of subtasks per iteration (default: 10, max: 1000)\n";
    std::cout << "  -w, --warmup <num>        Number of warmup iterations (default: 1000)\n";
    std::cout << "  -m, --mode <mode>         Mode to benchmark (default: full_tracing)\n";
    std::cout << "                            Modes: baseline, disabled, full_tracing,\n";
    std::cout << "                                   start_end_only, ctor_dtor, all\n";
    std::cout << "  -o, --output <path>       Write nvlog output to file (default: /tmp/test_instrumentation_v3_perf.log)\n";
    std::cout << "  -c, --core <num>          CPU core for benchmark thread (default: 8)\n";
    std::cout << "  --log-core <num>          CPU core for fmtlog polling thread (default: 9)\n";
    std::cout << "  -p, --pmu <type>          PMU counter type (default: 3)\n";
    std::cout << "                            0=disabled, 1=general, 2=topdown(Grace), 3=cache(Grace)\n";
    std::cout << "  -h, --help                Display this help message\n";
}

BenchmarkConfig parse_args(int argc, char* argv[]) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
            int64_t val = std::strtoll(argv[++i], nullptr, 10);
            if (val > 0) {
                config.num_iterations = val;
            } else {
                std::cerr << "Error: iterations must be > 0\n";
                exit(1);
            }
        }
        else if ((arg == "-s" || arg == "--subtasks") && i + 1 < argc) {
            int val = std::atoi(argv[++i]);
            if (val > 0 && val <= 1000) {
                config.num_subtasks = val;
            } else {
                std::cerr << "Error: subtasks must be between 1 and 1000\n";
                exit(1);
            }
        }
        else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            int64_t val = std::strtoll(argv[++i], nullptr, 10);
            if (val >= 0) {
                config.warmup_iterations = val;
            } else {
                std::cerr << "Error: warmup must be >= 0\n";
                exit(1);
            }
        }
        else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "baseline" || mode == "disabled" || mode == "full_tracing" ||
                mode == "start_end_only" || mode == "ctor_dtor" || mode == "all") {
                config.mode = mode;
            } else {
                std::cerr << "Error: unknown mode '" << mode << "'\n";
                std::cerr << "Valid modes: baseline, disabled, full_tracing, start_end_only, ctor_dtor, all\n";
                exit(1);
            }
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.log_file = argv[++i];
        }
        else if ((arg == "-c" || arg == "--core") && i + 1 < argc) {
            int val = std::atoi(argv[++i]);
            if (val >= 0) {
                config.bench_core = val;
            } else {
                std::cerr << "Error: core must be >= 0\n";
                exit(1);
            }
        }
        else if (arg == "--log-core" && i + 1 < argc) {
            int val = std::atoi(argv[++i]);
            if (val >= 0) {
                config.log_core = val;
            } else {
                std::cerr << "Error: log-core must be >= 0\n";
                exit(1);
            }
        }
        else if ((arg == "-p" || arg == "--pmu") && i + 1 < argc) {
            int val = std::atoi(argv[++i]);
            if (val >= 0 && val <= 3) {
                config.pmu_type = val;
            } else {
                std::cerr << "Error: pmu type must be 0-3\n";
                exit(1);
            }
        }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

// ============================================================================
// Thread Affinity Helpers
// ============================================================================

static void pin_current_thread(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        std::cerr << "  ERROR: failed to pin thread " << pthread_self()
                  << " to core " << core << "\n";
    } else {
        int actual = sched_getcpu();
        std::cout << "  Thread " << pthread_self() << " pinned to core " << core
                  << " (running on core " << actual << ")\n";
    }
}

// ============================================================================
// Performance Benchmark Code
// ============================================================================

class BenchmarkTimer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    int64_t elapsed_ns() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    }
};

volatile int g_dummy_sink = 0;
void sink_value(int val) { g_dummy_sink += val; }

// Benchmark baseline with PercentileTracker
void benchmark_baseline(perf_metrics::PercentileTracker& tracker, const BenchmarkConfig& config) {
    BenchmarkTimer timer;

    // Warmup
    for (int64_t i = 0; i < config.warmup_iterations; ++i) {
        int sum = 0;
        for (int j = 0; j < config.num_subtasks; ++j) sum += j;
        sink_value(sum);
    }

    // Actual benchmark - measure each iteration
    for (int64_t i = 0; i < config.num_iterations; ++i) {
        timer.start();
        int sum = 0;
        for (int j = 0; j < config.num_subtasks; ++j) sum += j;
        sink_value(sum);
        int64_t latency = timer.elapsed_ns();
        tracker.addValue(latency);
    }
}

// Benchmark a specific mode with PercentileTracker
void benchmark_mode(TracingMode mode, perf_metrics::PercentileTracker& tracker,
                    const BenchmarkConfig& config, PMUDeltaSummarizer* pmu) {
    TaskInstrumentationContext ctx(mode, 123, 45, 6, pmu);
    BenchmarkTimer timer;

    // Warmup
    for (int64_t i = 0; i < config.warmup_iterations; ++i) {
        {
            TaskInstrumentation ti(ctx, "Test Task", config.num_subtasks);
            int sum = 0;
            for (int j = 0; j < config.num_subtasks; ++j) {
                ti.add("Subtask");
                sum += j;
            }
            sink_value(sum);
        }
    }

    // Actual benchmark - measure each iteration including destructor
    for (int64_t i = 0; i < config.num_iterations; ++i) {
        timer.start();
        {
            TaskInstrumentation ti(ctx, "Test Task", config.num_subtasks);
            int sum = 0;
            for (int j = 0; j < config.num_subtasks; ++j) {
                ti.add("Subtask");
                sum += j;
            }
            sink_value(sum);
        }  // Destructor runs here before measurement
        int64_t latency = timer.elapsed_ns();
        tracker.addValue(latency);
    }
}

// Benchmark constructor/destructor only with PercentileTracker
void benchmark_ctor_dtor(perf_metrics::PercentileTracker& tracker,
                         const BenchmarkConfig& config, PMUDeltaSummarizer* pmu) {
    TaskInstrumentationContext ctx(TracingMode::DISABLED, 123, 45, 6, pmu);
    BenchmarkTimer timer;

    // Warmup
    for (int64_t i = 0; i < config.warmup_iterations; ++i) {
        {
            TaskInstrumentation ti(ctx, "Test Task", config.num_subtasks);
        }
    }

    // Actual benchmark - measure each iteration including destructor
    for (int64_t i = 0; i < config.num_iterations; ++i) {
        timer.start();
        {
            TaskInstrumentation ti(ctx, "Test Task", config.num_subtasks);
        }  // Destructor runs here before measurement
        int64_t latency = timer.elapsed_ns();
        tracker.addValue(latency);
    }
}

void print_detailed_stats(const char* mode_name, const perf_metrics::PercentileTracker& tracker,
                         const perf_metrics::PercentileTracker::Statistics& stats,
                         const perf_metrics::PercentileTracker::Statistics* baseline_stats = nullptr) {
    std::cout << "\n" << mode_name << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "  Iterations:        " << stats.totalCount << "\n";
    std::cout << "  Min:               " << stats.min << " ns\n";
    std::cout << "  Max:               " << stats.max << " ns\n";
    std::cout << "  Mean:              " << stats.mean << " ns\n";
    std::cout << "  Std Dev:           " << stats.stdDev << " ns\n";
    std::cout << "  50th percentile:   " << stats.p50 << " ns\n";
    std::cout << "  90th percentile:   " << stats.p90 << " ns\n";
    std::cout << "  99th percentile:   " << stats.p99 << " ns\n";
    std::cout << "  99.9th percentile: " << stats.p999 << " ns  [PRIMARY METRIC]\n";
    std::cout << "  99.99th pctl:      " << stats.p9999 << " ns\n";
    std::cout << "  99.999th pctl:     " << tracker.getPercentile(99.999) << " ns\n";

    if (baseline_stats) {
        double overhead_mean = static_cast<double>(stats.mean) - static_cast<double>(baseline_stats->mean);
        double overhead_p999 = static_cast<double>(stats.p999) - static_cast<double>(baseline_stats->p999);
        double overhead_pct_mean = (overhead_mean / static_cast<double>(baseline_stats->mean)) * 100.0;
        double overhead_pct_p999 = (overhead_p999 / static_cast<double>(baseline_stats->p999)) * 100.0;

        std::cout << "\n  Overhead vs Baseline:\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "    Mean:     +" << overhead_mean << " ns (+" << overhead_pct_mean << "%)\n";
        std::cout << "    99.9th:   +" << overhead_p999 << " ns (+" << overhead_pct_p999 << "%)\n";
    }
}

void print_comparison_table(const perf_metrics::PercentileTracker::Statistics& baseline,
                           const perf_metrics::PercentileTracker::Statistics& mode0,
                           const perf_metrics::PercentileTracker::Statistics& mode1,
                           const perf_metrics::PercentileTracker::Statistics& mode2,
                           const perf_metrics::PercentileTracker::Statistics& ctor_dtor) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Summary Comparison Table\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << std::left << std::setw(25) << "Mode"
              << std::right << std::setw(12) << "Mean (ns)"
              << std::setw(12) << "p99.9 (ns)"
              << std::setw(14) << "Overhead (%)"
              << std::setw(17) << "p99.9 OH (%)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::fixed << std::setprecision(0);

    // Baseline
    std::cout << std::left << std::setw(25) << "Baseline"
              << std::right << std::setw(12) << baseline.mean
              << std::setw(12) << baseline.p999
              << std::setw(14) << "-"
              << std::setw(17) << "-" << "\n";

    // DISABLED mode
    double mode0_oh_mean = ((static_cast<double>(mode0.mean) / baseline.mean) - 1.0) * 100.0;
    double mode0_oh_p999 = ((static_cast<double>(mode0.p999) / baseline.p999) - 1.0) * 100.0;
    std::cout << std::left << std::setw(25) << "DISABLED"
              << std::right << std::setw(12) << mode0.mean
              << std::setw(12) << mode0.p999
              << std::setw(13) << std::fixed << std::setprecision(2) << mode0_oh_mean << "%"
              << std::setw(16) << mode0_oh_p999 << "%" << "\n";

    // FULL_TRACING mode (PRIMARY TARGET)
    double mode1_oh_mean = ((static_cast<double>(mode1.mean) / baseline.mean) - 1.0) * 100.0;
    double mode1_oh_p999 = ((static_cast<double>(mode1.p999) / baseline.p999) - 1.0) * 100.0;
    std::cout << std::left << std::setw(25) << "FULL_TRACING *PRIMARY*"
              << std::right << std::setw(12) << std::fixed << std::setprecision(0) << mode1.mean
              << std::setw(12) << mode1.p999
              << std::setw(13) << std::fixed << std::setprecision(2) << mode1_oh_mean << "%"
              << std::setw(16) << mode1_oh_p999 << "%" << "\n";

    // START_END_ONLY mode
    double mode2_oh_mean = ((static_cast<double>(mode2.mean) / baseline.mean) - 1.0) * 100.0;
    double mode2_oh_p999 = ((static_cast<double>(mode2.p999) / baseline.p999) - 1.0) * 100.0;
    std::cout << std::left << std::setw(25) << "START_END_ONLY"
              << std::right << std::setw(12) << std::fixed << std::setprecision(0) << mode2.mean
              << std::setw(12) << mode2.p999
              << std::setw(13) << std::fixed << std::setprecision(2) << mode2_oh_mean << "%"
              << std::setw(16) << mode2_oh_p999 << "%" << "\n";

    // Constructor/Destructor
    double ctor_oh_mean = ((static_cast<double>(ctor_dtor.mean) / baseline.mean) - 1.0) * 100.0;
    double ctor_oh_p999 = ((static_cast<double>(ctor_dtor.p999) / baseline.p999) - 1.0) * 100.0;
    std::cout << std::left << std::setw(25) << "Ctor+Dtor only"
              << std::right << std::setw(12) << std::fixed << std::setprecision(0) << ctor_dtor.mean
              << std::setw(12) << ctor_dtor.p999
              << std::setw(13) << std::fixed << std::setprecision(2) << ctor_oh_mean << "%"
              << std::setw(16) << ctor_oh_p999 << "%" << "\n";

    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config = parse_args(argc, argv);

    const bool run_all = (config.mode == "all");

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Task Instrumentation v3 Performance Test\n";
    std::cout << "With PercentileTracker for Accurate Percentile Measurements\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Iterations:        " << config.num_iterations << "\n";
    std::cout << "  Subtasks:          " << config.num_subtasks << "\n";
    std::cout << "  Warmup iterations: " << config.warmup_iterations << "\n";
    std::cout << "  Mode:              " << config.mode << "\n";
    std::cout << "  Log output:        " << (config.log_file.empty() ? "console" : config.log_file) << "\n";
    std::cout << "  Bench core:        " << config.bench_core << "\n";
    std::cout << "  Log core:          " << config.log_core << "\n";

    const char* pmu_names[] = {"DISABLED", "GENERAL", "TOPDOWN (Grace)", "CACHE_METRICS (Grace)"};
    std::cout << "  PMU type:          " << config.pmu_type
              << " (" << pmu_names[config.pmu_type] << ")\n";

    std::cout << "  PercentileTracker: 0-50μs range, 100ns buckets\n";
    std::cout << "\n";

    // Pin to log_core first so the fmtlog polling thread inherits it,
    // then re-pin to bench_core for the actual benchmark loop.
    // This mirrors how cuphycontroller uses low_priority_core.
    std::cout << "Pinning main thread to core " << config.log_core
              << " (fmtlog inheritance target)...\n";
    pin_current_thread(config.log_core);
    if (!config.log_file.empty()) {
        std::cout << "Starting fmtlog polling thread (inherits core "
                  << config.log_core << ")...\n";
        nvlog_c_init(config.log_file.c_str());
    }
    std::cout << "Re-pinning main thread to core " << config.bench_core
              << " (benchmark core)...\n";
    pin_current_thread(config.bench_core);
    std::cout << "Thread layout: benchmark=core " << config.bench_core
              << ", fmtlog=core " << config.log_core << "\n\n";

    // Create PMU counters directly (no Worker dependency).
    // perf_event_open attaches to the calling thread, so this must run after core pinning.
    PMUDeltaSummarizer pmu(static_cast<PMU_TYPE>(config.pmu_type));
    std::cout << "PMU counters initialized on benchmark thread\n\n";

    // PercentileTracker(0, 50000, 100): 0-50us range, 100ns bucket size, 500 buckets
    perf_metrics::PercentileTracker baseline_tracker(0, 50000, 100);
    perf_metrics::PercentileTracker mode0_tracker(0, 50000, 100);  // DISABLED
    perf_metrics::PercentileTracker mode1_tracker(0, 50000, 100);  // FULL_TRACING
    perf_metrics::PercentileTracker mode2_tracker(0, 50000, 100);  // START_END_ONLY
    perf_metrics::PercentileTracker ctor_dtor_tracker(0, 50000, 100);

    std::cout << "Running benchmarks...\n\n";

    int total_steps = 1;  // baseline always runs
    if (run_all) {
        total_steps += 4;  // disabled, full_tracing, start_end_only, ctor_dtor
    } else if (config.mode != "baseline") {
        total_steps += 1;  // one selected mode
    }
    int step = 1;

    std::cout << "[" << step++ << "/" << total_steps << "] Baseline (no instrumentation)...";
    std::cout.flush();
    benchmark_baseline(baseline_tracker, config);
    std::cout << " DONE\n";

    if (run_all || config.mode == "disabled") {
        std::cout << "[" << step++ << "/" << total_steps << "] Mode DISABLED...";
        std::cout.flush();
        benchmark_mode(TracingMode::DISABLED, mode0_tracker, config, &pmu);
        std::cout << " DONE\n";
    }

    if (run_all || config.mode == "full_tracing") {
        std::cout << "[" << step++ << "/" << total_steps << "] Mode FULL_TRACING [PRIMARY TARGET]...";
        std::cout.flush();
        benchmark_mode(TracingMode::FULL_TRACING, mode1_tracker, config, &pmu);
        std::cout << " DONE\n";
    }

    if (run_all || config.mode == "start_end_only") {
        std::cout << "[" << step++ << "/" << total_steps << "] Mode START_END_ONLY...";
        std::cout.flush();
        benchmark_mode(TracingMode::START_END_ONLY, mode2_tracker, config, &pmu);
        std::cout << " DONE\n";
    }

    if (run_all || config.mode == "ctor_dtor") {
        std::cout << "[" << step++ << "/" << total_steps << "] Constructor + Destructor only...";
        std::cout.flush();
        benchmark_ctor_dtor(ctor_dtor_tracker, config, &pmu);
        std::cout << " DONE\n";
    }

    auto baseline_stats   = baseline_tracker.getStatistics();
    auto mode0_stats      = mode0_tracker.getStatistics();
    auto mode1_stats      = mode1_tracker.getStatistics();
    auto mode2_stats      = mode2_tracker.getStatistics();
    auto ctor_dtor_stats  = ctor_dtor_tracker.getStatistics();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Detailed Statistics (All values in nanoseconds)\n";
    std::cout << std::string(80, '=') << "\n";

    if (run_all) {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
        print_detailed_stats("DISABLED mode", mode0_tracker, mode0_stats, &baseline_stats);
        print_detailed_stats("FULL_TRACING mode [PRIMARY TARGET]", mode1_tracker, mode1_stats, &baseline_stats);
        print_detailed_stats("START_END_ONLY mode", mode2_tracker, mode2_stats, &baseline_stats);
        print_detailed_stats("Constructor + Destructor only", ctor_dtor_tracker, ctor_dtor_stats, &baseline_stats);
        print_comparison_table(baseline_stats, mode0_stats, mode1_stats, mode2_stats, ctor_dtor_stats);
    } else if (config.mode == "baseline") {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
    } else if (config.mode == "disabled") {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
        print_detailed_stats("DISABLED mode", mode0_tracker, mode0_stats, &baseline_stats);
    } else if (config.mode == "full_tracing") {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
        print_detailed_stats("FULL_TRACING mode [PRIMARY TARGET]", mode1_tracker, mode1_stats, &baseline_stats);
    } else if (config.mode == "start_end_only") {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
        print_detailed_stats("START_END_ONLY mode", mode2_tracker, mode2_stats, &baseline_stats);
    } else if (config.mode == "ctor_dtor") {
        print_detailed_stats("Baseline (no instrumentation)", baseline_tracker, baseline_stats);
        print_detailed_stats("Constructor + Destructor only", ctor_dtor_tracker, ctor_dtor_stats, &baseline_stats);
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Additional Information\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "sizeof(TaskInstrumentation): " << sizeof(TaskInstrumentation) << " bytes\n\n";

    if (config.mode == "full_tracing" || run_all) {
        std::cout << "Key Takeaway:\n";
        std::cout << "  Focus on FULL_TRACING 99.9th percentile = " << mode1_stats.p999 << " ns\n";
        std::cout << "  This is the critical latency target for production use.\n\n";
    }

    if (!config.log_file.empty()) {
        nvlog_c_close();
    }

    return 0;
}
