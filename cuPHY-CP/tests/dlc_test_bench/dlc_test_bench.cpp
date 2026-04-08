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

#include "dlc_test_bench.hpp"
#include "sendCplaneUnitTest.hpp"
#include "OranPcapComparator.hpp"
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>
#include "nvlog_fmt.hpp"
#include "CLI/CLI.hpp"

// Define benchmark-specific tag
constexpr int TAG_BENCHMARK = TAG_UNIT_TB_COMMON;

/**
 * Main function with command line argument parsing for DLC Test Bench
 * Supports Google Test framework and custom options
 */
int main(int argc, char** argv) {
    
    char root[1024];
    get_root_path(root, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::string yaml_path = std::string(root).append(NVLOG_DEFAULT_CONFIG_FILE);
    pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_path.c_str(), "nvlog_out.log", NULL);

    if (bg_thread_id == -1)
    {
        std::cout << "\n nvlog_fmtlog_init() failed! yaml_path:" << yaml_path.c_str() << std::endl; 
        return -1; 
    }

    // Check if user wants to run benchmarks
    bool run_benchmarks = false;

    CLI::App app{"dlc_test_bench application"};
    app.add_option("-p,--pattern", aerial_fh::TestConfig::instance().pattern_number, "Vector pattern number")->required();
    app.add_flag("--enable-pcap", aerial_fh::TestConfig::instance().enable_pcap, "Enable writing the C-Plane packets into a file")->capture_default_str();
    app.add_option("--pcap-file", aerial_fh::TestConfig::instance().pcap_file_name, "C-Plane packet file name. Saved in /tmp/ location.")->capture_default_str();
    app.add_flag("--benchmark", run_benchmarks, "Enable running the benchmarking suite")->capture_default_str();
    app.add_flag("--enable-perf-profiling", aerial_fh::TestConfig::instance().use_perf_profiler, "Enable Linux Perf tool profiling during benchmarks")->capture_default_str();

    try {
        app.allow_extras(); // Added to not fail on unrecognized google benchmark/ GTEST specific args
        app.parse(argc, argv); 
    } catch (const CLI::ParseError &e) {
        return app.exit(e); 
    }

    if (run_benchmarks) {
        // Initialize Google Benchmark (processes --benchmark_* arguments)
        ::benchmark::Initialize(&argc, argv);
        
        // Disable verification of produced packets in benchmarking mode.
        aerial_fh::TestConfig::instance().verify_cplane = false;
        
        NVLOGC_FMT(TAG_BENCHMARK, "Registering dynamic benchmarks for pattern {}...", 
                   aerial_fh::TestConfig::instance().pattern_number);
        aerial_fh::RegisterDynamicBenchmarks();
        NVLOGC_FMT(TAG_BENCHMARK, "Dynamic benchmark registration complete");
        
        // Display configuration using NVLOG
        NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
        NVLOGC_FMT(TAG_BENCHMARK, "DLC Test Bench Configuration [BENCHMARK mode]");
        NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
        NVLOGC_FMT(TAG_BENCHMARK, "Pattern Number: {}", aerial_fh::TestConfig::instance().pattern_number);
        NVLOGC_FMT(TAG_BENCHMARK, "PCAP Writing: {}", 
                   aerial_fh::TestConfig::instance().enable_pcap ? "Enabled" : "Disabled");
        NVLOGC_FMT(TAG_BENCHMARK, "Output verification: {}", 
                   aerial_fh::TestConfig::instance().verify_cplane ? "Enabled" : "Disabled");
        NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
        
        // CRITICAL: Flush all pending NVLOG messages before benchmark output
        // This ensures all test logs appear BEFORE benchmark results
        NVLOGC_FMT(TAG_BENCHMARK, "--- Starting Benchmark Execution ---");
        fmtlog::poll(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
               
        ::benchmark::RunSpecifiedBenchmarks();
        
        ::benchmark::Shutdown();
        
        // Flush any remaining logs after benchmarks
        fmtlog::poll(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Clean up global benchmark test instance to avoid HDF5 file close errors
        extern std::unique_ptr<aerial_fh::SendCPlaneUnitTest> g_benchmark_test_instance;
        if (g_benchmark_test_instance) {
            std::cout << "\n=== Cleaning up benchmark resources ===" << std::endl;
            std::cout.flush();
            g_benchmark_test_instance->TearDown();
            g_benchmark_test_instance.reset();
        }
        NVLOGC_FMT(TAG_BENCHMARK, "\n=== Benchmarks Complete ===\n");
       
        // Final flush and close logger before exiting benchmark path
        fmtlog::poll(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        nvlog_fmtlog_close(bg_thread_id);

        return 0;
    }
    
    // Otherwise, run Google Test
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners = 
        ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new SummaryLogger);  // Must be done before tests run
    
    // Display configuration using NVLOG
    NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
    NVLOGC_FMT(TAG_BENCHMARK, "DLC Test Bench Configuration [TEST mode]");
    NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
    NVLOGC_FMT(TAG_BENCHMARK, "Pattern Number: {}", aerial_fh::TestConfig::instance().pattern_number);
    NVLOGC_FMT(TAG_BENCHMARK, "PCAP Writing: {}", 
               aerial_fh::TestConfig::instance().enable_pcap ? "Enabled" : "Disabled");
    NVLOGC_FMT(TAG_BENCHMARK, "Output verification: {}", 
               aerial_fh::TestConfig::instance().verify_cplane ? "Enabled" : "Disabled");
    NVLOGC_FMT(TAG_BENCHMARK, "=============================================================================");
    
    // Flush before running tests
    fmtlog::poll(true);
    
    // Run all tests
    int result = RUN_ALL_TESTS();

    fmtlog::poll(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    nvlog_fmtlog_close(bg_thread_id);

    return result;
}
