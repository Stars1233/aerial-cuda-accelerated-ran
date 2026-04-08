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
 * @file perf_metrics_accumulator_unit_test.cpp
 * @brief Unit test for PerfMetricsAccumulator functionality
 */

#include "perf_metrics/perf_metrics_accumulator.hpp"
#include <iostream>
#include <chrono>
#include <thread>

// Use DRV.PERF_METRICS tag for testing (NVLOG_TAG_BASE_CUPHY_DRIVER + 48)
#define TAG_PERF_METRICS 248

void test_basic_functionality() {
    std::cout << "\n=== Testing Basic Functionality ===" << std::endl;
    
    // Create accumulator with pre-registered sections
    perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};
    
    // Run multiple timing cycles
    for (int ii = 0; ii < 10; ii++) {
        pma.startSection("Section 1");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        pma.stopSection("Section 1");
        
        pma.startSection("Section 2");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        pma.stopSection("Section 2");
    }
    
    // Log the accumulated durations
    std::cout << "Logging durations with INFO level:" << std::endl;
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::INFO>();
    
    // Reset and verify empty state
    std::cout << "\nAfter reset:" << std::endl;
    pma.reset();
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::INFO>();
}

void test_error_handling() {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
    perf_metrics::PerfMetricsAccumulator pma{"Valid Section"};
    
    // Test starting non-existent section
    std::cout << "Testing non-existent section (should show error):" << std::endl;
    pma.startSection("Non-existent Section");
    
    // Test stopping non-active section
    std::cout << "Testing stop without start (should show error):" << std::endl;
    pma.stopSection("Valid Section");
    
    // Test double start
    std::cout << "Testing double start (should show error):" << std::endl;
    pma.startSection("Valid Section");
    pma.startSection("Valid Section");  // Should error
    pma.stopSection("Valid Section");
}

void test_different_log_levels() {
    std::cout << "\n=== Testing Different Log Levels ===" << std::endl;
    
    perf_metrics::PerfMetricsAccumulator pma{"Test Section"};
    
    pma.startSection("Test Section");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    pma.stopSection("Test Section");
    
    std::cout << "Testing VERBOSE level:" << std::endl;
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::VERBOSE>();
    
    std::cout << "Testing DEBUG level:" << std::endl;
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::DEBUG>();
    
    std::cout << "Testing WARN level:" << std::endl;
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::WARN>();
    
    std::cout << "Testing ERROR level:" << std::endl;
    pma.logDurations<TAG_PERF_METRICS, perf_metrics::LogLevel::ERROR>();
}

int main() {
    std::cout << "PerfMetricsAccumulator Unit Test Starting..." << std::endl;
    
    try {
        test_basic_functionality();
        test_error_handling();
        test_different_log_levels();
        
        std::cout << "\n=== All Tests Completed ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
