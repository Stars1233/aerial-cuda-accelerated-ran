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
 * @file percentile_tracker_unit_test.cpp
 * @brief Unit test for PercentileTracker functionality
 */

#include "perf_metrics/percentile_tracker.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <random>

// Use DRV.PERF_METRICS tag for testing (NVLOG_TAG_BASE_CUPHY_DRIVER + 48)
#define TAG_PERF_METRICS 248

void test_basic_functionality() {
    std::cout << "\n=== Testing Basic Functionality ===" << std::endl;
    
    // Create tracker with default single-slot mode
    perf_metrics::PercentileTracker tracker(0, 100000, 1000); // 0-100μs range, 1μs buckets
    
    // Add some values
    for (int i = 0; i < 100; i++) {
        tracker.addValue(i * 1000); // Add values from 0μs to 99μs
    }
    
    // Get basic statistics
    int64_t min = tracker.getMinValue();
    int64_t max = tracker.getMaxValue();
    int64_t mean = tracker.getMean();
    int64_t count = tracker.getTotalCount();
    
    std::cout << "Statistics: count=" << count << ", min=" << min 
              << ", max=" << max << ", mean=" << mean << std::endl;
    
    // Get percentiles
    int64_t p50 = tracker.getPercentile(50.0);
    int64_t p90 = tracker.getPercentile(90.0);
    int64_t p99 = tracker.getPercentile(99.0);
    
    std::cout << "Percentiles: p50=" << p50 << ", p90=" << p90 << ", p99=" << p99 << std::endl;
    
    // Log using template method
    tracker.logStats<TAG_PERF_METRICS>(2, "BasicTest");
}

void test_statistics_bundle() {
    std::cout << "\n=== Testing Statistics Bundle ===" << std::endl;
    
    perf_metrics::PercentileTracker tracker(0, 50000, 500); // 0-50μs range, 0.5μs buckets
    
    // Add random values
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(1000, 40000);
    
    for (int i = 0; i < 1000; i++) {
        tracker.addValue(dis(gen));
    }
    
    // Get all statistics in one call
    perf_metrics::PercentileTracker::Statistics stats = tracker.getStatistics();
    
    std::cout << "Bundle Statistics:" << std::endl;
    std::cout << "  Count: " << stats.totalCount << std::endl;
    std::cout << "  Min: " << stats.min << std::endl;
    std::cout << "  Max: " << stats.max << std::endl;
    std::cout << "  Mean: " << stats.mean << std::endl;
    std::cout << "  StdDev: " << stats.stdDev << std::endl;
    std::cout << "  P50: " << stats.p50 << std::endl;
    std::cout << "  P90: " << stats.p90 << std::endl;
    std::cout << "  P99: " << stats.p99 << std::endl;
    std::cout << "  P99.9: " << stats.p999 << std::endl;
    std::cout << "  P99.99: " << stats.p9999 << std::endl;
}

void test_single_slot_mode() {
    std::cout << "\n=== Testing Single Slot Mode ===" << std::endl;
    
    // Create tracker in single-slot mode (default)
    perf_metrics::PercentileTracker tracker(0, 200000, 2000); // 0-200μs range, 2μs buckets
    
    // Add values using simple addValue method
    for (int i = 0; i < 50; i++) {
        tracker.addValue(i * 2000 + 1000); // Values from 1μs to 99μs
    }
    
    // Get statistics for slot 0 explicitly
    int64_t count0 = tracker.getTotalCount(0);
    int64_t min0 = tracker.getMinValue(0);
    int64_t max0 = tracker.getMaxValue(0);
    
    // Get statistics with default (all slots, which is just slot 0)
    int64_t countAll = tracker.getTotalCount();
    int64_t minAll = tracker.getMinValue();
    int64_t maxAll = tracker.getMaxValue();
    
    std::cout << "Slot 0: count=" << count0 << ", min=" << min0 << ", max=" << max0 << std::endl;
    std::cout << "All slots: count=" << countAll << ", min=" << minAll << ", max=" << maxAll << std::endl;
    std::cout << "Should be identical in single-slot mode" << std::endl;
    
    tracker.logStats<TAG_PERF_METRICS>(2, "SingleSlot", 0);
    tracker.logStats<TAG_PERF_METRICS>(2, "AllSlots", -1);
}

void test_multi_slot_mode() {
    std::cout << "\n=== Testing Multi-Slot Mode ===" << std::endl;
    
    // Create tracker with 20 slots (one frame worth)
    perf_metrics::PercentileTracker tracker(0, 100000, 1000, 20); // 20 slots
    
    // Add values to different slots using sfn/slot addressing
    for (uint16_t slot = 0; slot < 20; slot++) {
        for (int i = 0; i < 10; i++) {
            // Add slightly different values for each slot
            tracker.addValue((slot * 1000) + (i * 100), 0, slot);
        }
    }
    
    // Get statistics for specific slots
    std::cout << "Slot 0 stats:" << std::endl;
    tracker.logStats<TAG_PERF_METRICS>(2, "Slot0", 0);
    
    std::cout << "Slot 10 stats:" << std::endl;
    tracker.logStats<TAG_PERF_METRICS>(2, "Slot10", 10);
    
    std::cout << "All slots combined:" << std::endl;
    tracker.logStats<TAG_PERF_METRICS>(2, "Combined", -1);
    
    // Test combined statistics
    int64_t totalCount = tracker.getTotalCount(-1);
    std::cout << "Total count across all slots: " << totalCount << std::endl;
}

void test_reset_functionality() {
    std::cout << "\n=== Testing Reset Functionality ===" << std::endl;
    
    perf_metrics::PercentileTracker tracker(0, 100000, 1000, 5); // 5 slots
    
    // Add values to multiple slots
    for (int slot = 0; slot < 5; slot++) {
        for (int i = 0; i < 20; i++) {
            tracker.addValue(i * 1000, slot);
        }
    }
    
    std::cout << "Before reset:" << std::endl;
    std::cout << "  Slot 0 count: " << tracker.getTotalCount(0) << std::endl;
    std::cout << "  Slot 2 count: " << tracker.getTotalCount(2) << std::endl;
    std::cout << "  Total count: " << tracker.getTotalCount(-1) << std::endl;
    
    // Reset specific slot
    tracker.reset(0);
    std::cout << "\nAfter resetting slot 0:" << std::endl;
    std::cout << "  Slot 0 count: " << tracker.getTotalCount(0) << std::endl;
    std::cout << "  Slot 2 count: " << tracker.getTotalCount(2) << std::endl;
    std::cout << "  Total count: " << tracker.getTotalCount(-1) << std::endl;
    
    // Reset all slots
    tracker.reset(-1);
    std::cout << "\nAfter resetting all slots:" << std::endl;
    std::cout << "  Slot 0 count: " << tracker.getTotalCount(0) << std::endl;
    std::cout << "  Slot 2 count: " << tracker.getTotalCount(2) << std::endl;
    std::cout << "  Total count: " << tracker.getTotalCount(-1) << std::endl;
}

void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    perf_metrics::PercentileTracker tracker(1000, 100000, 1000); // 1μs - 100μs range
    
    // Test clamping - value below minimum
    std::cout << "Testing value below minimum (should clamp):" << std::endl;
    tracker.addValue(500);
    
    // Test clamping - value above maximum
    std::cout << "Testing value above maximum (should clamp):" << std::endl;
    tracker.addValue(200000);
    
    // Add some normal values
    for (int i = 0; i < 10; i++) {
        tracker.addValue(50000 + (i * 1000));
    }
    
    std::cout << "Statistics after edge case testing:" << std::endl;
    tracker.logStats<TAG_PERF_METRICS>(2, "EdgeCase");
    
    // Test out of bounds slot index
    std::cout << "\nTesting out of bounds slot index (should warn):" << std::endl;
    int64_t count = tracker.getTotalCount(100);
    std::cout << "Count from invalid slot index: " << count << std::endl;
}

void test_memory_footprint() {
    std::cout << "\n=== Testing Memory Footprint Logging ===" << std::endl;
    
    // Small tracker
    std::cout << "Small tracker (100 buckets, 1 slot):" << std::endl;
    perf_metrics::PercentileTracker small(0, 100000, 1000, 1);
    small.logMemoryFootprint<TAG_PERF_METRICS>(2);
    
    // Medium tracker
    std::cout << "\nMedium tracker (500 buckets, 20 slots):" << std::endl;
    perf_metrics::PercentileTracker medium(0, 500000, 1000, 20);
    medium.logMemoryFootprint<TAG_PERF_METRICS>(2);
    
    // Large tracker
    std::cout << "\nLarge tracker (1000 buckets, 80 slots):" << std::endl;
    perf_metrics::PercentileTracker large(0, 1000000, 1000, 80);
    large.logMemoryFootprint<TAG_PERF_METRICS>(2);
}

void test_move_semantics() {
    std::cout << "\n=== Testing Move Semantics ===" << std::endl;
    
    // Create and populate a tracker
    perf_metrics::PercentileTracker tracker1(0, 100000, 1000);
    for (int i = 0; i < 50; i++) {
        tracker1.addValue(i * 1000);
    }
    
    std::cout << "Original tracker stats:" << std::endl;
    tracker1.logStats<TAG_PERF_METRICS>(2, "Original");
    
    // Move construct
    perf_metrics::PercentileTracker tracker2(std::move(tracker1));
    std::cout << "\nAfter move construction:" << std::endl;
    tracker2.logStats<TAG_PERF_METRICS>(2, "MovedTo");
    
    // Move assign
    perf_metrics::PercentileTracker tracker3(0, 50000, 500);
    tracker3 = std::move(tracker2);
    std::cout << "\nAfter move assignment:" << std::endl;
    tracker3.logStats<TAG_PERF_METRICS>(2, "MoveAssigned");
}

void test_slot_index_calculation() {
    std::cout << "\n=== Testing Slot Index Calculation ===" << std::endl;
    
    // Create tracker with 80 slots (4 frames worth)
    perf_metrics::PercentileTracker tracker(0, 100000, 1000, 80);
    
    // Test various sfn/slot combinations
    std::cout << "Testing sfn/slot to index mapping:" << std::endl;
    std::cout << "  SFN=0, Slot=0 -> Index=" << tracker.getSlotIndex(0, 0) << std::endl;
    std::cout << "  SFN=0, Slot=19 -> Index=" << tracker.getSlotIndex(0, 19) << std::endl;
    std::cout << "  SFN=1, Slot=0 -> Index=" << tracker.getSlotIndex(1, 0) << std::endl;
    std::cout << "  SFN=1, Slot=19 -> Index=" << tracker.getSlotIndex(1, 19) << std::endl;
    std::cout << "  SFN=3, Slot=19 -> Index=" << tracker.getSlotIndex(3, 19) << std::endl;
}

// Helper function to get current time in nanoseconds
int64_t getCurrentTimeNs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

// Helper function to busy-spin for a specified duration
void busySpin(int64_t durationNs) {
    int64_t startTime = getCurrentTimeNs();
    int64_t endTime = startTime + durationNs;
    while (getCurrentTimeNs() < endTime) {
        // Busy wait
    }
}

void test_multi_slot_with_timing() {
    std::cout << "\n=== Testing Multi-Slot with Actual Timing ===" << std::endl;
    
    // Create tracker with 10 slots, tracking 0-1000μs range with 1μs buckets
    perf_metrics::PercentileTracker tracker(0, 1000000, 1000, 10);
    
    const int iterations = 10;  // 10 iterations per slot
    
    std::cout << "Running " << iterations << " iterations per slot..." << std::endl;
    
    // First 5 slots: measure sleep time (slot_index * 100μs)
    std::cout << "\nSlots 0-4: Testing sleep timing" << std::endl;
    for (int slotIdx = 0; slotIdx < 5; slotIdx++) {
        int64_t targetDelayUs = slotIdx * 100;  // 0, 100, 200, 300, 400 μs
        
        for (int i = 0; i < iterations; i++) {
            int64_t startTime = getCurrentTimeNs();
            std::this_thread::sleep_for(std::chrono::microseconds(targetDelayUs));
            int64_t endTime = getCurrentTimeNs();
            int64_t measuredTime = endTime - startTime;
            
            tracker.addValue(measuredTime, slotIdx);
        }
        
        auto stats = tracker.getStatistics(slotIdx);
        std::cout << "  Slot " << slotIdx << " (target=" << targetDelayUs << "μs): "
                  << "mean=" << stats.mean / 1000 << "μs, "
                  << "min=" << stats.min / 1000 << "μs, "
                  << "max=" << stats.max / 1000 << "μs" << std::endl;
    }
    
    // Second 5 slots: measure busy-spin time ((slot_index-5) * 100μs)
    std::cout << "\nSlots 5-9: Testing busy-spin timing" << std::endl;
    for (int slotIdx = 5; slotIdx < 10; slotIdx++) {
        int64_t targetDelayUs = (slotIdx - 5) * 100;  // 0, 100, 200, 300, 400 μs
        int64_t targetDelayNs = targetDelayUs * 1000;
        
        for (int i = 0; i < iterations; i++) {
            int64_t startTime = getCurrentTimeNs();
            busySpin(targetDelayNs);
            int64_t endTime = getCurrentTimeNs();
            int64_t measuredTime = endTime - startTime;
            
            tracker.addValue(measuredTime, slotIdx);
        }
        
        auto stats = tracker.getStatistics(slotIdx);
        std::cout << "  Slot " << slotIdx << " (target=" << targetDelayUs << "μs): "
                  << "mean=" << stats.mean / 1000 << "μs, "
                  << "min=" << stats.min / 1000 << "μs, "
                  << "max=" << stats.max / 1000 << "μs" << std::endl;
    }
    
    // Show combined statistics
    std::cout << "\nCombined statistics across all slots:" << std::endl;
    tracker.logStats<TAG_PERF_METRICS>(2, "AllSlots", -1);
    
    // Show per-slot statistics with logging
    std::cout << "\nDetailed per-slot statistics:" << std::endl;
    for (int slotIdx = 0; slotIdx < 10; slotIdx++) {
        char slotName[32];
        snprintf(slotName, sizeof(slotName), "Slot%d", slotIdx);
        tracker.logStats<TAG_PERF_METRICS>(2, slotName, slotIdx);
    }
    
    std::cout << "\nNote: Sleep timing typically has higher variance due to OS scheduling." << std::endl;
    std::cout << "Busy-spin timing should be more consistent but less power-efficient." << std::endl;
}

int main() {
    std::cout << "PercentileTracker Unit Test Starting..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_basic_functionality();
        test_statistics_bundle();
        test_single_slot_mode();
        test_multi_slot_mode();
        test_reset_functionality();
        test_edge_cases();
        test_memory_footprint();
        test_move_semantics();
        test_slot_index_calculation();
        test_multi_slot_with_timing();
        
        std::cout << "\n=========================================" << std::endl;
        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

