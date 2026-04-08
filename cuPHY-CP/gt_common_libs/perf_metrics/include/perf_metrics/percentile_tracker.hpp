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

#ifndef PERF_METRICS_PERCENTILE_TRACKER_HPP
#define PERF_METRICS_PERCENTILE_TRACKER_HPP

#include <cstdint>   // For uint16_t, uint32_t, etc.
#include <cstring>   // For strlen, strncpy
#include <cmath>     // For math functions
#include <limits>    // For std::numeric_limits
#include <algorithm> // For std::min, std::max
#include <time.h>    // For timespec
#include <stdio.h>   // For snprintf
#include <string>    // Needed for std::to_string
#include <cinttypes> // For PRId64 format specifiers
#include "nvlog.hpp"

// Ensure uint types are defined if not already defined by cstdint
#ifndef UINT32_MAX
typedef unsigned int uint32_t;
#endif
#ifndef UINT16_MAX

typedef unsigned short uint16_t;
#endif

// Maximum length for prefix strings
#define PT_MAX_PREFIX_LENGTH 64

namespace perf_metrics {

class PercentileTracker {
public:
    // Statistics bundle structure to hold all statistics values
    struct Statistics {
        int64_t totalCount;
        int64_t min;
        int64_t max;
        int64_t mean;
        int64_t stdDev;
        int64_t p50;
        int64_t p90;
        int64_t p99;
        int64_t p999;
        int64_t p9999;
    };
    
    PercentileTracker(int64_t lowestTrackableValue = 0, 
                      int64_t highestTrackableValue = 500000, // 500 μs in ns
                      int64_t bucketSize = 1000,              // Default bucket size of 1usec
                      int32_t numSlots = -1);                // Default to single slot mode
    ~PercentileTracker();
    
    // Add move semantics
    PercentileTracker(PercentileTracker&& other) noexcept;
    PercentileTracker& operator=(PercentileTracker&& other) noexcept;
    
    // Original method that accepts sfn and slot to determine which slot index to use
    void addValue(int64_t value, uint16_t sfn, uint16_t slot);
    
    // Overloaded method that only accepts value (for single slot mode)
    void addValue(int64_t value);
    
    // Overloaded method that accepts value and direct slot index
    void addValue(int64_t value, int slotIndex);
    
    // Retrieve percentile for a specific slot index or aggregated across all slots
    int64_t getPercentile(double percentile, int slotIndex = -1) const;
    
    // Reset statistics for a specific slot index or all slots
    void reset(int slotIndex = -1);
    
    // Gets all statistics for a specific slot index or aggregated across all slots
    Statistics getStatistics(int slotIndex = -1) const;
    
    // Log statistics with configurable tag and level
    // Log levels: 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG
    // Use template for tag to ensure it's a compile-time constant
    template<int PT_LOGGING_TAG>
    void logStats(uint32_t logLevel = 2, const char* prefix = "", 
                 int slotIndex = -1) const {
        // Get all statistics for the specified slot index
        Statistics stats = getStatistics(slotIndex);
        
        // Log using the helper function (pass slotIndex separately now)
        logStatistics<PT_LOGGING_TAG>(logLevel, prefix, stats, slotIndex);
    }
    
    // Additional utility methods - work on all slots by default
    int64_t getMinValue(int slotIndex = -1) const;
    int64_t getMaxValue(int slotIndex = -1) const;
    int64_t getMean(int slotIndex = -1) const;
    int64_t getStdDeviation(int slotIndex = -1) const;
    int64_t getTotalCount(int slotIndex = -1) const;
    
    // Convert sfn/slot to a slot index (0-79) in our buffer
    // Made public so it can be used externally
    uint32_t getSlotIndex(uint16_t sfn, uint16_t slot) const;
    
    // Log memory usage details of this object
    // Use template for tag to ensure it's a compile-time constant
    template<int PT_LOGGING_TAG>
    void logMemoryFootprint(uint32_t logLevel = 2) const {
        // Calculate memory usage
        size_t bucketSize = sizeof(HistogramBucket);
        size_t slotStatsSize = sizeof(SlotStats);
        size_t bucketArraySize = bucketSize * bucketCount_;
        size_t totalBucketArraysSize = bucketArraySize * numSlots_;
        size_t slotStatsArraySize = slotStatsSize * numSlots_;
        size_t totalObjectSize = sizeof(PercentileTracker) + slotStatsArraySize + totalBucketArraysSize;
        
        // Calculate total number of buckets
        size_t totalBuckets = bucketCount_ * numSlots_;
        
        double totalMB = static_cast<double>(totalObjectSize) / (1024 * 1024);
        
        // Use mutable buffer for formatting
        snprintf(formatBuffer_, PT_MAX_PREFIX_LENGTH, "Memory Usage");
        
        // Log at the specified level using compile-time tag
        switch (logLevel) {
            case 0: // ERROR
                NVLOGE_FMT(PT_LOGGING_TAG, AERIAL_CUPHYDRV_API_EVENT, "PercentileTracker {}: config={} buckets x {} slots, total_buckets={}, object={} bytes, slot_array={} bytes, bucket_arrays={} bytes, total={} bytes ({:.2f} MB)",
                          formatBuffer_, bucketCount_, numSlots_, totalBuckets,
                          sizeof(PercentileTracker), slotStatsArraySize, 
                          totalBucketArraysSize, totalObjectSize, totalMB);
                break;
            case 1: // WARN
                NVLOGW_FMT(PT_LOGGING_TAG, "PercentileTracker {}: config={} buckets x {} slots, total_buckets={}, object={} bytes, slot_array={} bytes, bucket_arrays={} bytes, total={} bytes ({:.2f} MB)",
                          formatBuffer_, bucketCount_, numSlots_, totalBuckets,
                          sizeof(PercentileTracker), slotStatsArraySize, 
                          totalBucketArraysSize, totalObjectSize, totalMB);
                break;
            case 2: // INFO (default)
                NVLOGI_FMT(PT_LOGGING_TAG, "PercentileTracker {}: config={} buckets x {} slots, total_buckets={}, object={} bytes, slot_array={} bytes, bucket_arrays={} bytes, total={} bytes ({:.2f} MB)",
                          formatBuffer_, bucketCount_, numSlots_, totalBuckets,
                          sizeof(PercentileTracker), slotStatsArraySize, 
                          totalBucketArraysSize, totalObjectSize, totalMB);
                break;
            case 3: // DEBUG
                NVLOGD_FMT(PT_LOGGING_TAG, "PercentileTracker {}: config={} buckets x {} slots, total_buckets={}, object={} bytes, slot_array={} bytes, bucket_arrays={} bytes, total={} bytes ({:.2f} MB)",
                          formatBuffer_, bucketCount_, numSlots_, totalBuckets,
                          sizeof(PercentileTracker), slotStatsArraySize, 
                          totalBucketArraysSize, totalObjectSize, totalMB);
                break;
            default: // Use INFO for any other value
                NVLOGI_FMT(PT_LOGGING_TAG, "PercentileTracker {}: config={} buckets x {} slots, total_buckets={}, object={} bytes, slot_array={} bytes, bucket_arrays={} bytes, total={} bytes ({:.2f} MB)",
                          formatBuffer_, bucketCount_, numSlots_, totalBuckets,
                          sizeof(PercentileTracker), slotStatsArraySize, 
                          totalBucketArraysSize, totalObjectSize, totalMB);
                break;
        }
    }

private:
    // Internal bucket structure
    struct HistogramBucket {
        int64_t count;
    };
    
    // Per-slot statistics structure
    struct SlotStats {
        HistogramBucket* buckets;
        int64_t totalCount;
        int64_t minValue;
        int64_t maxValue;
    };
    
    // Histogram configuration
    int64_t lowestTrackableValue_;
    int64_t highestTrackableValue_;
    int64_t bucketSize_;
    int bucketCount_;
    uint32_t numSlots_;
    
    // Data storage - one set of statistics per slot
    SlotStats* slotStats_;
    
    // Pre-allocated buffer for string formatting
    mutable char formatBuffer_[PT_MAX_PREFIX_LENGTH];
    
    // Helper methods
    int getBucketIndex(int64_t value) const;
    int64_t getBucketValueAtIndex(int index) const;
    
    // Calculate the midpoint value for a given bucket index
    inline int64_t getBucketMidpoint(const int bucketIndex) const {
        return lowestTrackableValue_ + (bucketIndex * bucketSize_) + (bucketSize_ / 2);
    }
    
    // Combined stats across all slots
    mutable int64_t combinedMinValue_;
    mutable int64_t combinedMaxValue_;
    mutable int64_t combinedTotalCount_;
    mutable bool statsCached_;
    
    // Recalculate combined stats across all slots
    void updateCombinedStats() const;
    
    // Internal logging helper that both public logging methods will use
    // Use template for tag to ensure it's a compile-time constant
    template<int PT_LOGGING_TAG>
    void logStatistics(uint32_t logLevel, const char* prefix, 
                      const Statistics& stats, int slotIndex = -1) const {
        // Format the statistics in a truly fixed-width tabular format
        char slotStr[16];
        char statsBuffer[256];
        char finalBuffer[512]; // For the complete formatted message
        
        // Format slot information
        if (numSlots_ > 1) {
            if (slotIndex >= 0) {
                snprintf(slotStr, sizeof(slotStr), "slot=%02d", slotIndex);
            } else {
                snprintf(slotStr, sizeof(slotStr), "slot=all");
            }
        } else {
            slotStr[0] = '\0';  // No slot info for single slot mode
        }
        
        // Use a fixed-width buffer with exactly positioned fields and separator bars
        char countStr[16], minStr[16], maxStr[16], meanStr[16], stdStr[16];
        char p50Str[16], p90Str[16], p99Str[16], p999Str[16], p9999Str[16];
        
        // Convert all values to strings with the same format
        snprintf(countStr, sizeof(countStr), "%" PRId64, stats.totalCount);
        snprintf(minStr, sizeof(minStr), "%" PRId64, stats.min);
        snprintf(maxStr, sizeof(maxStr), "%" PRId64, stats.max);
        snprintf(meanStr, sizeof(meanStr), "%" PRId64, stats.mean);
        snprintf(stdStr, sizeof(stdStr), "%" PRId64, stats.stdDev);
        snprintf(p50Str, sizeof(p50Str), "%" PRId64, stats.p50);
        snprintf(p90Str, sizeof(p90Str), "%" PRId64, stats.p90);
        snprintf(p99Str, sizeof(p99Str), "%" PRId64, stats.p99);
        snprintf(p999Str, sizeof(p999Str), "%" PRId64, stats.p999);
        snprintf(p9999Str, sizeof(p9999Str), "%" PRId64, stats.p9999);
        
        // Create the stats buffer with slot info if available
        if (slotStr[0] != '\0') {
            snprintf(statsBuffer, sizeof(statsBuffer),
                    "%-8s count=%-15s | min=%-8s max=%-8s mean=%-8s std=%-8s | p50=%-8s p90=%-8s p99=%-8s p99.9=%-8s p99.99=%-8s",
                    slotStr, countStr, minStr, maxStr, meanStr, stdStr,
                    p50Str, p90Str, p99Str, p999Str, p9999Str);
        } else {
            // Single slot mode - no slot info
            snprintf(statsBuffer, sizeof(statsBuffer),
                    "count=%-15s | min=%-8s max=%-8s mean=%-8s std=%-8s | p50=%-8s p90=%-8s p99=%-8s p99.9=%-8s p99.99=%-8s",
                    countStr, minStr, maxStr, meanStr, stdStr,
                    p50Str, p90Str, p99Str, p999Str, p9999Str);
        }
        
        // Now format the final output with prefix and pipe separator
        // No width constraint on prefix - let it be any length
        snprintf(finalBuffer, sizeof(finalBuffer), "%s | %s", prefix, statsBuffer);
        
        // Log at the specified level using compile-time tag
        switch (logLevel) {
            case 0: // ERROR
                NVLOGE_FMT(PT_LOGGING_TAG, AERIAL_CUPHYDRV_API_EVENT, "{}", finalBuffer);
                break;
            case 1: // WARN
                NVLOGW_FMT(PT_LOGGING_TAG, "{}", finalBuffer);
                break;
            case 2: // INFO (default)
                NVLOGI_FMT(PT_LOGGING_TAG, "{}", finalBuffer);
                break;
            case 3: // DEBUG
                NVLOGD_FMT(PT_LOGGING_TAG, "{}", finalBuffer);
                break;
            default: // Use INFO for any other value
                NVLOGI_FMT(PT_LOGGING_TAG, "{}", finalBuffer);
                break;
        }
    }
};

} // namespace perf_metrics

#endif // PERF_METRICS_PERCENTILE_TRACKER_HPP

