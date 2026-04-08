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

#include "perf_metrics/percentile_tracker.hpp"

#include <cstdint>  // This ensures uint16_t, uint32_t, etc. are defined
#include <limits>   // For std::numeric_limits
#include <vector>   // For std::vector
#include <cmath>    // For std::ceil, std::sqrt
#include <algorithm> // For std::min, std::max
#include <time.h>   // For timespec

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 48) // "DRV.PERF_METRICS"

namespace perf_metrics {

PercentileTracker::PercentileTracker(int64_t lowestTrackableValue, 
                                    int64_t highestTrackableValue, 
                                    int64_t bucketSize,
                                    int32_t numSlots)
    : lowestTrackableValue_(lowestTrackableValue),
      highestTrackableValue_(highestTrackableValue),
      bucketSize_(bucketSize),
      numSlots_(numSlots <= 0 ? 1 : static_cast<uint32_t>(numSlots)),
      slotStats_(nullptr),
      combinedMinValue_(std::numeric_limits<int64_t>::max()),
      combinedMaxValue_(0),
      combinedTotalCount_(0),
      statsCached_(false) {
    
    // Log constructor parameters
    NVLOGI_FMT(TAG, "PercentileTracker::PercentileTracker :: Creating with request values: low={}, high={}, bucketSize={}, numSlots={}",
               lowestTrackableValue, highestTrackableValue, bucketSize, numSlots);
    
    if (bucketSize <= 0) {
        NVLOGW_FMT(TAG, "PercentileTracker::PercentileTracker :: bucketSize must be positive");
        bucketSize_ = 1000; // Default to 100ns if invalid
    }
    
    if (lowestTrackableValue < 0) {
        NVLOGW_FMT(TAG, "PercentileTracker::PercentileTracker :: lowestTrackableValue must be non-negative");
        lowestTrackableValue_ = 0; // Default to 0 if invalid
    }
    
    if (highestTrackableValue <= lowestTrackableValue_) {
        NVLOGW_FMT(TAG, "PercentileTracker::PercentileTracker :: highestTrackableValue must be greater than lowestTrackableValue");
        highestTrackableValue_ = lowestTrackableValue_ + 500000; // Default to 500μs range if invalid
    }
    
    // Calculate number of buckets needed
    int64_t range = highestTrackableValue_ - lowestTrackableValue_;
    
    // Safety check - limit maximum number of buckets to prevent overflow and memory issues
    // 100K is a reasonable upper limit that won't consume excessive memory
    const int64_t MAX_BUCKETS = 100000;
    int64_t calculatedBuckets = (range + bucketSize_ - 1) / bucketSize_; // Ceiling division
    
    if (calculatedBuckets > MAX_BUCKETS) {
        NVLOGW_FMT(TAG, "PercentileTracker::PercentileTracker :: Calculated bucket count {} exceeds maximum limit. Adjusting bucket size to maintain full range.", 
                calculatedBuckets);
        bucketCount_ = static_cast<int>(MAX_BUCKETS);
        // Adjust bucket size to cover the full requested range with MAX_BUCKETS
        bucketSize_ = (range + bucketCount_ - 1) / bucketCount_;  // Ceiling division to ensure full coverage
        // Update highestTrackableValue to reflect actual coverage with new bucket size
        highestTrackableValue_ = lowestTrackableValue_ + (bucketCount_ * bucketSize_);
        NVLOGW_FMT(TAG, "PercentileTracker::PercentileTracker :: Increased bucket size from {} to {} to cover range [{}, {})",
                bucketSize, bucketSize_, lowestTrackableValue_, highestTrackableValue_);
    } else {
        bucketCount_ = static_cast<int>(calculatedBuckets);
    }
    
    NVLOGI_FMT(TAG, "PercentileTracker::PercentileTracker :: Final values: low={}, high={}, bucketSize={}, bucketCount={}, numSlots={}",
               lowestTrackableValue_, highestTrackableValue_, bucketSize_, bucketCount_, numSlots_);
    
    // Allocate and initialize slot statistics
    slotStats_ = new SlotStats[numSlots_];
    for (uint32_t i = 0; i < numSlots_; i++) {
        slotStats_[i].buckets = new HistogramBucket[bucketCount_];
        slotStats_[i].totalCount = 0;
        slotStats_[i].minValue = std::numeric_limits<int64_t>::max();
        slotStats_[i].maxValue = 0;
        
        // Initialize all buckets for this slot
        for (int j = 0; j < bucketCount_; j++) {
            slotStats_[i].buckets[j].count = 0;
        }
    }
    
    // Log memory footprint after initialization
    logMemoryFootprint<TAG>();
}

PercentileTracker::~PercentileTracker() {
    if (slotStats_) {
        for (uint32_t i = 0; i < numSlots_; i++) {
            delete[] slotStats_[i].buckets;
        }
        delete[] slotStats_;
    }
}

// Instance method for calculating slot index
uint32_t PercentileTracker::getSlotIndex(uint16_t sfn, uint16_t slot) const {
    if (numSlots_ == 1) {
        return 0;
    }
    
    // Keep the existing sfn/slot to slotIndex logic, but adjust for any numSlots value
    
    // Determine how many slots we have per frame
    uint32_t slotsPerFrame = 20;  // Default from 5G NR standard (for 30kHz SCS)
    
    // We adjust the calculation to work with any numSlots value
    // For each sfn, we allocate slots 0-19, then move to the next sfn portion
    uint32_t sfnContribution = (sfn % ((numSlots_ + slotsPerFrame - 1) / slotsPerFrame)) * slotsPerFrame;
    uint32_t index = slot + sfnContribution;
    
    // Double-check we're within bounds
    if (index >= numSlots_) {
        NVLOGW_FMT(TAG, "PercentileTracker::getSlotIndex :: Invalid slot index calculated: {} for SFN {}, slot {}. Using modulo.",
                index, sfn, slot);
        index %= numSlots_;
    }
    
    return index;
}

// Original method - takes sfn and slot
void PercentileTracker::addValue(int64_t value, uint16_t sfn, uint16_t slot) {
    // Convert sfn/slot to a slot index and delegate
    uint32_t slotIndex = getSlotIndex(sfn, slot);
    addValue(value, static_cast<int>(slotIndex));
}

// Overloaded method - only takes value (for single slot mode)
void PercentileTracker::addValue(int64_t value) {
    // Always use slot 0 in this case
    addValue(value, 0);
}

// Core implementation - takes value and direct slot index
void PercentileTracker::addValue(int64_t value, int slotIndex) {
    // Validate the slot index is within bounds
    if (slotIndex < 0 || static_cast<uint32_t>(slotIndex) >= numSlots_) {
        NVLOGW_FMT(TAG, "PercentileTracker::addValue :: slotIndex {} out of bounds (0-{}). Dropping value.", 
                  slotIndex, numSlots_ - 1);
        return;
    }
    
    uint32_t safeSlotIndex = static_cast<uint32_t>(slotIndex);
    
    // Clamp value to valid range
    if (value < lowestTrackableValue_) {
        NVLOGW_FMT(TAG, "PercentileTracker::addValue :: Value {} below minimum, clamping to {}", value, lowestTrackableValue_);
        value = lowestTrackableValue_;
    } else if (value >= highestTrackableValue_) {
        // Instead of resizing, clamp to the highest available bucket
        NVLOGW_FMT(TAG, "PercentileTracker::addValue :: Value {} above maximum, clamping to {}", value, highestTrackableValue_ - 1);
        value = highestTrackableValue_ - 1;
    }
    
    // Update min/max values for this slot
    slotStats_[safeSlotIndex].minValue = std::min(slotStats_[safeSlotIndex].minValue, value);
    slotStats_[safeSlotIndex].maxValue = std::max(slotStats_[safeSlotIndex].maxValue, value);
    
    // Increment the appropriate bucket
    int bucketIndex = getBucketIndex(value);
    slotStats_[safeSlotIndex].buckets[bucketIndex].count++;
    slotStats_[safeSlotIndex].totalCount++;
    
    // Mark combined stats as dirty
    statsCached_ = false;
}

int64_t PercentileTracker::getPercentile(double percentile, int slotIndex) const {
    if (percentile < 0.0 || percentile > 100.0) {
        NVLOGW_FMT(TAG, "PercentileTracker::getPercentile :: Percentile must be between 0.0 and 100.0");
        percentile = std::max(0.0, std::min(100.0, percentile));
    }
    
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getPercentile :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    // If slotIndex is negative, calculate across all slots
    if (slotIndex < 0) {
        // In single slot mode, -1 is equivalent to 0
        if (numSlots_ == 1) {
            slotIndex = 0;
        } else {
            // Collect all counts from all slots
            std::vector<int64_t> valueCounts;
            valueCounts.resize(bucketCount_);
            
            int64_t totalCount = 0;
            
            // Aggregate bucket counts from all slots
            for (uint32_t s = 0; s < numSlots_; s++) {
                for (int b = 0; b < bucketCount_; b++) {
                    valueCounts[b] += slotStats_[s].buckets[b].count;
                }
                totalCount += slotStats_[s].totalCount;
            }
            
            if (totalCount == 0) {
                return 0;
            }
            
            // Calculate the count at or below the requested percentile
            int64_t countAtPercentile = static_cast<int64_t>(std::ceil(percentile / 100.0 * totalCount));
            
            // Find the bucket that contains the percentile
            int64_t runningCount = 0;
            for (int i = 0; i < bucketCount_; i++) {
                runningCount += valueCounts[i];
                if (runningCount >= countAtPercentile) {
                    // Return the middle value of the bucket
                    return getBucketMidpoint(i);
                }
            }
            
            // Fallback - return the highest bucket midpoint
            return getBucketMidpoint(bucketCount_ - 1);
        }
    }
    
    // Calculate for a specific slot
    if (slotStats_[slotIndex].totalCount == 0) {
        return 0;
    }
    
    // Calculate the count at or below the requested percentile
    int64_t countAtPercentile = static_cast<int64_t>(std::ceil(percentile / 100.0 * slotStats_[slotIndex].totalCount));
    
    // Find the bucket that contains the percentile
    int64_t runningCount = 0;
    for (int i = 0; i < bucketCount_; i++) {
        runningCount += slotStats_[slotIndex].buckets[i].count;
        if (runningCount >= countAtPercentile) {
            // Return the middle value of the bucket
            return getBucketMidpoint(i);
        }
    }
    
    // Fallback to max value if something went wrong
    return slotStats_[slotIndex].maxValue;
}

void PercentileTracker::reset(int slotIndex) {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::reset :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    // If slotIndex is negative, reset all slots
    if (slotIndex < 0) {
        for (uint32_t s = 0; s < numSlots_; s++) {
            for (int i = 0; i < bucketCount_; i++) {
                slotStats_[s].buckets[i].count = 0;
            }
            slotStats_[s].totalCount = 0;
            slotStats_[s].minValue = std::numeric_limits<int64_t>::max();
            slotStats_[s].maxValue = 0;
        }
    } else {
        // Reset specific slot
        for (int i = 0; i < bucketCount_; i++) {
            slotStats_[slotIndex].buckets[i].count = 0;
        }
        slotStats_[slotIndex].totalCount = 0;
        slotStats_[slotIndex].minValue = std::numeric_limits<int64_t>::max();
        slotStats_[slotIndex].maxValue = 0;
    }
    
    // Reset combined stats
    combinedMinValue_ = std::numeric_limits<int64_t>::max();
    combinedMaxValue_ = 0;
    combinedTotalCount_ = 0;
    statsCached_ = false;
}

void PercentileTracker::updateCombinedStats() const {
    if (statsCached_) {
        return;
    }
    
    combinedMinValue_ = std::numeric_limits<int64_t>::max();
    combinedMaxValue_ = 0;
    combinedTotalCount_ = 0;
    
    // If there's only one slot, use its values directly
    if (numSlots_ == 1) {
        if (slotStats_[0].totalCount > 0) {
            combinedMinValue_ = slotStats_[0].minValue;
            combinedMaxValue_ = slotStats_[0].maxValue;
            combinedTotalCount_ = slotStats_[0].totalCount;
        }
    } else {
        // Combine statistics from all slots
        for (uint32_t s = 0; s < numSlots_; s++) {
            if (slotStats_[s].totalCount > 0) {
                combinedMinValue_ = std::min(combinedMinValue_, slotStats_[s].minValue);
                combinedMaxValue_ = std::max(combinedMaxValue_, slotStats_[s].maxValue);
                combinedTotalCount_ += slotStats_[s].totalCount;
            }
        }
    }
    
    if (combinedTotalCount_ == 0) {
        combinedMinValue_ = 0;
        combinedMaxValue_ = 0;
    }
    
    statsCached_ = true;
}

int64_t PercentileTracker::getMinValue(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getMinValue :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    if (slotIndex < 0) {
        updateCombinedStats();
        return combinedMinValue_;
    } else {
        return (slotStats_[slotIndex].totalCount > 0) ? slotStats_[slotIndex].minValue : 0;
    }
}

int64_t PercentileTracker::getMaxValue(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getMaxValue :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    if (slotIndex < 0) {
        updateCombinedStats();
        return combinedMaxValue_;
    } else {
        return (slotStats_[slotIndex].totalCount > 0) ? slotStats_[slotIndex].maxValue : 0;
    }
}

int64_t PercentileTracker::getMean(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getMean :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    // Handle a specific slot or single slot mode with negative index
    if (slotIndex >= 0 || numSlots_ == 1) {
        // For single slot mode, normalize negative index to 0
        if (numSlots_ == 1 && slotIndex < 0) {
            slotIndex = 0;
        }
        
        // Calculate mean for a specific slot
        if (slotStats_[slotIndex].totalCount == 0) {
            return 0;
        }
        
        int64_t sum = 0;
        for (int i = 0; i < bucketCount_; i++) {
            if (slotStats_[slotIndex].buckets[i].count > 0) {
                // Use middle value of bucket for approximation
                const int64_t midpoint = getBucketMidpoint(i);
                sum += midpoint * slotStats_[slotIndex].buckets[i].count;
            }
        }
        
        return sum / slotStats_[slotIndex].totalCount;
    } else {
        // Calculate mean across all slots in multi-slot mode
        int64_t sum = 0;
        int64_t totalCount = 0;
        
        for (uint32_t s = 0; s < numSlots_; s++) {
            if (slotStats_[s].totalCount > 0) {
                for (int i = 0; i < bucketCount_; i++) {
                    if (slotStats_[s].buckets[i].count > 0) {
                        // Use middle value of bucket for approximation
                        const int64_t midpoint = getBucketMidpoint(i);
                        sum += midpoint * slotStats_[s].buckets[i].count;
                    }
                }
                totalCount += slotStats_[s].totalCount;
            }
        }
        
        return (totalCount > 0) ? sum / totalCount : 0;
    }
}

int64_t PercentileTracker::getStdDeviation(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getStdDeviation :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    // For single slot mode, normalize negative index to 0
    if (numSlots_ == 1 && slotIndex < 0) {
        slotIndex = 0;
    }
    
    // For simplicity, this implementation only handles a specific slot
    if (slotIndex >= 0) {
        if (slotStats_[slotIndex].totalCount <= 1) {
            return 0;
        }
        
        int64_t mean = getMean(slotIndex);
        double variance = 0.0;
        
        for (int i = 0; i < bucketCount_; i++) {
            if (slotStats_[slotIndex].buckets[i].count > 0) {
                const int64_t midpoint = getBucketMidpoint(i);
                const double deviation = midpoint - mean;
                variance += (deviation * deviation) * slotStats_[slotIndex].buckets[i].count;
            }
        }
        
        variance /= (slotStats_[slotIndex].totalCount - 1);
        return static_cast<int64_t>(std::sqrt(variance));
    }
    
    // Calculate standard deviation across all slots in multi-slot mode
    int64_t totalCount = 0;
    
    // First, get total count across all slots
    for (uint32_t s = 0; s < numSlots_; s++) {
        totalCount += slotStats_[s].totalCount;
    }
    
    if (totalCount <= 1) {
        return 0;
    }
    
    // Get the mean across all slots
    int64_t mean = getMean(-1);
    double variance = 0.0;
    
    // Calculate variance using aggregated bucket counts from all slots
    for (uint32_t s = 0; s < numSlots_; s++) {
        for (int i = 0; i < bucketCount_; i++) {
            if (slotStats_[s].buckets[i].count > 0) {
                const int64_t midpoint = getBucketMidpoint(i);
                const double deviation = midpoint - mean;
                variance += (deviation * deviation) * slotStats_[s].buckets[i].count;
            }
        }
    }
    
    variance /= (totalCount - 1);
    return static_cast<int64_t>(std::sqrt(variance));
}

int64_t PercentileTracker::getTotalCount(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getTotalCount :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    if (slotIndex < 0) {
        updateCombinedStats();
        return combinedTotalCount_;
    } else {
        return slotStats_[slotIndex].totalCount;
    }
}

int PercentileTracker::getBucketIndex(int64_t value) const {
    int index = static_cast<int>((value - lowestTrackableValue_) / bucketSize_);

    // Defensive bounds check (should never trigger due to clamping in addValue)
    if (index < 0) {
        return 0;
    }
    if (index >= bucketCount_) {
        return bucketCount_ - 1;
    }
    return index;
}

int64_t PercentileTracker::getBucketValueAtIndex(int index) const {
    // Return the middle value of the bucket
    return getBucketMidpoint(index);
}

// Move constructor implementation
PercentileTracker::PercentileTracker(PercentileTracker&& other) noexcept :
    lowestTrackableValue_(other.lowestTrackableValue_),
    highestTrackableValue_(other.highestTrackableValue_),
    bucketSize_(other.bucketSize_),
    bucketCount_(other.bucketCount_),
    numSlots_(other.numSlots_),
    slotStats_(other.slotStats_),
    combinedMinValue_(other.combinedMinValue_),
    combinedMaxValue_(other.combinedMaxValue_),
    combinedTotalCount_(other.combinedTotalCount_),
    statsCached_(other.statsCached_)
{
    // Nullify the source object's pointer to prevent double deletion
    other.slotStats_ = nullptr;
    other.numSlots_ = 0;
    other.bucketCount_ = 0;
    
    NVLOGI_FMT(TAG, "PercentileTracker::PercentileTracker(move) :: Move constructor called. bucketCount_={}, numSlots_={}", 
               bucketCount_, numSlots_);
}

// Move assignment operator implementation
PercentileTracker& PercentileTracker::operator=(PercentileTracker&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (slotStats_) {
            for (uint32_t i = 0; i < numSlots_; i++) {
                delete[] slotStats_[i].buckets;
            }
            delete[] slotStats_;
        }
        
        // Transfer ownership
        lowestTrackableValue_ = other.lowestTrackableValue_;
        highestTrackableValue_ = other.highestTrackableValue_;
        bucketSize_ = other.bucketSize_;
        bucketCount_ = other.bucketCount_;
        numSlots_ = other.numSlots_;
        slotStats_ = other.slotStats_;
        combinedMinValue_ = other.combinedMinValue_;
        combinedMaxValue_ = other.combinedMaxValue_;
        combinedTotalCount_ = other.combinedTotalCount_;
        statsCached_ = other.statsCached_;
        
        // Nullify the source object's pointer
        other.slotStats_ = nullptr;
        other.numSlots_ = 0;
        other.bucketCount_ = 0;
        
        NVLOGI_FMT(TAG, "PercentileTracker::operator= :: Move assignment called. bucketCount_={}, numSlots_={}", 
                  bucketCount_, numSlots_);
    }
    return *this;
}

// Gets all statistics in a bundle
PercentileTracker::Statistics PercentileTracker::getStatistics(int slotIndex) const {
    // Check if slotIndex is out of bounds
    if (slotIndex >= static_cast<int>(numSlots_)) {
        NVLOGW_FMT(TAG, "PercentileTracker::getStatistics :: slotIndex {} out of bounds. Using 0.", slotIndex);
        slotIndex = 0;
    }
    
    Statistics stats;
    
    // Get all statistics
    stats.min = getMinValue(slotIndex);
    stats.max = getMaxValue(slotIndex);
    stats.mean = getMean(slotIndex);
    stats.stdDev = getStdDeviation(slotIndex);
    stats.totalCount = getTotalCount(slotIndex);
    
    // Calculate percentiles
    stats.p50 = getPercentile(50.0, slotIndex);
    stats.p90 = getPercentile(90.0, slotIndex);
    stats.p99 = getPercentile(99.0, slotIndex);
    stats.p999 = getPercentile(99.9, slotIndex);
    stats.p9999 = getPercentile(99.99, slotIndex);
    
    return stats;
}

} // namespace perf_metrics

