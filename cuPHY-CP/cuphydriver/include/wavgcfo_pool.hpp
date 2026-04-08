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

#ifndef WAVGCFO_POOL_H
#define WAVGCFO_POOL_H

#include <iostream>
#include <typeinfo>
#include <atomic>
#include <memory>
#include <array>
#include <unordered_map>
#include <mutex>
#include "gpudevice.hpp"
#include "time.hpp"
#include "fh.hpp"
#include "locks.hpp"
#include "cuphydriver_api.hpp"
#include "time.hpp"

// #define DEBUG_WAVGCFO_BUFFER

static constexpr uint8_t WAVGCFO_BUFFER_SIZE = 2 * sizeof(float);
static constexpr uint16_t WAVGCFO_POOL_SIZE = 1024; // 2KB
static constexpr uint32_t WAVGCFO_CLEANUP_HYSTERESIS = 5;
static constexpr uint32_t WAVGCFO_DEPLETION_THRESHOLD_PERCENT = 30;

/////////////////////////////////////////////////////////////////
//// Weighted Average CFO buffer
/////////////////////////////////////////////////////////////////
class WAvgCfoBuffer
{

protected:
    t_ns lastUsed_t;
    // Useful to release the buffer into the right pool
    size_t size;
    std::unique_ptr<dev_buf> buffer;
    // Useful to retrieve the right bucket where the buffer is
    uint32_t rnti;
    uint64_t cell_id;
    int cellDynIdx;
    uint16_t sfn;
    uint16_t slot;
#ifdef DEBUG_WAVGCFO_BUFFER
    std::atomic<int> ref_count;
#endif

public:
    /**
     * Constructor for Weighted Average CFO Buffer
     * 
     * @param[in] bdev Device buffer pointer
     * @param[in] _size Size of the buffer
     */
    WAvgCfoBuffer(dev_buf * bdev, size_t _size)
    {
        buffer.reset(bdev);
        buffer->clear();
        size = _size;
        rnti = 0;
        cell_id = 0;
        cellDynIdx = -1;
        sfn = 0;
        slot = 0;
        lastUsed_t = {};
#ifdef DEBUG_WAVGCFO_BUFFER
        ref_count.store(0);
#endif
    }

    /**
     * Get buffer address
     * 
     * @return Pointer to buffer address
     */
    [[nodiscard]] float* getAddr() const {
        return reinterpret_cast<float*>(buffer->addr());
    }

    /**
     * Get timestamp of last use
     * 
     * @return Timestamp in nanoseconds
     */
    [[nodiscard]] t_ns getTimestampLastUsed() const {
        return lastUsed_t;
    }

    /**
     * Set timestamp to current time
     */
    void setTimestampLastUsed() {
        lastUsed_t = Time::nowNs();
    }

    /**
     * Set cell dynamic index
     * 
     * @param[in] idx Cell dynamic index
     */
    void setCellDynIdx(int idx)
    {
        cellDynIdx = idx;
    }

    /**
     * Get cell dynamic index
     * 
     * @return Cell dynamic index
     */
    [[nodiscard]] int getCellDynIdx() const
    {
        return cellDynIdx;
    }

    /**
     * Get buffer size
     * 
     * @return Buffer size in bytes
     */
    [[nodiscard]] size_t getSize() const {
        return size;
    }

    /**
     * Initialize buffer with RNTI and cell ID
     * 
     * @param[in] _rnti Radio Network Temporary Identifier
     * @param[in] _cell_id Cell identifier
     * 
     * @return 0 on success
     */
    [[nodiscard]] int init(uint32_t _rnti, uint64_t _cell_id) {
        rnti = _rnti;
        lastUsed_t = Time::nowNs();
        cell_id = _cell_id;
        return 0;
    }

    /**
     * Get RNTI
     * 
     * @return Radio Network Temporary Identifier
     */
    [[nodiscard]] uint32_t getRnti() const {
        return rnti;
    }

    /**
     * Get Cell ID
     * 
     * @return Cell identifier
     */
    [[nodiscard]] uint64_t getCellId() const {
        return cell_id;
    }

    /**
     * Set SFN and slot
     * 
     * @param[in] _sfn System Frame Number
     * @param[in] _slot Slot number
     */
    void setSfnSlot(uint16_t _sfn, uint16_t _slot) {
        sfn = _sfn;
        slot = _slot;
    }

    /**
     * Get System Frame Number
     * 
     * @return System Frame Number
     */
    [[nodiscard]] uint16_t getSfn() const {
        return sfn;
    }

    /**
     * Get slot number
     * 
     * @return Slot number
     */
    [[nodiscard]] uint16_t getSlot() const {
        return slot;
    }

#ifdef DEBUG_WAVGCFO_BUFFER
    /**
     * Increment reference count (debug only)
     * 
     * @return Previous reference count value
     */
    [[nodiscard]] int refAdd() {
        return ref_count.fetch_add(1);
    }

    /**
     * Decrement reference count (debug only)
     * 
     * @return Previous reference count value
     */
    [[nodiscard]] int refSub() {
        return ref_count.fetch_sub(1);
    }
#endif

    /**
     * Clear buffer state
     */
    void clear() {
        lastUsed_t = Time::zeroNs();
        rnti = 0;
        cellDynIdx = -1;
    }

};

/////////////////////////////////////////////////////////////////
//// Cache for Weighted Average CFO buffers
/////////////////////////////////////////////////////////////////

/**
 * Cache structure to manage WAvgCfoBuffer pointers by RNTI and Cell ID
 * 
 * The cache uses a 32-bit composite key where:
 * - Bits 0-15: cell_id (16 bits)
 * - Bits 16-31: rnti (16 bits)
 */
struct WAvgCfoCache {
public:
    WAvgCfoCache() = default;
    ~WAvgCfoCache() = default;

    // Non-copyable, non-movable
    WAvgCfoCache(const WAvgCfoCache&) = delete;
    WAvgCfoCache& operator=(const WAvgCfoCache&) = delete;
    WAvgCfoCache(WAvgCfoCache&&) = delete;
    WAvgCfoCache& operator=(WAvgCfoCache&&) = delete;

    /**
     * Find a buffer in the cache
     * 
     * @param[in] rnti Radio Network Temporary Identifier
     * @param[in] cell_id Cell identifier
     * 
     * @return Pointer to WAvgCfoBuffer if found, nullptr otherwise
     */
    [[nodiscard]] WAvgCfoBuffer* find(uint16_t rnti, uint16_t cell_id);

    /**
     * Allocate (insert) a buffer into the cache
     * 
     * @param[in] rnti Radio Network Temporary Identifier
     * @param[in] cell_id Cell identifier
     * @param[in] buffer Pointer to WAvgCfoBuffer to cache
     * 
     * @return true if inserted successfully, false if key already exists or buffer is null
     */
    [[nodiscard]] bool allocate(uint16_t rnti, uint16_t cell_id, WAvgCfoBuffer* buffer);

    /**
     * Deallocate (remove) a buffer from the cache
     * 
     * @param[in] rnti Radio Network Temporary Identifier
     * @param[in] cell_id Cell identifier
     * 
     * @return Pointer to the removed buffer, or nullptr if not found
     */
    [[nodiscard]] WAvgCfoBuffer* deallocate(uint16_t rnti, uint16_t cell_id);

    /**
     * Get the number of entries in the cache
     * 
     * @return Number of cached buffers
     */
    [[nodiscard]] std::size_t size() const;

    /**
     * Clear all entries from the cache
     */
    void clear();

    /**
     * Check if cache is empty
     * 
     * @return true if cache is empty, false otherwise
     */
    [[nodiscard]] bool empty() const;

    /**
     * Remove unused buffers from cache and invoke callback on each removed buffer
     * 
     * This method iterates through all cached buffers, checks if they are in use,
     * and removes those that are not. For each removed buffer, it calls the provided
     * callback function.
     * 
     * @param[in] callback Function to call for each removed buffer pointer
     * 
     * @return Number of buffers removed from cache
     */
    template<typename Callback>
    [[nodiscard]] std::size_t removeUnused(Callback&& callback) {
        std::lock_guard<Mutex> lock(mutex_);
        std::size_t count = 0;
        
        for (auto it = cache_.begin(); it != cache_.end(); ) {
            WAvgCfoBuffer* buffer = it->second;
            if (buffer != nullptr) {
                // Check if buffer is actually unused by comparing timestamps
                // or add a proper "in_use" flag to WAvgCfoBuffer
                t_ns current_time = Time::nowNs();
                t_ns buffer_last_used = buffer->getTimestampLastUsed();
                
                // Consider buffer unused if not accessed recently (e.g., > 1 second)
                if (Time::getDifference(current_time, buffer_last_used).count() > 1000000000ULL) {
                    callback(buffer);
                    it = cache_.erase(it);
                    count++;
                } else {
                    ++it;
                }
            } else {
                ++it;
            }
        }
        return count;
    }

private:
    /**
     * Create a composite 32-bit key from RNTI and Cell ID
     * 
     * Key format:
     * - Bits 0-15: cell_id (16 bits)
     * - Bits 16-31: rnti (16 bits)
     * 
     * @param[in] rnti Radio Network Temporary Identifier (16 bits)
     * @param[in] cell_id Cell identifier (16 bits)
     * 
     * @return 32-bit composite key
     */
    [[nodiscard]] static constexpr uint32_t makeKey(const uint16_t rnti, const uint16_t cell_id) noexcept {
        return static_cast<uint32_t>(cell_id) | (static_cast<uint32_t>(rnti) << 16);
    }

    /**
     * Extract Cell ID from composite key
     * 
     * @param[in] key Composite key
     * 
     * @return Cell ID (bits 0-15)
     */
    [[nodiscard]] static constexpr uint16_t extractCellId(const uint32_t key) noexcept {
        return static_cast<uint16_t>(key & 0xFFFF);
    }

    /**
     * Extract RNTI from composite key
     * 
     * @param[in] key Composite key
     * 
     * @return RNTI (bits 16-31)
     */
    [[nodiscard]] static constexpr uint16_t extractRnti(const uint32_t key) noexcept {
        return static_cast<uint16_t>((key >> 16) & 0xFFFF);
    }

    std::unordered_map<uint32_t, WAvgCfoBuffer*> cache_;
    mutable Mutex mutex_;
};

/////////////////////////////////////////////////////////////////
//// Pool of free Weighted Average CFO buffers with fixed size
/////////////////////////////////////////////////////////////////

class WAvgCfoPool {

public:
    /**
     * Constructor for Weighted Average CFO Pool
     * 
     * @param[in] _gDev GPU device pointer
     * @param[in] _fhi_handle Fronthaul handle
     */
    WAvgCfoPool(GpuDevice* _gDev, FronthaulHandle _fhi_handle);
    
    /**
     * Destructor
     */
    ~WAvgCfoPool();

    /**
     * Get buffer size
     * 
     * @return Buffer size in bytes
     */
    [[nodiscard]] size_t getSize() const;

    /**
     * Count number of elements currently in use
     * 
     * @return Number of elements in use
     */
    [[nodiscard]] size_t countElements() const;

    /**
     * Get total GPU memory size allocated by this pool for memory footprint
     *
     * @return Total GPU memory size allocated by this pool in bytes
     */
    [[nodiscard]] size_t getGpuMemSize() const;

    /**
     * Pull a free buffer from the pool
     * 
     * @return Pointer to WAvgCfoBuffer or nullptr if pool is empty
     */
    [[nodiscard]] WAvgCfoBuffer * pullBuffer();
    
    /**
     * Re-insert a buffer into the pool
     * 
     * @param[in] buf Buffer to push back into pool
     * 
     * @return 0 on success, negative value on error, EINVAL if buf is null
     */
    [[nodiscard]] int pushBuffer(WAvgCfoBuffer * buf);

protected:
    // List of allocated WAvgCfo buffers (fixed size array)
    std::array<std::unique_ptr<WAvgCfoBuffer>, WAVGCFO_POOL_SIZE>    wavgcfo_list;
    // (DPDK) lockless ring to maintain the pool
    RingBufferHandle                               wavgcfo_ring;
    RingBufferInfo                                 ring_info;
    int                                            totRingElems;
    size_t                                         gpuMemSize;
    GpuDevice*                                     gDev;
};

/////////////////////////////////////////////////////////////////
//// Manager for Weighted Average CFO Pool and Cache
/////////////////////////////////////////////////////////////////

/**
 * Pool Manager that integrates WAvgCfoPool and WAvgCfoCache
 * 
 * Provides high-level buffer allocation with automatic caching.
 * Buffers are cached by (RNTI, Cell ID) for efficient reuse.
 */
struct WAvgCfoPoolManager {
public:
    /**
     * Constructor
     * 
     * @param[in] _pdh Phy driver handle
     * @param[in] _gDev GPU device pointer
     * @param[in] _fhi_handle Fronthaul handle
     */
    WAvgCfoPoolManager(phydriver_handle _pdh, GpuDevice* _gDev, FronthaulHandle _fhi_handle);
    
    /**
     * Destructor
     */
    ~WAvgCfoPoolManager();

    // Non-copyable, non-movable
    WAvgCfoPoolManager(const WAvgCfoPoolManager&) = delete;
    WAvgCfoPoolManager& operator=(const WAvgCfoPoolManager&) = delete;
    WAvgCfoPoolManager(WAvgCfoPoolManager&&) = delete;
    WAvgCfoPoolManager& operator=(WAvgCfoPoolManager&&) = delete;

    /**
     * Allocate a buffer for given RNTI and Cell ID
     * 
     * First checks the cache. If found, returns cached buffer.
     * If not found, pulls a buffer from the pool, caches it, and returns it.
     * 
     * @param[in] rnti Radio Network Temporary Identifier
     * @param[in] cell_id Cell identifier
     * 
     * @return Pointer to WAvgCfoBuffer or nullptr if allocation fails
     */
    [[nodiscard]] WAvgCfoBuffer* allocate(uint16_t rnti, uint16_t cell_id);

    /**
     * Deallocate a buffer for given RNTI and Cell ID
     * 
     * Removes buffer from cache and returns it to the pool.
     * 
     * @param[in] rnti Radio Network Temporary Identifier
     * @param[in] cell_id Cell identifier
     * 
     * @return 0 on success, negative value on error
     */
    [[nodiscard]] int deallocate(uint16_t rnti, uint16_t cell_id);

    /**
     * Get pool statistics
     * 
     * @return Number of buffers available in pool
     */
    [[nodiscard]] std::size_t getPoolAvailable() const;

    /**
     * Get cache statistics
     * 
     * @return Number of buffers currently cached
     */
    [[nodiscard]] std::size_t getCacheSize() const;

    /**
     * Clear all cached buffers and return them to pool
     * 
     * @return Number of buffers returned to pool
     */
    [[nodiscard]] std::size_t clearCache();

    /**
     * Check pool depletion and trigger cleanup if necessary
     * 
     * Monitors pool availability and clears cache when pool is running low
     * to reclaim unused buffers. Uses hysteresis to avoid excessive cleanup.
     */
    void checkPoolDepletion();

    MemFoot mf;

private:
    phydriver_handle pdh;
    std::unique_ptr<WAvgCfoPool> pool_;
    WAvgCfoCache cache_;
    uint32_t depletion_hysteresis_{};
};

#endif


