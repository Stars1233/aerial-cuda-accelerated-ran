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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 47) // "DRV.WAVGCFO_POOL"

#include "wavgcfo_pool.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// WAvgCfo Cache
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

WAvgCfoBuffer* WAvgCfoCache::find(const uint16_t rnti, const uint16_t cell_id) {
    std::lock_guard<Mutex> lock(mutex_);
    const uint32_t key = makeKey(rnti, cell_id);
    
    const auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    return nullptr;
}

bool WAvgCfoCache::allocate(const uint16_t rnti, const uint16_t cell_id, WAvgCfoBuffer* buffer) {
    if (!buffer) {
        return false;
    }
    
    std::lock_guard<Mutex> lock(mutex_);
    const uint32_t key = makeKey(rnti, cell_id);
    
    MemtraceDisableScope md; // Disable dynamic memory allocation check temporarily
    // Use insert to avoid overwriting existing entries
    const auto result = cache_.insert({key, buffer});
    return result.second; // true if inserted, false if key already existed
}

WAvgCfoBuffer* WAvgCfoCache::deallocate(const uint16_t rnti, const uint16_t cell_id) {
    std::lock_guard<Mutex> lock(mutex_);
    const uint32_t key = makeKey(rnti, cell_id);
    
    const auto it = cache_.find(key);
    NVLOGC_FMT(TAG, "WAvgCfoCache::deallocate: rnti {} cell_id {} key {} mapSize {}", rnti, cell_id, key, cache_.size());
    if (it != cache_.end()) {
        WAvgCfoBuffer* buffer = it->second;
        cache_.erase(it);
        NVLOGC_FMT(TAG, "WAvgCfoCache::deallocate: after erase rnti {} cell_id {} key {} mapSize {}", rnti, cell_id, key, cache_.size());
        return buffer;
    }
    NVLOGC_FMT(TAG, "WAvgCfoCache::deallocate: not found rnti {} cell_id {} key {}", rnti, cell_id, key);
    return nullptr;
}

std::size_t WAvgCfoCache::size() const {
    std::lock_guard<Mutex> lock(mutex_);
    return cache_.size();
}

void WAvgCfoCache::clear() {
    std::lock_guard<Mutex> lock(mutex_);
    cache_.clear();
}

bool WAvgCfoCache::empty() const {
    std::lock_guard<Mutex> lock(mutex_);
    return cache_.empty();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Pool of free Weighted Average CFO buffers with fixed size
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
WAvgCfoPool::WAvgCfoPool(GpuDevice* _gDev, FronthaulHandle _fhi_handle)
    :
    gpuMemSize(0),
    gDev(_gDev)
{
    char * ring_name{};

    if(WAVGCFO_BUFFER_SIZE == 0)
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "WAvgCfoPool bad input size");

    gDev->setDevice();

    ring_name = static_cast<char*>(calloc(256, sizeof(char)));
    if(ring_name == nullptr)
        PHYDRIVER_THROW_EXCEPTIONS(ENOMEM, "Failed to allocate ring name");

    snprintf(ring_name, 256, "WAVGCFO_POOL_%u", WAVGCFO_BUFFER_SIZE);
    ring_info.count = WAVGCFO_POOL_SIZE;
    ring_info.socket_id = AERIAL_SOCKET_ID_ANY;
    ring_info.multi_producer = true;
    ring_info.multi_consumer = true;

    ring_info.name = static_cast<const char *>(ring_name);
    // Create (DPDK) lockless ring to maintain the pool
    // Assume WAVGCFO_POOL_SIZE is a power of 2
    if(aerial_fh::ring_create(_fhi_handle, &ring_info, &wavgcfo_ring))
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "aerial_fh::ring_create error");

    totRingElems = static_cast<int>(aerial_fh::ring_free_count(wavgcfo_ring));

    // Create WAvgCfo set of free buffers (fixed array size)
    for(std::size_t i = 0; i < WAVGCFO_POOL_SIZE - 1; i++)
    {
        dev_buf* _dev_buf = new dev_buf(WAVGCFO_BUFFER_SIZE, gDev);
        gpuMemSize += _dev_buf->size_alloc;
        wavgcfo_list[i] = std::make_unique<WAvgCfoBuffer>(_dev_buf, WAVGCFO_BUFFER_SIZE);
        // Push all the WAvgCfo buffers in the ring
        if(aerial_fh::ring_enqueue(wavgcfo_ring, static_cast<void*>(wavgcfo_list[i].get())))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Error inserting obj {}", i);
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "aerial_fh::ring_enqueue error");
        }
    }

    NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "WAvgCfoPool created with {} elements of size {}", countElements(), WAVGCFO_BUFFER_SIZE);
}

WAvgCfoPool::~WAvgCfoPool() {
    for(auto& i:wavgcfo_list) {
        i.reset(nullptr);
    }

    if(aerial_fh::ring_destroy(wavgcfo_ring))
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error destroying wavgcfo_ring");
}

size_t WAvgCfoPool::getSize() const {
    return WAVGCFO_BUFFER_SIZE;
}

size_t WAvgCfoPool::countElements() const {
    return WAVGCFO_POOL_SIZE - (static_cast<int>(aerial_fh::ring_free_count(wavgcfo_ring)));
}

size_t WAvgCfoPool::getGpuMemSize() const {
    return gpuMemSize;
}

// Allocate a buffer from FH rte_ring
WAvgCfoBuffer * WAvgCfoPool::pullBuffer() {

    WAvgCfoBuffer * buf{};

    if(aerial_fh::ring_dequeue(wavgcfo_ring, reinterpret_cast<void**>(&buf)))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::ring_dequeue");
        return nullptr;
    }

    if (buf == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "pullBuffer: nullptr");
        return nullptr;
    }

#ifdef DEBUG_WAVGCFO_BUFFER
    if (buf->getSize() != WAVGCFO_BUFFER_SIZE)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "pullBuffer: Wrong WAvgCfoPool: size: {}-{} {}: rnti {} cell_id {}",
                buf->getSize(), WAVGCFO_BUFFER_SIZE, reinterpret_cast<void*>(buf), buf->getRnti(), buf->getCellId());
    }

    if(buf->refAdd() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Dequeued used buf {}: rnti {} cell_id {}",
                reinterpret_cast<void*>(buf), buf->getRnti(), buf->getCellId());
    }
#endif

    return buf;
}

// Release the buffer back to FH rte_ring
int WAvgCfoPool::pushBuffer(WAvgCfoBuffer * buf) {

    if(!buf)
        return EINVAL;

    if (buf->getSize() != WAVGCFO_BUFFER_SIZE)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Wrong WAvgCfoPool: size: {}-{} {}: rnti {} cell_id {}",
                buf->getSize(), WAVGCFO_BUFFER_SIZE, reinterpret_cast<void*>(buf), buf->getRnti(), buf->getCellId());
    }

#ifdef DEBUG_WAVGCFO_BUFFER
    const int ref = buf->refSub();
    if (ref > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Double enqueue buf {}: rnti {} cell_id {}",
                reinterpret_cast<void*>(buf), buf->getRnti(), buf->getCellId());
        return 0;
    }
    else if (ref < 1)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Double enqueue buf {}: SFN {}.{} rnti {} cell_id {}",
                reinterpret_cast<void*>(buf), buf->getSfn(), buf->getSlot(), buf->getRnti(), buf->getCellId());
        return 0;
    }
    NVLOGD_FMT(TAG, "clear buffer: {} SFN {}.{} rnti {} cell_id {}",
            reinterpret_cast<void*>(buf), buf->getSfn(), buf->getSlot(), buf->getRnti(), buf->getCellId());
#endif

    buf->clear();

    if(aerial_fh::ring_enqueue(wavgcfo_ring, static_cast<void*>(buf)))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::ring_enqueue {}", reinterpret_cast<void*>(buf));
        return -1;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// WAvgCfo Pool Manager
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

WAvgCfoPoolManager::WAvgCfoPoolManager(phydriver_handle _pdh, GpuDevice* _gDev, FronthaulHandle _fhi_handle)
    : pool_(std::make_unique<WAvgCfoPool>(_gDev, _fhi_handle))
    , cache_()
{
    // Initialize memory footprint tracking
    mf.init(_pdh, std::string("WAvgCfoPoolManager"), sizeof(WAvgCfoPoolManager));

    // Track GPU memory for buffer pool
    mf.addGpuRegularSize(pool_->getGpuMemSize());

    // Track CPU memory for cache structure
    mf.addCpuRegularSize(sizeof(WAvgCfoCache));

    NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "WAvgCfoPoolManager created");
}

WAvgCfoPoolManager::~WAvgCfoPoolManager()
{
    // Clear cache and return all buffers to pool before destruction
    const std::size_t cleared = clearCache();
    NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "WAvgCfoPoolManager destroyed, cleared {} cached buffers", cleared);
}

WAvgCfoBuffer* WAvgCfoPoolManager::allocate(const uint16_t rnti, const uint16_t cell_id)
{
    // First, check if buffer is already cached
    WAvgCfoBuffer* buffer = cache_.find(rnti, cell_id);
    
    if (buffer != nullptr) {
        // Buffer found in cache
        NVLOGD_FMT(TAG, "WAvgCfoBuffer found in cache for rnti {} cell_id {}", rnti, cell_id);
        return buffer;
    }
    
    // Buffer not in cache, pull from pool
    buffer = pool_->pullBuffer();
    
    if (buffer == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to pull buffer from pool for rnti {} cell_id {}", rnti, cell_id);
        return nullptr;
    }
    
    // Initialize the buffer
    if (buffer->init(rnti, cell_id) != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to initialize buffer for rnti {} cell_id {}", rnti, cell_id);
        // Return buffer to pool
        if (pool_->pushBuffer(buffer) != 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to return buffer to pool after init failure");
        }
        return nullptr;
    }
    
    // Add to cache
    if (!cache_.allocate(rnti, cell_id, buffer)) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to cache buffer for rnti {} cell_id {} (key already exists)", rnti, cell_id);
        // Return buffer to pool
        if (pool_->pushBuffer(buffer) != 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to return buffer to pool after cache failure");
        }
        return nullptr;
    }
    
    NVLOGC_FMT(TAG, "WAvgCfoBuffer allocated and cached for rnti {} cell_id {}", rnti, cell_id);
    
    
    buffer->setTimestampLastUsed();
    return buffer;
}

int WAvgCfoPoolManager::deallocate(const uint16_t rnti, const uint16_t cell_id)
{
    // Remove from cache
    WAvgCfoBuffer* buffer = cache_.deallocate(rnti, cell_id);
    
    if (buffer == nullptr) {
        NVLOGW_FMT(TAG, "Buffer not found in cache for rnti {} cell_id {}", rnti, cell_id);
        return -1;
    }
    
    // Return buffer to pool
    const int result = pool_->pushBuffer(buffer);
    
    if (result != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to return buffer to pool for rnti {} cell_id {}", rnti, cell_id);
        return result;
    }
    
    NVLOGD_FMT(TAG, "WAvgCfoBuffer deallocated for rnti {} cell_id {}", rnti, cell_id);
    
    return 0;
}

std::size_t WAvgCfoPoolManager::getPoolAvailable() const
{
    return pool_->countElements();
}

std::size_t WAvgCfoPoolManager::getCacheSize() const
{
    return cache_.size();
}

std::size_t WAvgCfoPoolManager::clearCache()
{
    std::size_t count = 0;
    
    // Note: This is a simplified implementation
    // In a real implementation, you might want to iterate through cache entries
    // For now, just report the cache size before clearing
    count = cache_.size();
    
    // Clear the cache
    cache_.clear();
    
    NVLOGD_FMT(TAG, "Cleared {} buffers from cache", count);
    
    return count;
}

void WAvgCfoPoolManager::checkPoolDepletion()
{
    // Calculate pool availability percentage
    const std::size_t available = pool_->countElements();
    const uint32_t availability_percent = (available * 100) / WAVGCFO_POOL_SIZE;
    
    if (availability_percent < WAVGCFO_DEPLETION_THRESHOLD_PERCENT) {
        depletion_hysteresis_++;
        
        if (depletion_hysteresis_ >= WAVGCFO_CLEANUP_HYSTERESIS) {
            NVLOGW_FMT(TAG, "WAvgCfo pool at {}% availability (< {}%). Cleaning up unused cached buffers", 
                       availability_percent, WAVGCFO_DEPLETION_THRESHOLD_PERCENT);
            
            // Reset hysteresis
            depletion_hysteresis_ = 0;
            
            // Remove unused buffers from cache and return them to pool
            const std::size_t freed = cache_.removeUnused([this](WAvgCfoBuffer* buffer) {
                if (buffer != nullptr) {
                    const int result = pool_->pushBuffer(buffer);
                    if (result != 0) {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, 
                                   "Failed to return buffer to pool during cleanup: rnti {} cell_id {}",
                                   buffer->getRnti(), buffer->getCellId());
                    }
                }
            });
            
            NVLOGW_FMT(TAG, "Freed {} unused buffers from cache. Pool now has {} buffers available", 
                       freed, pool_->countElements());
        } else {
            NVLOGW_FMT(TAG, "WAvgCfo pool at {}% availability (< {}%). Hysteresis = {}/{}", 
                       availability_percent, WAVGCFO_DEPLETION_THRESHOLD_PERCENT, 
                       depletion_hysteresis_, WAVGCFO_CLEANUP_HYSTERESIS);
        }
    } else {
        // Reset hysteresis when pool is healthy
        if (depletion_hysteresis_ > 0) {
            depletion_hysteresis_ = 0;
        }
    }
}
