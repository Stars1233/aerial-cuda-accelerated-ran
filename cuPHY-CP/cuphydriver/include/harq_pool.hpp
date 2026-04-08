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

#ifndef HARQB_H
#define HARQB_H

#include <unordered_map>
#include <iostream>
#include <typeinfo>
#include <atomic>
#include "gpudevice.hpp"
#include "time.hpp"
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "fh.hpp"

static constexpr int MAX_HARQ_POOLS = 3;                       ///< Maximum number of HARQ buffer pools (different buffer sizes)
static constexpr int HARQ_POOL_SIZE[] = {262144, 4000000, 8000000}; ///< HARQ pool buffer sizes in bytes (small, medium, large)
static constexpr int MAX_HARQ_RE_TX = 4;                       ///< Maximum number of HARQ retransmissions per harq process ID
static constexpr int MAX_BUCKET_ENTRY = 16;                    ///< Maximum number of bucket entries to group range of RNTI values
static constexpr int MAX_HARQ_PID = 16;                        ///< Maximum number of HARQ process IDs
static constexpr int MAX_BUCKET_MAP_ENTRIES = 64;              ///< Maximum number of HARQ buffer entries per bucket map cell
static constexpr float MAX_HARQB_NS_BUCKET = 1000000000;       ///< Maximum HARQ buffer retention time in nanoseconds (1 second)
static constexpr uint32_t HARQ_CLEANUP_HYSTERSIS = 5;          ///< Hysteresis count for HARQ buffer cleanup (delays cleanup to avoid thrashing)

// #define DEBUG_HARQ_POOL

/**
 * @brief HARQ buffer for storing transport blocks between transmissions
 *
 * Manages GPU device memory for HARQ (Hybrid Automatic Repeat Request) soft-combining.
 * Each buffer is associated with a specific UE (RNTI) and HARQ process ID, storing
 * decoded transport block data for potential retransmissions.
 */
typedef struct _HarqBuffer
{

protected:
    t_ns lastUsed_t;                                               ///< Timestamp of last buffer usage (for cleanup tracking)
    uint32_t txCount;                                              ///< Current transmission count (0 to MAX_HARQ_RE_TX-1)
    uint8_t max_tx_count;                                          ///< Maximum transmission count for this HARQ process
    size_t size;                                                   ///< Buffer size in bytes (used to return buffer to correct pool)
    std::unique_ptr<dev_buf> buffer;                               ///< GPU device buffer for storing transport block data
    uint32_t rnti;                                                 ///< Radio Network Temporary Identifier (UE identifier)
    uint32_t harq_pid;                                             ///< HARQ process ID (0-15, used to locate buffer in bucket)
    uint64_t cell_id;                                              ///< Cell identifier
    int cellDynIdx;                                                ///< Dynamic cell index
    uint16_t sfn;                                                  ///< System Frame Number when buffer was last used
    uint16_t slot;                                                 ///< Slot number when buffer was last used
    std::atomic<bool> in_use;                                      ///< Atomic flag indicating if buffer is currently in use
    std::atomic<int> ref_count;                                    ///< Number of active references to this buffer (decremented on callback regardless of CRC/timeout)
    uint8_t num_tti_bundled;                                       ///< Number of TTI slots bundled (includes slot gaps, e.g., slots 4 and 6 = 3 TTIs)

public:
    /**
     * @brief Construct HARQ buffer with device memory
     *
     * @param bdev  - GPU device buffer (ownership transferred to this object)
     * @param _size - Buffer size in bytes
     */
    _HarqBuffer(dev_buf * bdev, size_t _size)
    {
        buffer.reset(bdev);
        buffer->clear();
        size = _size;
        in_use = false;
        txCount = 0;
        max_tx_count = 0;
        rnti = 0;
        harq_pid = 0;
        cell_id = 0;
        cellDynIdx=-1;
        sfn = 0;
        slot = 0;
        lastUsed_t={};
        ref_count.store(0);
        num_tti_bundled = 0;
    }

    /**
     * @brief Get GPU device memory address
     *
     * @return Pointer to GPU device buffer
     */
    uint8_t * getAddr() const {
        return buffer->addr();
    }

    /**
     * @brief Get current transmission count
     *
     * @return Number of transmissions (0 to MAX_HARQ_RE_TX-1)
     */
    uint32_t getTxCount() const {
        return txCount;
    }
    /**
     * Get the maximum transmission count for this HARQ process
     * @return Maximum transmission count
     */
    uint32_t getMaxTxCount() const {
        return max_tx_count;
    }
    /**
     * Set the maximum transmission count for this HARQ process
     * @param[in] max_tx_count_ Maximum transmission count to set
     */
    void setMaxTxCount(uint8_t max_tx_count_) {
        max_tx_count = max_tx_count_;
    }
    /**
     * @brief Increment transmission count with wraparound
     *
     * @return true always (for compatibility)
     */
    bool increaseTxCount() {
        txCount = (txCount+1)%MAX_HARQ_RE_TX;
        //FIXME: what if it reaches MAX_HARQ_RE_TX ?
        return true;
    }

    /**
     * @brief Reset transmission count to zero
     */
    void resetTxCount() {
        txCount = 0;
    }

    /**
     * @brief Get timestamp of last buffer usage
     *
     * @return Timestamp in nanoseconds
     */
    t_ns getTimestampLastUsed() const {
        return lastUsed_t;
    }

    /**
     * @brief Update timestamp to current time
     */
    void setTimestampLastUsed() {
        lastUsed_t = Time::nowNs();
    }

    /**
     * @brief Set cell dynamic index
     *
     * @param idx - Cell dynamic index
     */
    void setCellDynIdx(int idx)
    {
        cellDynIdx = idx;
    }

    /**
     * @brief Get cell dynamic index
     *
     * @return Cell dynamic index
     */
    int getCellDynIdx()
    {
        return cellDynIdx;
    }

    /**
     * @brief Get buffer size
     *
     * @return Buffer size in bytes
     */
    size_t getSize() const {
        return size;
    }

    /**
     * @brief Initialize HARQ buffer with UE and cell information
     *
     * @param _rnti     - Radio Network Temporary Identifier
     * @param _harq_pid - HARQ process ID (0-15)
     * @param _cell_id  - Cell identifier
     * @return 0 on success
     */
    int init(uint32_t _rnti, uint32_t _harq_pid,uint64_t _cell_id) {
        rnti = _rnti;
        harq_pid = _harq_pid;
        txCount = 0;
        lastUsed_t = Time::nowNs();
        cell_id = _cell_id;
        ref_count.store(0);
        num_tti_bundled = 0;
        return 0;
    }

    /**
     * @brief Get UE RNTI
     *
     * @return Radio Network Temporary Identifier
     */
    uint32_t getRnti() const {
        return rnti;
    }

    /**
     * @brief Get HARQ process ID
     *
     * @return HARQ process ID (0-15)
     */
    uint32_t getHarqPid() const {
        return harq_pid;
    }
    
    /**
     * @brief Get cell ID
     *
     * @return Cell identifier
     */
    uint64_t getCellId() const {
        return cell_id;
    }

    /**
     * @brief Set SFN and slot
     *
     * @param _sfn  - System Frame Number
     * @param _slot - Slot number
     */
    void setSfnSlot(uint16_t _sfn, uint16_t _slot){
        sfn = _sfn;
        slot = _slot;
    }

    /**
     * @brief Get System Frame Number
     *
     * @return SFN when buffer was last used
     */
    uint16_t getSfn(){
        return sfn;
    }

    /**
     * @brief Get slot number
     *
     * @return Slot when buffer was last used
     */
    uint16_t getSlot(){
        return slot;
    }

    /**
     * @brief Increment reference count atomically
     *
     * Increments the number of active references to this HARQ buffer.
     * Used to track concurrent usage across multiple processing stages.
     *
     * @return Previous reference count before increment
     */
    int refAdd() {
        return ref_count.fetch_add(1);
    }

    /**
     * @brief Decrement reference count atomically
     *
     * Decrements when processing completes (callback received).
     * Called regardless of CRC status or timeout to track active usage.
     *
     * @return Previous reference count before decrement
     */
    int refSub() {
        return ref_count.fetch_sub(1);
    }

    /**
     * @brief Reset reference count to zero
     *
     * Used when buffer is freshly allocated or released back to pool.
     */
    void refReset() {
        ref_count.store(0);
    }

    /**
     * @brief Get current reference count
     *
     * @return Current number of active references
     */
    int getRefCount() {
        return ref_count.load();
    }

    /**
     * @brief Mark buffer as in use
     *
     * Sets atomic flag to indicate buffer is actively being used for transmission.
     */
    void setInUse() {
        in_use = true;
    }

    /**
     * @brief Mark buffer as not in use
     */
    void unsetInUse() {
        in_use = false;
    }

    /**
     * @brief Check if buffer is in use
     *
     * @return true if buffer is currently in use, false otherwise
     */
    bool isInUse() {
        bool ret = in_use;
        return ret;
    }
    /**
     * Get the number of TTI slots bundled for this HARQ process
     * @return Number of bundled TTI slots
     */
    uint32_t getNumTtiBundled() const {
        return num_tti_bundled;
    }
    /**
     * Increment the number of bundled TTI slots
     * @param[in] inc Number of TTI slots to add to the bundle count
     */
    void incrementNumTtiBundled(uint8_t inc) {
        this->num_tti_bundled += inc;
    }

    /**
     * @brief Reset bundled TTI count to zero
     *
     * Called when starting a new HARQ process or releasing buffer.
     */
    void resetNumTtiBundled() {
        num_tti_bundled = 0;
    }

    /**
     * @brief Clear buffer state (reset all tracking info)
     */
    void clear() {
        txCount=0;
        max_tx_count = 0;
        lastUsed_t = Time::zeroNs();
        rnti = 0;
        harq_pid = 0;
        in_use = false;
        cellDynIdx=-1;
        ref_count.store(0);
        num_tti_bundled = 0;
    }

} HarqBuffer;

/**
 * @brief HARQ bucket entry for holding active HARQ buffers
 *
 * Thread-safe container for HARQ buffers indexed by RNTI. Handles collisions
 * where multiple UEs map to the same bucket through fixed-size array.
 */
struct HarqBucketEntry {
    Mutex mlock;                                               ///< Mutex for thread-safe access to this bucket entry
    std::array<HarqBuffer *, MAX_BUCKET_MAP_ENTRIES> hb_rnti_v; ///< Array of HARQ buffer pointers (handles RNTI collisions)
};

/**
 * @brief HARQ bucket manager for organizing active buffers by RNTI
 *
 * Organizes active HARQ buffers into buckets based on RNTI modulo operation.
 * Each bucket contains a 2D map [bucket_entry][harq_pid] to handle RNTI collisions
 * and multiple HARQ processes per UE.
 */
class HarqBucket {

public:
    /**
     * @brief Construct HARQ bucket for RNTI range
     *
     * @param _rnti - RNTI modulo value this bucket handles
     */
    HarqBucket(int _rnti) : rnti(_rnti) {
        for(int i=0; i<MAX_BUCKET_ENTRY; i++ )
        {
            for(int j=0; j<MAX_HARQ_PID; j++ )
            {
                // bucket_map[i][j].hb_rnti_v.reserve(MAX_BUCKET_MAP_ENTRIES);
                for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++ )
                    bucket_map[i][j].hb_rnti_v[k] = nullptr;
            }
        }
            // bucket_map[i][j].hb_rnti_map.reserve(MAX_BUCKET_MAP_ENTRIES);
    }

    /**
     * @brief Destroy HARQ bucket
     */
    ~HarqBucket() {
        // for(int i=0; i<MAX_BUCKET_ENTRY; i++ )
        //     for(int j=0; j<MAX_HARQ_PID; j++ )
        //         bucket_map[i][j].hb_rnti_v.clear();
                // bucket_map[i][j].hb_rnti_map.clear();
    }

    /**
     * @brief Get bucket RNTI modulo value
     *
     * @return RNTI modulo value this bucket handles
     */
    int getBucketRnti() const;
    
    /**
     * @brief Push HARQ buffer into bucket
     *
     * @param hb_ptr   - HARQ buffer pointer to insert
     * @param rnti     - UE RNTI
     * @param harq_pid - HARQ process ID
     * @param cell_id  - Cell identifier
     * @return 0 on success, negative error code on failure
     */
    int pushBuffer(HarqBuffer * hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id);
    
    /**
     * @brief Get HARQ buffer from bucket (without removing)
     *
     * @param[out] hb_ptr - Pointer to receive buffer pointer
     * @param rnti          - UE RNTI
     * @param harq_pid      - HARQ process ID
     * @param cell_id       - Cell identifier
     * @return 0 on success, negative error code if not found
     */
    int getBuffer(HarqBuffer ** hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id);
    
    /**
     * @brief Remove HARQ buffer from bucket
     *
     * @param rnti     - UE RNTI
     * @param harq_pid - HARQ process ID
     * @param cell_id  - Cell identifier
     * @return 0 on success, negative error code if not found
     */
    int pullBuffer(uint32_t rnti, uint32_t harq_pid, uint64_t cell_id);

    struct HarqBucketEntry  bucket_map[MAX_BUCKET_ENTRY][MAX_HARQ_PID]; ///< 2D map of bucket entries [entry_index][harq_pid] for collision handling

protected:
    int                     rnti;                              ///< RNTI modulo value this bucket handles
};

/**
 * @brief HARQ buffer pool with fixed-size buffers
 *
 * Maintains a pool of pre-allocated HARQ buffers of uniform size for efficient
 * allocation and deallocation. Uses lockless ring buffer for thread-safe
 * free buffer management.
 */
class HarqPool {

public:
    /**
     * @brief Construct HARQ buffer pool
     *
     * @param _size        - Size of each HARQ buffer in bytes
     * @param _sizePool    - Number of buffers to pre-allocate in this pool
     * @param _gDev        - GPU device for buffer allocation
     * @param _fhi_handle  - Fronthaul handle for ring buffer creation
     */
    HarqPool(size_t _size, size_t _sizePool,GpuDevice* _gDev, FronthaulHandle _fhi_handle);
    
    /**
     * @brief Destroy HARQ buffer pool
     */
    ~HarqPool();

    /**
     * @brief Get buffer size for this pool
     *
     * @return Buffer size in bytes
     */
    size_t getSize() const;

    /**
     * @brief Count available free buffers
     *
     * @return Number of free buffers in pool
     */
    size_t countElements() const;

    /**
     * @brief Get total GPU memory size allocated by this pool for memory footprint
     *
     * @return Total GPU memory size allocated by this pool in bytes
     */
    size_t getGpuMemSize() const;

    /**
     * @brief Pull free buffer from pool
     *
     * @return Pointer to free HARQ buffer, nullptr if pool is empty
     */
    HarqBuffer * pullBuffer();
    
    /**
     * @brief Return buffer to pool
     *
     * @param buf - HARQ buffer pointer to return to pool
     * @return 0 on success, negative error code on failure
     */
    int pushBuffer(HarqBuffer * buf);

protected:
    std::vector<std::unique_ptr<HarqBuffer>>    hb_list;       ///< Vector of all allocated HARQ buffers (owns memory)
    RingBufferHandle                            hb_ring;       ///< Lockless ring buffer for free buffer pointers
    RingBufferInfo                              ring_info;     ///< Ring buffer configuration information
    int                                         totRingElems;  ///< Total number of ring elements (should match sizePool)
    size_t                                      size;          ///< Size of each buffer in bytes
    size_t                                      sizePool;      ///< Total number of buffers in this pool
    size_t                                      gpuMemSize;    ///< Total GPU memory size allocated by this pool in bytes
    GpuDevice*                                  gDev;          ///< GPU device pointer for memory operations
};

/**
 * @brief HARQ pool and bucket manager
 *
 * Top-level manager for all HARQ resources. Manages multiple buffer pools
 * (different sizes), RNTI-based buckets for active buffers, and provides
 * allocation/deallocation interface. Handles buffer lifecycle from allocation
 * through retransmissions to release.
 */
class HarqPoolManager {

public:
    /**
     * @brief Construct HARQ pool manager
     *
     * @param _pdh  - Physical layer driver handle
     * @param _gDev - GPU device for buffer allocation
     */
    HarqPoolManager(phydriver_handle _pdh, GpuDevice* _gDev);
    
    /**
     * @brief Destroy HARQ pool manager
     */
    ~HarqPoolManager();

    /**
     * @brief Allocate new pool for given buffer size
     *
     * @param buf_size - Buffer size in bytes
     * @return Pointer to allocated pool, nullptr on failure
     */
    HarqPool * poolAlloc(size_t buf_size);
    
    /**
     * @brief Find existing pool for buffer size
     *
     * @param buf_size - Buffer size in bytes
     * @return Pointer to pool if found, nullptr otherwise
     */
    HarqPool * poolFind(size_t buf_size);
    
    /**
     * @brief Check all pools for depletion and log warnings
     */
    void checkPoolDepletion(ReleasedHarqBufferInfo& released_harq_buffer_info);
    
    /**
     * @brief Find HARQ buffer in bucket by RNTI and process ID
     *
     * @param[out] hb_ptr - Pointer to receive buffer pointer
     * @param rnti          - UE RNTI
     * @param harq_pid      - HARQ process ID
     * @param cell_id       - Cell identifier
     * @return 0 if found, negative error code otherwise
     */
    int bucketFindBuffer(HarqBuffer ** hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id);

    /**
     * @brief Allocate and insert HARQ buffer into bucket
     *
     * If buffer doesn't exist in bucket, pulls a new free buffer of the specified
     * size from the appropriate HarqPool and places it into the bucket for the
     * given RNTI and HARQ process ID.
     *
     * @param[out] hb_ptr           - Pointer to receive allocated buffer pointer
     * @param buf_size               - Required buffer size in bytes
     * @param rnti                   - UE RNTI
     * @param harq_pid               - HARQ process ID
     * @param ndi                    - New Data Indicator pointer
     * @param cell_id                - Cell identifier
     * @param is_bundled_pdu         - Flag indicating if this is a bundled PDU
     * @param pusch_aggr_factor      - PUSCH aggregation factor for bundling
     * @param sfn                    - System Frame Number
     * @param slot                   - Slot number
     * @return 0 on success, negative error code on failure
     */
    int bucketAllocateBuffer(HarqBuffer ** hb_ptr, size_t buf_size, uint32_t rnti, uint32_t harq_pid, uint8_t* ndi, uint64_t cell_id, bool is_bundled_pdu, uint8_t pusch_aggr_factor, uint16_t sfn, uint16_t slot);

    /**
     * @brief Release HARQ buffer from bucket back to pool
     *
     * Removes the buffer from its bucket and returns it to the appropriate
     * HarqPool based on its size. Buffer state is cleared before returning.
     *
     * @param hb_ptr - HARQ buffer pointer to release
     * @return 0 on success, negative error code on failure
     */
    int bucketReleaseBuffer(HarqBuffer * hb_ptr);
    
    /**
     * @brief Mark HARQ buffer as not in use (for retransmissions)
     *
     * Buffer remains in bucket but is marked available for next transmission.
     *
     * @param hb_ptr - HARQ buffer pointer
     * @return 0 on success, negative error code on failure
     */
    int unsetInUse(HarqBuffer * hb_ptr);
    
    /**
     * @brief Clean up unused HARQ buffers from all buckets
     *
     * Scans all buckets and releases buffers that haven't been used recently.
     *
     * @return Number of buffers cleaned up
     */
    int cleanupHarqBuckets(ReleasedHarqBufferInfo& released_harq_buffer_info);
    
    /**
     * @brief Free all HARQ buffers (emergency cleanup)
     */
    void freeAllHarqBuffers(ReleasedHarqBufferInfo& released_harq_buffer_info);
    
    /**
     * @brief Get unique manager ID
     *
     * @return Manager identifier (timestamp-based)
     */
    uint64_t            getId() const;

    MemFoot             mf;                                    ///< Memory footprint tracker for all HARQ pools

protected:
    uint64_t                                    id;            ///< Unique manager identifier (timestamp-based)
    phydriver_handle                            pdh;           ///< Physical layer driver handle
    GpuDevice*                                  gDev;          ///< GPU device pointer for buffer operations
    std::vector<std::unique_ptr<HarqPool>>      hb_pool_list;  ///< List of HARQ pools (one per buffer size)
    std::vector<std::unique_ptr<HarqBucket>>    hb_bucket_list; ///< List of HARQ buckets (organized by RNTI % bucket_count)
};

#endif
