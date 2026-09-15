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

#ifndef NV_LOCKFREE_H_INCLUDED_
#define NV_LOCKFREE_H_INCLUDED_

#include <stddef.h>
#include <stdint.h>
#include "nv_ipc_ring.h"
#include "nv_ipc_mempool.h"

namespace nv {

#define LOCK_FREE_OPT_APP_INTERNAL (0)
#define LOCK_FREE_OPT_SHM_PRIMARY (1)
#define LOCK_FREE_OPT_SHM_SECONDARY (2)

// NOTE: name length (including '\0') <= 32. Long than 31 characters will be truncated so may cause error.

/**
 * Lock-free memory pool
 *
 * @tparam T Type of the buffer
 *
 * This class is a wrapper for the nv_ipc_mempool_t class. It provides a lock-free memory pool for the buffer type T.
 * The memory pool supports to be allocated on CPU or GPU.
 * The memory pool supports to be used in application internal or shared between primary and secondary processes.
 */
template <typename T>
class lock_free_mem_pool {
public:

    /**
     * Constructor for lock-free memory pool
     *
     * @param length Number of buffers in the pool
     * @param flags Flags for the pool (APP_INTERNAL, SHM_PRIMARY, SHM_SECONDARY)
     * @param name Name of the pool
     * @param cuda_device_id CUDA device ID. -1 for CPU pool, otherwise the CUDA device ID
     * @param buf_size Size of each buffer in bytes. Defaults to sizeof(T). Use this to
     *                 allocate variably-sized buffers (e.g. pool of raw bytes sized at runtime).
     */
    lock_free_mem_pool(uint32_t length, uint32_t flags = LOCK_FREE_OPT_APP_INTERNAL, const char* name = nullptr, int cuda_device_id = -1, size_t buf_size = sizeof(T)) {
        int primary = 0;
        if (flags == LOCK_FREE_OPT_SHM_PRIMARY) {
            primary = 1;
        } else if (flags == LOCK_FREE_OPT_SHM_SECONDARY) {
            primary = 0;
        } else {
            primary = 0xFF;
        }

        mempool = nv_ipc_mempool_open(primary, name, static_cast<int>(buf_size), length, cuda_device_id);
    }

    /**
     * Allocate a buffer from the memory pool. Return nullptr if the memory pool is empty
     *
     * @return Pointer to the allocated buffer
     */
    T* alloc() {
        if (mempool == nullptr) {
            return nullptr;
        }
        int32_t index = mempool->alloc(mempool);
        return reinterpret_cast<T*>(mempool->get_addr(mempool, index));
    }

    /**
     * Free a buffer back to the memory pool
     *
     * @param buf Pointer to the buffer to free
     * @return 0 on success, -1 on failure
     */
    int free(T* buf) {
        if (mempool == nullptr) {
            return -1;
        }
        int32_t index = mempool->get_index(mempool, buf);
        return  mempool->free(mempool, index);
    }

    /**
     * Get the buffer address by buffer index
     *
     * @param index Index of the buffer
     * @return Pointer to the buffer address
     */
    T* get_buf_addr(int32_t index) {
        if (mempool == nullptr) {
            return nullptr;
        }
        return reinterpret_cast<T*>(mempool->get_addr(mempool, index));
    }

    /**
     * Get the buffer index by buffer pointer
     *
     * @param buf Pointer to the buffer
     * @return Index of the buffer
     */
    int get_buf_index(T* buf) {
        if (mempool == nullptr) {
            return -1;
        }
        return mempool->get_index(mempool, buf);
    }

    /**
     * Get the pool length
     *
     * @return Length of the pool
     */
    int get_pool_len() {
        if (mempool == nullptr) {
            return -1;
        }
        return  mempool->get_pool_len(mempool);
    }

    /**
     * Get the free buffer count
     *
     * @return Free buffer count
     */
    int get_free_count() {
        if (mempool == nullptr) {
            return -1;
        }
        return  mempool->get_free_count(mempool);
    }

    /**
     * Get the size of each buffer in bytes
     *
     * @return Buffer size in bytes
     */
    int get_buf_size() {
        if (mempool == nullptr) {
            return -1;
        }
        return mempool->get_buf_size(mempool);
    }

    /**
     * Destructor for lock-free memory pool
     */
    ~lock_free_mem_pool() {
        if (mempool != nullptr) {
            mempool->close(mempool);
        }
    }

private:
    nv_ipc_mempool_t* mempool = nullptr; ///< Pointer to the nv_ipc_mempool_t instance
};

/**
 * Lock-free ring memory pool
 *
 * @tparam T Type of the buffer
 *
 * This class is a wrapper for the nv_ipc_ring_t class. It provides a lock-free ring memory pool for the buffer type T.
 * The ring memory pool supports to be used in application internal or shared between primary and secondary processes.
 */
template <typename T>
class lock_free_ring_pool {
public:

    /**
     * Constructor for lock-free ring memory pool
     *
     * @param name Name of the ring
     * @param length Length of the ring
     * @param buf_size Size of the buffer
     * @param flags Flags for the ring (APP_INTERNAL, SHM_PRIMARY, SHM_SECONDARY)
     */
    lock_free_ring_pool(const char* name, uint32_t length, uint32_t buf_size = sizeof(T), uint32_t flags = LOCK_FREE_OPT_APP_INTERNAL) {
        ring_type_t type;
        if (flags == LOCK_FREE_OPT_SHM_PRIMARY) {
            type = RING_TYPE_SHM_PRIMARY;
        } else if (flags == LOCK_FREE_OPT_SHM_SECONDARY) {
            type = RING_TYPE_SHM_SECONDARY;
        } else {
            type = RING_TYPE_APP_INTERNAL;
        }

        ring = nv_ipc_ring_open(type, name, length, buf_size);
    }

    /**
     * Allocate a buffer from the memory pool. Return nullptr if the memory pool is empty
     *
     * @return Pointer to the allocated buffer
     */
    T* alloc() {
        if (ring == nullptr) {
            return nullptr;
        }
        int32_t index = ring->alloc(ring);
        return index < 0 ? nullptr : reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    /**
     * Free a buffer back to the memory pool
     *
     * @param buf Pointer to the buffer to free
     * @return 0 on success, -1 on failure
     */
    int free(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        int32_t index = ring->get_index(ring, buf);
        return ring->free(ring, index);
    }

    /**
     * Enqueue a buffer pointer into the ring queue
     *
     * @param buf Pointer to the buffer to enqueue
     * @return 0 on success, -1 on failure
     */
    int enqueue(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        int32_t index = ring->get_index(ring, buf);
        return ring->enqueue_by_index(ring, index);
    }

    /**
     * Dequeue a buffer pointer from ring queue. Return nullptr if the ring queue is empty
     *
     * @return Pointer to the dequeued buffer
     */
    T* dequeue() {
        if (ring == nullptr) {
            return nullptr;
        }
        int32_t index = ring->dequeue_by_index(ring);
        return reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    /**
     * Automatically allocate a buffer, copy the source object into the buffer, and enqueue the buffer pointer to the ring pool
     *
     * @param obj Pointer to the source object
     * @return 0 on success, -1 on failure
     */
    int copy_enqueue(T* obj) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->enqueue(ring, obj);
    }

    /**
     * Automatically dequeue a buffer pointer from the ring queue, copy to the destination buffer, and free the source buffer to the memory pool
     *
     * @param obj Pointer to the destination buffer
     * @return 0 on success, -1 on failure
     */
    int copy_dequeue(T* obj) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->dequeue(ring, obj);
    }

    /**
     * Get the buffer pointer by buffer index
     *
     * @param index Index of the buffer
     * @return Pointer to the buffer address
     */
    T* get_buf_addr(int32_t index) {
        if (ring == nullptr) {
            return nullptr;
        }
        return index < 0 ? nullptr : reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    /**
     * Get the index pointer by buffer pointer
     *
     * @param buf Pointer to the buffer
     * @return Index of the buffer
     */
    int32_t get_buf_index(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->get_index(ring, buf);
    }

    /**
     * Get free buffer count in the ring memory pool
     *
     * @return Free buffer count
     */
    unsigned int get_free_count() {
        if (ring == nullptr) {
            return 0;
        }
        int free_count = ring->get_free_count(ring);
        return free_count < 0 ? 0 : free_count;
    }

    /**
     * Get enqueued object count
     *
     * @return Enqueued object count
     * @note ring_len = free_count + enqueued_count only when no flowing buffer (which was allocated but not enqueued).
     */
    unsigned int get_enqueued_count() {
        if (ring == nullptr) {
            return 0;
        }
        int count = ring->get_count(ring);
        return count < 0 ? 0 : count;
    }

    /**
     * Get the object buffer size
     *
     * @return Buffer size
     */
    unsigned int get_buf_size() {
        if (ring == nullptr) {
            return 0;
        }
        int buf_size = ring->get_buf_size(ring);
        return buf_size < 0 ? 0 : buf_size;
    }

    /**
     * Get the ring length
     *
     * @return Ring length
     */
    unsigned int get_ring_len() {
        if (ring == nullptr) {
            return 0;
        }
        int ring_len = ring->get_ring_len(ring);
        return ring_len < 0 ? 0 : ring_len;
    }

    /**
     * Destructor for lock-free ring memory pool
     */
    ~lock_free_ring_pool() {
        if (ring != nullptr) {
            ring->close(ring);
        }
    }

private:
    nv_ipc_ring_t* ring = nullptr; ///< Pointer to the nv_ipc_ring_t instance
};


} // namespace nv

#endif /* NV_LOCKFREE_H_INCLUDED_ */
