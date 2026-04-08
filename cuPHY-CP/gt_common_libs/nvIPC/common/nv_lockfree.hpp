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

#ifndef NV_RING_H_INCLUDED_
#define NV_RING_H_INCLUDED_

#include <stdint.h>
#include "nv_ipc_ring.h"
#include "nv_ipc_mempool.h"

namespace nv {

#define LOCK_FREE_OPT_APP_INTERNAL (0)
#define LOCK_FREE_OPT_SHM_PRIMARY (1)
#define LOCK_FREE_OPT_SHM_SECONDARY (2)

// NOTE: name length (including '\0') <= 32. Long than 31 characters will be truncated so may cause error.
template <typename T>
class lock_free_mem_pool {
public:
    lock_free_mem_pool(uint32_t length, uint32_t flags = LOCK_FREE_OPT_APP_INTERNAL, const char* name = nullptr) {
        int primary = 0;
        if (flags == LOCK_FREE_OPT_SHM_PRIMARY) {
            primary = 1;
        } else if (flags == LOCK_FREE_OPT_SHM_SECONDARY) {
            primary = 0;
        } else {
            primary = 0xFF;
        }

        mempool = nv_ipc_mempool_open(primary, name, sizeof(T), length, -1);
    }

    T* alloc() {
        if (mempool == nullptr) {
            return nullptr;
        }
        int32_t index = mempool->alloc(mempool);
        return reinterpret_cast<T*>(mempool->get_addr(mempool, index));
    }

    int free(T* buf) {
        if (mempool == nullptr) {
            return -1;
        }
        int32_t index = mempool->get_index(mempool, buf);
        return  mempool->free(mempool, index);
    }

    int get_pool_len(T* buf) {
        if (mempool == nullptr) {
            return -1;
        }
        return  mempool->get_pool_len(mempool);
    }

    int get_free_count(T* buf) {
        if (mempool == nullptr) {
            return -1;
        }
        return  mempool->get_free_count(mempool);
    }

    ~lock_free_mem_pool() {
        if (mempool != nullptr) {
            mempool->close(mempool);
        }
    }

private:
    nv_ipc_mempool_t* mempool = nullptr;
};

/* There's a object memory pool and a ring queue inside the ring */
template <typename T>
class lock_free_ring_pool {
public:
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

    // Allocate a buffer from the memory pool. Return nullptr if the memory pool is empty
    T* alloc() {
        if (ring == nullptr) {
            return nullptr;
        }
        int32_t index = ring->alloc(ring);
        return index < 0 ? nullptr : reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    // Free the buffer back to the memory pool
    int free(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        int32_t index = ring->get_index(ring, buf);
        return ring->free(ring, index);
    }

    // Enqueue the buffer pointer into the ring queue
    int enqueue(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        int32_t index = ring->get_index(ring, buf);
        return ring->enqueue_by_index(ring, index);
    }

    // Dequeue a buffer pointer from ring queue. Return nullptr if the ring queue is empty
    T* dequeue() {
        if (ring == nullptr) {
            return nullptr;
        }
        int32_t index = ring->dequeue_by_index(ring);
        return reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    // Automatically allocate a buffer, copy the source object into the buffer, and enqueue the buffer pointer to the ring pool
    int copy_enqueue(T* obj) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->enqueue(ring, obj);
    }

    // Automatically dequeue a buffer pointer from the ring queue, copy to the destination buffer, and free the source buffer to the memory pool
    int copy_dequeue(T* obj) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->dequeue(ring, obj);
    }

    // Get the buffer pointer by buffer index
    T* get_buf_addr(int32_t index) {
        if (ring == nullptr) {
            return nullptr;
        }
        return index < 0 ? nullptr : reinterpret_cast<T*>(ring->get_addr(ring, index));
    }

    // Get the index pointer by buffer pointer
    int32_t get_buf_index(T* buf) {
        if (ring == nullptr) {
            return -1;
        }
        return ring->get_index(ring, buf);
    }

    // Get free buffer count in the ring memory pool
    unsigned int get_free_count() {
        if (ring == nullptr) {
            return 0;
        }
        int free_count = ring->get_free_count(ring);
        return free_count < 0 ? 0 : free_count;
    }

    // Get enqueued object count
    // ring_len = free_count + enqueued_count only when no flowing buffer (which was allocated but not enqueued).
    unsigned int get_enqueued_count() {
        if (ring == nullptr) {
            return 0;
        }
        int count = ring->get_count(ring);
        return count < 0 ? 0 : count;
    }

    // Get the object buffer size
    unsigned int get_buf_size() {
        if (ring == nullptr) {
            return 0;
        }
        int buf_size = ring->get_buf_size(ring);
        return buf_size < 0 ? 0 : buf_size;
    }

    // Get the ring length
    unsigned int get_ring_len() {
        if (ring == nullptr) {
            return 0;
        }
        int ring_len = ring->get_ring_len(ring);
        return ring_len < 0 ? 0 : ring_len;
    }

    ~lock_free_ring_pool() {
        if (ring != nullptr) {
            ring->close(ring);
        }
    }

private:
    nv_ipc_ring_t* ring = nullptr;
};


} // namespace nv

#endif /* NV_RING_H_INCLUDED_ */
