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

#include "ring_buffer.hpp"

#include "dpdk.hpp"
#include "fronthaul.hpp"
#include "utils.hpp"

#define TAG "FH.RING"

namespace aerial_fh
{
RingBuffer::RingBuffer(Fronthaul* fhi, RingBufferInfo const* info) :
    fhi_{fhi},
    info_{*info},
    id_{0}
{
    NVLOGI_FMT(TAG, "Creating ring buffer '{}' of size {}", info_.name, info_.count);

    info_.count        = rte_align32pow2(info_.count);
    unsigned int flags = 0;

    if(!info_.multi_producer)
    {
        flags |= RING_F_SP_ENQ;
    }

    if(!info_.multi_consumer)
    {
        flags |= RING_F_SC_DEQ;
    }

    ring_ = rte_ring_create(info_.name, info_.count, info_.socket_id, flags);

    if(ring_ == nullptr)
    {
        THROW_FH(rte_errno, StringBuilder() << "Failed to create ring buffer '" << info_.name << "' of size " << info_.count << ": " << rte_strerror(rte_errno));
    }
}

RingBuffer::~RingBuffer()
{
    NVLOGI_FMT(TAG, "Destroying ring buffer '{}' of size {}", info_.name, info_.count);
    rte_ring_free(ring_);
}

Fronthaul* RingBuffer::get_fronthaul() const
{
    return fhi_;
}

int RingBuffer::enqueue(void* obj)
{
    return rte_ring_enqueue(ring_, obj);
}

size_t RingBuffer::enqueue_bulk(void* const* objs, size_t count)
{
    return rte_ring_enqueue_bulk(ring_, objs, count, nullptr);
}

size_t RingBuffer::enqueue_burst(void* const* objs, size_t count)
{
    return rte_ring_enqueue_burst(ring_, objs, count, nullptr);
}

int RingBuffer::dequeue(void** obj)
{
    return rte_ring_dequeue(ring_, obj);
}

size_t RingBuffer::dequeue_bulk(void** objs, size_t count)
{
    return rte_ring_dequeue_bulk(ring_, objs, count, nullptr);
}

size_t RingBuffer::dequeue_burst(void** objs, size_t count)
{
    return rte_ring_dequeue_burst(ring_, objs, count, nullptr);
}

size_t RingBuffer::free_count() const
{
    return rte_ring_free_count(ring_);
}

bool RingBuffer::full() const
{
    return rte_ring_full(ring_);
}

bool RingBuffer::empty() const
{
    return rte_ring_empty(ring_);
}

} // namespace aerial_fh
