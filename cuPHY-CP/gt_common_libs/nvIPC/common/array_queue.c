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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <time.h>

#include "array_queue.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 7) //"NVIPC.QUEUE"

#define PRINT_COUNTER 2000

const int32_t VALUE_NULL = -1;

typedef struct
{
    int32_t  next;
    uint32_t counter;
} note_t;

typedef union
{
    note_t   node;
    uint64_t ulong;
} cas_union_t;

typedef struct
{
    int                   primary;
    int32_t               queue_len;
    array_queue_header_t* header; // The array_queue_header_t structure shared by CPU SHM

    // For debug
    char* name;

    int enqueue_try_max;
    int dequeue_try_max;
} priv_data_t;

static inline priv_data_t* get_private_data(array_queue_t* queue)
{
    return (priv_data_t*)((int8_t*)queue + sizeof(array_queue_t));
}

// Enqueue at tail
static int array_queue_enqueue(array_queue_t* queue, int32_t value)
{
    priv_data_t*          priv_data = get_private_data(queue);
    array_queue_header_t* header    = priv_data->header;

    if(value < 0 || value >= priv_data->queue_len)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue error value %d", priv_data->name, value);
        return -1;
    }

    atomic_ulong* pnode_tail;
    cas_union_t   tail, node_new, node_tail, cas_new, cas_expected;

    // Check whether the value is already in queue
    node_new.ulong = atomic_load(&header->queue[value]);
    if(node_new.node.next != VALUE_NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: value %d already in queue", priv_data->name, value);
        return -1;
    }

    int try_count = 0;
    while(1)
    {
        try_count++;
        if(priv_data->enqueue_try_max < try_count)
        {
            priv_data->enqueue_try_max = try_count;
        }

        // Load tail, node_tail and re-check
        tail.ulong = atomic_load(&header->tail);
        if(tail.node.next == VALUE_NULL)
        {
            pnode_tail = &header->head;
        }
        else
        {
            if(tail.node.next < 0 || tail.node.next >= priv_data->queue_len)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid tail next %d", priv_data->name,
                        tail.node.next);
                return -1;
            }
            pnode_tail = &header->queue[tail.node.next];
        }
        node_tail.ulong = atomic_load(pnode_tail);
        if(tail.ulong != atomic_load(&header->tail))
        {
            // tail changed so node_tail need refresh
            continue;
        }

        if(tail.node.counter + 1 == node_tail.node.counter && tail.node.next != node_tail.node.next)
        {
            // If tail was fall behind, move tail
            // If last node was dequeued, the queue is empty, set head.next to NULL first
            if(node_tail.node.next == VALUE_NULL)
            { // && node_tail.node.counter == tail.node.counter + 1
                cas_expected.ulong   = tail.ulong;
                cas_new.node.next    = VALUE_NULL;
                cas_new.node.counter = tail.node.counter + 1;
                atomic_compare_exchange_strong(&header->head, &cas_expected.ulong, cas_new.ulong);
            }

            // Move tail
            cas_expected.ulong   = tail.ulong;
            cas_new.node.next    = node_tail.node.next;
            cas_new.node.counter = tail.node.counter + 1;
            atomic_compare_exchange_strong(&header->tail, &cas_expected.ulong, cas_new.ulong);
            continue;
        }
        else if(tail.node.next == node_tail.node.next && tail.node.counter == node_tail.node.counter)
        {
            if(tail.node.next == value)
            {
                // Normally should not run to here
                NVLOGF(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue error link self: tail:%d-%u node_tail:%d-%u value=%d", priv_data->name, tail.node.next, tail.node.counter, node_tail.node.next, node_tail.node.counter, value);
            }

            cas_new.node.next    = value;
            cas_new.node.counter = tail.node.counter + 1;

            // Set the new node value
            atomic_store(&header->queue[value], cas_new.ulong);

            // Enqueue: replace node_tail.next to the new node
            cas_expected.ulong = node_tail.ulong;
            if(atomic_compare_exchange_strong(pnode_tail, &cas_expected.ulong, cas_new.ulong) != 0) // Enqueue time point
            {
                // Succeeded, move tail and break
                cas_expected.ulong = tail.ulong;
                atomic_compare_exchange_strong(&header->tail, &cas_expected.ulong, cas_new.ulong);
                break;
            }
        }
        else
        {
            NVLOGF(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue error: tail:%d-%u node_tail:%d-%u value=%d", priv_data->name, tail.node.next, tail.node.counter, node_tail.node.next, node_tail.node.counter, value);
        }
    }

    long enq_count = atomic_fetch_add(&header->enq_count, 1);
    if(enq_count % PRINT_COUNTER == 0)
    {
        long deq_count = atomic_load(&header->deq_count);
        enq_count      = atomic_load(&header->enq_count);
        NVLOGI(TAG, "%s: enqueue try_max=%d counter: enq=%lu deq=%lu available~%lu", priv_data->name, priv_data->enqueue_try_max, enq_count, deq_count, enq_count - deq_count);
    }

    NVLOGV(TAG, "%s: enqueued tail:%d-%u node_tail:%d-%u value=%d", priv_data->name, tail.node.next, tail.node.counter, node_tail.node.next, node_tail.node.counter, value);
    return 0;
}

// Dequeue from head
static int array_queue_dequeue(array_queue_t* queue)
{
    priv_data_t*          priv_data = get_private_data(queue);
    array_queue_header_t* header    = priv_data->header;

    atomic_ulong *pnode_head, pnode_tail;
    cas_union_t   node_head, cas_new, head, tail, cas_expected;
    int           try_count = 0;
    while(1)
    {
        try_count++;
        if(priv_data->dequeue_try_max < try_count)
        {
            priv_data->dequeue_try_max = try_count;
        }

        // Load head
        head.ulong = atomic_load(&header->head);
        if(head.node.next == VALUE_NULL)
        {
            NVLOGV(TAG, "%s: dequeue empty 1", priv_data->name);
            // Empty
            return -1;
        }
        if(head.node.next < 0 || head.node.next >= priv_data->queue_len)
        {
            // Should never run to here
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: dequeue invalid head next %d", priv_data->name, head.node.next);
            return -1;
        }

        // Load node_head
        pnode_head      = &header->queue[head.node.next];
        node_head.ulong = atomic_load(pnode_head);

        // Load tail
        tail.ulong = atomic_load(&header->tail);

        // Reload head to check consistent
        if(head.ulong != atomic_load(&header->head))
        {
            continue;
        }

        if(node_head.node.next == VALUE_NULL)
        {
            NVLOGV(TAG, "%s: dequeue empty 2", priv_data->name);
            // Empty
            return -1;
        }

        if(node_head.node.next < 0 || node_head.node.next >= priv_data->queue_len)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid node_head next %d", priv_data->name,
                    node_head.node.next);
            return -1;
        }

        if(head.node.next == tail.node.next && node_head.node.next != head.node.next && node_head.node.counter == tail.node.counter + 1)
        {
            // Move tail
            cas_expected.ulong   = tail.ulong;
            cas_new.node.next    = node_head.node.next; // Error
            cas_new.node.counter = tail.node.counter + 1;
            atomic_compare_exchange_strong(&header->tail, &cas_expected.ulong, cas_new.ulong);
            continue;
        }
        else if(tail.node.next == VALUE_NULL && head.node.next != VALUE_NULL && head.node.counter == tail.node.counter + 1)
        {
            // Only one node, move tail to the first node
            cas_expected.ulong   = tail.ulong;
            cas_new.node.next    = head.node.next; // Error
            cas_new.node.counter = tail.node.counter + 1;
            atomic_compare_exchange_strong(&header->tail, &cas_expected.ulong, cas_new.ulong);
        }

        if(node_head.node.next == head.node.next && node_head.node.counter == head.node.counter)
        {
            // Only one node is in the queue
            cas_expected.ulong   = node_head.ulong;
            cas_new.node.next    = VALUE_NULL;
            cas_new.node.counter = node_head.node.counter + 1;
            if(atomic_compare_exchange_strong(pnode_head, &cas_expected.ulong, cas_new.ulong) != 0)
            {
                // Dequeue succeeded, next of the dequeued node has already been set to empty
                break;
            }
        }
        else if(node_head.node.next != head.node.next && node_head.node.counter == head.node.counter + 1)
        {
            // Multiple nodes are in the queue
            cas_expected.ulong   = head.ulong;
            cas_new.node.next    = node_head.node.next;
            cas_new.node.counter = head.node.counter + 1;
            if(atomic_compare_exchange_strong(&header->head, &cas_expected.ulong, cas_new.ulong) != 0)
            {
                // Dequeue succeeded, set next of the dequeued node to empty
                cas_new.node.next    = VALUE_NULL;
                cas_new.node.counter = node_head.node.counter + 1;
                atomic_store(pnode_head, cas_new.ulong);
                break;
            }
        }
        else
        {
            // Should never run to here
            NVLOGF(TAG, AERIAL_NVIPC_API_EVENT, "%s: dequeue error: tail:%d-%u head:%d-%u node_head:%d-%u ", priv_data->name, tail.node.next, tail.node.counter, head.node.next, head.node.counter, node_head.node.next, node_head.node.counter);
        }
    }

    long deq_count = atomic_fetch_add(&header->deq_count, 1);
    if(deq_count % PRINT_COUNTER == 0)
    {
        long enq_count = atomic_load(&header->enq_count);
        NVLOGI(TAG,
               "%s: dequeue try_max=%d counter: enq=%lu deq=%lu header_counter: enq=%u deq=%u available=%u",
               priv_data->name,
               priv_data->dequeue_try_max,
               enq_count,
               deq_count,
               tail.node.counter,
               head.node.counter,
               tail.node.counter - head.node.counter);
    }

    NVLOGV(TAG, "%s: dequeued value=%d tail:%d-%u head:%d-%u node_head:%d-%u ", priv_data->name, head.node.next, tail.node.next, tail.node.counter, head.node.next, head.node.counter, node_head.node.next, node_head.node.counter);

    return head.node.next;
}

static int array_queue_get_max_length(array_queue_t* queue)
{
    priv_data_t* priv_data = get_private_data(queue);
    return priv_data->queue_len;
}

static char* array_queue_get_name(array_queue_t* queue)
{
    priv_data_t* priv_data = get_private_data(queue);
    return priv_data->name;
}

static int array_queue_get_count(array_queue_t* queue)
{
    priv_data_t*          priv_data = get_private_data(queue);
    array_queue_header_t* header    = priv_data->header;

    long deq_count = atomic_load(&header->deq_count);
    long enq_count = atomic_load(&header->enq_count);
    long count     = enq_count - deq_count;
    if(count < 0)
    {
        count = 0;
    }
    else if(count > priv_data->queue_len)
    {
        count = priv_data->queue_len;
    }
    return (int)count;
}

int32_t array_queue_get_next(array_queue_t* queue, int32_t base)
{
    priv_data_t*          priv_data = get_private_data(queue);
    array_queue_header_t* header    = priv_data->header;

    cas_union_t   curr;
    if(base >= priv_data->queue_len)
    {
        return VALUE_NULL;
    }
    atomic_ulong* pnode_base = base < 0 ? &header->head : &header->queue[base];
    curr.ulong               = atomic_load(pnode_base);
    return curr.node.next;
}

static int array_queue_close(array_queue_t* queue)
{
    if(queue == NULL)
    {
        return -1;
    }

    priv_data_t* priv_data = get_private_data(queue);

    if(priv_data->name != NULL)
    {
        free(priv_data->name);
    }

    free(queue);

    NVLOGI(TAG, "%s: OK", __func__);
    return 0;
}

static int array_queue_init(array_queue_t* queue)
{
    priv_data_t* priv_data = get_private_data(queue);

    if(priv_data->primary)
    {
        cas_union_t init_val;
        init_val.node.counter = 0;
        init_val.node.next    = VALUE_NULL;
        atomic_init(&priv_data->header->head, init_val.ulong);
        atomic_init(&priv_data->header->tail, init_val.ulong);

        atomic_init(&priv_data->header->enq_count, 0);
        atomic_init(&priv_data->header->deq_count, 0);

        int32_t i;
        for(i = 0; i < priv_data->queue_len; i++)
        {
            atomic_init(&priv_data->header->queue[i], init_val.ulong);
        }
    }
    return 0;
}

int32_t array_queue_get_length(array_queue_t* queue)
{
    priv_data_t* priv_data = get_private_data(queue);
    return priv_data->queue_len;
}

int array_queue_reset(array_queue_t *queue)
{
    if (queue == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t *priv_data = get_private_data(queue);

    // Dequeue all existing elements
    for (int i = 0; i < priv_data->queue_len; i++)
    {
        if (array_queue_dequeue(queue) < 0)
        {
            break;
        }
    }
    return 0;
}

array_queue_t* array_queue_open(int primary, const char* name, void* header, int32_t length)
{
    int            size  = sizeof(array_queue_t) + sizeof(priv_data_t);
    array_queue_t* queue = malloc(size);
    if(queue == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed: name=%s", __func__, name);
        return NULL;
    }
    memset(queue, 0, size);

    priv_data_t* priv_data = get_private_data(queue);
    priv_data->primary     = primary;
    priv_data->queue_len   = length;
    priv_data->header      = header;

    queue->get_length = array_queue_get_length;
    queue->enqueue    = array_queue_enqueue;
    queue->dequeue    = array_queue_dequeue;
    queue->close      = array_queue_close;

    // Debug functions
    queue->get_name       = array_queue_get_name;
    queue->get_max_length = array_queue_get_max_length;
    queue->get_count      = array_queue_get_count;
    queue->get_next       = array_queue_get_next;

    // For debug log
    priv_data->enqueue_try_max = 0;
    priv_data->dequeue_try_max = 0;
    priv_data->name            = strdup(name);

    if(array_queue_init(queue) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: name=%s Failed", __func__, name);
        array_queue_close(queue);
        return NULL;
    }
    else
    {
        NVLOGI(TAG, "%s: name=%s length=%d OK", __func__, name, length);
        return queue;
    }
}
