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

#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <sys/queue.h>
#include <sys/epoll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <numa.h>
#include <sys/sysinfo.h>

#include "test_common.h"
#include "test_ipc.h"
#define TAG (NVLOG_TAG_BASE_NVIPC + 28) // "NVIPC.TEST"

/***** Stress test multiply *********/
#define SEND_THREAD_NUM 10
#define RECV_THREAD_NUM 12

#define APP_NUMA_ID 0 // Use NUMA Node 0 by default

#define CPU_NOP_COUNT 1

task_info_t task_info;

// Detect how many CPU cores are available on the machine, and assign them to primary and secondary nvipc_cunit processes.
int detect_cpu_cores()
{
    cpu_set_t cpuset;

    // Get the affinity mask for the current process
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == -1)
    {
        perror("sched_getaffinity failed");
        return -1;
    }

    // Get the number of CPUs in the set
    int core_num = CPU_COUNT(&cpuset);

    // Get the core id for each core in the set
    int core_id = -1;
    for (int i = 0; i < core_num; i++) {
        do {
            core_id++;
        } while (!CPU_ISSET(core_id, &cpuset) && core_id < MAX_CPU_CORE_NUM);

        if (core_id < MAX_CPU_CORE_NUM) {
            task_info.system_cpu_cores[i] = core_id;
        } else {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: i=%d core_id=%d is out of range", __func__, i, core_id);
            return -1;
        }
    }

    // Reserve 4 cores 0~3 for system usage
    int app_cores = core_num - 4;
    int start_core = 4;

    // For SHM IPC, allocate a half of the cores to primary process and another half to secondary processes
    if (ipc_transport == NV_IPC_TRANSPORT_SHM) {
        app_cores /= 2;
        if (is_primary()) {
            start_core += app_cores;
        }
    }

    task_info.free_core_num = app_cores;
    for (int i = 0; i < app_cores; i++) {
        task_info.cpu_cores[i] = task_info.system_cpu_cores[start_core + i];
        NVLOGC(TAG, "%s: CPU core for %s app: i=%d core_id=%d",
                __func__, is_primary() ? "primary" : "secondary", i, task_info.cpu_cores[i]);
    }

    return 0;
}

void test_assign_cpu(void)
{
    assign_cpu_for_process(task_info.cpu_cores[task_info.free_core_num - 1]);
    task_info.free_core_num--;
}

void set_task_name(const char* task_name) {
    snprintf(task_info.task_name, TASK_NAME_MAX_LEN, "%s", task_name);
}

void print_thread_name(const char* info) {
    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);
    NVLOGC(TAG, "%s: thread_name=%s", info, thread_name);
}

void* thread_func_wrapper(void *arg) {
    thread_info_t *thread_info = (thread_info_t *) arg;
    NVLOGI(TAG, "%s: set thread %lu name to %s", __func__, pthread_self(), thread_info->name);

    if (pthread_setname_np(pthread_self(), thread_info->name) != 0) {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set thread name %s failed", __func__,
                thread_info->name);
    }
    if (thread_info->cpu_core >= 0) {
        int ret = assin_cpu_for_thread(thread_info->cpu_core);
        CU_ASSERT_EQUAL(ret, 0);
    }
    return thread_info->func(thread_info);
}

static void launch_task_threads(task_info_t* tasks)
{
    NVLOGC(TAG, "%s: start: send_threads_num=%d recv_threads_num=%d", __func__,
            tasks->send_threads_num, tasks->recv_threads_num);
    // print_thread_name("before create");
    int i = 0;
    while(i < tasks->send_threads_num || i < tasks->recv_threads_num)
    {
        if(i < tasks->send_threads_num)
        {
            thread_info_t* info = &tasks->send_threads[i];
            if(pthread_create(&info->pthread_id, NULL, thread_func_wrapper, info) != 0)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: pthread_create failed: i=%d", __func__, i);
            }
        }
        if(i < tasks->recv_threads_num)
        {
            thread_info_t* info = &tasks->recv_threads[i];
            if(pthread_create(&info->pthread_id, NULL, thread_func_wrapper, info) != 0)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: pthread_create failed: i=%d", __func__, i);
            }
        }
        i++;
    }
    // print_thread_name("after create");

    // Join
    for (i = 0; i < tasks->send_threads_num; i++) {
        thread_info_t *info = &tasks->send_threads[i];
        if (pthread_join(info->pthread_id, NULL) != 0) {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: pthread_join failed: i=%d", __func__, i);
        }
    }

    if (tasks->need_exit_msg) {
        // Send exit message to exit all receiving threads
        for (i = 0; i < tasks->recv_threads_num; i++) {
            send_msg_exit();
        }
    }

    for (i = 0; i < tasks->recv_threads_num; i++) {
        thread_info_t *info = &tasks->recv_threads[i];
        if (pthread_join(info->pthread_id, NULL) != 0) {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: pthread_join failed: i=%d", __func__, i);
        }
    }

    NVLOGC(TAG, "%s: finished", __func__);
}

int has_all_received() {
    return task_info.total_recv_count == atomic_load(&task_info.expected_total_count);
}

int64_t run_duplex_tasks(void* (*send_func)(void*), void* (*recv_func)(void*), void *arg,
        int send_threads_num, int recv_threads_num, long total_count, int need_exit_msg) {

    if (send_threads_num + recv_threads_num > task_info.free_core_num) {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: CPU core not enough: %d + %d > %d", __func__,
                send_threads_num, recv_threads_num, task_info.free_core_num);
        CU_ASSERT_FATAL(send_threads_num + recv_threads_num <= task_info.free_core_num);
        return 0;
    }

    sync_together("duplex_multi_thread_run start ...");

    struct timespec ts_start;
    nvlog_gettime_rt(&ts_start);

    atomic_store(&task_info.expected_total_count, total_count);

    NVLOGC(TAG, "%s: %s finished: total_count=%ld send_cores=%d recv_cores=%d", __func__,
            task_info.task_name, total_count, send_threads_num, recv_threads_num);

    task_info.need_exit_msg = need_exit_msg;
    task_info.send_threads_num = send_threads_num;
    task_info.recv_threads_num = recv_threads_num;

    int i = 0;
    for(i = 0; i < task_info.send_threads_num; i++)
    {
        task_info.send_threads[i].tid   = i;
        task_info.send_threads[i].func  = send_func;
        task_info.send_threads[i].arg   = arg;
        task_info.send_threads[i].count = total_count / task_info.send_threads_num;
        task_info.send_threads[i].cpu_core = task_info.cpu_cores[i];

        // Thread name length has to be <= 15 characters; i % 10000 keeps
        // the suffix to at most 4 digits, well within any realistic core count.
        snprintf(task_info.send_threads[i].name, 16, "task_send_%d", i % 10000);
    }
    if (task_info.send_threads_num > 0) {
        task_info.send_threads[0].count += total_count % task_info.send_threads_num;
    }

    for(i = 0; i < task_info.recv_threads_num; i++)
    {
        task_info.recv_threads[i].tid   = i;
        task_info.recv_threads[i].arg   = arg;
        task_info.recv_threads[i].func  = recv_func;
        task_info.recv_threads[i].count = total_count / task_info.recv_threads_num;
        task_info.recv_threads[i].cpu_core = task_info.cpu_cores[send_threads_num + i];

        // Thread name length has to be <= 15 characters; i % 10000 keeps
        // the suffix to at most 4 digits, well within any realistic core count.
        snprintf(task_info.recv_threads[i].name, 16, "task_recv_%d", i % 10000);
    }
    if (task_info.recv_threads_num > 0) {
        task_info.recv_threads[0].count += total_count % task_info.recv_threads_num;
    }

    launch_task_threads(&task_info);

    // Assert result
    long total_recv_count = 0;
    for(i = 0; i < task_info.recv_threads_num; i++)
    {
        total_recv_count += task_info.recv_threads[i].count;
    }

    int64_t ns = nvlog_get_interval(&ts_start);
    NVLOGC(TAG, "%s: %s finished: total_count=%ld total_time=%ldms average_time=%ldns", __func__,
            task_info.task_name, total_count, ns / 1000L / 1000L, ns / total_count);

    CU_ASSERT_EQUAL(total_recv_count, total_count);

    sync_together("duplex_multi_thread_run finished");

    return ns;
}

#define DEQUEUE_CACHE_LEN 10
void* lockless_queue_test(void* arg)
{
    thread_info_t* info = (thread_info_t*)arg;
    NVLOGC(TAG, "%s: start %s thread %d pthread_id=%ld loop_count=%lu", __func__, info->name,
            info->tid, info->pthread_id, info->count);

    array_queue_t* queue  = (array_queue_t*)info->arg;
    int32_t        length = queue->get_length(queue);
    int32_t        value[DEQUEUE_CACHE_LEN];

    long test_counter = 0, idle_count = 0;
    long i;

    for(i = 0; i < info->count; i++)
    {
        int j;
        for(j = 0; j < DEQUEUE_CACHE_LEN; j++)
        {
            while((value[j] = queue->dequeue(queue)) < 0)
            {
                cpu_nop_time(CPU_NOP_COUNT);
                idle_count++;
            }
            NVLOGV(TAG, "%s: dequeued j=%d value=%d", __func__, j, value[j]);
        }
        for(j = 0; j < DEQUEUE_CACHE_LEN; j++)
        {
            while(queue->enqueue(queue, value[j]) < 0)
            {
                cpu_nop_time(CPU_NOP_COUNT);
                idle_count++;
            }
            NVLOGV(TAG, "%s: enqueued j=%d value=%d", __func__, j, value[j]);
        }
        test_counter++;
    }
    // info->count = idle_count;

    NVLOGC(TAG, "%s: exit %s thread %d pthread_id=%ld", __func__, info->name, info->tid,
            info->pthread_id);
    return NULL;
}

#define TEST_QUEUE_LEN 1024

void lockless_queue_check_empty(array_queue_t* queue, int32_t count)
{
    int32_t i, value, ret = 0;
    int32_t queue_len = queue->get_length(queue);

    for(i = 0; i < count; i++)
    {
        if((value = queue->dequeue(queue)) >= 0)
        {
            CU_ASSERT(0);
            ret = -1;
        }
    }
    NVLOGC(TAG, "%s: count=%d ret=%d", __func__, count, ret);
}

void lockless_queue_enqueue(array_queue_t* queue, int32_t sequence[], int32_t count)
{
    int32_t i = 0, ret = 0;
    int32_t queue_len = queue->get_length(queue);

    while(i < queue_len && i < count)
    {
        if(queue->enqueue(queue, sequence[i]) < 0)
        {
            CU_ASSERT(0);
            ret = -1;
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: error i=%d value=%d", __func__, i, sequence[i]);
        }
        else
        {
            NVLOGD(TAG, "%s: enqueued i=%d value=%d", __func__, i, sequence[i]);
        }
        i++;
    }
    NVLOGC(TAG, "%s: count=%d ret=%d", __func__, count, ret);
}

void lockless_queue_dequeue(array_queue_t* queue, int32_t sequence[], int32_t count)
{
    int32_t i = 0, ret = 0;
    int32_t queue_len = queue->get_length(queue);

    while(i < queue_len && i < count)
    {
        int32_t value = -1;
        if((value = queue->dequeue(queue)) < 0)
        {
            CU_ASSERT(0);
            ret = -1;
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: error i=%d value=%d", __func__, i, sequence[i]);
        }
        else
        {
            CU_ASSERT_EQUAL(value, sequence[i]);
            NVLOGD(TAG, "%s: dequeued i=%d value=%d", __func__, i, value);
        }
        i++;
    }
    NVLOGC(TAG, "%s: count=%d ret=%d", __func__, count, ret);
}

void lockless_queue_basic(array_queue_t* queue)
{
    int32_t i, value, ret;
    int32_t queue_len = queue->get_length(queue);
    int32_t sequence[TEST_QUEUE_LEN];

    for(i = 0; i < queue_len; i++)
    {
        sequence[i] = i;
    }
    lockless_queue_dequeue(queue, sequence, queue_len);
    lockless_queue_check_empty(queue, 2);

    for(i = 0; i < queue_len; i++)
    {
        sequence[i] = queue_len - i - 1;
    }
    lockless_queue_enqueue(queue, sequence, queue_len);
    lockless_queue_dequeue(queue, sequence, queue_len);
    lockless_queue_check_empty(queue, 1);

    lockless_queue_enqueue(queue, sequence, 1);
    lockless_queue_dequeue(queue, sequence, 1);
    lockless_queue_check_empty(queue, 1);

    lockless_queue_enqueue(queue, sequence, 1);
    lockless_queue_dequeue(queue, sequence, 1);
    lockless_queue_enqueue(queue, sequence, 1);
    lockless_queue_dequeue(queue, sequence, 1);
    lockless_queue_check_empty(queue, 1);

    lockless_queue_enqueue(queue, sequence, 2);
    lockless_queue_dequeue(queue, sequence, 2);
    lockless_queue_check_empty(queue, 1);

    lockless_queue_enqueue(queue, sequence, 2);
    lockless_queue_dequeue(queue, sequence, 2);
    lockless_queue_check_empty(queue, 2);

    // Enqueue all for next test
    lockless_queue_enqueue(queue, sequence, queue_len);
    lockless_queue_dequeue(queue, sequence, 2);
    NVLOGC(TAG, "%s: finished", __func__);
}

void test_lockless_queue(void)
{
    int            ret = 0;
    array_queue_t* queue;
    int32_t        queue_len = TEST_QUEUE_LEN;
    char           name[32]  = "test_queue";

    int           size = ARRAY_QUEUE_HEADER_SIZE(queue_len);
    nv_ipc_shm_t* shmpool;

    sync_primary_first("crate test queue");

    if((shmpool = nv_ipc_shm_open(is_primary(), name, size)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: nv_ipc_shm_open error", __func__);
        ret = -1;
        return;
    }

    if((queue = array_queue_open(is_primary(), name, shmpool->get_mapped_addr(shmpool), queue_len)) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: array_queue_open error", __func__);
        ret = -1;
        shmpool->close(shmpool);
        return;
    }
    sync_primary_end("crate test queue");

    if(is_primary())
    {
        int i;
        for(i = 0; i < queue_len; i++)
        {
            if(queue->enqueue(queue, i) < 0)
            {
                NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: enqueue initial value %d failed", __func__, i);
                ret = -1;
                break;
            }
        }

        lockless_queue_basic(queue);
    }

    set_task_name(__func__);

    int send_thread_num = task_info.free_core_num / 2;
    int recv_thread_num = task_info.free_core_num - send_thread_num;
    run_duplex_tasks(lockless_queue_test, lockless_queue_test, queue, send_thread_num,
            recv_thread_num, 1L * 100 * 1000, 0);

    sync_secondary_first("close test queue");
    if(queue->close(queue) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: queue close error", __func__);
        ret = -1;
    }
    if(shmpool->close(shmpool) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: shmpool close error", __func__);
        ret = -1;
    }
    sync_secondary_end("close test queue");

    CU_ASSERT(ret == 0);
}
