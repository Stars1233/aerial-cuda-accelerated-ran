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

#include "nvlog.hpp"
#include "nvlog.h"
#include "exit_handler.hpp"
#include <string>
#include <unistd.h>
#include <cstdlib>

#define TAG (NVLOG_TAG_BASE_NVLOG + 10) // "NVLOG.EXIT_HANDLER"

// Define initial value of exit_handler::instance
exit_handler* exit_handler::instance = nullptr;

const unsigned int exit_handler::EXIT_WATCHDOG_SLEEP_SEC = 5;

void* exit_handler::exit_watchdog_thread_func(void* arg)
{
    exit_handler* self = static_cast<exit_handler*>(arg);

    nv_assign_thread_cpu_core(self->exit_watchdog_cpu_core);

    // Set thread name to "exit_watchdog"
    pthread_setname_np(pthread_self(), "exit_watchdog");
    NVLOGC_FMT(TAG, "[exit_watchdog] thread started on core {}", sched_getcpu());

    // Wait for the semaphore to be posted to wake up the exit watchdog thread
    if (sem_wait(&self->exit_watchdog_sem) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "sem_wait(&exit_watchdog_sem) failed: {}", std::strerror(errno));
    }

    NVLOGC_FMT(TAG, "[exit_watchdog] thread received exiting notification, wait for {} seconds to force exit", EXIT_WATCHDOG_SLEEP_SEC);

    // Sleep several seconds to let main thread exit
    sleep(EXIT_WATCHDOG_SLEEP_SEC);

    NVLOGC_FMT(TAG, "[exit_watchdog] thread closed FMT log after and force exit");

    // If the main thread is still not exited after watchdog sleep, print a warning log and force exit
    nvlog_fmtlog_close();

    char ts[NVLOG_TIME_STRING_LEN] = {'\0'};
    nvlog_gettimeofday_string(ts, NVLOG_TIME_STRING_LEN);
    printf("%s EXIT from watchdog after %u seconds waiting\n", ts, EXIT_WATCHDOG_SLEEP_SEC);

    // Force exit the application
    exit(EXIT_FAILURE);
    return nullptr;
}

int exit_handler::start_exit_watchdog_thread(int cpu_id)
{
    exit_watchdog_cpu_core = cpu_id;

    if (pthread_create(&exit_watchdog_thread, nullptr, exit_watchdog_thread_func, this) != 0) {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "pthread_create(&exit_watchdog_thread) failed: {}", std::strerror(errno));
        return -1;
    }

    // Detach exit watchdog thread to avoid zombie thread
    pthread_detach(exit_watchdog_thread);
    return 0;
}

exit_handler::exit_handler()
{
    exit_handler_flag=L1_RUNNING;

    // Initialize exit watchdog semaphore and thread
    if (sem_init(&exit_watchdog_sem, 0, 0) != 0) {
        printf("sem_init(&exit_watchdog_sem) failed: %s\n", std::strerror(errno));
    }
}

exit_handler::~exit_handler()
{
    exit_handler_flag=L1_RUNNING;

    // Destroy exit watchdog semaphore
    if (sem_destroy(&exit_watchdog_sem) != 0) {
        printf("sem_destroy(&exit_watchdog_sem) failed: %s\n", std::strerror(errno));
    }
}

void exit_handler::set_exit_handler_flag(l1_state val)
{
    exit_handler_flag = val;

    if (val == L1_EXIT)
    {
        // Post the semaphore to wake up the exit watchdog thread to force exit the application after waiting
        sem_post(&exit_watchdog_sem);
    }
}

uint32_t exit_handler::get_l1_state()
{
    return exit_handler_flag;
}

void exit_handler::test_trigger_exit(const char* file, int line, const char* info)
{
    // Note: DO NOT call NVLOGF_FMT() in this function to avoid nested calls.

    // Atomic fetch and set
    l1_state curr_state = exit_handler_flag.exchange(L1_EXIT);
    bool old_flag = (curr_state == L1_EXIT) ? true : false;
    sem_post(&exit_watchdog_sem);
    char ts[NVLOG_TIME_STRING_LEN] = {'\0'};
    nvlog_gettimeofday_string(ts, NVLOG_TIME_STRING_LEN);

    int cpu_id = sched_getcpu();

    // Get thread name and CPU core number
    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);

    // Only one thread runs the exit callback and call exit(EXIT_FAILURE), other threads just wait.
    if(!old_flag)
    {
        NVLOG_FMT_EVT(fmtlog::FAT, TAG, AERIAL_SYSTEM_API_EVENT, "FATAL exit: Thread [{}] on core {} file {} line {}: additional info: {}", thread_name, cpu_id, file, line, info);
        printf("%s FATAL exit: Thread [%s] on core %d file %s line %d: additional info: %s\n", ts, thread_name, cpu_id, file, line, info);

        if (exit_cb != nullptr)
        {
            // exit_cb performs application cleanup and exit
            exit_cb();
            NVLOGC_FMT(TAG, "{}: exit_cb called", __func__);
        }
    }
    else
    {
        NVLOG_FMT_EVT(fmtlog::FAT, TAG, AERIAL_SYSTEM_API_EVENT, "FATAL already exiting: Thread [{}] on core {} file {} line {}: additional info: {}", thread_name, cpu_id, file, line, info);
        printf("%s FATAL already exiting: Thread [%s] on core %d file %s line %d: additional info: %s\n", ts, thread_name, cpu_id, file, line, info);
    }

    // Exit the caller thread to avoid running following code after NVLOGF_FMT() or other exiting code
    pthread_exit(NULL);
}

bool exit_handler::test_exit_in_flight()
{
    return ((exit_handler_flag==L1_EXIT)?true:false);
}

void exit_handler::set_exit_handler_cb(void (*exit_hdlr_cb)())
{
    exit_cb=exit_hdlr_cb;
}