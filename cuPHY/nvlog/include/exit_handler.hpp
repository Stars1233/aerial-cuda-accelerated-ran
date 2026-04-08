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

#ifndef EXIT_HANDLER_H
#define EXIT_HANDLER_H

#include <cstdint>
#include <atomic>
#include <semaphore.h>
#include <pthread.h>


class exit_handler
{
    public:
        /* l1_state describes the states L1 for goes through. It is the overarching state of l1 for all cells
        L1 comes up in L1_RUNNING state
        L1_RUNNING -> L1_RECOVERY : cuPhy objects unavailable for > (aggr obj unavailability threshold) consecutive slots
        L1_RECOVERY -> L1_RUNNING : cuPhy objects for all channels become available for (aggr obj unavailability threshold)*2 consecutive slots:
        L1_RECOVERY -> L1_EXIT    : L1 in L1_RECOVERY for > 160 ms
        */
        typedef enum l1_state_{
            L1_RUNNING,
            L1_EXIT,
            L1_RECOVERY
        }l1_state;
        static exit_handler& getInstance() {
            if (instance == nullptr) {
                // Create instance by "new" to avoid automatically deconstructing when exiting the application
                instance = new exit_handler();
            }
            return *instance;
        }
        ~exit_handler();
    	exit_handler(exit_handler const &) = delete;
    	exit_handler& operator=(exit_handler const &) = delete;         
        void set_exit_handler_flag(l1_state val);
        void test_trigger_exit(const char* file, int line, const char* format_fmt);
        bool test_exit_in_flight();
        void set_exit_handler_cb(void (*exit_hdlr_cb)());
        uint32_t get_l1_state();
        int start_exit_watchdog_thread(int cpu_id);

    private:
        static exit_handler* instance;
        exit_handler();
        std::atomic<l1_state> exit_handler_flag;
        void (*exit_cb)() = nullptr;

        /** Exit watchdog: waits on semaphore; when posted, sleeps EXIT_WATCHDOG_SLEEP_SEC seconds then exit() as backup. */
        int exit_watchdog_cpu_core = 0; ///< CPU core number to run exit_watchdog thread
        pthread_t exit_watchdog_thread = 0; ///< Thread ID of exit_watchdog thread
        sem_t exit_watchdog_sem; ///< Semaphore to signal exit_watchdog thread to wake up
        static void* exit_watchdog_thread_func(void* arg); ///< Function to run in exit watchdog thread
        static const unsigned int EXIT_WATCHDOG_SLEEP_SEC; ///< Sleep seconds to wait for main thread to exit
};

#endif

