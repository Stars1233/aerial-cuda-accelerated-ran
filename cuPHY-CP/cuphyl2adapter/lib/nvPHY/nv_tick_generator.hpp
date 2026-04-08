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

#if !defined(NV_TICK_GENERATOR_HPP_INCLUDED_)
#define NV_TICK_GENERATOR_HPP_INCLUDED_


#include "nv_phy_utils.hpp"
#include "nv_phy_epoll_context.hpp"
#include "yaml.hpp"
#include "cuphyoam.hpp"

#include <chrono>
#include <atomic>
#include <thread>
using namespace std;
using namespace std::chrono;

// #define dbg //Undefined so that it only prints when jitter occurs

//#define ABS(N) (std::abs(N))
#define ABS(N) (((N) < 0) ? (-(N)) : (N))
#define allowed_offset_nsec 100000
#define test_time_sec (20)
#define pre_window 200

#define nv_tick_thread_sleep_interval 80000

namespace nv
{
    class PHY_module;
    class tti_gen
    {
        
        public:
        explicit tti_gen(yaml::node config , PHY_module &m):
        module_(&m),
        current_ts(0ns),
        current_scheduled_ts(0ns)
        {
            if (config.has_key("timer_thread_config"))
            {
                timer_thread_cfg.reset(new thread_config);
                timer_thread_cfg->name = config["timer_thread_config"]["name"].as<std::string>();
                timer_thread_cfg->cpu_affinity =  config["timer_thread_config"]["cpu_affinity"].as<int>();
                timer_thread_cfg->sched_priority = config["timer_thread_config"]["sched_priority"].as<int>();
            }
            if (config.has_key("tick_generator_mode")) {
                tick_generator_mode = config["tick_generator_mode"].as<unsigned int>();
            }
            else
            {
                tick_generator_mode = 0;
            }
            thread_id = 0;
            window_nsec = 0;
        }

        tti_gen(const tti_gen&) = delete;
        tti_gen& operator=(const tti_gen&) = delete;

        tti_gen(tti_gen&& other):
        timer_thread(std::move(other.timer_thread)),
#ifdef dbg
        prev_tp(std::move(other.prev_tp)),
#endif
        timer_thread_cfg(std::move(other.timer_thread_cfg)),
        window_nsec(std::move(other.window_nsec)),
        tick_generator_mode(std::move(other.tick_generator_mode)),
        has_thread_cfg(std::move(other.has_thread_cfg)),
        started(std::move(other.started)),
        epoll_ctx(std::move(other.epoll_ctx)),
        timer_fd_p(std::move(other.timer_fd_p)),
        timer_mcb_p(std::move(other.timer_mcb_p)),
        current_ts(std::move(other.current_ts)),
        current_scheduled_ts(std::move(other.current_scheduled_ts)),
        thread_id(std::move(other.thread_id)),
        module_(std::move(other.module_))
        {
        }

        void set_module(PHY_module& m) { module_ = &m; } 

        void start_tick_generator();
        void stop_tick_generator();

        /**
         * Join the timer thread
         *
         * Blocks until the timer thread completes execution.
         */
        void timer_thread_join();

        private:
        uint64_t sys_clock_time_handler();
        int64_t get_first_slot_timestamp();
        void slot_indication_thread_poll_method();
        void slot_indication_thread_sleep_method();
        void slot_indication_thread_timer_fd_method();
        void slot_indication_handler();

        private:
        std::thread timer_thread; // timer thread
#ifdef dbg
        //Measure slot indication interval
        chrono::high_resolution_clock::time_point prev_tp;
#endif
        unique_ptr<thread_config> timer_thread_cfg;
        uint32_t      window_nsec;
        uint32_t      tick_generator_mode;
        volatile bool has_thread_cfg = true;
        volatile bool started        = false;

        phy_epoll_context                                epoll_ctx;
        unique_ptr<timer_fd>                             timer_fd_p;
        unique_ptr<member_event_callback<tti_gen>> timer_mcb_p;
        std::atomic<bool> stop_thread{false};
        nanoseconds current_ts;
        nanoseconds current_scheduled_ts;
        PHY_module* module_;
        pthread_t thread_id;
    };
}

#endif
