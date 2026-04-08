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

#include "nv_tick_generator.hpp"
#include "nv_phy_module.hpp"
#include "nvlog.hpp"
#include "memtrace.h"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 7) // "L2A.TICK"

namespace nv
{
    constexpr int64_t FRAME_PERIOD = 10000000LL;
    constexpr int64_t SFN_MAX = 1024LL;
    constexpr int64_t SFN_PERIOD = SFN_MAX * FRAME_PERIOD;
    constexpr int64_t TAI_GPS_EPOCH_DELTA = 315964800ULL; //(Jan 6th 1980(GPS epoch) - Jan 1st 1970 (TAI epoch)) /*(365x10 + 2(2 leap years)+5(5 additional days in 1980))x24x60x60*/
    constexpr int64_t GPS_TO_TAI_LAG = 19ULL; //GPS lags TAI by 19s
    static std::atomic<bool> thread_started{false};
    static std::atomic<bool> timer_thread_start{false};
    void tti_gen::start_tick_generator()
    {
        window_nsec = nv::mu_to_ns(module_->get_mu_highest());
        NVLOGI_FMT(TAG,"{} tick_generator_mode={} window_nsec={}", __FUNCTION__, tick_generator_mode, window_nsec);

        bool expected = false;
        if (thread_started.compare_exchange_strong(expected, true)) {
            std::thread t;

            if (tick_generator_mode == 0)
            {
                t = std::thread(&tti_gen::slot_indication_thread_poll_method, this);
            }
            else if (tick_generator_mode == 1)
            {
                MemtraceDisableScope md; // at the cell start only. Only once at init
                t = std::thread(&tti_gen::slot_indication_thread_sleep_method, this);
            }
            else if (tick_generator_mode == 2)
            {
                t = std::thread(&tti_gen::slot_indication_thread_timer_fd_method, this);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Error configuration: tick_generator_mode={}", tick_generator_mode);
                thread_started.store(false);
                return;
            }

            thread_id = t.native_handle();
            timer_thread.swap(t);

            int name_st = pthread_setname_np(timer_thread.native_handle(), "timer_thread");

            if (name_st != 0 )
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT ,"Timer Thread pthread_setname_np failed with status: {}",std::strerror(name_st));
            }
            sched_param sch;
            int         policy;
            int         status = 0;
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Set thread priority
            status = pthread_getschedparam(timer_thread.native_handle(), &policy, &sch);
            if(status != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "timer_thread pthread_getschedparam failed with status : {}", std::strerror(status));
            }
            sch.sched_priority = timer_thread_cfg->sched_priority;

            status = pthread_setschedparam(timer_thread.native_handle(), SCHED_FIFO, &sch);
            if(status != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "timer_thread setschedparam failed with status : {}" , std::strerror(status));
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Set thread CPU affinity
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(timer_thread_cfg->cpu_affinity, &cpuset);
            status = pthread_setaffinity_np(timer_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "timer_thread setaffinity_np  failed with status : {}" , std::strerror(status));
            }
            timer_thread_start.store(true);
        }
    }

    void tti_gen::stop_tick_generator()
    {
        NVLOGC_FMT(TAG, "Stopping TTI Generator timer thread");
        stop_thread.store(true);

        // Stop epoll event loop for timer_fd method (mode 2)
        if (tick_generator_mode == 2) {
            epoll_ctx.terminate();
        }
    }

    /**
     * Join the timer thread
     *
     * Blocks until the timer thread completes execution.
     */
    void tti_gen::timer_thread_join()
    {
        if (timer_thread.joinable()) {
            timer_thread.join();
        }
    }

    inline uint64_t tti_gen::sys_clock_time_handler()
    {
        using namespace std::chrono;
        current_ts = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
        return current_ts.count();
    }

    int64_t tti_gen::get_first_slot_timestamp()
    {
        uint64_t curr = sys_clock_time_handler();

#if 1
        uint64_t next_expected;
        uint64_t tia_to_gps_offset_ns = 0;
        int64_t  gps_offset = 0;
        int64_t  gps_sfn_offset = 0;
        uint64_t gps_curr = 0;
        uint64_t gps_next_sfn_0 = 0;
        int64_t gps_next_sfn_0_offset = 0;

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        phyDriver.l1_update_gps_alpha_beta(module_->gps_alpha(), module_->gps_beta());
        NVLOGC_FMT(TAG, "PTP Configs: gps_alpha: {} gps_beta: {}", module_->gps_alpha(), module_->gps_beta());
        NVLOGD_FMT(TAG, "gps_alpha offset: {} gps_beta offset: {}", (module_->gps_alpha() * 10000ULL) / 12288ULL, (module_->gps_beta() * 10000000ULL));
        gps_offset = (module_->gps_beta() * 10000000ULL) + ((module_->gps_alpha() * 10000ULL) / 12288ULL);
        tia_to_gps_offset_ns = (TAI_GPS_EPOCH_DELTA + GPS_TO_TAI_LAG) * 1000000000ULL;

        NVLOGD_FMT(TAG, "TAI offset: {}", AppConfig::getInstance().getTaiOffset());

        gps_curr = curr + AppConfig::getInstance().getTaiOffset() - tia_to_gps_offset_ns;
        // Get next SFN 0 time in GPS scale for alpha=0 beta=0
        gps_next_sfn_0 = gps_curr;
        gps_next_sfn_0 /= SFN_PERIOD;
        gps_next_sfn_0++;
        gps_next_sfn_0 *= SFN_PERIOD;
        // Accomodated by adding SFN periods to next SFN 0 with beta and alpha offset.
        gps_next_sfn_0_offset = gps_next_sfn_0 - gps_curr;

        // Add GPU Offset that will be reverted in nv_phy_module so that we start at SFN 0 still
        gps_next_sfn_0_offset += gps_offset % SFN_PERIOD;

        while(gps_next_sfn_0_offset < 0)
        {
            gps_next_sfn_0_offset += SFN_PERIOD;
        }

        next_expected = curr + (gps_next_sfn_0_offset % SFN_PERIOD);

        NVLOGC_FMT(TAG,"Start time: tick={}, seconds since epoch = {}, nanoseconds = {}",
             curr,
             curr / 1000000000ULL,
             curr % 1000000000ULL);

        NVLOGC_FMT(TAG,"FIRST tick scheduled for: tick={}, seconds since epoch = {}, nanoseconds = {}",
             next_expected,
             next_expected / 1000000000ULL,
             next_expected % 1000000000ULL);
        return next_expected;
#else
        return curr + window_nsec;
#endif
    }

    void set_timespec(struct timespec *ts, uint64_t epoch)
    {
        ts->tv_sec = epoch / (1000ULL * 1000 * 1000);
        ts->tv_nsec = epoch % (1000ULL * 1000 * 1000);
    }

    void tti_gen::slot_indication_thread_sleep_method()
    {
        while(!timer_thread_start.load())
        {
            usleep(1000);
        }
        nvlog_fmtlog_thread_init("timer_thread");
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);
        // enable dynamic memory allocation tracing in real-time code path
        // Use LD_PRELOAD=<special .so> when running cuphycontroller, otherwise this does nothing.
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

        struct timespec ts_expected, ts_remain;
        if(module_->get_enable_se_sync_cmd()){
            while(1)
            {
                //Sync command check for Spec Effeciency feature
                if(!module_->get_sfn_slot_sync_cmd_sent()){
                    if(module_->send_sfn_slot_sync_grpc_command())
                    {
                        module_->set_sfn_slot_sync_cmd_sent(false); //Failure
                        sleep(3); //Sleep for 3s before retrying
                        continue;
                    }
                    else
                    {
                        module_->set_sfn_slot_sync_cmd_sent(true); //Success
                    }
                }
                if(module_->get_target_node()==0) //Target:UE, Source : DU
                {
                    if(!module_->check_sync_rcvd_from_ue()) //Skip till SFN/slot sync command is received from Target Node (UE)
                    {
                        usleep(100000);
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                else //Target:DU, Source : UE
                {
                    if(!module_->check_sync_rcvd_from_du()) //Skip till SFN/slot sync command is received from Target Node (DU)
                    {
                        usleep(100000);
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        uint64_t next_expected = get_first_slot_timestamp();
        module_->set_first_tick(true);
        while(!stop_thread.load())
        {
            current_scheduled_ts = nanoseconds(next_expected);

            // Sleep to absolute time stamp
            set_timespec(&ts_expected, next_expected);
            int ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts_expected, &ts_remain);
            if(ret != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_nanosleep returned error ret: {}", ret);
            }
            // Call TTI handler to send SLOT.indication
            slot_indication_handler();

            // Add time interval to next expected slot
            next_expected += window_nsec;
        }

        NVLOGC_FMT(TAG, "{}: thread timer_thread exiting", __FUNCTION__);
    }

    void tti_gen::slot_indication_thread_poll_method()
    {
        // enable dynamic memory allocation tracing in real-time code path
        // Use LD_PRELOAD=<special .so> when running cuphycontroller, otherwise this does nothing.
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

        nvlog_fmtlog_thread_init("timer_thread");
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

#ifdef dbg
        if(has_thread_cfg)
        {
            sched_param sch;
            int         policy;
            int         status = 0;
            status             = pthread_getschedparam(pthread_self(), &policy, &sch);
            if(status == 0)
            {
                NVLOGD_FMT(TAG, "slot_indication_thread_poll_method sched_priority {}",sch.sched_priority);
                NVLOGD_FMT(TAG, "slot_indication_thread_poll_method on CPU {}" , sched_getcpu());
            }
            else
            {
                NVLOGD_FMT(TAG, "pthread_getschedparam failed with status: {}", std::strerror(status));
            }
        }
#endif

        assign_thread_cpu_core(timer_thread_cfg->cpu_affinity);

        bool     first = true;
        uint64_t last_actual =0;
        uint64_t next_expected =0;
        uint64_t first_time =0;
        uint64_t runtime =0;
        uint64_t count          = 0;
        uint64_t sum_abs_offset = 0;
        int32_t  min_offset     = 0;
        int32_t  max_offset     = 0;
        int32_t  max_abs_offset = 0;
        uint64_t slot_indication_start = 0;
        uint64_t slot_indication_end = 0;

        uint64_t tia_to_gps_offset_ns = 0;
        int64_t  gps_offset = 0;
        int64_t  gps_sfn_offset = 0;
        uint64_t gps_curr = 0;
        uint64_t gps_next_sfn_0 = 0;
        int64_t gps_next_sfn_0_offset = 0;
#ifdef dbg
        int32_t  debug_count = 0;
#endif
        //window_nsec  = nv::mu_to_ns(1);
        window_nsec  = nv::mu_to_ns(module_->get_mu_highest());

        while(!stop_thread.load())
        {
            uint64_t curr = sys_clock_time_handler();
            if(first)
            {
                first_time    = curr;
                last_actual   = curr;
                first         = false;

                if (module_->tickDynamicSfnSlotIsEnabled())
                {
                    NVLOGC_FMT(TAG, "PTP Configs: gps_alpha: {} gps_beta: {}", module_->gps_alpha(), module_->gps_beta());
                    NVLOGD_FMT(TAG, "gps_alpha offset: {} gps_beta offset: {}", (module_->gps_alpha() * 10000ULL) / 12288ULL, (module_->gps_beta() * 10000000ULL));
                    gps_offset = (module_->gps_beta() * 10000000ULL) + ((module_->gps_alpha() * 10000ULL) / 12288ULL);
                    tia_to_gps_offset_ns = (TAI_GPS_EPOCH_DELTA + GPS_TO_TAI_LAG) * 1000000000ULL;

                    gps_curr = curr - tia_to_gps_offset_ns;
                    // Get next SFN 0 time in GPS scale for alpha=0 beta=0
                    gps_next_sfn_0 = gps_curr;
                    gps_next_sfn_0 /= SFN_PERIOD;
                    gps_next_sfn_0++;
                    gps_next_sfn_0 *= SFN_PERIOD;
                    // Accomodated by adding SFN periods to next SFN 0 with beta and alpha offset.
                    gps_next_sfn_0_offset = gps_next_sfn_0 - gps_curr;

                    // Add GPU Offset that will be reverted in nv_phy_module so that we start at SFN 0 still
                    gps_next_sfn_0_offset += gps_offset % SFN_PERIOD;

                    while(gps_next_sfn_0_offset < 0)
                    {
                        gps_next_sfn_0_offset += SFN_PERIOD;
                    }

                    next_expected = curr + (gps_next_sfn_0_offset % SFN_PERIOD);

                    NVLOGI_FMT(TAG,"Start time: tick={}, seconds since epoch = {}, nanoseconds = {}",
                         curr,
                         curr / 1000000000ULL,
                         curr % 1000000000ULL);

                    NVLOGI_FMT(TAG,"FIRST tick scheduled for: tick={}, seconds since epoch = {}, nanoseconds = {}",
                         next_expected,
                         next_expected / 1000000000ULL,
                         next_expected % 1000000000ULL);
                }
                else
                {
                    next_expected = curr + window_nsec;
                }
                continue;
            }

            if(curr < last_actual)
            {
                NVLOGW_FMT(TAG, "error curr {} last {}", curr , last_actual);
            }

            int64_t diff = next_expected - curr;
            if(diff >= pre_window)
            {
                continue;
            }


#ifdef dbg
            uint64_t a = sys_clock_time_handler();
#endif
            current_scheduled_ts = nanoseconds(next_expected);
            slot_indication_start = sys_clock_time_handler();
            slot_indication_handler();
            slot_indication_end = sys_clock_time_handler();

            int32_t  offset       = (int32_t)((int64_t)curr - (int64_t)next_expected);
            int32_t  abs_offset   = ABS(offset);
            // stats
            if(offset < min_offset)
                min_offset = offset;
            if(offset > max_offset)
                max_offset = offset;
            if(abs_offset > max_abs_offset)
                max_abs_offset = abs_offset;
            count += 1;
            sum_abs_offset += (uint64_t)abs_offset;

#ifdef dbg
            NVLOGD_FMT(TAG, "{} {} {} {} {}", count , offset , abs_offset , sum_abs_offset, sum_abs_offset / count);
#endif
            runtime = curr - first_time;

            if(abs_offset > allowed_offset_nsec)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "jitter error, offset {} slot_indication_handler time: {}", offset, slot_indication_end - slot_indication_start);
            }

#ifdef dbg
            NVLOGI_FMT(TAG, "curr: {}, diff:  {}, offset:  {}" , curr , diff , offset);
#endif
            next_expected += window_nsec;
            last_actual   = curr;

#ifdef dbg
            uint64_t b = sys_clock_time_handler();
#endif

	    /*
            * It is found that the fixed-high-priority thread with SCHED_FIFO scheduling policy keeps running
            * without sleep can have unexpected side effects to the system(e.g., blocking other threads/processes
            * from running, the long existing "jitter error" we encountered). So here we yield the cpu for a short
            * window for it to perform housekeeping activities. The sleep interval can be configured intelligently
            * according to current numerology. For now, it is hard coded.
            */
            std::this_thread::sleep_for(std::chrono::nanoseconds(nv_tick_thread_sleep_interval));

#ifdef dbg
            uint64_t c = sys_clock_time_handler();

            int32_t  diff_1       = (int32_t)(b - a);
            int32_t  diff_2       = (int32_t)(c - b);

            if(++debug_count == 2000) {
                debug_count = 0;
                NVLOGI_FMT(TAG, "diff_1: {}, diff_2: {}" ,diff_1 ,diff_2 );
            }
#endif
        }

        // print stats
        NVLOGD_FMT(TAG, "total run time: {} sec",runtime / (1000000000));
        NVLOGD_FMT(TAG, "event count:    {}",count);
        NVLOGD_FMT(TAG, "sum:            {}",sum_abs_offset);
        NVLOGD_FMT(TAG, "min offset:     {}",min_offset);
        NVLOGD_FMT(TAG, "max offset:     {}",max_offset);
        NVLOGD_FMT(TAG, "max abs offset: {}",max_abs_offset);
        if(count > 0)
        {
            NVLOGD_FMT(TAG, "avg abs offset: {}",sum_abs_offset / count);
        }
        else
        {
            NVLOGD_FMT(TAG, "avg abs offset: N/A (no events)");
        }
    }


    void tti_gen::slot_indication_handler()
    {
#ifdef dbg
        auto   cur_tp = chrono::high_resolution_clock::now();
        double delay  = chrono::duration_cast<chrono::microseconds>(cur_tp - prev_tp).count();
        NVLOGI_FMT(TAG, "slot indication interval is : {} micro secs",delay);
        prev_tp = cur_tp;
#endif

        simulated_cpu_stall_checkpoint(L2A_TICK_THREAD,-1);
        module_->tick_received(current_scheduled_ts);

        if (tick_generator_mode == 2)
        {
            // clear timer fd signal
            timer_fd_p->clear();
        }
    }

    void tti_gen::slot_indication_thread_timer_fd_method()
    {
        // enable dynamic memory allocation tracing in real-time code path
        // Use LD_PRELOAD=<special .so> when running cuphycontroller, otherwise this does nothing.
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

        nvlog_fmtlog_thread_init("timer_thread");
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

#ifdef dbg
        if(has_thread_cfg)
        {
            sched_param sch;
            int         policy;

            int status = 0;
            status     = pthread_getschedparam(pthread_self(), &policy, &sch);
            if(status == 0)
            {
                NVLOGD_FMT(TAG, "slot_indication_thread_timer_fd_method sched_priority {}", sch.sched_priority);
                NVLOGD_FMT(TAG, "slot_indication_thread_timer_fd_method on CPU {}", sched_getcpu());
            }
            else
            {
                NVLOGD_FMT(TAG, "pthread_getschedparam failed with status: {}", std::strerror(status));
            }
        }
#endif

        unique_ptr<member_event_callback<tti_gen>> mcb_p(new member_event_callback<tti_gen>(this, &tti_gen::slot_indication_handler));
        unique_ptr<timer_fd>                             fd_p(new timer_fd(nv::mu_to_ns(1), true)); /// hard code for now
        //unique_ptr<timer_fd> fd_p(new timer_fd(500000, true));

        epoll_ctx.add_fd(fd_p->get_fd(), mcb_p.get());

        timer_fd_p  = std::move(fd_p);
        timer_mcb_p = std::move(mcb_p);
        try
        {
            epoll_ctx.start_event_loop();
        }
        catch(std::exception& e)
        {
            NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                    "tti_gen::slot_indication_thread_timer_fd_method() exception: {}\n",
                    e.what());
        }
        catch(...)
        {
            NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                    "tti_gen::slot_indication_thread_timer_fd_method() unknown exception\n");
        }
    }
}
