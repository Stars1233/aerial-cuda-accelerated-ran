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

#include <iostream>
#include <fstream>

#include "app_config.hpp"
#include "nv_ipc_ring.h"
#include "cumac_handler.hpp"
#include "oran_utils/conversion.hpp"

using namespace std;
using namespace std::chrono;

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 22) // "CUMAC.HANDLER"

#define CHECK_PTR_NULL_FATAL(ptr)                                                                                  \
    do                                                                                                             \
    {                                                                                                              \
        if((ptr) == nullptr)                                                                                       \
        {                                                                                                          \
            NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: pointer {} is nullptr", __func__, __LINE__, #ptr); \
        }                                                                                                          \
    } while(0);

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                              \
    do                                                                                                                             \
    {                                                                                                                              \
        if((v1) != (v2))                                                                                                           \
        {                                                                                                                          \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values doesn't equal: v1={} > v2={}", __func__, __LINE__, v1, v2); \
        }                                                                                                                          \
    } while(0);

#define CHECK_BUF_SIZE_ERR(size, max)                                                                                                                                                      \
    do                                                                                                                                                                                     \
    {                                                                                                                                                                                      \
        if((size) > (max))                                                                                                                                                                 \
        {                                                                                                                                                                                  \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: buffer size check failed: size={} > max={}", __func__, __LINE__, static_cast<uint32_t>(size), static_cast<uint32_t>(max)); \
        }                                                                                                                                                                                  \
    } while(0);

// If defined, force L2 to have an approx total scheduling time
// NB: This only works correctly for PTP GPS_ALPHA=GPS_BETA=0
// NB: See "schedule_time->set_limit(schedule_time, ...)" in cumac_handler.cpp if changing this
static constexpr long SLOT_TIME_BOUNDARY_NS = 500000;
static constexpr long SLOT_ADVANCE = 3; // TODO: If L2 knows this, use the variable instead of hard-coding

static cumac_handler* cumac_handler_instance = nullptr;

template <typename T>
static T* cumac_init_msg_header(nv_ipc_msg_t* msg, int msg_id, int cell_id)
{
    size_t msg_size = sizeof(T);
    msg->msg_id = msg_id;
    msg->cell_id = cell_id;
    msg->msg_len = msg_size;
    msg->data_len = 0;

    cumac_msg_header_t *header = (cumac_msg_header_t*) msg->msg_buf;
    header->message_count = 1;
    header->handle_id = cell_id;
    header->type_id = msg_id;
    header->body_len = msg_size - sizeof(cumac_msg_header_t);
    return reinterpret_cast<T*>(msg->msg_buf);
}

cumac_handler* get_cumac_handler_instance()
{
    return cumac_handler_instance;
}

static void reconfig_timer_handler(__sigval_t sigv)
{
    int cell_id = sigv.sival_int;
    NVLOGI_FMT(TAG, "{}: cell_id={}", __func__, cell_id);

    cumac_handler* _handler = get_cumac_handler_instance();
    _handler->reconfig(cell_id);
}

int cumac_handler::get_pending_config_cell_id()
{
    for (int cell_id = 0; cell_id < cell_num; cell_id ++)
    {
        if(cumac_cell_data[cell_id].pending_config.fetch_and(0) != 0)
        {
            return cell_id;
        }
    }
    return -1;
}

int cumac_handler::reconfig(int cell_id)
{
    stop_reconfig_timer(cell_id);

    if (config_retry_counter.load() > 0)
    {
        NVLOGC_FMT(TAG, "cell_config: cell {} timeout, retry config ...", cell_id);
        send_config_request(cell_id);
    }
    else
    {
        // Reset current cell config status and config next pending cell if exist
        current_config_cell_id.store(-1);
        int next_pending_cell = get_pending_config_cell_id();
        if (next_pending_cell >= 0)
        {
            send_config_request(next_pending_cell);
        }

    }
    return 0;
}

int cumac_handler::start_reconfig_timer(int cell_id, uint32_t interval_ms)
{
    NVLOGI_FMT(TAG, "cell_config: start timer: cell_id={} delay={}ms", cell_id, interval_ms);

    struct sigevent   sigev;
    struct itimerspec its;
    memset(&sigev, 0, sizeof(struct sigevent));
    sigev.sigev_notify          = SIGEV_THREAD;
    sigev.sigev_value.sival_int = cell_id;
    sigev.sigev_notify_function = reconfig_timer_handler;

    its.it_interval.tv_sec  = interval_ms / 1000;
    its.it_interval.tv_nsec = (interval_ms % 1000) * 1000 * 1000;
    its.it_value.tv_sec     = its.it_interval.tv_sec;
    its.it_value.tv_nsec    = its.it_interval.tv_nsec;

    timer_t* timer = &reconfig_timer;
    if(timer_create(CLOCK_REALTIME, &sigev, timer) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_create failed", __func__);
        return -1;
    }

    if(timer_settime(*timer, 0, &its, NULL) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_settime failedError: {}", __func__, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: delay={} OK", __func__, interval_ms);
        return 0;
    }
}

int cumac_handler::stop_reconfig_timer(int cell_id)
{
    // Delete the timer
    if(reconfig_timer != nullptr && timer_delete(reconfig_timer) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_delete failed cell_id={}", __func__, cell_id);
        reconfig_timer = nullptr;
    }
    NVLOGI_FMT(TAG, "cell_config: stop timer: cell_id={}", cell_id);
    return 0;
}

static void restart_timer_handler(__sigval_t sigv)
{
    int cell_id = sigv.sival_int;
    NVLOGC_FMT(TAG, "{}: cell_id={}", __func__, cell_id);

    cumac_handler* _handler = get_cumac_handler_instance();
    _handler->restart(cell_id);
}

int cumac_handler::restart(int cell_id)
{
    // Delete the timer
    timer_t* timer = cell_id == CELL_ID_ALL ? &restart_timer : &cumac_cell_data[cell_id].restart_timer;
    if(*timer != nullptr && timer_delete(*timer) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_delete failed cell_id={}", __func__, cell_id);
        *timer = nullptr;
    }

    if (cell_id != CELL_ID_ALL)
    {
        cell_init(cell_id);
    }
    else
    {
        for(int cell_id = 0; cell_id < get_cell_num(); cell_id++)
        {
            cell_init(cell_id);
        }
    }
    return 0;
}

int cumac_handler::set_restart_timer(int cell_id, int interval)
{
    if(interval <= 0 || (cell_id != CELL_ID_ALL && cell_id >= cumac_cell_data.size()))
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid cell_id={} interval={}", cell_id, interval);
        return -1;
    }

    NVLOGC_FMT(TAG, "{}: delay={}", __func__, interval);

    struct sigevent   sigev;
    struct itimerspec its;
    memset(&sigev, 0, sizeof(struct sigevent));
    sigev.sigev_notify          = SIGEV_THREAD;
    sigev.sigev_value.sival_int = cell_id;
    sigev.sigev_notify_function = restart_timer_handler;

    its.it_interval.tv_sec  = interval;
    its.it_interval.tv_nsec = 0;
    its.it_value.tv_sec     = interval;
    its.it_value.tv_nsec    = 0;

    timer_t* timer = cell_id == CELL_ID_ALL ? &restart_timer : &cumac_cell_data[cell_id].restart_timer;
    if(timer_create(CLOCK_REALTIME, &sigev, timer) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_create failed", __func__);
        return -1;
    }

    if(timer_settime(*timer, 0, &its, NULL) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: timer_settime failedError: {}", __func__, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: delay={} OK", __func__, interval);
        return 0;
    }
}

static void* cumac_receiver_thread_func(void* arg)
{
    cumac_handler *_cumac_handler = reinterpret_cast<cumac_handler*>(arg);
    config_thread_property(_cumac_handler->get_cumac_configs()->get_recv_thread_config());

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    test_cumac_configs* cumac_cfgs = _cumac_handler->get_cumac_configs();
    yaml::node cumac_root = _cumac_handler->get_cumac_configs()->get_cumac_yaml_root();
    phy_mac_transport* transp = new phy_mac_transport(cumac_root["transport"], NV_IPC_MODULE_MAC, cumac_cfgs->cumac_cell_num);
    _cumac_handler->setTransport(transp);

    cumac_cfgs->set_max_msg_size(nv_ipc_get_buf_size(transp->get_nv_ipc_config(), NV_IPC_MEMPOOL_CPU_MSG));
    cumac_cfgs->set_max_data_size(nv_ipc_get_buf_size(transp->get_nv_ipc_config(), NV_IPC_MEMPOOL_CPU_DATA));

    NVLOGC_FMT(TAG, "NVIPC instance created: max_msg_size={} max_data_size={}", cumac_cfgs->get_max_msg_size(), cumac_cfgs->get_max_data_size());

    _cumac_handler->receiver_thread_func();
    return nullptr;
}

static void* cumac_scheduler_thread_func(void* arg)
{
    cumac_handler *_cumac_handler = reinterpret_cast<cumac_handler*>(arg);
    config_thread_property(_cumac_handler->get_cumac_configs()->get_sched_thread_config());

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    _cumac_handler->scheduler_thread_func();
    return nullptr;
}

static void* cumac_builder_thread_func(void* arg)
{
    cumac_handler *_cumac_handler = reinterpret_cast<cumac_handler*>(arg);
    config_thread_property(_cumac_handler->get_cumac_configs()->get_builder_thread_config());

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    _cumac_handler->builder_thread_func();
    return nullptr;
}

static void* cumac_worker_thread_func(void* arg)
{
    cumac_handler *_cumac_handler = reinterpret_cast<cumac_handler*>(arg);
    _cumac_handler->worker_thread_func();
    return nullptr;
}

static void* cumac_tick_thread_func(void* arg)
{
    cumac_handler* _cumac_handler = reinterpret_cast<cumac_handler*>(arg);

    struct nv::thread_config tick_cfg;
    tick_cfg.name           = "cumac_tick";
    tick_cfg.sched_priority = 99;
    tick_cfg.cpu_affinity   = _cumac_handler->get_cumac_configs()->get_recv_thread_config().cpu_affinity;
    config_thread_property(tick_cfg);

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    struct timespec ts_expected, ts_remain;
    nvlog_gettime_rt(&ts_expected);
    ts_expected.tv_sec++;
    ts_expected.tv_nsec = 0;

    // Set init SFN/SLOT
    sched_info_t sched_info;
    sched_info.ss.u16.sfn = 0;
    sched_info.ss.u16.slot = 0;

    while(1)
    {
        // Sleep to absolute time stamp
        int ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts_expected, &ts_remain);
        if(ret != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_nanosleep returned error ret: {}", ret);
        }

        // Call TTI handler to send SLOT.indication
        sched_info.ts = ts_expected.tv_sec * 1e9 + ts_expected.tv_nsec;
        _cumac_handler->on_tick_event(sched_info);

        // Add time interval to next expected slot
        ts_expected.tv_nsec += SLOT_INTERVAL;
        if(ts_expected.tv_nsec >= 1e9)
        {
            ts_expected.tv_sec++;
            ts_expected.tv_nsec -= 1e9;
        }
        sched_info.ss = _cumac_handler->get_next_sfn_slot(sched_info.ss);
    }

    return nullptr;
}

void cumac_handler::start()
{
    NVLOGC_FMT(TAG, "{}", __func__);

    if(pthread_create(&cumac_recv_tid, NULL, cumac_receiver_thread_func, this) !=  0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create cuMAC receiver thread failed");
    }

    if(pthread_create(&cumac_sched_tid, NULL, cumac_scheduler_thread_func, this) !=  0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create cuMAC scheduler thread failed");
    }

    if(cumac_build_in_advance == 1)
    {
        for(int i = 0; i < cumac_configs->worker_cores.size(); i++)
        {
            pthread_t thread_id;
            if(pthread_create(&thread_id, NULL, cumac_worker_thread_func, this) != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create thread cuMAC worker_thread_func failed");
            }
        }
    }
    else if(cumac_build_in_advance == 2)
    {
        if(pthread_create(&cumac_builder_tid, NULL, cumac_builder_thread_func, this) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create cuMAC builder thread failed");
        }
    }
}

void cumac_handler::stop()
{
    NVLOGC_FMT(TAG, "{}", __func__);
}

void cumac_handler::join() {
    NVLOGC_FMT(TAG, "{}", __func__);
    if (cumac_sched_tid != 0 && pthread_join(cumac_sched_tid, NULL) != 0) {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Join cumac_sched thread failed");
    }
    if (cumac_recv_tid != 0) {
        pthread_cancel(cumac_recv_tid);
        pthread_join(cumac_recv_tid, NULL);
    }
    NVLOGC_FMT(TAG, "cumac_handler: [cumac_sched] and [cumac_recv] threads joined");
}

void cumac_handler::cell_init(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }
    cumac_thrputs[cell_id].reset();
    cumac_cell_data[cell_id].prach_state     = 0;
    cumac_cell_data[cell_id].prach_prambIdx  = -1;
    cumac_cell_data[cell_id].harq_process_id = 0;
    cumac_cell_data[cell_id].schedule_enable = false;
    cumac_cell_data[cell_id].cumac_state      = cumac_state_t::IDLE;

    if (first_init[cell_id] || testmac_configs->get_restart_option() != 0)
    {
        NVLOGC_FMT(TAG, "{}: cell_id={} global_tick={} first_init={}", __FUNCTION__, cell_id, global_tick, first_init[cell_id]);
        send_config_request(cell_id);
    }
    else
    {
        // Restart without CONFIG.request
        cell_start(cell_id);
    }
}

void cumac_handler::cell_start(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }
    NVLOGC_FMT(TAG, "{}: cell_id={} cumac_type=SCF global_tick={}", __FUNCTION__, cell_id, global_tick);
    cumac_thrputs[cell_id].reset();
    cumac_cell_data[cell_id].cumac_state = cumac_state_t::RUNNING;
    send_start_request(cell_id);
    cumac_cell_data[cell_id].schedule_enable = true;
}

void cumac_handler::cell_stop(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }
    NVLOGC_FMT(TAG, "{}: cell_id={} cumac_type=SCF global_tick={}", __FUNCTION__, cell_id, global_tick);
    cumac_cell_data[cell_id].schedule_enable = false;
    cumac_thrputs[cell_id].reset();
    send_stop_request(cell_id);
}

cumac_handler::cumac_handler(test_mac_configs* configs, const char* pattern_file, uint32_t ch_mask, uint64_t cell_mask)
{
    cumac_pattern* pattern = new cumac_pattern(configs);
    if(pattern->cumac_pattern_parsing(pattern_file, ch_mask, cell_mask) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: cumac_pattern parsing failed", __func__);
    }

    _transport = nullptr;

    cumac_handler_instance = this;

    testmac_configs = configs;
    cumac_configs = testmac_configs->cumac_configs;

    this->lp      = pattern;

    current_config_cell_id.store(-1);
    config_retry_counter.store(0);

    restart_timer                 = nullptr;
    global_tick                   = (uint64_t)0 - 1; // To start global_tick from 0
    conformance_test_slot_counter = 0;
    beam_id_position              = 0;

    slots_per_frame = lp->get_slots_per_frame();
    notify_mode = configs->get_ipc_sync_mode();

    data_buf_opt = configs->get_fapi_tb_loc();
    cell_num = lp->get_cell_num();
    cumac_thrputs.resize(cell_num);
    cumac_cell_data.resize(cell_num);
    first_init.resize(cell_num);

    configured_cell_num = 0;

    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        cumac_thrputs[cell_id].reset();
        cumac_cell_data[cell_id].prach_state     = 0;
        cumac_cell_data[cell_id].prach_prambIdx  = -1;
        cumac_cell_data[cell_id].harq_process_id = 0;
        cumac_cell_data[cell_id].schedule_enable = false;
        cumac_cell_data[cell_id].restart_timer   = nullptr;
        cumac_cell_data[cell_id].cumac_state      = cumac_state_t::IDLE;
        cumac_cell_data[cell_id].cell_id_remap = cell_id;
        cell_id_map[cell_id] = cell_id;
        cell_remap_event[cell_id] = false;

        cumac_cell_data[cell_id].slot_timing.resize(slots_per_frame);

        first_init[cell_id] = 1;
    }

    // Enable worker threads when worker_cores is configured
    if(cumac_configs->worker_cores.size() > 0)
    {
        cumac_build_in_advance = 1;
    }
    else if(cumac_configs->builder_thread_enable != 0)
    {
        cumac_build_in_advance = 2;
    }
    else
    {
        cumac_build_in_advance = 0;
    }

    sem_init(&cumac_worker_sem, 0, 0);

    sem_init(&cumac_build_sem, 0, 0);
    sem_init(&cumac_sched_ready, 0, 0);
    // sem_init(&cumac_sched_free, 0, 0);

    sem_init(&cumac_scheduler_sem, 0, 0);
    sched_info_ring = nv_ipc_ring_open(RING_TYPE_APP_INTERNAL, "cumac_scheduler", 256, sizeof(sched_info_t));

    ss_tick.store({.u32 = SFN_SLOT_INVALID});
    ts_tick = 0;
    ss_cumac_last.u32 = 0;

    if ((schedule_time = stat_log_open("MAC_SCHED", STAT_MODE_COUNTER, 2000)) != NULL)
    {
        schedule_time->set_limit(schedule_time, 0, 500);
    }
    if ((stat_debug = stat_log_open("MAC_DEBUG", STAT_MODE_COUNTER, 2000)) != NULL)
    {
        stat_debug->set_limit(stat_debug, 0, 1000LL * 1000 * 1000);
    }

    int ul_stat_interval = 100;
    int ul_timing_max = slots_per_frame * SLOT_INTERVAL;

    // UL CUMAC response timing
    if ((cumac_ul_time = stat_log_open("CUMAC_UL_TIME", STAT_MODE_COUNTER, ul_stat_interval)) != NULL)
    {
        cumac_ul_time->set_limit(cumac_ul_time, 0, ul_timing_max);
    }

    NVLOGC_FMT(TAG, "{} constructed: cell_num={} cumac_build_in_advance={}", __func__, cell_num, cumac_build_in_advance);
}

cumac_handler::~cumac_handler()
{
    lp->~cumac_pattern();
    if (schedule_time != NULL)
    {
        schedule_time->close(schedule_time);
    }
    if (stat_debug != NULL)
    {
        stat_debug->close(stat_debug);
    }

    if (cumac_ul_time != NULL)
    {
        cumac_ul_time->close(cumac_ul_time);
    }
}

void cumac_handler::slot_indication_handler(sfn_slot_t& ss, uint64_t slot_counter)
{
    if(cumac_build_in_advance == 1)
    {
        sem_wait(&cumac_sched_ready);

        // In multi-worker thread case, need wait last slot all building task finished then start new slot building
        cell_build_index.store(0);
        cell_build_counter.store(0);

        sfn_slot_t ss_next = get_next_sfn_slot(ss);
        ss_build.store(ss_next);

        NVLOGI_FMT(TAG, "SFN {}.{} STATE: send start: notify build SFN {}.{}", ss.u16.sfn, ss.u16.slot, ss_next.u16.sfn, ss_next.u16.slot);

        notify_worker_threads(cell_num);
    }
    else if(cumac_build_in_advance == 2)
    {
        sem_post(&cumac_build_sem);
        sem_wait(&cumac_sched_ready);
    }

    try
    {
        static_ul_dl_scheduler(ss.u16.sfn, ss.u16.slot, slot_counter);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "Send failed: {}", e.what());
    }

    // if (cumac_configs->builder_thread_enable)
    // {
    //     sem_post(&cumac_sched_free);
    // }
}

int cumac_handler::schedule_all_cell_restart(uint64_t slot_counter)
{
    int test_slots = cumac_configs->cumac_test_slots;
    if(test_slots > 0 && slot_counter > 0 && slot_counter > test_slots)
    {
        return 0;
    }

    if(test_slots > 0 && slot_counter > 0 && slot_counter % test_slots == 0)
    {
        int restart_interval = testmac_configs->get_restart_interval();
        NVLOGC_FMT(TAG, "Finished running {} slots test. slot_counter={} restart_interval={}", test_slots, slot_counter, restart_interval);

        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            // Stop slot scheduler
            cell_stop(cell_id);
        }

        // Set restart timer if restart_interval is configured
        if(restart_interval > 0)
        {
            set_restart_timer(CELL_ID_ALL, restart_interval);
        }
        NVLOGC_FMT(TAG, "Finished running {} slots test", test_slots);
        return 1;
    }
    return 0;
}

int cumac_handler::schedule_single_cell_restart(uint64_t slot_counter)
{
    // Single cell restart test if configured cell_run_slots > 0 && cell_stop_slots > 0
    int cell_run_slots = testmac_configs->get_cell_run_slots();
    int cell_stop_slots = testmac_configs->get_cell_stop_slots();
    if (cell_run_slots > 0 && cell_stop_slots > 0 && slot_counter > 0)
    {
        int restart_period = cell_run_slots + cell_stop_slots;
        int restart_slot_id = slot_counter % restart_period;
        if (restart_slot_id == cell_run_slots)
        {
            cell_stop(slot_counter / restart_period % cell_num);
        }
        else if (restart_slot_id == 0)
        {
            cell_init((slot_counter / restart_period + cell_num -1 ) % cell_num);
        }
    }

    return 0;
}

int cumac_handler::schedule_cell_update(uint64_t slot_counter)
{
    std::vector<cell_update_cmd_t>& cmds = testmac_configs->get_cell_update_commands();
    int size = cmds.size();
    if (size <= 0)
    {
        return 0;
    }

    int period = cmds[size -1].slot_point;
    int slot_index = slot_counter % period;

    for (cell_update_cmd_t cmd : cmds)
    {
        if (cmd.slot_point % period == slot_index)
        {
            if ((cmd.slot_point == 0 && slot_counter != 0) || (cmd.slot_point != 0 && slot_counter == 0))
            {
                // Update slot_point=0 configurations only at initialization
                continue;
            }

            for (cell_param_t param : cmd.cell_params)
            {
                char top_dir[1024];
                get_root_path(top_dir, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

                char sys_cmd[2048];
                int offset = snprintf(sys_cmd, 1024, "cd %s/build/cuPHY-CP/cuphyoam && ", top_dir);
                offset += snprintf(sys_cmd + offset, 1024, "python3 %s/cuPHY-CP/cuphyoam/examples/aerial_cell_param_net_update.py", top_dir);
                offset += snprintf(sys_cmd + offset, 1024, " %d %s %4X", param.cell_id + 1, param.mac.c_str(), (param.pcp << 13) + param.vlan);

                NVLOGC_FMT(TAG, "slot_counter={} test {}-{} stop cell {} for re-config net parameters",
                        slot_counter, period, slot_index, param.cell_id);
                NVLOGC_FMT(TAG, "CMD: {}", sys_cmd);

                string cmd_str(sys_cmd);
                std::thread oam_cmd_thread{[cmd_str] {
                    if(system(cmd_str.c_str()) != 0)
                    {
                        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: run system command failed: err={} - {}", __func__, errno, strerror(errno));
                    }
                }};

                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(0, &mask);

                int rc = pthread_setaffinity_np(oam_cmd_thread.native_handle(), sizeof(mask), &mask);
                if(rc != 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: set affinity failed: err={} - {}", __func__, rc, strerror(rc));
                }

                oam_cmd_thread.detach();

                if (slot_counter != 0)
                {
                    cell_stop(param.cell_id);
                    set_restart_timer(param.cell_id, testmac_configs->get_restart_interval());
                }
            }
        }
    }
    return 0;
}

void cumac_handler::static_ul_dl_scheduler(uint32_t sfn, uint32_t slot, uint64_t slot_counter)
{
    uint32_t cell_num = lp->get_cell_num();

    // Get CLOCK_REALTIME time stamp
    struct timespec ts_start;
    nvlog_gettime_rt(&ts_start);

    // NVLOGI_FMT(TAG, "SFN {}.{} slot_counter={} schedule ...", sfn, slot, slot_counter);

    if(schedule_all_cell_restart(slot_counter))
    {
        return;
    }

    schedule_single_cell_restart(slot_counter);

    if (slot_counter > 0)
    {
        schedule_cell_update(slot_counter);
    }

    sfn_slot_t ss;
    ss.u16.sfn = sfn;
    ss.u16.slot = slot;
    int cumac_count = 0;

#ifdef CUPTI_ENABLE_TRACING
    // Limited slots enabled for cupti tracing
    bool enable_this_slot = false;
    static int ramp_enable_slot_mask[80] {0};
    ramp_enable_slot_mask[20] = 1;
    ramp_enable_slot_mask[24] = 1;
    int ramp_slot_idx = (sfn % 4)*20 + slot;
    if (((sfn % 256) == 1) && (ramp_enable_slot_mask[ramp_slot_idx]))
    {
        enable_this_slot = true;
    }
#else
    bool enable_this_slot = true;
#endif

    if (enable_this_slot)
    {
        cumac_count = schedule_slot(ss);
    }

    // The time stamp of clock_gettime(CLOCK_REALTIME, &ts) and system_clock::now().time_since_epoch() should be the same
    struct timespec ts_now;
    nvlog_gettime_rt(&ts_now);

    int64_t handle_delay = (ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec - ts_tick) / 1000;
    int64_t mac_sched_time = (ts_now.tv_sec * 1000000000LL + ts_now.tv_nsec - ts_tick) / 1000; // us
    int64_t handle_time = mac_sched_time - handle_delay;

    if (schedule_time != NULL)
    {
        schedule_time->add(schedule_time, mac_sched_time); // us
    }

    NVLOGI_FMT(TAG_CUMAC, "SFN {}.{} schedule_end: slot_counter={} msg_count={} handle_delay={}us handle_time={}us total_time={}us",
            sfn, slot, slot_counter, cumac_count, handle_delay, handle_time, mac_sched_time);

    // Print throughput statistic info every second (print after finishing scheduling, do not include throughput of this slot)
    if(slot_counter > 0 && slot_counter % SLOTS_PER_SECOND == 0)
    {
        print_cumac_thrput(slot_counter);
    }

    // Add DL throughput of this slot
    if (enable_this_slot && testmac_configs->app_mode == 0)
    {
        update_dl_cumac_thrput(sfn, slot, slot_counter);
    }
}

int cumac_handler::set_cumac_delay(int cell_id, int slot, int cumac_mask, int delay_us)
{
    NVLOGC_FMT(TAG,"OAM Set CUMAC delay: cell_id={} slot={} cumac_mask=0x{:02X} delay_us={}",
            cell_id, slot, cumac_mask, delay_us);

    if (cell_id != 255 && cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: error: cell_id={} cell_num={}", __func__, cell_id, cell_num);
        return -1;
    }

    if (slot >= lp->get_sched_slot_num())
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: error: slot={} sched_slot_num={}", __func__, slot, lp->get_sched_slot_num());
        return -1;
    }

    // Set delay for 1 slot of the whole cumac pattern
    cumac_configs->cumac_stt.resize(lp->get_sched_slot_num());
    for (int i = 0; i < lp->get_sched_slot_num(); i++)
    {
        cumac_configs->cumac_stt[i] = i == slot ? delay_us * 1000 : 0;
    }

    for (int j = 0; j < CUMAC_REQ_SIZE; j++)
    {
        if (cumac_mask & (1 << j))
        {
            for (int k = 0; k < CUMAC_SCHED_BUFFER_SIZE; k++)
            {
                if (cell_id < cell_num)
                {
                    cumac_cell_data[cell_id].cumac_scheds[k].target_ts_offsets[j] = &cumac_configs->cumac_stt;
                }
                else
                {
                    for (int cid = 0; cid < cell_num; cid ++)
                    {
                        cumac_cell_data[cid].cumac_scheds[k].target_ts_offsets[j] = &cumac_configs->cumac_stt;
                    }
                }
            }
        }
    }

    return 0;
}

void cumac_handler::update_dl_cumac_thrput(uint32_t sfn, uint32_t slot, uint64_t slot_counter)
{
    vector<vector<vector<vector<cumac_req_t*>>>>& slot_cell_patterns = lp->get_slot_cell_patterns(slot_counter);

    uint32_t cell_num = lp->get_cell_num();
    uint32_t slot_idx = get_slot_in_frame(sfn, slot) % slot_cell_patterns.size();

    auto cells_size = slot_cell_patterns[slot_idx].size();
    auto start_cell_index = 0;
    uint16_t num_cumac_msgs_per_cell = 0;

    for(int cell_id = 0; cell_id < cells_size; cell_id++)
    {
        if (cumac_cell_data[cell_id].schedule_enable == false)
        {
            // Skip stopped cells
            continue;
        }
    }
}

void cumac_handler::print_cumac_thrput(uint64_t slot_counter)
{
    const int task_bitmask = cumac_configs->task_bitmask;

    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        cumac_thrput_t& cumac_thrput = cumac_thrputs[cell_id];

        // Build output string with CUMAC count and individual task counters using snprintf
        char output[512];
        int offset = 0;

        // Add cell ID and overall CUMAC counter
        offset += snprintf(output + offset, sizeof(output) - offset,
                          "Cell %2d | CUMAC %3u |",
                          cell_id, cumac_thrput.cumac_slots.load());

        // Add task counters for enabled tasks
        if (task_bitmask & (0x1 << CUMAC_TASK_UE_SELECTION)) {
            offset += snprintf(output + offset, sizeof(output) - offset,
                              " UE_SEL %3u |", cumac_thrput.task_slots[CUMAC_TASK_UE_SELECTION].load());
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) {
            offset += snprintf(output + offset, sizeof(output) - offset,
                              " PRB_ALLOC %3u |", cumac_thrput.task_slots[CUMAC_TASK_PRB_ALLOCATION].load());
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_LAYER_SELECTION)) {
            offset += snprintf(output + offset, sizeof(output) - offset,
                              " LAYER_SEL %3u |", cumac_thrput.task_slots[CUMAC_TASK_LAYER_SELECTION].load());
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_MCS_SELECTION)) {
            offset += snprintf(output + offset, sizeof(output) - offset,
                              " MCS_SEL %3u |", cumac_thrput.task_slots[CUMAC_TASK_MCS_SELECTION].load());
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_PFM_SORT)) {
            offset += snprintf(output + offset, sizeof(output) - offset,
                              " PFM_SORT %3u |", cumac_thrput.task_slots[CUMAC_TASK_PFM_SORT].load());
        }

        // Add error, invalid and slots counters
        offset += snprintf(output + offset, sizeof(output) - offset,
                " ERR %4u | INV %4u | Slots %lu", cumac_thrput.error.load(), cumac_thrput.invalid.load(), slot_counter);

        if (offset >= sizeof(output) - 1)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: error: throughput string buffer length is not enough", __func__);
        }

        // Console log print per second
        NVLOGC_FMT(TAG, "{}", output);

        cumac_thrput.reset();
    }
}

void cumac_handler::on_tick_event(sched_info_t sched_info)
{
    if (is_app_exiting())
    {
        // Wake up cumac_sched thread to exit the loop
        sem_post(&cumac_scheduler_sem);
        return;
    }

    if (configured_cell_num < cell_num) {
        NVLOGI_FMT(TAG, "SFN {}.{} on_tick_event ts={} skipped before cumac_cp init finishing", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, sched_info.ts);
        return;
    }

    NVLOGI_FMT(TAG, "SFN {}.{} STATE: tick received: ts={}", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, sched_info.ts);

    if (sched_info_ring->enqueue(sched_info_ring, &sched_info) == 0)
    {
        sem_post(&cumac_scheduler_sem);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "Error: testCUMAC scheduler is full, please check SLOT.ind sending frequency");
    }
}

void cumac_handler::on_msg(nv::phy_mac_msg_desc& msg)
{
    if(msg.msg_buf == NULL)
    {
        NVLOGI_FMT(TAG, "No more message");
        return;
    }

    int      cell_id = msg.cell_id;
    uint16_t msg_id  = msg.msg_id;

    if(cell_id < 0 || cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "RECV: cell_id error: msg_id=0x{:02X} cell_id={} pool={}", msg.msg_id, cell_id, msg.data_pool);
        return;
    }

    cumac_validate vald(cumac_configs->validate_enable, cumac_configs->validate_log_opt);

    switch(msg_id)
    {
    case CUMAC_CONFIG_RESPONSE: {
        auto resp = reinterpret_cast<cumac_config_resp_t*>(msg.msg_buf);
        NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} {}: error_code={}", cell_id, msg_id, get_cumac_msg_name(msg_id), resp->error_code);

        // If error_code != 0, will retry sending CONFIG.req when reconfig_timer timeout
        if(resp->error_code == 0)
        {
            // Successfully CONFIG.resp received, current cell CONFIG was done
            NVLOGC_FMT(TAG, "cell_config: cell_id={} OK", cell_id);

            configured_cell_num ++;

            if(first_init[cell_id] == 1)
            {
                first_init[cell_id] = 0;
            }
            // Send START.req if not controlled by OAM
            if(testmac_configs->oam_cell_ctrl_cmd == 0)
            {
                cell_start(cell_id);
            }

            if(testmac_configs->cell_config_wait < 0)
            {
                // No wait for CONFIG.resp, skip
                break;
            }

            if(current_config_cell_id.load() != cell_id)
            {
                NVLOGC_FMT(TAG, "cell_config: got cell_id={} response when current cell_id={}", current_config_cell_id.load(), cell_id);
                // Skip, do not interrupt current cell config procedure
                break;
            }

            config_retry_counter.store(0);
            stop_reconfig_timer(cell_id);

            if(testmac_configs->cell_config_wait > 0)
            {
                // Start the timer to wait for <cell_config_wait> ms then config next cell if exist
                start_reconfig_timer(cell_id, testmac_configs->cell_config_wait);
            }
            else
            {
                // Immediately config next pending cell if exist
                current_config_cell_id.store(-1);
                int next_pending_cell = get_pending_config_cell_id();
                if(next_pending_cell >= 0)
                {
                    send_config_request(next_pending_cell);
                }
            }
        }
        else
        {
            NVLOGC_FMT(TAG, "cell_config: cell_id={} failed: error_code={}", cell_id, resp->error_code);
        }
    }
    break;

    case CUMAC_START_RESPONSE: {
        NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} {}", cell_id, msg_id, get_cumac_msg_name(msg_id));
    }
    break;

    case CUMAC_STOP_RESPONSE: {
        auto resp = reinterpret_cast<cumac_stop_resp_t*>(msg.msg_buf);
        NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} {}: error_code={}", cell_id, msg_id, get_cumac_msg_name(msg_id), resp->error_code);
        cumac_cell_data[cell_id].cumac_state = cumac_state_t::IDLE;
        NVLOGC_FMT(TAG, "cell {} stopped", cell_id);
    }
    break;

    case CUMAC_ERROR_INDICATION: {
        auto resp = reinterpret_cast<cumac_err_ind_t*>(msg.msg_buf);
        NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} {}: error_code={} reason_code={}", cell_id, msg_id, get_cumac_msg_name(msg_id), resp->error_code, resp->reason_code);
        cumac_thrputs[cell_id].error++;
        // NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} CUMAC_ERROR_INDICATION: err_code=0x{:02X} err_msg_id=0x{:02X}",
        //         static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.err_code, resp.msg_id);

        switch(resp->msg_id)
        {
        case CUMAC_START_REQUEST:
            NVLOGW_FMT(TAG, "START failed: cell_id={} error_code={}", cell_id, resp->error_code);
            cumac_cell_data[cell_id].schedule_enable = false;
            cumac_cell_data[cell_id].cumac_state     = cumac_state_t::IDLE;
            break;
        default:
            break;
        }
    }
    break;

    case CUMAC_TTI_ERROR_INDICATION: {
        auto resp = reinterpret_cast<cumac_tti_err_ind_t*>(msg.msg_buf);
        NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} {}: error_code={} reason_code={} err_msg_id={}", resp->sfn, resp->slot, cell_id, msg_id, get_cumac_msg_name(msg_id), resp->error_code, resp->reason_code, resp->msg_id);
        cumac_thrputs[cell_id].error++;
    }
    break;

    case CUMAC_SCH_TTI_RESPONSE: {
        auto resp = reinterpret_cast<cumac_sch_tti_resp_t*>(msg.msg_buf);
        handle_sch_tti_response(msg, vald);
    }
    break;

    default: {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "RECV: Unknown CUMAC MSG: cell_id={} msg_id=0x{:02X} {}", cell_id, msg_id, get_cumac_msg_name(msg_id));
    }
    break;
    }
}

int cumac_handler::handle_sch_tti_response(nv::phy_mac_msg_desc& msg_desc, cumac_validate& vald)
{
    auto resp = reinterpret_cast<cumac_sch_tti_resp_t*>(msg_desc.msg_buf);

    uint8_t* buf_home = static_cast<uint8_t*>(msg_desc.data_buf);

    cumac_req_t* req = get_cumac_req_data(msg_desc.cell_id, resp->sfn, resp->slot);
    if (req == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} {} req=nullptr",
            resp->sfn, resp->slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        return -1;
    }

    if (req->tv_data == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} {} tv_data=nullptr",
            resp->sfn, resp->slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        return -1;
    }

    cumac_cell_configs_t& cell_cfg = lp->get_cumac_cell_configs(msg_desc.cell_id);

    // CUMAC MSG validation: use actual response sizes (nUeSchd, nActiveUe) from cuMAC-CP
    vald.msg_start(msg_desc.cell_id, msg_desc.msg_id, resp->sfn, resp->slot);
    vald.set_cumac_req(req);
    uint32_t nUeSchd = resp->nUeSchd;   // Actual scheduled UEs this TTI (may be < nMaxSchUePerCell)

    if(cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        // Output data buffers
        uint16_t* setSchdUePerCellTTI = reinterpret_cast<uint16_t*>(buf_home + resp->offsets.setSchdUePerCellTTI);

        // Validate setSchdUePerCellTTI buffer
        CUMAC_VALIDATE_BYTES_ERR(&vald, setSchdUePerCellTTI, req->tv_data->resp.setSchdUePerCellTTI, nUeSchd * sizeof(uint16_t));

        if(cumac_configs->debug_option & 0x2)
        {
            print_array(msg_desc, "setSchdUePerCellTTI", setSchdUePerCellTTI, nUeSchd);
        }

        // Increment task counter for UE_SELECTION
        cumac_thrputs[msg_desc.cell_id].task_slots[CUMAC_TASK_UE_SELECTION]++;
    }

    if(cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        int16_t* allocSol     = reinterpret_cast<int16_t*>(buf_home + resp->offsets.allocSol);
        uint32_t  allocSol_num = cell_cfg.allocType == 0 ? req->tv_data->req.nPrbGrp : 2 * nUeSchd;

        CUMAC_VALIDATE_BYTES_ERR(&vald, allocSol, req->tv_data->resp.allocSol, allocSol_num * sizeof(int16_t));

        if(cumac_configs->debug_option & 0x2)
        {
            print_array(msg_desc, "allocSol", allocSol, allocSol_num);
        }

        // Increment task counter for PRB_ALLOCATION
        cumac_thrputs[msg_desc.cell_id].task_slots[CUMAC_TASK_PRB_ALLOCATION]++;
    }

    if(cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
    {
        // Output data buffers
        uint8_t* layerSelSol = reinterpret_cast<uint8_t*>(buf_home + resp->offsets.layerSelSol);

        // Validate layerSelSol buffer
        CUMAC_VALIDATE_BYTES_ERR(&vald, layerSelSol, req->tv_data->resp.layerSelSol, nUeSchd * sizeof(uint8_t));

        if(cumac_configs->debug_option & 0x2)
        {
            print_array(msg_desc, "layerSelSol", layerSelSol, nUeSchd);
        }

        // Increment task counter for LAYER_SELECTION
        cumac_thrputs[msg_desc.cell_id].task_slots[CUMAC_TASK_LAYER_SELECTION]++;
    }

    if(cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_MCS_SELECTION))
    {
        // Output data buffers
        int16_t* mcsSelSol = reinterpret_cast<int16_t*>(buf_home + resp->offsets.mcsSelSol);

        // Validate mcsSelSol buffer. TODO: provide the right mcsSelSol TV instead of just check range (0~31)
        // CUMAC_VALIDATE_BYTES_ERR(&vald, mcsSelSol, req->tv_data->resp.mcsSelSol, nUeSchd * sizeof(int16_t));
        uint32_t invalid_num = 0;
        for(int i = 0; i < nUeSchd; i++)
        {
            int16_t val = *(mcsSelSol + i);
            if(val < 0 || val > 31)
            {
                invalid_num++;
            }
        }
        if(invalid_num > nUeSchd / 2)
        {
            vald.report_text(VALD_ENABLE_ERR, "SFN %u.%u cell_id=%d mcsSelSol invalid_num=%d", resp->sfn, resp->slot, msg_desc.cell_id, invalid_num);
            // CUMAC_VALIDATE_TEXT_ERR((&vald), "SFN %u.%u cell_id=%d mcsSelSol invalid_num=%d", resp->sfn, resp->slot, msg_desc.cell_id, invalid_num)
        }

        if(cumac_configs->debug_option & 0x2)
        {
            print_array(msg_desc, "mcsSelSol", mcsSelSol, nUeSchd);
        }

        // Increment task counter for MCS_SELECTION
        cumac_thrputs[msg_desc.cell_id].task_slots[CUMAC_TASK_MCS_SELECTION]++;
    }

    if(cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        // Output data buffers
        cumac_pfm_output_cell_info_t* pfmSortSol = reinterpret_cast<cumac_pfm_output_cell_info_t*>(buf_home + resp->offsets.pfmSortSol);

        // Validate pfmSortSol buffer
        CUMAC_VALIDATE_BYTES_ERR(&vald, pfmSortSol, req->tv_data->resp.pfmSortSol, sizeof(cumac_pfm_output_cell_info_t));
        if(cumac_configs->debug_option & 0x2)
        {
            print_array(msg_desc, "pfmSortSol", reinterpret_cast<uint8_t*>(pfmSortSol), sizeof(cumac_pfm_output_cell_info_t));
        }

        // Increment task counter for PFM_SORT
        cumac_thrputs[msg_desc.cell_id].task_slots[CUMAC_TASK_PFM_SORT]++;
    }

    int vald_ret = vald.msg_ended();

    NVLOGI_FMT(TAG_CUMAC, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} {} VALD={}", resp->sfn, resp->slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id), vald_ret ? "FAIL" : "OK");

    // Increases the slot number in cumac_thrput
    cumac_thrputs[msg_desc.cell_id].cumac_slots++;

    return 0;
}

void cumac_handler::receiver_thread_func()
{
    //wait for IPC connection
    sleep(1);

    // Create tick thread and start tick
    if(get_cumac_configs()->cumac_cp_standalone)
    {
        if(pthread_create(&cumac_tick_tid, NULL, cumac_tick_thread_func, this) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create cuMAC tick thread failed");
        }
    }

    // Send OAM cell update message if configured
    // schedule_cell_update(0);

    try
    {
        if(testmac_configs->oam_cell_ctrl_cmd == 0)
        {
            for(int cell_id = 0; cell_id < get_cell_num(); cell_id++)
            {
                cell_init(cell_id);
            }
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "IPC send failed, please check whether cumac_cp is running properly", e.what());
    }

    if(cumac_build_in_advance != 0)
    {
        build_first_slot();
    }

    nv::phy_mac_msg_desc msg_desc;
    while (1) {
        try {
            _transport->rx_wait();
            while (_transport->rx_recv(msg_desc) >= 0) {
                on_msg(msg_desc);
                _transport->rx_release(msg_desc);
            }
        } catch (std::exception &e) {
            NVLOGW_FMT(TAG, "cumac receiver_thread_func: exception: {}", e.what());
        }
    }
}

void cumac_handler::scheduler_thread_func()
{
    sched_info_t sched_info;
    while(sem_wait(&cumac_scheduler_sem) == 0)
    {
        if (is_app_exiting())
        {
            NVLOGC_FMT(TAG, "App is exiting, stop all cuMAC RUNNING cells");
            // Stop all RUNNING cells
            for (int cell_id = 0; cell_id < cell_num; cell_id++)
            {
                if (cumac_cell_data[cell_id].cumac_state == cumac_state_t::RUNNING)
                {
                    cell_stop(cell_id);
                }
            }

            int wait_count = 0;
            for (int cell_id = 0; cell_id < cell_num; cell_id++)
            {
                while (wait_count < 10 && cumac_cell_data[cell_id].cumac_state == cumac_state_t::RUNNING)
                {
                    // Still has at least one RUNNING cell, wait 100ms for it to stop
                    usleep(100 * 1000);
                    wait_count++;
                }
            }
            NVLOGC_FMT(TAG, "Waited for {} ms for all RUNNING cells to stop", wait_count * 100);

            // Exit the while loop to exit the scheduler thread
            break;
        }

        while (sched_info_ring->dequeue(sched_info_ring, &sched_info) == 0)
        {
            // No need worry about overflow. Even 1us a tick, the overflow time is 1.84*10^19 / 10^6 / 86400 / 365 = 5.83*10^5 years.
            global_tick ++;
            ss_tick.store(sched_info.ss);
            ts_tick = sched_info.ts;

            struct timespec ts_start;
            nvlog_gettime_rt(&ts_start);
            int64_t start_delay = ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec - ts_tick;

            NVLOGI_FMT(TAG, "SFN {}.{} STATE: schedule start: global_tick={} start_delay={}ns", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, global_tick, start_delay);

            // handle 0th based index
            slot_indication_handler(sched_info.ss, global_tick);

            int64_t end_time = std::chrono::system_clock::now().time_since_epoch().count();
            NVLOGI_FMT(TAG, "SFN {}.{} STATE: schedule end: global_tick={} start_delay={}ns total_time={}", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, global_tick, start_delay, end_time - ts_tick);

            lp->check_init_pattern_finishing(global_tick + 1);
        }
    }
    NVLOGC_FMT(TAG, "[cumac_sched] thread exiting");
}

void cumac_handler::build_first_slot()
{
    NVLOGC_FMT(TAG, "{}: BUILD SFN 0.0 in advance", __func__);

    // Build SFN 0.0 CUMAC messages
    sfn_slot_t ss = {.u32 = 0};
    schedule_cumac_reqs({.u32 = 0}, -500 * 1000);

    NVLOGI_FMT(TAG, "SFN {}.{} STATE: build ready: cell_num={}", ss.u16.sfn, ss.u16.slot, cell_num);
    sem_post(&cumac_sched_ready);
}

void cumac_handler::builder_thread_func()
{
    // Wait for SLOT.ind to trigger semaphore post, then build next slot CUMAC messages in advance
    while(sem_wait(&cumac_build_sem) == 0)
    {
        // sem_wait(&cumac_sched_free);

        sfn_slot_t ss_curr = ss_tick.load();
        if (ss_cumac_last.u32 != ss_curr.u32)
        {
            NVLOGW_FMT(TAG, "Slot missed: SFN curr: {}.{} last: {}.{}",
                    ss_curr.u16.sfn, ss_curr.u16.slot, ss_cumac_last.u16.sfn, ss_cumac_last.u16.slot);
        }

        sfn_slot_t ss_cumac = get_next_sfn_slot(ss_curr);
        // ss_cumac.u32 = ss_curr.u32 == SFN_SLOT_INVALID ? 0 : ss_curr.u32;

        schedule_cumac_reqs(ss_cumac, -500 * 1000);

        ss_cumac_last = ss_cumac;

        sem_post(&cumac_sched_ready);
    }
}

int cumac_handler::poll_build_task()
{
    int poll_result = -1;

    // Poll and process cuMAC message build task if available
    int cell_id = cell_build_index.fetch_add(1);
    if(cell_id < cell_num && cell_id >= 0)
    {
        poll_result   = 0;
        sfn_slot_t ss = ss_build.load();

        // NVLOGI_FMT(TAG, "{}: SFN {}.{} BUILD {} cell_id={} start", __func__, ss.u16.sfn, ss.u16.slot, cell_num, cell_id);

        schedule_cumac_reqs(ss, -500 * 1000, cell_id);

        int build_counter = cell_build_counter.fetch_add(1);
        NVLOGI_FMT(TAG, "{}: SFN {}.{} BUILD {}-{} cell_id={} done", __func__, ss.u16.sfn, ss.u16.slot, cell_num, build_counter, cell_id);
        if(build_counter == cell_num - 1)
        {
            // Notify cumac_sched thread when all cells building finished
            NVLOGI_FMT(TAG, "SFN {}.{} STATE: build ready: cell_build_index={}", ss.u16.sfn, ss.u16.slot, cell_build_index.load());
            sem_post(&cumac_sched_ready);
        }

        // Only for error check, should never run to here
        if(build_counter > cell_num)
        {
            NVLOGF_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: ERROR: cell_num={} build_counter={}", __func__, cell_num, build_counter);
        }
    }

    return poll_result;
}

void cumac_handler::worker_thread_func()
{
    int wid = cumac_worker_id.fetch_add(1);

    if (wid >= cumac_configs->worker_cores.size())
    {
        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "{}: invalid worker id: wid={} worker_cores.size={}", __FUNCTION__, wid, cumac_configs->worker_cores.size());
        return;
    }

    char name[16];
    snprintf(name, 16, "cumac_worker_%02d", wid);

    struct nv::thread_config config;
    config.name = std::string(name);
    config.cpu_affinity = cumac_configs->worker_cores[wid];
    config.sched_priority = 80;
    config_thread_property(config);

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread cuMAC {} on CPU {} initialized fmtlog", name, sched_getcpu());

    while(sem_wait(&cumac_worker_sem) == 0)
    {
        while(poll_build_task() >= 0)
        {
            // Keep polling util all task done
        }
    }
}

void cumac_handler::notify_worker_threads(uint32_t num)
{
    for(uint32_t i = 0; i < num; i++)
    {
        sem_post(&cumac_worker_sem);
    }
}

bool cumac_handler::cell_id_sanity_check(int cell_id)
{
    if(cell_id >= cell_num)
    {
        NVLOGW_FMT(TAG, "{}: cell_id {} out of bounds, ignored. Valid range is [0 ~ {}]", __FUNCTION__, cell_id, cell_num - 1);
        return false;
    }
    return true;
}

int cumac_handler::send_config_request(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return -1;
    }

    if (testmac_configs->cell_config_wait >= 0)
    {
        int32_t cas_expected = -1;
        bool cas_result = current_config_cell_id.compare_exchange_strong(cas_expected, cell_id);
        if (cas_result == true)
        {
            // New config, send CONFIG.req
            config_retry_counter.store(testmac_configs->cell_config_retry);
        }
        else if (cas_expected == cell_id)
        {
            // Current cell CONFIG is on going, retry config by re-send CONFIG.req
            config_retry_counter.fetch_sub(1);
        }
        else
        {
            // Another cell config is ongoing, set this cell to pending and return
            cumac_cell_data[cell_id].pending_config.store(1);
            return 0;
        }

        // Start timer to check no CONFIG.resp when timeout
        start_reconfig_timer(cell_id, testmac_configs->cell_config_timeout);
    }

    nv::phy_mac_msg_desc msg_desc;
    if(transport().tx_alloc(msg_desc) < 0)
    {
        return -1;
    }

    cumac_config_req_t* req = cumac_init_msg_header<cumac_config_req_t>(&msg_desc, CUMAC_CONFIG_REQUEST, cell_id);
    cumac_cell_configs_t& cfgs = lp->get_cumac_cell_configs(cell_id);
    memcpy(req->body, &lp->get_cumac_cell_configs(cell_id), sizeof(cumac_cell_configs_t));
    req->header.body_len += sizeof(cumac_cell_configs_t);
    msg_desc.msg_len = req->header.body_len + sizeof(cumac_msg_header_t);

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transport().tx_send(msg_desc);
    transport().tx_post();
    return 0;
}

int cumac_handler::send_start_request(int cell_id)
{
    nv::phy_mac_msg_desc msg_desc;
    if(transport().tx_alloc(msg_desc) < 0)
    {
        return -1;
    }

    auto req = cumac_init_msg_header<cumac_start_req_t>(&msg_desc, CUMAC_START_REQUEST, cell_id);
    req->start_param = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transport().tx_send(msg_desc);
    transport().tx_post();

    return 0;
}

int cumac_handler::send_stop_request(int cell_id)
{
    nv::phy_mac_msg_desc msg_desc;
    if(transport().tx_alloc(msg_desc) < 0)
    {
        return -1;
    }

    auto req = cumac_init_msg_header<cumac_stop_req_t>(&msg_desc, CUMAC_STOP_REQUEST, cell_id);
    req->stop_param = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transport().tx_send(msg_desc);
    transport().tx_post();

    return 0;
}

vector<cumac_req_t*>& cumac_handler::get_cumac_req_list(int cell_id, sfn_slot_t ss, cumac_group_t group_id)
{
    cumac_slot_pattern_t &slot_cell_patterns = lp->get_slot_cell_patterns(get_slot_in_frame(ss));
    uint32_t slot_idx = get_slot_in_frame(ss) % slot_cell_patterns.size();
    vector<vector<cumac_req_t*>> &cumac_groups = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
    return cumac_groups[group_id];
}

cumac_req_t* cumac_handler::get_cumac_req_data(int cell_id, uint16_t sfn, uint16_t slot)
{
    vector<vector<vector<vector<cumac_req_t*>>>>& slot_cell_patterns = lp->get_slot_cell_patterns(get_slot_in_frame(sfn, slot));
    uint32_t                                      slot_idx           = get_slot_in_frame(sfn, slot) % slot_cell_patterns.size();
    vector<vector<cumac_req_t*>>&                 cumac_groups       = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
    vector<cumac_req_t*>&                         cumac_reqs         = cumac_groups[CUMAC_SCH_TTI_REQ];

    if(cumac_reqs.size() > 0)
    {
        cumac_req_t* req = cumac_reqs[0];
        return req;
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} cumac_reqs.size=0", __func__, sfn, slot);
        return nullptr;
    }
}

template <typename T>
void cumac_handler::print_array(phy_mac_msg_desc& msg_desc, const char* info, T* array, uint32_t num)
{
    cumac_slot_msg_header_t* header = reinterpret_cast<cumac_slot_msg_header_t*>(msg_desc.msg_buf);

    char info_str[64];
    snprintf(info_str, 64, "ARRAY SFN %u.%u %s-%d", header->sfn, header->slot, info, msg_desc.cell_id);
    uint8_t* buf = reinterpret_cast<uint8_t*>(msg_desc.data_buf);

    NVLOGI_FMT_ARRAY(TAG, info_str, array, num);
}

template <typename T, typename U>
int cumac_handler::copy_to_ipc_buf(phy_mac_msg_desc& msg_desc, T* src, const char* info, uint32_t& src_offset, uint32_t& dst_offset, uint32_t num)
{
    size_t size = sizeof(T) * num;

    if(dst_offset + size > cumac_configs->get_max_data_size())
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error: IPC data buffer overflow: {} buf_size={} dst_offset={} size={}", info, cumac_configs->get_max_data_size(), dst_offset, size);
        return -1;
    }

    transport().copy_to_data_buf(msg_desc, dst_offset, src, size);

    if(cumac_configs->debug_option & 0x1)
    {
        uint8_t* buf = reinterpret_cast<uint8_t*>(msg_desc.data_buf);
        print_array(msg_desc, info, reinterpret_cast<U*>(buf + dst_offset), size / sizeof(U));
    }

    src_offset = dst_offset;
    dst_offset += size;

    return 0;
}

int cumac_handler::copy_complex_to_ipc_buf(phy_mac_msg_desc& msg_desc, cuComplex* src, const char* info, uint32_t& src_offset, uint32_t& dst_offset, uint32_t num)
{
    size_t size = sizeof(cuComplex) * num;

    if(dst_offset + size > cumac_configs->get_max_data_size())
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error: IPC data buffer overflow: {} buf_size={} dst_offset={} size={}", info, cumac_configs->get_max_data_size(), dst_offset, size);
        return -1;
    }

    transport().copy_to_data_buf(msg_desc, dst_offset, src, size);

    if(cumac_configs->debug_option & 0x1)
    {
        uint8_t* buf = reinterpret_cast<uint8_t*>(msg_desc.data_buf);
        print_array(msg_desc, info, reinterpret_cast<float*>(buf + dst_offset), num);
    }

    src_offset = dst_offset;
    dst_offset += size;

    return 0;
}

int cumac_handler::build_sch_tti_request(int cell_id, vector<cumac_req_t*>& cumac_reqs, cumac_sch_tti_req_t& head, phy_mac_msg_desc& msg_desc)
{
    size_t offset = 0;

    for(cumac_req_t* cumac_req : cumac_reqs)
    {
        // Use CLOCK_MONOTONIC to avoid time jump when system clock is adjusted by NTP
        struct timespec ts_start, ts_end;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        cumac_tti_req_tv_t& tv = cumac_req->tv_data->req;
        cumac_tti_req_payload_t& req = head.payload;

        req.taskBitMask = cumac_configs->task_bitmask;
        req.cellID = cell_id; // Ignore cellID from TV
        req.ULDLSch = tv.ULDLSch;
        req.nActiveUe = tv.nActiveUe;
        req.nSrsUe = tv.nSrsUe;

        req.nPrbGrp = tv.nPrbGrp;
        req.nBsAnt = tv.nBsAnt;
        req.nUeAnt = tv.nUeAnt;
        req.sigmaSqrd = tv.sigmaSqrd;

        // Initiate all offsets to INVALID_CUMAC_BUF_OFFSET=0xFFFFFFFF
        memset(&req.offsets, INVALID_CUMAC_BUF_OFFSET, sizeof(req.offsets));

        // Copy the elements to nvipc data buffer
        uint32_t offset = 0;

        // multiCellUeSelection buffers

        cumac_cell_configs_t& cell_cfg = lp->get_cumac_cell_configs(msg_desc.cell_id);

        // CRNTI is not needed for cuMAC-CP
        copy_to_ipc_buf(msg_desc, tv.prgMsk, "prgMsk", req.offsets.prgMsk, offset, tv.nPrbGrp);
        copy_to_ipc_buf(msg_desc, tv.wbSinr, "wbSinr", req.offsets.wbSinr, offset, tv.nActiveUe * tv.nUeAnt);
        copy_to_ipc_buf(msg_desc, tv.avgRatesActUe, "avgRatesActUe", req.offsets.avgRatesActUe, offset, tv.nActiveUe);

        if(req.taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // multiCellScheduler buffers
        {
            copy_to_ipc_buf(msg_desc, tv.postEqSinr, "postEqSinr", req.offsets.postEqSinr, offset, tv.nActiveUe * tv.nPrbGrp * tv.nUeAnt);
            copy_to_ipc_buf(msg_desc, tv.sinVal, "sinVal", req.offsets.sinVal, offset, cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt);

            uint32_t prdLen, detLen, hLen;
            if(tv.ULDLSch == 1)
            { // DL
                prdLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nBsAnt * tv.nBsAnt;
                detLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt * tv.nUeAnt;
            }
            else
            { // UL
                prdLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt * tv.nUeAnt;
                detLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nBsAnt * tv.nBsAnt;
            }
            hLen = tv.nPrbGrp * cell_cfg.nMaxSchUePerCell * cell_cfg.nMaxCell * tv.nBsAnt * tv.nUeAnt;

            copy_complex_to_ipc_buf(msg_desc, tv.detMat, "detMat", req.offsets.detMat, offset, detLen);
            copy_complex_to_ipc_buf(msg_desc, tv.estH_fr, "estH_fr", req.offsets.estH_fr, offset, hLen);
            copy_complex_to_ipc_buf(msg_desc, tv.prdMat, "prdMat", req.offsets.prdMat, offset, prdLen);
        }

        if(req.taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION)) // mcsSelectionLUT buffers
        {
            copy_to_ipc_buf(msg_desc, tv.tbErrLastActUe, "tbErrLastActUe", req.offsets.tbErrLastActUe, offset, tv.nActiveUe);
        }

        if(req.taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT)) // pfmSort buffers
        {
            copy_to_ipc_buf<cumac_pfm_cell_info_t, uint8_t>(msg_desc, tv.pfmCellInfo, "pfmCellInfo", req.offsets.pfmCellInfo, offset, 1);
        }

        msg_desc.msg_len = sizeof(cumac_sch_tti_req_t);
        msg_desc.data_len = offset;

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        int64_t build_time = nvlog_timespec_interval(&ts_start, &ts_end);
        NVLOGI_FMT(TAG, "SFN {}.{}: BUILD: SCH_TTI.req cell_id={} cellID={} ULDLSch={} nActiveUe={} nBsAnt={} nUeAnt={} msg_len={} data_len={} time={}ns",
                head.sfn, head.slot, cell_id, req.cellID, req.ULDLSch, req.nActiveUe, req.nBsAnt, req.nUeAnt, msg_desc.msg_len, msg_desc.data_len, build_time);
    }

    if(offset > cumac_configs->get_max_data_size())
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error: IPC data buffer overflowed: buf_size={} offset={}", cumac_configs->get_max_data_size(), offset);
        return -1;
    }
    else
    {
        return 0;
    }
}

// Return successfully sent CUMAC message count
int cumac_handler::schedule_cumac_request(int cell_id, sfn_slot_t ss, cumac_group_t group_id, int32_t ts_offset)
{
#ifdef PREPONE_TX_DATA_REQ // Strongly not recommended, caution to enable in release version.
    if (group_id == TX_DATA_REQ)
    {
        ss = get_next_sfn_slot(ss);
    }
#endif

    cumac_sched_t& cumac_sched = cumac_cell_data[cell_id].cumac_scheds[ss.u16.slot & 0x3];
    nv::phy_mac_msg_desc& msg_desc = cumac_sched.cumac_msg_cache[group_id];

    sfn_slot_t ss_curr = ss_tick.load();
    // Build CUMAC message at one of the two: (1) builder thread with ts_offset < 0; (2) non-builder thread with ts_offset = 0.
    if ((cumac_build_in_advance != 0 && ts_offset < 0) || (cumac_build_in_advance == 0 && ts_offset == 0))
    {
        if (msg_desc.msg_buf != nullptr)
        {
            sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
            NVLOGI_FMT(TAG, "Current SFN {}.{} dropping CUMAC: SFN {}.{} cell_id={} msg_id=0x{:02X}",
                    ss_curr.u16.sfn, ss_curr.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
            transport().tx_release(msg_desc);
            msg_desc.reset();
        }

        vector<cumac_req_t*>& cumac_reqs = get_cumac_req_list(cell_id, ss, group_id);

        if (cumac_reqs.size() == 0)
        {
            // Empty slot, skip
            return 0;
        }

        switch(group_id)
        {
            case CUMAC_SCH_TTI_REQ: { //TODO
                if(data_buf_opt == 1)
                {
                    msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
                }
                else if(data_buf_opt == 2)
                {
                    NVLOGI_FMT(TAG, "Cannot use GPU pools for TX DATA yet");
                    msg_desc.data_pool = NV_IPC_MEMPOOL_CUDA_DATA;
                    return -1;
                }
                else if(data_buf_opt == 3)
                {
                    //NVLOGI_FMT(TAG, "Creating GPU pools(with GDR copy) for TX DATA data_buf_opt = {}",data_buf_opt);
                    msg_desc.data_pool = NV_IPC_MEMPOOL_GPU_DATA;
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Must use separate buffers in SCF!");
                    return -1;
                }

                if(transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGF_FMT(TAG, AERIAL_TEST_MAC_EVENT, "SFN {}.{} allocate nvipc buffer failed: cell_id={} msg_buf={}",
                            ss_curr.u16.sfn, ss_curr.u16.slot, msg_desc.cell_id, msg_desc.msg_buf);
                    return 0;
                }

                cumac_sch_tti_req_t* req = cumac_init_msg_header<cumac_sch_tti_req_t>(&msg_desc, CUMAC_SCH_TTI_REQUEST, cell_id);
                req->sfn = ss.u16.sfn;
                req->slot = ss.u16.slot;

                build_sch_tti_request(cell_id, cumac_reqs, *req, msg_desc);
            }
            break;

            default: {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unknown CUMAC group_id={}", static_cast<unsigned>(group_id));
                return 0;
            }
            break;
        }
        cumac_sched.cumac_build_num ++;
    }

    // Send or cache the CUMAC message
    int cumac_sent_count = 0;

    // Get the default CUMAC delay configured by yaml
    std::vector<int32_t>& target_ts_offsets = *(cumac_sched.target_ts_offsets[group_id]);
    int32_t target_ts_offset = target_ts_offsets[get_slot_in_frame(ss) % target_ts_offsets.size()];

#if 0 // Only for debug: overwrite default CUMAC delay configuration for specific cell_id, SFN, SLOT and CUMAC message
    if (cell_id == 2 && ss.u16.slot == 0 && group_id == UL_TTI_REQ)
    {
        target_ts_offset = 0;
    }
#endif

    NVLOGD_FMT(TAG, "SFN {}.{} {}: cell_id={} group_id={} ts_offset={} target_ts_offset={}",
            ss.u16.sfn, ss.u16.slot, __func__, cell_id, +group_id, ts_offset, target_ts_offset);

    if (msg_desc.msg_buf != nullptr && (ts_offset >= target_ts_offset || ts_offset > 0))
    {
        NVLOGI_FMT(TAG, "SFN {}.{} SEND: cell_id={} msg_id=0x{:02X} {}", ss.u16.sfn,  ss.u16.slot, cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

        transport().tx_send(msg_desc);
        cumac_sent_count ++;
        cumac_sched.cumac_sent_num ++;

        msg_desc.reset();
        if (notify_mode == IPC_SYNC_PER_MSG) {
            // Notify once every CUMAC message
            transport().notify(1);
        }
    }

    return cumac_sent_count;
}

int cumac_handler::send_tti_end(int cell_id, sfn_slot_t& ss)
{
    nv::phy_mac_msg_desc msg_desc;
    if(transport().tx_alloc(msg_desc) < 0)
    {
        return -1;
    }

    auto req = cumac_init_msg_header<cumac_tti_end_t>(&msg_desc, CUMAC_TTI_END, cell_id);
    req->sfn = ss.u16.sfn;
    req->slot = ss.u16.slot;
    req->end_param = 0;

    NVLOGI_FMT(TAG, "SFN {}.{} SEND: cell_id={} msg_id=0x{:02X} {}", req->sfn, req->slot, cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transport().tx_send(msg_desc);

    if(notify_mode == IPC_SYNC_PER_MSG)
    {
        // Notify once every CUMAC message
        transport().notify(1);
    }
    return 0;
}

/*
 * Function schedule_cumac_reqs() may be called multi-times for 1 slot:
 * (1) If builder_thread_enable = 1, call one time when receiving SLOT.ind of previous slot
 * (2) Call one time when receiving SLOT.ind of current slot
 * (3) If cumac_stt > 0, call one time at SLOT.ind + cumac_stt
 */
int cumac_handler::schedule_cumac_reqs(sfn_slot_t ss, int ts_offset, int specified_cell_id)
{
    int total_count = 0;
    int32_t delay_time_idx = get_slot_in_frame(ss) % cumac_configs->cumac_stt.size();
    int32_t cumac_stt = cumac_configs->cumac_stt[delay_time_idx];
    for (int cell_id = 0; cell_id < cell_num; cell_id ++)
    {
        if(specified_cell_id >= 0 && specified_cell_id != cell_id)
        {
            // If specified_cell_id is non-negative value, only scheudle specified cell requests
            continue;
        }

        cumac_sched_t &cumac_sched = cumac_cell_data[cell_id].cumac_scheds[ss.u16.slot & 0x3];
        if (ts_offset < 0 || (ts_offset == 0 && cumac_build_in_advance == 0))
        {
            // Reset the cumac_scheds buffer
            // cumac_cell_data[cell_id].cumac_scheds[ss.u16.slot & 0x3].reset();
            for (nv::phy_mac_msg_desc& msg_desc:  cumac_sched.cumac_msg_cache)
            {
                if (msg_desc.msg_buf != nullptr) {
                    sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
                    NVLOGI_FMT(TAG, "Current SFN {}.{} dropping CUMAC: SFN {}.{} cell_id={} msg_id=0x{:02X}",
                            ss.u16.sfn, ss.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
                    transport().tx_release(msg_desc);
                    msg_desc.reset();
                }
            }
        }

        if (cumac_cell_data[cell_id].schedule_enable == false)
        {
            // Skip stopped cells
            continue;
        }

        total_count += 1;

        int cumac_count = 0;
        for (int group_id = 0; group_id < CUMAC_REQ_SIZE; group_id++)
        {
            cumac_count += schedule_cumac_request(cell_id, ss, (cumac_group_t)group_id, ts_offset);
        }

        if (cumac_count > 0 && cumac_sched.cumac_sent_num == cumac_sched.cumac_build_num)
        {
#ifdef ENABLE_L2_SLT_RSP
            send_tti_end(cell_id, ss);
            cumac_count++;
#else
            // Cell ended
            if (notify_mode == IPC_SYNC_PER_CELL)
            {
                // Notify once every cell
                NVLOGI_FMT(TAG, "SFN {}.{} NOTIFY: cell_id={} cumac_build_num={} cumac_sent_num={}", ss.u16.sfn, ss.u16.slot, cell_id, cumac_sched.cumac_build_num, cumac_sched.cumac_sent_num);
                transport().notify(cumac_sched.cumac_sent_num);
            }
#endif
            total_count += cumac_count;
        }

        if (notify_mode == IPC_SYNC_PER_CELL && cumac_stt > 0 && cell_id == (cell_num-1) && ts_offset > 0)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} NOTIFY: cell_id={} cumac_build_num={} cumac_sent_num={}", ss.u16.sfn, ss.u16.slot, cell_id, cumac_sched.cumac_build_num, cumac_sched.cumac_sent_num);
            // Trigger the last sync event after sleeping to ts_offset
            transport().notify(cumac_sched.cumac_sent_num);
        }
    }
    return total_count;
}

int cumac_handler::schedule_slot(sfn_slot_t ss)
{

    //Variables used for MAC.PROCESSING_TIMES message
    std::chrono::nanoseconds cumac1_start(0);
    std::chrono::nanoseconds cumac1_stop(0);
    int cumac1_count = 0;
    std::chrono::nanoseconds cumac2_start(0);
    std::chrono::nanoseconds cumac2_stop(0);
    int cumac2_count = 0;
    std::chrono::nanoseconds notify_start(0);
    std::chrono::nanoseconds notify_stop(0);

    // Schedule non delaying CUMAC messages
    cumac1_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    int total_sent = schedule_cumac_reqs(ss, 0);
    cumac1_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    cumac1_count = total_sent;

    auto t_last = std::chrono::system_clock::now().time_since_epoch();

    // Sleep for delaying CUMAC messages / event notification if configured
    int t_sleep_ns = 0;

    int32_t delay_time_idx = get_slot_in_frame(ss) % cumac_configs->cumac_stt.size();
    int32_t cumac_stt = cumac_configs->cumac_stt[delay_time_idx];

    // Determine CUMAC message sending start time
    int64_t schedule_start_time = cumac_stt;

    //Determine tick time
    uint64_t t_slot = sfn_to_tai(ss.u16.sfn, ss.u16.slot, ts_tick + AppConfig::getInstance().getTaiOffset(), 0, 0, 1) - AppConfig::getInstance().getTaiOffset();
    t_slot -= SLOT_ADVANCE*SLOT_TIME_BOUNDARY_NS; // Subtract off tick slot advance.

    if (cumac_stt > 0) {
        // Force L2 scheduler to take the actual L2 scheduling time plus whatever t_sleep adds up to the desired total time.
        // Also include the offset due to SLOT.indication reception latency, assuming taking credit for any time after
        // the slot time boundary.  NB: This code only works for PTP GPS_ALPHA = GPS_BETA = 0.  For non-zero values, a
        // correction factor is necessary.

        int t_slot_modulo = ts_tick % SLOT_TIME_BOUNDARY_NS;
        t_sleep_ns = schedule_start_time - (t_last.count() - t_slot);
        NVLOGI_FMT(TAG,"Sleeping {} for SFN.slot {}.{} tick {} from {}",t_sleep_ns,ss.u16.sfn,ss.u16.slot,t_slot,t_last.count());
        if (t_sleep_ns < 0)
        {
            t_sleep_ns = 0;
        }
        struct timespec t_sleep_tspec;
        t_sleep_tspec.tv_sec = 0;
        t_sleep_tspec.tv_nsec = t_sleep_ns;
        nanosleep(&t_sleep_tspec, NULL);

        // Schedule delayed CUMAC messages and event notify
        cumac2_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        total_sent += schedule_cumac_reqs(ss, cumac_stt);
        cumac2_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        cumac2_count = total_sent - cumac1_count;
    }

    notify_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    if(total_sent > 0 && notify_mode == IPC_SYNC_PER_TTI)
    {
        // Notify once every TTI tick
        transport().notify(total_sent);
    }
    notify_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto t_end = std::chrono::system_clock::now().time_since_epoch();
    NVLOGI_FMT(TAG, "{{TI}} <testCUMAC Scheduler,{},{},0,0> Start Task:{},Last Cell:{},Sleep:{},End Task:{},sleep_ns={} cumac_stt={}",
            ss.u16.sfn, ss.u16.slot, ts_tick, t_last.count(), t_last.count() + t_sleep_ns, t_end.count(), t_sleep_ns, cumac_stt);

    NVLOGI_FMT(TAG, "SFN {}.{} {} tick={} slot_indication={} cumac1_start={} cumac1_stop={} cumac1_count={} sleep_time={} cumac2_start={} cumac2_stop={} cumac2_count={} notify_start={} notify_stop={} slot_total={}",
               ss.u16.sfn, ss.u16.slot, __func__,
               t_slot,
               ts_tick,
               cumac1_start.count(),cumac1_stop.count(),cumac1_count,
               t_sleep_ns,
               cumac2_start.count(),cumac2_stop.count(),cumac2_count,
               notify_start.count(),notify_stop.count(),
               t_end.count() - cumac1_start.count());

    return total_sent;
}
