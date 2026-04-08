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

#ifndef _CUMAC_HANDLER_HPP_
#define _CUMAC_HANDLER_HPP_

#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <semaphore.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>

#include "nv_mac.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "cumac_defines.hpp"
#include "test_mac_configs.hpp"
#include "cumac_pattern.hpp"
#include "test_mac_stats.hpp"
#include "cumac_validate.hpp"
#include "stat_log.h"
#include "nv_ipc_ring.h"

#include "cumac_msg.h"

using namespace std;
using namespace nv;

// cuMAC end request modes
#define END_REQUEST_NONE (0)     //!< No end request sent
#define END_REQUEST_PER_SLOT (1) //!< Send end request per slot
#define END_REQUEST_PER_CELL (2) //!< Send end request per cell

/**
 * cuMAC cell state enumeration
 * 
 * Tracks the lifecycle state of a cuMAC cell from initialization through operation
 */
enum class cumac_state_t
{
    IDLE       = 0, //!< Cell is idle, not configured
    CONFIGURED = 1, //!< Cell is configured but not running
    RUNNING    = 2, //!< Cell is actively running
    INVALID    = 3, //!< Invalid state
};

#define CUMAC_SCHED_BUFFER_SIZE 4 //!< Size of cuMAC scheduling buffer ring

/**
 * cuMAC scheduling data structure
 * 
 * Manages cuMAC message building and sending for a single slot.
 * Uses a ring buffer approach with CUMAC_SCHED_BUFFER_SIZE entries.
 */
struct cumac_sched_t
{
public:
    cumac_sched_t() :
        cumac_build_num(0),
        cumac_sent_num(0),
        zero_target_ts_offsets({0})
    {
        for (int i = 0; i < CUMAC_REQ_SIZE; i++)
        {
            cumac_msg_cache[i].reset();
            target_ts_offsets[i] = &zero_target_ts_offsets;
        }
    }

    /**
     * Reset scheduling state for new slot
     * Note: Preserves target_ts_offset configuration
     */
    void reset()
    {
        cumac_build_num = 0;
        cumac_sent_num = 0;
        for (int i = 0; i < CUMAC_REQ_SIZE; i++)
        {
            cumac_msg_cache[i].reset();
        }
    }

    int32_t cumac_build_num;  //!< Number of cuMAC messages built for this slot
    int32_t cumac_sent_num;   //!< Number of cuMAC messages sent for this slot
    std::vector<int32_t> zero_target_ts_offsets; //!< Default zero offset vector
    std::vector<int32_t> *target_ts_offsets[CUMAC_REQ_SIZE]; //!< Target timestamp offsets for each request type

    nv::phy_mac_msg_desc cumac_msg_cache[CUMAC_REQ_SIZE]; //!< Cached cuMAC message descriptors
};

/**
 * cuMAC per-cell data structure
 * 
 * Contains all runtime state and scheduling data for a single cuMAC cell.
 * Thread-safe through atomic variables and mutexes.
 */
class cumac_cell_data_t {
public:
    cumac_cell_data_t()
    {
        cell_id_remap   = 0;
        prach_state     = 0;
        prach_prambIdx  = 0;
        harq_process_id = 0;
        restart_timer   = nullptr;
        schedule_enable = false;
        cumac_state.store(cumac_state_t::IDLE);
        pending_config.store(0);
    }

    cumac_cell_data_t(const cumac_cell_data_t& obj)
    {
        cell_id_remap   = obj.cell_id_remap;
        prach_state     = obj.prach_state;
        prach_prambIdx  = obj.prach_prambIdx;
        harq_process_id = obj.harq_process_id;
        restart_timer   = obj.restart_timer;
        schedule_enable = false;
        cumac_state.store(obj.cumac_state.load());
        pending_config.store(0);
    }

    ~cumac_cell_data_t() {}

    int cell_id_remap; //!< Remapped cell ID for OAM cell re-attaching test

    uint16_t prach_state;     //!< PRACH state machine state
    int8_t   prach_prambIdx;  //!< PRACH preamble index
    uint8_t  harq_process_id; //!< Current HARQ process ID

    std::atomic<bool> schedule_enable; //!< Enable/disable scheduling for this cell

    std::atomic<uint32_t> pending_config; //!< Flag indicating pending CONFIG.request

    timer_t restart_timer; //!< Timer for single cell restart

    std::atomic<cumac_state_t> cumac_state = cumac_state_t::IDLE; //!< Current cuMAC state for this cell

    std::mutex queue_mutex; //!< Mutex protecting message queue

    std::queue<nv::phy_mac_msg_desc> msg_queues; //!< Queue of pending messages

    cumac_sched_t cumac_scheds[CUMAC_SCHED_BUFFER_SIZE]; //!< Ring buffer of scheduling data

    std::vector<slot_timing_t> slot_timing; //!< Timing information per slot
};

/**
 * cuMAC handler class
 * 
 * Main class for managing cuMAC (CUDA-accelerated MAC) operations including:
 * - Multi-cell cuMAC message scheduling and building
 * - Message transport and IPC communication
 * - Cell lifecycle management (init, start, stop, restart)
 * - Worker thread management for parallel message building
 * - Slot timing and synchronization
 */
class cumac_handler {
public:
    /**
     * Construct cuMAC handler
     * 
     * @param[in] configs Test MAC configurations
     * @param[in] pattern_file Path to cuMAC pattern file
     * @param[in] ch_mask Channel mask for enabled channels
     * @param[in] cell_mask Bit mask of enabled cells
     */
    cumac_handler(test_mac_configs* configs, const char* pattern_file, uint32_t ch_mask, uint64_t cell_mask);
    ~cumac_handler();

    /**
     * Start cuMAC handler threads (receiver, scheduler, builder, workers)
     */
    void start();
    
    /**
     * Stop cuMAC handler threads
     */
    void stop();
    
    /**
     * Wait for cuMAC receiver thread to complete
     */
    void join();

    /**
     * Initialize a cell - send CONFIG.request
     * 
     * @param[in] cell_id Cell ID to initialize
     */
    void cell_init(int cell_id);
    
    /**
     * Start a configured cell - send START.request
     * 
     * @param[in] cell_id Cell ID to start
     */
    void cell_start(int cell_id);
    
    /**
     * Stop a running cell - send STOP.request
     * 
     * @param[in] cell_id Cell ID to stop
     */
    void cell_stop(int cell_id);
    
    /**
     * Process received cuMAC message
     * 
     * @param[in] msg_desc Message descriptor containing received message
     */
    void on_msg(nv::phy_mac_msg_desc& msg_desc);

    /**
     * Handle tick event for slot scheduling
     * 
     * @param[in] sched_info Scheduling information with SFN/slot and timestamp
     */
    void on_tick_event(sched_info_t sched_info);

    /**
     * Schedule all cuMAC messages for a slot
     * 
     * @param[in] ss SFN/slot to schedule
     * @return Number of messages sent
     */
    int schedule_slot(sfn_slot_t ss);
    
    /**
     * Schedule cuMAC requests for specific cells
     * 
     * @param[in] ss SFN/slot to schedule
     * @param[in] ts_offset Timestamp offset for delayed sending
     * @param[in] specified_cell_id Specific cell to schedule, -1 for all cells
     * @return Number of messages sent
     */
    int schedule_cumac_reqs(sfn_slot_t ss, int ts_offset, int specified_cell_id = -1);

    /**
     * Receiver thread main function - handles IPC message reception
     */
    void receiver_thread_func();
    
    /**
     * Builder thread main function - pre-builds messages for next slot
     */
    void builder_thread_func();

    /**
     * Scheduler thread main function - schedules slot processing
     */
    void scheduler_thread_func();

    /**
     * Worker thread main function - parallel message building
     */
    void worker_thread_func();

    /**
     * Notify worker threads to start processing
     * 
     * @param[in] num Number of worker threads to notify
     */
    void notify_worker_threads(uint32_t num = 1);

    int poll_build_task();

    void build_first_slot();

    int build_sch_tti_request(int cell_id, vector<cumac_req_t*>& cumac_reqs, cumac_sch_tti_req_t& req, phy_mac_msg_desc& msg_desc);

    int send_config_request(int cell_id);
    int send_start_request(int cell_id);
    int send_stop_request(int cell_id);
    int send_tti_end(int cell_id, sfn_slot_t& ss);

    int handle_sch_tti_response(nv::phy_mac_msg_desc& msg_desc, cumac_validate& vald);

    void update_dl_cumac_thrput(uint32_t sfn, uint32_t slot, uint64_t slot_counter);
    void print_cumac_thrput(uint64_t slot_counter);

    int restart(int cell_id);
    int schedule_cell_update(uint64_t slot_counter);

    int reconfig(int cell_id);
    int start_reconfig_timer(int cell_id, uint32_t interval_ms);
    int stop_reconfig_timer(int cell_id);
    int get_pending_config_cell_id();

    // OAM negative test
    int set_cumac_delay(int cell_id, int slot, int cumac_mask, int delay_us);

    bool cell_id_sanity_check(int cell_id);

    int schedule_cumac_request(int cell_id, sfn_slot_t ss, cumac_group_t group_id, int32_t ts_offset);

    vector<cumac_req_t*>& get_cumac_req_list(int cell_id, sfn_slot_t ss, cumac_group_t group_id);

    cumac_req_t* get_cumac_req_data(int cell_id, uint16_t sfn, uint16_t slot);

    phy_mac_transport& transport()
    {
        return *_transport;
    }

    void setTransport(phy_mac_transport* transport)
    {
        _transport = transport;
    }

    test_cumac_configs* get_cumac_configs() {
        return testmac_configs->cumac_configs;
    }

    int get_cell_num()
    {
        return cell_num;
    }

    int get_slots_per_frame()
    {
        return slots_per_frame;
    }

    int cell_id_remap(int src_cell, int target_cell)
    {
        if(cell_remap_event[src_cell])
        {
            return -1;
        }
        cell_id_map_tmp[src_cell] = target_cell;
        cell_remap_event[src_cell] = true;
        return 0;
    }

    bool is_stopped()
    {
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            if(cumac_cell_data[cell_id].cumac_state == cumac_state_t::RUNNING)
            {
                return false;
            }
        }
        return true;
    }

    cumac_state_t get_cumac_state(int cell_id)
    {
        if(cell_id < cumac_cell_data.size())
        {
            return cumac_cell_data[cell_id].cumac_state;
        }
        else
        {
            return cumac_state_t::INVALID;
        }
    }

    cumac_cell_data_t* get_cumac_cell_data(int cell_id)
    {
        return &cumac_cell_data[cell_id];
    }

    cumac_thrput_t* get_cumac_thrput(int cell_id)
    {
        return &cumac_thrputs[cell_id];
    }

    sfn_slot_t get_next_sfn_slot(sfn_slot_t& ss)
    {
        sfn_slot_t next = ss;
        next.u16.slot++;
        if(next.u16.slot >= slots_per_frame)
        {
            next.u16.slot = 0;
            next.u16.sfn  = next.u16.sfn >= FAPI_SFN_MAX - 1 ? 0 : next.u16.sfn + 1;
        }
        return next;
    }

    uint32_t get_slot_in_frame(sfn_slot_t& ss)
    {
        uint32_t index = ss.u16.sfn; // extend to uint32_t
        return index * slots_per_frame + ss.u16.slot;
    }

    uint32_t get_slot_in_frame(uint16_t sfn, uint16_t slot)
    {
        uint32_t index = sfn; // extend to uint32_t
        return index * slots_per_frame + slot;
    }

protected:

    void slot_indication_handler(sfn_slot_t& ss, uint64_t slot_counter);
    void static_ul_dl_scheduler(uint32_t sfn, uint32_t slot, uint64_t slot_counter);

    int schedule_all_cell_restart(uint64_t slot_counter);
    int schedule_single_cell_restart(uint64_t slot_counter);

    int set_restart_timer(int cell_id, int interval);

    template <typename T>
    void print_array(phy_mac_msg_desc& msg_desc, const char *info, T *array, uint32_t num);

    template <typename T, typename U = T>
    int copy_to_ipc_buf(phy_mac_msg_desc& msg_desc, T* src, const char* debug_info, uint32_t& src_offset, uint32_t& dst_offset, uint32_t num);

    int copy_complex_to_ipc_buf(phy_mac_msg_desc& msg_desc, cuComplex* src, const char* debug_info, uint32_t& src_offset, uint32_t& dst_offset, uint32_t num);

    int data_buf_opt = 1; //!< Data buffer option: 0=msg_buf, 1=CPU_DATA, 2=CUDA_DATA, 3=GPU_DATA

    int cell_num = 0;            //!< Total number of cells
    int configured_cell_num = 0; //!< Number of cells successfully configured

    int notify_mode = 0; //!< IPC notification mode: per-cell, per-TTI, or per-message

    uint16_t slots_per_frame = 0; //!< Number of slots per frame (depends on numerology)

    // cuMAC thread IDs
    pthread_t cumac_recv_tid = 0;    //!< Receiver thread ID
    pthread_t cumac_sched_tid = 0;   //!< Scheduler thread ID
    pthread_t cumac_builder_tid = 0; //!< Builder thread ID
    pthread_t cumac_tick_tid = 0;    //!< Tick thread ID (for standalone mode)

    std::atomic<int32_t> current_config_cell_id = -1; //!< Cell ID currently being configured (-1 if none)
    std::atomic<int32_t> config_retry_counter = 0;   //!< Retry counter for cell configuration

    sem_t cumac_build_sem; // CUMAC builder thread wait on it to start building CUMAC message
    sem_t cumac_sched_ready; // cumac_scheds buffer is ready for next slot
    // sem_t cumac_sched_free; // Slot scheduled, free cumac_scheds buffer

    sem_t cumac_scheduler_sem; // Scheduler thread wait on it to start scheduling a slot
    nv_ipc_ring_t* sched_info_ring; // Slot info pending to schedule

    std::atomic<sfn_slot_t> ss_build; // SFN/SLOT of the current building tick for building or worker thread
    std::atomic<sfn_slot_t> ss_tick; // SFN/SLOT of the current scheduling tick
    int64_t                 ts_tick = 0; // Time stamp of the current scheduling tick
    sfn_slot_t              ss_cumac_last;

    uint64_t global_tick = 0;
    uint64_t conformance_test_slot_counter = 0;
    uint16_t beam_id_position = 0; //!< Beam ID position

    // The throughput test result
    std::vector<cumac_thrput_t> cumac_thrputs; //!< Throughput test result

    // summary test result
    // std::vector<results_summary_t> cell_summary;

    // The status data for each cell
    std::vector<cumac_cell_data_t> cumac_cell_data;

    // Cell first initialization flag, MUST send CONFIG.request when first_init == true
    std::vector<int> first_init;

    test_mac_configs*  testmac_configs = nullptr;
    test_cumac_configs* cumac_configs = nullptr;
    cumac_pattern*    lp = nullptr;
    phy_mac_transport* _transport = nullptr;

    //For OAM cell re-attaching to different RUs test, the idea is to replace current cell CUMACs with these of the target cell
    std::unordered_map<int, int> cell_id_map;
    std::unordered_map<int, int> cell_id_map_tmp;
    std::unordered_map<int, bool> cell_remap_event;

    int cumac_build_in_advance;

    // Worker thread inscreasing index and synchronization semaphore
    std::atomic<int> cumac_worker_id = 0;
    sem_t            cumac_worker_sem;

    std::atomic<int> cell_build_index   = 0; // Cell index to build cuMAC messages
    std::atomic<int> cell_build_counter = 0; // Cell counter which have finished building cuMAC messages

    // Timer for restart all cells
    timer_t restart_timer = nullptr;

    // Timer for check cell config status
    timer_t reconfig_timer = nullptr;

    stat_log_t* schedule_time = nullptr;
    stat_log_t* stat_debug = nullptr;

    stat_log_t* cumac_ul_time = nullptr;
};

cumac_handler* get_cumac_handler_instance();

#endif /* _CUMAC_HANDLER_HPP_ */
