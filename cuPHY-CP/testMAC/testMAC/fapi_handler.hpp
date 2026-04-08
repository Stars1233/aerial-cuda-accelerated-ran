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

#ifndef _FAPI_HANDLER_HPP_
#define _FAPI_HANDLER_HPP_

#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <semaphore.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <list>
#include "nv_mac.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "nv_phy_fapi_msg_common.hpp"
#include "fapi_defines.hpp"
#include "fapi_validate.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "test_mac_stats.hpp"
#include "stat_log.h"
#include "nv_ipc_ring.h"

using namespace std;
using namespace nv;

#define END_REQUEST_NONE (0)
#define END_REQUEST_PER_SLOT (1)
#define END_REQUEST_PER_CELL (2)

#define FAPI_MAX_UE 64 
#define FAPI_MAX_UL_HARQ_ID 16 // Refer to SCF 222, harqProcessID value range: 0 -> 15
#define FAPI_MAX_HARQ_RETX 3

// For STT estimation feature
#define STT_STATISTIC_PERIOD (2000)       // Statistic period for calculating min, max, average time cost. Unit: number of slots
#define STT_ESTIMATE_COUNT_MAX (3)        // How many times to estimate
#define STT_ESTIMATE_ADJUST_OFFSET (2500) // Jitter to adjust including nanosleep wakeup time cost. Unit: ns.

// For FAPI TX deadline feature.
static constexpr int DEADLINE_STATISTIC_PERIOD = 2000; // Statistic period for calculating min, max, average deadline difference. Unit: number of slots
static constexpr int DEADLINE_DIFF_MAX_NS = 30 * 1000; // Max deadline difference for statistic logging. Unit: ns.
static constexpr int MAX_SCHEDULE_AHEAD_TIME_NS = 15 * 1000; // Max ahead time for FAPI TX deadline scheduling. Unit: ns.
static constexpr int AVG_SCHEDULE_AHEAD_TIME_NS = 10 * 1000; // Average ahead time for FAPI TX deadline scheduling. Unit: ns.
static constexpr int SYSTEM_WAKEUP_TIME_COST_NS = 2500; // Average system wakeup time cost (system timer/thread context switch time). Unit: ns

// This is required to ensure the FAPI TX deadline scheduling is valid.
static_assert(MAX_SCHEDULE_AHEAD_TIME_NS > AVG_SCHEDULE_AHEAD_TIME_NS + SYSTEM_WAKEUP_TIME_COST_NS,
    "MAX_SCHEDULE_AHEAD_TIME_NS must be greater than AVG_SCHEDULE_AHEAD_TIME_NS + SYSTEM_WAKEUP_TIME_COST_NS");

enum class fapi_state_t
{
    IDLE       = 0,
    CONFIGURED = 1,
    RUNNING    = 2,
    INVALID    = 3,
};

typedef struct ul_harq_handle_t
{
public:
    ul_harq_handle_t()
    {
        harq_id = 0;
        retx_count = 0;
        rv = 0;
        pusch_pdu_NDI = 0;
        ul_dci_NDI = 0;
    }

    uint8_t harq_id;
    uint8_t retx_count;
    uint8_t rv;
    uint8_t pusch_pdu_NDI;
    uint8_t ul_dci_NDI;
    /* data */
}ul_harq_handle_t;

#define FAPI_SCHED_BUFFER_SIZE 4
struct fapi_sched_t
{
public:
    fapi_sched_t() :
        fapi_build_num(0),
        fapi_sent_num(0),
        zero_target_ts_offsets({0})
    {
        for (int i = 0; i < FAPI_REQ_SIZE; i++)
        {
            fapi_msg_cache[i].reset();
            target_ts_offsets[i] = &zero_target_ts_offsets;
        }
    }

    // Initiate everything except target_ts_offset
    void reset()
    {
        fapi_build_num = 0;
        fapi_sent_num = 0;
        for (int i = 0; i < FAPI_REQ_SIZE; i++)
        {
            fapi_msg_cache[i].reset();
        }
    }

    int32_t fapi_build_num;
    int32_t fapi_sent_num;
    std::vector<int32_t> zero_target_ts_offsets;
    std::vector<int32_t> *target_ts_offsets[FAPI_REQ_SIZE];

    nv::phy_mac_msg_desc fapi_msg_cache[FAPI_REQ_SIZE];
};

class cell_data_t {
public:
    cell_data_t()
    {
        cell_id_remap   = 0;
        prach_state     = 0;
        prach_prambIdx  = 0;
        harq_process_id = 0;
        restart_timer   = nullptr;
        schedule_enable = false;
        fapi_state.store(fapi_state_t::IDLE);
        pending_config.store(0);
    }

    cell_data_t(const cell_data_t& obj)
    {
        cell_id_remap   = obj.cell_id_remap;
        prach_state     = obj.prach_state;
        prach_prambIdx  = obj.prach_prambIdx;
        harq_process_id = obj.harq_process_id;
        restart_timer   = obj.restart_timer;
        schedule_enable = false;
        fapi_state.store(obj.fapi_state.load());
        pending_config.store(0);
    }

    ~cell_data_t() {}

    int cell_id_remap;

    uint16_t prach_state;
    int8_t   prach_prambIdx;
    uint8_t  harq_process_id;

    std::atomic<bool> schedule_enable;

    // Pending to send a CONFIG.request
    std::atomic<uint32_t> pending_config;

    // Timer for restart single cell
    timer_t restart_timer = nullptr;

    std::atomic<fapi_state_t> fapi_state = fapi_state_t::IDLE;

    std::mutex queue_mutex;

    std::queue<nv::phy_mac_msg_desc> msg_queues;

    fapi_sched_t fapi_scheds[FAPI_SCHED_BUFFER_SIZE];

    std::vector<slot_timing_t> slot_timing;
};

typedef struct
{
    int         estimate_counter;
    int         statistic_period;
    int         estimate_send_time;
    stat_log_t* stat_send; // Statistic for average time cost of 1 sending message
    stat_log_t* stat_diff; // Statistic for difference between the expected scheduling end time and actual end time
} stt_estimate_t;

class fapi_handler {
public:
    fapi_handler(phy_mac_transport& transport, test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats* conformance_test_stats);
    virtual ~fapi_handler();

    virtual void cell_init(int cell_id)       = 0;
    virtual void cell_start(int cell_id)      = 0;
    virtual void cell_stop(int cell_id)       = 0;
    virtual void on_msg(nv_ipc_msg_t& msg)    = 0;
    virtual int  schedule_slot(sfn_slot_t ss) = 0;

    /**
     * Stop all cells and exit the FAPI handler
     */
    virtual void terminate() = 0;

    virtual void builder_thread_func() = 0;

    virtual void scheduler_thread_func() = 0;

    // Worker thread function
    virtual void worker_thread_func() = 0;

    virtual void notify_worker_threads() = 0;

    virtual int send_config_request(int cell_id) = 0;
    virtual int send_start_request(int cell_id)  = 0;
    virtual int send_stop_request(int cell_id)   = 0;

    void update_dl_thrput(uint32_t sfn, uint32_t slot, uint64_t slot_counter);
    void print_thrput(uint64_t slot_counter);

    int restart(int cell_id);
    int schedule_cell_update(uint64_t slot_counter);

    int reconfig(int cell_id);
    int start_reconfig_timer(int cell_id, uint32_t interval_ms);
    int stop_reconfig_timer(int cell_id);
    int get_pending_config_cell_id();

    // OAM negative test
    int set_fapi_delay(int cell_id, int slot, int fapi_mask, int delay_us);

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
            if(cell_data[cell_id].fapi_state == fapi_state_t::RUNNING)
            {
                return false;
            }
        }
        return true;
    }

    fapi_state_t get_fapi_state(int cell_id)
    {
        if(cell_id < cell_data.size())
        {
            return cell_data[cell_id].fapi_state;
        }
        else
        {
            return fapi_state_t::INVALID;
        }
    }

    cell_data_t* get_cell_data(int cell_id)
    {
        return &cell_data[cell_id];
    }

    thrput_t* get_thrput(int cell_id)
    {
        return &thrputs[cell_id];
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

    void set_rnti_test_mode(int test_mode)
    {
        rnti_test_mode = test_mode;
    }

    int get_rnti_test_mode()
    {
        return rnti_test_mode;
    }

    test_mac_configs* get_configs()
    {
        return configs;
    }

    void set_first_init_flag(int cell_id, bool flag)
    {
        first_init[cell_id] = flag;
    }

    void initSrsChestBufIdxQueue(uint32_t cell_id, bool isMimoEnabled);
    uint16_t getSrsChestBufIdx(uint32_t cell_id);
    void updateMapOfRntiToSrsChestBufIdx(uint32_t cell_id, uint32_t rnti, uint16_t srsChestBufferIndex);
    void setMapOfSrsReqToIndRntiToSrsChestBufIdx(uint32_t cell_id, uint32_t rnti, uint16_t srsChestBufferIndex);
    uint16_t getMapOfSrsReqToIndRntiToSrsChestBufIdx (uint32_t cell_id, uint32_t rnti);
    uint16_t getSrsChestBufferIndexFromMapOfRnti(uint32_t cell_id, uint32_t rnti);
    void insertIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex);
    bool isIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex);
    void removeIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex);

protected:
    phy_mac_transport& transport()
    {
        return _transport;
    }

    void slot_indication_handler(uint32_t sfn, uint32_t slot, uint64_t slot_counter);
    void static_ul_dl_scheduler(uint32_t sfn, uint32_t slot, uint64_t slot_counter);
    void dump_conformance_test_stats();

    int schedule_all_cell_restart(uint64_t slot_counter);
    int schedule_single_cell_restart(uint64_t slot_counter);

    int set_prach_msg2_rapid(int cell_id, uint8_t* rapid);
    int get_pramble_params(int cell_id, uint16_t sfn, uint16_t slot, preamble_params_t* preamble);

    vector<fapi_req_t*>& get_fapi_req_list(int cell_id, sfn_slot_t ss, fapi_group_t group_id);

    fapi_req_t* get_fapi_req_data(int cell_id, uint16_t sfn, uint16_t slot, channel_type_t channel);

    pusch_tv_data_t* get_pusch_tv_pdu(fapi_req_t* pusch_req, int pdu_id, int bitmap);
    pucch_tv_data_t* get_pucch_tv_pdu(fapi_req_t* pucch_req, int pdu_id);
    pucch_tv_data_t* get_pucch_tv_pf234_pdu(fapi_req_t* pucch_req, int pdu_id);

    srs_tv_data_t* get_srs_tv_pdu(fapi_req_t* srs_req, uint32_t handle);

    int get_pucch_tv_pf234_pdu_num(fapi_req_t* pucch_req);

    int set_restart_timer(int cell_id, int interval);

    int data_buf_opt = 1; //!< Data buffer option: 0=msg_buf, 1=CPU_DATA, 2=CUDA_DATA, 3=GPU_DATA

    int cell_num = 0;

    int notify_mode = 0;

    uint16_t slots_per_frame = 0;

    // The cell_id for cell which is in configuring
    std::atomic<int32_t> current_config_cell_id;
    std::atomic<int32_t> config_retry_counter;

    sem_t slot_sem; // FAPI builder thread wait on it to start building FAPI message
    sem_t fapi_sched_ready; // fapi_scheds buffer is ready for next slot
    sem_t fapi_sched_free; // Slot scheduled, free fapi_scheds buffer

    sem_t scheduler_sem; // Scheduler thread wait on it to start scheduling a slot
    nv_ipc_ring_t* sched_info_ring; // Slot info pending to schedule

    std::atomic<sfn_slot_t> ss_tick; //!< SFN/SLOT of the tick
    int64_t                 ts_tick; //!< Time stamp of the tick
    int64_t                 ts_tick_tai = 0; //!< The ideal tick time stamp aligned to TAI
    sfn_slot_t              ss_fapi_last; //!< SFN/SLOT of the last slot

    uint64_t global_tick = 0; //!< Global tick counter
    uint64_t conformance_test_slot_counter = 0; //!< Conformance test slot counter
    uint16_t beam_id_position = 0; //!< Beam ID position

    // The throughput test result
    std::vector<thrput_t> thrputs;

    // summary test result
    std::vector<results_summary_t> cell_summary;

    // The status data for each cell
    std::vector<cell_data_t> cell_data;

    // Cell first initialization flag, MUST send CONFIG.request when first_init == true
    std::vector<int> first_init;

    test_mac_configs*  configs = nullptr;
    launch_pattern*    lp = nullptr;
    phy_mac_transport& _transport;

    //For OAM cell re-attaching to different RUs test, the idea is to replace current cell FAPIs with these of the target cell

    std::unordered_map<int, int> cell_id_map;
    std::unordered_map<int, int> cell_id_map_tmp;
    std::unordered_map<int, bool> cell_remap_event;
    std::mutex srsChestBufQueueMutex[MAX_CELLS_PER_SLOT];
    std::mutex mapOfRntiToSrsChestBufIdxMutex[MAX_CELLS_PER_SLOT];
    std::mutex mapOfSrsReqToIndRntiToSrsChestBufIdxMutex[MAX_CELLS_PER_SLOT];
    std::mutex mapBfwSrsChestBufIdxListMutex[MAX_CELLS_PER_SLOT];
    std::unordered_map<uint32_t, std::queue<uint16_t>> mapOfSrsChestBufIdx;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t,uint16_t>> mapOfRntiToSrsChestBufIdx{};
    std::unordered_map<uint32_t, std::unordered_map<uint32_t,uint16_t>> mapOfSrsReqToIndRntiToSrsChestBufIdx{};
    std::unordered_map<uint32_t, std::list<uint16_t>> mapBfwSrsChestBufIdxList{};
    

    // Timer for restart all cells
    timer_t restart_timer = nullptr;

    // Timer for check cell config status
    timer_t reconfig_timer = nullptr;

    ch8_conformance_test_stats* conformance_test_stats = nullptr;

    // fapi_validate vald;

    stat_log_t* schedule_time = nullptr;
    stat_log_t* stat_debug = nullptr;

    stat_log_t* uci_time = nullptr;
    stat_log_t* crc_time = nullptr;
    stat_log_t* rx_data_time = nullptr;
    stat_log_t* srs_ind_time = nullptr;

    stat_log_t* stat_stt_send = nullptr; // Statistic for average time cost of 1 sending message
    stat_log_t* stat_stt_diff = nullptr; // Statistic for difference between the expected scheduling end time and actual end time

    stat_log_t* stat_deadline_diff = nullptr; // Statistic for difference between the expected FAPI deadline and actual FAPI sending end time
    stat_log_t* stat_tx_per_msg = nullptr; // Statistic for average time cost of 1 sending message

    std::vector<stt_estimate_t> stt_estimate;

    // Negative test flags
    int rnti_test_mode = 0; // 0 - default, normal mode; 1 - negative test

    ul_harq_handle_t ul_harq_handle[MAX_CELLS_PER_SLOT][FAPI_MAX_UE][FAPI_MAX_UL_HARQ_ID];
    uint8_t ul_dci_freq_domain_bits = 0; 
    std::map<uint16_t, uint16_t> rnti_2_ueid_map;
};

fapi_handler* get_fapi_handler_instance();

#endif /* _FAPI_HANDLER_HPP_ */
