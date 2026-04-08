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

#include "nv_ipc_ring.h"
#include "fapi_handler.hpp"
#include <iostream>
#include <fstream>

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 2) // "MAC.FAPI"

static fapi_handler* fapi_handler_instance = nullptr;

fapi_handler* get_fapi_handler_instance()
{
    return fapi_handler_instance;
}

static void reconfig_timer_handler(__sigval_t sigv)
{
    int cell_id = sigv.sival_int;
    NVLOGI_FMT(TAG, "{}: cell_id={}", __func__, cell_id);

    fapi_handler* _handler = get_fapi_handler_instance();
    _handler->reconfig(cell_id);
}

int fapi_handler::get_pending_config_cell_id()
{
    for (int cell_id = 0; cell_id < cell_num; cell_id ++)
    {
        if(cell_data[cell_id].pending_config.fetch_and(0) != 0)
        {
            return cell_id;
        }
    }
    return -1;
}

int fapi_handler::reconfig(int cell_id)
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

int fapi_handler::start_reconfig_timer(int cell_id, uint32_t interval_ms)
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

int fapi_handler::stop_reconfig_timer(int cell_id)
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

    fapi_handler* _handler = get_fapi_handler_instance();
    _handler->restart(cell_id);
}

int fapi_handler::restart(int cell_id)
{
    // Delete the timer
    timer_t* timer = cell_id == CELL_ID_ALL ? &restart_timer : &cell_data[cell_id].restart_timer;
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

int fapi_handler::set_restart_timer(int cell_id, int interval)
{
    if(interval <= 0 || (cell_id != CELL_ID_ALL && cell_id >= cell_data.size()))
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

    timer_t* timer = cell_id == CELL_ID_ALL ? &restart_timer : &cell_data[cell_id].restart_timer;
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

    void fapi_handler::initSrsChestBufIdxQueue(uint32_t cell_id, bool isMimoEnabled)
    {
        int maxSrsChestBufSize = isMimoEnabled ? MAX_SRS_CHEST_BUFFERS_PER_CELL : MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL;
        for (uint32_t idx = 0; idx< maxSrsChestBufSize; idx++)
        {
            uint32_t tempIdx = (idx + (maxSrsChestBufSize/2)) % maxSrsChestBufSize;
            mapOfSrsChestBufIdx[cell_id].push(tempIdx);
        }
        NVLOGC_FMT(TAG, "After {}: initialized mapOfSrsChestBufIdx of cell_id={} size={}", __func__, cell_id, mapOfSrsChestBufIdx[cell_id].size());
    }

    uint16_t fapi_handler::getSrsChestBufIdx(uint32_t cell_id)
    {
        if (mapOfSrsChestBufIdx[cell_id].empty())
        {
            NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "{}: mapOfSrsChestBufIdx Queue is Empty for cell_id={} !!", __func__, cell_id);
            return 0xFFFF;
        }
        uint16_t temp = mapOfSrsChestBufIdx[cell_id].front();
        mapOfSrsChestBufIdx[cell_id].pop();
        //NVLOGD_FMT(TAG, "{}: Before while mapOfSrsChestBufIdx for cell_id={} temp={} !!", __func__, cell_id, temp);
        while(isIdxInMapBfwSrsChestBufIdxList(cell_id, temp))
        {
            //NVLOGD_FMT(TAG, "{}: mapOfSrsChestBufIdx cell_id={} temp={} !!", __func__, cell_id, temp);
            mapOfSrsChestBufIdx[cell_id].push(temp);
            temp = mapOfSrsChestBufIdx[cell_id].front();
            mapOfSrsChestBufIdx[cell_id].pop();
        }
        return temp;
    }

    void fapi_handler::updateMapOfRntiToSrsChestBufIdx(uint32_t cell_id, uint32_t rnti, uint16_t srsChestBufferIndex)
    {
        auto it = mapOfRntiToSrsChestBufIdx[cell_id].find(rnti);
        if (it != mapOfRntiToSrsChestBufIdx[cell_id].end())
        {
            uint32_t old_index = mapOfRntiToSrsChestBufIdx[cell_id][rnti];
            mapOfSrsChestBufIdx[cell_id].push(old_index);
        }
        mapOfRntiToSrsChestBufIdx[cell_id][rnti] = srsChestBufferIndex;
        //NVLOGD_FMT(TAG, "{} cell_id={} rnti={} srsChestBufferIndex={}", __func__, cell_id, rnti, srsChestBufferIndex);
    }

    void fapi_handler::setMapOfSrsReqToIndRntiToSrsChestBufIdx(uint32_t cell_id, uint32_t rnti, uint16_t srsChestBufferIndex)
    {
        mapOfSrsReqToIndRntiToSrsChestBufIdx[cell_id][rnti] = srsChestBufferIndex;
    }

    uint16_t fapi_handler::getMapOfSrsReqToIndRntiToSrsChestBufIdx (uint32_t cell_id, uint32_t rnti)
    {
        auto it = mapOfSrsReqToIndRntiToSrsChestBufIdx[cell_id].find(rnti);
        if (it == mapOfSrsReqToIndRntiToSrsChestBufIdx[cell_id].end())
        {
            //NVLOGD_FMT(TAG, "{}: mapOfSrsReqToIndRntiToSrsChestBufIdx is Emprty for cell_id={} rnti={} !!", __func__, cell_id, rnti);
            return 0xFFFF;
        }
        return mapOfSrsReqToIndRntiToSrsChestBufIdx[cell_id].at(rnti);
    }

    uint16_t fapi_handler::getSrsChestBufferIndexFromMapOfRnti(uint32_t cell_id, uint32_t rnti)
    {
        mapOfRntiToSrsChestBufIdxMutex[cell_id].lock();
        auto it = mapOfRntiToSrsChestBufIdx[cell_id].find(rnti);
        if (it == mapOfRntiToSrsChestBufIdx[cell_id].end())
        {
            mapOfRntiToSrsChestBufIdxMutex[cell_id].unlock();
            //NVLOGD_FMT(TAG, "{}: mapOfRntiToSrsChestBufIdx is Emprty for cell_id={} rnti={} !!", __func__, cell_id, rnti);
            return 0xFFFF;
        }
        uint16_t result = mapOfRntiToSrsChestBufIdx[cell_id].at(rnti);
        mapOfRntiToSrsChestBufIdxMutex[cell_id].unlock();
        return result;
    }

    void fapi_handler::insertIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex)
    {
        //mapBfwSrsChestBufIdxList[cell_id].push_back(srsChestBufferIndex);
        // If cell_id does not exist, create an empty list
        if (mapBfwSrsChestBufIdxList.find(cell_id) == mapBfwSrsChestBufIdxList.end())
        {
            mapBfwSrsChestBufIdxList[cell_id] = {};
        }
        // Check for duplicates before inserting
        auto& lst = mapBfwSrsChestBufIdxList[cell_id];
        if (std::find(lst.begin(), lst.end(), srsChestBufferIndex) == lst.end())
        {
            lst.push_back(srsChestBufferIndex);
            NVLOGD_FMT(TAG, "{}: mapOfRntiToSrsChestBufIdx is Emprty for cell_id={} srsChestBufferIndex={} !!", __func__, cell_id, srsChestBufferIndex);
        }
    }

    void fapi_handler::removeIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex)
    {
        // Check if the cell_id exists and list is non-empty
        if (mapBfwSrsChestBufIdxList.find(cell_id) != mapBfwSrsChestBufIdxList.end() && !mapBfwSrsChestBufIdxList[cell_id].empty())
        {
            auto& lst = mapBfwSrsChestBufIdxList[cell_id]; // Reference to the list

            auto it = std::find(lst.begin(), lst.end(), srsChestBufferIndex);
            if (it != lst.end())
            {
                lst.erase(it); // Remove the matched srsChestBufferIndex
                NVLOGD_FMT(TAG, "{}: mapBfwSrsChestBufIdxList is Emprty for cell_id={} srsChestBufferIndex={} !!", __func__, cell_id, srsChestBufferIndex);
            }

            // If the list becomes empty after removal, erase the cell_id from the map
            if (lst.empty())
            {
                mapBfwSrsChestBufIdxList.erase(cell_id);
            }
        }
    }

    bool fapi_handler::isIdxInMapBfwSrsChestBufIdxList(uint32_t cell_id, uint16_t srsChestBufferIndex)
    {
        auto found = std::find(mapBfwSrsChestBufIdxList[cell_id].begin(), mapBfwSrsChestBufIdxList[cell_id].end(), srsChestBufferIndex);
        if (found != mapBfwSrsChestBufIdxList[cell_id].end())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

fapi_handler::fapi_handler(phy_mac_transport& transport, test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats * conformance_test_stats) :
    _transport(transport)
{
    fapi_handler_instance = this;

    this->configs = configs;
    this->lp      = lp;
    this->conformance_test_stats = conformance_test_stats;

    current_config_cell_id.store(-1);
    config_retry_counter.store(0);

    restart_timer                 = nullptr;
    global_tick                   = (uint64_t)0 - 1; // To start global_tick from 0
    conformance_test_slot_counter = 0;
    beam_id_position              = 0;

    ul_dci_freq_domain_bits = 0;

    rnti_test_mode = 0;

    slots_per_frame = lp->get_slots_per_frame();
    notify_mode = configs->get_ipc_sync_mode();
    // vald.enable = configs->validate_enable;
    // vald.log_opt = configs->validate_log_opt;

    data_buf_opt = configs->get_fapi_tb_loc();
    cell_num = lp->get_cell_num();
    thrputs.resize(cell_num);
    cell_data.resize(cell_num);
    first_init.resize(cell_num);
    cell_summary.resize(cell_num);

    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        thrputs[cell_id].reset();
        memset(&cell_summary[cell_id], 0, sizeof(results_summary_t));
        cell_data[cell_id].prach_state     = 0;
        cell_data[cell_id].prach_prambIdx  = -1;
        cell_data[cell_id].harq_process_id = 0;
        cell_data[cell_id].schedule_enable = false;
        cell_data[cell_id].restart_timer   = nullptr;
        cell_data[cell_id].fapi_state      = fapi_state_t::IDLE;
        cell_data[cell_id].cell_id_remap = cell_id;
        cell_id_map[cell_id] = cell_id;
        cell_remap_event[cell_id] = false;

        cell_data[cell_id].slot_timing.resize(slots_per_frame);

        first_init[cell_id] = 1;

        for (int j = 0; j < FAPI_REQ_SIZE; j++)
        {
            if (configs->fapi_delay_bit_mask & (1 << j))
            {
                for (int k = 0; k < FAPI_SCHED_BUFFER_SIZE; k++)
                {
                    cell_data[cell_id].fapi_scheds[k].target_ts_offsets[j] = &configs->schedule_total_time;
                }
            }
        }
    }

    sem_init(&slot_sem, 0, 0);
    sem_init(&fapi_sched_ready, 0, 0);
    sem_init(&fapi_sched_free, 0, 0);

    sem_init(&scheduler_sem, 0, 0);
    sched_info_ring = nv_ipc_ring_open(RING_TYPE_APP_INTERNAL, "scheduler", 256, sizeof(sched_info_t));

    ss_tick.store({.u32 = SFN_SLOT_INVALID});
    ts_tick = 0;
    ss_fapi_last.u32 = 0;

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

    // UL FAPI response timing
    if ((uci_time = stat_log_open("UCI_TIME", STAT_MODE_COUNTER, ul_stat_interval)) != NULL)
    {
        uci_time->set_limit(uci_time, 0, ul_timing_max);
    }
    if ((crc_time = stat_log_open("CRC_TIME", STAT_MODE_COUNTER, ul_stat_interval)) != NULL)
    {
        crc_time->set_limit(crc_time, 0, ul_timing_max);
    }
    if ((rx_data_time = stat_log_open("RX_DATA_TIME", STAT_MODE_COUNTER, ul_stat_interval)) != NULL)
    {
        rx_data_time->set_limit(rx_data_time, 0, ul_timing_max);
    }
    if ((srs_ind_time = stat_log_open("SRS_IND_TIME", STAT_MODE_COUNTER, ul_stat_interval)) != NULL)
    {
        srs_ind_time->set_limit(srs_ind_time, 0, ul_timing_max);
    }

    if ((stat_stt_diff = stat_log_open("MAC_STT_DIFF", STAT_MODE_COUNTER, STT_STATISTIC_PERIOD)) != NULL)
    {
        stat_stt_diff->set_limit(stat_stt_diff, -200 * 1000, 10 * 1000);
    }
    if ((stat_stt_send = stat_log_open("MAC_STT_SEND", STAT_MODE_COUNTER, STT_STATISTIC_PERIOD)) != NULL)
    {
        stat_stt_send->set_limit(stat_stt_send, 0, 500 * 1000);
    }
    if ((stat_deadline_diff = stat_log_open("TX:DEADLINE_DIFF", STAT_MODE_COUNTER, DEADLINE_STATISTIC_PERIOD)) != NULL)
    {
        stat_deadline_diff->set_limit(stat_deadline_diff, -DEADLINE_DIFF_MAX_NS, DEADLINE_DIFF_MAX_NS);
    }
    if ((stat_tx_per_msg = stat_log_open("TX:PER_MSG_TIME", STAT_MODE_COUNTER, STT_STATISTIC_PERIOD)) != NULL)
    {
        stat_tx_per_msg->set_limit(stat_tx_per_msg, 0, 100 * 1000);
    }

    int pattern_len = lp->get_slot_cell_patterns().size();
    stt_estimate.resize(pattern_len);
    for(int slot_idx = 0; slot_idx < pattern_len; slot_idx++)
    {
        stt_estimate_t& stt    = stt_estimate[slot_idx];
        stt.estimate_counter   = 0;
        stt.statistic_period   = STT_STATISTIC_PERIOD / pattern_len;
        stt.estimate_send_time = configs->estimate_send_time;
        char name[64];
        snprintf(name, 64, "MAC_STT_DIFF_%d", slot_idx);
        if((stt.stat_diff = stat_log_open(name, STAT_MODE_COUNTER, stt.statistic_period)) != NULL)
        {
            stt.stat_diff->set_limit(stt.stat_diff, -200 * 1000, 10 * 1000);
        }
        snprintf(name, 64, "MAC_STT_SEND_%d", slot_idx);
        if((stt.stat_send = stat_log_open(name, STAT_MODE_COUNTER, stt.statistic_period)) != NULL)
        {
            stt.stat_send->set_limit(stt.stat_send, 0, 500 * 1000);
        }
    }

    // Init ul_harq_handle to avoid converity scan defects
    for (int i = 0; i < MAX_CELLS_PER_SLOT; i ++) {
        for (int j = 0; j < FAPI_MAX_UE; j ++) {
            for (int k = 0; k < FAPI_MAX_UL_HARQ_ID; k++) {
                ul_harq_handle[i][j][k].harq_id = 0;
                ul_harq_handle[i][j][k].retx_count = 0;
                ul_harq_handle[i][j][k].rv = 0;
                ul_harq_handle[i][j][k].pusch_pdu_NDI = 0;
                ul_harq_handle[i][j][k].ul_dci_NDI = 0;
            }
        }
    }
}

fapi_handler::~fapi_handler()
{
    lp->~launch_pattern();
    if (schedule_time != NULL)
    {
        schedule_time->close(schedule_time);
    }
    if (stat_debug != NULL)
    {
        stat_debug->close(stat_debug);
    }
    if (uci_time != NULL)
    {
        uci_time->close(uci_time);
    }
    if (crc_time != NULL)
    {
        crc_time->close(crc_time);
    }
    if (rx_data_time != NULL)
    {
        rx_data_time->close(rx_data_time);
    }
    if (srs_ind_time != NULL)
    {
        srs_ind_time->close(srs_ind_time);
    }
    if (stat_stt_diff != NULL)
    {
        stat_stt_diff->close(stat_stt_diff);
    }
    if (stat_stt_send != NULL)
    {
        stat_stt_send->close(stat_stt_send);
    }
    if (stat_deadline_diff != NULL)
    {
        stat_deadline_diff->close(stat_deadline_diff);
    }
    if (stat_tx_per_msg != NULL)
    {
        stat_tx_per_msg->close(stat_tx_per_msg);
    }
}

// Set the RACH_INDICATION preamble index to MSG 2 RAPID
int fapi_handler::set_prach_msg2_rapid(int cell_id, uint8_t* rapid)
{
    *rapid &= 0xC0;
    if(cell_data[cell_id].prach_prambIdx >= 0)
    {
        *rapid |= cell_data[cell_id].prach_prambIdx & 0x3F;
    }
    NVLOGI_FMT(TAG, "{}: prambIdx={} RAPID={}", __FUNCTION__, cell_data[cell_id].prach_prambIdx, *rapid & 0x3F);
    return 0;
}

int fapi_handler::get_pramble_params(int cell_id, uint16_t sfn, uint16_t slot, preamble_params_t* preamble)
{
    vector<vector<vector<vector<fapi_req_t*>>>>& slot_cell_patterns = lp->get_slot_cell_patterns(get_slot_in_frame(sfn, slot));
    uint32_t                                     slot_idx           = get_slot_in_frame(sfn, slot) % slot_cell_patterns.size();
    vector<vector<fapi_req_t*>>&                 fapi_groups        = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
    vector<fapi_req_t*>&                         fapi_reqs          = fapi_groups[UL_TTI_REQ];

    preamble->tv_numPrmb = 0;
    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t* req = fapi_reqs[i];
        if(req->channel == channel_type_t::PRACH)
        {
            for(auto it: req->tv_data->prach_tv.data)
            {
                preamble->tv_delay.insert(preamble->tv_delay.end(),it.ref.delay_v.begin(), it.ref.delay_v.end());
                preamble->tv_peak.insert(preamble->tv_peak.end(), it.ref.peak_v.begin(), it.ref.peak_v.end());
                preamble->tv_numPrmb    += it.ref.numPrmb;
                preamble->tv_prmbIdx.insert(preamble->tv_prmbIdx.end(), it.ref.prmbIdx_v.begin(), it.ref.prmbIdx_v.end());
            }
            auto it2 = preamble->tv_nPreambPwr.end();
            auto it3 = preamble->tv_nTA.end();
            preamble->tv_nPreambPwr.resize(preamble->tv_peak.size());
            preamble->tv_nTA.resize(preamble->tv_delay.size());
            std::transform(preamble->tv_peak.begin(),
                    preamble->tv_peak.end(),
                    preamble->tv_nPreambPwr.begin(),
                    [](float tv_peak)->uint32_t
                    {
                        return (140000 + (10000 * (std::log10(tv_peak))));
                    });
            float TC = nv::TC;
            uint32_t mu = lp->get_cell_configs(cell_id).mu;
            std:: transform(preamble->tv_delay.begin(),
                    preamble->tv_delay.end(),
                    preamble->tv_nTA.begin(),
                    [TC, mu](float tv_delay)
                    {
                        return tv_delay / (16 * 64 *TC / (1 << mu));
                    });

            //NVLOGD_FMT(TAG, "{}: delay={} nTA={} peak={} prmbPwr={} numPrmb={} prmbIdx={}", __FUNCTION__, preamble->tv_delay, preamble->tv_nTA, preamble->tv_peak, preamble->tv_nPreambPwr, preamble->tv_numPrmb, preamble->tv_prmbIdx);
        }
    }
    return 0;
}

pusch_tv_data_t* fapi_handler::get_pusch_tv_pdu(fapi_req_t* pusch_req, int pdu_id, int bitmap)
{
    if (pusch_req->tv_data == nullptr)
    {
        return nullptr;
    }
    int tv_pdu_id = 0;
    for (pusch_tv_data_t* tv_data : pusch_req->tv_data->pusch_tv.data)
    {
        if (tv_data->pduBitmap & bitmap)
        {
            if (tv_pdu_id == pdu_id)
            {
                return tv_data;
            }
            tv_pdu_id ++;
        }
    }
    return nullptr;
}

srs_tv_data_t* fapi_handler::get_srs_tv_pdu(fapi_req_t* srs_req, uint32_t handle)
{
    if (srs_req->tv_data == nullptr)
    {
        return nullptr;
    }
    for (srs_tv_data_t& tv_data : srs_req->tv_data->srs_tv.data)
    {
        if (tv_data.srsPduIdx == handle)
        {
            return &tv_data;
        }
    }
    return nullptr;
}

pucch_tv_data_t* fapi_handler::get_pucch_tv_pdu(fapi_req_t* pucch_req, int pdu_id)
{
    if (pucch_req->tv_data == nullptr)
    {
        return nullptr;
    }
    int tv_pdu_id = 0;
    for (pucch_tv_data_t& tv_data : pucch_req->tv_data->pucch_tv.data)
    {
        if (tv_pdu_id == pdu_id)
        {
            return &tv_data;
        }
        tv_pdu_id ++;
    }
    return nullptr;
}

int fapi_handler::get_pucch_tv_pf234_pdu_num(fapi_req_t* pucch_req)
{
    if (pucch_req->tv_data == nullptr)
    {
        return 0;
    }
    int pf234_pdu_num = 0;
    for (pucch_tv_data_t& tv_data : pucch_req->tv_data->pucch_tv.data)
    {
        if (tv_data.FormatType >= 2)
        {
            pf234_pdu_num ++;
        }
    }
    return pf234_pdu_num;
}

pucch_tv_data_t* fapi_handler::get_pucch_tv_pf234_pdu(fapi_req_t* pucch_req, int pdu_id)
{
    if (pucch_req->tv_data == nullptr)
    {
        return nullptr;
    }
    int tv_pdu_id = 0;
    for (pucch_tv_data_t& tv_data : pucch_req->tv_data->pucch_tv.data)
    {
        if (tv_data.FormatType >= 2)
        {
            if (tv_pdu_id == pdu_id)
            {
                return &tv_data;
            }
            tv_pdu_id ++;
        }
    }
    return nullptr;
}

fapi_req_t* fapi_handler::get_fapi_req_data(int cell_id, uint16_t sfn, uint16_t slot, channel_type_t channel)
{
    vector<vector<vector<vector<fapi_req_t*>>>>& slot_cell_patterns = lp->get_slot_cell_patterns(get_slot_in_frame(sfn, slot));
    uint32_t                                     slot_idx           = get_slot_in_frame(sfn, slot) % slot_cell_patterns.size();
    vector<vector<fapi_req_t*>>&                 fapi_groups        = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
    vector<fapi_req_t*>&                         fapi_reqs          = fapi_groups[UL_TTI_REQ];

    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t* req = fapi_reqs[i];
        if(req->channel == channel)
        {
            // NVLOGD_FMT(TAG, "{}: SFN {}.{} channel {} found: fapi_reqs={}", __func__, sfn, slot, get_channel_name(channel), fapi_reqs.size());
            return req;
        }
    }
    // NVLOGW_FMT(TAG, "{}: SFN {}.{} channel {} not found fapi_reqs={}", __func__, sfn, slot, get_channel_name(channel), fapi_reqs.size());
    return nullptr;
}

vector<fapi_req_t*>& fapi_handler::get_fapi_req_list(int cell_id, sfn_slot_t ss, fapi_group_t group_id)
{
    slot_pattern_t &slot_cell_patterns = lp->get_slot_cell_patterns(get_slot_in_frame(ss));
    uint32_t slot_idx = get_slot_in_frame(ss) % slot_cell_patterns.size();
    vector<vector<fapi_req_t*>> &fapi_groups = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
    return fapi_groups[group_id];
}

void fapi_handler::slot_indication_handler(uint32_t sfn, uint32_t slot, uint64_t slot_counter)
{

    auto t_start = std::chrono::system_clock::now().time_since_epoch();
    if((configs->get_conformance_test_params()->conformance_test_enable == true)&&
    (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
    {
        conformance_test_slot_counter++;
        if(configs->get_conformance_test_params()->conformance_test_slots <= conformance_test_slot_counter)
        {
            configs->get_conformance_test_params()->conformance_test_enable = false;
            dump_conformance_test_stats();

            NVLOGC_FMT(TAG,"*************************SLOT.indication Handler {}{} Conformance Test Over ***************",sfn,slot);
        }
    }
    auto t_conformance = std::chrono::system_clock::now().time_since_epoch();

    if (configs->builder_thread_enable)
    {
        sem_post(&slot_sem);
        sem_wait(&fapi_sched_ready);
    }

    try
    {
        static_ul_dl_scheduler(sfn, slot, slot_counter);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "Send failed: {}", e.what());
    }

    if (configs->builder_thread_enable)
    {
        sem_post(&fapi_sched_free);
    }

    auto t_end = std::chrono::system_clock::now().time_since_epoch();

    NVLOGI_FMT(TAG,
       "{{TI}} <SLOT.indication Handler,{},{},0,0> Start Task:{},Conformance Test:{},End Task:{},",
       sfn,
       slot,
       t_start.count(),
       t_conformance.count(),
       t_end.count());
}

int fapi_handler::schedule_all_cell_restart(uint64_t slot_counter)
{
    int test_slots = configs->get_test_slots();
    if(test_slots > 0 && slot_counter > 0 && slot_counter % test_slots == 0)
    {
        int restart_interval = configs->get_restart_interval();
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

int fapi_handler::schedule_single_cell_restart(uint64_t slot_counter)
{
    // Single cell restart test if configured cell_run_slots > 0 && cell_stop_slots > 0
    int cell_run_slots = configs->get_cell_run_slots();
    int cell_stop_slots = configs->get_cell_stop_slots();
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

int fapi_handler::schedule_cell_update(uint64_t slot_counter)
{
    std::vector<cell_update_cmd_t>& cmds = configs->get_cell_update_commands();
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
                    set_restart_timer(param.cell_id, configs->get_restart_interval());
                }
            }
        }
    }
    return 0;
}

void fapi_handler::static_ul_dl_scheduler(uint32_t sfn, uint32_t slot, uint64_t slot_counter)
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
    int fapi_count = 0;

// #ifdef CUPTI_ENABLE_TRACING
//     // Limited slots enabled for cupti tracing
//     bool enable_this_slot = false;
//     static int ramp_enable_slot_mask[80] {0};
//     ramp_enable_slot_mask[4] = 1;
//     int ramp_slot_idx = (sfn % 4)*20 + slot;
//     if (ramp_enable_slot_mask[ramp_slot_idx])
//     {
//         enable_this_slot = true;
//     }
// #else
    bool enable_this_slot = true;
// #endif

    if (enable_this_slot)
    {
        fapi_count = schedule_slot(ss);
    }

    // The time stamp of clock_gettime(CLOCK_REALTIME, &ts) and system_clock::now().time_since_epoch() should be the same
    struct timespec ts_now;
    nvlog_gettime_rt(&ts_now);

    int64_t handle_delay = ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec - ts_tick;
    int64_t mac_sched_time = (ts_now.tv_sec * 1000000000LL + ts_now.tv_nsec - ts_tick_tai) / 1000; // us

    if (schedule_time != NULL)
    {
        schedule_time->add(schedule_time, mac_sched_time); // us
    }

    NVLOGI_FMT(TAG, "SFN {}.{} slot_counter={} fapi_count={} scheduled in {} us - SLOT.ind handle delay={} tai_diff={}",
            sfn, slot, slot_counter, fapi_count, mac_sched_time, handle_delay, ts_tick_tai - ts_tick);

    // Print throughput statistic info every second (print after finishing scheduling, do not include throughput of this slot)
    if(slot_counter > 0 && slot_counter % SLOTS_PER_SECOND == 0)
    {
        print_thrput(slot_counter);
    }

    // Add DL throughput of this slot
    if (enable_this_slot && configs->app_mode == 0)
    {
        update_dl_thrput(sfn, slot, slot_counter);
    }
}

int fapi_handler::set_fapi_delay(int cell_id, int slot, int fapi_mask, int delay_us)
{
    NVLOGC_FMT(TAG,"OAM Set FAPI delay: cell_id={} slot={} fapi_mask=0x{:02X} delay_us={}",
            cell_id, slot, fapi_mask, delay_us);

    if (cell_id != 255 && cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: error: cell_id={} cell_num={}", __func__, cell_id, cell_num);
        return -1;
    }

    if (slot >= lp->get_sched_slot_num())
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: error: slot={} sched_slot_num={}", __func__, slot, lp->get_sched_slot_num());
        return -1;
    }

    // Set delay for 1 slot of the whole launch pattern
    configs->schedule_total_time.resize(lp->get_sched_slot_num());
    for (int i = 0; i < lp->get_sched_slot_num(); i++)
    {
        configs->schedule_total_time[i] = i == slot ? delay_us * 1000 : 0;
    }

    for (int j = 0; j < FAPI_REQ_SIZE; j++)
    {
        if (fapi_mask & (1 << j))
        {
            for (int k = 0; k < FAPI_SCHED_BUFFER_SIZE; k++)
            {
                if (cell_id < cell_num)
                {
                    cell_data[cell_id].fapi_scheds[k].target_ts_offsets[j] = &configs->schedule_total_time;
                }
                else
                {
                    for (int cid = 0; cid < cell_num; cid ++)
                    {
                        cell_data[cid].fapi_scheds[k].target_ts_offsets[j] = &configs->schedule_total_time;
                    }
                }
            }
        }
    }

    return 0;
}

void fapi_handler::update_dl_thrput(uint32_t sfn, uint32_t slot, uint64_t slot_counter)
{
    vector<vector<vector<vector<fapi_req_t*>>>>& slot_cell_patterns = lp->get_slot_cell_patterns(slot_counter);

    uint32_t cell_num = lp->get_cell_num();
    uint32_t slot_idx = get_slot_in_frame(sfn, slot) % slot_cell_patterns.size();

    auto cells_size = slot_cell_patterns[slot_idx].size();
    auto start_cell_index = 0;
    uint16_t num_fapi_msgs_per_cell = 0;

    for(int cell_id = 0; cell_id < cells_size; cell_id++)
    {
        if (cell_data[cell_id].schedule_enable == false)
        {
            // Skip stopped cells
            continue;
        }

        vector<vector<fapi_req_t*>>& fapi_groups = slot_cell_patterns[slot_idx][cell_id_map[cell_id]];
        for(int group_id = 0; group_id <= fapi_groups.size(); group_id++)
        {
            if (group_id == TX_DATA_REQ)
            {
                vector<fapi_req_t*>& fapi_reqs = fapi_groups[group_id];
                for (fapi_req_t* req : fapi_reqs)
                {
                    if (req->channel == channel_type_t::PDSCH)
                    {
                        for(const pdsch_tv_data_t* tv_data : req->tv_data->pdsch_tv.data)
                        {
                            if(tv_data->tb_size != 0 && tv_data->tb_buf != NULL)
                            {
                                thrputs[cell_id].dl_thrput += tv_data->tb_size;
                            }
                        }
                        thrputs[cell_id].slots[PDSCH]++;
                    }
                }
            }
        }
    }
}

void fapi_handler::print_thrput(uint64_t slot_counter)
{
    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        thrput_t& thrput = thrputs[cell_id];
        results_summary_t& summary = cell_summary[cell_id];
        // Console log print per second
        if((thrput.ul_ind.ontime != 0 || thrput.ul_ind.late != 0) && (thrput.prach_ind.ontime != 0 || thrput.prach_ind.late != 0) && (thrput.uci.ontime != 0 || thrput.uci.late != 0) && (thrput.srs_ind.late != 0 || thrput.srs_ind.early != 0 || thrput.srs_ind.ontime != 0))
        {
            float ul_ind_pct = (summary.ul_ind.ontime * 100.0) / (summary.ul_ind.ontime + summary.ul_ind.late);
            float prach_ind_pct = (summary.prach_ind.ontime * 100.0) / (summary.prach_ind.ontime + summary.prach_ind.late);
            float uci_pct = (summary.uci.ontime * 100.0) / (summary.uci.ontime + summary.uci.late);
            float srs_ind_pct = (summary.srs_ind.ontime * 100.0) / (summary.srs_ind.ontime + summary.srs_ind.late + thrput.srs_ind.early);
            NVLOGC_FMT(TAG, "Cell {:2} | DL {:7.2f} Mbps {:4} Slots | UL {:7.2f} Mbps {:4} Slots | Prmb {:4} | HARQ {:4} | SR {:4} | CSI1 {:4} | CSI2 {:4} | SRS {:4} | ERR {:4} | INV {:4} | UL ONTIME {:3.2f} | PRACH ONTIME {:3.2f} | UCI ONTIME {:3.2f} | SRS ONTIME {:3.2f} | Slots {}",
                    cell_id,
                    ((double)thrput.dl_thrput.load() * 8 / 1000000),
                    thrput.slots[PDSCH].load(),
                    ((double)thrput.ul_thrput.load() * 8 / 1000000),
                    thrput.slots[PUSCH].load(),
                    thrput.prmb.load(),
                    thrput.harq.load(),
                    thrput.sr.load(),
                    thrput.csi1.load(),
                    thrput.csi2.load(),
                    thrput.srs.load(),
                    thrput.error.load(),
                    thrput.invalid.load(),
                    ul_ind_pct,
                    prach_ind_pct,
                    uci_pct,
                    srs_ind_pct,
                    slot_counter
                    );
        }
        else if((thrput.uci.ontime != 0 || thrput.uci.late != 0) && (thrput.srs_ind.late != 0 || thrput.srs_ind.early != 0 || thrput.srs_ind.ontime != 0))
        {
            float pct = (summary.uci.ontime * 100.0) / (summary.uci.ontime + summary.uci.late);
            float srs_ind_pct = (summary.srs_ind.ontime * 100.0) / (summary.srs_ind.ontime + summary.srs_ind.late + thrput.srs_ind.early);
            NVLOGC_FMT(TAG, "Cell {:2} | DL {:7.2f} Mbps {:4} Slots | UL {:7.2f} Mbps {:4} Slots | Prmb {:4} | HARQ {:4} | SR {:4} | CSI1 {:4} | CSI2 {:4} | SRS {:4} | ERR {:4} | INV {:4} | UCI ONTIME {:3.2f} | SRS ONTIME {:3.2f} | Slots {}",
                    cell_id,
                    ((double)thrput.dl_thrput.load() * 8 / 1000000),
                    thrput.slots[PDSCH].load(),
                    ((double)thrput.ul_thrput.load() * 8 / 1000000),
                    thrput.slots[PUSCH].load(),
                    thrput.prmb.load(),
                    thrput.harq.load(),
                    thrput.sr.load(),
                    thrput.csi1.load(),
                    thrput.csi2.load(),
                    thrput.srs.load(),
                    thrput.error.load(),
                    thrput.invalid.load(),
                    pct,
                    srs_ind_pct,
                    slot_counter
                    );
        }
        else if(thrput.uci.ontime != 0 || thrput.uci.late != 0)
        {
            float pct = (summary.uci.ontime * 100.0) / (summary.uci.ontime + summary.uci.late);
            NVLOGC_FMT(TAG, "Cell {:2} | DL {:7.2f} Mbps {:4} Slots | UL {:7.2f} Mbps {:4} Slots | Prmb {:4} | HARQ {:4} | SR {:4} | CSI1 {:4} | CSI2 {:4} | SRS {:4} | ERR {:4} | INV {:4} | UCI ONTIME {:3.2f} | Slots {}",
                    cell_id,
                    ((double)thrput.dl_thrput.load() * 8 / 1000000),
                    thrput.slots[PDSCH].load(),
                    ((double)thrput.ul_thrput.load() * 8 / 1000000),
                    thrput.slots[PUSCH].load(),
                    thrput.prmb.load(),
                    thrput.harq.load(),
                    thrput.sr.load(),
                    thrput.csi1.load(),
                    thrput.csi2.load(),
                    thrput.srs.load(),
                    thrput.error.load(),
                    thrput.invalid.load(),
                    pct,
                    slot_counter
                    );
        }
        else if(thrput.srs_ind.late != 0 || thrput.srs_ind.early != 0 || thrput.srs_ind.ontime != 0)
        {
            float srs_ind_pct = (summary.srs_ind.ontime * 100.0) / (summary.srs_ind.ontime + summary.srs_ind.late + thrput.srs_ind.early);
            NVLOGC_FMT(TAG, "Cell {:2} | DL {:7.2f} Mbps {:4} Slots | UL {:7.2f} Mbps {:4} Slots | Prmb {:4} | HARQ {:4} | SR {:4} | CSI1 {:4} | CSI2 {:4} | SRS {:4} | ERR {:4} | INV {:4} | SRS ONTIME {:3.2f} | Slots {}",
                    cell_id,
                    ((double)thrput.dl_thrput.load() * 8 / 1000000),
                    thrput.slots[PDSCH].load(),
                    ((double)thrput.ul_thrput.load() * 8 / 1000000),
                    thrput.slots[PUSCH].load(),
                    thrput.prmb.load(),
                    thrput.harq.load(),
                    thrput.sr.load(),
                    thrput.csi1.load(),
                    thrput.csi2.load(),
                    thrput.srs.load(),
                    thrput.error.load(),
                    thrput.invalid.load(),
                    srs_ind_pct,
                    slot_counter
                    );
        }
        else
        {
            NVLOGC_FMT(TAG, "Cell {:2} | DL {:7.2f} Mbps {:4} Slots | UL {:7.2f} Mbps {:4} Slots | Prmb {:4} | HARQ {:4} | SR {:4} | CSI1 {:4} | CSI2 {:4} | SRS {:4} | ERR {:4} | INV {:4} | Slots {}",
                    cell_id,
                    ((double)thrput.dl_thrput.load() * 8 / 1000000),
                    thrput.slots[PDSCH].load(),
                    ((double)thrput.ul_thrput.load() * 8 / 1000000),
                    thrput.slots[PUSCH].load(),
                    thrput.prmb.load(),
                    thrput.harq.load(),
                    thrput.sr.load(),
                    thrput.csi1.load(),
                    thrput.csi2.load(),
                    thrput.srs.load(),
                    thrput.error.load(),
                    thrput.invalid.load(),
                    slot_counter
                    );
        }
        thrput.reset();
    }
}

void fapi_handler::dump_conformance_test_stats()
{
   std::string conformance_test_stats_log = "************ CH8_CONFORMACE_TEST_STATS****************\n";
   conformance_test_stats_log.append("Prach Stats:\n");
   conformance_test_stats_log.append("prach_occassion = ").append(std::to_string(conformance_test_stats->get_prach_stats().prach_occassion)).append("\n");
   conformance_test_stats_log.append("preamble_detected = ").append(std::to_string(conformance_test_stats->get_prach_stats().preamble_detected)).append("\n");
   conformance_test_stats_log.append("preamble_error = ").append(std::to_string(conformance_test_stats->get_prach_stats().preamble_error)).append("\n\n");
   conformance_test_stats_log.append("PUCCH format 0 Stats:\n");
   conformance_test_stats_log.append("pf0_occassion = ").append(std::to_string(conformance_test_stats->get_pf0_stats().pf0_occassion)).append("\n");
   conformance_test_stats_log.append("pf0_ack = ").append(std::to_string(conformance_test_stats->get_pf0_stats().pf0_ack)).append("\n");
   conformance_test_stats_log.append("pf0_nack = ").append(std::to_string(conformance_test_stats->get_pf0_stats().pf0_nack)).append("\n");
   conformance_test_stats_log.append("pf0_dtx = ").append(std::to_string(conformance_test_stats->get_pf0_stats().pf0_dtx)).append("\n\n");
   conformance_test_stats_log.append("PUCCH format 1 Stats:\n");
   conformance_test_stats_log.append("pf1_occassion = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_occassion)).append("\n");
   conformance_test_stats_log.append("pf1_ack_nack_pattern = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern)).append("\n");
   conformance_test_stats_log.append("pf1_ack_bits = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_ack_bits)).append("\n");
   conformance_test_stats_log.append("pf1_nack_bits = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_nack_bits)).append("\n");
   conformance_test_stats_log.append("pf1_dtx_bits = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_dtx_bits)).append("\n");
   conformance_test_stats_log.append("pf1_nack_to_ack_bits = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_nack_to_ack_bits)).append("\n");
   conformance_test_stats_log.append("pf1_ack_to_nack_bits = ").append(std::to_string(conformance_test_stats->get_pf1_stats().pf1_ack_to_nack_bits)).append("\n\n");
   conformance_test_stats_log.append("PUCCH format 2 Stats:\n");
   conformance_test_stats_log.append("pf2_harq_occassion = ").append(std::to_string(conformance_test_stats->get_pf2_stats().pf2_harq_occassion)).append("\n");
   conformance_test_stats_log.append("pf2_csi_occassion = ").append(std::to_string(conformance_test_stats->get_pf2_stats().pf2_csi_occassion)).append("\n");
   conformance_test_stats_log.append("pf2_harq_ack_bits = ").append(std::to_string(conformance_test_stats->get_pf2_stats().pf2_harq_ack_bits)).append("\n");
   conformance_test_stats_log.append("pf2_harq_nack_bits = ").append(std::to_string(conformance_test_stats->get_pf2_stats().pf2_harq_nack_bits)).append("\n");
   conformance_test_stats_log.append("pf2_bler = ").append(std::to_string(conformance_test_stats->get_pf2_stats().pf2_bler)).append("\n\n");
   conformance_test_stats_log.append("PUCCH format 3 Stats:\n");
   conformance_test_stats_log.append("Pf3_occassion = ").append(std::to_string(conformance_test_stats->get_pf3_stats().pf3_occassion)).append("\n");
   conformance_test_stats_log.append("Pf3_bler= ").append(std::to_string(conformance_test_stats->get_pf3_stats().pf3_bler)).append("\n\n");

   ofstream conformance_test_stats_file;
   std::string conformance_test_stats_file_name = std::string("/tmp/") + configs->get_conformance_test_params()->conformance_test_stats_file ;
   conformance_test_stats_file.open (conformance_test_stats_file_name.c_str());
   conformance_test_stats_file <<  conformance_test_stats_log;
   conformance_test_stats_file.close();
   cell_stop(0);
}
