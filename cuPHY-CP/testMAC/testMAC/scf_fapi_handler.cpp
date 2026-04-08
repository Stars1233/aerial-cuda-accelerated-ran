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

#include "app_config.hpp"
#include <algorithm>
#include <string.h>
#include <type_traits>
#include "scf_fapi_handler.hpp"
#include "fapi_validate.hpp"
#include "nvlog.hpp"
#include "oran_utils/conversion.hpp"

#ifdef AERIAL_CUMAC_ENABLE
#include "cumac_handler.hpp"
#endif

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 4) // "MAC.SCF"
#define TAG_PROCESSING_TIMES (NVLOG_TAG_BASE_TEST_MAC + 9) // "MAC.PROCESSING_TIMES"

// If defined, force L2 to have an approx total scheduling time
// NB: This only works correctly for PTP GPS_ALPHA=GPS_BETA=0
// NB: See "schedule_time->set_limit(schedule_time, ...)" in fapi_handler.cpp if changing this
static constexpr long SLOT_TIME_BOUNDARY_NS = 500000;
static constexpr long SLOT_ADVANCE = 3; // TODO: If L2 knows this, use the variable instead of hard-coding

// Dynamic slot test parameters for spectral efficiency
static constexpr uint8_t DYN_TEST_START_SYMBOL_INDEX = 0;
static constexpr uint16_t DYN_TEST_DMRS_SYM_POS = 0x0402; // Symbol 2 and 11

// define it to 1, to get GDRCOPY and COPY timing stats
#define COPY_TIME_STAT_ENABLED 0

static constexpr int MAX_MIB_BITS = 24;

static inline void to_u8_array(uint32_t bchPayload, uint8_t* mib)
{
    for(int i = 0; i < MAX_MIB_BITS / 8; i++)
    {
        mib[i] = bchPayload >> ((MAX_MIB_BITS / 8 - i - 1) * 8);
    }
}

const char* get_scf_fapi_msg_name(uint8_t msg_id)
{
    switch(msg_id)
    {
    case SCF_FAPI_PARAM_REQUEST:
        return "PARAM.req";
    case SCF_FAPI_PARAM_RESPONSE:
        return "PARAM.resp";
    case SCF_FAPI_CONFIG_REQUEST:
        return "CONFIG.req";
    case SCF_FAPI_CONFIG_RESPONSE:
        return "CONFIG.resp";

    case SCF_FAPI_START_REQUEST:
        return "START.req";
    case SCF_FAPI_STOP_REQUEST:
        return "STOP.req";
    case SCF_FAPI_STOP_INDICATION:
        return "STOP.ind";
    case SCF_FAPI_ERROR_INDICATION:
        return "ERR.ind";

    case SCF_FAPI_SLOT_INDICATION:
        return "SLOT.ind";

    case SCF_FAPI_DL_TTI_REQUEST:
        return "DL_TTI.req";
    case SCF_FAPI_UL_TTI_REQUEST:
        return "UL_TTI.req";
    case SCF_FAPI_TX_DATA_REQUEST:
        return "TX_DATA.req";
    case SCF_FAPI_UL_DCI_REQUEST:
        return "UL_DCI.req";
    case SCF_FAPI_DL_BFW_CVI_REQUEST:
        return "DL_BFW_CVI.req";
    case SCF_FAPI_UL_BFW_CVI_REQUEST:
        return "UL_BFW_CVI.req";
    case SCF_FAPI_RX_DATA_INDICATION:
        return "RX_DATA.ind";
    case SCF_FAPI_CRC_INDICATION:
        return "CRC.ind";
    case SCF_FAPI_UCI_INDICATION:
        return "UCI.ind";
    case SCF_FAPI_SRS_INDICATION:
        return "SRS.ind";
    case SCF_FAPI_RACH_INDICATION:
        return "RACH.ind";

    case SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION:
        return "PE_NOISE_VARIANCE.ind";
    case SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION:
        return "PF_234_INTERFERENCE.ind";
    case SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION:
        return "PRACH_INTERFERENCE.ind";

    case SCF_FAPI_SLOT_RESPONSE:
        return "SLOT.resp";
    case CV_MEM_BANK_CONFIG_REQUEST:
        return "CV_MEM_BANK_CONFIG.req";
    case CV_MEM_BANK_CONFIG_RESPONSE:
        return "CV_MEM_BANK_CONFIG.resp";

    default:
        return "UNKNOWN_SCF_FAPI";
    }
}

const char* get_fapi_group_name(fapi_group_t group_id)
{
    switch(group_id)
    {
    case DL_TTI_REQ:
        return "DL_TTI.req";
    case UL_TTI_REQ:
        return "UL_TTI.req";
    case TX_DATA_REQ:
        return "TX_DATA.req";
    case UL_DCI_REQ:
        return "UL_DCI.req";
    case DL_BFW_CVI_REQ:
        return "DL_BFW_CVI.req";
    case UL_BFW_CVI_REQ:
        return "UL_BFW_CVI.req";
    default:
        return "UNKNOWN_FAPI_GROUP_ID";
    }
}

inline uint8_t to_scf_ch_seg_type(channel_segment_t& seg) {
    uint8_t val = 0;
    switch (seg.type)
    {
        case tv_channel_type_t::TV_PUSCH: // PUSCH
            /* code */
            return scf_fapi_ch_seg_type_t::SCF_CHAN_SEG_PUSCH;
            break;
        default:
            break;
    }
    return val;
}

scf_fapi_handler::scf_fapi_handler(phy_mac_transport& ipc_transport, test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats* conformance_test_stats):fapi_handler(ipc_transport, configs, lp, conformance_test_stats)
{
    sem_init(&worker_sem, 0, 0);

    // Initiate schedule_item_list and schedule_sequence. Size is cell_num * FAPI_REQ_SIZE for both.
    int fapi_req_size = static_cast<int>(FAPI_REQ_SIZE);
    schedule_item_list.resize(cell_num * fapi_req_size);
    schedule_sequence.resize(static_cast<size_t>(cell_num * fapi_req_size));
    for(int i = 0; i < cell_num * fapi_req_size; i++)
    {
        // Initialize schedule_item_list
        schedule_item_t& item = schedule_item_list[i];
        item.cell_id = i / fapi_req_size;
        item.group_id = static_cast<fapi_group_t>(i % fapi_req_size);
        item.tx_deadline = static_cast<int64_t>(configs->get_fapi_tx_deadline_ns(static_cast<size_t>(item.group_id)));
        item.remain_num = 0;
        item.exist = 0;

        // Default sequence is 0 ~ cell_num * FAPI_REQ_SIZE - 1
        schedule_sequence[i] = i;
    }

    // Dynamic slot test parameters for spectral efficiency
    size_t max_data_size = configs->get_max_data_size();
    if (max_data_size > 0)
    {
        dyn_tb_data_gen_buf.resize(max_data_size);
        for (size_t i = 0; i < max_data_size; i++) {
            dyn_tb_data_gen_buf[i] = i % 100; // Populate cycle of 0, 1, 2, 3, ..., 100 to the buffers for test
        }
    }
}

int scf_fapi_handler::reorder_schedule_sequence(sfn_slot_t ss)
{
    int fapi_req_size = static_cast<int>(FAPI_REQ_SIZE);
    int n = cell_num * fapi_req_size;
    for (int i = 0; i < n; i++)
    {
        schedule_item_t& item = schedule_item_list[static_cast<size_t>(i)];
        item.remain_num = 0;
        if (cell_data[item.cell_id].schedule_enable && get_fapi_req_list(item.cell_id, ss, item.group_id).size() > 0)
        {
            item.exist = 1;
        }
        else
        {
            item.exist = 0;
        }
    }

    // Sort so exist==1 items are at head, ordered by deadline, cell_id, group_id; exist==0 items after
    std::sort(schedule_sequence.begin(), schedule_sequence.end(),
        [this](int a, int b) {
            int ea = schedule_item_list[static_cast<size_t>(a)].exist;
            int eb = schedule_item_list[static_cast<size_t>(b)].exist;
            if (ea != eb)
                return ea > eb;  // exist=1 before exist=0
            if (ea == 0)
                return a < b;   // among exist=0, keep order
            int64_t ta = schedule_item_list[static_cast<size_t>(a)].tx_deadline;
            int64_t tb = schedule_item_list[static_cast<size_t>(b)].tx_deadline;
            if (ta != tb)
                return ta < tb;
            if (schedule_item_list[static_cast<size_t>(a)].cell_id != schedule_item_list[static_cast<size_t>(b)].cell_id)
                return schedule_item_list[static_cast<size_t>(a)].cell_id < schedule_item_list[static_cast<size_t>(b)].cell_id;
            return schedule_item_list[static_cast<size_t>(a)].group_id < schedule_item_list[static_cast<size_t>(b)].group_id;
        });

    // Fill remaining_same_deadline_count for exist=1 items (count same deadline from this position to end, among exist=1 only)
    for (int i = 0; i < n; i++)
    {
        int idx = schedule_sequence[static_cast<size_t>(i)];
        schedule_item_t& item = schedule_item_list[static_cast<size_t>(idx)];
        if (item.exist == 0)
            continue;
        int64_t d = item.tx_deadline;
        int count = 1;
        int j;
        bool ended = false;
        for (j = i + 1; j < n; j++)
        {
            int jidx = schedule_sequence[static_cast<size_t>(j)];
            if (schedule_item_list[static_cast<size_t>(jidx)].exist == 0)
            {
                ended = true;
                break;
            }
            if (schedule_item_list[static_cast<size_t>(jidx)].tx_deadline == d)
                count++;
            else
                break;
        }
        if (j == n)
        {
            ended = true;
        }

        item.remain_num = count;
        if (ended)
        {
            item.remain_num += cell_num; // Each cell has one SLOT.resp, so add cell_num to the remaining number
        }
    }
    return 0;
}

void scf_fapi_handler::terminate()
{
    // Wake up scheduler thread to exit the loop
    sem_post(&scheduler_sem);
}

void scf_fapi_handler::scheduler_thread_func()
{
    sched_info_t sched_info;
    while(sem_wait(&scheduler_sem) == 0)
    {
        if (is_app_exiting())
        {
            NVLOGC_FMT(TAG, "App is exiting, stop all RUNNING cells");
            // Stop all RUNNING cells
            for (int cell_id = 0; cell_id < cell_num; cell_id++)
            {
                if (cell_data[cell_id].fapi_state == fapi_state_t::RUNNING)
                {
                    cell_stop(cell_id);
                }
            }

            int wait_count = 0;
            for (int cell_id = 0; cell_id < cell_num; cell_id++)
            {
                while (wait_count < 10 && cell_data[cell_id].fapi_state == fapi_state_t::RUNNING)
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

            NVLOGI_FMT(TAG, "SFN {}.{} global_tick={} start", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, global_tick);

            // handle 0th based index
            slot_indication_handler(sched_info.ss.u16.sfn, sched_info.ss.u16.slot, global_tick);

            lp->check_init_pattern_finishing(global_tick + 1);
        }
    }
    NVLOGC_FMT(TAG, "[mac_sched] thread exiting");
}

void scf_fapi_handler::builder_thread_func()
{
    // All the fapi_scheds buffers are free at initial
    for (int i = 0; i < FAPI_SCHED_BUFFER_SIZE; i++) {
        sem_post(&fapi_sched_free);
    }

    // Build SFN 0.0 FAPI messages
    sem_wait(&fapi_sched_free);
    schedule_fapi_reqs({.u32 = 0}, -500 * 1000);
    sem_post(&fapi_sched_ready);

    // Wait for SLOT.ind to trigger semaphore post, then build next slot FAPI messages in advance
    while(sem_wait(&slot_sem) == 0)
    {
        sem_wait(&fapi_sched_free);

        sfn_slot_t ss_curr = ss_tick.load();
        if (ss_fapi_last.u32 != ss_curr.u32)
        {
            NVLOGW_FMT(TAG, "Slot missed: SFN curr: {}.{} last: {}.{}",
                    ss_curr.u16.sfn, ss_curr.u16.slot, ss_fapi_last.u16.sfn, ss_fapi_last.u16.slot);
        }

        sfn_slot_t ss_fapi = get_next_sfn_slot(ss_curr);
        // ss_fapi.u32 = ss_curr.u32 == SFN_SLOT_INVALID ? 0 : ss_curr.u32;

        schedule_fapi_reqs(ss_fapi, -500 * 1000);

        ss_fapi_last = ss_fapi;

        // Switch to SCHED pattern after finishing INIT pattern if exist
        lp->check_init_pattern_finishing(get_slot_in_frame(ss_fapi));

        sem_post(&fapi_sched_ready);
    }
}

void scf_fapi_handler::worker_thread_func()
{
    int wid = worker_id.fetch_add(1);

    if (wid >= configs->worker_cores.size())
    {
        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "{}: invalid worker id: wid={} worker_cores.size={}", __FUNCTION__, wid, configs->worker_cores.size());
        return;
    }

    char name[16];
    snprintf(name, 16, "mac_worker_%02d", wid);

    struct nv::thread_config config;
    config.name = std::string(name);
    config.cpu_affinity = configs->worker_cores[wid];
    config.sched_priority = 97;
    config_thread_property(config);

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", name, sched_getcpu());

    nv::phy_mac_msg_desc msg_desc;
    while(sem_wait(&worker_sem) == 0)
    {
        while(_transport.rx_recv(msg_desc) >= 0)
        {
            on_msg(msg_desc);
            _transport.rx_release(msg_desc);
        }
    }
}

void scf_fapi_handler::notify_worker_threads()
{
    sem_post(&worker_sem);
}

bool scf_fapi_handler::cell_id_sanity_check(int cell_id)
{
    if(cell_id >= cell_num)
    {
        NVLOGW_FMT(TAG, "{}: cell_id {} out of bounds, ignored. Valid range is [0 ~ {}]", __FUNCTION__, cell_id, cell_num - 1);
        return false;
    }
    return true;
}

void scf_fapi_handler::cell_init(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }

    // Do not reset the FAPI state and cell data when OAM re-configure cell
    if (configs->oam_cell_ctrl_cmd == 0)
    {
        thrputs[cell_id].reset();
        cell_data[cell_id].prach_state     = 0;
        cell_data[cell_id].prach_prambIdx  = -1;
        cell_data[cell_id].harq_process_id = 0;
        cell_data[cell_id].schedule_enable = false;
        cell_data[cell_id].fapi_state      = fapi_state_t::IDLE;
    }

    if (first_init[cell_id] || configs->get_restart_option() != 0)
    {
        NVLOGC_FMT(TAG, "{}: cell_id={} fapi_type=SCF global_tick={} first_init={}", __FUNCTION__, cell_id, global_tick, first_init[cell_id]);
        auto& cv_configs = lp->get_mem_bank_configs(cell_id);
        if (cv_configs.data.size() != 0) {
            send_mem_bank_cv_config_req(cell_id);
        } else {
            send_config_request(cell_id);
        }
    }
    else
    {
        // Restart without CONFIG.request
        cell_start(cell_id);
    }
}

void scf_fapi_handler::cell_start(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }
    NVLOGC_FMT(TAG, "{}: cell_id={} fapi_type=SCF global_tick={}", __FUNCTION__, cell_id, global_tick);
    thrputs[cell_id].reset();
    cell_data[cell_id].fapi_state = fapi_state_t::RUNNING;
    send_start_request(cell_id);
    cell_data[cell_id].schedule_enable = true;
}

void scf_fapi_handler::cell_stop(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return;
    }
    NVLOGC_FMT(TAG, "{}: cell_id={} fapi_type=SCF global_tick={} prach_reconfig={}", __FUNCTION__, cell_id, global_tick, lp->get_prach_reconfig_test());
    cell_data[cell_id].schedule_enable = false;
    thrputs[cell_id].reset();
    send_stop_request(cell_id);

    // Deprecated
    // if (lp->get_prach_reconfig_test())
    // {
    //     lp->apply_reconfig_pattern(cell_id);
    // }
}

int scf_fapi_handler::encode_tx_beamforming(uint8_t* buf, const tx_beamforming_data_t& beam_data, channel_type_t ch_type)
{
    scf_fapi_tx_precoding_beamforming_t* bf = reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(buf);
    bf->num_prgs                            = beam_data.numPRGs;
    bf->prg_size                            = beam_data.prgSize;
    bf->dig_bf_interfaces                   = beam_data.digBFInterfaces;  //lp->get_beam_idx_size() > 0 ? 1 : 0

    // Note: Because of NVIPC msg_buff size limitation we do not encode beamIdx & PMidx for MU-MIMO UE's and consider dig_bf_interfaces=0 and num_prgs=0.
    uint16_t num_prgs = 0;
    if (bf->dig_bf_interfaces != 0)
    {
        num_prgs = bf->num_prgs;
    }

    int offset = sizeof(scf_fapi_tx_precoding_beamforming_t);
    for(int i = 0; i < num_prgs; i++)
    {
        // Append PMidx if num_prgs != 0
        uint16_t* pm_idx = reinterpret_cast<uint16_t*>(buf + offset);
        *(pm_idx)        = beam_data.PMidx_v[i];
        offset += sizeof(uint16_t);
        //NVLOGI_FMT(TAG, "testMac pm_idx ={}",*(pm_idx));

        // Append beamIdx if dig_bf_interfaces != 0
        uint16_t* beam_idx =  reinterpret_cast<uint16_t*>(buf + offset);
        for(int j = 0; j < bf->dig_bf_interfaces; j++)
        {
            *(beam_idx + j) = beam_data.beamIdx_v[j]; //lp->get_beam_idx(beam_id_position);
            //NVLOGI_FMT(TAG, "TX beam_idx: {}-{} = {}", bf->dig_bf_interfaces, j, beam_data.beamIdx_v[j]);
        }
        offset += sizeof(uint16_t) * bf->dig_bf_interfaces;
    }

    return offset;
}

int scf_fapi_handler::encode_rx_beamforming(uint8_t* buf, const rx_beamforming_data_t& beam_data, channel_type_t ch_type)
{
    scf_fapi_rx_beamforming_t* bf = reinterpret_cast<scf_fapi_rx_beamforming_t*>(buf);
    #ifdef SCF_FAPI_10_04_SRS
    bf->trp_scheme                          = 0;
    #endif
    bf->num_prgs                            = beam_data.numPRGs;
    bf->prg_size                            = beam_data.prgSize;
    bf->dig_bf_interfaces                   = beam_data.digBFInterfaces; // lp->get_beam_idx_size() > 0 ? 1 : 0;

    // Note: Because of NVIPC msg_buff size limitation we do not encode beamIdx for MU-MIMO UE's and consider dig_bf_interfaces=0 and num_prgs=0.
    uint16_t num_prgs = 0;
    if (bf->dig_bf_interfaces != 0)
    {
        num_prgs = bf->num_prgs;
    }

    int offset = sizeof(scf_fapi_rx_beamforming_t);
    for(int i = 0; i < num_prgs; i++)
    {
        // Add beamIdx array if digBFInterfaces != 0
        uint16_t* beam_idx = reinterpret_cast<uint16_t*>(buf + offset);
        for(int j = 0; j < bf->dig_bf_interfaces; j++)
        {
            *(beam_idx + j) = beam_data.beamIdx_v[j]; // lp->get_beam_idx(beam_id_position);
            //NVLOGD_FMT(TAG, "RX beam_idx: {}-{} = {}", bf->dig_bf_interfaces, i, beam_data.beamIdx_v[i]);
        }
        offset += sizeof(uint16_t) * bf->dig_bf_interfaces;
    }

    return offset;
}

#ifdef SCF_FAPI_10_04
int scf_fapi_handler::encode_srs_rx_beamforming(uint8_t* buf, const rx_srs_beamforming_data_t& beam_data)
{
    scf_fapi_rx_beamforming_t* bf = reinterpret_cast<scf_fapi_rx_beamforming_t*>(buf);
    #ifdef SCF_FAPI_10_04_SRS
    bf->trp_scheme                          = 0;
    #endif
    bf->num_prgs                            = beam_data.numPRGs;
    bf->prg_size                            = beam_data.prgSize;
    bf->dig_bf_interfaces                   = beam_data.digBFInterfaces; // lp->get_beam_idx_size() > 0 ? 1 : 0;

    uint16_t num_prgs = 0;
    if (bf->dig_bf_interfaces != 0)
    {
        num_prgs = bf->num_prgs;
    }
    int offset = sizeof(scf_fapi_rx_beamforming_t);

    // Add beamIdx array if digBFInterfaces != 0
    for (uint8_t prgIdx = 0; prgIdx < num_prgs; prgIdx++)
    {
        uint16_t* beam_idx = reinterpret_cast<uint16_t*>(buf + offset);
        for (uint8_t digBfIdx = 0; digBfIdx < bf->dig_bf_interfaces ; digBfIdx++)
        {
            *(beam_idx + prgIdx + digBfIdx) = beam_data.beamIdx_v[prgIdx][digBfIdx]; // lp->get_beam_idx(beam_id_position);
            NVLOGD_FMT(TAG, "RX beam_idx: prgIdx={}, digBfIdx={},digBfIdx={}", prgIdx, digBfIdx, beam_data.beamIdx_v[prgIdx][digBfIdx]);
        }
    }
    offset += (sizeof(uint16_t) * (bf->num_prgs) * (bf->dig_bf_interfaces));

    return offset;
}
#endif

int scf_fapi_handler::send_slot_response(int cell_id, sfn_slot_t& ss)
{
    nv::phy_mac_msg_desc msg_desc;
    if(transport().tx_alloc(msg_desc) < 0)
    {
        NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} SLOT.resp", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
        return -1;
    }

    auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_slot_rsp_t>(msg_desc, SCF_FAPI_SLOT_RESPONSE, cell_id, false);
    auto& req  = *reinterpret_cast<scf_fapi_slot_rsp_t*>(fapi);
    req.sfn = ss.u16.sfn;
    req.slot = ss.u16.slot;
    msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
    NVLOGI_FMT(TAG, "SFN {}.{} SEND: cell_id={} {} msg_len={} data_len={}", ss.u16.sfn, ss.u16.slot, cell_id, get_scf_fapi_msg_name(msg_desc.msg_id), msg_desc.msg_len, msg_desc.data_len);
    transport().tx_send(msg_desc);
    if(notify_mode == IPC_SYNC_PER_MSG)
    {
        // Notify once every FAPI message
        transport().notify(1);
    }
    return 0;
}

int scf_fapi_handler::build_dyn_dl_tti_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_tti_req_t& req, dyn_slot_param_t& dyn_param)
{
    req.num_pdus    = 0;
    req.ngroup      = 0;
    uint8_t* data   = reinterpret_cast<uint8_t*>(&req.payload[0]);
    size_t   offset = 0;

    int pdsch_pdu_index = 0;
    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t* fapi_req = fapi_reqs[i];
        if(fapi_req->channel == channel_type_t::PDSCH)
        {
            pdsch_static_param_t& static_param = lp->get_static_slot_param(cell_id).pdsch;
            for(dyn_pdu_param_t& dyn_pdu : dyn_param.pdus)
            {
                auto& pdu     = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
                pdu.pdu_type  = DL_TTI_PDU_TYPE_PDSCH;
                pdu.pdu_size  = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next = pdu.pdu_config;

                // scf_fapi_pdsch_pdu_t and scf_fapi_pdsch_codeword_t[]
                auto& dlschInfo = *reinterpret_cast<scf_fapi_pdsch_pdu_t*>(next);
                // DL tb_pars
                dlschInfo.pdu_bitmap    = static_param.pduBitmap;
                dlschInfo.rnti          = dyn_pdu.rnti;
                dlschInfo.num_codewords = static_param.NrOfCodeWords;
                dlschInfo.pdu_index     = pdsch_pdu_index++;

                dlschInfo.codewords[0].mcs_index        = dyn_pdu.mcs;
                dlschInfo.codewords[0].mcs_table        = dyn_pdu.mcs_table;
                dlschInfo.codewords[0].target_code_rate = dyn_pdu.target_code_rate;
                dlschInfo.codewords[0].qam_mod_order    = dyn_pdu.modulation_order;
                dlschInfo.codewords[0].rv_index         = static_param.rvIndex;
                dlschInfo.codewords[0].tb_size          = dyn_pdu.tb_size;

                dlschInfo.bwp.bwp_start     = static_param.BWPStart;
                dlschInfo.bwp.bwp_size      = static_param.BWPSize;
                dlschInfo.bwp.scs           = static_param.SubCarrierSpacing;
                dlschInfo.bwp.cyclic_prefix = static_param.CyclicPrefix;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
                req.testMode = 0;
#endif
                pdu.pdu_size += sizeof(scf_fapi_pdsch_pdu_t) + dlschInfo.num_codewords * sizeof(scf_fapi_pdsch_codeword_t);
                next = reinterpret_cast<uint8_t*>(dlschInfo.codewords + dlschInfo.num_codewords);

                // scf_fapi_pdsch_pdu_t
                auto& end = *reinterpret_cast<scf_fapi_pdsch_pdu_end_t*>(next);

                end.num_of_layers             = dyn_pdu.layer;
                end.dmrs_config_type          = 0; // type 1
                end.dl_dmrs_sym_pos           = dyn_pdu.dmrs_sym_loc_bmsk;
                end.sc_id                     = static_param.scid;
                end.num_dmrs_cdm_grps_no_data = static_param.numDmrsCdmGrpsNoData[dyn_pdu.layer - 1]; // 1 - 1/2 layers; 2 - 3/4 layers.
                end.dmrs_ports                = dyn_pdu.dmrs_port_bmsk;

                end.resource_alloc     = static_param.resourceAlloc; // type 1
                end.rb_start           = dyn_pdu.prb.prbStart;
                end.rb_size            = dyn_pdu.prb.prbEnd - dyn_pdu.prb.prbStart + 1;
                end.vrb_to_prb_mapping = static_param.VRBtoPRBMapping; // non-interleaved

                end.start_sym_index = DYN_TEST_START_SYMBOL_INDEX;
                end.num_symbols     = dyn_pdu.nrOfSymbols;

                end.ref_point       = static_param.refPoint;
                end.resource_alloc  = static_param.resourceAlloc;

                for(int k = 0; k < static_param.rbBitmap.size(); k++)
                {
                    end.rb_bitmap[k] = static_param.rbBitmap[k];
                }

                end.dl_dmrs_scrambling_id     = static_param.dlDmrsScrmablingId;
                end.data_scrambling_id        = static_param.dataScramblingId;

                pdu.pdu_size += sizeof(scf_fapi_pdsch_pdu_end_t);
                next = end.next;

                // BEAMFORMING CONFIGS
                int bf_size = encode_tx_beamforming(next, static_param.tx_beam_data, fapi_req->channel);
                pdu.pdu_size += bf_size;
                next += bf_size;

                scf_fapi_tx_power_info_t* tp = reinterpret_cast<scf_fapi_tx_power_info_t*>(next);

                tp->power_control_offset    = static_param.powerControlOffset;
                tp->power_control_offset_ss = static_param.powerControlOffsetSS;

                pdu.pdu_size += sizeof(scf_fapi_tx_power_info_t);
                next += sizeof(scf_fapi_tx_power_info_t);

                offset += pdu.pdu_size;
                req.num_pdus++;
            }
        }
    }

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} DL_TTI.req fapi_reqs.size={} dyn_param.pdus.size={} nPDU: PDSCH={}",
            static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, fapi_reqs.size(), dyn_param.pdus.size(), req.num_pdus);
    return offset;
}

// Populate DL_TTI_REQ with values from TVs
int scf_fapi_handler::build_dl_tti_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_tti_req_t& req)
{
    req.num_pdus    = 0;
    req.ngroup      = 0;
    uint8_t* data   = reinterpret_cast<uint8_t*>(&req.payload[0]);
    size_t   offset = 0;

    int pdsch_pdu_index = 0;
    int pbch_num = 0, pdsch_num = 0, pdcch_dl_num = 0, csi_rs_num = 0;
    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t*    fapi_req = fapi_reqs[i];
        const test_vector_t& tv = *fapi_req->tv_data;

        if(fapi_req->channel == channel_type_t::PBCH)
        { // PBCH
            // Add BCH PDU parameters
            for(const pbch_tv_data_t& pbch_pdu : tv.pbch_tv.data)
            {
                auto& pdu    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
                pdu.pdu_type = DL_TTI_PDU_TYPE_SSB;
                pdu.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);

                auto& bchInfo = *reinterpret_cast<scf_fapi_ssb_pdu_t*>(&pdu.pdu_config[0]);
                // TODO: Check the SSB parameter map
                bchInfo.phys_cell_id     = lp->get_cell_configs(cell_id_map[cell_id]).phyCellId;
                bchInfo.beta_pss         = pbch_pdu.betaPss;
                bchInfo.bch_payload_flag = pbch_pdu.bchPayload;

                bchInfo.ssb_subcarrier_offset = pbch_pdu.ssbSubcarrierOffset;
                bchInfo.ssb_block_index       = pbch_pdu.ssbBlockIndex;
                bchInfo.ssb_offset_point_a    = pbch_pdu.SsbOffsetPointA;
                bchInfo.mib_pdu.agg           = pbch_pdu.bchPayload;
                if (configs->get_tick_dynamic_sfn_slot_is_enabled())
                {
                    bchInfo.mib_pdu.agg |= (static_cast<uint32_t>(req.sfn) & 0x3f0) << 13;
                }
                pdu.pdu_size += sizeof(scf_fapi_ssb_pdu_t);

                uint8_t* next = reinterpret_cast<uint8_t*>(bchInfo.pc_and_bf);
                pdu.pdu_size += encode_tx_beamforming(next, pbch_pdu.tx_beam_data, fapi_req->channel);

                offset += pdu.pdu_size;
                req.num_pdus++;
                pbch_num++;
            }
        }

        if(fapi_req->channel == channel_type_t::PDCCH_DL)
        { // PDCCH_DL
            // Add PDCCH_DL PDU parameters
            for(auto& coreset : tv.pdcch_tv.coreset)
            {
                auto& pdu    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
                pdu.pdu_type = DL_TTI_PDU_TYPE_PDCCH;

                auto& dciInfo             = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&pdu.pdu_config[0]);
                dciInfo.bwp.bwp_size      = coreset.BWPSize;  // tb_pars.numPrb;
                dciInfo.bwp.bwp_start     = coreset.BWPStart; // tb_pars.startPrb;
                dciInfo.bwp.scs           = coreset.SubcarrierSpacing;
                dciInfo.bwp.cyclic_prefix = coreset.CyclicPrefix;
                dciInfo.coreset_type      = coreset.CoreSetType;
                dciInfo.num_dl_dci        = coreset.numDlDci;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
                req.testMode              = coreset.testModel;
#endif
                //dciInfo.nBeamId                = 0;
                dciInfo.start_sym_index = coreset.StartSymbolIndex;
                dciInfo.duration_sym    = coreset.DurationSymbols;

                // memcpy copies in the reverse order
                // memcpy(&dciInfo.freq_domain_resource[0], (uint8_t*)&coreset.FreqDomainResource + 2, sizeof(dciInfo.freq_domain_resource));
                for(int i = 0; i < sizeof(dciInfo.freq_domain_resource); ++i)
                {
                    dciInfo.freq_domain_resource[i] = ((uint8_t*)&coreset.FreqDomainResource)[7 - i];
                }
                dciInfo.cce_reg_mapping_type = coreset.CceRegMappingType;
                dciInfo.reg_bundle_size      = coreset.RegBundleSize;
                dciInfo.interleaver_size     = coreset.InterleaverSize;
                dciInfo.shift_index          = coreset.ShiftIndex;

                std::size_t dciOffset = 0;
                uint8_t*    dldci_buf = reinterpret_cast<uint8_t*>(dciInfo.dl_dci);
                for(uint16_t i = 0; i < coreset.numDlDci; i++)
                {
                    auto& dci                     = coreset.dciList[i];
                    auto& dci_payload             = *reinterpret_cast<scf_fapi_dl_dci_t*>(dldci_buf + dciOffset); // Only 1 for now
                    dci_payload.rnti              = dci.RNTI;
                    dci_payload.scrambling_id     = dci.ScramblingId;
                    dci_payload.scrambling_rnti   = dci.ScramblingRNTI;
                    dci_payload.cce_index         = dci.CceIndex;
                    dci_payload.aggregation_level = dci.AggregationLevel;

                    // BEAMFORMING CONFIGS
                    int bf_size = encode_tx_beamforming(dci_payload.payload, dci.tx_beam_data, fapi_req->channel);

                    // TX POWER INFO CONFIGS
                    auto& pwr_info                   = *reinterpret_cast<scf_fapi_pdcch_tx_power_info_t*>(&dci_payload.payload[bf_size]);
                    pwr_info.beta_pdcch_1_0          = dci.beta_PDCCH_1_0;
#ifdef SCF_FAPI_10_04
                    pwr_info.power_control_offset_ss_profile_nr = dci.powerControlOffsetSSProfileNR;
#else
                    pwr_info.power_control_offset_ss = dci.powerControlOffsetSS;
#endif
                    auto tx_power_info_size          = sizeof(scf_fapi_pdcch_tx_power_info_t);

                    // DCI CONFIGS END
                    auto& dci_end             = *reinterpret_cast<scf_fapi_pdcch_dci_payload_t*>(&dci_payload.payload[bf_size + tx_power_info_size]); // Only 1 for now
                    dci_end.payload_size_bits = dci.PayloadSizeBits;

                    int len = (dci_end.payload_size_bits + 7) / 8;
                    std::copy(dci.Payload, dci.Payload + len, dci_end.payload);
                    dciOffset += sizeof(scf_fapi_dl_dci_t) + bf_size + tx_power_info_size + sizeof(scf_fapi_pdcch_dci_payload_t) + len;
                    // NVLOGI_FMT(TAG, "DL_DCI {} {} dci_offset={} bf_size={} tx_power_info_size={} len={}", __FUNCTION__, i, dciOffset, bf_size, tx_power_info_size , len);
                }
                pdu.pdu_size = sizeof(scf_fapi_generic_pdu_info_t) + sizeof(scf_fapi_pdcch_pdu_t) + dciOffset;
                offset += pdu.pdu_size;
                req.num_pdus++;
                pdcch_dl_num++;
            }
        }

        if(fapi_req->channel == channel_type_t::PDSCH)
        {
            for(pdsch_tv_data_t* tv_data : tv.pdsch_tv.data)
            {
                struct tb_pars& tb_pars = tv_data->tbpars;

                auto& pdu     = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
                pdu.pdu_type  = DL_TTI_PDU_TYPE_PDSCH;
                pdu.pdu_size  = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next = pdu.pdu_config;

                // scf_fapi_pdsch_pdu_t and scf_fapi_pdsch_codeword_t[]
                auto& dlschInfo = *reinterpret_cast<scf_fapi_pdsch_pdu_t*>(next);
                // DL tb_pars
                dlschInfo.pdu_bitmap    = 0;
                dlschInfo.rnti          = tb_pars.nRnti; // rnti_list[cell_id];
                dlschInfo.num_codewords = 1;
                dlschInfo.pdu_index     = pdsch_pdu_index++;

                dlschInfo.codewords[0].mcs_index = tb_pars.mcsIndex;
                dlschInfo.codewords[0].mcs_table = tb_pars.mcsTableIndex;
                dlschInfo.codewords[0].target_code_rate = tb_pars.targetCodeRate; // FYI cuPHY PDSCH has a runtime check to ensure code rate isn't 0
                dlschInfo.codewords[0].qam_mod_order    = tb_pars.qamModOrder;
                dlschInfo.codewords[0].tb_size   = tv_data->tb_size;
                dlschInfo.codewords[0].rv_index  = tb_pars.rv;

                dlschInfo.bwp.bwp_start = tv_data->BWPStart;
                dlschInfo.bwp.bwp_size  = tv_data->BWPSize;
                dlschInfo.bwp.scs       = tv_data->SubcarrierSpacing;
                dlschInfo.bwp.cyclic_prefix = tv_data->CyclicPrefix;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
                req.testMode              = tv_data->testModel;
#endif
                pdu.pdu_size += sizeof(scf_fapi_pdsch_pdu_t) + dlschInfo.num_codewords * sizeof(scf_fapi_pdsch_codeword_t);
                next = reinterpret_cast<uint8_t*>(dlschInfo.codewords + dlschInfo.num_codewords);

                // scf_fapi_pdsch_pdu_t
                auto& end = *reinterpret_cast<scf_fapi_pdsch_pdu_end_t*>(next);

                end.rb_start        = tb_pars.startPrb;
                end.rb_size         = tb_pars.numPrb;
                end.start_sym_index = tb_pars.startSym;
                end.num_symbols     = tb_pars.numSym;
                end.num_of_layers   = tb_pars.numLayers;
                end.ref_point       = tv_data->ref_point;
                end.resource_alloc  = tv_data->resourceAlloc;

                end.vrb_to_prb_mapping  = tv_data->VRBtoPRBMapping;
                end.transmission_scheme = tv_data->transmissionScheme;

                for (int k = 0; k < tv_data->rbBitmap.size(); k++)
                {
                    end.rb_bitmap[k] = tv_data->rbBitmap[k];
                }

                end.dmrs_config_type = tb_pars.dmrsType;
                end.dl_dmrs_sym_pos  = tv_data->dmrsSymLocBmsk;
                //end.nDMRSAddPos       = tb_pars.dmrsAddlPosition;
                //end.nNrOfDMRSSymbols  = tb_pars.dmrsMaxLength;
                end.sc_id                 = tb_pars.nSCID;
                end.dl_dmrs_scrambling_id = tb_pars.dmrsScramId;
                end.data_scrambling_id    = tb_pars.dataScramId;
                // end.dmrs_ports        = (!!(tb_pars.nPortIndex & 0xf0000000) << 0) |
                //                         (!!(tb_pars.nPortIndex & 0x0f000000) << 1) |
                //                         (!!(tb_pars.nPortIndex & 0x00f00000) << 2) |
                //                         (!!(tb_pars.nPortIndex & 0x000f0000) << 3) |
                //                         (!!(tb_pars.nPortIndex & 0x0000f000) << 4) |
                //                         (!!(tb_pars.nPortIndex & 0x00000f00) << 5) |
                //                         (!!(tb_pars.nPortIndex & 0x000000f0) << 6) |
                //                         (!!(tb_pars.nPortIndex & 0x0000000f) << 7);
                end.dmrs_ports                = static_cast<uint16_t>(tv_data->tbpars.nPortIndex);
                end.num_dmrs_cdm_grps_no_data = tv_data->numDmrsCdmGrpsNoData;

                pdu.pdu_size += sizeof(scf_fapi_pdsch_pdu_end_t);
                next = end.next;

                // scf_fapi_pdsch_ptrs_t if pdu_bitmap bit1=1
                if(dlschInfo.pdu_bitmap & 0x1)
                {
                    scf_fapi_pdsch_ptrs_t* ptrs         = reinterpret_cast<scf_fapi_pdsch_ptrs_t*>(next);
                    ptrs->ptrs_port_index               = 0;
                    ptrs->ptrs_time_density             = 0;
                    ptrs->ptrs_freq_density             = 0;
                    ptrs->ptrs_re_offset                = 0;
                    ptrs->n_epre_ratio_of_pdsch_to_ptrs = 0;

                    pdu.pdu_size += sizeof(scf_fapi_pdsch_ptrs_t);
                    next = ptrs->next;
                }

                // BEAMFORMING CONFIGS
                int bf_size = encode_tx_beamforming(next, tv_data->tx_beam_data, fapi_req->channel);
                pdu.pdu_size += bf_size;
                next += bf_size;

                scf_fapi_tx_power_info_t* tp = reinterpret_cast<scf_fapi_tx_power_info_t*>(next);

                tp->power_control_offset     = tv_data->powerControlOffset;
                tp->power_control_offset_ss  = tv_data->powerControlOffsetSS;

                pdu.pdu_size += sizeof(scf_fapi_tx_power_info_t);
                next += sizeof(scf_fapi_tx_power_info_t);

                if(dlschInfo.pdu_bitmap & 0x2)
                {
                    // TODO: Add CBG fields according to SCF 222, section 3.4.2.2
                }

                offset += pdu.pdu_size;
                req.num_pdus++;
                pdsch_num++;
            }
        }

        if(fapi_req->channel == channel_type_t::CSI_RS)
        {
            for(auto& csirs_pdu : tv.csirs_tv.data)
            {
                auto& pdu                                 = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
                pdu.pdu_type                              = DL_TTI_PDU_TYPE_CSI_RS;
                pdu.pdu_size                              = sizeof(scf_fapi_generic_pdu_info_t);
                auto& pdu_info                            = *reinterpret_cast<scf_fapi_csi_rsi_pdu_t*>(pdu.pdu_config);
                pdu_info.bwp.bwp_size                     = csirs_pdu.BWPSize;
                pdu_info.bwp.bwp_start                    = csirs_pdu.BWPStart;
                pdu_info.bwp.scs                          = csirs_pdu.SubcarrierSpacing;
                pdu_info.bwp.cyclic_prefix                = csirs_pdu.CyclicPrefix;
                pdu_info.start_rb                         = csirs_pdu.StartRB;
                pdu_info.num_of_rbs                       = csirs_pdu.NrOfRBs;
                pdu_info.csi_type                         = csirs_pdu.CSIType;
                pdu_info.row                              = csirs_pdu.Row;
                pdu_info.freq_domain                      = csirs_pdu.FreqDomain;
                pdu_info.sym_l0                           = csirs_pdu.SymbL0;
                pdu_info.sym_l1                           = csirs_pdu.SymbL1;
                pdu_info.cdm_type                         = csirs_pdu.CDMType;
                pdu_info.freq_density                     = csirs_pdu.FreqDensity;
                pdu_info.scrambling_id                    = csirs_pdu.ScrambId;
                pdu_info.tx_power.power_control_offset    = csirs_pdu.powerControlOffset;
                pdu_info.tx_power.power_control_offset_ss = csirs_pdu.powerControlOffsetSS;

                pdu.pdu_size += sizeof(scf_fapi_csi_rsi_pdu_t) - sizeof(scf_fapi_tx_precoding_beamforming_t);

                // BEAMFORMING CONFIGS
                uint8_t* bf = reinterpret_cast<uint8_t*>(&pdu_info.pc_and_bf);
                pdu.pdu_size += encode_tx_beamforming(bf, csirs_pdu.tx_beam_data, fapi_req->channel);

                NVLOGD_FMT(TAG, "{}: CSI_RS PDU: CSIType={}", __FUNCTION__, csirs_pdu.CSIType);

                offset += pdu.pdu_size;
                req.num_pdus++;
                csi_rs_num++;
            }
        }
    }

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} DL_TTI.req nPDU: PBCH={} PDSCH={} PDCCH_DL={} CSI_RS={}",
            static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, pbch_num, pdsch_num, pdcch_dl_num, csi_rs_num);
    return offset;
}

int get_tx_data_len(vector<fapi_req_t*>& fapi_reqs)
{
    int total_tb_size = 0;
    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t*    fapi_req = fapi_reqs[i];
        const test_vector_t& tv = *fapi_req->tv_data;
        if(fapi_req->channel == channel_type_t::PDSCH)
        { // PDSCH_DL
            for(auto tv_data : tv.pdsch_tv.data)
            {
                if(tv_data->tb_size == 0 && tv_data->tb_buf == NULL)
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: TB size is empty. tb_size={}", __FUNCTION__, tv_data->tb_size);
                    continue;
                }

                total_tb_size += tv_data->tb_size;
            }
        }
    }
    return total_tb_size;
}

int scf_fapi_handler::build_dyn_tx_data_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_tx_data_req_t& req, phy_mac_msg_desc& msg_desc, dyn_slot_param_t& dyn_param)
{
    int      pdsch_pdu_index = 0; // Make sure pdu_index the same with which in DL_CONFIG_Request.
    size_t   msg_offset      = 0;
    size_t   pdu_offset      = 0;
    uint8_t* pdu_addr;
    uint8_t* msg_addr = reinterpret_cast<uint8_t*>(req.payload);

    req.num_pdus = 0;

#if COPY_TIME_STAT_ENABLED
    struct timespec beg, end;
    int64_t         tdiff = 0;
    clock_gettime(CLOCK_MONOTONIC, &beg);
#endif

    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t*          fapi_req = fapi_reqs[i];
        if(fapi_req->channel == channel_type_t::PDSCH)
        { // PDSCH_DL
            for(dyn_pdu_param_t& dyn_pdu : dyn_param.pdus)
            {
                auto& dlpdu = *(reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(msg_addr + msg_offset));

                // Overwrite tb_size from JSON dynamic parameter
                size_t pdu_tb_size = dyn_pdu.tb_size;

                if(pdu_tb_size == 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: TB size is empty. tb_size={}", __FUNCTION__, pdu_tb_size);
                    continue;
                }

                dlpdu.pdu_index = pdsch_pdu_index++;
                dlpdu.num_tlv   = 1;
#ifdef SCF_FAPI_10_04
                dlpdu.pdu_len = sizeof(scf_fapi_tx_data_pdu_info_t) + dlpdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));
#else
                dlpdu.pdu_len = pdu_tb_size;
#endif
                scf_fapi_tl_t* tlv = dlpdu.tlvs;
                tlv->Set<uint32_t>(SCF_TX_DATA_OFFSET, pdu_offset);
#ifdef SCF_FAPI_10_04
                tlv->length = pdu_tb_size;
#endif

                NVLOGD_FMT(TAG, "{}: PDSCH: pdu_index={} num_tlv={} pdu_len={} pdu_offset={}", __FUNCTION__, static_cast<unsigned short>(dlpdu.pdu_index), static_cast<unsigned>(dlpdu.num_tlv), static_cast<unsigned>(dlpdu.pdu_len), pdu_offset);

                // Add counter and offset
                req.num_pdus++;
                msg_offset += sizeof(scf_fapi_tx_data_pdu_info_t) + dlpdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));

                // Calculate padding byte number to be appended
                uint32_t padding_nbytes = (~pdu_tb_size + 1) & (configs->pdsch_align_bytes -1);

                pdu_offset += pdu_tb_size + padding_nbytes;
            }

            // Copy the TB data once per cell
            // Handle both CPU_DATA and CPU_LARGE pools with memcpy
            if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA || msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_LARGE)
            {
                memcpy(msg_desc.data_buf, dyn_tb_data_gen_buf.data(), pdu_offset);
            }
            else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
            {
                transport().tx_copy(msg_desc.data_buf, dyn_tb_data_gen_buf.data(), pdu_offset, msg_desc.data_pool);
            }
        }
        else
        {
            NVLOGD_FMT(TAG, "{}: channel is not supported in TX_DATA request: {}", __FUNCTION__, +fapi_req->channel);
            continue;
        }

        NVLOGD_FMT(TAG, "{}: add pdu: {}-{} channel={} nPDU={} pdu_offset={}", __FUNCTION__, fapi_reqs.size(), i, +fapi_req->channel, static_cast<int>(req.num_pdus), pdu_offset);
    }

#if COPY_TIME_STAT_ENABLED
    clock_gettime(CLOCK_MONOTONIC, &end);
    tdiff = (end.tv_sec - beg.tv_sec) * 1000000000LL + end.tv_nsec - beg.tv_nsec;
    if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA)
    {
        NVLOGI_FMT(TAG, "COPY: cell_id {} tdiff {} us", cell_id, tdiff / 1000);
    }
    else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
    {
        NVLOGI_FMT(TAG, "GDRCOPY: cell_id {} tdiff {} us", cell_id, tdiff / 1000);
    }
#endif

    req.msg_hdr.length = sizeof(scf_fapi_tx_data_req_t) - sizeof(scf_fapi_body_header_t) + msg_offset;
    msg_desc.msg_len   = req.msg_hdr.length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
    msg_desc.data_len  = pdu_offset;

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} TX_DATA.req nPDU={}", static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, static_cast<int>(req.num_pdus));

    if(msg_desc.data_len > configs->get_max_data_size())
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}:{} Error: IPC data buffer overflowed: buf_size={} pdu_offset={}", __FILE__, __LINE__, configs->get_max_data_size(), pdu_offset);
        return -1;
    }
    else
    {
        return 0;
    }

    return 0;
}

int scf_fapi_handler::build_tx_data_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_tx_data_req_t& req, phy_mac_msg_desc& msg_desc)
{
    int      pdsch_pdu_index = 0; // Make sure pdu_index the same with which in DL_CONFIG_Request.
    size_t   msg_offset      = 0;
    size_t   pdu_offset      = 0;
    uint8_t* pdu_addr;
    uint8_t* msg_addr = reinterpret_cast<uint8_t*>(req.payload);

    req.num_pdus = 0;

#if COPY_TIME_STAT_ENABLED
    struct timespec beg, end;
    int64_t tdiff = 0;
    clock_gettime(CLOCK_MONOTONIC, &beg);
#endif

    for(int i = 0; i < fapi_reqs.size(); i++)
    {
        fapi_req_t*    fapi_req = fapi_reqs[i];
        const test_vector_t& tv = *fapi_req->tv_data;
        if(fapi_req->channel == channel_type_t::PDSCH)
        { // PDSCH_DL
            // Copy the TB data once per cell
            // Handle both CPU_DATA and CPU_LARGE pools with memcpy
            if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA || msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_LARGE)
            {
                memcpy(msg_desc.data_buf, tv.pdsch_tv.data_buf.data(), tv.pdsch_tv.data_size);
            }
            else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
            {
               transport().tx_copy(msg_desc.data_buf, tv.pdsch_tv.data_buf.data(), tv.pdsch_tv.data_size, msg_desc.data_pool);
            }

            for(auto tv_data : tv.pdsch_tv.data)
            {
                auto& dlpdu = *(reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(msg_addr + msg_offset));

                if(tv_data->tb_size == 0 && tv_data->tb_buf == NULL)
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: TB size is empty. tb_size={}", __FUNCTION__, tv_data->tb_size);
                    continue;
                }

                dlpdu.pdu_index = pdsch_pdu_index++;
                dlpdu.num_tlv   = 1;
#ifdef SCF_FAPI_10_04
                dlpdu.cw_index = 0; // Not used, set to 0 by default.
                dlpdu.pdu_len   = sizeof(scf_fapi_tx_data_pdu_info_t) + dlpdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));
#else
                // FAPI 10.02 tlv->length is uint16_t which may not be enough for tb_size. So use pdu_len instead.
                dlpdu.pdu_len = tv_data->tb_size;
#endif
                scf_fapi_tl_t* tlv = dlpdu.tlvs;
                tlv->Set<uint32_t>(SCF_TX_DATA_OFFSET, pdu_offset);
#ifdef SCF_FAPI_10_04
                tlv->length = tv_data->tb_size;
#endif

                NVLOGD_FMT(TAG, "{}: data_buf={} tb_buf={} tb_size={} pdu_offset={}", __func__, reinterpret_cast<void*>(msg_desc.data_buf), reinterpret_cast<void*>(tv_data->tb_buf), tv_data->tb_size, pdu_offset);

                if (configs->get_tick_dynamic_sfn_slot_is_enabled())
                {
                    if(tv.pdsch_tv.type == tv_type_t::TV_PRACH_MSG2)
                    {
                        // RAPID is in the 6 lower bits of the first byte of PDU
                        set_prach_msg2_rapid(cell_id, (uint8_t*)msg_desc.data_buf + pdu_offset);
                    }
                }

                // Add counter and offset
                req.num_pdus++;
                msg_offset += sizeof(scf_fapi_tx_data_pdu_info_t) + dlpdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));

                // Calculate padding byte number to be appended
                uint32_t padding_nbytes = (~tv_data->tb_size + 1) & (configs->pdsch_align_bytes -1);

                NVLOGD_FMT(TAG, "{}: PDSCH: pdu_index={} pdu_len={} tb_size={} padding_nbytes={} pdu_offset={}",
                        __FUNCTION__, static_cast<unsigned short>(dlpdu.pdu_index), static_cast<unsigned>(dlpdu.pdu_len), tv_data->tb_size, padding_nbytes, pdu_offset);

                pdu_offset += tv_data->tb_size + padding_nbytes;
            }
        }
        else
        {
            NVLOGD_FMT(TAG, "{}: channel is not supported in TX_DATA request: {}", __FUNCTION__, +fapi_req->channel);
            continue;
        }

        NVLOGD_FMT(TAG, "{}: add pdu: {}-{} channel={} nPDU={} pdu_offset={}", __FUNCTION__, fapi_reqs.size(), i, +fapi_req->channel, static_cast<int>(req.num_pdus), pdu_offset);
    }

#if COPY_TIME_STAT_ENABLED
    clock_gettime(CLOCK_MONOTONIC, &end);
    tdiff = (end.tv_sec - beg.tv_sec) * 1000000000LL + end.tv_nsec - beg.tv_nsec;
    if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA)
    {
        NVLOGI_FMT(TAG,"COPY: cell_id {} tdiff {} us",cell_id,tdiff/1000);
    }
    else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
    {
        NVLOGI_FMT(TAG,"GDRCOPY: cell_id {} tdiff {} us",cell_id,tdiff/1000);
    }
#endif

    req.msg_hdr.length = sizeof(scf_fapi_tx_data_req_t) - sizeof(scf_fapi_body_header_t) + msg_offset;
    msg_desc.msg_len   = req.msg_hdr.length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
    msg_desc.data_len  = pdu_offset;

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} TX_DATA.req nPDU={}", static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, static_cast<int>(req.num_pdus));

    if(msg_desc.data_len > configs->get_max_data_size())
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}:{} Error: IPC data buffer overflowed: buf_size={} pdu_offset={}", __FILE__, __LINE__, configs->get_max_data_size(), pdu_offset);
        return -1;
    }
    else
    {
        return 0;
    }

    return 0;
}
void scf_fapi_handler :: update_prach_ocassion(const prach_tv_data_t&  prach_pars)
{
    //Start incrementing all counters after first successful detection.
    if((0 == conformance_test_stats->get_prach_stats().preamble_detected))
    {
        conformance_test_slot_counter = 0;
        conformance_test_stats->get_prach_stats().prach_occassion = 0;
        conformance_test_stats->get_prach_stats().preamble_error = 0;
        conformance_test_stats->get_prach_stats().preamble_timing_offset_error = 0;
    }
    conformance_test_stats->get_prach_stats().preamble_id = prach_pars.ref.prmbIdx_v[0];
    conformance_test_stats->get_prach_stats().timing_advance_range_low =
        (prach_pars.ref.delay_v[0] - configs->get_conformance_test_params()->prach_params.delay_error_tolerance);
    conformance_test_stats->get_prach_stats().timing_advance_range_high =
        (prach_pars.ref.delay_v[0] + configs->get_conformance_test_params()->prach_params.delay_error_tolerance);
    conformance_test_stats->get_prach_stats().prach_occassion++;
    NVLOGD_FMT(TAG,"update_prach_ocassion prach_occassion={} conformance_test_slot_counter {}",
    conformance_test_stats->get_prach_stats().prach_occassion,
    conformance_test_slot_counter);

}


void scf_fapi_handler:: update_pucch_ocassion(scf_fapi_pucch_pdu_t& pucch_pdu)
{

    //Till first  HARQ ACK or CRC sucess for CSI part1 slot count will not increase.
    if((conformance_test_stats->get_pf0_stats().pf0_ack |
    conformance_test_stats->get_pf1_stats().pf1_ack_bits |
    conformance_test_stats->get_pf2_stats().pf2_harq_ack_bits|
    conformance_test_stats->get_pf2_stats().pf2_harq_nack_bits|
    conformance_test_stats->get_pf2_stats().pf2_csi_success|
    conformance_test_stats->get_pf3_stats().pf3_csi_success)== 0)
    {
        conformance_test_slot_counter = 0;
        memset(&conformance_test_stats->get_pf0_stats(), 0, sizeof(pf0_conformance_test_t));
        memset(&conformance_test_stats->get_pf1_stats(), 0, sizeof(pf1_conformance_test_t));
        memset(&conformance_test_stats->get_pf2_stats(), 0, sizeof(pf2_conformance_test_t));
        memset(&conformance_test_stats->get_pf3_stats(), 0, sizeof(pf3_conformance_test_t));
    }
    switch(pucch_pdu.format_type)
    {
        case 0:
        {
            conformance_test_stats->get_pf0_stats().pf0_occassion++;
        }
        break;
        case 1:
        {
            conformance_test_stats->get_pf1_stats().pf1_occassion++;
            conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern =
            configs->get_conformance_test_params()->pucch_params.ack_nack_pattern;
        }
        break;
        case 2:
        {
            if(pucch_pdu.bit_len_harq > 0)
            conformance_test_stats->get_pf2_stats().pf2_harq_occassion++;
            if(pucch_pdu.bit_len_csi_part_1 > 0)
            conformance_test_stats->get_pf2_stats().pf2_csi_occassion++;

        }
        break;
        case 3:
        {
            conformance_test_stats->get_pf3_stats().pf3_occassion++;
        }
        break;
        default:
        break;
    }

}

int scf_fapi_handler::build_dyn_ul_tti_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_tti_req_t& req, dyn_slot_param_t& dyn_param)
{
    size_t offset    = 0;
    req.num_pdus     = 0;
    req.num_ulsch    = 0;
    req.num_ulcch    = 0;
    req.rach_present = 0;
    req.ngroup       = 0;

    uint32_t handle_id_pusch_uci  = 0;
    uint32_t handle_id_pusch_data = 0;
    uint32_t handle_id_pucch      = 0;

    uint8_t* buffer    = reinterpret_cast<uint8_t*>(req.payload);
    for(fapi_req_t* fapi_req : fapi_reqs)
    {
        if(fapi_req->channel == channel_type_t::PUSCH)
        {
            pusch_static_param_t static_param = lp->get_static_slot_param(cell_id).pusch;
            int pdu_id = 0;
            for(dyn_pdu_param_t& dyn_pdu : dyn_param.pdus)
            {
                auto& pduinfo    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buffer + offset));
                pduinfo.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next    = pduinfo.pdu_config;
                pduinfo.pdu_type = UL_TTI_PDU_TYPE_PUSCH;

                auto& ulsch_pdu = *reinterpret_cast<scf_fapi_pusch_pdu_t*>(next);

                ulsch_pdu.handle                      = rand(); //Use random value for handle for testing
                ulsch_pdu.rnti                        = dyn_pdu.rnti;
                ulsch_pdu.rb_start                    = dyn_pdu.prb.prbStart;
                ulsch_pdu.rb_size                     = dyn_pdu.prb.prbEnd - dyn_pdu.prb.prbStart + 1;
                ulsch_pdu.start_symbol_index          = DYN_TEST_START_SYMBOL_INDEX;
                ulsch_pdu.num_of_symbols              = dyn_pdu.nrOfSymbols;
                ulsch_pdu.num_of_layers               = dyn_pdu.layer;
                ulsch_pdu.mcs_table                   = dyn_pdu.mcs_table;
                ulsch_pdu.mcs_index                   = dyn_pdu.mcs;
                ulsch_pdu.ul_dmrs_scrambling_id       = static_param.ulDmrsScramblingId;
                ulsch_pdu.dmrs_config_type            = static_param.dmrsConfigType;
                ulsch_pdu.data_scrambling_id          = static_param.dataScramblingId;
                ulsch_pdu.ul_dmrs_sym_pos             = dyn_pdu.dmrs_sym_loc_bmsk;
                ulsch_pdu.dmrs_ports                  = dyn_pdu.dmrs_port_bmsk;
                ulsch_pdu.num_dmrs_cdm_groups_no_data = static_param.numDmrsCdmGrpsNoData[dyn_pdu.layer - 1]; // 1 - 1/2 layers; 2 - 3/4 layers.
                ulsch_pdu.pusch_identity              = static_param.puschIdentity;
                ulsch_pdu.scid                        = static_param.scid;
                ulsch_pdu.qam_mod_order               = dyn_pdu.modulation_order;

                ulsch_pdu.bwp.scs                    = static_param.SubCarrierSpacing;
                ulsch_pdu.bwp.cyclic_prefix          = static_param.CyclicPrefix;
                ulsch_pdu.target_code_rate           = dyn_pdu.target_code_rate;
                ulsch_pdu.frequency_hopping          = static_param.FrequencyHopping;
                ulsch_pdu.tx_direct_current_location = static_param.txDirectCurrentLocation;
                ulsch_pdu.ul_frequency_shift_7p5_khz = static_param.uplinkFrequencyShift7p5khz;

                ulsch_pdu.bwp.bwp_start       = static_param.BWPStart;
                ulsch_pdu.bwp.bwp_size        = static_param.BWPSize;
                ulsch_pdu.pdu_bitmap          = static_param.pduBitmap;
                ulsch_pdu.transform_precoding = 1; // 1 - disabled. TODO: set static_param.transformPrecoding

                pduinfo.pdu_size += sizeof(scf_fapi_pusch_pdu_t);
                next = ulsch_pdu.payload;

                if(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DATA)
                {
                    ulsch_pdu.handle = handle_id_pusch_data++;
                    auto& pusch_data = *reinterpret_cast<scf_fapi_pusch_data_t*>(next);
                    // HARQ indication
                    pusch_data.rv_index           = static_param.rvIndex;
                    pusch_data.new_data_indicator = static_param.newDataIndicator;
                    if(configs->get_conformance_test_params()->conformance_test_enable)
                    {
                        pusch_data.harq_process_id            = static_param.harqProcessID;
                        uint16_t          ue_id               = rnti_2_ueid_map[ulsch_pdu.rnti];
                        ul_harq_handle_t& curr_ul_harq_handle = ul_harq_handle[cell_id][ue_id][cell_data[cell_id].harq_process_id];
                        pusch_data.new_data_indicator         = curr_ul_harq_handle.pusch_pdu_NDI;
                        pusch_data.rv_index                   = curr_ul_harq_handle.rv;
                    }
                    else
                    {
                        if(lp->get_config_static_harq_proc_id() > 0)
                        {
                            pusch_data.harq_process_id = static_param.harqProcessID;
                        }
                        else
                        {
                            pusch_data.harq_process_id         = cell_data[cell_id].harq_process_id;
                            cell_data[cell_id].harq_process_id = (cell_data[cell_id].harq_process_id + 1) % FAPI_MAX_UL_HARQ_ID;
                        }
                    }
                    // Save harq_process_id for validation latter
                    // lp->save_harq_pid(cell_id, req.sfn, req.slot, pdu_id++, pusch_data.harq_process_id);

                    pusch_data.tb_size = dyn_pdu.tb_size;
                    pusch_data.num_cb  = static_param.numCb;

                    int nbytes = (pusch_data.num_cb + 7) / 8;
                    for(int i = 0; i < nbytes; i++)
                    {
                        pusch_data.cb_present_and_position[i] = static_param.cbPresentAndPosition;
                    }
                    pduinfo.pdu_size += sizeof(scf_fapi_pusch_data_t) + nbytes;
                    next = pusch_data.cb_present_and_position + nbytes;
                }

                // PUSCH DFT-s-OFDM
                if(ulsch_pdu.transform_precoding == 0)
                {
                    if(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DFTSOFDM)
                    {
                        scf_fapi_pusch_dftsofdm_t* puschDftsOfdm = reinterpret_cast<scf_fapi_pusch_dftsofdm_t*>(next);
                        puschDftsOfdm->lowPaprGroupNumber        = 0;
                        puschDftsOfdm->lowPaprSequenceNumber     = 0;
                        pduinfo.pdu_size += sizeof(scf_fapi_pusch_dftsofdm_t);
                        next += sizeof(scf_fapi_pusch_dftsofdm_t);
                    }
                }

                // BEAMFORMING CONFIGS
                int bf_size = encode_rx_beamforming(next, static_param.rx_beam_data, fapi_req->channel);
                pduinfo.pdu_size += bf_size;
                next += bf_size;

                // PUSCH Maintenance Parameters
#ifdef SCF_FAPI_10_04
                scf_fapi_pusch_maintenance_t* puschMaintenance = reinterpret_cast<scf_fapi_pusch_maintenance_t*>(next);
                if(ulsch_pdu.transform_precoding == 0)
                {
                    if(!(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DFTSOFDM))
                    {
                        puschMaintenance->groupOrSequenceHopping = 0; // TODO: provide value for transform_precoding = 0 enabled case
                    }
                }
                pduinfo.pdu_size += sizeof(scf_fapi_pusch_maintenance_t);
                next += sizeof(scf_fapi_pusch_maintenance_t);
#endif

                for(int i = 0; i < req.ngroup; i++)
                {
                    scf_fapi_ue_group_t* gp = reinterpret_cast<scf_fapi_ue_group_t*>(next);
                    gp->num_ue              = 1; // TODO: set the right UE number in each group
                    for(int j = 0; j < gp->num_ue; j++)
                    {
                        gp->pdu_index[j] = j; // TODO: set the right pdu index if UE group exists
                    }
                    pduinfo.pdu_size += sizeof(scf_fapi_ue_group_t) + gp->num_ue;
                    next = gp->pdu_index + gp->num_ue;
                }

                offset += pduinfo.pdu_size;
                req.num_pdus++;
                req.num_ulsch++;

                // Log print the TB size and offset
                NVLOGD_FMT(TAG, "{}: PUSCH handle={} pdu_bitmap={} offset={} rnti={} hpid={}", __FUNCTION__, static_cast<unsigned>(ulsch_pdu.handle), static_cast<unsigned short>(ulsch_pdu.pdu_bitmap), offset, static_cast<unsigned short>(ulsch_pdu.rnti), cell_data[cell_id].harq_process_id);
            }
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_NO_SUPPORT_EVENT, "{}: channel is not supported yet: {}", __FUNCTION__, static_cast<unsigned>(fapi_req->channel));
        }
    }

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} UL_TTI.req num_pdus={}", static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, req.num_pdus);
    return offset;
}

int scf_fapi_handler::build_ul_tti_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_tti_req_t& req)
{
    size_t offset    = 0;
    req.num_pdus     = 0;
    req.num_ulsch    = 0;
    req.num_ulcch    = 0;
    req.rach_present = 0;
    req.ngroup       = 0;

    uint32_t handle_id_pusch_uci = 0;
    uint32_t handle_id_pusch_data = 0;
    uint32_t handle_id_pucch = 0;

    uint8_t* buffer = reinterpret_cast<uint8_t*>(req.payload);
    int prach_num = 0, pusch_num = 0, pucch_num = 0, srs_num = 0;
    for(fapi_req_t* fapi_req : fapi_reqs)
    {
        const test_vector_t& tv = *fapi_req->tv_data;
        if(fapi_req->channel == channel_type_t::PRACH)
        {
            for(auto& prach_pars : tv.prach_tv.data)
            {
                auto& pduinfo    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buffer + offset));
                pduinfo.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next    = pduinfo.pdu_config;

                pduinfo.pdu_type             = UL_TTI_PDU_TYPE_PRACH;
                auto& prach_pdu              = *reinterpret_cast<scf_fapi_prach_pdu_t*>(next);
                prach_pdu.phys_cell_id       = prach_pars.physCellID;
                prach_pdu.num_prach_ocas     = prach_pars.NumPrachOcas;
                prach_pdu.num_cs             = prach_pars.numCs;
                prach_pdu.num_ra             = prach_pars.numRa;
                prach_pdu.prach_format       = prach_pars.prachFormat;
                prach_pdu.prach_start_symbol = prach_pars.prachStartSymbol;

                pduinfo.pdu_size += sizeof(scf_fapi_prach_pdu_t) - sizeof(scf_fapi_rx_beamforming_t);

                // BEAMFORMING CONFIGS
                uint8_t* bf = reinterpret_cast<uint8_t*>(&prach_pdu.beam_index);
                pduinfo.pdu_size += encode_rx_beamforming(bf, prach_pars.rx_beam_data, fapi_req->channel);

                offset += pduinfo.pdu_size;
                req.num_pdus++;
                prach_num++;

                req.rach_present = 1;
                // Save expected preamble id and TA value range for conformance test
                if((configs->get_conformance_test_params()->conformance_test_enable)&&
                (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
                {
                    update_prach_ocassion(prach_pars);
                }
            }
        }
        else if(fapi_req->channel == channel_type_t::PUSCH)
        {
            int pdu_id = 0;
            for(const pusch_tv_data_t* tv_data : tv.pusch_tv.data)
            {
                auto& pduinfo    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buffer + offset));
                pduinfo.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next    = pduinfo.pdu_config;

                pduinfo.pdu_type = UL_TTI_PDU_TYPE_PUSCH;

                auto& ulsch_pdu = *reinterpret_cast<scf_fapi_pusch_pdu_t*>(next);
                // UL tb_pars
                const struct tb_pars& tb_pars         = tv_data->tbpars;
                ulsch_pdu.handle                      = rand();        //Use random value for handle for testing
                ulsch_pdu.rnti                        = get_rnti_test_mode() == 0 ? tb_pars.nRnti : req.sfn;
                ulsch_pdu.rb_start                    = tb_pars.startPrb;
                ulsch_pdu.rb_size                     = tb_pars.numPrb;
                ulsch_pdu.start_symbol_index          = tb_pars.startSym;
                ulsch_pdu.num_of_symbols              = tb_pars.numSym;
                ulsch_pdu.num_of_layers               = tb_pars.numLayers;
                ulsch_pdu.mcs_table                   = tb_pars.mcsTableIndex;
                ulsch_pdu.mcs_index                   = tb_pars.mcsIndex;
                ulsch_pdu.ul_dmrs_scrambling_id       = tb_pars.dmrsScramId;
                ulsch_pdu.dmrs_config_type            = tb_pars.dmrsType;
                ulsch_pdu.data_scrambling_id          = tb_pars.dataScramId;
                ulsch_pdu.ul_dmrs_sym_pos             = tv_data->dmrsSymLocBmsk;
                ulsch_pdu.dmrs_ports                  = tb_pars.nPortIndex;
                ulsch_pdu.num_dmrs_cdm_groups_no_data = tv_data->numDmrsCdmGrpsNoData;
                ulsch_pdu.pusch_identity              = tv_data->puschIdentity;
                ulsch_pdu.scid                        = tb_pars.nSCID;
                ulsch_pdu.qam_mod_order               = tb_pars.qamModOrder;
                //ulsch_pdu.pusch_identity              = lp->get_cell_configs(cell_id_map[cell_id]).phyCellId;

                ulsch_pdu.bwp.scs                    = tv_data->SubcarrierSpacing;
                ulsch_pdu.bwp.cyclic_prefix          = tv_data->CyclicPrefix;
                ulsch_pdu.target_code_rate           = tv_data->targetCodeRate;
                ulsch_pdu.frequency_hopping          = tv_data->FrequencyHopping;
                ulsch_pdu.tx_direct_current_location = tv_data->txDirectCurrentLocation;
                ulsch_pdu.ul_frequency_shift_7p5_khz = tv_data->uplinkFrequencyShift7p5khz;

                ulsch_pdu.bwp.bwp_start = tv_data->BWPStart;
                ulsch_pdu.bwp.bwp_size  = tv_data->BWPSize;
                ulsch_pdu.pdu_bitmap    = tv_data->pduBitmap;
                ulsch_pdu.transform_precoding = tv_data->TransformPrecoding;
                // ulsch_pdu.qam_mod_order = tv_data->qamModOrder;

                pduinfo.pdu_size += sizeof(scf_fapi_pusch_pdu_t);
                next = ulsch_pdu.payload;

                if(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DATA)
                {
                    ulsch_pdu.handle = handle_id_pusch_data++;
                    auto& pusch_data = *reinterpret_cast<scf_fapi_pusch_data_t*>(next);
                    // HARQ indication
                    pusch_data.rv_index = tb_pars.rv;
                    pusch_data.new_data_indicator = tv_data->newDataIndicator;
                    if(configs->get_conformance_test_params()->conformance_test_enable)
                    {
                        pusch_data.harq_process_id         = tv_data->harqProcessID;
                        uint16_t ue_id = rnti_2_ueid_map[ulsch_pdu.rnti];
                        ul_harq_handle_t & curr_ul_harq_handle = ul_harq_handle[cell_id][ue_id][cell_data[cell_id].harq_process_id];
                        pusch_data.new_data_indicator = curr_ul_harq_handle.pusch_pdu_NDI;
                        pusch_data.rv_index = curr_ul_harq_handle.rv;
                    }
                    else
                    {
                        if(lp->get_config_static_harq_proc_id() > 0)
                        {
                            pusch_data.harq_process_id = tv_data->harqProcessID;
                        }
                        else
                        {
                            pusch_data.harq_process_id         = cell_data[cell_id].harq_process_id;
                            cell_data[cell_id].harq_process_id = (cell_data[cell_id].harq_process_id + 1) % FAPI_MAX_UL_HARQ_ID;
                        }
                    }
                    // Save harq_process_id for validation latter
                    lp->save_harq_pid(cell_id, req.sfn, req.slot, pdu_id++, pusch_data.harq_process_id);

                    pusch_data.tb_size = tv_data->tb_size;
                    pusch_data.num_cb  = 0;

                    int nbytes = (pusch_data.num_cb + 7) / 8;
                    for(int i = 0; i < nbytes; i++)
                    {
                        pusch_data.cb_present_and_position[i] = 0;
                    }
                    pduinfo.pdu_size += sizeof(scf_fapi_pusch_data_t) + nbytes;
                    next = pusch_data.cb_present_and_position + nbytes;
                }

                if(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_UCI)
                {
                    ulsch_pdu.handle = handle_id_pusch_uci ++;
                    // Add puschUci if exists
                    scf_fapi_pusch_uci_t* puschUci  = reinterpret_cast<scf_fapi_pusch_uci_t*>(next);
                    puschUci->harq_ack_bit_length   = tv_data->harqAckBitLength;
                    puschUci->csi_part_1_bit_length = tv_data->csiPart1BitLength;
                    
#ifdef SCF_FAPI_10_04
                    puschUci->flag_csi_part2 = tv_data->flagCsiPart2;
#else
                    puschUci->csi_part_2_bit_length = tv_data->csiPart2BitLength;
#endif
                    puschUci->alpha_scaling         = tv_data->alphaScaling;
                    puschUci->beta_offset_harq_ack  = tv_data->betaOffsetHarqAck;
                    puschUci->beta_offset_csi_1     = tv_data->betaOffsetCsi1;
                    puschUci->beta_offset_csi_2     = tv_data->betaOffsetCsi2;
                    pduinfo.pdu_size += sizeof(scf_fapi_pusch_uci_t);
                    next += sizeof(scf_fapi_pusch_uci_t);
                }

                if(ulsch_pdu.pdu_bitmap & 0x4)
                {
                    // TODO: Add puschPtrs if exists
                }

                // PUSCH DFT-s-OFDM
                if(ulsch_pdu.transform_precoding==0)
                {
                    if(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DFTSOFDM)
                    {
                        scf_fapi_pusch_dftsofdm_t* puschDftsOfdm  = reinterpret_cast<scf_fapi_pusch_dftsofdm_t*>(next);
                        puschDftsOfdm->lowPaprGroupNumber    = tv_data->lowPaprGroupNumber;
                        puschDftsOfdm->lowPaprSequenceNumber = tv_data->lowPaprSequenceNumber;
                        pduinfo.pdu_size += sizeof(scf_fapi_pusch_dftsofdm_t);
                        next += sizeof(scf_fapi_pusch_dftsofdm_t);
                    }
                }

                // BEAMFORMING CONFIGS
                int bf_size = encode_rx_beamforming(next, tv_data->rx_beam_data, fapi_req->channel);
                pduinfo.pdu_size += bf_size;
                next += bf_size;

            #ifdef SCF_FAPI_10_04
                // PUSCH Maintenance Parameters
                scf_fapi_pusch_maintenance_t* puschMaintenance  = reinterpret_cast<scf_fapi_pusch_maintenance_t*>(next);
                if(ulsch_pdu.transform_precoding==0)
                {
                    if(!(ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_DFTSOFDM))
                    {
                        puschMaintenance->groupOrSequenceHopping = tv_data->groupOrSequenceHopping;
                    }
                }
                pduinfo.pdu_size += sizeof(scf_fapi_pusch_maintenance_t);
                next += sizeof(scf_fapi_pusch_maintenance_t);

                // PUSCH UCI 
                if (ulsch_pdu.pdu_bitmap & PUSCH_BITMAP_UCI && !!tv_data->flagCsiPart2 && tv_data->numPart2s > 0) {
                    scf_uci_csip2_info_t* uci_csip2_info  = reinterpret_cast<scf_uci_csip2_info_t*>(next);
                    uci_csip2_info->numPart2s = tv_data->numPart2s;
                    pduinfo.pdu_size += sizeof(scf_uci_csip2_info_t);
                    next += sizeof(scf_uci_csip2_info_t);
                    size_t offset = 0;
                    for (uint16_t i = 0; i < tv_data->numPart2s; i++) {
                        auto csip2_part = reinterpret_cast<scf_uci_csip2_part_t*>(next + offset);
                        auto& tv_csip2_part = tv_data->csip2_v3_parts[i];
                        csip2_part->priority = tv_csip2_part.priority;
                        csip2_part->numPart1Params = tv_csip2_part.numPart1Params;
                        offset += sizeof(scf_uci_csip2_part_t) ;
                        auto csip2_part_offset = reinterpret_cast<scf_uci_csip2_part_param_offset_t*>(next + offset);
                        auto csip2_part_size_offset = reinterpret_cast<scf_uci_csip2_part_param_size_t*>(next + offset + csip2_part->numPart1Params *  sizeof(uint16_t));

                        for ( uint16_t j = 0; j < csip2_part->numPart1Params; j++) {
                            csip2_part_offset->paramOffsets[j] = tv_csip2_part.paramOffsets[j];
                            csip2_part_size_offset->paramSizes[j] = tv_csip2_part.paramSizes[j];
                            auto val =  reinterpret_cast<uint16_t>(csip2_part_offset->paramOffsets[j]);
                            auto valsize =  reinterpret_cast<uint8_t>(csip2_part_size_offset->paramSizes[j]);
                            // NVLOGD_FMT(TAG, "FAPI: paramOffsets [{}] paramSizes [{}] TV: paramOffsets [{}] paramSizes [{}] part2SizeMapScope {}", val, valsize, tv_csip2_part.paramOffsets[j], tv_csip2_part.paramSizes[j], tv_csip2_part.part2SizeMapScope);
                        }
                        offset +=  csip2_part->numPart1Params * ((sizeof(uint16_t) + sizeof(uint8_t)));
                        auto csip2_part_scope = reinterpret_cast<scf_uci_csip2_part_scope_t*>(&uci_csip2_info->payload[0] + offset);
                        csip2_part_scope->part2SizeMapIndex = tv_csip2_part.part2SizeMapIndex;
                        NVLOGD_FMT(TAG, "part2SizeMapIndex {}", static_cast<uint16_t>(csip2_part_scope->part2SizeMapIndex));
                        csip2_part_scope->part2SizeMapScope = tv_csip2_part.part2SizeMapScope;
                        offset +=sizeof(scf_uci_csip2_part_scope_t);
                    }
                    pduinfo.pdu_size += offset;
                    next += offset;
                }
                cell_configs_t& cell_configs = lp->get_cell_configs(cell_id_map[cell_id]);
                // PUSCH Extension
                if (cell_configs.enableWeightedAverageCfo == 1) {
                    scf_fapi_pusch_extension_t* puschExtension = reinterpret_cast<scf_fapi_pusch_extension_t*>(next);
                    // for weighted average CFO estimation
                    puschExtension->fo_forget_coeff = static_cast<uint8_t>(tv_data->foForgetCoeff * 100.0f);
                    puschExtension->n_iterations = tv_data->nIterations;
                    puschExtension->ldpc_early_termination = tv_data->ldpcEarlyTermination;
                    pduinfo.pdu_size += sizeof(scf_fapi_pusch_extension_t);
                    next += sizeof(scf_fapi_pusch_extension_t);
                }
            #endif
                for(int i = 0; i < req.ngroup; i++)
                {
                    scf_fapi_ue_group_t* gp = reinterpret_cast<scf_fapi_ue_group_t*>(next);
                    gp->num_ue              = 1; // TODO: set the right UE number in each group
                    for(int j = 0; j < gp->num_ue; j++)
                    {
                        gp->pdu_index[j] = j; // TODO: set the right pdu index if UE group exists
                    }
                    pduinfo.pdu_size += sizeof(scf_fapi_ue_group_t) + gp->num_ue;
                    next = gp->pdu_index + gp->num_ue;
                }

                offset += pduinfo.pdu_size;
                req.num_pdus++;
                req.num_ulsch++;
                pusch_num++;

                // Log print the TB size and offset
                NVLOGD_FMT(TAG, "{}: PUSCH handle={} pdu_bitmap={} offset={} rnti={} hpid={}", __FUNCTION__, static_cast<unsigned>(ulsch_pdu.handle), static_cast<unsigned short>(ulsch_pdu.pdu_bitmap), offset, static_cast<unsigned short>(ulsch_pdu.rnti), cell_data[cell_id].harq_process_id);
            }
        }
        else if(fapi_req->channel == channel_type_t::PUCCH)
        {
            for(auto& tv_data : tv.pucch_tv.data)
            {
                auto& pduinfo    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buffer + offset));
                pduinfo.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next    = pduinfo.pdu_config;

                pduinfo.pdu_type = UL_TTI_PDU_TYPE_PUCCH;

                scf_fapi_pucch_pdu_t& pucch_pdu = *reinterpret_cast<scf_fapi_pucch_pdu_t*>(pduinfo.pdu_config);

                pucch_pdu.rnti              = tv_data.RNTI;
                pucch_pdu.handle            = handle_id_pucch ++;
                pucch_pdu.bwp.bwp_start     = tv_data.BWPStart;
                pucch_pdu.bwp.bwp_size      = tv_data.BWPSize;
                pucch_pdu.bwp.scs           = tv_data.SubcarrierSpacing;
                pucch_pdu.bwp.cyclic_prefix = tv_data.CyclicPrefix;

                pucch_pdu.format_type             = tv_data.FormatType;
                pucch_pdu.multi_slot_tx_indicator = tv_data.multiSlotTxIndicator;
                pucch_pdu.pi_2_bpsk               = tv_data.pi2Bpsk;

                pucch_pdu.prb_start          = tv_data.prbStart;
                pucch_pdu.prb_size           = tv_data.prbSize;
                pucch_pdu.start_symbol_index = tv_data.StartSymbolIndex;
                pucch_pdu.num_of_symbols     = tv_data.NrOfSymbols;

                pucch_pdu.freq_hop_flag        = tv_data.freqHopFlag;
                pucch_pdu.second_hop_prb       = tv_data.secondHopPRB;
                pucch_pdu.group_hop_flag       = tv_data.groupHopFlag;
                pucch_pdu.seq_hop_flag         = tv_data.sequenceHopFlag;
                pucch_pdu.hopping_id           = tv_data.hoppingId;
                pucch_pdu.initial_cyclic_shift = tv_data.InitialCyclicShift;

                pucch_pdu.data_scrambling_id  = tv_data.dataScramblingId;
                pucch_pdu.time_domain_occ_idx = tv_data.TimeDomainOccIdx;
                pucch_pdu.pre_dft_occ_idx     = tv_data.PreDftOccIdx;
                pucch_pdu.pre_dft_occ_len     = tv_data.PreDftOccLen;

                pucch_pdu.add_dmrs_flag      = tv_data.AddDmrsFlag;
                pucch_pdu.dmrs_scrambling_id = tv_data.DmrsScramblingId;
                pucch_pdu.dmrs_cyclic_shift  = tv_data.DMRScyclicshift;

                pucch_pdu.sr_flag            = pucch_pdu.format_type < 2 ? tv_data.SRFlag : tv_data.bitLenSr;
                pucch_pdu.bit_len_harq       = tv_data.BitLenHarq;
                pucch_pdu.bit_len_csi_part_1 = tv_data.BitLenCsiPart1;
                pucch_pdu.bit_len_csi_part_2 = tv_data.BitLenCsiPart2;
                //pucch_pdu.pucch_maintenance_params.max_code_rate = tv_data.maxCodeRate;
                pduinfo.pdu_size += sizeof(scf_fapi_pucch_pdu_t);
                next = pucch_pdu.payload;

                // BEAMFORMING CONFIGS
                int bf_size = encode_rx_beamforming(next, tv_data.rx_beam_data, fapi_req->channel);
                pduinfo.pdu_size += bf_size;
                next += bf_size;

                offset += pduinfo.pdu_size;
                req.num_pdus++;
                req.num_ulcch++;
                pucch_num++;

#if 0 // For temporary hard-code test only
                pucch_pdu.format_type = 1; // Select PUCCH format 0/1. Format 2/3/4 are not supported in cuphy header yet.
                pucch_pdu.sr_flag = 1; // Include SR or not
                pucch_pdu.bit_len_harq = 1; // Include HARQ or not
                pucch_pdu.bit_len_csi_part_1 = 10; // Include CSI Part 1 or not. Not supported in l2adpater yet.
                pucch_pdu.bit_len_csi_part_2 = 10; // Include CSI Part 2 or not. Not supported in l2adpater yet.
#endif
                if((configs->get_conformance_test_params()->conformance_test_enable)&&
                (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
                    update_pucch_ocassion(pucch_pdu);

                NVLOGD_FMT(TAG, "{}: PUCCH handle={} format={} sr_flag={} bit_len_harq={} CSI: part1={} part2={} offset={}", __FUNCTION__, static_cast<unsigned>(pucch_pdu.handle), static_cast<unsigned short>(pucch_pdu.format_type),
                        static_cast<unsigned short>(pucch_pdu.sr_flag), static_cast<unsigned short>(pucch_pdu.bit_len_harq), static_cast<unsigned short>(pucch_pdu.bit_len_csi_part_1), static_cast<unsigned short>(pucch_pdu.bit_len_csi_part_1), offset);
            }
        }
        else if(fapi_req->channel == channel_type_t::SRS)
        {
            NVLOGD_FMT(TAG, "{}: Inside SRS ", __FUNCTION__);
            static uint8_t ctr = 0;
            static uint8_t isInitSrsChestBufIdx[MAX_CELLS_PER_SLOT] = {false};
            if (!isInitSrsChestBufIdx[cell_id])
            {
                srsChestBufQueueMutex[cell_id].lock();
                bool isMimoEnabled = (lp->get_mmimo_enabled() != 0 || (lp->get_mmimo_static_dynamic_enabled() != 0));
                initSrsChestBufIdxQueue(cell_id, isMimoEnabled);
                isInitSrsChestBufIdx[cell_id] = true;
                srsChestBufQueueMutex[cell_id].unlock();
            }
            for(const srs_tv_data_t& tv_data : tv.srs_tv.data)
            {
                auto& pduinfo    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buffer + offset));
                pduinfo.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
                uint8_t* next    = pduinfo.pdu_config;

                pduinfo.pdu_type = UL_TTI_PDU_TYPE_SRS;

                scf_fapi_srs_pdu_t& srs_pdu = *reinterpret_cast<scf_fapi_srs_pdu_t*>(next);
                srs_pdu.rnti = tv_data.RNTI;
                srs_pdu.handle = tv_data.srsPduIdx;
                NVLOGD_FMT(TAG, "SFN {}.{} >>> 0 Test Mac Before srs_pdu.handle={}", static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), static_cast<unsigned int>(srs_pdu.handle));
#ifdef SCF_FAPI_10_04
                srsChestBufQueueMutex[cell_id].lock();
                auto srsChestBuffIdx = 0;
                if (configs->get_enable_srs_l1_limit_testing() == 1) {
                    srs_pdu.handle |= (static_cast<uint32_t>(tv_data.srsChestBufferIndex) << 8);
                }
                else {
                    srsChestBuffIdx = getSrsChestBufIdx(cell_id);
                    mapOfSrsReqToIndRntiToSrsChestBufIdxMutex[cell_id].lock();
                    if(srsChestBuffIdx != 0xFFFF)
                    {
                        srs_pdu.handle |= (static_cast<uint32_t>(srsChestBuffIdx) << 8);
                        setMapOfSrsReqToIndRntiToSrsChestBufIdx(cell_id, srs_pdu.rnti, srsChestBuffIdx);
                    }
                    else
                    {
                        srs_pdu.handle |= (static_cast<uint32_t>(tv_data.srsChestBufferIndex) << 8);
                    }
                }
                NVLOGD_FMT(TAG, "cell_id={} SFN={} SLOT={} rnti={} After srsChestBufferIndex={} handle={} srsChestBuffIdx={}", 
                           cell_id, static_cast<unsigned>(req.sfn), 
                           static_cast<unsigned>(req.slot), 
                           static_cast<unsigned int>(tv_data.RNTI), 
                           static_cast<unsigned int>(static_cast<uint16_t>((srs_pdu.handle >> 8) & 0xFFFF)),
                           static_cast<unsigned int>(srs_pdu.handle),
                           static_cast<unsigned int>(srsChestBuffIdx));
                mapOfSrsReqToIndRntiToSrsChestBufIdxMutex[cell_id].unlock();
                srsChestBufQueueMutex[cell_id].unlock();
#endif
                srs_pdu.bwp.bwp_start     = tv_data.BWPStart;
                srs_pdu.bwp.bwp_size      = tv_data.BWPSize;
                srs_pdu.bwp.scs           = lp->get_cell_configs(cell_id_map[cell_id]).mu; //tv_data.SubcarrierSpacing;
                srs_pdu.bwp.cyclic_prefix = 0; //tv_data.CyclicPrefix;
                srs_pdu.num_ant_ports = tv_data.numAntPorts;
                srs_pdu.num_symbols = tv_data.numSymbols;
                srs_pdu.num_repetitions = tv_data.numRepetitions;
                srs_pdu.time_start_position = tv_data.timeStartPosition;
                srs_pdu.config_index = tv_data.configIndex;
                srs_pdu.sequenceId = tv_data.sequenceId;
                srs_pdu.bandwidth_index = tv_data.bandwidthIndex;
                srs_pdu.comb_size = tv_data.combSize;
                srs_pdu.comb_offset = tv_data.combOffset;
                srs_pdu.cyclic_shift = tv_data.cyclicShift;
                srs_pdu.frequency_position = tv_data.frequencyPosition;
                srs_pdu.frequency_shift = tv_data.frequencyShift;
                srs_pdu.frequency_hopping = tv_data.frequencyHopping;
                srs_pdu.group_or_sequence_hopping = tv_data.groupOrSequenceHopping;
                srs_pdu.resource_type = tv_data.resourceType;
                srs_pdu.t_srs = tv_data.Tsrs;
                srs_pdu.t_offset = tv_data.Toffset;

                pduinfo.pdu_size += sizeof(scf_fapi_srs_pdu_t);
#if 0
                // BEAMFORMING CONFIGS
                int bf_size = encode_srs_rx_beamforming(next, tv_data.rx_beam_data);
                pduinfo.pdu_size += bf_size;
                next += bf_size;
#endif
#ifdef SCF_FAPI_10_04_SRS
                next = srs_pdu.payload;
                scs_fapi_v4_srs_params_t& srs_v4_parms = *reinterpret_cast<scs_fapi_v4_srs_params_t*>(next);

                srs_v4_parms.srs_bw_size = 272;
                memset (srs_v4_parms.srs_bw_sq_info, 0, sizeof(srs_v4_parms.srs_bw_sq_info));

                srs_v4_parms.usage = tv_data.fapi_v4_params.usage;

    #if 1
                //srs_v4_parms.usage = 1 << (tv_data.fapi_v4_params.usage);
                // FIXME:
                //srs_v4_parms.report_type = tv_data.fapi_v4_params.usage;
                srs_v4_parms.report_type = 1; /* only report type 1 is supported now */

    #else
                if (srs_v4_parms.usage & SRS_REPORT_FOR_BEAM_MANAGEMENT)
                {
                    srs_v4_parms.report_type[SRS_USAGE_FOR_BEAM_MANAGEMENT] = 1; /* only report type 1 is supported now */
                }
                if (srs_v4_parms.usage & SRS_REPORT_FOR_CODEBOOK)
                {
                    srs_v4_parms.report_type[SRS_USAGE_FOR_CODEBOOK] = 1; /* only report type 1 is supported now */
                }
                if (srs_v4_parms.usage & SRS_REPORT_FOR_NON_CODEBOOK)
                {
                    srs_v4_parms.report_type[SRS_USAGE_FOR_NON_CODEBOOK] = 1; /* only report type 1 is supported now */
                }
                if (srs_v4_parms.usage & SRS_REPORT_FOR_ANTENNA_SWITCHING)
                {
                    srs_v4_parms.report_type[SRS_USAGE_FOR_ANTENNA_SWITCHING] = 1; /* only report type 1 is supported now */
                }
    #endif
                srs_v4_parms.sing_val_rep = 0xFF;
    #if 0
                srs_v4_parms.iq_repr = INDEX_IQ_REPR_FP32_COMPLEX;
                srs_v4_parms.prg_size = 137;
                srs_v4_parms.num_of_tot_ue_ant = 2;
                srs_v4_parms.ue_ant_in_this_srs_res_set = 0;
                srs_v4_parms.samp_ue_ant = 3;
    #else
                srs_v4_parms.iq_repr = INDEX_IQ_REPR_FP32_COMPLEX;
                srs_v4_parms.prg_size = tv_data.prgSize;
                srs_v4_parms.num_of_tot_ue_ant = tv_data.fapi_v4_params.numTotalUeAntennas;
                srs_v4_parms.ue_ant_in_this_srs_res_set = tv_data.fapi_v4_params.ueAntennasInThisSrsResourceSet;
                srs_v4_parms.samp_ue_ant = tv_data.fapi_v4_params.sampledUeAntennas;
    #endif
                srs_v4_parms.rep_scope = 0;
                srs_v4_parms.num_ul_spat_strm_ports = 0;

                pduinfo.pdu_size += ((sizeof(scs_fapi_v4_srs_params_t)) + srs_v4_parms.num_ul_spat_strm_ports);
                next = srs_v4_parms.ul_spat_strm_ports;
#endif
                offset += pduinfo.pdu_size;
                req.num_pdus++;
                srs_num++;
            }
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_NO_SUPPORT_EVENT, "{}: channel is not supported yet: {}", __FUNCTION__, static_cast<unsigned>(fapi_req->channel));
        }
    }

    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} UL_TTI.req nPDU: PRACH={} PUSCH={} PUCCH={} SRS={}",
            static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, prach_num, pusch_num, pucch_num, srs_num);
    return offset;
}
int insertOrUpdate(std::map<uint16_t, uint16_t>& myMap, int key) {
    uint16_t mapSizeBeforeInsertion = myMap.size(); // Store size of map before insertion

    // Check if key exists in map
    if (myMap.count(key)) {
        // Key exists in map, return associated value
        return myMap[key];
    } else {
        // Key does not exist in map, create new entry
        myMap[key] = mapSizeBeforeInsertion;
        return mapSizeBeforeInsertion;
    }
}
template <typename T, typename std::enable_if<std::is_unsigned_v<T>, bool>::type = true>
T reverse_bits(T num, uint8_t bitlength)
{
    uint8_t bit_count = bitlength -1 ;
    T result = 0;
    while(num)
    {
        result |= (num & 1) << bit_count;
        num >>= 1;
        bit_count--;
    }

    return result;
}
int scf_fapi_handler::build_ul_dci_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_dci_t& req)
{
    uint8_t* buf    = reinterpret_cast<uint8_t*>(req.payload);
    size_t   offset = 0;

    req.num_pdus = 0;
    for(auto& fapi_req : fapi_reqs)
    {
        const test_vector_t& tv = *fapi_req->tv_data;
        for(auto& coreset : tv.pdcch_tv.coreset)
        {
            auto& uldci_pdu    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buf + offset));
            uldci_pdu.pdu_type = 0; //PDCCH PDU
            uldci_pdu.pdu_size = sizeof(scf_fapi_generic_pdu_info_t);

            // PDCCH PDU CONFIGS
            auto& pdcch_pdu             = *(reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&uldci_pdu.pdu_config[0]));
            pdcch_pdu.bwp.bwp_size      = coreset.BWPSize;  // tb_pars.numPrb;
            pdcch_pdu.bwp.bwp_start     = coreset.BWPStart; // tb_pars.startPrb;
            pdcch_pdu.bwp.scs           = coreset.SubcarrierSpacing;
            pdcch_pdu.bwp.cyclic_prefix = coreset.CyclicPrefix;
            pdcch_pdu.start_sym_index   = coreset.StartSymbolIndex;
            pdcch_pdu.duration_sym      = coreset.DurationSymbols;

            for(int i = 0; i < sizeof(pdcch_pdu.freq_domain_resource); ++i)
            {
                pdcch_pdu.freq_domain_resource[i] = ((uint8_t*)&coreset.FreqDomainResource)[7 - i];
            }

            pdcch_pdu.cce_reg_mapping_type = coreset.CceRegMappingType;
            pdcch_pdu.reg_bundle_size      = coreset.RegBundleSize;
            pdcch_pdu.interleaver_size     = coreset.InterleaverSize;
            pdcch_pdu.shift_index          = coreset.ShiftIndex;
            pdcch_pdu.precoder_granularity = coreset.precoderGranularity;
            pdcch_pdu.coreset_type         = coreset.CoreSetType;

            uldci_pdu.pdu_size += sizeof(scf_fapi_pdcch_pdu_t);

            // DCI CONFIGS
            // Num DCIs
            pdcch_pdu.num_dl_dci   = coreset.numDlDci;
            std::size_t dci_offset = 0;
            uint8_t* dci_params = reinterpret_cast<uint8_t*>(&pdcch_pdu.dl_dci[0]);
            for(uint16_t i = 0; i < coreset.numDlDci; i++)
            {
                auto& dci                     = coreset.dciList[i];
                auto& dci_payload             = *reinterpret_cast<scf_fapi_dl_dci_t*>(dci_params + dci_offset);
                dci_payload.rnti              = dci.RNTI;
                dci_payload.scrambling_id     = dci.ScramblingId;
                dci_payload.scrambling_rnti   = dci.ScramblingRNTI;
                dci_payload.cce_index         = dci.CceIndex;
                dci_payload.aggregation_level = dci.AggregationLevel;

                uldci_pdu.pdu_size += sizeof(scf_fapi_dl_dci_t);
                dci_offset += sizeof(scf_fapi_dl_dci_t);
                // BEAMFORMING CONFIGS
                auto& bf             = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&dci_payload.payload[0]);
                bf.num_prgs          = 0;
                bf.dig_bf_interfaces = 0;
                auto bf_size         = encode_tx_beamforming(dci_payload.payload, dci.tx_beam_data, fapi_req->channel);

                uldci_pdu.pdu_size += bf_size;
                dci_offset += bf_size;

                // TX POWER INFO CONFIGS
                auto& pwr_info                   = *reinterpret_cast<scf_fapi_pdcch_tx_power_info_t*>(&dci_payload.payload[bf_size]);
                pwr_info.beta_pdcch_1_0          = dci.beta_PDCCH_1_0;
#ifdef SCF_FAPI_10_04
                pwr_info.power_control_offset_ss_profile_nr = dci.powerControlOffsetSSProfileNR;
#else
                pwr_info.power_control_offset_ss = dci.powerControlOffsetSS;
#endif
                auto tx_power_info_size          = sizeof(scf_fapi_pdcch_tx_power_info_t);

                uldci_pdu.pdu_size += tx_power_info_size;
                dci_offset += tx_power_info_size;

                // DCI CONFIGS END
                auto& dci_end = *reinterpret_cast<scf_fapi_pdcch_dci_payload_t*>(&dci_payload.payload[bf_size + tx_power_info_size]); // Only 1 for now

                int nbytes                = (dci.PayloadSizeBits + 7) / 8;
                dci_end.payload_size_bits = dci.PayloadSizeBits;
                std::copy(dci.Payload, dci.Payload + nbytes, dci_end.payload);
                if(configs->get_conformance_test_params()->conformance_test_enable)
                {

                    uint8_t harq_byte_pos = (ul_dci_freq_domain_bits + 14) >> 3;
                    uint8_t NDI_byte_pos = (ul_dci_freq_domain_bits + 11) >> 3;
                    uint8_t RV_byte_pos = (ul_dci_freq_domain_bits + 12) >> 3;

                    uint8_t harq_bit_pos  = (ul_dci_freq_domain_bits + 14) & 0x7;
                    uint8_t NDI_bit_pos  = (ul_dci_freq_domain_bits + 11) & 0x7;
                    uint8_t RV_bit_pos  = (ul_dci_freq_domain_bits + 12) & 0x7;

                    uint8_t harq_id_bit_size = 4;
                    uint16_t harq_id = 0;
                    if(harq_bit_pos + harq_id_bit_size > 8)
                    {
                        uint8_t harq_bits_2 = harq_bit_pos + harq_id_bit_size - 8;
                        uint8_t harq_bits_1 = harq_id_bit_size - harq_bits_2;
                        uint8_t byte1 = reverse_bits(dci.Payload[harq_byte_pos], 8);
                        uint8_t byte2 = reverse_bits(dci.Payload[harq_byte_pos + 1], 8);
                        harq_id = static_cast<unsigned short> ((byte1 & (((1 << harq_bits_1) - 1 )<< harq_bit_pos)) >> harq_bit_pos);
                        harq_id |= (static_cast<unsigned short> (byte2 & ((1 << harq_bits_2) -1)) << harq_bits_1);
                    }
                    else
                    {
                        uint8_t byte1 = reverse_bits(dci.Payload[harq_byte_pos], 8);
                        uint16_t harq_id = static_cast<unsigned short> ((byte1 & (0xf << harq_bit_pos)) >> harq_bit_pos) ;
                    }
                    harq_id  = reverse_bits(harq_id, harq_id_bit_size);
                    int ue_id = insertOrUpdate(rnti_2_ueid_map, dci_payload.rnti);
                    ul_harq_handle_t & curr_ul_harq_handle = ul_harq_handle[cell_id][ue_id][harq_id];
                    if(RV_bit_pos == 7)
                    {
                        if(curr_ul_harq_handle.rv & 1)
                        {
                            dci_end.payload[RV_byte_pos + 1] |=   1<< 7;
                        }
                        else
                        {
                            dci_end.payload[RV_byte_pos + 1] &= ~(1<< 7);
                        }
                        if(curr_ul_harq_handle.rv & 2)
                        {
                            dci_end.payload[RV_byte_pos] |=   (1 << (7 - RV_bit_pos));
                        }
                        else
                        {
                            dci_end.payload[RV_byte_pos -1 ] &=  ~(1 << (7 - RV_bit_pos)) ;
                        }
                    }
                    else
                    {
                        if(curr_ul_harq_handle.rv & 1)
                        {
                            dci_end.payload[RV_byte_pos] |=   (1 << ( 7 - RV_bit_pos - 1));
                        }
                        else
                        {
                            dci_end.payload[RV_byte_pos] &=   ~(1 << ( 7 - RV_bit_pos - 1));
                        }
                        if(curr_ul_harq_handle.rv & 2)
                        {
                            dci_end.payload[RV_byte_pos] |=   (1 << (7 - RV_bit_pos) );
                        }
                        else
                        {
                            dci_end.payload[RV_byte_pos] &=  ~(1 << (7 - RV_bit_pos));
                        }
                    }
                    if(curr_ul_harq_handle.ul_dci_NDI)
                    {
                        dci_end.payload[NDI_byte_pos] |= 1 << (7 - NDI_bit_pos);
                    }
                    else
                    {
                        dci_end.payload[NDI_byte_pos] &= ~(1 << (7 - NDI_bit_pos));
                    }
                }

                uldci_pdu.pdu_size += sizeof(scf_fapi_pdcch_dci_payload_t) + nbytes;
                dci_offset += sizeof(scf_fapi_pdcch_dci_payload_t) + nbytes;
                offset += uldci_pdu.pdu_size;
                NVLOGD_FMT(TAG, "{}: nDCI={} nTotalBits={} offset={} dci_offset={}", __FUNCTION__, static_cast<unsigned>(req.num_pdus),static_cast<unsigned short>(dci_end.payload_size_bits), offset, dci_offset);
            }
            req.num_pdus++;
        }
    }
    NVLOGI_FMT(TAG, "SFN {}.{} BUILD: cell_id={} UL_DCI.req nPDU={}",
            static_cast<unsigned>(req.sfn), static_cast<unsigned>(req.slot), cell_id, req.num_pdus);
    return offset;
}

int scf_fapi_handler::build_dl_bfw_cvi_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_bfw_cvi_request_t* req)
{
#if 1
    size_t   offset = 0;
    req->npdus = 0;
    for(fapi_req_t* fapi_req : fapi_reqs)
    {
        const test_vector_t& tv = *fapi_req->tv_data;

        if(fapi_req->channel == channel_type_t::BFW_DL || fapi_req->channel == channel_type_t::BFW_UL)
        {
           //bfw_cv_data_t data = tv.bfw_tv.data[0];
           NVLOGD_FMT(TAG, "req->npdus={}, offset={}", req->npdus, offset);
            uint8_t  *dl_bfw_config_pdu_ptr = reinterpret_cast<uint8_t*> (&req->config_pdu[0]);

            for(auto& data : tv.bfw_tv.data)
            {
                scf_fapi_dl_bfw_group_config_t  *dl_bfw_config_pdu = reinterpret_cast<scf_fapi_dl_bfw_group_config_t*>(dl_bfw_config_pdu_ptr + offset);
                dl_bfw_config_pdu->pdu_size = sizeof(dl_bfw_config_pdu->pdu_size);
                dl_bfw_config_pdu->dl_bfw_cvi_config.nUes = data.nUes;
                dl_bfw_config_pdu->dl_bfw_cvi_config.rb_start = data.rbStart;
                dl_bfw_config_pdu->dl_bfw_cvi_config.rb_size = data.rbSize;
                dl_bfw_config_pdu->dl_bfw_cvi_config.num_prgs = data.numPRGs;
                dl_bfw_config_pdu->dl_bfw_cvi_config.prg_size = data.prgSize;
                dl_bfw_config_pdu->pdu_size += sizeof(scf_dl_bfw_config_t);
                uint8_t* ptr = &(dl_bfw_config_pdu->dl_bfw_cvi_config.payload[0]);
                for(uint32_t i = 0; i < data.nUes; i++)
                {
                    scf_dl_bfw_config_start_t* config = reinterpret_cast<scf_dl_bfw_config_start_t*>(ptr);
                    config->rnti = data.ue_grp_data[i].RNTI;
                    config->handle = 0;
                    //config->srsChestBufferIndex = data.ue_grp_data[i].srsChestBufferIndex;
                    auto bufferIndex = getSrsChestBufferIndexFromMapOfRnti(cell_id, data.ue_grp_data[i].RNTI);
                    if (bufferIndex != 0xFFFF)
                    {
                        config->handle |= (bufferIndex << 8);
                        mapBfwSrsChestBufIdxListMutex[cell_id].lock();
                        removeIdxInMapBfwSrsChestBufIdxList(cell_id, data.ue_grp_data[i].srsChestBufferIndex);
                        mapBfwSrsChestBufIdxListMutex[cell_id].unlock();
                    }
                    else
                    {
                        config->handle |= (data.ue_grp_data[i].srsChestBufferIndex << 8);
                        mapBfwSrsChestBufIdxListMutex[cell_id].lock();
                        insertIdxInMapBfwSrsChestBufIdxList(cell_id, data.ue_grp_data[i].srsChestBufferIndex);
                        mapBfwSrsChestBufIdxListMutex[cell_id].unlock();
                    }
                    config->pduIndex = data.ue_grp_data[i].pduIndex;
                    config->gnb_ant_index_start = data.ue_grp_data[i].gNbAntIdxStart;
                    config->gnb_ant_index_end = data.ue_grp_data[i].gNbAntIdxEnd;
                    config->num_ue_ants = data.ue_grp_data[i].numOfUeAnt;
                    dl_bfw_config_pdu->pdu_size += sizeof(scf_dl_bfw_config_start_t) + config->num_ue_ants;
                    uint8_t* ue_ant = &config->payload[0];
                    for(uint32_t j=0; j < config->num_ue_ants; j++)
                    {
                        ue_ant[j] = data.ue_grp_data[i].ueAntIndexes[j];
                        NVLOGD_FMT(TAG, "{}: nUe={} ue_ant[{}]={}\n",__func__,i,j,ue_ant[j]);
                    }
                    ptr += sizeof(scf_dl_bfw_config_start_t) + config->num_ue_ants;
                    NVLOGD_FMT(TAG, "{}: rnti={} pduIdx={} gNbAntStartIdx={} gNbAntEndIdx={} numUeAnt={}\n",
                        __func__,static_cast<uint16_t>(config->rnti),static_cast<uint16_t>(config->pduIndex),
                        static_cast<uint8_t>(config->gnb_ant_index_start),static_cast<uint8_t>(config->gnb_ant_index_end),
                        static_cast<uint8_t>(config->num_ue_ants));
                }
                offset += dl_bfw_config_pdu->pdu_size;
                //dl_bfw_config_pdu_ptr = static_cast<scf_fapi_dl_bfw_group_config_t *> (ptr);
                //offset += sizeof(dl_bfw_config_pdu->pdu_size);
                req->npdus++;
            }

        }
    }
    NVLOGD_FMT(TAG, "req->npdus={}, offset={}", req->npdus, offset);
    return offset;
#else
    for(fapi_req_t* fapi_req : fapi_reqs)
    {
        const test_vector_t& tv = *fapi_req->tv_data;

        if(fapi_req->channel == channel_type_t::BFW_DL || fapi_req->channel == channel_type_t::BFW_UL)
        {
            req->npdus     = 1;
            bfw_cv_data_t data = tv.bfw_tv.data[0];
            req->config_pdu[0].dl_bfw_cvi_config.nUes = data.nUes;
            req->config_pdu[0].dl_bfw_cvi_config.rb_start = data.rbStart;
            req->config_pdu[0].dl_bfw_cvi_config.rb_size = data.rbSize;
            req->config_pdu[0].dl_bfw_cvi_config.num_prgs = data.numPRGs;
            req->config_pdu[0].dl_bfw_cvi_config.prg_size = data.prgSize;
            req->config_pdu[0].pdu_size = 9;
            uint8_t* ptr = &(req->config_pdu[0].dl_bfw_cvi_config.payload[0]);
            for(uint32_t i = 0; i < data.nUes; i++)
            {
                 scf_dl_bfw_config_start_t* config = reinterpret_cast<scf_dl_bfw_config_start_t*>(ptr);
                 config->rnti = data.ue_grp_data[i].RNTI;
                 config->pduIndex = data.ue_grp_data[i].pduIndex;
                 config->gnb_ant_index_start = data.ue_grp_data[i].gNbAntIdxStart;
                 config->gnb_ant_index_end = data.ue_grp_data[i].gNbAntIdxEnd;
                 config->num_ue_ants = data.ue_grp_data[i].numOfUeAnt;
                 req->config_pdu[0].pdu_size += 7 + config->num_ue_ants;
                 uint8_t* ue_ant = &config->payload[0];
                 for(uint32_t j=0; j < config->num_ue_ants; j++)
                 {
                     ue_ant[j] = data.ue_grp_data[i].ueAntIndexes[j];
                     NVLOGD_FMT(TAG, "{}: nUe={} ue_ant[{}]={}\n",__func__,i,j,ue_ant[j]);
                 }
                 ptr += sizeof(scf_dl_bfw_config_start_t) + config->num_ue_ants;
                 NVLOGD_FMT(TAG, "{}: rnti={} pduIdx={} gNbAntStartIdx={} gNbAntEndIdx={} numUeAnt={}\n",
                     __func__,static_cast<uint16_t>(config->rnti),static_cast<uint16_t>(config->pduIndex),
                     static_cast<uint8_t>(config->gnb_ant_index_start),static_cast<uint8_t>(config->gnb_ant_index_end),
                     static_cast<uint8_t>(config->num_ue_ants));
            }
        }
    }
    return req->config_pdu[0].pdu_size + 2;
#endif
}

// Add a TLV and return next TLV address. Align to 4 bytes, refer to Section 3.3.1.4 in SCF 222
static scf_fapi_tl_t* add_tlv(scf_fapi_config_request_msg_t& req, scf_fapi_tl_t* tlv, uint16_t tag, uint16_t len, uint32_t val)
{
    req.msg_body.num_tlvs++;
    tlv->tag    = tag;
    tlv->length = len;
    memcpy(tlv->val, &val, sizeof(val));
    uint8_t* next = tlv->val + ((len + 3) / 4) * 4;
    return reinterpret_cast<scf_fapi_tl_t*>(next);
}

// Add a TLV and return next TLV address. Align to 4 bytes, refer to Section 3.3.1.4 in SCF 222
static scf_fapi_tl_t* add_tlv(scf_fapi_config_request_msg_t& req, scf_fapi_tl_t* tlv, uint16_t tag, uint16_t len, uint8_t* val)
{
    req.msg_body.num_tlvs++;
    tlv->tag    = tag;
    tlv->length = len;
    memcpy(tlv->val, val, len);
    uint8_t* next = tlv->val + ((len + 3) / 4) * 4;
    return reinterpret_cast<scf_fapi_tl_t*>(next);
}

int scf_fapi_handler::send_config_request(int cell_id)
{
    if(!cell_id_sanity_check(cell_id))
    {
        return -1;
    }

    if (configs->cell_config_wait >= 0)
    {
        int32_t cas_expected = -1;
        bool cas_result = current_config_cell_id.compare_exchange_strong(cas_expected, cell_id);
        if (cas_result == true)
        {
            // New config, send CONFIG.req
            config_retry_counter.store(configs->cell_config_retry);
        }
        else if (cas_expected == cell_id)
        {
            // Current cell CONFIG is on going, retry config by re-send CONFIG.req
            config_retry_counter.fetch_sub(1);
        }
        else
        {
            // Another cell config is ongoing, set this cell to pending and return
            cell_data[cell_id].pending_config.store(1);
            return 0;
        }

        // Start timer to check no CONFIG.resp when timeout
        start_reconfig_timer(cell_id, configs->cell_config_timeout);
    }

    // Determine target_cell_id first for correct DBT and config retrieval
    int target_cell_id = cell_remap_event[cell_id] ? cell_id_map_tmp[cell_id] : cell_id_map[cell_id];

    nv::phy_mac_msg_desc msg_desc;

    // Use target_cell_id to get DBT info for the correct cell during reconfiguration
    auto* dbt_enabled = lp->get_dbt_info(target_cell_id);
    if(dbt_enabled != nullptr && dbt_enabled->bf_stat_dyn_enabled)
    {
        //msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
        // Using NVIPC Large Buffer
        msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
    }

    if(transport().tx_alloc(msg_desc) < 0)
    {
        NVLOGW_FMT(TAG, "Failed to allocate nvipc buffer for cell {} CONFIG.req", cell_id);
        return -1;
    }

    auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_config_request_msg_t>(msg_desc, SCF_FAPI_CONFIG_REQUEST, cell_id, false);
    auto& req  = *reinterpret_cast<scf_fapi_config_request_msg_t*>(fapi);

    req.msg_body.num_tlvs        = 0;
    cell_configs_t& cell_configs = lp->get_cell_configs(target_cell_id);
    yaml::node&     yaml_config  = configs->get_yaml_config();
    if(!yaml_config.has_key("data"))
    {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{} parameters for CONFIG.request not found", __func__);
        return -1;
    }
    yaml::node config_req_params = yaml_config["data"];

    scf_fapi_tl_t* ptr = reinterpret_cast<scf_fapi_tl_t*>(req.msg_body.tlvs);

    NVLOGC_FMT(TAG, "cell_config: cell_id={} target_cell_id={} phyCellId={}", cell_id, target_cell_id, cell_configs.phyCellId);

    // Basic parameters for the first time initialization and restart_option = 1
    if(cell_remap_event[cell_id] || first_init[cell_id] || configs->get_restart_option() == 1)
    {
        ptr = add_tlv(req, ptr, CONFIG_TLV_PHY_CELL_ID, 2, cell_configs.phyCellId);
        ptr = add_tlv(req, ptr, CONFIG_TLV_FRAME_DUPLEX_TYPE, 1, cell_configs.frameDuplexType);
        ptr = add_tlv(req, ptr, CONFIG_TLV_DL_BANDWIDTH, 2, cell_configs.dlBandwidth);
        ptr = add_tlv(req, ptr, CONFIG_TLV_UL_BANDWIDTH, 2, cell_configs.ulBandwidth);
        if(configs->get_conformance_test_params()->conformance_test_enable)
        {
            uint16_t ul_bwp_size = cell_configs.ulBandwidth;
            ul_dci_freq_domain_bits = std::ceil(std::log2(ul_bwp_size * (ul_bwp_size + 1)/ 2));
        }
        ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_RX_ANT, 2, cell_configs.numRxAnt);
        ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_TX_ANT, 2, cell_configs.numTxAnt);
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_NUM_TX_PORT, 2, cell_configs.numTxPort);
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_NUM_RX_PORT, 2, cell_configs.numRxPort);
        ptr = add_tlv(req, ptr, CONFIG_TLV_DL_FREQ, 4, config_req_params["nDLAbsFrePointA"].as<uint32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_UL_FREQ, 4 , config_req_params["nULAbsFrePointA"].as<uint32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SCS_COMMON, 1, cell_configs.mu);
        // ptr = add_tlv(req, ptr, CONFIG_TLV_FRAME_DUPLEX_TYPE, 1, config_req_params["nFrameDuplexType"].as<int32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_PBCH_POWER, 4, config_req_params["nSSBPwr"].as<int32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_PERIOD, 1, config_req_params["nSSBPeriod"].as<int32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_SUBCARRIER_OFFSET, 1, config_req_params["nSSBSubcOffset"].as<int32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_MASK, 4, config_req_params["nSSBMask"].child(0).as<int32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_MASK, 4, config_req_params["nSSBMask"].child(1).as<int32_t>());
        // Measurement parameters
    #ifdef SCF_FAPI_10_04
        ptr = add_tlv(req, ptr, CONFIG_TLV_RSRP_MEAS, 1, config_req_params["rsrpMeasurement"].as<uint32_t>());
    #endif
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_NOISE_VAR_MEAS, 1, config_req_params["pnMeasurement"].as<uint32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_PF_234_INTERFERENCE_MEAS, 1, config_req_params["pf_234_interference"].as<uint32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_PRACH_INTERFERENCE_MEAS, 1, config_req_params["prach_interference"].as<uint32_t>());
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_PUSCH_AGGR_FACTOR, 1, config_req_params["pusch_aggr_factor"].as<uint8_t>());
    }

    // PRACH parameters can be included at initialization, restart_option = 1 or 2.
    if(cell_remap_event[cell_id] || first_init[cell_id] || configs->get_restart_option() == 1 || configs->get_restart_option() == 2)
    {
        // PRACH
        prach_configs_t& prach_configs = lp->get_prach_configs(target_cell_id);

        ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_SEQ_LEN, 1, prach_configs.prachSequenceLength);
        ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_SUBC_SPACING, 1, prach_configs.prachSubCSpacing);
        ptr = add_tlv(req, ptr, CONFIG_TLV_RESTRICTED_SET_CONFIG, 1, prach_configs.restrictedSetConfig);
        ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_CONFIG_INDEX, 1, prach_configs.prachConfigIndex);
        ptr = add_tlv(req, ptr, CONFIG_TLV_SSB_PER_RACH, 1, prach_configs.SsbPerRach);
        ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_MULT_CARRIERS_IN_BAND, 1, prach_configs.prachMultipleCarriersInABand);
        ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_PRACH_FD_OCCASIONS, 1, prach_configs.numPrachFdOccasions);

        NVLOGI_FMT(TAG, "send_config_request: cell_id={} with prachConfigIndex={} restrictedSetConfig={}",
            cell_id, prach_configs.prachConfigIndex, prach_configs.restrictedSetConfig);

        for(int i = 0; i < prach_configs.numPrachFdOccasions; i++)
        {
            prach_fd_occasion_config_t& fd_occasion = prach_configs.prachFdOccasions[i];

            ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_ROOT_SEQ_INDEX, 2, fd_occasion.prachRootSequenceIndex);
            ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_ROOT_SEQ, 1, fd_occasion.numRootSequences);
            ptr = add_tlv(req, ptr, CONFIG_TLV_K1, 2, fd_occasion.k1);
            ptr = add_tlv(req, ptr, CONFIG_TLV_PRACH_ZERO_CORR_CONF, 1, fd_occasion.prachZeroCorrConf);
            ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_UNUSED_ROOT_SEQ, 2, fd_occasion.numUnusedRootSequences);
            for(int j = 0; j < fd_occasion.numUnusedRootSequences; j++)
            {
                ptr = add_tlv(req, ptr, CONFIG_TLV_NUM_UNUSED_ROOT_SEQ, 2, fd_occasion.unusedRootSequences[j]);
            }
        }
    }

    // Other parameters for the first time initialization and restart_option = 1
    if(cell_remap_event[cell_id] || first_init[cell_id] || configs->get_restart_option() == 1)
    {
        uint32_t nMIB[3];
        nMIB[0]      = config_req_params["nMIB"].child(0).as<uint32_t>();
        nMIB[1]      = config_req_params["nMIB"].child(1).as<uint32_t>();
        nMIB[2]      = config_req_params["nMIB"].child(2).as<uint32_t>();
        uint32_t mib = (nMIB[2] << 16) | (nMIB[1] << 9) | nMIB[0];

        ptr = add_tlv(req, ptr, CONFIG_TLV_MIB, 4, mib);

        uint16_t dlGridSize[5]      = {0};
        dlGridSize[cell_configs.mu] = cell_configs.dlGridSize;
        uint16_t ulGridSize[5]      = {0};
        ulGridSize[cell_configs.mu] = cell_configs.ulGridSize;

        ptr = add_tlv(req, ptr, CONFIG_TLV_DL_GRID_SIZE, sizeof(dlGridSize), reinterpret_cast<uint8_t*>(dlGridSize));
        ptr = add_tlv(req, ptr, CONFIG_TLV_UL_GRID_SIZE, sizeof(ulGridSize), reinterpret_cast<uint8_t*>(ulGridSize));

        std::vector<precoding_matrix_t>& precoding_matrix_v = lp->get_precoding_matrix_v(target_cell_id);
        for(int i = 0; i < precoding_matrix_v.size(); i++)
        {
            precoding_matrix_t& matrix = precoding_matrix_v[i];

            uint16_t length = matrix.numLayers * matrix.numAntPorts;
            if(matrix.precoderWeight_v.size() != length)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PM {} _coef precoderWeight_v length error: expected={} v_size={}", matrix.PMidx, length, matrix.precoderWeight_v.size());
            }

            int      size       = sizeof(uint16_t) * 3 + sizeof(complex_int16_t) * length;
            uint8_t* matrix_buf = reinterpret_cast<uint8_t*>(malloc(size));
            if (matrix_buf == nullptr)
            {
                NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: malloc failed", __func__);
                return -1;
            }

            uint16_t* pm_data = reinterpret_cast<uint16_t*>(matrix_buf);
            *pm_data++        = matrix.PMidx;
            *pm_data++        = matrix.numLayers;
            *pm_data++        = matrix.numAntPorts;

            complex_int16_t* pw_data = reinterpret_cast<complex_int16_t*>(pm_data);
            for(complex_int16_t precoderWeight : matrix.precoderWeight_v)
            {
                pw_data->re = precoderWeight.re;
                pw_data->im = precoderWeight.im;
                pw_data++;
            }
            ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_PRECODING_MATRIX, size, matrix_buf);
            free(matrix_buf);
            NVLOGI_FMT(TAG, "PM {} layer={}, ports={} v_length={}", matrix.PMidx, matrix.numLayers, matrix.numAntPorts, matrix.precoderWeight_v.size());
        }

        //TODO: Should be read and stored from the TV's. Also instead of msg_buf, the contents for the DBT PDU should encoded to msg_buff
        // Use target_cell_id to get DBT info for the correct cell during reconfiguration
        auto* dbt = lp->get_dbt_info(target_cell_id);
        //if (dbt != nullptr && dbt->bf_stat_dyn_enabled && msg_desc.data_buf != nullptr) {
        // Using NVIPC Large Buffer
        if (dbt != nullptr && dbt->bf_stat_dyn_enabled && msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_LARGE &&msg_desc.data_buf != nullptr) {
            uint16_t numDigBeams = dbt->num_static_beamIdx;
            uint16_t numTXRUs = dbt->num_TRX_beamforming;
            std::size_t size = ((sizeof(uint16_t) * 2)+ (numDigBeams * (sizeof(uint16_t) + (sizeof(complex_int16_t) * numTXRUs))));
            //  uint8_t* dbt_iq_buf = reinterpret_cast<uint8_t*>(dbt.dbt_data_buf.data());

            uint16_t* dbt_data = reinterpret_cast<uint16_t*>(msg_desc.data_buf);
            *dbt_data++        = numDigBeams;
            *dbt_data++        = numTXRUs;

            for(int i = 0; i < numDigBeams; i++)
            {
                *dbt_data++ = i+1;
                complex_int16_t* dbt_wt_data = reinterpret_cast<complex_int16_t*>(dbt_data);
                for(int j = 0; j < numTXRUs; j++)
                {
                    const std::size_t dbt_index = (static_cast<std::size_t>(i) * numTXRUs) + static_cast<std::size_t>(j);
                    dbt_wt_data->re = dbt->dbt_data_buf[dbt_index].re;
                    dbt_wt_data->im = dbt->dbt_data_buf[dbt_index].im;
                    dbt_wt_data++;
                }
                dbt_data = reinterpret_cast<uint16_t*>(dbt_wt_data);
                NVLOGI_FMT(TAG, "DBT beamIdx={} numDigBeams={}, numTXRUs={} DBTSize={}", i+1, numDigBeams, numTXRUs, size);
            }
            msg_desc.data_len = size;
            NVLOGI_FMT(TAG, "DBT PDU size={}", msg_desc.data_len);

            ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU, 0, nullptr);
            // if (msg_desc.data_buf != nullptr) {
            //     memcpy(msg_desc.data_buf, dbt_iq_buf, size);
            // }

    }

        ptr = add_tlv(req, ptr, CONFIG_TLV_RSSI_MEAS, sizeof(uint8_t), 1);
    }
#ifdef SCF_FAPI_10_04
    uint8_t * indPerSlotPtr = configs->get_indication_per_slot();
    ptr = add_tlv(req, ptr, CONFIG_TLV_INDICATION_INSTANCES_PER_SLOT, 6, (indPerSlotPtr)); // Size of test_mac_configs#indication_per_slot is 6
#endif

    if (config_req_params["channel_segment_timelines"].as<uint32_t>() != 0) {
        auto& timelines = lp->get_channel_segment(cell_id);

        auto allocation = sizeof(uint8_t) + timelines.size() * sizeof(scf_channel_segment_info_t);
        auto m_buf = std::vector<uint8_t>(allocation);
        uint8_t* ptr_buf = m_buf.data();
        auto segment = reinterpret_cast<scf_channel_segment_t*>(ptr_buf);
        segment->nPduSegments = static_cast<uint8_t>(timelines.size());
        uint8_t* segment_buf = reinterpret_cast<uint8_t*>(segment->payload);

        for (auto &timeline: timelines) {
            auto seg = reinterpret_cast<scf_channel_segment_info_t*>(segment_buf);
            seg->type = to_scf_ch_seg_type(timeline);
            seg->chan_start_offset = timeline.channel_start_offset;
            seg->chan_duration = timeline.channel_duration;
            segment_buf += sizeof(scf_channel_segment_info_t);
        }
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_CHAN_SEGMENT, allocation, m_buf.data());
    }
#ifdef SCF_FAPI_10_04
    if(lp->get_srs_enabled() == 1)
    {
        ptr = add_tlv(req, ptr, CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS, 4, MAX_SRS_CHEST_BUFFERS_PER_CELL);
        NVLOGI_FMT(TAG, "CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS = {}", MAX_SRS_CHEST_BUFFERS_PER_CELL);
    }
#endif

#ifdef SCF_FAPI_10_04
    auto* csi2MapInfo = lp->get_csi2_maps(cell_id);
    NVLOGD_FMT(TAG, "Parsing CONFIG_TLV_UCI_CONFIG");
    if ((csi2MapInfo!= nullptr) && (csi2MapInfo->nCsi2Maps > 0)) {

        std::vector<uint8_t> mapData;
        mapData.reserve(csi2MapInfo->totalSizeInBytes);
        auto mapDataBuf = mapData.data();
        uint16_t* buf16 = reinterpret_cast<uint16_t*>(mapDataBuf);
        uint32_t len = 0;
        *buf16++ = csi2MapInfo->nCsi2Maps;
        mapDataBuf+=sizeof(uint16_t);
        len+=sizeof(uint16_t);
        // NVLOGD_FMT(TAG, "len = {}", len);

        for (uint32_t i = 0; i < csi2MapInfo->nCsi2Maps; i++) {
            auto& mapParams = csi2MapInfo->mapParams[i];
            *mapDataBuf++ = mapParams.numPart1Params;
            len+=sizeof(uint8_t);
            // NVLOGC_FMT(TAG, "i = {} len {}", i, len);
            std::copy(mapParams.sizePart1Params.data(), mapParams.sizePart1Params.data() + mapParams.numPart1Params,  mapDataBuf);
            mapDataBuf +=  sizeof(uint8_t) * mapParams.numPart1Params;
            len +=  sizeof(uint8_t) * mapParams.numPart1Params;
            // NVLOGD_FMT(TAG, "i = {} len {} size1PartParams", i, len);
            buf16 = reinterpret_cast<uint16_t*>(mapDataBuf);
            std::copy( mapParams.map.data(),  mapParams.map.data() + mapParams.map.size(), buf16);
            mapDataBuf+= sizeof(uint16_t) * mapParams.map.size();
            len += sizeof(uint16_t) * mapParams.map.size();
            // NVLOGD_FMT(TAG, "CONFIG_TLV_UCI_CONFIG i = {} len {} map", i, len);
            NVLOGD_FMT(TAG, "CONFIG_TLV_UCI_CONFIG numPart1Params {} mapsize {} first 0x{:04X} last 0x{:04X}", mapParams.numPart1Params, mapParams.map.size(), *buf16, *(buf16 + mapParams.map.size() - 1));
        }
        
        NVLOGD_FMT(TAG, "CONFIG_TLV_UCI_CONFIG len 0x{:02X}", len);

        ptr = add_tlv(req, ptr,CONFIG_TLV_UCI_CONFIG, len, mapData.data());
        mapData.clear();
    }
#endif
    // Update message length
    req.msg_hdr.length           = (uint8_t*)ptr - (uint8_t*)&req.msg_body;
    msg_desc.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + req.msg_hdr.length;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} MAC_PHY_CELL_CONFIG_REQ", cell_id, static_cast<unsigned>(req.msg_hdr.type_id));
    NVLOGI_FMT(TAG, "MAC_PHY_CELL_CONFIG_REQ: cell_id={} msg_len={} PHY_CELL_ID {} DL_BANDWIDTH {} UL_BANDWIDTH {} NUM_RX_ANT {} NUM_RX_PORT {} NUM_TX_ANT {} NUM_TX_PORT {} SCS_COMMON {} PRACH_SUBC_SPACING {}",
            cell_id, msg_desc.msg_len, cell_configs.phyCellId, cell_configs.dlGridSize, cell_configs.ulGridSize, cell_configs.numRxAnt, cell_configs.numRxPort, cell_configs.numTxAnt, cell_configs.numTxPort, cell_configs.mu, cell_configs.mu);

    // Send the message over the transport
    transport().tx_send(msg_desc);
    transport().tx_post();
    return 0;
}

int scf_fapi_handler::send_start_request(int cell_id)
{
    nv::phy_mac_msg_desc msg_desc;
    if (transport().tx_alloc(msg_desc) < 0)
    {
        NVLOGW_FMT(TAG, "Failed to allocate nvipc buffer for cell {} START.req", cell_id);
        return -1;
    }

    auto            fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_body_header_t>(msg_desc, SCF_FAPI_START_REQUEST, cell_id, false);
    auto&           req  = *reinterpret_cast<scf_fapi_body_header_t*>(fapi);

    req.length = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} MAC_PHY_CELL_START_REQ", cell_id, static_cast<unsigned>(req.type_id));

    transport().tx_send(msg_desc);
    transport().tx_post();
    return 0;
}

int scf_fapi_handler::send_stop_request(int cell_id)
{
    nv::phy_mac_msg_desc msg_desc;
    if (transport().tx_alloc(msg_desc) < 0)
    {
        NVLOGW_FMT(TAG, "Failed to allocate nvipc buffer for cell {} STOP.req", cell_id);
        return -1;
    }

    auto            fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_body_header_t>(msg_desc, SCF_FAPI_STOP_REQUEST, cell_id, false);
    auto&           req  = *reinterpret_cast<scf_fapi_body_header_t*>(fapi);

    req.length = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} MAC_PHY_CELL_STOP_REQ", cell_id, static_cast<unsigned>(req.type_id));

    transport().tx_send(msg_desc);
    transport().tx_post();
    return 0;
}

// Return successfully sent FAPI message count
int scf_fapi_handler::schedule_fapi_request(int cell_id, sfn_slot_t ss, fapi_group_t group_id, int32_t ts_offset)
{
#ifdef PREPONE_TX_DATA_REQ // Strongly not recommended, caution to enable in release version.
    if (group_id == TX_DATA_REQ)
    {
        ss = get_next_sfn_slot(ss);
    }
#endif

    fapi_sched_t& fapi_sched = cell_data[cell_id].fapi_scheds[ss.u16.slot & 0x3];
    nv::phy_mac_msg_desc& msg_desc = fapi_sched.fapi_msg_cache[group_id];

    sfn_slot_t ss_curr = ss_tick.load();
    // Build FAPI message at one of the two: (1) builder thread with ts_offset < 0; (2) non-builder thread with ts_offset = 0.
    if ((configs->builder_thread_enable != 0 && ts_offset < 0) || (configs->builder_thread_enable == 0 && ts_offset == 0))
    {
        if (msg_desc.msg_buf != nullptr)
        {
            sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
            NVLOGI_FMT(TAG, "Current SFN {}.{} dropping FAPI: SFN {}.{} cell_id={} msg_id=0x{:02X}",
                    ss_curr.u16.sfn, ss_curr.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
            transport().tx_release(msg_desc);
            msg_desc.reset();
        }

        vector<fapi_req_t*>& fapi_reqs = get_fapi_req_list(cell_id, ss, group_id);

        if (fapi_reqs.size() == 0)
        {
            // If UL_TTI_REQ doesn't present and dummy_tti_enabled is true, create a blank DL_TTI.req, else skip
            if (group_id != DL_TTI_REQ || !configs->get_dummy_tti_enabled() || get_fapi_req_list(cell_id, ss, UL_TTI_REQ).size() > 0)
            {
                return 0;
            }
        }

        switch(group_id)
        {
            case DL_TTI_REQ: {
                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} DL_TTI.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_dl_tti_req_t>(msg_desc, SCF_FAPI_DL_TTI_REQUEST, cell_id, false);
                auto& req  = *reinterpret_cast<scf_fapi_dl_tti_req_t*>(fapi);
                req.sfn = ss.u16.sfn;
                req.slot = ss.u16.slot;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
                req.testMode =  0;
#endif
                if (configs->app_mode == 0) {
                    fapi->length += build_dl_tti_request(cell_id, fapi_reqs, req);
                } else {
                    fapi->length += build_dyn_dl_tti_request(cell_id, fapi_reqs, req, lp->get_dyn_slot_param(cell_id, ss));
                }
                msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
            }
            break;

            case TX_DATA_REQ: {
                if(data_buf_opt == 1)
                {
#ifdef ENABLE_32DL
                    NVLOGI_FMT(TAG, "Using large CPU pools for TX DATA for 32 layer DL");
                    msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
#else
                    NVLOGI_FMT(TAG, "Using small CPU pools for TX DATA for 16 layer DL");
                    msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
#endif
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

                // msg_desc.data_len = get_tx_data_len(fapi_reqs);

                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} TX_DATA.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                scf_fapi_header_t* hdr = reinterpret_cast<scf_fapi_header_t*>(msg_desc.msg_buf);

                hdr->message_count = 1;
                hdr->handle_id     = cell_id;
                auto body_hdr      = reinterpret_cast<scf_fapi_tx_data_req_t*>(&hdr->payload);

                body_hdr->msg_hdr.type_id = SCF_FAPI_TX_DATA_REQUEST;
                msg_desc.msg_id           = SCF_FAPI_TX_DATA_REQUEST;
                body_hdr->sfn = ss.u16.sfn;
                body_hdr->slot = ss.u16.slot;
                if (configs->app_mode == 0) {
                    build_tx_data_request(cell_id, fapi_reqs, *body_hdr, msg_desc);
                } else {
                    build_dyn_tx_data_request(cell_id, fapi_reqs, *body_hdr, msg_desc, lp->get_dyn_slot_param(cell_id, ss));
                }
                msg_desc.cell_id = cell_id;
            }
            break;

            case UL_TTI_REQ: {
                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} UL_TTI.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_ul_tti_req_t>(msg_desc, SCF_FAPI_UL_TTI_REQUEST, cell_id, false);
                auto& req  = *reinterpret_cast<scf_fapi_ul_tti_req_t*>(fapi);
                req.sfn = ss.u16.sfn;
                req.slot = ss.u16.slot;
                if (configs->app_mode == 0) {
                    fapi->length += build_ul_tti_request(cell_id, fapi_reqs, req);
                } else {
                    fapi->length += build_dyn_ul_tti_request(cell_id, fapi_reqs, req, lp->get_dyn_slot_param(cell_id, ss));
                }
                msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
            }
            break;

            case UL_DCI_REQ: {
                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} UL_DCI.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_ul_dci_t>(msg_desc, SCF_FAPI_UL_DCI_REQUEST, cell_id, false);
                auto& req  = *reinterpret_cast<scf_fapi_ul_dci_t*>(fapi);
                req.sfn = ss.u16.sfn;
                req.slot = ss.u16.slot;
                fapi->length += build_ul_dci_request(cell_id, fapi_reqs, req);
                msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
            }
            break;
            case DL_BFW_CVI_REQ: {
                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} DL_BFW_CVI.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_dl_bfw_cvi_request_t>(msg_desc, SCF_FAPI_DL_BFW_CVI_REQUEST, cell_id, false);
                scf_fapi_dl_bfw_cvi_request_t* req  = reinterpret_cast<scf_fapi_dl_bfw_cvi_request_t*>(fapi);
                req->sfn = ss.u16.sfn;
                req->slot = ss.u16.slot;
                fapi->length += build_dl_bfw_cvi_request(cell_id, fapi_reqs, req) + 5;
                msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
            }
            break;
            case UL_BFW_CVI_REQ: {
                if (transport().tx_alloc(msg_desc) < 0)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{} Failed to allocate nvipc buffer for cell {} UL_BFW_CVI.req", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot), cell_id);
                    return 0;
                }

                auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_ul_bfw_cvi_request_t>(msg_desc, SCF_FAPI_UL_BFW_CVI_REQUEST, cell_id, false);
                scf_fapi_ul_bfw_cvi_request_t* req  = reinterpret_cast<scf_fapi_ul_bfw_cvi_request_t*>(fapi);
                req->sfn = ss.u16.sfn;
                req->slot = ss.u16.slot;
                fapi->length += build_dl_bfw_cvi_request(cell_id, fapi_reqs, req) + 5;
                msg_desc.msg_len = fapi->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
            }
            break;
            default: {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unknown FAPI group_id={}", static_cast<unsigned>(group_id));
                return 0;
            }
        }
        fapi_sched.fapi_build_num ++;
    }

    // Send or cache the FAPI message
    int fapi_sent_count = 0;

    // Get the default FAPI delay configured by yaml
    std::vector<int32_t>& target_ts_offsets = *(fapi_sched.target_ts_offsets[group_id]);
    int32_t target_ts_offset = target_ts_offsets[get_slot_in_frame(ss) % target_ts_offsets.size()];

#if 0 // Only for debug: overwrite default FAPI delay configuration for specific cell_id, SFN, SLOT and FAPI message
    if (cell_id == 2 && ss.u16.slot == 0 && group_id == UL_TTI_REQ)
    {
        target_ts_offset = 0;
    }
#endif

    NVLOGD_FMT(TAG, "SFN {}.{} {}: cell_id={} group_id={} ts_offset={} target_ts_offset={}",
            ss.u16.sfn, ss.u16.slot, __func__, cell_id, +group_id, ts_offset, target_ts_offset);

    bool tx = (!configs->builder_thread_enable || target_ts_offset == 0) ? ts_offset >= target_ts_offset : ts_offset > 0;
    if (msg_desc.msg_buf != nullptr && tx)
    {
        NVLOGI_FMT(TAG, "SFN {}.{} SEND: cell_id={} {} msg_len={} data_len={}", ss.u16.sfn, ss.u16.slot, cell_id, get_scf_fapi_msg_name(msg_desc.msg_id), msg_desc.msg_len, msg_desc.data_len);

        if (msg_desc.msg_id == SCF_FAPI_UL_TTI_REQUEST) {
            slot_timing_t& timing = cell_data[cell_id].slot_timing[ss.u16.slot];
            timing.ss = ss;
            nvlog_gettime_rt(&timing.ts_ul_tti);
        }

        if (configs->app_mode != 0 && msg_desc.msg_id == SCF_FAPI_TX_DATA_REQUEST)
        {
            if(msg_desc.data_len > 0)
            {
                thrputs[cell_id].dl_thrput += msg_desc.data_len;
                thrputs[cell_id].slots[PDSCH]++;
            }
        }

        // Duplicate FAPI message sending if configured (For negative test: duplicate FAPI messages)
        if (configs->duplicate_fapi_bit_mask & (1 << group_id))
        {
            NVLOGI_FMT(TAG, "SFN {}.{} duplicate msg_id={} {} for {} times",
                    ss.u16.sfn, ss.u16.slot, msg_desc.msg_id, get_scf_fapi_msg_name(msg_desc.msg_id), configs->duplicate_fapi_num_max);
            int duplicate_num = rand() % configs->duplicate_fapi_num_max + 1;
            for (int i = 0; i < duplicate_num; i++) {
                // Allocate memory with the same data_pool for the duplicate FAPI message
                nv::phy_mac_msg_desc msg_desc_dup;
                msg_desc_dup.data_pool = msg_desc.data_pool;
                if (transport().tx_alloc(msg_desc_dup) < 0) {
                    NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "SFN {}.{} Failed to allocate memory for duplicate FAPI message", static_cast<unsigned>(ss.u16.sfn), static_cast<unsigned>(ss.u16.slot));
                    break;
                }
                // Copy the original FAPI header and body
                msg_desc_dup.msg_id = msg_desc.msg_id;
                msg_desc_dup.cell_id = msg_desc.cell_id;
                msg_desc_dup.msg_len = msg_desc.msg_len;
                msg_desc_dup.data_len = msg_desc.data_len;
                msg_desc_dup.data_pool = msg_desc.data_pool;
                memcpy(msg_desc_dup.msg_buf, msg_desc.msg_buf, msg_desc.msg_len);
                if (msg_desc.data_buf != nullptr) {
                    memcpy(msg_desc_dup.data_buf, msg_desc.data_buf, msg_desc.data_len);
                }
                // Send the duplicate FAPI message
                transport().tx_send(msg_desc_dup);
            }
        }

        transport().tx_send(msg_desc);
        fapi_sent_count ++;
        fapi_sched.fapi_sent_num ++;

        msg_desc.reset();
        if (notify_mode == IPC_SYNC_PER_MSG) {
            // Notify once every FAPI message
            transport().notify(1);
        }
    }

    return fapi_sent_count;
}

/*
 * Function schedule_fapi_reqs() may be called multi-times for 1 slot:
 * (1) If builder_thread_enable = 1, call one time when receiving SLOT.ind of previous slot
 * (2) Call one time when receiving SLOT.ind of current slot
 * (3) If schedule_total_time > 0, call one time at SLOT.ind + schedule_total_time
 */
int scf_fapi_handler::schedule_fapi_reqs(sfn_slot_t ss, int ts_offset)
{
    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        fapi_sched_t& fapi_sched = cell_data[cell_id].fapi_scheds[ss.u16.slot & 0x3];
        if(ts_offset < 0 || (ts_offset == 0 && configs->builder_thread_enable == 0))
        {
            // Reset the fapi_scheds buffer
            for(nv::phy_mac_msg_desc& msg_desc : fapi_sched.fapi_msg_cache)
            {
                if(msg_desc.msg_buf != nullptr)
                {
                    sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
                    NVLOGI_FMT(TAG, "Current SFN {}.{} dropping FAPI: SFN {}.{} cell_id={} msg_id=0x{:02X}", ss.u16.sfn, ss.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
                    transport().tx_release(msg_desc);
                    msg_desc.reset();
                }
            }
            fapi_sched.reset();
        }
    }

    int total_count = 0;

    // Variables for calculating per message sending time in contiguous sending
    int64_t before_contiguous_send = 0;
    int contiguous_send_num = 0;
    int contiguous_remain_num = 0;

    int MAX_FAPI_NUM = cell_num * FAPI_REQ_SIZE;
    for(int i = 0; i < MAX_FAPI_NUM; i++)
    {
        // ts_offset < 0 means it's running FAPI building in builder thread, use i as sequence_id directly
        int sequence_id = ts_offset < 0 ? i : schedule_sequence[i];

        schedule_item_t& item = schedule_item_list[sequence_id];
        int cell_id = item.cell_id;
        fapi_group_t group_id = item.group_id;

        fapi_sched_t& fapi_sched = cell_data[cell_id].fapi_scheds[ss.u16.slot & 0x3];
        if(cell_data[cell_id].schedule_enable == false)
        {
            // Skip stopped cells
            continue;
        }

        if(item.exist == 0 && ts_offset > 0 && configs->get_fapi_tx_deadline_enable())
        {
            // Skip non-existent items sending
            continue;
        }

        // Sleep to a proper time to send the FAPI message before deadline (ts_offset > 0 means time-controlled sending or STT sending)
        if (ts_offset > 0 && configs->get_fapi_tx_deadline_enable())
        {
            int64_t now_ns = static_cast<int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
            if (before_contiguous_send == 0)
            {
                before_contiguous_send = now_ns;
                contiguous_remain_num = item.remain_num;
            }

            int64_t send_time_ns = item.tx_deadline - static_cast<int64_t>(item.remain_num) * configs->get_fapi_tx_time_per_msg_ns();
            int64_t sleep_ns = ts_tick_tai + send_time_ns - now_ns;
            if (sleep_ns > (SLOT_INTERVAL * 2)) // Max sleep time should < 2 slots (1000 us for 15kHz SCS)
            {
                NVLOGW_FMT(TAG, "SFN {}.{} LONG_SLEEP: cell_id={} group_id={}-{} remain_num={} sleep_ns={} tx_deadline={} ts_tick_tai={} now_ns={}",
                    ss.u16.sfn, ss.u16.slot, cell_id, group_id, get_fapi_group_name(group_id), item.remain_num, sleep_ns, item.tx_deadline, ts_tick_tai, now_ns);
                sleep_ns = SLOT_INTERVAL * 2;
            }

            // Do not sleep if the time gap is less than MAX_SCHEDULE_AHEAD_TIME_NS
            if (sleep_ns > MAX_SCHEDULE_AHEAD_TIME_NS)
            {
                // Subtract target AVG_SCHEDULE_AHEAD_TIME_NS and SYSTEM_WAKEUP_TIME_COST_NS
                sleep_ns -= (AVG_SCHEDULE_AHEAD_TIME_NS + SYSTEM_WAKEUP_TIME_COST_NS);
                struct timespec t_sleep;
                t_sleep.tv_sec = sleep_ns / 1000000000LL;
                t_sleep.tv_nsec = sleep_ns % 1000000000LL;
                nanosleep(&t_sleep, nullptr);

                // Reset the contiguous send time and counter for calculating average message sending time
                before_contiguous_send = static_cast<int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
                contiguous_remain_num = item.remain_num;
                contiguous_send_num = 0;

                NVLOGI_FMT(TAG, "SFN {}.{} SLEEP: cell_id={} group_id={}-{} remain_num={} sleep_ns={} tx_deadline={} ts_tick_tai={} now_ns={} wakeup_diff={}",
                    ss.u16.sfn, ss.u16.slot, cell_id, group_id, get_fapi_group_name(group_id), item.remain_num, sleep_ns, item.tx_deadline, ts_tick_tai, now_ns, before_contiguous_send - sleep_ns - now_ns);
            }
            contiguous_send_num ++;
        }

        int fapi_sent = schedule_fapi_request(cell_id, ss, group_id, ts_offset);

        // NVLOGV_FMT(TAG, "SFN {}.{} SCHED: cell_id={} group_id={} fapi_build_num={} fapi_sent_num={}", ss.u16.sfn, ss.u16.slot, cell_id, group_id, fapi_sched.fapi_build_num, fapi_sched.fapi_sent_num);

        // When sent the last FAPI message of a cell, append a SLOT.resp or notify transport if configured
        if ((configs->builder_thread_enable != 0 && fapi_sent > 0 && fapi_sched.fapi_sent_num == fapi_sched.fapi_build_num) // When builder thread is enabled, check if send number equals to build number
            || (configs->builder_thread_enable == 0 && group_id == FAPI_REQ_SIZE - 1)) // When builder thread is disabled, check if this is the last fapi group of a cell
        {
            // A cell has sent all FAPI messages
#ifdef ENABLE_L2_SLT_RSP
            send_slot_response(cell_id, ss);
            contiguous_send_num ++;
            total_count ++;
#else
            // Cell ended
            if(notify_mode == IPC_SYNC_PER_CELL)
            {
                // Notify once every cell
                NVLOGI_FMT(TAG, "SFN {}.{} NOTIFY: cell_id={} fapi_build_num={} fapi_sent_num={}", ss.u16.sfn, ss.u16.slot, cell_id, fapi_sched.fapi_build_num, fapi_sched.fapi_sent_num);
                transport().notify(fapi_sched.fapi_sent_num);
            }
#endif
            total_count += fapi_sched.fapi_sent_num;
        }

        if (ts_offset > 0 && configs->get_fapi_tx_deadline_enable() && (item.remain_num == 1 || contiguous_remain_num == contiguous_send_num))
        {
            int64_t now_ns = static_cast<int64_t>(std::chrono::system_clock::now().time_since_epoch().count());
            int64_t finish_time_ns = now_ns - ts_tick_tai;
            int64_t deadline_diff_ns = item.tx_deadline - finish_time_ns;
            int64_t fapi_tx_per_msg = contiguous_send_num > 0 ? (now_ns - before_contiguous_send) / contiguous_send_num : 0;
            NVLOGI_FMT(TAG, "SFN {}.{} DEADLINE: cell_id={} group_id={}-{} tx_deadline_ns={} finish_time_ns={} diff_ns={} fapi_tx_per_msg={} contiguous_send_num={}",
                ss.u16.sfn, ss.u16.slot, cell_id, group_id, get_fapi_group_name(group_id), item.tx_deadline, finish_time_ns, deadline_diff_ns, fapi_tx_per_msg, contiguous_send_num);

            if (stat_deadline_diff != nullptr)
            {
                stat_deadline_diff->add(stat_deadline_diff, deadline_diff_ns);
            }
            if (stat_tx_per_msg != nullptr)
            {
                stat_tx_per_msg->add(stat_tx_per_msg, fapi_tx_per_msg);
            }
        }
    }

    return total_count;
}

int scf_fapi_handler::schedule_slot(sfn_slot_t ss)
{

    //Variables used for MAC.PROCESSING_TIMES message
    std::chrono::nanoseconds fapi1_start(0);
    std::chrono::nanoseconds fapi1_stop(0);
    int fapi1_count = 0;
    std::chrono::nanoseconds fapi2_start(0);
    std::chrono::nanoseconds fapi2_stop(0);
    int fapi2_count = 0;
    std::chrono::nanoseconds notify_start(0);
    std::chrono::nanoseconds notify_stop(0);

    // Schedule non delaying FAPI messages
    fapi1_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

    if (configs->get_fapi_tx_deadline_enable())
    {
        reorder_schedule_sequence(ss);
    }

    // Schedule slot with ts_offset = 0
    int total_sent = schedule_fapi_reqs(ss, 0);

    fapi1_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    fapi1_count = total_sent;

    auto t_last = std::chrono::system_clock::now().time_since_epoch();

    // Sleep for delaying FAPI messages / event notification if configured
    int t_sleep_ns = 0;

    uint32_t slot_idx = get_slot_in_frame(ss) % stt_estimate.size();
    stt_estimate_t& stt = stt_estimate[slot_idx];

    int32_t delay_time_idx = get_slot_in_frame(ss) % configs->schedule_total_time.size();
    int32_t schedule_total_time = configs->schedule_total_time[delay_time_idx];

    // Determine FAPI sending start time
    int64_t schedule_start_time = schedule_total_time;
    int fapi_num_to_send = 0;
    if (configs->builder_thread_enable > 0 && schedule_total_time > 0) {
        for (int cell_id = 0; cell_id < cell_num; cell_id ++)
        {
            if (cell_data[cell_id].schedule_enable == false)
            {
                // Skip stopped cells
                continue;
            }

            fapi_sched_t& fapi_sched = cell_data[cell_id].fapi_scheds[ss.u16.slot & 0x3];
            fapi_num_to_send += fapi_sched.fapi_build_num + 1; // Add one SLOT.resp (END.req)
        }

        if (stt.estimate_send_time < 0) {
            // Start scheduling FAPI message from T0 + schedule_total_time. No sending in advance
            schedule_start_time = schedule_total_time;
        } else if (stt.estimate_send_time == 0) {
            // Estimate 1 us for 1 message sending time cost by default
            schedule_start_time = schedule_total_time - 1000 * fapi_num_to_send - STT_ESTIMATE_ADJUST_OFFSET;
        } else {
            // Use estimate_send_time calculated from the last second practical data
            schedule_start_time = schedule_total_time - stt.estimate_send_time - STT_ESTIMATE_ADJUST_OFFSET;
        }
    }

    //Determine tick time
    uint64_t t_slot = sfn_to_tai(ss.u16.sfn, ss.u16.slot, ts_tick + AppConfig::getInstance().getTaiOffset(), 0, 0, 1) - AppConfig::getInstance().getTaiOffset();
    t_slot -= SLOT_ADVANCE*SLOT_TIME_BOUNDARY_NS; // Subtract off tick slot advance.
    ts_tick_tai = t_slot;

    if (configs->get_fapi_tx_deadline_enable())
    {
        // Schedule slot with ts_offset > 0 (Here only need ts_offset> 0 to represent time-controlled sending)
        total_sent += schedule_fapi_reqs(ss, 1);
    } else if (schedule_total_time > 0) {
        // Force L2 scheduler to take the actual L2 scheduling time plus whatever t_sleep adds up to the desired total time.
        // Also include the offset due to SLOT.indication reception latency, assuming taking credit for any time after
        // the slot time boundary.  NB: This code only works for PTP GPS_ALPHA = GPS_BETA = 0.  For non-zero values, a
        // correction factor is necessary.

        int t_slot_modulo = ts_tick % SLOT_TIME_BOUNDARY_NS;
        t_sleep_ns = schedule_start_time - (t_last.count() - t_slot);
        NVLOGI_FMT(TAG,"Sleeping {} for SFN.slot {}.{} tick {} from {}",t_sleep_ns,ss.u16.sfn,ss.u16.slot,t_slot,t_last.count());
        if (t_sleep_ns > 0)
        {
            struct timespec t_sleep_tspec;
            t_sleep_tspec.tv_sec = 0;
            t_sleep_tspec.tv_nsec = t_sleep_ns;
            nanosleep(&t_sleep_tspec, NULL);
        }

        // Schedule delayed FAPI messages and event notify
        fapi2_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

        // Schedule slot with ts_offset = schedule_total_time
        total_sent += schedule_fapi_reqs(ss, schedule_total_time);

        fapi2_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        fapi2_count = total_sent - fapi1_count;

        // If estimate_send_time feature was not disabled (-1) by yaml config
        if (stt.estimate_send_time >= 0)
        {
            // Calculate schedule start time, end time, average time cost, the difference between expected end time and actual end time
            int64_t schedule_end_time = fapi2_stop.count() - t_slot;
            int64_t stt_diff = schedule_end_time - schedule_total_time;
            stt.stat_diff->add(stt.stat_diff, stt_diff); // ns
            stat_stt_diff->add(stat_stt_diff, stt_diff); // ns

            // Update average estimate_send_time for each slot of the launch pattern every second
            if(stt.stat_send->get_counter(stt.stat_send) == stt.statistic_period - 1)
            {
                if(stt.estimate_counter < STT_ESTIMATE_COUNT_MAX)
                {
                    stt.estimate_send_time = stt.stat_send->get_avg_value(stt.stat_send);
                    stt.estimate_counter++;
                }

                // Narror down the statistic warning window of stt_diff to +/-10us after the first second
                if(stt.estimate_send_time == 0)
                {
                    stt.stat_diff->set_limit(stt.stat_diff, -10 * 1000, 10 * 1000);
                    stt.stat_diff->set_limit(stt.stat_diff, -10 * 1000, 10 * 1000);
                }
            }

            // Calculate send time of this slot and update statistic info
            int64_t slot_send_time = fapi2_stop.count() - fapi2_start.count();
            stt.stat_send->add(stt.stat_send, slot_send_time); // ns
            stat_stt_send->add(stat_stt_send, slot_send_time); // ns

            NVLOGI_FMT(TAG, "SFN {}.{}: STT={} schedule_start_time={} schedule_end_time={} schedule_time_diff={} fapi_num_sent={} slot_send_time={}",
                    ss.u16.sfn, ss.u16.slot, schedule_total_time, schedule_start_time, schedule_end_time, stt_diff, fapi_num_to_send, slot_send_time);
        }
    }

    notify_start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    if(total_sent > 0 && notify_mode == IPC_SYNC_PER_TTI)
    {
        // Notify once every TTI tick
        transport().notify(total_sent);
    }
    notify_stop = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto t_end = std::chrono::system_clock::now().time_since_epoch();

    NVLOGI_FMT(TAG, "{{TI}} <testMAC Scheduler,{},{},0,0> Start Task:{},Last Cell:{},Sleep:{},End Task:{},sleep_ns={}",
            ss.u16.sfn, ss.u16.slot, ts_tick, t_last.count(), t_last.count() + t_sleep_ns, t_end.count(), t_sleep_ns);

    //MAC.PROCESSING_TIMES message containing full timeline of testmac message sending - used to debug L2 (testmac) vs L2A message processing latency
    NVLOGI_FMT(TAG_PROCESSING_TIMES, "SFN {}.{} {} tick={} slot_indication={} fapi1_start={} fapi1_stop={} fapi1_count={} sleep_time={} fapi2_start={} fapi2_stop={} fapi2_count={} notify_start={} notify_stop={}",
               ss.u16.sfn, ss.u16.slot, __func__,
               t_slot,
               ts_tick,
               fapi1_start.count(),fapi1_stop.count(),fapi1_count,
               t_sleep_ns,
               fapi2_start.count(),fapi2_stop.count(),fapi2_count,
               notify_start.count(),notify_stop.count());

    return total_sent;
}

int scf_fapi_handler::send_fapi_from_queue(int cell_id, sfn_slot_t ss_curr)
{
    nv::phy_mac_msg_desc msg_desc;
    int count = 0;
    cell_data[cell_id].queue_mutex.lock();
    while (cell_data[cell_id].msg_queues.size() > 0) {
        msg_desc = cell_data[cell_id].msg_queues.front();
        sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
        if (ss_msg.u32 == ss_curr.u32)
        {
            count ++;
            cell_data[cell_id].msg_queues.pop();
            transport().tx_send(msg_desc);
            if(notify_mode == IPC_SYNC_PER_MSG)
            {
                // Notify once every FAPI message
                transport().notify(1);
            }
        }
        else if (ss_msg.u32 == get_next_sfn_slot(ss_curr).u32)
        {
            // SLOT message for target slot ended
            break;
        }
        else
        {
            cell_data[cell_id].msg_queues.pop();
            // Invalid message, drop it
            NVLOGW_FMT(TAG, "SFN {}.{}: dropping invalid FAPI for SFN {}.{}",
                    ss_curr.u16.sfn, ss_curr.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot);
            transport().tx_release(msg_desc);
        }
    }
    cell_data[cell_id].queue_mutex.unlock();
    return count;
}

int scf_fapi_handler::send_mem_bank_cv_config_req(int cell_id) {

    auto& cv_mem_bank_configs = lp->get_mem_bank_configs(cell_id);
    nv::phy_mac_msg_desc msg_desc[MAX_NVIPC_FOR_MEM_BANK_CV_CONFIG_REQ];

    NVLOGI_FMT(TAG, "send_mem_bank_cv_config_req: cell_id={}, target_cell_id={}", cell_id, cell_id_map[cell_id]);

    std::size_t offset = 0;
    uint8_t idx = 0;
    uint8_t i = 0;
    uint8_t numUes = cv_mem_bank_configs.data.size();
    uint8_t startUeIdx = 0;
    uint8_t numUesSrsSamples = 0;
    numUesSrsSamples = ((numUes >= NUM_SUPPORTED_SRS_PDU) ? NUM_SUPPORTED_SRS_PDU:numUes);
    configs->num_mem_bank_cv_config_req_sent[cell_id] = ((numUes + NUM_SUPPORTED_SRS_PDU - 1) / NUM_SUPPORTED_SRS_PDU);

    if(configs->num_mem_bank_cv_config_req_sent[cell_id] > MAX_NVIPC_FOR_MEM_BANK_CV_CONFIG_REQ)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "send_mem_bank_cv_config_req: cell_id={}, numUes={}, numUesSrsSamples={}, configs->num_mem_bank_cv_config_req_sent={} is more than MAX_NVIPC_FOR_MEM_BANK_CV_CONFIG_REQ {}", 
            cell_id, numUes, numUesSrsSamples, configs->num_mem_bank_cv_config_req_sent[cell_id], MAX_NVIPC_FOR_MEM_BANK_CV_CONFIG_REQ);
        return -1;
    }
    for (uint8_t idx = 0; idx < configs->num_mem_bank_cv_config_req_sent[cell_id] ; idx++)
    {
        msg_desc[idx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;
        if(data_buf_opt == 1)
        {
            msg_desc[idx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;
        }
        else if(data_buf_opt == 2)
        {
            NVLOGI_FMT(TAG, "Cannot use GPU pools for TX DATA yet");
            msg_desc[idx].data_pool = NV_IPC_MEMPOOL_CUDA_DATA;
            return -1;
        }
        else if(data_buf_opt == 3)
        {
            //NVLOGI(TAG, "Creating GPU pools(with GDR copy) for TX DATA data_buf_opt = %d\n",data_buf_opt);
            //msg_desc.data_pool = NV_IPC_MEMPOOL_GPU_DATA;
            return -1;
        }
        if (transport().tx_alloc(msg_desc[idx]) < 0)
        {
            NVLOGW_FMT(TAG, "Failed to allocate nvipc buffer for cell {} CV_MEM_BANK_CONFIG.req", cell_id);
            return -1;
        }

        auto  fapi = scf_5g_fapi::add_scf_fapi_hdr<cv_mem_bank_config_request_msg_t>(msg_desc[idx], CV_MEM_BANK_CONFIG_REQUEST, cell_id, false);
        auto& req  = *reinterpret_cast<cv_mem_bank_config_request_msg_t*>(fapi);

        cv_mem_bank_config_request_body_t& req_body = req.msg_body;
        req_body.numUes = numUesSrsSamples - startUeIdx;

        offset = 0;

        msg_desc[idx].data_len  = 0;
        uint8_t* ptr = reinterpret_cast<uint8_t*> (&req_body.cv_info);
        for (i = startUeIdx; i < numUesSrsSamples; i++) {
            auto ue_cv_ptr = reinterpret_cast<ue_cv_info*>(ptr);
            auto &cv_info = cv_mem_bank_configs.data[i];
            ue_cv_ptr->rnti = cv_info.RNTI;
            ue_cv_ptr->reportType = cv_info.reportType;
            ue_cv_ptr->startPrbGrp = cv_info.startPrbGrp;
            ue_cv_ptr->srsPrbGrpSize = cv_info.srsPrbGrpSize;
            ue_cv_ptr->nPrbGrps = cv_info.nPrbGrps;
            ue_cv_ptr->nGnbAnt = cv_info.nGnbAnt;
            ue_cv_ptr->nUeAnt = cv_info.nUeAnt;
            ue_cv_ptr->offset = msg_desc[idx].data_len;
            NVLOGD_FMT(TAG, "send_mem_bank_cv_config_req: cell_id={} rnti={}, reportType={} startPrbGrp={} srsPrbGrpSize={} nPrbGrps={} nGnbAnt={} nUeAnt={} offset={} \n",
                cell_id, static_cast<uint16_t>(ue_cv_ptr->rnti),static_cast<uint8_t>(ue_cv_ptr->reportType),
                static_cast<uint16_t>(ue_cv_ptr->startPrbGrp),static_cast<uint32_t>(ue_cv_ptr->srsPrbGrpSize),static_cast<uint16_t>(ue_cv_ptr->nPrbGrps),
                static_cast<uint8_t>(ue_cv_ptr->nGnbAnt),static_cast<uint8_t>(ue_cv_ptr->nUeAnt),
                static_cast<uint32_t>(ue_cv_ptr->offset));

            memcpy((uint8_t*)(msg_desc[idx].data_buf) + msg_desc[idx].data_len, cv_info.cv_samples.data(), cv_info.cv_samples.size());

            //uint8_t* print_ptr = ((uint8_t*)(msg_desc[idx].data_buf)) + msg_desc[idx].data_len;
            //NVLOGD_FMT(TAG, "{}: srsChEst = {} {} {} {}\n",__func__,*print_ptr, *(print_ptr+1), *(print_ptr+2), *(print_ptr+3));

            msg_desc[idx].data_len += cv_info.cv_samples.size();
            ptr += sizeof(ue_cv_info);
            offset += sizeof(ue_cv_info);
        }
        req.msg_hdr.length = sizeof(cv_mem_bank_config_request_msg_t) - sizeof(scf_fapi_body_header_t) + offset;
        msg_desc[idx].msg_len   = req.msg_hdr.length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
        transport().tx_send(msg_desc[idx]);
        transport().notify(1);
        startUeIdx = numUesSrsSamples;
        numUesSrsSamples += (((numUes - numUesSrsSamples)>= NUM_SUPPORTED_SRS_PDU) ? NUM_SUPPORTED_SRS_PDU : (numUes - numUesSrsSamples));
        NVLOGD_FMT(TAG, "startUeIdx {} , numUes {}, numUesSrsSamples {} idx {}", startUeIdx, numUes, numUesSrsSamples, idx);
    }
    /*This needs some correction. There was a misunderstanding that there is one to one relationship
    between UE and cv_mem_bank_configs read from TVs. So data buffer was allocated for msg_desc
    and cv_mem_bank_configs were copied on this data buffer and ue_cv_ptr->buffer points to the databuffer memory segment
    where cv_mem_bank_configs of that UE was copied.
    But it turns out the cv_mem_bank_configs are more than number of UE present in TV. So this one to one relation doesnot
    exist. All the cv_mem_bank_configs are to be sent as a seperate message.
    */
    return 0;
}

int scf_fapi_handler::compare_ul_measurement(fapi_validate* vald, ul_measurement_t& tv, uint8_t* payload, uint16_t pdu_type)
{
    if (vald->cell_id >= cell_data.size() || pdu_type >= UCI_PDU_TYPE_NUM)
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: invalid vald->cell_id={} cell_data.size()={}", __func__, vald->cell_id, cell_data.size());
        return -1;
    }

    int err = 0;
    ul_measurement_t& tolerance = lp->get_cell_configs(vald->cell_id).tolerance.ul_meas[pdu_type];

#ifdef SCF_FAPI_10_04
    scf_fapi_ul_meas_common_t* meas = reinterpret_cast<scf_fapi_ul_meas_common_t*>(payload);
    NVLOGD_FMT(TAG, "{} UL measurement: ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}", vald->get_msg_name(),
            static_cast<int>(meas->ul_sinr_metric),
            static_cast<unsigned>(meas->timing_advance),
            static_cast<int>(meas->timing_advance_ns),
            static_cast<unsigned>(meas->rssi),
            static_cast<unsigned>(meas->rsrp));
    err += FAPI_VALIDATE_U16_WARN(vald, meas->timing_advance, tv.TimingAdvance, tolerance.TimingAdvance);
    err += FAPI_VALIDATE_I16_WARN(vald, meas->timing_advance_ns, tv.TimingAdvanceNs, tolerance.TimingAdvanceNs);
    err += FAPI_VALIDATE_U16_WARN(vald, meas->rssi, tv.RSSI, tolerance.RSSI);
    err += FAPI_VALIDATE_U16_WARN(vald, meas->rsrp, tv.RSRP, tolerance.RSRP);
#else
    fapi_ul_measure_10_02_t* meas = reinterpret_cast<fapi_ul_measure_10_02_t*>(payload);
    NVLOGD_FMT(TAG, "{} UL measurement: ul_cqi={} ta={} rssi={}", vald->get_msg_name(),
            static_cast<unsigned short>(meas->ul_cqi), static_cast<unsigned short>(meas->timing_advance), static_cast<unsigned short>(meas->rssi));
    err += FAPI_VALIDATE_U8_WARN(vald, meas->ul_cqi, tv.UL_CQI, tolerance.UL_CQI);
    err += FAPI_VALIDATE_U16_WARN(vald, meas->timing_advance, tv.TimingAdvance, tolerance.TimingAdvance);
    err += FAPI_VALIDATE_U16_WARN(vald, meas->rssi, tv.RSSI, tolerance.RSSI);
#endif
    return err;
}

int scf_fapi_handler::compare_ul_measurement_ehq(fapi_validate* vald, ul_measurement_t& tv, uint8_t* payload, uint16_t pdu_type)
{
    if (vald->cell_id >= cell_data.size() || pdu_type >= UCI_PDU_TYPE_NUM)
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: invalid vald->cell_id={} cell_data.size()={}", __func__, vald->cell_id, cell_data.size());
        return -1;
    }
    int err = 0;
    ul_measurement_t& tolerance = lp->get_cell_configs(vald->cell_id).tolerance.ul_meas[pdu_type];
    cell_configs_t& cell_configs = lp->get_cell_configs(vald->cell_id);
#ifdef SCF_FAPI_10_04
    scf_fapi_ul_meas_common_t* meas = reinterpret_cast<scf_fapi_ul_meas_common_t*>(payload);
    NVLOGD_FMT(TAG, "{} UL measurement: ul-sinr={}", vald->get_msg_name(),
            static_cast<int>(meas->ul_sinr_metric));
    if(cell_configs.pusch_sinr_selector == 2)
        err += FAPI_VALIDATE_I16_WARN(vald, meas->ul_sinr_metric, tv.SNR_ehq, tolerance.SNR);
        
    err += FAPI_VALIDATE_U16_WARN(vald, meas->rssi, tv.RSSI_ehq, tolerance.RSSI);
    err += FAPI_VALIDATE_U16_WARN(vald, meas->rsrp, tv.RSRP_ehq, tolerance.RSRP);
#endif
    return err;
}

int compare_pf01_sr(fapi_validate* vald, pucch_uci_ind_t& tv, uint8_t* payload)
{
    int err = 0;
    scf_fapi_sr_format_0_1_info_t* sr = reinterpret_cast<scf_fapi_sr_format_0_1_info_t*>(payload);
    err += FAPI_VALIDATE_U32_ERR(vald, sr->sr_indication, tv.SRindication);
    err += FAPI_VALIDATE_U32_ERR(vald, sr->sr_confidence_level, tv.SRconfidenceLevel);
    return err;
}

int compare_pf01_harq(fapi_validate* vald, pucch_uci_ind_t& tv, uint8_t* payload)
{
    scf_fapi_harq_info_t* harq = reinterpret_cast<scf_fapi_harq_info_t*>(payload);
    int err = 0;
    err += FAPI_VALIDATE_U32_ERR(vald, harq->num_harq, tv.NumHarq);
    err += FAPI_VALIDATE_U32_WARN(vald, harq->harq_confidence_level, tv.HarqconfidenceLevel);
    if (harq->num_harq > 0 && harq->num_harq == tv.NumHarq)
    {
        uint8_t* tv_HarqValue = tv.HarqValue.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, harq->harq_value, tv_HarqValue, harq->num_harq);
    }
    return err;
}

int compare_pf234_sr(fapi_validate* vald, pucch_uci_ind_t& tv, uint8_t* payload)
{
    int err = 0;
    scf_fapi_sr_format_2_3_4_info_t *sr = reinterpret_cast<scf_fapi_sr_format_2_3_4_info_t*>(payload);
    err += FAPI_VALIDATE_U32_ERR(vald, sr->sr_bit_len, tv.SrBitLen);
    if (sr->sr_bit_len > 0 && sr->sr_bit_len == tv.SrBitLen)
    {
        int nbytes = (sr->sr_bit_len + 7) / 8;
        uint8_t* tv_SrPayload = tv.SrPayload.data();
        err += FAPI_VALIDATE_BYTES_ERR(vald, sr->sr_payload, tv_SrPayload, nbytes);
    }
    return err;
}

int compare_pf234_harq(fapi_validate* vald, pusch_uci_ind_t& tv, uint8_t* payload, bool early_harq_enabled)
{
    scf_fapi_harq_format_2_3_4_info_t* harq = reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(payload);
    int err = 0;

#ifdef SCF_FAPI_10_04
    if((early_harq_enabled) && (tv.isEarlyHarq))
        err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_detection_status, tv.HarqDetStatus_earlyHarq);
    else
        err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_detection_status, tv.HarqDetectionStatus);
#else
    if(harq->harq_bit_len > CUPHY_N_MAX_UCI_BITS_RM)
    {
        err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_crc, tv.HarqCrc);
    }
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_bit_len, tv.HarqBitLen);
    if (harq->harq_bit_len > 0 && harq->harq_bit_len == tv.HarqBitLen)
    {
        int nbytes = (harq->harq_bit_len + 7) / 8;
#ifdef SCF_FAPI_10_04
        uint8_t* tv_HarqPayload = nullptr;
        if((early_harq_enabled) && (tv.isEarlyHarq))
            tv_HarqPayload = tv.HarqPayload_earlyHarq.data();
        else
            tv_HarqPayload = tv.HarqPayload.data();
#else
        uint8_t* tv_HarqPayload = tv.HarqPayload.data();
#endif
        err += FAPI_VALIDATE_BYTES_WARN(vald, harq->harq_payload, tv_HarqPayload, nbytes);
    }

    return err;
}

bool is_harq_crc_pass(scf_fapi_harq_format_2_3_4_info_t* harq)
{
    bool crc_pass = false;

#ifdef SCF_FAPI_10_04
    // FAPI 10.04: 1 - CRC pass; 2 - CRC fail; 3 - DTX; 4 - No DTX; 5 - DTX not checked
    if(harq->harq_detection_status == 1)
    {
        crc_pass = true;
    }
#else
    // FAPI 10.02: 0 - CRC pass; 1 - CRC fail; 3 - Not present
    if(harq->harq_crc == 0)
    {
        crc_pass = true;
    }
#endif

    return crc_pass;
}

int compare_pf234_harq(fapi_validate* vald, pucch_uci_ind_t& tv, uint8_t* payload)
{
    scf_fapi_harq_format_2_3_4_info_t* harq = reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(payload);
    int err = 0;

#ifdef SCF_FAPI_10_04
    err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_detection_status, tv.HarqDetectionStatus);
#else
    err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_crc, tv.HarqCrc);
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, harq->harq_bit_len, tv.HarqBitLen);
    if (harq->harq_bit_len > 0 && harq->harq_bit_len == tv.HarqBitLen && is_harq_crc_pass(harq))
    {
        int nbytes = (harq->harq_bit_len + 7) / 8;
        uint8_t* tv_HarqPayload = tv.HarqPayload.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, harq->harq_payload, tv_HarqPayload, nbytes);
    }

    return err;
}

int compare_pf234_csi_p1(fapi_validate* vald, csi_part_t& tv, uint8_t* payload)
{
    scf_fapi_csi_part_1_t* csi = reinterpret_cast<scf_fapi_csi_part_1_t*>(payload);
    int err = 0;
#ifdef SCF_FAPI_10_04
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part1_detection_status, tv.DetectionStatus);
#else
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_1_crc, tv.Crc);
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_1_bit_len, tv.BitLen);
    if (csi->csi_part_1_bit_len > 0 && csi->csi_part_1_bit_len == tv.BitLen)
    {
        int nbytes = (csi->csi_part_1_bit_len + 7) / 8;
        uint8_t* tv_CsiPartPayload = tv.Payload.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, csi->csi_part_1_payload, tv_CsiPartPayload, nbytes);
    }

    return err;
}

int compare_pusch_csi_p1(fapi_validate* vald, csi_part_t& tv, uint8_t* payload)
{
    scf_fapi_csi_part_1_t* csi = reinterpret_cast<scf_fapi_csi_part_1_t*>(payload);
    int err = 0;
#ifdef SCF_FAPI_10_04
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part1_detection_status, tv.DetectionStatus);
#else
    if(csi->csi_part_1_bit_len > CUPHY_N_MAX_UCI_BITS_RM)
    {
        err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_1_crc, tv.Crc);
    }
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_1_bit_len, tv.BitLen);
    if (csi->csi_part_1_bit_len > 0 && csi->csi_part_1_bit_len == tv.BitLen)
    {
        int nbytes = (csi->csi_part_1_bit_len + 7) / 8;
        uint8_t* tv_CsiPartPayload = tv.Payload.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, csi->csi_part_1_payload, tv_CsiPartPayload, nbytes);
    }

    return err;
}

int compare_pf234_csi_p2(fapi_validate* vald, csi_part_t& tv, uint8_t* payload)
{
    scf_fapi_csi_part_2_t* csi = reinterpret_cast<scf_fapi_csi_part_2_t*>(payload);
    int err = 0;
#ifdef SCF_FAPI_10_04
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part2_detection_status, tv.DetectionStatus);
#else
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_2_crc, tv.Crc);
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_2_bit_len, tv.BitLen);
    if (csi->csi_part_2_bit_len > 0 && csi->csi_part_2_bit_len == tv.BitLen)
    {
        int nbytes = (csi->csi_part_2_bit_len + 7) / 8;
        uint8_t* tv_CsiPartPayload = tv.Payload.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, csi->csi_part_2_payload, tv_CsiPartPayload, nbytes);
    }

    return err;
}

int compare_pusch_csi_p2(fapi_validate* vald, csi_part_t& tv, uint8_t* payload)
{
    scf_fapi_csi_part_2_t* csi = reinterpret_cast<scf_fapi_csi_part_2_t*>(payload);
    int err = 0;
#ifdef SCF_FAPI_10_04
    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part2_detection_status, tv.DetectionStatus);
#else
    if(csi->csi_part_2_bit_len > CUPHY_N_MAX_UCI_BITS_RM)
    {
        err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_2_crc, tv.Crc);
    }
#endif

    err += FAPI_VALIDATE_U32_ERR(vald, csi->csi_part_2_bit_len, tv.BitLen);
    if (csi->csi_part_2_bit_len > 0 && csi->csi_part_2_bit_len == tv.BitLen)
    {
        int nbytes = (csi->csi_part_2_bit_len + 7) / 8;
        uint8_t* tv_CsiPartPayload = tv.Payload.data();
        err += FAPI_VALIDATE_BYTES_WARN(vald, csi->csi_part_2_payload, tv_CsiPartPayload, nbytes);
    }

    return err;
}

int scf_fapi_handler::parse_pf01_sr(int cell_id, uci_pdu_format_t format, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_sr_format_0_1_info_t* sr = reinterpret_cast<scf_fapi_sr_format_0_1_info_t*>(payload);
    offset += sizeof(scf_fapi_sr_format_0_1_info_t);

    if(sr->sr_indication == 1)
    {
        thrputs[cell_id].sr++;
    }
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: cell_id={} SR: sr_indication={} sr_confidence_level={} num_sr={}",
            cell_id, sr->sr_indication, sr->sr_confidence_level, thrputs[cell_id].sr.load());

    return offset;
}

int scf_fapi_handler::parse_pf01_harq(int cell_id, uci_pdu_format_t format, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_harq_info_t* harq = reinterpret_cast<scf_fapi_harq_info_t*>(payload);
    offset += sizeof(scf_fapi_harq_info_t) + harq->num_harq;
    if(harq->num_harq != 0)
    {
        thrputs[cell_id].harq++;
    }

    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: cell_id={} HARQ: harq_num={}", cell_id, harq->num_harq);
    if((configs->get_conformance_test_params()->conformance_test_enable)&&
    (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
    {

    // Conformance test
        for(int i = 0 ; i < harq->num_harq; i++)
        {
            if(format == UCI_PDU_PF0)
            {
                //NVLOGC_FMT(TAG, "SCF_FAPI_UCI_INDICATION: cell_id={} harq_confidence_level {} harq_value{}", cell_id, harq->harq_confidence_level, harq->harq_value[i]);
                if((harq->harq_confidence_level == 0)||(harq->harq_confidence_level == 0xFF))
                {
                    switch(harq->harq_value[i])
                    {
#ifndef SCF_FAPI_10_04
                        case 0:
                        conformance_test_stats->get_pf0_stats().pf0_ack++;
                        break;
                        case 1:
                            conformance_test_stats->get_pf0_stats().pf0_nack++;
                            break;
                        case 2:
                            conformance_test_stats->get_pf0_stats().pf0_dtx++;
                            break;
#else
                        case 1:
                        conformance_test_stats->get_pf0_stats().pf0_ack++;
                        break;
                        case 0:
                            conformance_test_stats->get_pf0_stats().pf0_nack++;
                            break;
                        case 2:
                            conformance_test_stats->get_pf0_stats().pf0_dtx++;
                            break;
#endif
                    }
                }
                else if (harq->harq_confidence_level == 1)
                {
                    conformance_test_stats->get_pf0_stats().pf0_dtx++;
                }
            }
            else if(format == UCI_PDU_PF1)
            {
                if((harq->harq_confidence_level == 0)||(harq->harq_confidence_level == 0xFF))
                {
                    switch(harq->harq_value[i])
                    {
#ifndef SCF_FAPI_10_04
                        case 0:
                            conformance_test_stats->get_pf1_stats().pf1_ack_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 1)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_nack_to_ack_bits++;
                            }
                        break;
                        case 1:
                            conformance_test_stats->get_pf1_stats().pf1_nack_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 0)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_ack_to_nack_bits++;
                            }

                        break;
                        case 2:
                            conformance_test_stats->get_pf1_stats().pf1_dtx_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 0)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_ack_to_dtx_bits++;
                            }
                            else
                            {
                                conformance_test_stats->get_pf1_stats().pf1_nack_to_dtx_bits++;
                            }
                        break;
#else
                        case 1:
                            conformance_test_stats->get_pf1_stats().pf1_ack_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 0)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_nack_to_ack_bits++;
                            }
                        break;
                        case 0:
                            conformance_test_stats->get_pf1_stats().pf1_nack_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 1)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_ack_to_nack_bits++;
                            }

                        break;
                        case 2:
                            conformance_test_stats->get_pf1_stats().pf1_dtx_bits++;
                            if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 1)
                            {
                                conformance_test_stats->get_pf1_stats().pf1_ack_to_dtx_bits++;
                            }
                            else
                            {
                                conformance_test_stats->get_pf1_stats().pf1_nack_to_dtx_bits++;
                            }
                        break;
#endif
                    }

                }
                else
                {
                    conformance_test_stats->get_pf1_stats().pf1_dtx_bits++;
#ifndef SCF_FAPI_10_04
                    if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 0)
#else
                    if((conformance_test_stats->get_pf1_stats().pf1_ack_nack_pattern & 1 << i) == 1)
#endif
                    {
                        conformance_test_stats->get_pf1_stats().pf1_ack_to_dtx_bits++;
                    }
                    else
                    {
                        conformance_test_stats->get_pf1_stats().pf1_nack_to_dtx_bits++;
                    }
                }
            }
        }
    }

    return offset;
}

int scf_fapi_handler::parse_pf234_sr(int cell_id, uci_pdu_format_t format, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_sr_format_2_3_4_info_t* sr = reinterpret_cast<scf_fapi_sr_format_2_3_4_info_t*>(payload);
    offset += sizeof(scf_fapi_sr_format_2_3_4_info_t) + (sr->sr_bit_len + 7) / 8;

    if(sr->sr_bit_len > 0)
    {
        thrputs[cell_id].sr++;
    }
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: cell_id={} SR: sr_bit_len={} num_sr={}",
         cell_id, static_cast<unsigned short>(sr->sr_bit_len), thrputs[cell_id].sr.load());

    return offset;
}

int scf_fapi_handler::parse_pf234_harq(int cell_id, uci_pdu_format_t format, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_harq_format_2_3_4_info_t* harq = reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(payload);
    offset += sizeof(scf_fapi_harq_format_2_3_4_info_t) + (harq->harq_bit_len + 7) / 8;

    thrputs[cell_id].harq++;

#ifdef SCF_FAPI_10_04
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: harq_bit_len={} HARQ detection status {}", static_cast<int>(harq->harq_bit_len), harq->harq_detection_status);
#else
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: harq_bit_len={} harq_crc={}", static_cast<int>(harq->harq_bit_len), static_cast<unsigned short>(harq->harq_crc));
#endif

    // Conformance test
    if((harq->harq_bit_len != 0)&&(configs->get_conformance_test_params()->conformance_test_enable) &&
    (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
    {
        if(format == UCI_PDU_PF2) //PUCCH format 2
        {
#ifdef  SCF_FAPI_10_04
            if((harq->harq_detection_status == 1) || (harq->harq_detection_status == 4))
#else
            if((harq->harq_crc == 0) ||(harq->harq_crc == 2))
#endif
            {
                for(int i = 0; i < harq->harq_bit_len; i++)
                {
                    if(harq->harq_payload[i >> 3] & (1 << (i & 7)))
                        conformance_test_stats->get_pf2_stats().pf2_harq_ack_bits++;
                    else
                        conformance_test_stats->get_pf2_stats().pf2_harq_nack_bits++;
                }
            }
            // Need to check what should be done if CRC fails
        }
    }

    return offset;
}

int scf_fapi_handler::parse_pf234_csi(int cell_id, uci_pdu_format_t format, int csi_id, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_csi_part_1_t* csi_1 = reinterpret_cast<scf_fapi_csi_part_1_t*>(payload);
    offset += sizeof(scf_fapi_csi_part_1_t) + (csi_1->csi_part_1_bit_len + 7) / 8;

#ifdef SCF_FAPI_10_04
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: PUCCH F234 CSI Part{} detection status {} bit len {}", csi_id + 1, csi_1->csi_part1_detection_status,static_cast<int>(csi_1->csi_part_1_bit_len));
#endif

    if (csi_id == 0)
    {
        thrputs[cell_id].csi1++;
    }
    else
    {
        thrputs[cell_id].csi2++;
    }

    // Conformance test
    if((configs->get_conformance_test_params()->conformance_test_enable)&&
    (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
    {
#ifdef SCF_FAPI_10_04
        if(csi_1->csi_part1_detection_status == 2)
#else
        if(csi_1->csi_part_1_crc== 1)
#endif
        {
            if(format == UCI_PDU_PF2)
            {
                conformance_test_stats->get_pf2_stats().pf2_bler++;
            }
            else if(format == UCI_PDU_PF3)
            {
                conformance_test_stats->get_pf3_stats().pf3_bler++;
            }
        }
#ifdef SCF_FAPI_10_04
        else if(csi_1->csi_part1_detection_status == 1)
#else
        else if(csi_1->csi_part_1_crc == 0)
#endif
        {
            if(format == UCI_PDU_PF2)
            {
                conformance_test_stats->get_pf2_stats().pf2_csi_success++;
            }
            else if (format == UCI_PDU_PF3)
            {

                conformance_test_stats->get_pf3_stats().pf3_csi_success++;
            }
        }
    }
    return offset;
}

int scf_fapi_handler::parse_pusch_csi(int cell_id, int csi_id, uint8_t* payload)
{
    int offset = 0;
    scf_fapi_csi_part_1_t* csi_1 = reinterpret_cast<scf_fapi_csi_part_1_t*>(payload);
    offset += sizeof(scf_fapi_csi_part_1_t) + (csi_1->csi_part_1_bit_len + 7) / 8;

#ifdef SCF_FAPI_10_04
    NVLOGD_FMT(TAG, "SCF_FAPI_UCI_INDICATION: PUSCH CSI Part{} detection status {} bit len {}", csi_id + 1, csi_1->csi_part1_detection_status,static_cast<int>(csi_1->csi_part_1_bit_len));
#endif

    if (csi_id == 0)
    {
        thrputs[cell_id].csi1++;
    }
    else
    {
        thrputs[cell_id].csi2++;
    }
    return offset;
}

void scf_fapi_handler::validate_timing_srs_indication(int cell_id, uint64_t handle_start_time, scf_fapi_srs_ind_t& resp)
{
    // TODO fix alpha and beta to match l2adapter
    uint64_t t0_slot = sfn_to_tai(resp.sfn, resp.slot, handle_start_time, 0, 0, 1);
    // T0 + 11
    if(handle_start_time-t0_slot > configs->get_srs_late_deadline_ns())
    {
        cell_summary[cell_id].srs_ind.late++;
        thrputs[cell_id].srs_ind.late++;
    }
    else if(handle_start_time-t0_slot < configs->get_srs_early_deadline_ns())
    {
        cell_summary[cell_id].srs_ind.early++;
        thrputs[cell_id].srs_ind.early++;
    }
    else
    {
        cell_summary[cell_id].srs_ind.ontime++;
        thrputs[cell_id].srs_ind.ontime++;
    }
    //NVLOGC_FMT(TAG, "testMAC received SRS.IND for {}.{} at {} ns T0 of slot {} ns, diff {} ns", (int)resp.sfn, (int)resp.slot, handle_start_time, t0_slot, handle_start_time-t0_slot);
}

void scf_fapi_handler::validate_indication_timing(int cell_id, uint64_t handle_start_time, int sfn, int slot, int deadline_ns, timing_t& summary_timing, timing_t& thrputs_timing)
{
    // TODO fix alpha and beta to match l2adapter
    uint64_t t0_slot = sfn_to_tai(sfn, slot, handle_start_time, 0, 0, 1);
    if(handle_start_time-t0_slot > deadline_ns)
    {
        summary_timing.late++;
        thrputs_timing.late++;
    }
    else
    {
        summary_timing.ontime++;
        thrputs_timing.ontime++;
    }
    // if(handle_start_time-t0_slot > 1900000)
    //     NVLOGC_FMT(TAG, "testMAC received UCI for {}.{} at {} ns T0 of slot {} ns, diff {} ns", (int)resp.sfn, (int)resp.slot, handle_start_time, t0_slot, handle_start_time-t0_slot);
}

int scf_fapi_handler::handle_uci_indication(int cell_id, scf_fapi_uci_ind_t& resp, fapi_validate& vald)
{
    auto handle_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    bool is_early_harq = false;
    int msg_id = SCF_FAPI_UCI_INDICATION;
#ifdef SCF_FAPI_10_04
    bool early_harq_enabled = (configs->get_indication_per_slot()[2] == 2)? true: false;
#endif
    NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_UCI_INDICATION num_ucis={}",
            static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<unsigned>(resp.num_ucis));

    int pusch_pdu_id = 0;
    int pucch_pdu_id = 0;
    fapi_req_t* pusch_req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUSCH);
    fapi_req_t* pucch_req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUCCH);

    bool should_count_pusch_slot = false;
    if (pusch_req != nullptr && pusch_req->tv_data->pusch_tv.data_size == 0)
    {
        // The PUSCH channel only contains UCI on PUSCH, no TB data
        should_count_pusch_slot = true;
    }

    vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);

    // UL time/slot cost validation
    slot_timing_t &timing = cell_data[cell_id].slot_timing[resp.slot];
    int64_t ul_ns = nvlog_get_interval(&timing.ts_ul_tti);

    uci_time->add(uci_time, ul_ns);

    if (resp.sfn != timing.ss.u16.sfn) {
        // Had delayed more than a frame
        FAPI_VALIDATE_TEXT_ERR(&vald, "UL cost more than a frame", timing.ss.u16.sfn,
                timing.ss.u16.slot);
    } else {
        sfn_slot_t ss_curr = ss_tick.load();
        int ul_slot = (ss_curr.u16.slot + slots_per_frame - timing.ss.u16.slot) % slots_per_frame;
        if (lp->get_mmimo_enabled() == 0 && (lp->get_mmimo_static_dynamic_enabled() == 0))
        {
            if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW) {
                FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW);
            }
        }
        else
        {
            if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R) {
                FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R);
            }
        }
    }

    uint32_t uci_offset = 0; // UCI PDU offset
    for(int i = 0; i < resp.num_ucis; i++)
    {
        scf_fapi_uci_pdu_t* pdu = reinterpret_cast<scf_fapi_uci_pdu_t*>(resp.payload + uci_offset);
        uci_offset += pdu->pdu_size;

        // PDU parse offset
        int offset = 0;
        if (pdu->pdu_type == 0)
        {
            if (vald.pdu_start(pusch_pdu_id, channel_type_t::PUSCH, pusch_req) < 0)
            {
                continue;
            }
        }
        else if (pdu->pdu_type < 3)
        {
            if (vald.pdu_start(pucch_pdu_id, channel_type_t::PUCCH, pucch_req) < 0)
            {
                continue;
            }
        }
        else
        {
            FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid pdu_type %d", pdu->pdu_type);
            continue;
        }

        switch(pdu->pdu_type)
        {
            case 0: // SCF222: UCI indication PDU carried on PUSCH, see Section 3.4.9.1
            {
                scf_fapi_uci_pusch_pdu_t* uci_pusch = reinterpret_cast<scf_fapi_uci_pusch_pdu_t*>(pdu->payload);
                pusch_tv_data_t* tv_data = get_pusch_tv_pdu(pusch_req, uci_pusch->handle, PUSCH_BITMAP_UCI);
#ifdef SCF_FAPI_10_04
                if ((should_count_pusch_slot)&&
                    (!((early_harq_enabled) && ((uci_pusch->pdu_bitmap == 0x02) && (tv_data != nullptr && tv_data->uci_ind.isEarlyHarq == 1)))))
#else
                if (should_count_pusch_slot)
#endif
                {
                    should_count_pusch_slot = false;
                    thrputs[cell_id].slots[PUSCH]++;
                }

                if (configs->app_mode == 0 && tv_data == nullptr)
                {
                    FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUSCH_UCI pdu_id %d", pusch_pdu_id);
                    vald.pdu_ended(pusch_pdu_id++, -1);
                    break;
                }

                // no check for negative case
                if (configs->app_mode == 0 && tv_data->tbErr != 0)
                {
                    //FAPI_VALIDATE_TEXT_WARN(&vald, "Negative PUSCH_UCI pdu_id %d", pusch_pdu_id);
                    NVLOGD_FMT(TAG, "NO check for PUSCH_UCI negative case (tbErr = 1).");
                    vald.pdu_ended(pusch_pdu_id++, -1);
                    break;
                }

                offset += sizeof(scf_fapi_uci_pusch_pdu_t);
                if (configs->app_mode == 0) {
                    FAPI_VALIDATE_U32_ERR(&vald, uci_pusch->rnti, tv_data->tbpars.nRnti, 0);
                }

//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: PUSCH PDU Handle={} rnti={} bitmap=0x{:x}", uci_pusch->handle, uci_pusch->rnti,uci_pusch->pdu_bitmap);
//#ifdef SCF_FAPI_10_04
//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: UCI on PUSCH: ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}",uci_pusch->measurement.ul_sinr_metric,
//                uci_pusch->measurement.timing_advance, uci_pusch->measurement.timing_advance_ns, uci_pusch->measurement.rssi, uci_pusch->measurement.rsrp);
//#endif

                // Validate UCI_on_PUSCH UL measurement parameters
                if (configs->app_mode == 0) {
#ifdef SCF_FAPI_10_04
                    if((!((early_harq_enabled) && ((uci_pusch->pdu_bitmap == 0x02  ) && (tv_data != nullptr && tv_data->uci_ind.isEarlyHarq == 1)))))
                    {
                        NVLOGD_FMT(TAG, " Comparing Full slot UCI Indication Measurements");
                        compare_ul_measurement(&vald, tv_data->uci_ind.meas, reinterpret_cast<uint8_t*>(&uci_pusch->measurement), pdu->pdu_type);
                    }
                    else if((early_harq_enabled) && ((uci_pusch->pdu_bitmap & 0x02) && (tv_data != nullptr && tv_data->uci_ind.isEarlyHarq == 1)))
                    {
                        bool isIndEarlyHarqEnabled = ((uci_pusch->pdu_bitmap & 0x02) && !((uci_pusch->pdu_bitmap & 0x04) || (uci_pusch->pdu_bitmap & 0x08))); // HARQ Only is Present but no CSI Part 1 and CSI Part2
                        if (isIndEarlyHarqEnabled) {
                            NVLOGD_FMT(TAG, "isIndEarlyHarqEnabled = {}, Comparing Early HARQ UCI Indication Measurements", isIndEarlyHarqEnabled);
                            compare_ul_measurement_ehq(&vald, tv_data->uci_ind.meas, reinterpret_cast<uint8_t*>(&uci_pusch->measurement), pdu->pdu_type);
                        } else {
                            NVLOGD_FMT(TAG, "isIndEarlyHarqEnabled = {}, Comparing Full slot UCI Indication Measurements", isIndEarlyHarqEnabled);
                            compare_ul_measurement(&vald, tv_data->uci_ind.meas, reinterpret_cast<uint8_t*>(&uci_pusch->measurement), pdu->pdu_type);
                        }
                        
                    }
#else
                    compare_ul_measurement(&vald, tv_data->uci_ind.meas, &uci_pusch->ul_cqi, pdu->pdu_type);
#endif
                }

                if(uci_pusch->pdu_bitmap & 0x02) // Bit 1: HARQ
                {
                    if (configs->app_mode == 0) {
#ifdef SCF_FAPI_10_04
                        compare_pf234_harq(&vald, tv_data->uci_ind, pdu->payload + offset, early_harq_enabled);
#else
                        compare_pf234_harq(&vald, tv_data->uci_ind, pdu->payload + offset, false);
#endif
                    }
                    offset += parse_pf234_harq(cell_id, UCI_PDU_PUSCH, pdu->payload + offset);
                }

                if(uci_pusch->pdu_bitmap & 0x04) // Bit 2: CSI part 1
                {
                    if (configs->app_mode == 0) {
                        compare_pusch_csi_p1(&vald, tv_data->uci_ind.csi_parts[0], pdu->payload + offset);
                    }
                    offset += parse_pusch_csi(cell_id, 0, pdu->payload + offset);
                }

                // Add CSI Part 2 if exists
                if(uci_pusch->pdu_bitmap & 0x08) // Bit 3: CSI part 2
                {
                    if (configs->app_mode == 0) {
                        compare_pusch_csi_p2(&vald, tv_data->uci_ind.csi_parts[1], pdu->payload + offset);
                    }
                    offset += parse_pusch_csi(cell_id, 1, pdu->payload + offset);
                }
                if (configs->app_mode == 0) {
                    vald.pdu_ended(pusch_pdu_id++, tv_data->uci_ind.idxInd);
                }
#ifdef SCF_FAPI_10_04
                if(early_harq_enabled && uci_pusch->pdu_bitmap == 0x02)
                {
                    is_early_harq = true;
                }
#endif
            }
            break;
            case 1: // SCF222: UCI indication PDU carried on PUCCH Format 0 or 1, see Section 3.4.9.2
            {
                scf_fapi_pucch_format_hdr* uci_pucch_01 = reinterpret_cast<scf_fapi_pucch_format_hdr*>(pdu->payload);
                pucch_tv_data_t* tv_data = get_pucch_tv_pdu(pucch_req, uci_pucch_01->handle);
                if (tv_data == nullptr)
                {
                    FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUCCH_PF01 pdu_id %d", pucch_pdu_id);
                    vald.pdu_ended(pucch_pdu_id++, -1);
                    break;
                }

                offset += sizeof(scf_fapi_pucch_format_hdr);

                FAPI_VALIDATE_U32_ERR(&vald, uci_pucch_01->rnti, tv_data->RNTI, 0);

//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: PUCCH Format 0 or 1 PDU Handle = {}", uci_pucch_01->handle);
//#ifdef SCF_FAPI_10_04
//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: PUCCH Format 0/1 ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}",uci_pucch_01->measurement.ul_sinr_metric,
//                uci_pucch_01->measurement.timing_advance, uci_pucch_01->measurement.timing_advance_ns, uci_pucch_01->measurement.rssi, uci_pucch_01->measurement.rsrp);
//#endif

                // Validate PUCCH PF 0/1 UL measurement parameters
#ifdef SCF_FAPI_10_04
                compare_ul_measurement(&vald, tv_data->uci_ind.meas, reinterpret_cast<uint8_t*>(&uci_pucch_01->measurement), pdu->pdu_type);
#else
                compare_ul_measurement(&vald, tv_data->uci_ind.meas, &uci_pucch_01->ul_cqi, pdu->pdu_type);
#endif

                if(uci_pucch_01->pdu_bitmap & 0x01) // Bit 0: SR
                {
                    compare_pf01_sr(&vald, tv_data->uci_ind, pdu->payload + offset);
                    offset += parse_pf01_sr(cell_id, (uci_pdu_format_t) uci_pucch_01->pucch_format, pdu->payload + offset);
                }
                if(uci_pucch_01->pdu_bitmap & 0x02) // Bit 1: HARQ
                {
                    compare_pf01_harq(&vald, tv_data->uci_ind, pdu->payload + offset);
                    offset += parse_pf01_harq(cell_id, (uci_pdu_format_t) uci_pucch_01->pucch_format, pdu->payload + offset);
                }
                vald.pdu_ended(pucch_pdu_id++, tv_data->uci_ind.idxInd);
#ifdef SCF_FAPI_10_04
                if(early_harq_enabled && uci_pucch_01->pdu_bitmap == 0x02)
                {
                    is_early_harq = true;
                }
#endif
            }
            break;
            case 2: // SCF222: UCI indication PDU carried on PUCCH Format 2, 3 or 4, see Section 3.4.9.3
            {
                scf_fapi_pucch_format_hdr* uci_pucch_234 = reinterpret_cast<scf_fapi_pucch_format_hdr*>(pdu->payload);
                pucch_tv_data_t* tv_data = get_pucch_tv_pdu(pucch_req, uci_pucch_234->handle);
                if (tv_data == nullptr)
                {
                    FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUCCH_PF234 pdu_id %d", pucch_pdu_id);
                    vald.pdu_ended(pucch_pdu_id++, -1);
                    break;
                }

                offset += sizeof(scf_fapi_pucch_format_hdr);

                FAPI_VALIDATE_U32_ERR(&vald, uci_pucch_234->rnti, tv_data->RNTI, 0);

//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: PUCCH Format 2,3 or 4 PDU Handle = {}", uci_pucch_234->handle);
//#ifdef SCF_FAPI_10_04
//                NVLOGD_FMT(TAG, "RECV: SCF_FAPI_UCI_INDICATION: PUCCH Format 2/3/4 ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}",uci_pucch_234->measurement.ul_sinr_metric,
//                uci_pucch_234->measurement.timing_advance, uci_pucch_234->measurement.timing_advance_ns, uci_pucch_234->measurement.rssi, uci_pucch_234->measurement.rsrp);
//#endif

                // Validate PUCCH PF 2/3/4 UL measurement parameters
#ifdef SCF_FAPI_10_04
                compare_ul_measurement(&vald, tv_data->uci_ind.meas, reinterpret_cast<uint8_t*>(&uci_pucch_234->measurement), pdu->pdu_type);
#else
                compare_ul_measurement(&vald, tv_data->uci_ind.meas, &uci_pucch_234->ul_cqi, pdu->pdu_type);
#endif

                if(uci_pucch_234->pdu_bitmap & 0x01) // Bit 0: SR
                {
                    compare_pf234_sr(&vald, tv_data->uci_ind, pdu->payload + offset);
                    offset += parse_pf234_sr(cell_id, (uci_pdu_format_t)(uci_pucch_234->pucch_format + 2), pdu->payload + offset);
                }
                if(uci_pucch_234->pdu_bitmap & 0x02) // Bit 1: HARQ
                {
                    compare_pf234_harq(&vald, tv_data->uci_ind, pdu->payload + offset);
                    offset += parse_pf234_harq(cell_id, (uci_pdu_format_t)(uci_pucch_234->pucch_format + 2), pdu->payload + offset);
                }
                if(uci_pucch_234->pdu_bitmap & 0x04) // Bit 2: CSI part 1
                {
                    compare_pf234_csi_p1(&vald, tv_data->uci_ind.csi_parts[0], pdu->payload + offset);
                    offset += parse_pf234_csi(cell_id, (uci_pdu_format_t)(uci_pucch_234->pucch_format + 2), 0, pdu->payload + offset);
                }

                // Add CSI Part 2 if exists
                if(uci_pucch_234->pdu_bitmap & 0x08) // Bit 3: CSI part 2
                {
                    compare_pf234_csi_p2(&vald, tv_data->uci_ind.csi_parts[1], pdu->payload + offset);
                    offset += parse_pf234_csi(cell_id, (uci_pdu_format_t)(uci_pucch_234->pucch_format + 2), 1, pdu->payload + offset);
                }
                vald.pdu_ended(pucch_pdu_id++, tv_data->uci_ind.idxInd);
#ifdef SCF_FAPI_10_04
                if(early_harq_enabled && uci_pucch_234->pdu_bitmap == 0x02)
                {
                    is_early_harq = true;
                }
#endif
            }
            break;
            default:
                NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: error UCI PDU type: {}", __FUNCTION__, static_cast<unsigned short>(pdu->pdu_type));
                break;
        }
    }

    vald.msg_ended();

    // only validate timing for UCI indications with pduBitmap & 0x2 and slot 4
    if(is_early_harq && resp.slot % 10 == 4)
    {
        // Early HARQ UCI Indication
        validate_indication_timing(cell_id, handle_start_time.count(), resp.sfn, resp.slot, configs->early_harq_deadline_ns, cell_summary[cell_id].uci, thrputs[cell_id].uci);
    }
    else
    {
        // Non early HARQ UCI Indication calculated in normal UL Indication
        validate_indication_timing(cell_id, handle_start_time.count(), resp.sfn, resp.slot, configs->uci_ind_deadline_ns, cell_summary[cell_id].uci, thrputs[cell_id].uci);
    }

    return 0;
}

void scf_fapi_handler::on_msg(nv_ipc_msg_t& msg)
{
    if(msg.msg_buf == NULL)
    {
        NVLOGI_FMT(TAG, "No more message");
        return;
    }

    scf_fapi_header_t *hdr = reinterpret_cast<scf_fapi_header_t*>(msg.msg_buf);
    int cell_id = msg.cell_id;
    if(hdr->handle_id != cell_id || cell_id < 0 || cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "RECV: cell_id error: msg_id=0x{:02X} handle_id={} cell_id={} pool={}", msg.msg_id, hdr->handle_id, cell_id, msg.data_pool);
        return;
    }

    /* Variables to validate multiple SRS.IND per slot */
    /* srsIndMap<cell_id, <sfn, slot, tvPdu_id>> */
    static std::map<int, std::tuple<uint16_t, uint16_t, int>> srsIndMap;

    fapi_validate vald(configs->validate_enable, configs->validate_log_opt);

    uint32_t body_offset = 0;
    for(uint8_t m = 0; m < hdr->message_count; m++)
    {
        scf_fapi_body_header_t* body = reinterpret_cast<scf_fapi_body_header_t*>(hdr->payload + body_offset);
        body_offset += body->length;

        // Check whether the msg_len is set correctly. Note: there's only 1 FAPI message in the iterator
        uint32_t head_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
        uint32_t body_len = body->length;
        if (msg.msg_len != head_len + body_len)
        {
            NVLOGW_FMT(TAG, "RECV: msg_len is not set correctly: cell_id={} msg_id=0x{:02X} msg_len={} head_len={} body_len={}",
                    msg.cell_id, msg.msg_id, msg.msg_len, head_len, body_len);
        }

        uint16_t msg_id = body->type_id;
        if(msg.data_pool == NV_IPC_MEMPOOL_CPU_MSG)
        {
            switch(msg_id)
            {
            case SCF_FAPI_CONFIG_RESPONSE: {
                auto resp = reinterpret_cast<scf_fapi_config_response_msg_t*>(body);
                NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} PHY_MAC_CELL_CONFIG_RESP: error_code={}", cell_id, msg_id, resp->msg_body.error_code);

                // If error_code != 0, will retry sending CONFIG.req when reconfig_timer timeout
                if (resp->msg_body.error_code == 0)
                {
                    // Successfully CONFIG.resp received, current cell CONFIG was done
                    NVLOGC_FMT(TAG, "cell_config: cell_id={} OK", cell_id);

                    if (first_init[cell_id] == 1) {
                        first_init[cell_id] = 0;
                    }
                    // Send START.req if not controlled by OAM
                    if(configs->oam_cell_ctrl_cmd == 0)
                    {
                        cell_start(cell_id);
                    }

                    if(cell_remap_event[cell_id])
                    {
                        cell_id_map[cell_id] = cell_id_map_tmp[cell_id];
                        cell_remap_event[cell_id] = false;
                    }

                    if (configs->cell_config_wait < 0)
                    {
                        // No wait for CONFIG.resp, skip
                        break;
                    }

                    if (current_config_cell_id.load() != cell_id)
                    {
                        NVLOGC_FMT(TAG, "cell_config: got cell_id={} response when current cell_id={}", current_config_cell_id.load(), cell_id);
                        // Skip, do not interrupt current cell config procedure
                        break;
                    }

                    config_retry_counter.store(0);
                    stop_reconfig_timer(cell_id);

                    if (configs->cell_config_wait > 0)
                    {
                        // Start the timer to wait for <cell_config_wait> ms then config next cell if exist
                        start_reconfig_timer(cell_id, configs->cell_config_wait);
                    }
                    else
                    {
                        // Immediately config next pending cell if exist
                        current_config_cell_id.store(-1);
                        int next_pending_cell = get_pending_config_cell_id();
                        if (next_pending_cell >= 0)
                        {
                            send_config_request(next_pending_cell);
                        }
                    }
                }
                else if(resp->msg_body.error_code == SCF_ERROR_CODE_MSG_INVALID_CONFIG)
                {
                    NVLOGC_FMT(TAG, "cell_config: cell_id={} failed: error_code={}", cell_id, resp->msg_body.error_code);
                    //exit(EXIT_SUCCESS);
                }
                else
                {
                    NVLOGC_FMT(TAG, "cell_config: cell_id={} failed: error_code={}", cell_id, resp->msg_body.error_code);
                }
            }
            break;

            case SCF_FAPI_STOP_INDICATION: {
                NVLOGI_FMT(TAG, "RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_STOP_INDICATION", cell_id, msg_id);
                cell_data[cell_id].fapi_state = fapi_state_t::IDLE;
                NVLOGC_FMT(TAG, "cell {} stopped", cell_id);
            }
            break;

            case SCF_FAPI_SLOT_INDICATION: {
                auto& resp = *reinterpret_cast<scf_fapi_slot_ind_t*>(body);

                sched_info_t sched_info;
                sched_info.ss = nv_ipc_get_sfn_slot(&msg);
                sched_info.ts = transport().get_ts_send(msg);

                int64_t nvipc_delay = std::chrono::system_clock::now().time_since_epoch().count() - sched_info.ts;
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: SLOT.ind global_tick={} nvipc_delay={}", sched_info.ss.u16.sfn, sched_info.ss.u16.slot, global_tick, nvipc_delay);

                if (sched_info_ring->enqueue(sched_info_ring, &sched_info) == 0)
                {
                    sem_post(&scheduler_sem);
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "Error: testMAC scheduler is full, please check SLOT.ind sending frequency");
                }

#ifdef AERIAL_CUMAC_ENABLE
                cumac_handler* _cumac_handler = get_cumac_handler_instance();
                if (_cumac_handler != nullptr)
                {
                    _cumac_handler->on_tick_event(sched_info);
                }
#endif
            }
            break;

            case SCF_FAPI_ERROR_INDICATION: {
                scf_fapi_error_ind_t& resp = *reinterpret_cast<scf_fapi_error_ind_t*>(body);
                thrputs[cell_id].error++;
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_ERROR_INDICATION: err_code=0x{:02X} err_msg_id=0x{:02X}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.err_code, resp.msg_id);

                switch (resp.msg_id)
                {
                    case SCF_FAPI_START_REQUEST:
                        // NVLOGW_FMT(TAG, "START failed: cell_id={} error_code={}", cell_id, resp.err_code);
                        // cell_data[cell_id].schedule_enable = false;
                        // cell_data[cell_id].fapi_state = fapi_state_t::IDLE;
                        break;
                    case SCF_FAPI_ERROR_INDICATION: {
                        if (resp.err_code == SCF_ERROR_CODE_RELEASED_HARQ_BUFFER_INFO) {
                            scf_fapi_error_ind_with_released_harq_buffer_ext_t* msg_body =
                                reinterpret_cast<scf_fapi_error_ind_with_released_harq_buffer_ext_t*>(body);
                            NVLOGI_FMT(TAG, "SFN {}.{} cell_id={} num_released_rscs={}", static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, static_cast<unsigned>(msg_body->num_released_rscs));
                            for (int i = 0; i < msg_body->num_released_rscs; i++) {
                                NVLOGD_FMT(TAG, "RNTI={} HarqID={} SFN={} Slot={}", static_cast<unsigned>(msg_body->released_harq_buffers[i].rnti), static_cast<unsigned>(msg_body->released_harq_buffers[i].harq_pid), static_cast<unsigned>(msg_body->released_harq_buffers[i].sfn), static_cast<unsigned>(msg_body->released_harq_buffers[i].slot));
                            }
                        } else {
                            const char* err_str = "UNKNOWN";
                            switch (static_cast<scf_fapi_error_codes_t>(resp.err_code)) {
                                case SCF_ERROR_CODE_L1_P1_EXIT_ERROR:       err_str = "L1_P1_EXIT"; break;
                                case SCF_ERROR_CODE_L1_P2_EXIT_ERROR:       err_str = "L1_P2_EXIT"; break;
                                case SCF_ERROR_CODE_PTP_SVC_ERROR:          err_str = "PTP_SVC_ERROR"; break;
                                case SCF_ERROR_CODE_PTP_SYNCED:             err_str = "PTP_SYNCED"; break;
                                case SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR: err_str = "RHOCP_PTP_EVENTS_ERROR"; break;
                                case SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED: err_str = "RHOCP_PTP_EVENTS_SYNCED"; break;
                                default: break;
                            }
                            NVLOGI_FMT(TAG, "SFN {}.{} cell_id={} msg_id=0x{:02X} err_code=0x{:02X} ({})", static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, static_cast<unsigned>(resp.msg_id), static_cast<unsigned>(resp.err_code), err_str);
                        }
                        break;
                    }
                    default:
                    break;
                }
            }
            break;
            case SCF_FAPI_CRC_INDICATION: {
                auto& resp = *reinterpret_cast<scf_fapi_crc_ind_t*>(body);
                auto handle_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_CRC_INDICATION: num_crc={}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<int>(resp.num_crcs));

                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                vald.set_fapi_req(get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUSCH));

                // UL time/slot cost validation
                slot_timing_t &timing = cell_data[cell_id].slot_timing[resp.slot];
                int64_t ul_ns = nvlog_get_interval(&timing.ts_ul_tti);
                crc_time->add(crc_time, ul_ns);

                if (resp.sfn != timing.ss.u16.sfn) {
                    // Had delayed more than a frame
                    FAPI_VALIDATE_TEXT_ERR(&vald, "UL cost more than a frame", timing.ss.u16.sfn,
                            timing.ss.u16.slot);
                } else {
                    sfn_slot_t ss_curr = ss_tick.load();
                    int ul_slot = (ss_curr.u16.slot + slots_per_frame - timing.ss.u16.slot) % slots_per_frame;
                    if (lp->get_mmimo_enabled() == 0 && (lp->get_mmimo_static_dynamic_enabled() == 0))
                    {
                        if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW) {
                            FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW);
                        }
                    }
                    else
                    {
                        if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R) {
                            FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R);
                        }
                    }
                }

                uint8_t* next = reinterpret_cast<uint8_t*>(resp.crc_info);
                for (uint16_t i = 0; i < resp.num_crcs; i++)
                {
                    validate_crc_ind(cell_id, resp.sfn, resp.slot, i, reinterpret_cast<scf_fapi_crc_info_t*>(next), vald);

                    scf_fapi_crc_info_t& crc = *(reinterpret_cast<scf_fapi_crc_info_t*>(next));
                    next += sizeof(scf_fapi_crc_info_t) + (crc.num_cb + 7) / 8;

                    scf_fapi_crc_end_info_t *end = reinterpret_cast<scf_fapi_crc_end_info_t*>(next);
                    next += sizeof(scf_fapi_crc_end_info_t);

                    NVLOGD_FMT(TAG, "RECV: SCF_FAPI_CRC_INDICATION: MAC CRC[{}]: RNTI={} HarqID={} TbCrcStatus={} NumCb={} Handle={}", i, static_cast<unsigned short>(crc.rnti), static_cast<unsigned short>(crc.harq_id), static_cast<unsigned short>(crc.tb_crc_status), static_cast<unsigned short>(crc.num_cb), static_cast<unsigned>(crc.handle));

//#ifdef SCF_FAPI_10_04
//                    NVLOGD_FMT(TAG, "RECV: SCF_FAPI_CRC_INDICATION ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}",end->measurement.ul_sinr_metric,
//                    end->measurement.timing_advance, end->measurement.timing_advance_ns, end->measurement.rssi, end->measurement.rsrp);
//#endif
                }
                vald.msg_ended();
                validate_indication_timing(cell_id, handle_start_time.count(), resp.sfn, resp.slot, configs->ul_ind_deadline_ns, cell_summary[cell_id].ul_ind, thrputs[cell_id].ul_ind);
            }
            break;

            case SCF_FAPI_RACH_INDICATION: {
                auto& resp = *reinterpret_cast<scf_fapi_rach_ind_t*>(body);
                auto handle_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());

                if(resp.num_pdus == 0 || resp.pdu_info[0].num_preamble == 0)
                {
                    NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RACH_INDICATION num_pdus={} empty preamble",
                            static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus);
                    break;
                }

                // FAPI validation
                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                fapi_req_t* req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PRACH);
                vald.set_fapi_req(req);
                if (req != nullptr)
                {
                    FAPI_VALIDATE_U32_ERR(&vald, resp.num_pdus, req->tv_data->prach_tv.data.size(), 0);
                }

                preamble_params_t preamble_prarms;
                get_pramble_params(cell_id, resp.sfn, resp.slot, &preamble_prarms);

                int num_detectedPrmb = 0;

                uint8_t* next = reinterpret_cast<uint8_t*>(resp.pdu_info);
                for(int pdu_id = 0; pdu_id < resp.num_pdus; pdu_id++)
                {
                    scf_fapi_prach_ind_pdu_t* rach_pdu = reinterpret_cast<scf_fapi_prach_ind_pdu_t*>(next);
                    next += sizeof(scf_fapi_prach_ind_pdu_t) + sizeof(scf_fapi_prach_preamble_info_t) * rach_pdu->num_preamble;

                    if(validate_rach_ind(cell_id, resp.sfn, resp.slot, pdu_id, rach_pdu, vald) == 0)
                    {
                        num_detectedPrmb += rach_pdu->num_preamble;
                    }
                    if(configs->get_conformance_test_params()->conformance_test_enable)
                    NVLOGI_FMT(TAG, "RECV: SCF_FAPI_RACH_INDICATION: pdu_id {}, num_preamble {}", pdu_id, rach_pdu->num_preamble);
                    for(int pream_id = 0; pream_id < rach_pdu->num_preamble; pream_id++)
                    {
                        scf_fapi_prach_preamble_info_t& preamble = rach_pdu->preamble_info[pream_id];

                        cell_data[cell_id].prach_state    = 1;
                        cell_data[cell_id].prach_prambIdx = preamble.preamble_index;

                        NVLOGD_FMT(TAG, "RECV: SCF_FAPI_RACH_INDICATION: MAC prmbIdx={}, prmbPwr={}, Timing adv={}", static_cast<unsigned short>(preamble.preamble_index), static_cast<unsigned>(preamble.preamble_power), static_cast<unsigned short>(preamble.timing_advance));
                        // if(preamble_prarms.tv_nTA == preamble.timing_advance && preamble_prarms.tv_nPreambPwr == preamble.preamble_power)
                        // {
                        // }
                        if((configs->get_conformance_test_params()->conformance_test_enable)&&
                        (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
                        {
                            if(preamble.preamble_index != conformance_test_stats->get_prach_stats().preamble_id)
                            {
                                float timing_advance = (float) preamble.timing_advance*16*64/2* 1/(480*1000*4096);
                                //Refer sec 4.2 TS 38.213 NTA = TA*16*64/2^mu
                                //Refer sec 4.3.1 T = NTA*Tc
                                //Refer sec 4.1 Tc = 1/(480*10^3*4096)
                                if((timing_advance > conformance_test_stats->get_prach_stats().timing_advance_range_high)||
                                (timing_advance < conformance_test_stats->get_prach_stats().timing_advance_range_low))
                                {
                                    conformance_test_stats->get_prach_stats().preamble_timing_offset_error++;
                                }

                                conformance_test_stats->get_prach_stats().preamble_error++;
                                continue;
                            }
                            conformance_test_stats->get_prach_stats().preamble_detected++;

                        }
                    }
                }
                thrputs[cell_id].prmb += num_detectedPrmb;
                scf_fapi_prach_preamble_info_t& preamble0 = resp.pdu_info[0].preamble_info[0];
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RACH_INDICATION: num_pdu={} numPrmb={}, preamble0: prmbIdx={} prmbPwr={} Timing adv={}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus, num_detectedPrmb, preamble0.preamble_index, static_cast<int>(preamble0.preamble_power), static_cast<unsigned>(preamble0.timing_advance));

                vald.msg_ended();
                validate_indication_timing(cell_id, handle_start_time.count(), resp.sfn, resp.slot, configs->prach_ind_deadline_ns, cell_summary[cell_id].prach_ind, thrputs[cell_id].prach_ind);
            }
            break;

            case SCF_FAPI_UCI_INDICATION: {
                handle_uci_indication(cell_id, *reinterpret_cast<scf_fapi_uci_ind_t*>(body), vald);
            }
            break;

            // case SCF_FAPI_SRS_INDICATION: {

            // }
            break;

            case SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION: {
                scf_fapi_rx_measurement_ind_t& resp = *reinterpret_cast<scf_fapi_rx_measurement_ind_t*>(body);
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION: num_meas={} meas_info[0].meas={}",
                    static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<unsigned>(resp.num_meas), resp.num_meas > 0 ? resp.meas_info[0].meas : 0);

                // FAPI validation
                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                fapi_req_t* req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUSCH);
                vald.set_fapi_req(req);
                if (configs->app_mode == 0 && req != nullptr)
                {
                    FAPI_VALIDATE_U32_WARN(&vald, resp.num_meas, req->tv_data->pusch_tv.data.size(), 0);
                }

                for (int n = 0; n < resp.num_meas; n++)
                {
                    uint16_t meas = resp.meas_info[n].meas;
                    NVLOGD_FMT(TAG, "SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION: meas[{}]={}", n, meas);

                    validate_pe_noise_interference_ind(cell_id, resp.sfn, resp.slot, n, &resp.meas_info[n], vald);
                }
                vald.msg_ended();
            }
            break;

            case SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION: {
                scf_fapi_rx_measurement_ind_t& resp = *reinterpret_cast<scf_fapi_rx_measurement_ind_t*>(body);
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION: num_meas={} meas_info[0].meas={}",
                    static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<unsigned>(resp.num_meas), resp.num_meas > 0 ? resp.meas_info[0].meas : 0);

                // FAPI validation
                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                fapi_req_t* req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUCCH);
                vald.set_fapi_req(req);
                if (req != nullptr)
                {
                    int pf234_pdu_num = get_pucch_tv_pf234_pdu_num(req);
                    FAPI_VALIDATE_U32_WARN(&vald, resp.num_meas, pf234_pdu_num, 0);
                }

                for (int n = 0; n < resp.num_meas; n++)
                {
                    uint16_t meas = resp.meas_info[n].meas;
                    NVLOGD_FMT(TAG, "SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION: meas[{}]={}", n, meas);

                    validate_pf234_interference_ind(cell_id, resp.sfn, resp.slot, n, &resp.meas_info[n], vald);
                }
                vald.msg_ended();
            }
            break;

            case SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION: {
                scf_fapi_prach_interference_ind_t& resp = *reinterpret_cast<scf_fapi_prach_interference_ind_t*>(body);
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION: num_meas={} meas_info[0].meas={}",
                    static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<unsigned>(resp.num_meas), resp.num_meas > 0 ? resp.meas_info[0].meas : 0);

                // FAPI validation
                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                fapi_req_t* req = get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PRACH);
                vald.set_fapi_req(req);
                if (req != nullptr)
                {
                    FAPI_VALIDATE_U32_WARN(&vald, resp.num_meas, req->tv_data->prach_tv.data.size(), 0);
                }

                for(int n = 0; n < resp.num_meas; n++)
                {
                    scf_fapi_prach_interference_t& meas_info = resp.meas_info[n];
                    NVLOGD_FMT(TAG, "SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION: meas[{}]={} phyCellId={} freqIndex={}",
                            n, static_cast<unsigned short>(meas_info.meas), static_cast<unsigned short>(meas_info.phyCellId), static_cast<unsigned short>(meas_info.freqIndex));

                    validate_prach_interference_ind(cell_id, resp.sfn, resp.slot, n, &resp.meas_info[n], vald);
                }
                vald.msg_ended();
            }
            break;

            case CV_MEM_BANK_CONFIG_RESPONSE: {
                configs->num_mem_bank_cv_config_req_sent[cell_id]--;
                NVLOGI_FMT(TAG,"CV_MEM_BANK_CONFIG_RESPONSE Received num_mem_bank_cv_config_req_sent[cell_id]={}", configs->num_mem_bank_cv_config_req_sent[cell_id]);
                if (configs->num_mem_bank_cv_config_req_sent[cell_id] == 0)
                {
                    NVLOGI_FMT(TAG,"cell_id={} first_init={} get_restart_option={}",cell_id,  first_init[cell_id] , configs->get_restart_option());
                    if (first_init[cell_id] || configs->get_restart_option() != 0) {
                        send_config_request(cell_id);
                    }
                }
            }
            break;
            case SCF_FAPI_SRS_INDICATION: {
                std::chrono::nanoseconds ts_msg_start = std::chrono::system_clock::now().time_since_epoch();
                scf_fapi_srs_ind_t& resp = *reinterpret_cast<scf_fapi_srs_ind_t*>(body);
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SRS.ind num_pdu={}",
                            static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus);
                uint16_t srs_ind_sfn       = resp.sfn;
                uint16_t srs_ind_slot      = resp.slot;

                int tvPdu_id = 0;
                auto iter = srsIndMap.find(cell_id);
                if (iter == srsIndMap.end())
                {
                    srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                    NVLOGD_FMT(TAG, "Cell id = {} Very first srs.ind {}{} tvPdu_id={}",cell_id, srs_ind_sfn, srs_ind_slot,tvPdu_id);
                }
                else
                {
                    if((srs_ind_sfn == std::get<0>(iter->second)) && (srs_ind_slot == std::get<1>(iter->second)))
                    {
                        tvPdu_id = std::get<2>(iter->second);
                        /* Grouped SRS.ind*/
                        NVLOGD_FMT(TAG, "Cell id = {} Grouped srs.ind tvPdu_id={}",cell_id, tvPdu_id);
                    }
                    else
                    {
                        tvPdu_id = 0; /* re Init */
                        /* new SRS.Ind with different SFN and SLOT */
                        srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                        NVLOGD_FMT(TAG, "Cell id = {} New srs.ind {}{} tvPdu_id={}",cell_id, srs_ind_sfn, srs_ind_slot,tvPdu_id);
                    }
                }

                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);

                // UL time/slot cost validation
                slot_timing_t &timing = cell_data[cell_id].slot_timing[resp.slot];
                int64_t ul_ns = nvlog_get_interval(&timing.ts_ul_tti);

                srs_ind_time->add(srs_ind_time, ul_ns);

                if (resp.sfn != timing.ss.u16.sfn) {
                    // Had delayed more than a frame
                    FAPI_VALIDATE_TEXT_ERR(&vald, "UL cost more than a frame", timing.ss.u16.sfn,
                            timing.ss.u16.slot);
                } else {
                    sfn_slot_t ss_curr = ss_tick.load();
                    int ul_slot = (ss_curr.u16.slot + slots_per_frame - timing.ss.u16.slot) % slots_per_frame;
                        if (ul_slot > VALD_TOLERANCE_SRS_SLOT_WINDOW_64T64R) {
                            FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_SRS_SLOT_WINDOW_64T64R);
                        }
                }

                vald.set_fapi_req(get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::SRS));
                uint8_t* next = reinterpret_cast<uint8_t*>(resp.srs_info);
                std::size_t data_offset = 0, offset = 0;
                uint8_t* srs_report_buffer = NULL;
                bool free_srs_chest_buffer_index_l2 = true;
                for(int pdu_id = 0; pdu_id < resp.num_pdus; pdu_id++)
                {
                    int rbSnrOffset = 0;
                    scf_fapi_srs_info_t* srs_info = reinterpret_cast<scf_fapi_srs_info_t*>(next + offset);
                    NVLOGD_FMT(TAG, "Cell id = {} Validate tvPdu_id={}", cell_id, tvPdu_id);
                    validate_srs_ind(cell_id, resp.sfn, resp.slot, pdu_id, srs_info, srs_report_buffer, srs_info->handle, &rbSnrOffset, vald, free_srs_chest_buffer_index_l2);

                    offset+= sizeof(scf_fapi_srs_info_t);
                    offset += rbSnrOffset;

                    NVLOGD_FMT(TAG, "sizeof(scf_fapi_srs_info_t)= {} rbSnrOffset={}",sizeof(scf_fapi_srs_info_t),rbSnrOffset);
                    thrputs[cell_id].srs ++;
                    tvPdu_id++;
                    srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                }
                vald.msg_ended();

                std::chrono::nanoseconds ts_msg_end = std::chrono::system_clock::now().time_since_epoch();
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SRS.ind num_pdu={} handle_time={} total_delay={}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus,
                        ts_msg_end.count() - ts_msg_start.count(), ts_msg_end.count() - transport().get_ts_send(msg));

                validate_timing_srs_indication(cell_id, ts_msg_start.count(), resp);

                }
            break;
            default:
                NVLOGW_FMT(TAG, "RECV: Unknown_message: hdr.msgId=0x{:02X} msg_len={} ", msg_id, static_cast<unsigned>(body->length));
                break;
            }
        }
        else if(msg.data_pool == NV_IPC_MEMPOOL_CPU_DATA || msg.data_pool == NV_IPC_MEMPOOL_CUDA_DATA || msg.data_pool == NV_IPC_MEMPOOL_CPU_LARGE)
        {
            switch(msg_id)
            {
                case SCF_FAPI_SRS_INDICATION: {
                std::chrono::nanoseconds ts_msg_start = std::chrono::system_clock::now().time_since_epoch();
#if 0
                    uint8_t* buffer_ptr = (uint8_t*)msg.data_buf + (sizeof(scf_fapi_norm_ch_iq_matrix_info_t));
                    float2* value = (float2*)buffer_ptr;
                    NVLOGI_FMT(TAG, "First SRS CV = {} + j{}",value[0].x, value[0].y);
#endif


                scf_fapi_srs_ind_t& resp = *reinterpret_cast<scf_fapi_srs_ind_t*>(body);
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SRS.ind with DATA num_pdu={}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus);

                uint16_t srs_ind_sfn       = resp.sfn;
                uint16_t srs_ind_slot      = resp.slot;

                int tvPdu_id = 0;
                auto iter = srsIndMap.find(cell_id);
                if (iter == srsIndMap.end())
                {
                    srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                    NVLOGD_FMT(TAG, "Cell id = {} Very first srs.ind {}{} tvPdu_id={}",cell_id, srs_ind_sfn, srs_ind_slot,tvPdu_id);
                }
                else
                {
                    if((srs_ind_sfn == std::get<0>(iter->second)) && (srs_ind_slot == std::get<1>(iter->second)))
                    {
                        tvPdu_id = std::get<2>(iter->second);
                        /* Grouped SRS.ind*/
                        NVLOGD_FMT(TAG, "Cell id = {} Grouped srs.ind tvPdu_id={}",cell_id, tvPdu_id);
                    }
                    else
                    {
                        tvPdu_id = 0; /* re Init */
                        /* new SRS.Ind with different SFN and SLOT */
                        srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                        NVLOGD_FMT(TAG, "Cell id = {} New srs.ind {}{} tvPdu_id={}",cell_id, srs_ind_sfn, srs_ind_slot,tvPdu_id);
                    }
                }

                vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);

                // UL time/slot cost validation
                slot_timing_t &timing = cell_data[cell_id].slot_timing[resp.slot];
                int64_t ul_ns = nvlog_get_interval(&timing.ts_ul_tti);

                srs_ind_time->add(srs_ind_time, ul_ns);
                //NVLOGC_FMT(TAG, "If Cell id = {} resp.sfn={} timing.ss.u16.sfn={} ul_ns={}", cell_id, static_cast<unsigned>(resp.sfn), static_cast<unsigned>(timing.ss.u16.sfn), ul_ns);

                if (resp.sfn != timing.ss.u16.sfn) {
                    // Had delayed more than a frame
                    FAPI_VALIDATE_TEXT_ERR(&vald, "UL cost more than a frame", timing.ss.u16.sfn,
                            timing.ss.u16.slot);
                } else {
                    sfn_slot_t ss_curr = ss_tick.load();
                    int ul_slot = (ss_curr.u16.slot + slots_per_frame - timing.ss.u16.slot) % slots_per_frame;
                        if (ul_slot > VALD_TOLERANCE_SRS_SLOT_WINDOW_64T64R) {
                            FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_SRS_SLOT_WINDOW_64T64R);
                        }
                }

                vald.set_fapi_req(get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::SRS));

                uint8_t* next = reinterpret_cast<uint8_t*>(resp.srs_info);
                std::size_t data_offset = 0, offset = 0;
                uint8_t* srs_report_buffer = reinterpret_cast<uint8_t*>(msg.data_buf);
                uint32_t prv_handle = 0;
                bool free_srs_chest_buffer_index_l2 = false;
                for(int pdu_id = 0; pdu_id < resp.num_pdus; pdu_id++)
                {
                    int rbSnrOffset = 0;
                    scf_fapi_srs_info_t* srs_info = reinterpret_cast<scf_fapi_srs_info_t*>(next + offset);
                    /*
                    next += sizeof(scf_fapi_srs_ind_t);
                    for (int i = 0; i < srs_pdu->num_reported_symbols; i ++)
                    {
                        scf_fapi_srs_info_t* snr = reinterpret_cast<scf_fapi_srs_info_t*>(next);
                        next += sizeof(scf_fapi_srs_ind_pdu_end_t) + snr->num_rbs;
                    }
                    */
                    NVLOGD_FMT(TAG, "Cell id = {} Validate tvPdu_id={}",cell_id, tvPdu_id);
#ifdef SCF_FAPI_10_04
                    NVLOGD_FMT(TAG,"tag={}",reinterpret_cast<uint16_t>(srs_info->srs_report_tlv.tag));
                    if(srs_info->srs_report_tlv.tag == 1)
                    {
                        data_offset = srs_info->srs_report_tlv.value;
                    }
                    else if(srs_info->srs_report_tlv.tag == 2)
                    {
                        data_offset = 0;
                    }
#endif
                    /* This is done to handle the mutiple usage case, Eg: USAGE_BEAM_MGMT+USAGE_CODEBOOK where data is encoded in 2 SRS PDU's
                       in SRS.IND but in TV they are part of the same PDU entry or RNTI
                       TODO: The counter "thrputs[cell_id].srs" is incremented twice if there are 2 bits set in usage which if needed can be fixed  */
                    if(prv_handle == srs_info->handle)
                    {
                        tvPdu_id--;
                        free_srs_chest_buffer_index_l2 = false;
                    }
                    else
                    {
                        free_srs_chest_buffer_index_l2 = true;
                    }
                    uint8_t srsPduIdx = static_cast<uint8_t>(srs_info->handle & 0xFF);

                    NVLOGD_FMT(TAG, "Cell id = {} Validate pdu_id={} tvPdu_id={} prv_handle={} srs_info->handle={} srsPduIdx={} free_srs_chest_buffer_index_l2={}", cell_id, pdu_id, tvPdu_id, prv_handle,static_cast<unsigned>(srs_info->handle), srsPduIdx, free_srs_chest_buffer_index_l2);

                    validate_srs_ind(cell_id, resp.sfn, resp.slot, pdu_id, srs_info, srs_report_buffer+data_offset, srsPduIdx, &rbSnrOffset, vald, free_srs_chest_buffer_index_l2);
                    offset+= (sizeof(scf_fapi_srs_info_t) + rbSnrOffset);
                    prv_handle = srs_info->handle;

                    NVLOGD_FMT(TAG, "sizeof(scf_fapi_srs_info_t)= {} rbSnrOffset={}",sizeof(scf_fapi_srs_info_t),rbSnrOffset);
                    if(free_srs_chest_buffer_index_l2)
                    {
                        thrputs[cell_id].srs ++;
                    }
                    tvPdu_id++;
                    srsIndMap[cell_id] = std::make_tuple(srs_ind_sfn, srs_ind_slot,tvPdu_id);
                }
                vald.msg_ended();
                //NVLOGW_FMT(TAG, "RECV: SRS_IND Not Supported: hdr.msgId=0x{:02X} msg_len={} ", msg_id, static_cast<unsigned>(body->length));

                std::chrono::nanoseconds ts_msg_end = std::chrono::system_clock::now().time_since_epoch();
                NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SRS.ind with DATA num_pdu={} handle_time={} total_delay={}",
                        static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, resp.num_pdus,
                        ts_msg_end.count() - ts_msg_start.count(), ts_msg_end.count() - transport().get_ts_send(msg));


                validate_timing_srs_indication(cell_id, ts_msg_start.count(), resp);

                // Only enable stat_debug code during debug
                // stat_debug->add(stat_debug, ts_msg_end.count() - transport().get_ts_send(msg));
                }
                break;
                case SCF_FAPI_RX_DATA_INDICATION: {
                    auto&                   resp       = *reinterpret_cast<scf_fapi_rx_data_ind_t*>(body);
                    auto handle_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
                    int                     tb_size    = 0;
                    int                     tb_offset  = 0;
                    size_t                  msg_offset = 0;
                    scf_fapi_rx_data_pdu_t* pdu;
                    uint8_t* tb_data = nullptr;

                    vald.msg_start(cell_id, msg_id, resp.sfn, resp.slot);
                    vald.set_fapi_req(get_fapi_req_data(cell_id, resp.sfn, resp.slot, channel_type_t::PUSCH));

                    // UL time/slot cost validation
                    slot_timing_t &timing = cell_data[cell_id].slot_timing[resp.slot];
                    int64_t ul_ns = nvlog_get_interval(&timing.ts_ul_tti);
                    rx_data_time->add(rx_data_time, ul_ns);

                    if (resp.sfn != timing.ss.u16.sfn) {
                        // Had delayed more than a frame
                        FAPI_VALIDATE_TEXT_ERR(&vald, "UL cost more than a frame", timing.ss.u16.sfn,
                                timing.ss.u16.slot);
                    } else {
                        sfn_slot_t ss_curr = ss_tick.load();
                        int ul_slot = (ss_curr.u16.slot + slots_per_frame - timing.ss.u16.slot) % slots_per_frame;
                        if (lp->get_mmimo_enabled() == 0 && (lp->get_mmimo_static_dynamic_enabled() == 0))
                        {
                            if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW) {
                                FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW);
                            }
                        }
                        else
                        {
                            if (ul_slot > VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R) {
                                FAPI_VALIDATE_TEXT_WARN(&vald, "ul_slot=%d > %d", ul_slot, VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R);
                            }
                        }
                    }

                    for(int i = 0; i < resp.num_pdus; i++)
                    {
                        pdu = (scf_fapi_rx_data_pdu_t*)((uint8_t*)resp.pdus + msg_offset);

                        if(msg.data_pool == NV_IPC_MEMPOOL_CPU_DATA)
                        {
                            // Get current PDU DATA address
                            tb_data = reinterpret_cast<uint8_t*>(msg.data_buf) + tb_offset;

                            // Next PDU offset
                            msg_offset += sizeof(scf_fapi_rx_data_pdu_t);
                        }
                        else
                        {
                            NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "Error: data_buf_opt={}", data_buf_opt);
                            return;
                        }

                        if(validate_rx_data_ind(cell_id, resp.sfn, resp.slot, i ,pdu, tb_data, vald) == 0)
                        {
                            tb_size += pdu->pdu_len;
                        }

                        tb_offset += pdu->pdu_len; //+ sizeof(scf_fapi_rx_data_pdu_t);
                    }
                    NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SCF_FAPI_RX_DATA_INDICATION numPDUs={} tb_size={}",
                            static_cast<unsigned>(resp.sfn), static_cast<unsigned>(resp.slot), cell_id, msg_id, static_cast<int>(resp.num_pdus), tb_size);

                    vald.msg_ended();

                    if(hdr->handle_id != cell_id)
                    {
                        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "RECV: cell_id error: msg_id=0x{:02X} carrier_id={} cell_id={}", msg.msg_id, hdr->handle_id, cell_id);
                        break;
                    }

                    if(resp.num_pdus == 0)
                    {
                        break;
                    }

                    thrputs[cell_id].ul_thrput += tb_size;
                    thrputs[cell_id].slots[PUSCH]++;
                    validate_indication_timing(cell_id, handle_start_time.count(), resp.sfn, resp.slot, configs->ul_ind_deadline_ns, cell_summary[cell_id].ul_ind, thrputs[cell_id].ul_ind);
                }
                break;

                default:
                    NVLOGI_FMT(TAG, "RECV: Unknown_message: hdr.msgId=0x{:02X} msg_len={} ", msg_id, static_cast<int>(body->length));
                break;
            }
        }
        else
        {
            NVLOGW_FMT(TAG, "Invalid data_pool: {}", msg.data_pool);
            return;
        }
    }
}

int scf_fapi_handler::validate_rach_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_prach_ind_pdu_t* prmb, fapi_validate& vald)
{
    if (vald.pdu_start(pdu_id, channel_type_t::PRACH) < 0)
    {
        return -1;
    }

    fapi_req_t* req = vald.get_fapi_req();
    prach_tv_data_t* tv_data = &req->tv_data->prach_tv.data[pdu_id];
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid RACH pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    prach_ind_t& tv = tv_data->ind;
    uint32_t phyCellId = lp->get_cell_configs(cell_id_map[cell_id]).phyCellId;
    FAPI_VALIDATE_U32_ERR(&vald, prmb->phys_cell_id, phyCellId);
    FAPI_VALIDATE_U32_ERR(&vald, prmb->symbol_index, tv.SymbolIndex);
    FAPI_VALIDATE_U32_ERR(&vald, prmb->slot_index, slot);
    FAPI_VALIDATE_U32_ERR(&vald, prmb->freq_index, tv.FreqIndex);
    FAPI_VALIDATE_U32_WARN(&vald, prmb->num_preamble, tv.numPreamble);
    FAPI_VALIDATE_U32_WARN(&vald, prmb->avg_rssi, tv.avgRssi, VALD_TOLERANCE_RACH_AVG_RSSI);
    FAPI_VALIDATE_U32_ERR(&vald, prmb->avg_snr, tv.avgSnr, VALD_TOLERANCE_RACH_AVG_SNR);

    // All preambles should come but may not in the same order
    for (int i = 0; i < prmb->num_preamble; i++)
    {
        scf_fapi_prach_preamble_info_t &prmb_info = prmb->preamble_info[i];

        bool found = false;
        for (int j = 0; j < tv.numPreamble; j++)
        {
            if (prmb_info.preamble_index == tv.preambleIndex_v[j])
            {
                FAPI_VALIDATE_U32_ERR(&vald, prmb_info.timing_advance, tv.TimingAdvance_v[j], VALD_TOLERANCE_TIMING_ADVANCE);
                // preamble_power = (10*log10(PDU2_peak) + 140 - 48.68) * 1000 // BB2RF_powerOffset = -48.68
                FAPI_VALIDATE_U32_WARN(&vald, prmb_info.preamble_power, tv.PreamblePwr_v[j], VALD_TOLERANCE_RACH_PREAMBLE_PWR);
                found = true;
            }
        }
        if (!found)
        {
            FAPI_VALIDATE_TEXT_WARN(&vald, "Prmb[%d].preamble_index=%d not found", i, prmb_info.preamble_index);
        }
    }

    return vald.pdu_ended(pdu_id, tv.idxInd);
}

int scf_fapi_handler::validate_crc_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_crc_info_t* crc, fapi_validate& vald)
{
    if (configs->app_mode != 0) {
        return 0;
    }

    if (vald.pdu_start(pdu_id, channel_type_t::PUSCH) < 0)
    {
        return -1;
    }

    pusch_tv_data_t* tv_data = get_pusch_tv_pdu(vald.get_fapi_req(), pdu_id, PUSCH_BITMAP_DATA);
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUSCH_DATA pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    // no check for negative case
    if (tv_data->tbErr != 0)
    {
        //FAPI_VALIDATE_TEXT_WARN(&vald, "Negative PUSCH_DATA pdu_id %d", pdu_id);
        NVLOGD_FMT(TAG, "NO check for PUSCH negative case (tbErr = 1).");
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    scf_fapi_crc_end_info_t *crc_end = reinterpret_cast<scf_fapi_crc_end_info_t*>(crc->cb_crc_status + (crc->num_cb + 7) / 8);
    pusch_data_ind_t& tv = tv_data->data_ind;

    uint8_t harq_pid = 0;
    lp->read_harq_pid(cell_id, sfn, slot, pdu_id, &harq_pid);
    FAPI_VALIDATE_U32_ERR(&vald, crc->harq_id, harq_pid);
    FAPI_VALIDATE_U32_ERR(&vald, crc->rnti, tv_data->tbpars.nRnti);
    FAPI_VALIDATE_U32_ERR(&vald, crc->tb_crc_status, tv.TbCrcStatus);
    if(configs->get_conformance_test_params()->conformance_test_enable)
    {
        uint16_t ue_id = insertOrUpdate(rnti_2_ueid_map,crc->rnti);
        ul_harq_handle_t & curr_ul_harq_handle = ul_harq_handle[cell_id][ue_id][crc->harq_id];
        if(crc->tb_crc_status == 0 || curr_ul_harq_handle.retx_count == FAPI_MAX_HARQ_RETX)
        {

            curr_ul_harq_handle.ul_dci_NDI ^= 1;
            curr_ul_harq_handle.pusch_pdu_NDI = 1;
            curr_ul_harq_handle.rv = 0;
            curr_ul_harq_handle.retx_count = 0;
            //conformance_test_slot_counter++;
        }
        else
        {
            // curr_ul_harq_handle.ul_dci_NDI = curr_ul_harq_handle.ul_dci_NDI;
            curr_ul_harq_handle.pusch_pdu_NDI = 0;
            if(curr_ul_harq_handle.retx_count == 0)
                curr_ul_harq_handle.rv = 2;
            else
                curr_ul_harq_handle.rv == 2 ? curr_ul_harq_handle.rv = 3 : curr_ul_harq_handle.rv = 1;
            curr_ul_harq_handle.retx_count++;
           // if(conformance_test_slot_counter > 0)
           //     conformance_test_slot_counter++;
        }
    }
    // Compare CRC num_cb with 0 instead of tv.NumCb since CBG is not supported
    FAPI_VALIDATE_U32_ERR(&vald, crc->num_cb, 0);


    // Validate PUSCH UL measurement parameters in CRC.ind
#ifdef SCF_FAPI_10_04
    compare_ul_measurement(&vald, tv_data->data_ind.meas, reinterpret_cast<uint8_t*>(&crc_end->measurement), 0);
#else
    compare_ul_measurement(&vald, tv_data->data_ind.meas, &crc_end->ul_cqi, 0);
#endif
    if((configs->get_conformance_test_params()->conformance_test_enable)&&
    (global_tick > configs->get_conformance_test_params()->conformance_test_start_time))
    {
        if(tv.TbCrcStatus)
        {

        }
    }
    return vald.pdu_ended(pdu_id, tv_data->data_ind.idxInd);
}

int scf_fapi_handler::validate_rx_data_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_rx_data_pdu_t* pdu, uint8_t* tb_data, fapi_validate& vald)
{
    if (configs->app_mode != 0) {
        return 0;
    }

    if (vald.pdu_start(pdu_id, channel_type_t::PUSCH) < 0)
    {
        return -1;
    }

    pusch_tv_data_t* tv_data = get_pusch_tv_pdu(vald.get_fapi_req(), pdu_id, PUSCH_BITMAP_DATA);
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUSCH_DATA pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    // no check for negative case
    if (tv_data->tbErr != 0)
    {
        //FAPI_VALIDATE_TEXT_WARN(&vald, "Negative PUSCH_DATA pdu_id %d", pdu_id);
        NVLOGD_FMT(TAG, "NO check for PUSCH negative case (tbErr = 1).");
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

#ifndef SCF_FAPI_10_04
    // Validate PUSCH UL measurement parameters in RX_DATA.ind
    compare_ul_measurement(&vald, tv_data->data_ind.meas, &pdu->ul_cqi, 0);
#endif

    uint8_t harq_pid = 0;
    lp->read_harq_pid(cell_id, sfn, slot, pdu_id, &harq_pid);

    // Validate other
    FAPI_VALIDATE_U32_ERR(&vald, pdu->rnti, tv_data->tbpars.nRnti);
    FAPI_VALIDATE_U32_ERR(&vald, pdu->harq_id, harq_pid);
    FAPI_VALIDATE_U32_ERR(&vald, pdu->pdu_len, tv_data->tb_size);

    if(tv_data->tb_size > 0 && pdu->pdu_len == tv_data->tb_size)
    {
        FAPI_VALIDATE_BYTES_ERR(&vald, tb_data, tv_data->tb_buf, tv_data->tb_size);
    }

    return vald.pdu_ended(pdu_id, tv_data->data_ind.idxInd);
}

int scf_fapi_handler::validate_srs_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_srs_info_t* pdu, uint8_t* iq_report_buffer, uint32_t handle, int * pRbSnrOffset, fapi_validate& vald, bool free_srs_chest_buffer_index_l2)
{
    if (configs->app_mode != 0) {
        NVLOGW_FMT(TAG, "App mode is not 0, skipping SRS validation");
        return 0;
    }
/*TODO: Uncomment this once we have a way to validate SRS pdu_id with usage=3.
  Currently, we are not validating SRS pdu_id with usage=3 becasue the SRS_PDU's will be more than the SRS.IND.
  This is a temporary fix to avoid the validation of SRS pdu_id with usage=3.*/
#if 0
    if (vald.pdu_start(pdu_id, channel_type_t::SRS) < 0)
    {
        NVLOGW_FMT(TAG, "Failed to start SRS pdu_id {}", pdu_id);
        return -1;
    }
#endif
    fapi_req_t* req = vald.get_fapi_req();
    // srs_tv_data_t* tv_data = &req->tv_data->srs_tv.data[tvPdu_id];
    srs_tv_data_t* tv_data = get_srs_tv_pdu(req, handle);
    if (tv_data == nullptr)
    {
        NVLOGW_FMT(TAG, "Invalid SRS pdu_id {}", pdu_id);
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid SRS pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    uint8_t* next = NULL;
    srs_ind_t& tv = tv_data->ind;
    FAPI_VALIDATE_U32_ERR(&vald, pdu->rnti, tv_data->RNTI, 0);

    if(std::isnan(pdu->timing_advance)) // special case: timing_advance = NaN, FAPI TV saved as 0, see saveTV_FAPI.m
    {
        uint32_t pdu_timing_advance_nanAs0 = 0;
        FAPI_VALIDATE_U32_ERR(&vald, pdu_timing_advance_nanAs0, tv.ind0.taOffset, VALD_TOLERANCE_TIMING_ADVANCE);
    }
    else
    {
        FAPI_VALIDATE_U32_ERR(&vald, pdu->timing_advance, tv.ind0.taOffset, VALD_TOLERANCE_TIMING_ADVANCE);
    }

#ifdef SCF_FAPI_10_04_SRS
#ifdef SCF_FAPI_10_04
    auto srsChestBuffIndxFrmHandle = static_cast<uint16_t>((pdu->handle >> 8) & 0xFFFF);
    if(free_srs_chest_buffer_index_l2) // This is to update the map of rnti to srs chest buffer index
    {
        mapOfRntiToSrsChestBufIdxMutex[cell_id].lock();
        updateMapOfRntiToSrsChestBufIdx(cell_id, pdu->rnti, srsChestBuffIndxFrmHandle);
        NVLOGD_FMT(TAG, "{}.{} If Test Mac validate_srs_ind::cell_id={} rnti={} handle={} srsChestBufferIndex={} usage={}", sfn, slot, cell_id, static_cast<unsigned int>(pdu->rnti), static_cast<unsigned int>(pdu->handle), static_cast<unsigned int>(srsChestBuffIndxFrmHandle), pdu->srs_usage);
        mapOfRntiToSrsChestBufIdxMutex[cell_id].unlock();
    }
    else
    {
        NVLOGD_FMT(TAG, "{}.{} Else Test Mac validate_srs_ind::cell_id={} rnti={} handle={} srsChestBufferIndex={} usage={}", sfn, slot, cell_id, static_cast<unsigned int>(pdu->rnti), static_cast<unsigned int>(pdu->handle), static_cast<unsigned int>(srsChestBuffIndxFrmHandle), pdu->srs_usage);
    }

    if(std::isnan(pdu->timing_advance_ns)) // special case: timing_advance_ns = NaN, FAPI TV saved as -16800, see saveTV_FAPI.m
    {
        int16_t pdu_timing_advance_ns_nanAsNeg16800 = -16800;
        FAPI_VALIDATE_I16_ERR(&vald, pdu_timing_advance_ns_nanAsNeg16800, tv.ind0.taOffsetNs, static_cast<int16_t>(10));
    }
    else
    {
        FAPI_VALIDATE_I16_ERR(&vald, pdu->timing_advance_ns, tv.ind0.taOffsetNs, static_cast<int16_t>(10));
    }
    if(free_srs_chest_buffer_index_l2)
    {
        mapOfSrsReqToIndRntiToSrsChestBufIdxMutex[cell_id].lock();
        uint32_t srsChestBufferIndex = getMapOfSrsReqToIndRntiToSrsChestBufIdx(cell_id, pdu->rnti);
        mapOfSrsReqToIndRntiToSrsChestBufIdxMutex[cell_id].unlock();
        if (srsChestBufferIndex != 0xFFFF)
        {
            FAPI_VALIDATE_U32_ERR(&vald, srsChestBuffIndxFrmHandle, srsChestBufferIndex);
        }
    }
#endif

    auto usage_bitmap_from_tv = (1 << pdu->srs_usage) & tv_data->fapi_v4_params.usage;
    FAPI_VALIDATE_U8_ERR(&vald, (1 << pdu->srs_usage),  usage_bitmap_from_tv);

    if(pdu->srs_report_tlv.length != 0)
    {
        if((pdu->srs_usage & SRS_USAGE_FOR_CODEBOOK) || (pdu->srs_usage & SRS_USAGE_FOR_NON_CODEBOOK))
        {
            if (iq_report_buffer == nullptr)
            {
                FAPI_VALIDATE_TEXT_ERR(&vald, "SRS PDU iq_report_buffer is nullptr");
            }
            else
            {
                auto info = reinterpret_cast<scf_fapi_norm_ch_iq_matrix_info_t*>(iq_report_buffer);
                if(info->num_prgs)
                {
                    auto expected_iq_repr = INDEX_IQ_REPR_32BIT_NORMALIZED_;
                    FAPI_VALIDATE_U8_ERR(&vald, info->norma_iq_repr, expected_iq_repr);
                    FAPI_VALIDATE_U16_ERR(&vald, info->num_gnb_ant_elmts, tv.ind1.numGnbAntennaElements);
                    FAPI_VALIDATE_U16_ERR(&vald, info->num_ue_srs_ports, tv.ind1.numUeSrsAntPorts);
                    FAPI_VALIDATE_U16_ERR(&vald, info->num_prgs, tv.ind1.numPRGs);
                    FAPI_VALIDATE_U16_ERR(&vald, info->prg_size, tv.ind1.prgSize);

                    auto iq_buf = reinterpret_cast<uint8_t*>(iq_report_buffer + (sizeof(scf_fapi_norm_ch_iq_matrix_info_t)));
                    short2* value = reinterpret_cast<short2*>(iq_buf);
                    // validate SNR by sum of the ratio errors
                    //auto& iq_sample_tv = tv.ind1.report_iq_data;
                    auto& iq_sample_buf = value;
                    uint32_t vald_num_samples = configs->srs_vald_sample_num > 0 ? configs->srs_vald_sample_num : tv.ind1.report_iq_data.size();
                    FAPI_VALIDATE_COMPLEX_INTEGER_APPROX_RATIO_ERR(&vald, iq_sample_buf, tv.ind1.report_iq_data.data(), tv.ind1.report_iq_data.size(), 0.001f, vald_num_samples);
                }
            }
        }
        else if(pdu->srs_usage == SRS_USAGE_FOR_BEAM_MANAGEMENT)
        {
            next = reinterpret_cast<uint8_t*>(pdu) + sizeof(scf_fapi_srs_info_t);
            scf_fapi_v3_bf_report_t& srs_bf_report = *(reinterpret_cast<scf_fapi_v3_bf_report_t*>(next));
            FAPI_VALIDATE_U8_ERR(&vald, srs_bf_report.num_symbols , tv.ind0.numSymbols);
            if(srs_bf_report.wideband_snr != 0xFF)
            {
                FAPI_VALIDATE_U8_ERR(&vald, srs_bf_report.wideband_snr, tv.ind0.wideBandSNR,1);
            }
            FAPI_VALIDATE_U8_ERR(&vald, srs_bf_report.num_reported_symbols, 1);
            uint8_t *ptrSnrRpt = reinterpret_cast<uint8_t*>(srs_bf_report.num_prg_snr_info);
            (*pRbSnrOffset) = sizeof(scf_fapi_v3_bf_report_t);

            for(int i = 0; i < srs_bf_report.num_reported_symbols; i++)
            {
                FAPI_VALIDATE_U16_ERR(&vald, *reinterpret_cast<uint16_t *>(ptrSnrRpt), (tv.ind1.numPRGs));
                ptrSnrRpt += sizeof(uint16_t);
                (*pRbSnrOffset) += sizeof(uint16_t);

                // Optimize based on known prbGrpSize values: 1, 2, 4, 16
                // For 272 PRBs: size=1 (272 PRGs), size=2 (136 PRGs), size=4 (68 PRGs), size=16 (17 PRGs)
                const int totalPrbs = tv.ind0.numPRGs * tv.ind0.prgSize;
                switch(tv.ind1.prgSize)
                {
                    case 1: {
                        // No averaging needed - direct copy with 8x loop unrolling
                        int j = 0;
                        for (; j + 7 < totalPrbs; j += 8)
                        {
                            const uint8_t rb_snr0 = static_cast<uint8_t>(tv_data->SNRval[j + 0]);
                            const uint8_t rb_snr1 = static_cast<uint8_t>(tv_data->SNRval[j + 1]);
                            const uint8_t rb_snr2 = static_cast<uint8_t>(tv_data->SNRval[j + 2]);
                            const uint8_t rb_snr3 = static_cast<uint8_t>(tv_data->SNRval[j + 3]);
                            const uint8_t rb_snr4 = static_cast<uint8_t>(tv_data->SNRval[j + 4]);
                            const uint8_t rb_snr5 = static_cast<uint8_t>(tv_data->SNRval[j + 5]);
                            const uint8_t rb_snr6 = static_cast<uint8_t>(tv_data->SNRval[j + 6]);
                            const uint8_t rb_snr7 = static_cast<uint8_t>(tv_data->SNRval[j + 7]);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 0), rb_snr0, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 1), rb_snr1, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 2), rb_snr2, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 3), rb_snr3, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 4), rb_snr4, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 5), rb_snr5, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 6), rb_snr6, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 7), rb_snr7, 1);
                            ptrSnrRpt += 8;
                            (*pRbSnrOffset) += 8;
                        }
                        for (; j < totalPrbs; j++)
                        {
                            const uint8_t rb_snr = static_cast<uint8_t>(tv_data->SNRval[j]);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), rb_snr, 1);
                            ptrSnrRpt += sizeof(uint8_t);
                            (*pRbSnrOffset) += sizeof(uint8_t);
                        }
                        break;
                    }

                    case 2: {
                        // Average 2 PRBs per group with 8x loop unrolling
                        int j = 0;
                        for (; j + 15 < totalPrbs; j += 16)
                        {
                            const uint8_t rb_snr0 = static_cast<uint8_t>((tv_data->SNRval[j +  0] + tv_data->SNRval[j +  1]) * 0.5f);
                            const uint8_t rb_snr1 = static_cast<uint8_t>((tv_data->SNRval[j +  2] + tv_data->SNRval[j +  3]) * 0.5f);
                            const uint8_t rb_snr2 = static_cast<uint8_t>((tv_data->SNRval[j +  4] + tv_data->SNRval[j +  5]) * 0.5f);
                            const uint8_t rb_snr3 = static_cast<uint8_t>((tv_data->SNRval[j +  6] + tv_data->SNRval[j +  7]) * 0.5f);
                            const uint8_t rb_snr4 = static_cast<uint8_t>((tv_data->SNRval[j +  8] + tv_data->SNRval[j +  9]) * 0.5f);
                            const uint8_t rb_snr5 = static_cast<uint8_t>((tv_data->SNRval[j + 10] + tv_data->SNRval[j + 11]) * 0.5f);
                            const uint8_t rb_snr6 = static_cast<uint8_t>((tv_data->SNRval[j + 12] + tv_data->SNRval[j + 13]) * 0.5f);
                            const uint8_t rb_snr7 = static_cast<uint8_t>((tv_data->SNRval[j + 14] + tv_data->SNRval[j + 15]) * 0.5f);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 0), rb_snr0, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 1), rb_snr1, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 2), rb_snr2, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 3), rb_snr3, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 4), rb_snr4, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 5), rb_snr5, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 6), rb_snr6, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 7), rb_snr7, 1);
                            ptrSnrRpt += 8;
                            (*pRbSnrOffset) += 8;
                        }
                        for (; j < totalPrbs; j += 2)
                        {
                            const uint8_t rb_snr = static_cast<uint8_t>((tv_data->SNRval[j] + tv_data->SNRval[j + 1]) * 0.5f);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), rb_snr, 1);
                            ptrSnrRpt += sizeof(uint8_t);
                            (*pRbSnrOffset) += sizeof(uint8_t);
                        }
                        break;
                    }

                    case 4: {
                        // Average 4 PRBs per group with 4x loop unrolling
                        int j = 0;
                        for (; j + 15 < totalPrbs; j += 16)
                        {
                            const uint8_t rb_snr0 = static_cast<uint8_t>((tv_data->SNRval[j +  0] + tv_data->SNRval[j +  1] + tv_data->SNRval[j +  2] + tv_data->SNRval[j +  3]) * 0.25f);
                            const uint8_t rb_snr1 = static_cast<uint8_t>((tv_data->SNRval[j +  4] + tv_data->SNRval[j +  5] + tv_data->SNRval[j +  6] + tv_data->SNRval[j +  7]) * 0.25f);
                            const uint8_t rb_snr2 = static_cast<uint8_t>((tv_data->SNRval[j +  8] + tv_data->SNRval[j +  9] + tv_data->SNRval[j + 10] + tv_data->SNRval[j + 11]) * 0.25f);
                            const uint8_t rb_snr3 = static_cast<uint8_t>((tv_data->SNRval[j + 12] + tv_data->SNRval[j + 13] + tv_data->SNRval[j + 14] + tv_data->SNRval[j + 15]) * 0.25f);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 0), rb_snr0, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 1), rb_snr1, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 2), rb_snr2, 1);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt + 3), rb_snr3, 1);
                            ptrSnrRpt += 4;
                            (*pRbSnrOffset) += 4;
                        }
                        for (; j < totalPrbs; j += 4)
                        {
                            const uint8_t rb_snr = static_cast<uint8_t>((tv_data->SNRval[j] + tv_data->SNRval[j + 1] + 
                                                                        tv_data->SNRval[j + 2] + tv_data->SNRval[j + 3]) * 0.25f);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), rb_snr, 1);
                            ptrSnrRpt += sizeof(uint8_t);
                            (*pRbSnrOffset) += sizeof(uint8_t);
                        }
                        break;
                    }

                    case 16: {
                        // Average 16 PRBs per group - no outer loop unrolling needed (only 17 iterations)
                        for(int j = 0; j < totalPrbs; j += 16)
                        {
                            const float avg_snr = (tv_data->SNRval[j +  0] + tv_data->SNRval[j +  1] +
                                                  tv_data->SNRval[j +  2] + tv_data->SNRval[j +  3] +
                                                  tv_data->SNRval[j +  4] + tv_data->SNRval[j +  5] +
                                                  tv_data->SNRval[j +  6] + tv_data->SNRval[j +  7] +
                                                  tv_data->SNRval[j +  8] + tv_data->SNRval[j +  9] +
                                                  tv_data->SNRval[j + 10] + tv_data->SNRval[j + 11] +
                                                  tv_data->SNRval[j + 12] + tv_data->SNRval[j + 13] +
                                                  tv_data->SNRval[j + 14] + tv_data->SNRval[j + 15]) * 0.0625f;
                            const uint8_t rb_snr = static_cast<uint8_t>(avg_snr);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), rb_snr, 1);
                            ptrSnrRpt += sizeof(uint8_t);
                            (*pRbSnrOffset) += sizeof(uint8_t);
                        }
                        break;
                    }

                    default:
                        // Fallback for unexpected sizes
                        for(int j = 0; j < totalPrbs; j += tv.ind1.prgSize)
                        {
                            float sum_snr = 0.0f;
                            for(int k = 0; k < tv.ind1.prgSize; k++) {
                                sum_snr += tv_data->SNRval[j + k];
                            }
                            const uint8_t rb_snr = static_cast<uint8_t>(sum_snr / tv.ind1.prgSize);
                            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), rb_snr, 1);
                            ptrSnrRpt += sizeof(uint8_t);
                            (*pRbSnrOffset) += sizeof(uint8_t);
                        }
                        break;
                }
            }
            //TODO: Enable below code if we want report to be alligned to 32 bit (4 Bytes)
            #if 0
                uint8_t pad_bytes = (4 - ((*pRbSnrOffset) % 4));
                (*pRbSnrOffset) += pad_bytes;
            #endif
        }
    }
    // FAPI_VALIDATE_BYTES_ERR(&vald, iq_buf, tv.ind1.report_iq_data.data(), tv.ind1.report_iq_data.size());
    /*
    uint8_t* next = pdu->payload;
    for (int i = 0; i < pdu->num_reported_symbols; i ++)
    {
        scf_fapi_srs_ind_pdu_end_t* snr = reinterpret_cast<scf_fapi_srs_ind_pdu_end_t*>(next);
        next += sizeof(scf_fapi_srs_ind_pdu_end_t) + snr->num_rbs;
        for (int j = 0; j < snr->num_rbs; j ++)
        {
            // TODO: validate snr
        }
    }
    */
#else
    //10_02
    FAPI_VALIDATE_U8_ERR(&vald, pdu->numSymbols, tv.ind0.numSymbols);
    FAPI_VALIDATE_U8_ERR(&vald, pdu->wideBandSNR, tv.ind0.wideBandSNR, VALD_TOLERANCE_SRS_WIDEBAND_SNR);
    FAPI_VALIDATE_U8_ERR(&vald, pdu->numReportedSymbols, 1);
    uint8_t *ptrSnrRpt = pdu->report;
    //NVLOGD_FMT(TAG,"tv.ind0.numPRGs {} tv.ind0.prgSize {} tv.ind1.numPRGs {} tv.ind1.prgSize {}",tv.ind0.numPRGs,tv.ind0.prgSize,tv.ind1.numPRGs,tv.ind1.prgSize);
    for(int i = 0; i < pdu->numReportedSymbols; i++)
    {
        FAPI_VALIDATE_U16_ERR(&vald, *reinterpret_cast<uint16_t *>(ptrSnrRpt), (tv.ind1.numPRGs*tv.ind1.prgSize));
        ptrSnrRpt += sizeof(uint16_t);
        (*pRbSnrOffset) += sizeof(uint16_t);
        for(int j =0 ; j < (tv.ind0.numPRGs*tv.ind0.prgSize); j++)
        {
            FAPI_VALIDATE_U8_ERR(&vald, *reinterpret_cast<uint8_t *>(ptrSnrRpt), tv_data->SNRval[j], VALD_TOLERANCE_SRS_PER_RB_SNR);
            ptrSnrRpt += sizeof(uint8_t);
            (*pRbSnrOffset) += sizeof(uint8_t);
        }
    }
#endif
    return vald.pdu_ended(pdu_id, tv.idxInd);
}

int scf_fapi_handler::validate_pe_noise_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_meas_t* pdu, fapi_validate& vald)
{
    if (configs->app_mode != 0) {
        return 0;
    }

    if (vald.pdu_start(pdu_id, channel_type_t::PUSCH) < 0)
    {
        return -1;
    }

    cell_configs_t& cell_configs = lp->get_cell_configs(vald.cell_id);
    fapi_req_t* req = vald.get_fapi_req();
    auto& pusch_noise_var_tolerance = cell_configs.tolerance.pusch_pe_noiseVardB;

    pusch_tv_data_t* tv_data = req->tv_data->pusch_tv.data[pdu_id];
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUSCH pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    int idxInd = -1;
    if(tv_data->pduBitmap & PUSCH_BITMAP_DATA)
    {
        if (tv_data->tbErr == 0)
        {
            if (cell_configs.pusch_sinr_selector == 1) { // Post EQ
                FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->data_ind.postEqNoiseVardB, pusch_noise_var_tolerance);
            } else if (cell_configs.pusch_sinr_selector == 2) {  // PreEQ
                FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->data_ind.noiseVardB, pusch_noise_var_tolerance);
            } else { // Disabled
                // Skip validating
            }
            // FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->data_ind.noiseVardB, pusch_noise_var_tolerance);
        }
        idxInd = tv_data->data_ind.idxInd;
    }
    else if (tv_data->pduBitmap & PUSCH_BITMAP_UCI)
    {
        if (tv_data->tbErr == 0)
        {
            if (cell_configs.pusch_sinr_selector == 1) { // Post EQ
                FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->uci_ind.postEqNoiseVardB, pusch_noise_var_tolerance);
            } else if (cell_configs.pusch_sinr_selector == 2) {  // PreEQ
                FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->uci_ind.noiseVardB, pusch_noise_var_tolerance);
            } else { // Disabled
                // Skip validating
            }
            // FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv_data->uci_ind.noiseVardB, pusch_noise_var_tolerance);
        }
        idxInd = tv_data->uci_ind.idxInd;
    }
    else
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUSCH pduBitmap 0x%X", tv_data->pduBitmap);
    }

    return vald.pdu_ended(pdu_id, idxInd);
}

int scf_fapi_handler::validate_pf234_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_meas_t* pdu, fapi_validate& vald)
{
    if (configs->app_mode != 0) {
        return 0;
    }

    if (vald.pdu_start(pdu_id, channel_type_t::PUCCH) < 0)
    {
        return -1;
    }

    fapi_req_t* req = vald.get_fapi_req();
    pucch_tv_data_t* tv_data = get_pucch_tv_pdu(req, pdu->handle);
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid PUCCH pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    pucch_uci_ind_t& tv = tv_data->uci_ind;

    FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv.noiseVardB, VALD_TOLERANCE_MEAS_PUCCH_NOISE);

    return vald.pdu_ended(pdu_id, tv.idxInd);
}

int scf_fapi_handler::validate_prach_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_prach_interference_t* pdu, fapi_validate& vald)
{
    if (configs->app_mode != 0) {
        return 0;
    }

    if (vald.pdu_start(pdu_id, channel_type_t::PRACH) < 0)
    {
        return -1;
    }

    fapi_req_t* req = vald.get_fapi_req();
    prach_tv_data_t* tv_data = &req->tv_data->prach_tv.data[pdu_id];
    if (tv_data == nullptr)
    {
        FAPI_VALIDATE_TEXT_ERR(&vald, "Invalid RACH pdu_id %d", pdu_id);
        vald.pdu_ended(pdu_id, -1);
        return -1;
    }

    prach_ind_t& tv = tv_data->ind;

    FAPI_VALIDATE_I16_WARN(&vald, pdu->meas, tv.avgNoise, VALD_TOLERANCE_MEAS_RACH_AVG_NOISE);

    return vald.pdu_ended(pdu_id, tv.idxInd);
}
