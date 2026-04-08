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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 24) // "DRV.MAP_UL"
#define TAG_VERBOSE (NVLOG_TAG_BASE_CUPHY_DRIVER + 41) // "DRV.MAP_UL_VERBOSE"
#define TAG_SYMBOL_TIMINGS (NVLOG_TAG_BASE_CUPHY_DRIVER + 36) // "DRV.SYMBOL_TIMINGS"
#define TAG_PACKET_TIMINGS (NVLOG_TAG_BASE_CUPHY_DRIVER + 37) // "DRV.PACKET_TIMINGS"
#define TAG_SYMBOL_TIMINGS_SRS (NVLOG_TAG_BASE_CUPHY_DRIVER + 45) // "DRV.SYMBOL_TIMINGS_SRS"
#define TAG_PACKET_TIMINGS_SRS (NVLOG_TAG_BASE_CUPHY_DRIVER + 46) // "DRV.PACKET_TIMINGS_SRS"

#include "slot_map_ul.hpp"
#include "context.hpp"
#include "nvlog_fmt.hpp"
#include <atomic>
#include <chrono>
#include <mutex>

#define ENABLE_RX_TS_LOGGING

SlotMapUl::SlotMapUl(phydriver_handle _pdh, uint64_t _id) :
    pdh(_pdh),
    id(_id)
{
    empty        = Time::zeroNs();
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    aggr_pusch = nullptr;
    aggr_pucch = nullptr;
    aggr_prach = nullptr;
    aggr_srs   = nullptr;
    aggr_ulbfw = nullptr;
    aggr_slot_params = nullptr;
    aggr_order_entity=nullptr;

    num_prach_occa.reserve(UL_MAX_CELLS_PER_SLOT);
    num_prach_occa.clear();
    aggr_cell_list.reserve(UL_MAX_CELLS_PER_SLOT);
    aggr_ulbuf_st1.reserve(UL_MAX_CELLS_PER_SLOT);
    aggr_ulbuf_st2.reserve(UL_MAX_CELLS_PER_SLOT);
    aggr_ulbuf_st3.reserve(UL_MAX_CELLS_PER_SLOT * PRACH_MAX_OCCASIONS);
    aggr_ulbuf_pcap_capture.reserve(UL_MAX_CELLS_PER_SLOT);
    aggr_ulbuf_pcap_capture_ts.reserve(UL_MAX_CELLS_PER_SLOT);

    for(int i = 0; i < UL_MAX_CELLS_PER_SLOT; i++)
    {
        aggr_slot_info[i]=nullptr;
    }

    num_active_cells = 0;

    cleanupTimes();
    task_current_number     = 0;
    atom_active             = false;
    run_order_done          = false;
    early_uci_task_done     = false;
    ulbfw_task_done         = false;
    atom_num_cells          = 0;
    atom_ul_channel_end_threads=0;
    atom_ul_end_threads     = 0;
    atom_ulc_tasks_complete = 0;
    tasks_num=0;
    tasks_ts_exec={};
    tasks_ts_enq={};
    isEarlyHarqPresent=0;
    isFrontLoadedDmrsPresent=0;
    atom_ul_cplane_info_for_uplane_rdy_count.store(0);

    mf.init(_pdh, std::string("SlotMapUl"), sizeof(SlotMapUl));
}

SlotMapUl::~SlotMapUl(){
    // delete ul_si;
    // delete dl_si;
};

int SlotMapUl::reserve()
{
    if(atom_active.load() == true)
        return -1;

    atom_active = true;

    return 0;
}

int SlotMapUl::release(int num_cells, bool enable_task_run_times)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    //////////////////////////////////////////////////////////////////
    //// Only last thread releases the slot map objects
    //////////////////////////////////////////////////////////////////
    int prev_cells = std::atomic_fetch_add(&(atom_num_cells), num_cells);
    if((num_active_cells > 0) && (prev_cells + num_cells < num_active_cells))
        return 0;

    bool isSrs = false;
    bool isNonSrsUl = false;
    bool ulBfwPrinted = false;

    if(num_active_cells > 0)
    {
        if(enable_task_run_times)
        {
            if(aggr_pusch || aggr_pucch || aggr_prach)
                isNonSrsUl = true;
            if(aggr_srs)
                isSrs = true;

            printTimes(isNonSrsUl, isSrs);
            ulBfwPrinted = true;
            //printTimes();
        }

        if(aggr_pusch != nullptr)
        {
            aggr_pusch->clearUciFlags(false);
            aggr_pusch->cleanup();
            aggr_pusch->release();
        }
        aggr_pusch = nullptr;

        if(aggr_pucch != nullptr)
        {
            aggr_pucch->clearUciFlags();
            aggr_pucch->cleanup();
            aggr_pucch->release();
        }
        aggr_pucch = nullptr;

        if(aggr_prach != nullptr)
        {
            aggr_prach->cleanup();
            aggr_prach->release();
        }
        aggr_prach = nullptr;
        if(aggr_srs != nullptr)
        {
            aggr_srs->cleanup();
            aggr_srs->release();
        }
        aggr_srs = nullptr;

        aggr_order_entity->cleanup();
        aggr_order_entity->release();

        for(int idx = 0; idx < aggr_ulbuf_st1.size(); idx++)
        {
            if(aggr_ulbuf_st1[idx] != nullptr)
            {
                aggr_ulbuf_st1[idx]->release();
            }
        }
        aggr_ulbuf_st1.clear();

        for(int idx = 0; idx < aggr_ulbuf_st3.size(); idx++)
        {
            if(aggr_ulbuf_st3[idx] != nullptr)
            {
                aggr_ulbuf_st3[idx]->release();
            }
        }
        aggr_ulbuf_st3.clear();

        for(int idx = 0; idx < aggr_ulbuf_st2.size(); idx++)
        {
            if(aggr_ulbuf_st2[idx] != nullptr)
            {
                aggr_ulbuf_st2[idx]->release();
            }
        }
        aggr_ulbuf_st2.clear();


        for(int idx = 0; idx < aggr_ulbuf_pcap_capture.size(); idx++)
        {
            if(aggr_ulbuf_pcap_capture[idx] != nullptr)
            {
                aggr_ulbuf_pcap_capture[idx]->release();
            }
        }
        aggr_ulbuf_pcap_capture.clear();

        for(int idx = 0; idx < aggr_ulbuf_pcap_capture_ts.size(); idx++)
        {
            if(aggr_ulbuf_pcap_capture_ts[idx] != nullptr)
            {
                aggr_ulbuf_pcap_capture_ts[idx]->release();
            }
        }
        aggr_ulbuf_pcap_capture_ts.clear();


        num_prach_occa.clear();

        for(int idx = 0; idx < aggr_cell_list.size(); idx++)
        {
	    if(aggr_slot_params) {
		//in case aggr pusch object is not available, aggr_slot_params is null and slot map release is called
                aggr_cell_list[idx]->setPuschDynPrmIndex(aggr_slot_params->si->slot_, -1);
	    }
        }

        aggr_cell_list.clear();
        aggr_slot_params = nullptr;
        num_active_cells = 0;
    }

    if(aggr_ulbfw != nullptr){
        if(!ulBfwPrinted)
        {
            printTimes(isNonSrsUl, isSrs);
        }
        aggr_ulbfw->cleanup();
        aggr_ulbfw->release();
    }
    aggr_ulbfw = nullptr;

    task_current_number = 0; //1
    cleanupTimes();
    atom_num_cells = 0;
    atom_active    = false;
    run_order_done = false;
    early_uci_task_done = false;
    ulbfw_task_done = false;
    atom_ul_channel_end_threads=0;
    atom_ul_end_threads     = 0;
    atom_ulc_tasks_complete = 0;
    isEarlyHarqPresent=0;
    isFrontLoadedDmrsPresent=0;
    atom_ul_cplane_info_for_uplane_rdy_count.store(0);
    return 0;
}

void SlotMapUl::getCellMplaneIdxList(std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& cell_idx_list, int *pcellCount)
{
    for(auto &c : aggr_cell_list)
    {
        cell_idx_list[*pcellCount]=c->getMplaneId()-1;
        (*pcellCount)++;
    }
}

int SlotMapUl::aggrSetCells(Cell* c, slot_command_api::phy_slot_params * _phy_slot_params, ULInputBuffer * ulbuf_st1,
                            ULInputBuffer * ulbuf_st2,std::array<ULInputBuffer*, PRACH_MAX_OCCASIONS> ul_bufst3_v,
                            const uint32_t rach_occasion, ULInputBuffer * ulbuf_pcap, ULInputBuffer * ulbuf_pcap_ts)
{
    if(c == nullptr || (ulbuf_st1 == nullptr && ulbuf_st2 == nullptr && rach_occasion == 0))
        return EINVAL;

    if(num_active_cells >= UL_MAX_CELLS_PER_SLOT)
        return ENOMEM;

    aggr_ulbuf_st1.push_back(ulbuf_st1);
    aggr_ulbuf_st2.push_back(ulbuf_st2);
    aggr_ulbuf_pcap_capture.push_back(ulbuf_pcap);
    aggr_ulbuf_pcap_capture_ts.push_back(ulbuf_pcap_ts);

    num_prach_occa.push_back(rach_occasion);

    for(int i = 0; i < rach_occasion; i++)
    {
        aggr_ulbuf_st3.push_back(ul_bufst3_v[i]);
    }

    aggr_cell_list.push_back(c);
    // slot_params* curr_slot_params
    // aggr_slot_params.push_back(curr_slot_params);
    aggr_slot_info[num_active_cells] = _phy_slot_params->sym_prb_info.get();
    // aggr_slot_oran_ind.push_back(si);
    num_active_cells++;

    return 0;
}

int SlotMapUl::aggrSetOrderEntity(OrderEntity* oentity)
{
    aggr_order_entity=oentity;
    return 0;
}
OrderEntity* SlotMapUl::aggrGetOrderEntity()
{
    return aggr_order_entity;
}

int SlotMapUl::aggrSetPhy(PhyPuschAggr* pusch, PhyPucchAggr* pucch, PhyPrachAggr* prach, PhySrsAggr* srs ,PhyUlBfwAggr* ulbfw,slot_params_aggr * _aggr_slot_params)
{
    //For now let's use and order kernel per cell
    if((pusch == nullptr) && (pucch == nullptr) && (prach == nullptr) && (srs==nullptr) && (ulbfw==nullptr)) // || oentity == nullptr)
        return EINVAL;

    aggr_pusch = pusch;
    aggr_pucch = pucch;
    aggr_prach = prach;
    aggr_srs = srs;
    aggr_ulbfw = ulbfw;
    // aggr_oentity = nullptr; //oentity;
    aggr_slot_params = _aggr_slot_params;
    return 0;
}


int SlotMapUl::getNumCells()
{
    return num_active_cells;
}

phydriver_handle SlotMapUl::getPhyDriverHandler(void) const
{
    return pdh;
}

uint64_t SlotMapUl::getId() const
{
    return id;
}

int SlotMapUl::setTasksTs(int _tasks_num, const std::array<t_ns, TASK_MAX_PER_SLOT + 1> _tasks_ts_exec, t_ns _tasks_ts_enq)
{
    tasks_num           = _tasks_num;
    tasks_ts_exec       = _tasks_ts_exec;
    tasks_ts_enq        = _tasks_ts_enq;

    return 0;
}

t_ns& SlotMapUl::getTaskTsExec(int task_num)
{
    if(task_num < TASK_MAX_PER_SLOT + 1)
        return tasks_ts_exec[task_num];
    return empty;
}

t_ns& SlotMapUl::getTaskTsEnq()
{
    return tasks_ts_enq;
}

int SlotMapUl::checkCurrentTask(int task_number, int tot_cells)
{
    int current = task_current_number.load();
    if(current == -1)
        //This represents an abort
        return -1;
    if(current >= (tot_cells * (task_number-1)))
    {
        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "task {} current {} tot_cells {}", task_number, current, tot_cells);
        return 1;
    }

    return 0;
}

bool SlotMapUl::unlockNextTask(int task_number, int num_cells)
{
    bool ret      = true;
    // int  expected = task_number;
    // ret           = std::atomic_compare_exchange_strong(&task_current_number, &expected, task_number + 1);
    task_current_number.fetch_add(num_cells);
    // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task {} add {} cells tot is {}", task_number, num_cells, task_current_number.load());

    return ret;
}

bool SlotMapUl::unlockChannelTask()
{
    bool ret      = true;
    run_order_done.store(true);

    return ret;
}

int SlotMapUl::waitChannelStartTask()
{
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "Wait for UL channel start for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait end Map {} num_active_cells = {} atomvar = {} ",
        //                 getId(), num_active_cells, atom_dl_end_threads.load());

    } while(run_order_done.load() != true);

    return 0;
}

int SlotMapUl::addChannelEndTask() {
    std::atomic_fetch_add(&(atom_ul_channel_end_threads),1);
    return 0;
}

int SlotMapUl::waitChannelEndTask(int num_channels) {
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait UL channel threads for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait end Map {} num_active_cells = {} atomvar = {} ",
        //                 getId(), num_active_cells, atom_dl_end_threads.load());

    } while(atom_ul_channel_end_threads != num_channels);

    return 0;
}

int SlotMapUl::addULCTasksComplete() {
    std::atomic_fetch_add(&(atom_ulc_tasks_complete),1);
    return 0;
}

int SlotMapUl::waitULCTasksComplete(int num_tasks) {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGW_FMT(TAG, "waitULCTasksComplete for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(atom_ulc_tasks_complete != num_tasks);

    return 0;
}

int SlotMapUl::addSlotEndTask() {
    std::atomic_fetch_add(&(atom_ul_end_threads),1);
    return 0;
}

int SlotMapUl::waitSlotEndTask(int num_tasks) {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait UL threads for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(atom_ul_end_threads != num_tasks);

    return 0;
}

int SlotMapUl::setEarlyUciEndTask() {
    early_uci_task_done=true;
    return 0;
}

bool SlotMapUl::waitEarlyUciEndTaskNonBlocking() {
    return early_uci_task_done;
}

int SlotMapUl::waitEarlyUciEndTask() {
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitEarlyUciEndTask for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(early_uci_task_done != true);

    return 0;
}

int SlotMapUl::setUlBfwEndTask() {
    ulbfw_task_done=true;
    return 0;
}

int SlotMapUl::waitUlBfwEndTask() {
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitUlBfwEndTask for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(ulbfw_task_done != true);

    return 0;
}

void SlotMapUl::abortTasks()
{
    task_current_number.store(-1);
}

bool SlotMapUl::tasksAborted()
{
    return task_current_number.load()==-1;
}

//void SlotMapUl::printTimes()
void SlotMapUl::printTimes(bool isNonSrsUl, bool isSrs)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    Cell* cell_ptr = nullptr;
    OrderEntity* oentity = nullptr;
    int sfn = 0, slot = 0;

    if(aggr_slot_params) {
        sfn = aggr_slot_params->si->sfn_;
        slot = aggr_slot_params->si->slot_;
    }
    oentity = aggr_order_entity;
    
    // Static variables for frame-based warmup tracking
    static int32_t last_sfn{-1};
    static int32_t transition_count{0};
    static std::mutex warmup_mutex;
    
    static bool warmup_completed{false};
    uint32_t warmup_frame_count = pdctx->getUlWarmupFrameCount();
    bool in_warmup_period = false;
    
    // Only track warmup if warmup_frame_count > 0 and we haven't completed warmup yet
    if (warmup_frame_count > 0 && !warmup_completed) {
        // Use lock to ensure thread-safe frame transition detection
        std::lock_guard<std::mutex> lock(warmup_mutex);

        // Establish the last SFN on the first UL slot encountered
        if (last_sfn == -1) {
            last_sfn = sfn;
            in_warmup_period = true;

        } else {

            // Calculate wrapped SFN difference (handles 1024 wrap-around)
            int sfn_diff = ((sfn - last_sfn + 512) % 1024) - 512;
            if (sfn_diff > 0) {
                transition_count++;
                last_sfn = sfn;

                in_warmup_period = (transition_count < warmup_frame_count);
                warmup_completed = (transition_count >= warmup_frame_count);
            } else {
                in_warmup_period = true;
            }

        }
    }
    
    // Avoid calling the same getGPU*Time* function multiple times for the same object, as this results in many unnecessary CUDA event queries and other API calls under the hood.
    // When a single order kernel is used, call the appropriate getGPU*Time* function once instead of calling it once per cell.
    float gpu_order_idle = (isNonSrsUl)?oentity->getGPUIdleTime():0;
    float gpu_order_run = (isNonSrsUl)?oentity->getGPUOrderTime():0;
    float gpu_order_idle_srs = (isSrs)?oentity->getGPUIdleTimeSrs():0;
    float gpu_order_run_srs = (isSrs)?oentity->getGPUOrderTimeSrs():0;
    for(int i = 0; i < getNumCells(); i++)
    {
        cell_ptr = aggr_cell_list[i];
        NVLOGI_FMT(TAG_VERBOSE, "[PHYDRV] SFN {}.{} UL Communication Tasks Cell {} Map {} times ===> "
            "[C-plane] {{ START: {} END: {} DURATION: {} us }} \n",
            sfn, slot, cell_ptr->getPhyId(), getId(),
            timings.start_t_ul_cplane[i].count(), timings.end_t_ul_cplane[i].count(), Time::NsToUs(timings.end_t_ul_cplane[i] - timings.start_t_ul_cplane[i]).count()
        );

            if(oentity != nullptr)
        {
            auto early=0,ontime=0,late=0;
            if(!(pdctx->cpuCommEnabled())||(pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevel()>=1))
            {
                early = oentity->getEarlyRxPackets(i);
                ontime = oentity->getOnTimeRxPackets(i);
                late = oentity->getLateRxPackets(i);
            }
            uint32_t earlySRS=0,ontimeSRS=0,lateSRS=0;
            if(isSrs){
                earlySRS = oentity->getEarlyRxPacketsSRS(i);
                ontimeSRS = oentity->getOnTimeRxPacketsSRS(i);
                lateSRS = oentity->getLateRxPacketsSRS(i);
                cell_ptr->update_peer_rx_metrics(oentity->getRxPacketCountSRS(i), oentity->getRxByteCountSRS(i));
            }

            if(i == 0) {
                NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} ORDER {:x} Cell {} Map {} times ===> "
                "[CPU CUDA time] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU RX U-plane] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU Free U-plane] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU Order Run] {{ START: {} END: {} DURATION: {} us}} "
                "[GPU Order Idle] {{ DURATION: {:.2f} us }} "
                "[GPU Order Run] {{ DURATION: {:.2f} us Exit Condition {} }} "
                "[GPU Order Idle SRS] {{ DURATION: {:.2f} us }} "
                "[GPU Order Run SRS] {{ DURATION: {:.2f} us Exit Condition {} }} "
                "[RX Packet Times] {{ EARLY: {} ONTIME: {} LATE: {} }} "
                "[SRS RX Packet Times] {{ EARLY: {} ONTIME: {} LATE: {} }}"
                "[RX Packet Dropped Count] {{ {} }} "
                "[SRS RX Packet Dropped Count] {{ {} }} "
                "\n",
                sfn, slot, oentity->getId(), cell_ptr->getPhyId(), getId(),
                timings.start_t_ul_order_cuda.count(), timings.end_t_ul_order_cuda.count(), Time::NsToUs(timings.end_t_ul_order_cuda - timings.start_t_ul_order_cuda).count(),
                timings.start_t_ul_rx_pkts[i].count(), timings.end_t_ul_rx_pkts[i].count(), Time::NsToUs(timings.end_t_ul_rx_pkts[i] - timings.start_t_ul_rx_pkts[i]).count(),
                timings.start_t_ul_freemsg[i].count(), timings.end_t_ul_freemsg[i].count(), Time::NsToUs(timings.end_t_ul_freemsg[i] - timings.start_t_ul_freemsg[i]).count(),
                timings.start_t_ul_order.count(), timings.end_t_ul_order.count(), Time::NsToUs(timings.end_t_ul_order - timings.start_t_ul_order).count(),
                gpu_order_idle, gpu_order_run, (isNonSrsUl)?oentity->getOrderExitCondition(i):0,
                gpu_order_idle_srs, gpu_order_run_srs, (isSrs)?oentity->getOrderSrsExitCondition(i):0,
                early,
                ontime,
                late,
                earlySRS,
                ontimeSRS,
                lateSRS,
                oentity->getRxPacketsDroppedCount(i),
                oentity->getRxPacketsDroppedCountSRS(i));
            }

            NVLOGI_FMT(TAG_VERBOSE, "[PHYDRV] SFN {}.{} ORDER {:x} Cell {} Map {} times ===> "
                "[CPU CUDA time] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU RX U-plane] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU Free U-plane] {{ START: {} END: {} DURATION: {} us }} "
                "[CPU Order Run] {{ START: {} END: {} DURATION: {} us}} "
                "[GPU Order Idle] {{ DURATION: {:.2f} us }} "
                "[GPU Order Run] {{ DURATION: {:.2f} us Exit Condition {} }} "
                "[GPU Order Idle SRS] {{ DURATION: {:.2f} us }} "
                "[GPU Order Run SRS] {{ DURATION: {:.2f} us Exit Condition {} }} "
                "[RX Packet Times] {{ EARLY: {} ONTIME: {} LATE: {} }} "
                "[SRS RX Packet Times] {{ EARLY: {} ONTIME: {} LATE: {} }}"
                "[RX Packet Dropped Count] {{ {} }} "
                "[SRS RX Packet Dropped Count] {{ {} }} "
                "\n",
                sfn, slot, oentity->getId(), cell_ptr->getPhyId(), getId(),
                timings.start_t_ul_order_cuda.count(), timings.end_t_ul_order_cuda.count(), Time::NsToUs(timings.end_t_ul_order_cuda - timings.start_t_ul_order_cuda).count(),
                timings.start_t_ul_rx_pkts[i].count(), timings.end_t_ul_rx_pkts[i].count(), Time::NsToUs(timings.end_t_ul_rx_pkts[i] - timings.start_t_ul_rx_pkts[i]).count(),
                timings.start_t_ul_freemsg[i].count(), timings.end_t_ul_freemsg[i].count(), Time::NsToUs(timings.end_t_ul_freemsg[i] - timings.start_t_ul_freemsg[i]).count(),
                timings.start_t_ul_order.count(), timings.end_t_ul_order.count(), Time::NsToUs(timings.end_t_ul_order - timings.start_t_ul_order).count(),
                gpu_order_idle, gpu_order_run, (isNonSrsUl)?oentity->getOrderExitCondition(i):0,
                gpu_order_idle_srs, gpu_order_run_srs, (isSrs)?oentity->getOrderSrsExitCondition(i):0,
                early,
                ontime,
                late,
                earlySRS,
                ontimeSRS,
                lateSRS,
                oentity->getRxPacketsDroppedCount(i),
                oentity->getRxPacketsDroppedCountSRS(i));

            // Only gather UL packet statistics after warmup period
            if (!in_warmup_period) {
                auto frame_slots = ORAN_MAX_SLOT_X_SUBFRAME_ID;
                auto frame_cycle = MAX_LAUNCH_PATTERN_SLOTS / frame_slots;
                auto slot_80 = (sfn % frame_cycle) * frame_slots + slot;
                
                pdctx->getULPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::EARLY, slot_80, early);
                pdctx->getULPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::ONTIME, slot_80, ontime);
                pdctx->getULPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::LATE, slot_80, late);
                pdctx->getULPacketStatistics()->set_active_slot(slot_80);

                if(isSrs)
                {
                    pdctx->getSRSPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::EARLY, slot_80, earlySRS);
                    pdctx->getSRSPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::ONTIME, slot_80, ontimeSRS);
                    pdctx->getSRSPacketStatistics()->increment_counter(i, Packet_Statistics::timing_type::LATE, slot_80, lateSRS);
                    pdctx->getSRSPacketStatistics()->set_active_slot(slot_80);
                }
            }
            size_t slot_packets_count = 0;
            size_t slot_bytes_count = 0;

            if((!pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevel() >=2)||(pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevel()>=3))
            {          
                if(isNonSrsUl)
                {
                    int packet_counts[ORAN_PUSCH_SYMBOLS_X_SLOT];
                    for(int sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                        char packet_times_str[1024];
                        int packet_times_offset = 0;

                        uint64_t min_packet_time = 0;
                        uint64_t max_packet_time = 0;

                        packet_times_offset += sprintf(&packet_times_str[packet_times_offset], "Sym%i ",sym_idx);

                        packet_counts[sym_idx] = oentity->getRxPacketCount(i, sym_idx);
                        slot_packets_count += packet_counts[sym_idx];
                        slot_bytes_count += oentity->getRxByteCount(i, sym_idx);
                        for(int pkt_idx=0;pkt_idx<packet_counts[sym_idx];pkt_idx++) {
                            uint64_t current_time = oentity->getRxPacketTs(i,sym_idx,pkt_idx);
                            if(pkt_idx == 0) {
                                max_packet_time = current_time;
                                min_packet_time = current_time;
                            }
                            if(current_time > max_packet_time) {
                                max_packet_time = current_time;
                            }
                            if(current_time < min_packet_time) {
                                min_packet_time = current_time;
                            }

                            packet_times_offset += sprintf(&packet_times_str[packet_times_offset], "%lu ", current_time);
                        }
                        NVLOGI_FMT(TAG_PACKET_TIMINGS,"SFN {}.{} Cell {} ts per packet: {}",sfn,slot,cell_ptr->getPhyId(),packet_times_str);
                    }

                    cell_ptr->update_peer_rx_metrics(slot_packets_count, slot_bytes_count);

                    NVLOGI_FMT(TAG_PACKET_TIMINGS,"SFN {}.{} Cell {} packet counts: {} {} {} {} {} {} {} {} {} {} {} {} {} {}",sfn,slot,cell_ptr->getPhyId(),
                    packet_counts[0],
                    packet_counts[1],
                    packet_counts[2],
                    packet_counts[3],
                    packet_counts[4],
                    packet_counts[5],
                    packet_counts[6],
                    packet_counts[7],
                    packet_counts[8],
                    packet_counts[9],
                    packet_counts[10],
                    packet_counts[11],
                    packet_counts[12],
                    packet_counts[13]);
                }
            }
            if(!pdctx->cpuCommEnabled() && pdctx->getUlRxPktTracingLevelSrs() >=2)
            {
                if(isSrs)
                {
                    int packet_counts[ORAN_SRS_SYMBOLS_X_SLOT];
                    for(int sym_idx=0;sym_idx<ORAN_SRS_SYMBOLS_X_SLOT;sym_idx++){
                        char packet_times_str[65536];
                        int packet_times_offset = 0;

                        uint64_t min_packet_time = 0;
                        uint64_t max_packet_time = 0;

                        packet_times_offset += sprintf(&packet_times_str[packet_times_offset], "Sym%i ",sym_idx);

                        packet_counts[sym_idx] = oentity->getRxPacketCountPerSymSRS(i,sym_idx);
                        for(int pkt_idx=0;pkt_idx<packet_counts[sym_idx];pkt_idx++) {
                            uint64_t current_time = oentity->getRxPacketTsSRS(i,sym_idx,pkt_idx);
                            if(pkt_idx == 0) {
                                max_packet_time = current_time;
                                min_packet_time = current_time;
                            }
                            if(current_time > max_packet_time) {
                                max_packet_time = current_time;
                            }
                            if(current_time < min_packet_time) {
                                min_packet_time = current_time;
                            }

                            packet_times_offset += sprintf(&packet_times_str[packet_times_offset], "%lu ", current_time);
                        }
                        NVLOGI_FMT(TAG_PACKET_TIMINGS_SRS,"[SRS] SFN {}.{} Cell {} ts per packet: {}",sfn,slot,cell_ptr->getPhyId(),packet_times_str);
                    }

                    NVLOGI_FMT(TAG_PACKET_TIMINGS_SRS,"[SRS] SFN {}.{} Cell {} packet counts: {} {} {} {} {} {} {} {} {} {} {} {} {} {}",sfn,slot,cell_ptr->getPhyId(),
                    packet_counts[0],
                    packet_counts[1],
                    packet_counts[2],
                    packet_counts[3],
                    packet_counts[4],
                    packet_counts[5],
                    packet_counts[6],
                    packet_counts[7],
                    packet_counts[8],
                    packet_counts[9],
                    packet_counts[10],
                    packet_counts[11],
                    packet_counts[12],
                    packet_counts[13]);
                }
            }            

            if((!pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevel() >=1)||(pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevel()>=2))
            {
                if(isNonSrsUl) {

                    uint64_t earliest_times[ORAN_PUSCH_SYMBOLS_X_SLOT];
                    uint64_t latest_times[ORAN_PUSCH_SYMBOLS_X_SLOT];
                    for(int sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                        earliest_times[sym_idx] = oentity->getRxPacketTsEarliest(i,sym_idx);
                        latest_times[sym_idx] = oentity->getRxPacketTsLatest(i,sym_idx);
                    }
                    NVLOGI_FMT(TAG_SYMBOL_TIMINGS,"SFN {}.{} Cell {} early/late ts per sym: {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}",
                    sfn,slot,cell_ptr->getPhyId(),
                    earliest_times[0],latest_times[0],
                    earliest_times[1],latest_times[1],
                    earliest_times[2],latest_times[2],
                    earliest_times[3],latest_times[3],
                    earliest_times[4],latest_times[4],
                    earliest_times[5],latest_times[5],
                    earliest_times[6],latest_times[6],
                    earliest_times[7],latest_times[7],
                    earliest_times[8],latest_times[8],
                    earliest_times[9],latest_times[9],
                    earliest_times[10],latest_times[10],
                    earliest_times[11],latest_times[11],
                    earliest_times[12],latest_times[12],
                    earliest_times[13],latest_times[13]
                    );

                }
            }
            if(!pdctx->cpuCommEnabled()&&pdctx->getUlRxPktTracingLevelSrs() >=1)
            {
                if(isSrs)
                {
                    uint64_t earliest_times[ORAN_SRS_SYMBOLS_X_SLOT];
                    uint64_t latest_times[ORAN_SRS_SYMBOLS_X_SLOT];
                    for(int sym_idx=0;sym_idx<ORAN_SRS_SYMBOLS_X_SLOT;sym_idx++){
                        earliest_times[sym_idx] = oentity->getRxPacketTsEarliestSRS(i,sym_idx);
                        latest_times[sym_idx] = oentity->getRxPacketTsLatestSRS(i,sym_idx);
                    }
                    NVLOGI_FMT(TAG_SYMBOL_TIMINGS_SRS,"[SRS] SFN {}.{} Cell {} early/late ts per sym: {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}",
                    sfn,slot,cell_ptr->getPhyId(),
                    earliest_times[0],latest_times[0],
                    earliest_times[1],latest_times[1],
                    earliest_times[2],latest_times[2],
                    earliest_times[3],latest_times[3],
                    earliest_times[4],latest_times[4],
                    earliest_times[5],latest_times[5],
                    earliest_times[6],latest_times[6],
                    earliest_times[7],latest_times[7],
                    earliest_times[8],latest_times[8],
                    earliest_times[9],latest_times[9],
                    earliest_times[10],latest_times[10],
                    earliest_times[11],latest_times[11],
                    earliest_times[12],latest_times[12],
                    earliest_times[13],latest_times[13]
                    );                    
                }
            }
        }
    }

    if(aggr_pusch != nullptr)
    {
        // Avoid calling the same getGPU*Time function more than once, as this performs multiple CUDA event queries under the hood.
        float pusch_gpu_setup_ph1 = (aggr_pusch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pusch->getGPUSetupPh1Time():0;
        float pusch_gpu_setup_ph2 = (aggr_pusch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pusch->getGPUSetupPh2Time():0;
        float pusch_gpu_run_time  = (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_pusch->getGPURunTime():0;

        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} PUSCH Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup Ph1] {{ DURATION: {:.2f} us }} "
            "[GPU Setup Ph2] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[GPU Run EH] {{ DURATION: {:.2f} us }} "
            "[GPU Run Post EH] {{ DURATION: {:.2f} us }} "
            "[GPU Run Phase 2] {{ DURATION: {:.2f} us }} "
            "[GPU Run Gap] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_pusch->getId(), getId(),
            timings.start_t_ul_pusch_cuda[0].count(), timings.start_t_ul_pusch_run[0].count(), Time::NsToUs(timings.start_t_ul_pusch_run[0] - timings.start_t_ul_pusch_cuda[0]).count(), +aggr_pusch->getSetupStatus(),
            timings.start_t_ul_pusch_run[0].count(), timings.end_t_ul_pusch_cuda[0].count(), Time::NsToUs(timings.end_t_ul_pusch_cuda[0] - timings.start_t_ul_pusch_run[0]).count(), +aggr_pusch->getRunStatus(),
            pusch_gpu_setup_ph1, pusch_gpu_setup_ph2,
            pusch_gpu_run_time,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isEarlyHarqPresent())?aggr_pusch->getGPURunSubSlotTime():0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isEarlyHarqPresent())?aggr_pusch->getGPURunPostSubSlotTime():0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isEarlyHarqPresent())?aggr_pusch->getGPUPhaseRunTime(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY):0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isEarlyHarqPresent())?aggr_pusch->getGPURunGapTime():0,
            timings.start_t_ul_pusch_cuda[0].count(), timings.end_t_ul_pusch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pusch_compl[0] - timings.start_t_ul_pusch_cuda[0]).count(),
            timings.start_t_ul_pusch_compl[0].count(), timings.end_t_ul_pusch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pusch_compl[0] - timings.start_t_ul_pusch_compl[0]).count(),
            timings.start_t_ul_pusch_cb[0].count(), timings.end_t_ul_pusch_cb[0].count(), Time::NsToUs(timings.end_t_ul_pusch_cb[0] - timings.start_t_ul_pusch_cb[0]).count()
        );

        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} PUSCH Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup Ph1] {{ DURATION: {:.2f} us }} "
            "[GPU Setup Ph2] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[GPU Run Front-loaded DMRS] {{ DURATION: {:.2f} us }} "
            "[GPU Run Post Front-loaded DMRS] {{ DURATION: {:.2f} us }} "
            "[GPU Run Phase 2] {{ DURATION: {:.2f} us }} "
            "[GPU Run Gap] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_pusch->getId(), getId(),
            timings.start_t_ul_pusch_cuda[0].count(), timings.start_t_ul_pusch_run[0].count(), Time::NsToUs(timings.start_t_ul_pusch_run[0] - timings.start_t_ul_pusch_cuda[0]).count(), +aggr_pusch->getSetupStatus(),
            timings.start_t_ul_pusch_run[0].count(), timings.end_t_ul_pusch_cuda[0].count(), Time::NsToUs(timings.end_t_ul_pusch_cuda[0] - timings.start_t_ul_pusch_run[0]).count(), +aggr_pusch->getRunStatus(),
            pusch_gpu_setup_ph1, pusch_gpu_setup_ph2,
            pusch_gpu_run_time,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isFrontLoadedDmrsPresent() && (!aggr_pusch->isEarlyHarqPresent()))?aggr_pusch->getGPURunSubSlotTime():0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isFrontLoadedDmrsPresent() && (!aggr_pusch->isEarlyHarqPresent()))?aggr_pusch->getGPURunPostSubSlotTime():0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isFrontLoadedDmrsPresent() && (!aggr_pusch->isEarlyHarqPresent()))?aggr_pusch->getGPUPhaseRunTime(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY):0,
            (aggr_pusch->getRunStatus()==CH_RUN_DONE_NO_ERROR && aggr_pusch->isFrontLoadedDmrsPresent() && (!aggr_pusch->isEarlyHarqPresent()))?aggr_pusch->getGPURunGapTime():0,
            timings.start_t_ul_pusch_cuda[0].count(), timings.end_t_ul_pusch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pusch_compl[0] - timings.start_t_ul_pusch_cuda[0]).count(),
            timings.start_t_ul_pusch_compl[0].count(), timings.end_t_ul_pusch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pusch_compl[0] - timings.start_t_ul_pusch_compl[0]).count(),
            timings.start_t_ul_pusch_cb[0].count(), timings.end_t_ul_pusch_cb[0].count(), Time::NsToUs(timings.end_t_ul_pusch_cb[0] - timings.start_t_ul_pusch_cb[0]).count()
        );
    }

    if(aggr_pucch != nullptr)
    {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} PUCCH Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }}"
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_pucch->getId(), getId(),
            timings.start_t_ul_pucch_cuda[0].count(), timings.start_t_ul_pucch_run[0].count(), Time::NsToUs(timings.start_t_ul_pucch_run[0] - timings.start_t_ul_pucch_cuda[0]).count(), +aggr_pucch->getSetupStatus(),
            timings.start_t_ul_pucch_run[0].count(), timings.end_t_ul_pucch_cuda[0].count(), Time::NsToUs(timings.end_t_ul_pucch_cuda[0] - timings.start_t_ul_pucch_run[0]).count(), +aggr_pucch->getRunStatus(),
            (aggr_pucch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pucch->getGPUSetupTime():0, (aggr_pucch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pucch->getGPURunTime():0,
            timings.start_t_ul_pucch_cuda[0].count(), timings.end_t_ul_pucch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pucch_compl[0] - timings.start_t_ul_pucch_cuda[0]).count(),
            timings.start_t_ul_pucch_compl[0].count(), timings.end_t_ul_pucch_compl[0].count(), Time::NsToUs(timings.end_t_ul_pucch_compl[0] - timings.start_t_ul_pucch_compl[0]).count(),
            timings.start_t_ul_pucch_cb[0].count(), timings.end_t_ul_pucch_cb[0].count(), Time::NsToUs(timings.end_t_ul_pucch_cb[0] - timings.start_t_ul_pucch_cb[0]).count()
        );
    }

    if(aggr_prach != nullptr)
    {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} PRACH Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_prach->getId(), getId(),
            timings.start_t_ul_prach_cuda[0].count(), timings.start_t_ul_prach_run[0].count(), Time::NsToUs(timings.start_t_ul_prach_run[0] - timings.start_t_ul_prach_cuda[0]).count(), +aggr_prach->getSetupStatus(),
            timings.start_t_ul_prach_run[0].count(), timings.end_t_ul_prach_cuda[0].count(), Time::NsToUs(timings.end_t_ul_prach_cuda[0] - timings.start_t_ul_prach_run[0]).count(), +aggr_prach->getRunStatus(),
            (aggr_prach->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_prach->getGPUSetupTime():0, (aggr_prach->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_prach->getGPURunTime():0,
            timings.start_t_ul_prach_cuda[0].count(), timings.end_t_ul_prach_compl[0].count(), Time::NsToUs(timings.end_t_ul_prach_compl[0] - timings.start_t_ul_prach_cuda[0]).count(),
            timings.start_t_ul_prach_compl[0].count(), timings.end_t_ul_prach_compl[0].count(), Time::NsToUs(timings.end_t_ul_prach_compl[0] - timings.start_t_ul_prach_compl[0]).count(),
            timings.start_t_ul_prach_cb[0].count(), timings.end_t_ul_prach_cb[0].count(), Time::NsToUs(timings.end_t_ul_prach_cb[0] - timings.start_t_ul_prach_cb[0]).count()
        );
    }

    if(aggr_srs != nullptr)
    {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} SRS Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_srs->getId(), getId(),
            timings.start_t_ul_srs_cuda[0].count(), timings.start_t_ul_srs_run[0].count(), Time::NsToUs(timings.start_t_ul_srs_run[0] - timings.start_t_ul_srs_cuda[0]).count(), +aggr_srs->getSetupStatus(),
            timings.start_t_ul_srs_run[0].count(), timings.end_t_ul_srs_cuda[0].count(), Time::NsToUs(timings.end_t_ul_srs_cuda[0] - timings.start_t_ul_srs_run[0]).count(), +aggr_srs->getRunStatus(),
            (aggr_srs->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_srs->getGPUSetupTime():0, (aggr_srs->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_srs->getGPURunTime():0,
            timings.start_t_ul_srs_cuda[0].count(), timings.end_t_ul_srs_compl[0].count(), Time::NsToUs(timings.end_t_ul_srs_compl[0] - timings.start_t_ul_srs_cuda[0]).count(),
            timings.start_t_ul_srs_compl[0].count(), timings.end_t_ul_srs_compl[0].count(), Time::NsToUs(timings.end_t_ul_srs_compl[0] - timings.start_t_ul_srs_compl[0]).count(),
            timings.start_t_ul_srs_cb[0].count(), timings.end_t_ul_srs_cb[0].count(), Time::NsToUs(timings.end_t_ul_srs_cb[0] - timings.start_t_ul_srs_cb[0]).count()
        );
    }

    if(aggr_ulbfw != nullptr)
    {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} UL_BFW Aggr {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[Callback] {{ START: {} END: {} DURATION: {} us }}\n",
            sfn, slot,
            aggr_ulbfw->getId(), getId(),
            timings.start_t_ul_bfw_cuda[0].count(), timings.start_t_ul_bfw_run[0].count(), Time::NsToUs(timings.start_t_ul_bfw_run[0] - timings.start_t_ul_bfw_cuda[0]).count(), +aggr_ulbfw->getSetupStatus(),
            timings.start_t_ul_bfw_run[0].count(), timings.end_t_ul_bfw_cuda[0].count(), Time::NsToUs(timings.end_t_ul_bfw_cuda[0] - timings.start_t_ul_bfw_run[0]).count(), +aggr_ulbfw->getRunStatus(),
            (aggr_ulbfw->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_ulbfw->getGPUSetupTime():0, (aggr_ulbfw->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_ulbfw->getGPURunTime():0,
            timings.start_t_ul_bfw_cuda[0].count(), timings.end_t_ul_bfw_compl[0].count(), Time::NsToUs(timings.end_t_ul_bfw_compl[0] - timings.start_t_ul_bfw_cuda[0]).count(),
            timings.start_t_ul_bfw_compl[0].count(), timings.end_t_ul_bfw_compl[0].count(), Time::NsToUs(timings.end_t_ul_bfw_compl[0] - timings.start_t_ul_bfw_compl[0]).count(),
            timings.start_t_ul_bfw_cb[0].count(), timings.end_t_ul_bfw_cb[0].count(), Time::NsToUs(timings.end_t_ul_bfw_cb[0] - timings.start_t_ul_bfw_cb[0]).count()
        );
    }
}

void SlotMapUl::setSlot3GPP(slot_command_api::slot_indication si) {
    slot_3gpp.sfn_ = si.sfn_;
    slot_3gpp.slot_ = si.slot_;
    slot_3gpp.tick_ = si.tick_;
    slot_3gpp.t0_valid_ = si.t0_valid_;
    slot_3gpp.t0_ = si.t0_;
}

int16_t SlotMapUl::getDynBeamIdOffset() const
{
    return dyn_beam_id_offset;
}

void SlotMapUl::setDynBeamIdOffset(int16_t beam_id_offset)
{
    dyn_beam_id_offset = beam_id_offset;
}

struct slot_command_api::slot_indication SlotMapUl::getSlot3GPP() const {
    return slot_3gpp;
}

void SlotMapUl::cleanupTimes()
{
    memset(&timings, 0, sizeof(struct ul_slot_timings));
}
