/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 16) // "DRV.MAP_DL"
#define TAG_VERBOSE (NVLOG_TAG_BASE_CUPHY_DRIVER + 40) // "DRV.MAP_DL_VERBOSE"

#include "slot_map_dl.hpp"
#include "context.hpp"
#include "nvlog_fmt.hpp"

SlotMapDl::SlotMapDl(phydriver_handle _pdh, uint64_t _id, uint8_t _enableBatchedMemcpy) :
    pdh(_pdh),
    id(_id),
    m_batchedMemcpyHelper(MAX_CELLS_PER_SLOT,
                          batchedMemcpySrcHint::srcIsHost, 
                          batchedMemcpyDstHint::dstIsDevice, 
                          _enableBatchedMemcpy)
{
    empty        = Time::zeroNs();

    aggr_pdsch = nullptr;
    aggr_pdcch_dl = nullptr;
    aggr_pdcch_ul = nullptr;
    aggr_pbch = nullptr;
    aggr_csirs = nullptr;
    aggr_dlbfw = nullptr;
    aggr_slot_params = nullptr;
    for(int i = 0; i < DL_MAX_CELLS_PER_SLOT; i++)
    {
        aggr_slot_info[i]=nullptr;
    }

    num_active_cells = 0;

    aggr_cell_list.reserve(DL_MAX_CELLS_PER_SLOT);
    aggr_dlbuf_list.reserve(DL_MAX_CELLS_PER_SLOT);

    cleanupTimes();
    atom_active             = false;
    atom_num_cells          = 0;

    atom_fhcb_done.store(false);
    atom_dlc_done.store(0);
    atom_uplane_prep_done.store(0);
    atom_cplane_done_mask.store(0);

    atom_dl_cplane_start    = 0;
    atom_dl_end_threads     = 0;
    atom_dl_channel_end_threads=0;
    atom_dl_gpu_comm_end=false;
    atom_dl_cpu_door_bell_task_done=false;
    atom_dl_comp_end=false;
    dl_si=nullptr;
    tasks_num=0;
    tasks_ts_exec={};
    tasks_ts_enq={};
    pdsch_cb_done=false;
    atom_dl_cplane_info_for_uplane_rdy_count.store(0);

    mf.init(_pdh, std::string("SlotMapDl"), sizeof(SlotMapDl));

}

SlotMapDl::~SlotMapDl(){ };

int SlotMapDl::reserve()
{
    if(atom_active.load() == true)
        return -1;

    atom_active = true;

    return 0;
}

int SlotMapDl::waitCplaneReady(int num_threads) {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    std::atomic_fetch_add(&(atom_dl_cplane_start), 1);

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait DL Cplane ready for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(atom_dl_cplane_start != num_threads);

    return 0;
}

void SlotMapDl::setFHCBDone() {
    atom_fhcb_done.store(true);
}

// This denotes completion of a cell's FHCB
void SlotMapDl::setCellFHCBDone(int cell) {
    atom_cell_fhcb_done.at(cell).store(true);
}


int SlotMapDl::waitFHCBDone() {
    t_ns start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitFHCBDone for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(!atom_fhcb_done.load());

    return 0;
}

// Polls until timeout/atomic var set denoting either non-completion (timeout) or 
// completion (atomic var set) of FHCB per cell
int SlotMapDl::waitCellFHCBDone(int cell) {
    t_ns start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitFHCBDone for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(!atom_cell_fhcb_done.at(cell).load());

    return 0;
}

void SlotMapDl::incDLCDone() {
    std::atomic_fetch_add(&(atom_dlc_done), 1);
}

void SlotMapDl::incUplanePrepDone() {
    std::atomic_fetch_add(&(atom_uplane_prep_done), 1);
}

void SlotMapDl::setCplaneDoneForTask(const int task_num) {
    const uint32_t mask = 1U << task_num;
    atom_cplane_done_mask.fetch_or(mask);
}

int SlotMapDl::waitCplaneDoneForTask(const int task_num) {
    const uint32_t mask = 1U << task_num;
    t_ns start_wait = Time::nowNs();
    do {
        if (Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2)) {
            NVLOGI_FMT(TAG, "waitCplaneDoneForTask for Map {} task {} is taking more than {} ns", getId(), task_num, (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while ((atom_cplane_done_mask.load() & mask) == 0);

    return 0;
}

int SlotMapDl::waitDLCDone(int num_dlc_tasks) {
    t_ns start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitDLCDone for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_dlc_done.load() < num_dlc_tasks);

    return 0;
}

int SlotMapDl::waitPeerUpdateDone() {
    t_ns start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitPeerUpdateDone for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_dl_cplane_info_for_uplane_rdy_count.load() < num_active_cells);

    return 0;
}

int SlotMapDl::waitUplanePrepDone(int num_uplane_prep_tasks) {
    t_ns start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGI_FMT(TAG, "waitUplanePrepDone for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_uplane_prep_done.load() < num_uplane_prep_tasks);

    return 0;
}

int SlotMapDl::addSlotEndTask() {
    std::atomic_fetch_add(&(atom_dl_end_threads),1);
    return 0;
}

int SlotMapDl::addSlotChannelEnd() {
    std::atomic_fetch_add(&(atom_dl_channel_end_threads),1);
    return 0;
}


int SlotMapDl::setDlGpuCommEnd() {
    atom_dl_gpu_comm_end=true;
    return 0;
}

int SlotMapDl::setDlCpuDoorBellTaskDone() {
    atom_dl_cpu_door_bell_task_done=true;
    return 0;
}

int SlotMapDl::setDlCompEnd() {
    atom_dl_comp_end=true;
    return 0;
}

int SlotMapDl::waitDlGpuCommEnd() {
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait DL GPU Comms for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_dl_gpu_comm_end != true);

    return 0;
}

int SlotMapDl::waitDlCompEnd() {
    t_ns start_wait;

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait DL Compression for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_dl_comp_end != true);

    return 0;
}

int SlotMapDl::waitDlCpuDoorBellTaskDone() {
    t_ns start_wait;
    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait DL CPU Door Bell Task for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(atom_dl_cpu_door_bell_task_done != true);
    return 0;
}

int SlotMapDl::waitSlotEndTask(int num_tasks) {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait Slot End Task for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
    } while(atom_dl_end_threads != num_tasks);

    return 0;
}

int SlotMapDl::waitSlotChannelEnd(int num_channels) {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    start_wait = Time::nowNs();
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait Slot Channel End for Map {} is taking more than {} ns", getId(), (GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }

    } while(atom_dl_channel_end_threads != num_channels);

    return 0;
}

int SlotMapDl::addSlotEnd(int num_cells) {
    std::atomic_fetch_add(&(atom_dl_end_threads), num_cells);
    return 0;
}

void SlotMapDl::setSlot3GPP(slot_command_api::slot_indication si) {
    slot_3gpp.sfn_ = si.sfn_;
    slot_3gpp.slot_ = si.slot_;
    slot_3gpp.tick_ = si.tick_;
    slot_3gpp.t0_valid_ = si.t0_valid_;
    slot_3gpp.t0_ = si.t0_;
}

const struct slot_command_api::slot_indication SlotMapDl::getSlot3GPP() const {
    return slot_3gpp;
}

void SlotMapDl::getCellMplaneIdxList(std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>& cell_idx_list, int *pcellCount)
{
    for(auto &c : aggr_cell_list)
    {
        cell_idx_list[*pcellCount]=c->getMplaneId()-1;
        (*pcellCount)++;
    }
}

int SlotMapDl::release(int num_cells)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    //////////////////////////////////////////////////////////////////
    //// Only last thread releases the slot map objects
    //////////////////////////////////////////////////////////////////
    int prev_cells = std::atomic_fetch_add(&(atom_num_cells), num_cells);
    if((num_active_cells > 0) && (prev_cells + num_cells < num_active_cells))
        return 0;

    bool dlBfwPrinted = false;
    if(num_active_cells > 0)
    {
        printTimes();
        dlBfwPrinted = true;

        if (aggr_pdsch) {
            aggr_pdsch->cleanup();
            aggr_pdsch->release();
            aggr_pdsch = nullptr;
        }

        if (aggr_pdcch_dl) {
            aggr_pdcch_dl->cleanup();
            aggr_pdcch_dl->release();
            aggr_pdcch_dl = nullptr;
        }

        if (aggr_pdcch_ul) {
            aggr_pdcch_ul->cleanup();
            aggr_pdcch_ul->release();
            aggr_pdcch_ul = nullptr;
        }

        if (aggr_pbch) {
            aggr_pbch->cleanup();
            aggr_pbch->release();
            aggr_pbch = nullptr;
        }

        if (aggr_csirs) {
            aggr_csirs->cleanup();
            aggr_csirs->release();
            aggr_csirs = nullptr;
        }

        aggr_dlbuf_list.clear();
        aggr_cell_list.clear();
        //aggr_slot_info.clear();

        //Valid for all the channels
        // for(int idx = 0; idx < aggr_slot_params.size(); idx++)
        //     aggr_slot_params[idx]->slot_phy_prms.pdsch = nullptr;
        // aggr_slot_params.clear();

        aggr_slot_params = nullptr;
        num_active_cells = 0;
    }

    if(aggr_dlbfw){
        if(!dlBfwPrinted)
        {
            printTimes();
        }
        aggr_dlbfw->cleanup();
        aggr_dlbfw->release();
        aggr_dlbfw = nullptr;
    }

    cleanupTimes();
    atom_num_cells = 0;

    atom_fhcb_done.store(false);
    for (int cell = 0; cell < DL_MAX_CELLS_PER_SLOT; ++cell) {
        atom_cell_fhcb_done.at(cell).store(false); 
    }
    atom_dlc_done.store(0);
    atom_uplane_prep_done.store(0);
    atom_cplane_done_mask.store(0);

    atom_dl_cplane_start    = 0;
    atom_dl_end_threads     = 0;
    atom_dl_channel_end_threads=0;
    atom_dl_gpu_comm_end=false;
    atom_dl_comp_end=false;
    atom_dl_cpu_door_bell_task_done=false;
    atom_active    = false;
    pdsch_cb_done = false;
    atom_dl_cplane_info_for_uplane_rdy_count.store(0);
    return 0;
}


int SlotMapDl::aggrSetCells(Cell* c, slot_command_api::phy_slot_params * _phy_slot_params, DLOutputBuffer * dlbuf)
{
    if(c == nullptr || dlbuf == nullptr)
        return EINVAL;

    if(num_active_cells >= DL_MAX_CELLS_PER_SLOT)
        return ENOMEM;

    aggr_dlbuf_list.push_back(dlbuf);
    aggr_cell_list.push_back(c);
    aggr_slot_info[num_active_cells] = _phy_slot_params->sym_prb_info.get();

    num_active_cells++;

    return 0;
}

int SlotMapDl::aggrSetPhy(PhyPdschAggr* pdsch, PhyPdcchAggr * pdcch_dl, PhyPdcchAggr * pdcch_ul, PhyPbchAggr * pbch, PhyCsiRsAggr * csirs,PhyDlBfwAggr* dlbfw ,slot_params_aggr * _aggr_slot_params)
{
    if(pdsch == nullptr && pdcch_dl == nullptr && pdcch_ul == nullptr && pbch == nullptr && csirs == nullptr && dlbfw==nullptr)
        return EINVAL;

    aggr_pdsch = pdsch;
    aggr_pdcch_dl = pdcch_dl;
    aggr_pdcch_ul = pdcch_ul;
    aggr_pbch = pbch;
    aggr_csirs = csirs;
    aggr_dlbfw = dlbfw;
    aggr_slot_params = _aggr_slot_params;

    return 0;
}


int SlotMapDl::getNumCells()
{
    return num_active_cells;
}

phydriver_handle SlotMapDl::getPhyDriverHandler(void) const
{
    return pdh;
}

uint64_t SlotMapDl::getId() const
{
    return id;
}

int16_t SlotMapDl::getDynBeamIdOffset() const
{
    return dyn_beam_id_offset;
}

void SlotMapDl::setDynBeamIdOffset(int16_t beam_id_offset)
{
    dyn_beam_id_offset = beam_id_offset;
}


int SlotMapDl::setTasksTs(int _tasks_num, const std::array<t_ns, TASK_MAX_PER_SLOT + 1> _tasks_ts_exec, t_ns _tasks_ts_enq)
{
    tasks_num           = _tasks_num;
    tasks_ts_exec       = _tasks_ts_exec;
    tasks_ts_enq        = _tasks_ts_enq;

    return 0;
}

t_ns& SlotMapDl::getTaskTsExec(int task_num)
{
    if(task_num < TASK_MAX_PER_SLOT + 1)
        return tasks_ts_exec[task_num];
    return empty;
}

t_ns& SlotMapDl::getTaskTsEnq()
{
    return tasks_ts_enq;
}

void SlotMapDl::printTimes()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    Cell* cell_ptr = nullptr;
    int sfn = 0, slot = 0;

    if(aggr_slot_params) {
        sfn = aggr_slot_params->si->sfn_;
        slot = aggr_slot_params->si->slot_;
    }

    // int num_cells = getNumCells();
    // if(num_cells > 0) {
    // }

    for(int i = 0; i < getNumCells(); i++) {
        cell_ptr = aggr_cell_list[i];
        if(cell_ptr != nullptr)
        {
            if(i == 0) {
                NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} DL Communication Tasks Cell {} Map {} times ===> "
                    "[C-plane] {{ START: {} END: {} DURATION: {} us }} "
                    "[U-plane prepare] {{ START: {} END: {} DURATION: {} us }} "
                    "[U-plane tx] {{ START: {} END: {} DURATION: {} us }} "
                    "[Callback] {{ START: {} END: {} DURATION: {} us }}",
                    sfn, slot, cell_ptr->getPhyId(), getId(),
                    timings.start_t_dl_cplane[i].count(), timings.end_t_dl_cplane[i].count(), Time::NsToUs(timings.end_t_dl_cplane[i] - timings.start_t_dl_cplane[i]).count(),
                    timings.start_t_dl_uprep[i].count(), timings.end_t_dl_uprep[i].count(), Time::NsToUs(timings.end_t_dl_uprep[i] - timings.start_t_dl_uprep[i]).count(),
                    timings.start_t_dl_utx[i].count(), timings.end_t_dl_utx[i].count(), Time::NsToUs(timings.end_t_dl_utx[i] - timings.start_t_dl_utx[i]).count(),
                    timings.start_t_dl_callback.count(), timings.end_t_dl_callback.count(), Time::NsToUs(timings.end_t_dl_callback - timings.start_t_dl_callback).count()
                );
            }

            NVLOGI_FMT(TAG_VERBOSE, "[PHYDRV] SFN {}.{} DL Communication Tasks Cell {} Map {} times ===> "
                "[C-plane] {{ START: {} END: {} DURATION: {} us }} "
                "[U-plane prepare] {{ START: {} END: {} DURATION: {} us }} "
                "[U-plane tx] {{ START: {} END: {} DURATION: {} us }} "
                "[Callback] {{ START: {} END: {} DURATION: {} us }}",
                sfn, slot, cell_ptr->getPhyId(), getId(),
                timings.start_t_dl_cplane[i].count(), timings.end_t_dl_cplane[i].count(), Time::NsToUs(timings.end_t_dl_cplane[i] - timings.start_t_dl_cplane[i]).count(),
                timings.start_t_dl_uprep[i].count(), timings.end_t_dl_uprep[i].count(), Time::NsToUs(timings.end_t_dl_uprep[i] - timings.start_t_dl_uprep[i]).count(),
                timings.start_t_dl_utx[i].count(), timings.end_t_dl_utx[i].count(), Time::NsToUs(timings.end_t_dl_utx[i] - timings.start_t_dl_utx[i]).count(),
                timings.start_t_dl_callback.count(), timings.end_t_dl_callback.count(), Time::NsToUs(timings.end_t_dl_callback - timings.start_t_dl_callback).count()
            );

            if(cell_ptr->getDLBitWidth() != BFP_NO_COMPRESSION)
            {
                auto prepare_1_enabled = std::abs(timings.prepare_execution_duration1[0]) < 0.0001f ? 0 : 1;
                auto prepare_2_enabled = std::abs(timings.prepare_execution_duration1[1]) < 0.0001f ? 0 : 1;

                if(i==0) {
                    NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} COMPRESSION DL {} bitwidths Cell {} Map {} times ===> "
                        "[CPU CUDA time] {{ START: {} END: {} DURATION: {} us }} "
                        "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }} "
                        "[S0 Enabled] {{ {} }} [S0 Prepare Execution Duration 1] {{ DURATION: {:.2f} us}} "
                        "[S0 Prepare Execution Duration 2] {{ DURATION: {:.2f} us}} "
                        "[S0 Prepare Execution Duration 3] {{ DURATION: {:.2f} us}} "
                        "[S1 Enabled] {{ {} }} [S1 Prepare Execution Duration 1] {{ DURATION: {:.2f} us}} "
                        "[S1 Prepare Execution Duration 2] {{ DURATION: {:.2f} us}} "
                        "[S1 Prepare Execution Duration 3] {{ DURATION: {:.2f} us}} "
                        "[Channel To Compression Gap] {{ DURATION: {:.2f} us}} "
                        "[GPU Execution Duration] {{ DURATION: {:.2f} us}} "
                        "[PrePrepare To Compression Gap] {{ DURATION: {:.2f} us}} "
                        "[Packet memcopy dur per sym] {{ DURATION: {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us {:.2f} us }}",
                        sfn, slot, cell_ptr->getDLBitWidth(), cell_ptr->getPhyId(), getId(),
                        timings.start_t_dl_compression_cuda.count(), timings.end_t_dl_compression_cuda.count(), Time::NsToUs(timings.end_t_dl_compression_cuda - timings.start_t_dl_compression_cuda).count(),
                        timings.start_t_dl_compression_compl.count(), timings.end_t_dl_compression_compl.count(), Time::NsToUs(timings.end_t_dl_compression_compl - timings.start_t_dl_compression_compl).count(),
                        prepare_1_enabled,
                        timings.prepare_execution_duration1[0],
                        timings.prepare_execution_duration2[0],
                        timings.prepare_execution_duration3[0],
                        prepare_2_enabled,
                        timings.prepare_execution_duration1[1],
                        timings.prepare_execution_duration2[1],
                        timings.prepare_execution_duration3[1],
                        timings.channel_to_compression_gap,
                        timings.compression_execution_duration,
                        timings.prePrepare_to_compression_gap[0],
                        timings.packet_mem_copy_per_symbol_dur_us[0],timings.packet_mem_copy_per_symbol_dur_us[1],timings.packet_mem_copy_per_symbol_dur_us[2],timings.packet_mem_copy_per_symbol_dur_us[3],timings.packet_mem_copy_per_symbol_dur_us[4],timings.packet_mem_copy_per_symbol_dur_us[5],timings.packet_mem_copy_per_symbol_dur_us[6],
                        timings.packet_mem_copy_per_symbol_dur_us[7],timings.packet_mem_copy_per_symbol_dur_us[8],timings.packet_mem_copy_per_symbol_dur_us[9],timings.packet_mem_copy_per_symbol_dur_us[10],timings.packet_mem_copy_per_symbol_dur_us[11],timings.packet_mem_copy_per_symbol_dur_us[12],timings.packet_mem_copy_per_symbol_dur_us[13]
                    );
                }
            }
        }
    }

    if (aggr_pdsch != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} PDSCH Aggr {:x} Map {} times ===> "
            "[PDSCH H2D Copy] {{ DURATION: {:.2f} us }} "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_pdsch->getId(), getId(),
            aggr_pdsch->getPdschH2DCopyTime(static_cast<uint8_t>(slot)),
            timings.start_t_dl_pdsch_setup[0].count(), timings.end_t_dl_pdsch_setup[0].count(), Time::NsToUs(timings.end_t_dl_pdsch_setup[0] - timings.start_t_dl_pdsch_setup[0]).count(), +aggr_pdsch->getSetupStatus(),
            timings.start_t_dl_pdsch_run[0].count(), timings.end_t_dl_pdsch_run[0].count(), Time::NsToUs(timings.end_t_dl_pdsch_run[0] - timings.start_t_dl_pdsch_run[0]).count(), +aggr_pdsch->getRunStatus(),
            (aggr_pdsch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pdsch->getGPUSetupTime():0,
            (aggr_pdsch->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_pdsch->getGPURunTime():0,
            timings.start_t_dl_pdsch_setup[0].count(), timings.end_t_dl_pdsch_compl[0].count(), Time::NsToUs(timings.end_t_dl_pdsch_compl[0] - timings.start_t_dl_pdsch_setup[0]).count(),
            timings.start_t_dl_pdsch_compl[0].count(), timings.end_t_dl_pdsch_compl[0].count(), Time::NsToUs(timings.end_t_dl_pdsch_compl[0] - timings.start_t_dl_pdsch_compl[0]).count()
        );
    }

    if (aggr_pdcch_dl != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} Aggr PDCCH DL {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_pdcch_dl->getId(), getId(),
            timings.start_t_dl_pdcchdl_setup[0].count(), timings.end_t_dl_pdcchdl_setup[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchdl_setup[0] - timings.start_t_dl_pdcchdl_setup[0]).count(), +aggr_pdcch_dl->getSetupStatus(),
            timings.start_t_dl_pdcchdl_run[0].count(), timings.end_t_dl_pdcchdl_run[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchdl_run[0] - timings.start_t_dl_pdcchdl_run[0]).count(), +aggr_pdcch_dl->getRunStatus(),
            (aggr_pdcch_dl->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pdcch_dl->getGPUSetupTime():0,
            (aggr_pdcch_dl->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_pdcch_dl->getGPURunTime():0,
            timings.start_t_dl_pdcchdl_setup[0].count(), timings.end_t_dl_pdcchdl_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchdl_compl[0] - timings.start_t_dl_pdcchdl_setup[0]).count(),
            timings.start_t_dl_pdcchdl_compl[0].count(), timings.end_t_dl_pdcchdl_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchdl_compl[0] - timings.start_t_dl_pdcchdl_compl[0]).count()
        );
    }

    if (aggr_pdcch_ul != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} Aggr PDCCH UL {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_pdcch_ul->getId(), getId(),
            timings.start_t_dl_pdcchul_setup[0].count(), timings.end_t_dl_pdcchul_setup[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchul_setup[0] - timings.start_t_dl_pdcchul_setup[0]).count(), +aggr_pdcch_ul->getSetupStatus(),
            timings.start_t_dl_pdcchul_run[0].count(), timings.end_t_dl_pdcchul_run[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchul_run[0] - timings.start_t_dl_pdcchul_run[0]).count(), +aggr_pdcch_ul->getRunStatus(),
            (aggr_pdcch_ul->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pdcch_ul->getGPUSetupTime():0,
            (aggr_pdcch_ul->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_pdcch_ul->getGPURunTime():0,
            timings.start_t_dl_pdcchul_setup[0].count(), timings.end_t_dl_pdcchul_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchul_compl[0] - timings.start_t_dl_pdcchul_setup[0]).count(),
            timings.start_t_dl_pdcchul_compl[0].count(), timings.end_t_dl_pdcchul_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pdcchul_compl[0] - timings.start_t_dl_pdcchul_compl[0]).count()
        );
    }

    if (aggr_pbch != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} Aggr PBCH {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_pbch->getId(), getId(),
            timings.start_t_dl_pbch_setup[0].count(), timings.end_t_dl_pbch_setup[0].count(),
            Time::NsToUs(timings.end_t_dl_pbch_setup[0] - timings.start_t_dl_pbch_setup[0]).count(), +aggr_pbch->getSetupStatus(),
            timings.start_t_dl_pbch_run[0].count(), timings.end_t_dl_pbch_run[0].count(),
            Time::NsToUs(timings.end_t_dl_pbch_run[0] - timings.start_t_dl_pbch_run[0]).count(), +aggr_pbch->getRunStatus(),
            (aggr_pbch->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_pbch->getGPUSetupTime():0,
            (aggr_pbch->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_pbch->getGPURunTime():0,
            timings.start_t_dl_pbch_setup[0].count(), timings.end_t_dl_pbch_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pbch_compl[0] - timings.start_t_dl_pbch_setup[0]).count(),
            timings.start_t_dl_pbch_compl[0].count(), timings.end_t_dl_pbch_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_pbch_compl[0] - timings.start_t_dl_pbch_compl[0]).count()
        );
    }

if (aggr_csirs != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} Aggr CSI-RS {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_csirs->getId(), getId(),
            timings.start_t_dl_csirs_setup[0].count(), timings.end_t_dl_csirs_setup[0].count(),
            Time::NsToUs(timings.end_t_dl_csirs_setup[0] - timings.start_t_dl_csirs_setup[0]).count(), +aggr_csirs->getSetupStatus(),
            timings.start_t_dl_csirs_run[0].count(), timings.end_t_dl_csirs_run[0].count(),
            Time::NsToUs(timings.end_t_dl_csirs_run[0] - timings.start_t_dl_csirs_run[0]).count(), +aggr_csirs->getRunStatus(),
            (aggr_csirs->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_csirs->getGPUSetupTime():0,
            (aggr_csirs->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_csirs->getGPURunTime():0,
            timings.start_t_dl_csirs_setup[0].count(), timings.end_t_dl_csirs_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_csirs_compl[0] - timings.start_t_dl_csirs_setup[0]).count(),
            timings.start_t_dl_csirs_compl[0].count(), timings.end_t_dl_csirs_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_csirs_compl[0] - timings.start_t_dl_csirs_compl[0]).count()
        );
    }

    if (aggr_dlbfw != nullptr) {
        NVLOGI_FMT(TAG, "[PHYDRV] SFN {}.{} Aggr DL_BFW {:x} Map {} times ===> "
            "[CPU CUDA Setup] {{ START: {} END: {} DURATION: {} us Setup STATUS: {} }} "
            "[CPU CUDA Run] {{ START: {} END: {} DURATION: {} us Run STATUS: {} }} "
            "[GPU Setup] {{ DURATION: {:.2f} us }} "
            "[GPU Run] {{ DURATION: {:.2f} us }} "
            "[Max GPU Time] {{ START: {} END: {} DURATION: {} us }} "
            "[CPU wait GPU Time] {{ START: {} END: {} DURATION: {} us }}",
            sfn, slot, aggr_dlbfw->getId(), getId(),
            timings.start_t_dl_bfw_setup[0].count(), timings.end_t_dl_bfw_setup[0].count(),
            Time::NsToUs(timings.end_t_dl_bfw_setup[0] - timings.start_t_dl_bfw_setup[0]).count(), +aggr_dlbfw->getSetupStatus(),
            timings.start_t_dl_bfw_run[0].count(), timings.end_t_dl_bfw_run[0].count(),
            Time::NsToUs(timings.end_t_dl_bfw_run[0] - timings.start_t_dl_bfw_run[0]).count(), +aggr_dlbfw->getRunStatus(),
            (aggr_dlbfw->getSetupStatus()==CH_SETUP_DONE_NO_ERROR)?aggr_dlbfw->getGPUSetupTime():0,
            (aggr_dlbfw->getRunStatus()==CH_RUN_DONE_NO_ERROR)?aggr_dlbfw->getGPURunTime():0,
            timings.start_t_dl_bfw_setup[0].count(), timings.end_t_dl_bfw_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_bfw_compl[0] - timings.start_t_dl_bfw_setup[0]).count(),
            timings.start_t_dl_bfw_compl[0].count(), timings.end_t_dl_bfw_compl[0].count(),
            Time::NsToUs(timings.end_t_dl_bfw_compl[0] - timings.start_t_dl_bfw_compl[0]).count()
        );
    }
}

void SlotMapDl::cleanupTimes()
{
    memset(&timings, 0, sizeof(struct dl_slot_timings));
}
