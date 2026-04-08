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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 17) // "DRV.FUNC_DL"
#define TAG_DRV_CUPHY_PTI ("DRV.CUPHY_PTI")

#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "context.hpp"
#include "time.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "phychannel.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "order_entity.hpp"
#include <unordered_map>
#include "aerial-fh-driver/oran.hpp"
#include <sched.h>
#include <unistd.h>
#include "task_instrumentation_v3.hpp"
#include "task_instrumentation_v3_factories.hpp"
#include "memtrace.h"
#include "nvlog_fmt.hpp"
#include "cupti_helper.hpp"
#include "cuphy_pti.hpp"
#include <optional>
#include "scf_5g_fapi.h"
#include "app_config.hpp"

#define SIGNAL_COMPLETION_W_EVENTS 1

int non_blocking_event_wait_with_timeout(cudaEvent_t event, t_ns timeout) {
    t_ns wait_start_t = Time::nowNs();
    t_ns wait_thresh_t(timeout);
    cudaError_t cuError = cudaErrorNotReady;
    while(cuError != cudaSuccess){
        cuError=cudaEventQuery(event);
        if(cuError != cudaSuccess && ((Time::nowNs()-wait_start_t)>wait_thresh_t))
        {
            //Timeout has occurred
            return -1;
        }
        //std::this_thread::sleep_for(std::chrono::microseconds(1)); // could add and adapt granularity of sleep to see if it helps
    }

    //Success
    return 0;
}

/**
 * Populate compression parameters array based on cell compression methods
 * 
 * Iterates through cells and populates the compression_params array indexed by
 * the compression method from aerial_fh::UserDataCompressionMethod enum.
 * 
 * @param[out] cparams_array Array of compression_params structures, one per compression method
 * @param[in] slot_map Pointer to the slot map containing cell information
 * @param[in] pdctx Pointer to the PHY driver context
 * @param[in] first_cell Index of the first cell to process
 * @param[in] num_cells Number of cells to process
 * @return Total number of valid cells processed across all compression methods
 */
int populateCompressionParams(
    std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array,
    SlotMapDl* slot_map,
    PhyDriverCtx* pdctx,
    const int first_cell,
    const int num_cells)
{
    using CompMethod = aerial_fh::UserDataCompressionMethod;
    
    // Cell index counters for each compression method
    std::array<int, NUM_USER_DATA_COMPRESSION_METHODS> cell_idx{};
    int valid_cell_count{};
    
    // Initialize compression params for each method
    for(std::size_t i = 0; i < NUM_USER_DATA_COMPRESSION_METHODS; ++i)
    {
        cparams_array[i].num_cells = 0;
        cparams_array[i].gpu_comms = pdctx->gpuCommDlEnabled();
    }
    
    // Populate compression params for each cell
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        
        const uint8_t comp_meth = cell_ptr->getDLCompMeth();
        
        // Validate compression method index
        if(comp_meth >= NUM_USER_DATA_COMPRESSION_METHODS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, 
                "Invalid compression method {} for cell {}", comp_meth, i);
            continue;
        }
        
        // Map NO_COMPRESSION to BLOCK_FLOATING_POINT array for processing
        // Both methods can be handled by the same kernel path
        // comp_meth[] stays the cell's real method (not array_idx) for kernel_compress.
        const uint8_t array_idx = (comp_meth == static_cast<uint8_t>(CompMethod::NO_COMPRESSION)) 
                                    ? static_cast<uint8_t>(CompMethod::BLOCK_FLOATING_POINT) 
                                    : comp_meth;
        
        compression_params& cparams = cparams_array[array_idx];
        int& curr_cell_idx = cell_idx[array_idx];
        
        // Build the compression parameters structure
        cparams.input_ptrs[curr_cell_idx]   = dlbuf->getBufD();
        cparams.prb_ptrs[curr_cell_idx]     = pdctx->gpuCommDlEnabled() ? dlbuf->getPrbPtrs() : nullptr;
        cparams.beta[curr_cell_idx]         = cell_ptr->getBetaDlPowerScaling();
        cparams.comp_meth[curr_cell_idx]    = comp_meth;
        cparams.bit_width[curr_cell_idx]    = cell_ptr->getDLBitWidth();
        cparams.num_antennas[curr_cell_idx] = cell_ptr->geteAxCNumPdsch();
        cparams.max_num_prb_per_symbol[curr_cell_idx] = cell_ptr->getDLGridSize();
        cparams.num_prbs[curr_cell_idx]     = ORAN_MAX_PRB * ORAN_PUSCH_SYMBOLS_X_SLOT * cell_ptr->geteAxCNumPdsch();
        
        // Initialize mod_compression config pointers if ModComp is enabled
        if(comp_meth == static_cast<uint8_t>(CompMethod::MODULATION_COMPRESSION))
        {
            cparams.mod_compression_config[curr_cell_idx] = dlbuf->getModCompressionConfig();
        }
        
        curr_cell_idx++;
        cparams.num_cells++;
        valid_cell_count++;
        NVLOGI_FMT(TAG, "Compression params for cell {} (comp_meth: {}): num_cells: {}, gpu_comms: {}, curr_cell_idx: {}",
            i, comp_meth, cparams.num_cells, cparams.gpu_comms, curr_cell_idx);
    }
    
    return valid_cell_count;
}

int task_work_function_debug(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "Debug Task", 13);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    int                                                                          sfn = 0, slot = 0;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    // Compression DL Buf is the first dlbuf FIXME: create a compression object
    DLOutputBuffer *compression_dlbuf = nullptr;
    DLOutputBuffer *prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);
    int num_dlc_tasks = get_num_dlc_tasks(pdctx->getNumDLWorkers(),pdctx->gpuCommEnabledViaCpu(),pdctx->getmMIMO_enable());
    std::array<std::string,MAX_NUM_OF_NIC_SUPPORTED> nic_name_arr;
    t_ns t_1,t_2;
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }

    //Condition check for DL BFW only scheduling on the slot (return immediately)
    if(slot_map->aggr_dlbfw && (slot_map->getNumCells()==0))
    {
        ti.add("End Task");
        return 0;
    }

    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        DLOutputBuffer * dlbuf = slot_map->aggr_dlbuf_list[i];
        Cell * cell_ptr = slot_map->aggr_cell_list[i];

        if(dlbuf == nullptr) continue;
        if(cell_ptr == nullptr) continue;

        auto nic_index = cell_ptr->getNicIndex();

        if (!compression_dlbuf) {
            compression_dlbuf = dlbuf;
        }
        if (!prepare_tx_dlbuf_per_nic[nic_index])
            prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;
        nic_name_arr[nic_index] = cell_ptr->getNicName();
    }

    // Extra check to protect against segfault in case the SlotMapDl::release() for that slot is called before this task_work_function_debug completes.
    // If that happens, the slot_map->getNumCells() will be 0, and the compression_dlbuf will stay nullptr, as the loop above won't execute.
    if(compression_dlbuf == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got nullptr for compression_dlbuf for SFN {}.{} (slot Map has {} cells)", task_num , slot_map->getId(), sfn, slot, slot_map->getNumCells());
        return -1;
    }

    ti.add("Wait DL Gpu Comm End"); // Synchronize end of GPU Comm Task
    if(slot_map->waitDlGpuCommEnd() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitDlGpuCommEnd returned error for SFN {}.{} Map {}", sfn, slot, slot_map->getId());
        EXIT_L1(EXIT_FAILURE);
    }    

    //Wait for PrePrepare to start
    ti.add("PrePrepare Stop Wait");
    if(prepare_tx_dlbuf_per_nic[0]->waitPrePrepareStop() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got PrePrepare Stop wait error", task_num , slot_map->getId());
        return -1;
    }
    else
    {
        t_1=Time::nowNs();
    }

    //Wait for compression to start
    ti.add("Compression Start Wait");
    if(compression_dlbuf->waitCompressionStart() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got COMPRESSION DL wait start error", task_num , slot_map->getId());
        return -1;
    }
    else
    {
        t_2=Time::nowNs();
    }

    //Wait for compression to stop
    ti.add("Compression Wait");
    PUSH_RANGE_PHYDRV("DL WAIT COMP", 3);
    slot_map->timings.start_t_dl_compression_compl = Time::nowNs();
    if(compression_dlbuf->waitCompressionStop() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got COMPRESSION DL wait error", task_num , slot_map->getId());
        return -1;
    }
    slot_map->timings.end_t_dl_compression_compl = Time::nowNs();
    POP_RANGE_PHYDRV;

    //ti.add("Prepare Tracing"); // Could consider adding this extra tag here to measure effect of event queries
    //NVLOGC_FMT(TAG, "Prepare tracing start");
    t_ns wait_thresh_t(pdctx->getcuphy_dl_channel_wait_th());

    if(pdctx->enablePrepareTracing()) {
        //Note: prepare events are always before compression execution, therefore no waits are needed here
        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
        {
            if(prepare_tx_dlbuf_per_nic[i])
            {
                int failure = non_blocking_event_wait_with_timeout(prepare_tx_dlbuf_per_nic[i]->getPrepareStopEvt(),wait_thresh_t);
                if(failure != 0)
                {
                    slot_map->timings.prepare_execution_duration1[i] = 0.0;
                    slot_map->timings.prepare_execution_duration2[i] = 0.0;
                    slot_map->timings.prepare_execution_duration3[i] = 0.0;                    
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"Debug Task Prepare kernel wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),wait_thresh_t.count());
                    return -1;
                }
                else {
                    slot_map->timings.prepare_execution_duration1[i] = prepare_tx_dlbuf_per_nic[i]->getPrepareExecutionTime1();
                    slot_map->timings.prepare_execution_duration2[i] = prepare_tx_dlbuf_per_nic[i]->getPrepareExecutionTime2();
                    slot_map->timings.prepare_execution_duration3[i] = prepare_tx_dlbuf_per_nic[i]->getPrepareExecutionTime3();
                    slot_map->timings.prePrepare_to_compression_gap[i] = (float)(t_2.count()-t_1.count())/1000.0;                
                }
            }
        }
    } else {
        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
        {
            slot_map->timings.prepare_execution_duration1[i] = 0.0;
            slot_map->timings.prepare_execution_duration2[i] = 0.0;
            slot_map->timings.prepare_execution_duration3[i] = 0.0;
        }
    }

    slot_map->timings.channel_to_compression_gap = compression_dlbuf->getChannelToCompressionGap();
    slot_map->timings.compression_execution_duration = compression_dlbuf->getCompressionExecutionTime();
    //NVLOGC_FMT(TAG, "Prepare tracing end");

    ti.add("Trigger synchronize");
    if(pdctx->gpuCommEnabledViaCpu())
    {
        if(slot_map->waitDlCpuDoorBellTaskDone() < 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitDlCpuDoorBellTaskDone returned error");
            return -1;
        }
    }
    else {
        //Note: Since we have already waited on compression, all prepare work has been submitted and we can access prepare events
        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
        {
            if (prepare_tx_dlbuf_per_nic[i])
            {
                int failure = non_blocking_event_wait_with_timeout(prepare_tx_dlbuf_per_nic[i]->getTxEndEvt(),wait_thresh_t);
                if(failure != 0) {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"Debug Task Trigger kernel wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),wait_thresh_t.count());
                    return -1;
                }
                else
                {
                    if(pdctx->enableDlCqeTracing() && (slot_map->tx_v_for_slot_map[i].size!=0))
                    {
                        t_ns trigger_time = Time::nowNs();
                        fhproxy->setTriggerTsGpuComm(nic_name_arr[i],((sfn%256)*20+slot), trigger_time.count());
                        fhproxy->triggerCqeTracerCb(nic_name_arr[i],&slot_map->tx_v_for_slot_map[i]);
                    }
                }
            }
        }    
    }    

#ifdef CUPHY_PTI_ENABLE_TRACING
    struct cuphy_pti_all_stats_t* stats;
    cuphy_pti_get_record_all_activities(stats);

    NVLOGI_FMT(TAG_DRV_CUPHY_PTI,"<DL,{},{},{}> PrePrep:{},{},{} Prep:{},{},{} Trigger:{},{},{}",
               sfn,
               slot,
               cpu,
               stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREPREP]-stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREPREP],
               stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREP],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREP],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_PREP]-stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_PREP],
               stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_TRIGGER],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_TRIGGER],
               stats->dh_gpu_stop_times[CUPHY_PTI_ACTIVITY_TRIGGER]-stats->dh_gpu_start_times[CUPHY_PTI_ACTIVITY_TRIGGER]
    );
    memset(stats->dh_gpu_start_times, 0, sizeof(uint64_t)*CUPHY_PTI_ACTIVITIES_MAX);
    memset(stats->dh_gpu_stop_times, 0, sizeof(uint64_t)*CUPHY_PTI_ACTIVITIES_MAX);
#endif

    ti.add("End Task");

    return 0;
}

int task_work_function_dl_aggr_bfw(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task BFW", 13);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    t_ns                                                                         start_t_1, start_t_2, start_tx;
    t_ns end_time;
    int                                                                          sfn = 0, slot = 0;
#if 1
    t_ns run_start_timeout_thresh_t(DL_BFW_TX_RUN_START_THRESHOLD_MS * NS_X_MS);
    t_ns run_completion_timeout_thresh_t(DL_BFW_TX_RUN_COMPLETION_THRESHOLD_MS * NS_X_MS);
#else
    t_ns run_start_timeout_thresh_t(NS_X_MS);
    t_ns run_completion_timeout_thresh_t(NS_X_MS);
#endif
    bool dlbfw_started = false;
    bool dlbfw_finished = false;
    bool dlbfw_timeout = false;
    t_ns start_t;
    PhyDlBfwAggr* dlbfw = slot_map->aggr_dlbfw;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    struct slot_command_api::dl_slot_callbacks dl_cb;
    std::array<uint32_t,DL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    if(dlbfw == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No DLBFW in SlotMap DL");
        goto exit_error;
    }

    //NVLOGI(TAG,"DL Task (BFW) started CPU %u for Map %d, SFN slot (%d,%d)\n", cpu, slot_map->getId(), sfn, slot);

    start_t_1 = Time::nowNs();

    /////////////////////////////////////////////////////////////////////////////////////
    //// cuPHY Setup + Run
    /////////////////////////////////////////////////////////////////////////////////////

    PUSH_RANGE_PHYDRV((std::string("DL BFW") + std::to_string(slot_map->getId())).c_str(), 2);
    slot_map->timings.start_t_dl_bfw_setup[0] = Time::nowNs();
    if (dlbfw) {
        try
        {
            ti.add("Cuda Setup");
            if (dlbfw->setup(slot_map->aggr_cell_list))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL BFW Setup returned error for Map {} SFN {}.{}",slot_map->getId(), sfn, slot);
                dlbfw->setSetupStatus(CH_SETUP_DONE_ERROR);
                //goto exit_error;
            }
            else
                dlbfw->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            slot_map->timings.end_t_dl_bfw_setup[0] = Time::nowNs();
            slot_map->timings.start_t_dl_bfw_run[0] = slot_map->timings.end_t_dl_bfw_setup[0];

            ti.add("Cuda Run");
            if(dlbfw->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL BFW run returned error for Map {} SFN {}.{}",slot_map->getId(), sfn, slot);
                dlbfw->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_DL_BFW_CVI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
                //goto exit_error;
            }
            else
                dlbfw->setRunStatus(CH_RUN_DONE_NO_ERROR);

            ti.add("Signal Completion");
            if(dlbfw->signalRunCompletionEvent(false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL BFW signalCompletion returned error SFN {}.{}\n", sfn, slot);
                goto exit_error;
            }
            pdctx->recordDlBFWCompletion(slot);
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_dl_bfw_run[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Slot End Task");
    slot_map->addSlotEndTask();
    start_t_2 = Time::nowNs();

    ti.add("Wait Run Start");
    start_t = Time::nowNs();
    do
    {
        dlbfw_started = (dlbfw->waitStartRunEventNonBlocking()==1);
        dlbfw_timeout = (Time::nowNs() - start_t > run_start_timeout_thresh_t);
    } while(!dlbfw_started && !dlbfw_timeout);
    if(dlbfw_timeout)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: DL BFW task waiting to start more than {} ns for Slot Map {} SFN {}.{}",
                        run_start_timeout_thresh_t.count(),slot_map->getId(), sfn, slot);
        goto exit_error;
    }

    ti.add("Wait Run Completion");
    start_t = Time::nowNs();
    slot_map->timings.start_t_dl_bfw_compl[0] = start_t;
    do
    {
        dlbfw_finished = (dlbfw->waitRunCompletionEventNonBlocking()==1);
        dlbfw_timeout = (Time::nowNs() - start_t > run_completion_timeout_thresh_t);
    } while(!dlbfw_finished && !dlbfw_timeout);
    if(dlbfw_timeout)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: DL BFW task waiting for run completion event more than {} ns for Slot Map {} SFN {}.{}",
                        run_completion_timeout_thresh_t.count(),slot_map->getId(), sfn, slot);
        goto exit_error;
    }
    else
    {
        slot_map->timings.end_t_dl_bfw_compl[0] = Time::nowNs();

        dlbfw->validate();

        slot_map->timings.start_t_dl_bfw_cb[0] = Time::nowNs();
        dlbfw->callback();
        slot_map->timings.end_t_dl_bfw_cb[0] = Time::nowNs();
    }

    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " DLBFW Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " us on CPU " << (int)cpu;

    return 0;

//FIXME: abort the whole DL slot in case of error
exit_error:
    return -1;
}


int task_work_function_dl_aggr_1_pdsch(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task PDSCH", 13);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    t_ns                                                                         start_t_1, start_t_2, start_tx;
    t_ns end_time;
    int                                                                          sfn = 0, slot = 0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    uint32_t cpu;
    struct slot_command_api::dl_slot_callbacks dl_cb;
    std::array<uint32_t,DL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    if(pdsch == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No PDSCH channel in SlotMap DL");
        goto exit_error;
    }

    //NVLOGI_FMT(TAG,"DL Task 1(PDSCH) started CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);


    start_t_1 = Time::nowNs();

    /////////////////////////////////////////////////////////////////////////////////////
    //// cuPHY Setup + Run
    /////////////////////////////////////////////////////////////////////////////////////

    PUSH_RANGE_PHYDRV((std::string("DL PDSCH") + std::to_string(slot_map->getId())).c_str(), 2);
    slot_map->timings.start_t_dl_pdsch_setup[0] = Time::nowNs();
    if (pdsch) {
        try
        {
            ti.add("Cuda Setup");
            if (pdsch->setup(slot_map->aggr_cell_list, slot_map->aggr_dlbuf_list))
            {
                // Logging takes place in phypdsch_aggr.cpp
                NVLOGW_FMT(TAG, "PDSCH setup returned error for Map {}",slot_map->getId());
                pdsch->setSetupStatus(CH_SETUP_DONE_ERROR);
                //goto exit_error;
            }
            else
                pdsch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            slot_map->timings.end_t_dl_pdsch_setup[0] = Time::nowNs();
            slot_map->timings.start_t_dl_pdsch_run[0] = slot_map->timings.end_t_dl_pdsch_setup[0];

            ti.add("Cuda Run");
            if(pdsch->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDSCH run returned error for Map {}",slot_map->getId());
                pdsch->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_DL_TTI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
                //goto exit_error;
            }
            else
                pdsch->setRunStatus(CH_RUN_DONE_NO_ERROR);
#if SIGNAL_COMPLETION_W_EVENTS

            ti.add("Signal Completion");
            if(pdsch->signalRunCompletionEvent(false))
#else
            if(pdsch->signalRunCompletion())
#endif
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDSCH signalCompletion returned error");
                goto exit_error;
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_dl_pdsch_run[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Slot Channel End");
    slot_map->addSlotChannelEnd();
    ti.add("Signal Slot End Task");
    slot_map->addSlotEndTask();
    start_t_2 = Time::nowNs();

    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " PDSCH Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " us on CPU " << (int)cpu;

    return 0;

//FIXME: abort the whole DL slot in case of error
exit_error:
    return -1;
}



int task_work_function_dl_aggr_control(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task Control", 13);
    ti.add("Start Task");
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    PhyPdcchAggr* pdcch_dl = slot_map->aggr_pdcch_dl;
    PhyPdcchAggr* pdcch_ul  = slot_map->aggr_pdcch_ul;
    PhyPbchAggr*  pbch      = slot_map->aggr_pbch;
    PhyCsiRsAggr* csirs     = slot_map->aggr_csirs;
    int tasks = 0;
    int ret;
    int sfn  = slot_map->getSlot3GPP().sfn_;
    int slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    std::string map_str = std::to_string(slot_map->getId());
    struct slot_command_api::dl_slot_callbacks dl_cb;
    std::array<uint32_t,DL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;

    // Set to true to skip run, if a misconfig. or any other issue during setup happens

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }

    if (pdcch_dl == nullptr &&
        pdcch_ul == nullptr &&
        pbch     == nullptr &&
        csirs    == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Control task called with no active control channels");
        return -1;
    }

    try {
        // Setup for each control channel
        ti.add("PDCCH_DL Setup");
        if(pdcch_dl != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PDCCH DL Setup") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pdcchdl_setup[0] = Time::nowNs();
            if (pdcch_dl->setup(slot_map->aggr_cell_list, slot_map->aggr_dlbuf_list)) {
                NVLOGW_FMT(TAG, "PDCCH DL setup returned error for Map {}",slot_map->getId());
                pdcch_dl->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                pdcch_dl->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pdcchdl_setup[0] = Time::nowNs();
            tasks++;
        }

        ti.add("PDCCH_UL Setup");
        if (pdcch_ul != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PDCCH UL Setup") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pdcchul_setup[0] = Time::nowNs();
            if (pdcch_ul->setup(slot_map->aggr_cell_list, slot_map->aggr_dlbuf_list)) {
                NVLOGW_FMT(TAG, "PDCCH UL setup returned error for Map {}", slot_map->getId());
                pdcch_ul->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                pdcch_ul->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pdcchul_setup[0] = Time::nowNs();
            tasks++;
        }

        ti.add("PBCH Setup");
        if (pbch != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PBCH Setup") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pbch_setup[0] = Time::nowNs();
            if (pbch->setup(slot_map->aggr_dlbuf_list, slot_map->aggr_cell_list)) {
                NVLOGW_FMT(TAG,"PBCH setup returned error for Map {}",slot_map->getId());
                pbch->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                pbch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pbch_setup[0] = Time::nowNs();
            tasks++;
        }

        ti.add("CSIRS Setup");
        if (csirs != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL CSIRS Setup") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_csirs_setup[0] = Time::nowNs();
            if (csirs->setup(slot_map->aggr_dlbuf_list,  slot_map->aggr_cell_list)) {
                NVLOGW_FMT(TAG, "CSI RS setup returned error for Map {}",slot_map->getId());
                csirs->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                csirs->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_csirs_setup[0] = Time::nowNs();
            tasks++;
        }

        // Run
        ti.add("PDCCH_DL Run");
        if (pdcch_dl != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PDCCH DL Run") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pdcchdl_run[0] = Time::nowNs();
            if (pdcch_dl->run()) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH DL run returned error for Map {}",slot_map->getId());
                pdcch_dl->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_DL_TTI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
            }
            else
                pdcch_dl->setRunStatus(CH_RUN_DONE_NO_ERROR);
    #if SIGNAL_COMPLETION_W_EVENTS
            if(pdcch_dl->signalRunCompletionEvent(false))
    #else
            if(pdcch_dl->signalRunCompletion())
    #endif
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH DL signalCompletion returned error");
                return -1;
            }

            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pdcchdl_run[0] = Time::nowNs();
        }

        ti.add("PDCCH_UL Run");
        if (pdcch_ul != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PDCCH UL Run") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pdcchul_run[0] = Time::nowNs();
            if (pdcch_ul->run()) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH UL run returned error for Map {}",slot_map->getId());
                pdcch_ul->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_DCI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
            }
            else
                pdcch_ul->setRunStatus(CH_RUN_DONE_NO_ERROR);
    #if SIGNAL_COMPLETION_W_EVENTS
            if(pdcch_ul->signalRunCompletionEvent(false))
    #else
            if(pdcch_ul->signalRunCompletion())
    #endif
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH UL signalCompletion returned error");
                return -1;
            }

            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pdcchul_run[0] = Time::nowNs();
        }

        ti.add("PBCH Run");
        if (pbch != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL PBCH Run") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_pbch_run[0] = Time::nowNs();
            if (pbch->run()) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PBCH run returned error for Map {}",slot_map->getId());
                pbch->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_DL_TTI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
            }
            else
                pbch->setRunStatus(CH_RUN_DONE_NO_ERROR);

    #if SIGNAL_COMPLETION_W_EVENTS
            if(pbch->signalRunCompletionEvent(false))
    #else
            if(pbch->signalRunCompletion())
    #endif
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PBCH signalCompletion returned error");
                return -1;
            }

            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_pbch_run[0] = Time::nowNs();
        }

        ti.add("CSIRS Run");
        if (csirs != nullptr) {
            PUSH_RANGE_PHYDRV((std::string("DL CSIRS Run") + map_str).c_str(), 2);
            slot_map->timings.start_t_dl_csirs_run[0] = Time::nowNs();
            if (csirs->run()) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "CSIRS run returned error for Map {}",slot_map->getId());
                csirs->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getDlCb(dl_cb))
                {
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_DL_TTI_REQUEST,SCF_ERROR_CODE_L1_DL_CH_ERROR,cell_idx_list,cell_count);
                }
            }
            else
                csirs->setRunStatus(CH_RUN_DONE_NO_ERROR);

    #if SIGNAL_COMPLETION_W_EVENTS
            if(csirs->signalRunCompletionEvent(false))
    #else
            if(csirs->signalRunCompletion())
    #endif
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "CSIRS signalCompletion returned error");
                return -1;
            }

            POP_RANGE_PHYDRV;
            slot_map->timings.end_t_dl_csirs_run[0] = Time::nowNs();
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()

    ti.add("Signal Slot Channel End");
    slot_map->addSlotChannelEnd();
    ti.add("Signal Slot End Task");
    slot_map->addSlotEndTask();

    ti.add("End Task");

    return 0;
}

int task_work_function_dl_fh_cb(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task FH Callback", 3);
    ti.add("Start Task");
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    struct slot_command_api::dl_slot_callbacks dl_cb;

    int sfn  = slot_map->getSlot3GPP().sfn_;
    int slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    pdctx->getDlCtx()->setCtx();

    uint32_t cpu;
    int ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }

    const bool valid = pdctx->getDlCb(dl_cb); 

    for (int cell = slot_map->getNumCells()-1; cell >= 0; --cell) {
        const Cell* cell_ptr = slot_map->aggr_cell_list[cell];
        if(cell_ptr == nullptr) continue;

        if (valid) {
            dl_cb.fh_prepare_callback_fn(dl_cb.fh_prepare_callback_fn_context, slot_map->aggr_slot_params->cgcmd, cell_ptr->getIdx());
        }
        slot_map->setCellFHCBDone(cell_ptr->getIdx()); 
    }
    
    slot_map->setFHCBDone();
    slot_map->addSlotEndTask();

    ti.add("End Task");

    return 0;
}

int task_work_function_cplane(Worker* worker, void* param, int task_num, int first_cell, int num_tasks)
{
    char name[64];
    sprintf(name, "DL Task C-Plane %d", task_num + 1);
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, name, 14);
    ti.add("Start Task");
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    Cell * cell_ptr = nullptr;
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_tx;
    int sfn = slot_map->getSlot3GPP().sfn_;
    int slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    struct slot_command_api::dl_slot_callbacks dl_cb;
    std::array<uint32_t,DL_MAX_CELLS_PER_SLOT> cplane_tx_err_cell_idx_list={};
    int cplane_tx_err_cell_count=0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;

    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);


    uint32_t cpu;
    int ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }

    pdctx->getDlCtx()->setCtx();


    // Wait for the callback above to finish. If this is the task calling the callback (the first),
    // it will already be complete.


    /////////////////////////////////////////////////////////////////////////////////////
    //// C-Plane
    /////////////////////////////////////////////////////////////////////////////////////
    if (!pdctx->isCPlaneDisabled())
    {
        PUSH_RANGE_PHYDRV((std::string("DL CPLANE") + std::to_string(slot_map->getId())).c_str(), 2);
        try
        {

            ti.add("Wait DLBFW Completion");
            int prevSlotDlBfwCompStatus = 1;
            if(pdctx->getmMIMO_enable() && task_num < slot_map->getNumCells())
            {
                // Determine which slot to check for previous slot DL BFW completion
                int previous_slot = (slot_oran_ind.osfid_*2 + slot_oran_ind.oslotid_ -1 +  SLOTS_PER_FRAME)%SLOTS_PER_FRAME;

                //Perform wait based on first cell's T1aMaxCpDlNs
                t_ns current_time = Time::nowNs();
                cell_ptr = slot_map->aggr_cell_list[task_num];
                start_tx = slot_map->getTaskTsExec(0) + t_ns(Cell::getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead()) - (t_ns(cell_ptr->getTcpAdvDlNs()) + t_ns(cell_ptr->getT1aMaxUpNs()));
                t_ns deadline_time = start_tx - t_ns(pdctx->getSendCPlane_dlbfw_backoff_th_ns());
                prevSlotDlBfwCompStatus = pdctx->queryDlBFWCompletion(previous_slot);
                while(!prevSlotDlBfwCompStatus)
                {
                    current_time = Time::nowNs();
                    if(current_time >= deadline_time)
                    {
                        // DL BFW pipeline is not completed before sendCPlane
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: Previous slot DLBFW pipeline is not complete! oframe_id_ {} osfid_ {} oslotid_ {} error code {}",
                                                                    slot_oran_ind.oframe_id_, slot_oran_ind.osfid_, slot_oran_ind.oslotid_, (int)pdctx->queryDlBFWCompletion_v2(previous_slot));
                        break;
                    }
                    prevSlotDlBfwCompStatus = pdctx->queryDlBFWCompletion(previous_slot);
                }

            }
            
            ti.add("CPlane Prepare");
            const bool valid = pdctx->getDlCb(dl_cb);
            const uint8_t dlc_packing_scheme = pdctx->get_dlc_core_packing_scheme();
            
            bool send_nonbfw[MAX_CELLS_PER_SLOT]{}; 

            for(int i = 0; i < slot_map->getNumCells(); i++)
            {
                // Apply packing scheme filter
                if (dlc_packing_scheme == 0) {
                    // Scheme 0 (default): strided iteration - task_num handles cells task_num, task_num+num_tasks, etc.
                    if ((i % num_tasks) != task_num) continue;
                } else if (dlc_packing_scheme == 1) {
                    // Scheme 1 (fixed per-cell): check if cell's dlc_core_index matches this task_num
                    Cell* check_cell = slot_map->aggr_cell_list[i];
                    if (check_cell == nullptr) continue;
                    if (check_cell->getDlcCoreIndex() != task_num) continue;
                }
                // Note: Scheme 2 (dynamic workload-based) is not yet supported and validated during init

                if(pdsch)
                {
                    if(pdctx->getUeMode()) //Skip sending DLC plane packets in UE mode (Spec effeciency)
                    {
                        break;
                    }
                }
                cell_ptr = slot_map->aggr_cell_list[i];
                if(cell_ptr == nullptr) continue;
                
                if (slot_map->waitCellFHCBDone(cell_ptr->getIdx()) < 0) {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitCellFHCBDone cell:{} returned error", cell_ptr->getIdx()); 
                    continue; // Continue to process subsequent cells
                }

                slot_map->timings.start_t_dl_cplane[i] = Time::nowNs();
                start_tx = slot_map->getTaskTsExec(0) + t_ns(Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) - (t_ns(cell_ptr->getTcpAdvDlNs()) + t_ns(cell_ptr->getT1aMaxUpNs()));

                // auto slot_info = slot_map->aggr_slot_info[i];
                uint8_t* bfw_header = nullptr;
                uint8_t count = 0;

                // To ease any pressure in NIC w/ C-Plane BFW packets - the TX start time is offset by 
                // (cell_ID * (TX_WINDOW) / NUM_CELLS). 
                uint64_t tx_cell_start_ofs_ns = pdctx->getFhProxy()->getDlcBfwEnableDividePerCell() ? 
                                   i * ((cell_ptr->getT1aMaxCpDlNs() - cell_ptr->getT1aMinCpDlNs()) / slot_map->getNumCells()) :
                                   0; 
                
                // tis facilitates nested calls to add additional subtask TIs
                ti_subtask_info tis{}; 
                if(slot_map->aggr_slot_info[i]) //slot_info != nullptr)
                {
                    if((ret=fhproxy->prepareCPlaneInfo(
                        cell_ptr->getIdx(),
                        cell_ptr->getRUType(),
                        cell_ptr->getPeerId(),
                        cell_ptr->getDLCompMeth(),
                        start_tx,
                        tx_cell_start_ofs_ns,
                        DIRECTION_DOWNLINK,
                        slot_oran_ind, //it_ulchannels.second.slot_oran_ind,
                        *(slot_map->aggr_slot_info[i]), //*slot_map->aggr_slot_info[i], //*slot_info,
                        0, slot_map->getDynBeamIdOffset(), 0, 0,&bfw_header,t_ns(0), prevSlotDlBfwCompStatus, tis))!=SEND_CPLANE_NO_ERROR)
                    {
                        //Do not return with a negative error code if DLC error
                        cplane_tx_err_cell_idx_list[cplane_tx_err_cell_count++]=cell_ptr->getIdx();
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL C-plane prepare error for task num {} error type {} Map {}",i,ret,slot_map->getId());
                        // If the first step failed, none of the sendCPlaneMMIMO will execute, so setup any unblocking signals here. 
                        slot_map->atom_dl_cplane_info_for_uplane_rdy_count.fetch_add(1);
                        slot_map->aggr_slot_info[i]->section_id_ready.store(true);
                    }

                    if (ret == SEND_CPLANE_NO_ERROR && pdctx->getmMIMO_enable()) {
                        // Construct & send BFW packets first as they have stricter deadline reqs
                        if ((ret = fhproxy->sendCPlaneMMIMO(
                                   true /* BFW packets */,
                                   cell_ptr->getIdx(),
                                   cell_ptr->getPeerId(),
                                   DIRECTION_DOWNLINK,
                                   tis)) != SEND_CPLANE_NO_ERROR) {
                            cplane_tx_err_cell_idx_list[cplane_tx_err_cell_count++]=cell_ptr->getIdx();
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL C-plane MMIMO BFW Send error for task num {} error type {} Map {}",i,ret,slot_map->getId());
                        }
                        send_nonbfw[i] = true; // Pending sending non-BFW packets.
                    }

                    // Appends any subtask instrumentation added by the sendCPlane/sendCPlaneMMIMO call.
                    ti.appendList(tis);
                }

                if(valid)
                {
                    if(bfw_header != nullptr)
                    {
                        dl_cb.fh_bfw_coeff_usage_done_fn(dl_cb.fh_bfw_coeff_usage_done_fn_context, bfw_header);
                    }
                }

                if (!pdctx->getmMIMO_enable()) {
                    // For nonMMIMO case, this is the end time for when a cell's C-Plane processing is done.
                    slot_map->timings.end_t_dl_cplane[i] = Time::nowNs();
                }
            } // for(int i = 0; i < slot_map->getNumCells(); i++)
            
            // Handle the pending non-BFW packet construction & sending. 
            for(int i = 0; i < slot_map->getNumCells(); i++)
            {
                if (send_nonbfw[i]) 
                {
                    ti_subtask_info tis{}; 
                    cell_ptr = slot_map->aggr_cell_list[i];
                    if ((ret=fhproxy->sendCPlaneMMIMO(false /* nonBFW packets */, cell_ptr->getIdx(), cell_ptr->getPeerId(), DIRECTION_DOWNLINK, tis)) != SEND_CPLANE_NO_ERROR) {
                        cplane_tx_err_cell_idx_list[cplane_tx_err_cell_count++]=cell_ptr->getIdx();
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL C-plane MMIMO non-BFW Send error for task num {} error type {} Map {}",i,ret,slot_map->getId()); 
                    }

                    // Appends any subtask instrumentation added by the sendCPlaneMMIMO call.
                    ti.appendList(tis);
                }
                // For MMIMO case, this is the end time for when a cell's C-Plane processing is done.
                slot_map->timings.end_t_dl_cplane[i] = Time::nowNs();
                // Now since the cell's workload is completed successfully (Prepare, BFW and non-BFW) - setup unblocking signals.
                slot_map->atom_dl_cplane_info_for_uplane_rdy_count.fetch_add(1);
                slot_map->aggr_slot_info[i]->section_id_ready.store(true);
            }

            if(valid)
            {
                if(cplane_tx_err_cell_count>0)
                {
                    auto* pdcch_ul = slot_map->aggr_pdcch_ul;
                    auto* bfw = slot_map->aggr_dlbfw;
                    auto msg_id = SCF_FAPI_DL_TTI_REQUEST;
                    if (!pdcch_ul)
                    {
                        msg_id = SCF_FAPI_UL_DCI_REQUEST;
                    } else if (!bfw)
                    {
                        msg_id = SCF_FAPI_DL_BFW_CVI_REQUEST;
                    }
                    dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_ind,msg_id,SCF_ERROR_CODE_L1_DL_CPLANE_TX_ERROR,cplane_tx_err_cell_idx_list,cplane_tx_err_cell_count);
                }
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
        POP_RANGE_PHYDRV
    } // if (!pdctx->isCPlaneDisabled)

    ti.add("Signal completion");
    
    slot_map->incDLCDone();
    slot_map->addSlotEndTask();

    // Signal C-plane completion for this task when DL affinity is disabled
    // This allows task_work_function_dl_aggr_2_gpu_comm_prepare to synchronize. Applicable only when mMIMO is enabled.
    if (pdctx->getmMIMO_enable() && pdctx->get_enable_dl_core_affinity() == 0) {
        slot_map->setCplaneDoneForTask(task_num);
    }

    ti.add("End Task");

    return 0;
}

int task_work_function_dl_aggr_1_compression(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task Compression", 14);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_t_1, start_t_2, start_tx;
    t_ns end_time;
    int                                                                          sfn = 0, slot = 0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;
    PhyPdcchAggr* pdcch_dl = slot_map->aggr_pdcch_dl;
    PhyPdcchAggr* pdcch_ul = slot_map->aggr_pdcch_ul;
    PhyPbchAggr*  pbch = slot_map->aggr_pbch;
    PhyCsiRsAggr* csirs = slot_map->aggr_csirs;
    PhyDlBfwAggr* dlbfw = slot_map->aggr_dlbfw;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    struct slot_command_api::dl_slot_callbacks dl_cb;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }

    pdctx->dl_aggr_compression_task.lock(); //Mutex lock to ensure that Compression kernel for future slots dont get launched before compression kernel for present slot
    //NVLOGI_FMT(TAG,"DL Task 1(Compression) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);
    ti.add("Buffer Selection");
    pdctx->getDlCtx()->setCtx();
    cudaStream_t first_strm;
    int local_cell_cnt = 0;
    int first_valid_cell = 0;
    int valid_cell_idx = 0;
    int num_dlc_tasks = get_num_dlc_tasks(pdctx->getNumDLWorkers(),pdctx->gpuCommEnabledViaCpu(),pdctx->getmMIMO_enable());
    int num_tasks_to_wait = 0;
    int num_tasks_ignore_for_wait = 0;

    if(dlbfw)
    {
        if(pdctx->gpuCommEnabledViaCpu())
            num_tasks_ignore_for_wait = 5 + (num_dlc_tasks<<1); //DLBFW + FHCb + Compression + CPU Doorbell ring + DL Task 3 Buf Cleanup + DLC tasks (x2 to factor in UPlane prepare)
        else
            num_tasks_ignore_for_wait = 4 + (num_dlc_tasks<<1); //DLBFW + FHCb + Compression + CPU Doorbell ring + DL Task 3 Buf Cleanup + DLC tasks (x2 to factor in UPlane prepare)            
    }
    else
    {
        if(pdctx->gpuCommEnabledViaCpu())
            num_tasks_ignore_for_wait = 4 + (num_dlc_tasks<<1); //FHCb + Compression + CPU Doorbell ring + DL Task 3 Buf Cleanup + DLC tasks (x2 to factor in UPlane prepare)
        else
            num_tasks_ignore_for_wait = 3 + (num_dlc_tasks<<1); //FHCb + Compression + CPU Doorbell ring + DL Task 3 Buf Cleanup + DLC tasks (x2 to factor in UPlane prepare)
    }

    if(pdctx->gpuCommEnabledViaCpu())
        num_tasks_to_wait = num_dl_tasks - num_tasks_ignore_for_wait;
    else
        num_tasks_to_wait = num_dl_tasks - num_tasks_ignore_for_wait;
    DLOutputBuffer *compression_dlbuf = nullptr;
    DLOutputBuffer *prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    int            cell_index_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {0};
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell * cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer * dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        auto nic_index = cell_ptr->getNicIndex();

        if (!compression_dlbuf) {
            compression_dlbuf = dlbuf;
        }
        if (!prepare_tx_dlbuf_per_nic[nic_index])
            prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;

        if (local_cell_cnt == 0)  { first_strm = cell_ptr->getDlStream(); first_valid_cell = i; }
        local_cell_cnt += 1;
    }

    /////////////////////////////////////////////////////////////////////////////////////
    //// Compression Preparation
    /////////////////////////////////////////////////////////////////////////////////////
    ti.add("Create Compression Params");
    
    // Array of compression params - one per compression method (indexed by aerial_fh::UserDataCompressionMethod)
    std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS> cparams_array{};
    
    // Populate compression parameters for all cells, grouped by compression method
    valid_cell_idx = populateCompressionParams(cparams_array, slot_map, pdctx, first_cell, num_cells);

    // For compression additionally wait for all enabled DL channel and GPU init comms
    // tasks to end. GPU init comms prepares the PRB buffers needed by compression
    ti.add("Wait Slot Channel End");
    if(slot_map->waitSlotChannelEnd(num_tasks_to_wait) < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitSlotChannelEnd returned error for Map {} num_dl_tasks {} num_dlc_tasks {} num_tasks_ignore_for_wait {}",slot_map->getId(),num_dl_tasks,num_dlc_tasks,num_tasks_ignore_for_wait);
        goto exit_error;
    }
    start_t_1 = Time::nowNs();

    ti.add("Channel Waits");
#if SIGNAL_COMPLETION_W_EVENTS
    if (pbch && pbch->waitRunCompletionGPUEvent(first_strm, pdctx->getDlCtx())) {
#else
    if (pbch && pbch->waitRunCompletionGPU(first_strm, pdctx->getDlCtx())) {
#endif
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PBCH waitRunCompletionGPU returned error");
        goto exit_error;
    }

#if SIGNAL_COMPLETION_W_EVENTS
    if (pdcch_dl && pdcch_dl->waitRunCompletionGPUEvent(first_strm, pdctx->getDlCtx())) {
#else
    if (pdcch_dl && pdcch_dl->waitRunCompletionGPU(first_strm, pdctx->getDlCtx())) {
#endif
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH DL waitRunCompletionGPU returned error");
        goto exit_error;
    }

#if SIGNAL_COMPLETION_W_EVENTS
    if (pdcch_ul && pdcch_ul->waitRunCompletionGPUEvent(first_strm, pdctx->getDlCtx())) {
#else
    if (pdcch_ul && pdcch_ul->waitRunCompletionGPU(first_strm, pdctx->getDlCtx())) {
#endif
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDCCH UL waitRunCompletionGPU returned error");
        goto exit_error;
    }

#if SIGNAL_COMPLETION_W_EVENTS
    if (csirs && csirs->waitRunCompletionGPUEvent(first_strm, pdctx->getDlCtx())) {
#else
    if (csirs && csirs->waitRunCompletionGPU(first_strm, pdctx->getDlCtx())) {
#endif
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "CSI_RS UL waitRunCompletionGPU returned error");
        goto exit_error;
    }

#if SIGNAL_COMPLETION_W_EVENTS
    if (pdsch && pdsch->waitRunCompletionGPUEvent(first_strm, pdctx->getDlCtx())) {
#else
    if (pdsch && pdsch->waitRunCompletionGPU(first_strm, pdctx->getDlCtx())) {
#endif
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PDSCH waitRunCompletionGPU returned error");
        goto exit_error;
    }

    if(pdsch||csirs||pdcch_dl||pdcch_ul||pbch)
    {
        //Record that the channels have completed
        ti.add("Record Channels Done");
        if(compression_dlbuf==nullptr)
            goto exit_error;
        {
            MemtraceDisableScope md;
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(compression_dlbuf->getAllChannelsDoneEvt(), first_strm));
        }

        ti.add("Wait DL Gpu Comm End"); // Synchronize end of GPU Comm Task
        if(slot_map->waitDlGpuCommEnd() < 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitDlGpuCommEnd returned error");
            EXIT_L1(EXIT_FAILURE);
        }        

        // Wait for GPU init comms prepare to finish making the buffers needed for compression
        // Note: this event is in another stream and another context!!!
        ti.add("Stream Wait Prepare");
        if(pdctx->gpuCommDlEnabled()) {
            for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
            {
                if(prepare_tx_dlbuf_per_nic[i])
                    CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(first_strm, prepare_tx_dlbuf_per_nic[i]->getPrePrepareStopEvt()));
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////
        //// Compression
        /////////////////////////////////////////////////////////////////////////////////////
        ti.add("Compression Run");
        PUSH_RANGE_PHYDRV((std::string("DL COMPRESS") + std::to_string(slot_map->getId())).c_str(), 2);
        slot_map->timings.start_t_dl_compression_cuda = Time::nowNs();
        
        // Run batched compression for all compression methods
        // The array contains compression params for all 7 compression methods (indexed by aerial_fh::UserDataCompressionMethod)
        // runCompression and launch_kernel_compression will process only those methods that have cells (num_cells > 0)
        if(compression_dlbuf->runCompression(cparams_array, pdctx->getDlCtx(), first_strm))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "COMPRESSION cell returned error");
            goto exit_error;
        }

        slot_map->timings.end_t_dl_compression_cuda = Time::nowNs();
        POP_RANGE_PHYDRV

        ti.add("Set Ready Flag");
        if(pdctx->gpuCommDlEnabled()) {
            /* Notify GComm output buffer is ready */
            for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
            {
                if(prepare_tx_dlbuf_per_nic[i])
                    prepare_tx_dlbuf_per_nic[i]->setReadyFlag(first_strm);
            }
        }
    }

    ti.add("Signal Slot End Task");
    slot_map->setDlCompEnd();
    slot_map->addSlotEndTask();
    start_t_2 = Time::nowNs();

    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " Compression Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " us on CPU " << (int)cpu;

    pdctx->dl_aggr_compression_task.unlock();
    return 0;

//FIXME: abort the whole DL slot in case of error
exit_error:
    pdctx->dl_aggr_compression_task.unlock();
    return -1;
}


void* queue_log_thread_func(void* arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "queue_log_thread thread {} affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "queue_log") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    struct args {
       Cell * cell_ptr;
       FhProxy* fhproxy;
    };
    struct args *args_p = reinterpret_cast<args*>(arg);
    //sleep(1);
    args_p->fhproxy->print_max_delays(args_p->cell_ptr->getNicName());
    free(args_p);
    return nullptr;
}

int task_work_function_dl_aggr_2(Worker* worker, void* param, int first_cell, int num_cells,int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task CPU Comms", 13);
    ti.add("Start Task");
    int                                                                          task_num = 2, ret = 0;
    uint16_t                                                                     numPrb, startPrb, endPrb;
    uint16_t                                                                     _numPrb, _startPrb, _endPrb;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_tx;
    t_ns                                                                         start_t_1, start_t_2;
    int                                                                          sfn = 0, slot = 0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;
    PhyPdcchAggr* pdcch_dl = slot_map->aggr_pdcch_dl;
    PhyPdcchAggr* pdcch_ul = slot_map->aggr_pdcch_ul;
    PhyPbchAggr* pbch = slot_map->aggr_pbch;
    PhyCsiRsAggr* csirs = slot_map->aggr_csirs;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    uint32_t cpu;
    DLOutputBuffer *first_dlbuf = nullptr;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    int local_cell_cnt = 0;
    cudaStream_t first_dl_stream = nullptr;
    cuphyBatchedMemcpyHelper& batchedMemcpyHelper = slot_map->getBatchedMemcpyHelper();
    batchedMemcpyHelper.reset(); // reset for upcoming batch of updateMemcpy calls
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    start_t_1 = Time::nowNs();

    if(pdctx->getmMIMO_enable()) {
        ti.add("Wait DLC");
        int num_dlc_tasks = get_num_dlc_tasks(pdctx->getNumDLWorkers(),pdctx->gpuCommEnabledViaCpu(),pdctx->getmMIMO_enable());
        slot_map->waitDLCDone(num_dlc_tasks);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    //// Prepare U-plane
    /////////////////////////////////////////////////////////////////////////////////////
    ti.add("UPlane Prepare");
    PUSH_RANGE_PHYDRV("DL UPL PREP", 3);
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        if (!first_dlbuf) {
            first_dlbuf = dlbuf;
        }
        
        if (local_cell_cnt == 0)  {first_dl_stream = cell_ptr->getDlStream();}
        local_cell_cnt += 1;

        start_tx = slot_map->getTaskTsExec(0) + (t_ns((Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) - cell_ptr->getT1aMaxUpNs()));
        slot_map->timings.start_t_dl_uprep[i] = Time::nowNs();

        if(slot_map->aggr_slot_info[i])
        {
            if(fhproxy->prepareUPlanePackets(
                cell_ptr->getRUType(),
                cell_ptr->getPeerId(),
                first_dl_stream, 
                start_tx,
                slot_oran_ind, //it_ulchannels.second.slot_oran_ind,
                *(slot_map->aggr_slot_info[i]), // *slot_map->aggr_slot_info[i], //*slot_info,
                dlbuf->getTxMsgContainer(),
                dlbuf->getSize(),
                dlbuf->getModCompressionConfig(),
                dlbuf->getModCompressionTempConfig(),
                dlbuf, t_ns(SYMBOL_DURATION_NS),
                batchedMemcpyHelper))
            {
                EXIT_L1(EXIT_FAILURE); //This should not happen
            }
        }

        slot_map->timings.end_t_dl_uprep[i] = Time::nowNs();

        // NVSLOGD(TAG) << "Task DL " << task_num << "Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " Prepare symbol took " << (Time::NsToUs(Time::nowNs()-uplane_symbol_t)).count() << " us";
    }
    // Avoid calling batched memcpy if there is no update done inside the Uplane prepare
    if((first_dl_stream != nullptr) && (batchedMemcpyHelper.getMemcpyCount() > 0)) {
        cuphyStatus_t batched_memcpy_status = batchedMemcpyHelper.launchBatchedMemcpy(first_dl_stream);
        if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
        {
           NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got launchBatchedMemcpy error", task_num, slot_map->getId());
           POP_RANGE_PHYDRV
           goto cleanup;
        }
    }
    POP_RANGE_PHYDRV

    /////////////////////////////////////////////////////////////////////////////////////
    //// Wait + TX U-plane
    /////////////////////////////////////////////////////////////////////////////////////

    ti.add("Channel Waits");
    if(pdsch != nullptr)
    {
        PUSH_RANGE_PHYDRV("DL WAIT PDSCH", 3);
        slot_map->timings.start_t_dl_pdsch_compl[0] = Time::nowNs();
        if(pdsch->waitRunCompletion(4 * 1000 * 1000) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got PDSCH wait error", task_num, slot_map->getId());
            goto cleanup;
        }
        slot_map->timings.end_t_dl_pdsch_compl[0] = Time::nowNs();
        POP_RANGE
        // NVSLOGD(TAG)  << "Task DL " << task_num << " Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
    }

    if(pdcch_dl != nullptr)
    {
        PUSH_RANGE_PHYDRV("DL WAIT PDCCH DL", 3);
        slot_map->timings.start_t_dl_pdcchdl_compl[0] = Time::nowNs();
        if(pdcch_dl->waitRunCompletion(4 * 1000 * 1000) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got PDCCH DL wait error", task_num, slot_map->getId());
            goto cleanup;
        }
        slot_map->timings.end_t_dl_pdcchdl_compl[0] = Time::nowNs();
        POP_RANGE
        // NVSLOGD(TAG)  << "Task DL " << task_num << " Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
    }

    if(pdcch_ul != nullptr)
    {
        PUSH_RANGE_PHYDRV("DL WAIT PDCCH UL", 3);
        slot_map->timings.start_t_dl_pdcchul_compl[0] = Time::nowNs();
        if(pdcch_ul->waitRunCompletion(4 * 1000 * 1000) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got PDCCH UL wait error", task_num, slot_map->getId());
            goto cleanup;
        }
        slot_map->timings.end_t_dl_pdcchul_compl[0] = Time::nowNs();
        POP_RANGE
        // NVSLOGD(TAG)  << "Task DL " << task_num << " Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
    }

    if(pbch != nullptr)
    {
        PUSH_RANGE_PHYDRV("DL PBCH", 3);
        slot_map->timings.start_t_dl_pbch_compl[0] = Time::nowNs();
        if(pbch->waitRunCompletion(4 * 1000 * 1000) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got PBCH wait error", task_num , slot_map->getId());
            goto cleanup;
        }
        slot_map->timings.end_t_dl_pbch_compl[0] = Time::nowNs();
        POP_RANGE
        // NVSLOGD(TAG)  << "Task DL " << task_num << " Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
    }

    if(csirs != nullptr)
    {
        PUSH_RANGE_PHYDRV("DL CSIRS", 3);
        slot_map->timings.start_t_dl_csirs_compl[0] = Time::nowNs();
        if(csirs->waitRunCompletion(4 * 1000 * 1000) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got CSIRS wait error", task_num, slot_map->getId());
            goto cleanup;
        }
        slot_map->timings.end_t_dl_csirs_compl[0] = Time::nowNs();
        POP_RANGE
        // NVSLOGD(TAG)  << "Task DL " << task_num << " Map " << slot_map->getId() << " CSI RS " << (long int)csirs->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
    }

    ti.add("Compression Wait");
    PUSH_RANGE_PHYDRV("DL WAIT COMP", 3);
    slot_map->timings.start_t_dl_compression_compl = Time::nowNs();
    if(first_dlbuf==nullptr)
        goto cleanup;
    if(first_dlbuf->waitCompressionStop() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got COMPRESSION DL wait error", task_num ,slot_map->getId());
        goto cleanup;
    }


    slot_map->timings.prepare_execution_duration1[0] = 0.0; //Note: prepare memcpy/memset
    slot_map->timings.prepare_execution_duration2[0] = 0.0; //Note: no prepare kernel
    slot_map->timings.channel_to_compression_gap = first_dlbuf->getChannelToCompressionGap();
    slot_map->timings.compression_execution_duration = first_dlbuf->getCompressionExecutionTime();
    slot_map->timings.end_t_dl_compression_compl = Time::nowNs();
    POP_RANGE
    POP_RANGE_PHYDRV

    ti.add("Uplane TX");
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;

        PUSH_RANGE_PHYDRV("DL UPL TX", 3);
        slot_map->timings.start_t_dl_utx[i] = Time::nowNs();
        fhproxy->UserPlaneSendPackets(cell_ptr->getPeerId(), dlbuf->getTxMsgContainer());
        slot_map->timings.end_t_dl_utx[i] = Time::nowNs();
        POP_RANGE_PHYDRV
    }

    //FIXME: only PDSCH should call the callback if present?
    ti.add("PDSCH Callback");
    PUSH_RANGE_PHYDRV("DL CALLBACK", 3);
    if(pdsch != nullptr)
    {
        slot_map->timings.start_t_dl_callback = Time::nowNs();
        pdsch->callback(/* slot_map->aggr_slot_params, */ slot_map->getSlot3GPP());
        slot_map->timings.end_t_dl_callback = Time::nowNs();
    }

    ti.add("Metric Update");
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        cell_ptr->updateMetric(CellMetric::kDlSlotsTotal, 1);
    }

    POP_RANGE_PHYDRV


    if (0) if(((sfn % 100) == 0) && (slot == 10))
    {
        Cell * cell_ptr = slot_map->aggr_cell_list[0];
#if 1
        struct args {
           Cell * cell_ptr;
           FhProxy* fhproxy;
        };

        struct args *queue_log_args = (struct args *)malloc(sizeof(struct args));

        queue_log_args->cell_ptr = cell_ptr;
        queue_log_args->fhproxy = fhproxy;

        pthread_t thread_id;
        pthread_create(&thread_id, NULL, queue_log_thread_func, queue_log_args);
#else
        fhproxy->print_max_delays(cell_ptr->getNicName());
#endif
    }

cleanup:
    slot_map->addSlotEndTask();

    start_t_2 = Time::nowNs();

    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " us on CPU " << (int)cpu;

    return 0;

exit_error:
    return -1;
}

int task_work_function_dl_aggr_2_gpu_comm(Worker* worker, void* param, int first_cell, int num_cells,int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task GPU Comms", 13);
    ti.add("Start Task");
    int                                                                          task_num = 2, ret = 0;
    uint16_t                                                                     numPrb, startPrb, endPrb;
    uint16_t                                                                     _numPrb, _startPrb, _endPrb;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_tx,start_t,h2d_wait_th(pdctx->geth2d_copy_wait_th());
    t_ns                                                                         start_t_1, start_t_2;
    int                                                                          sfn = 0, slot = 0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;
    PhyPdcchAggr* pdcch_dl = slot_map->aggr_pdcch_dl;
    PhyPdcchAggr* pdcch_ul = slot_map->aggr_pdcch_ul;
    PhyPbchAggr* pbch = slot_map->aggr_pbch;
    PhyCsiRsAggr* csirs = slot_map->aggr_csirs;
    PhyDlBfwAggr* dlbfw = slot_map->aggr_dlbfw;
    bool dlbfw_complete;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    bool pdsch_tb_h2d_done=false;
    t_ns timeout_thresh_t(ORDER_KERNEL_WAIT_TIMEOUT_MS * 2 * NS_X_MS);

    TxRequestGpuPercell tx_v_cells[MAX_NUM_OF_NIC_SUPPORTED] = {};

    PreparePRBInfo prb_info{{nullptr}};
    PreparePRBInfo prb_info_per_nic[MAX_NUM_OF_NIC_SUPPORTED];
    DLOutputBuffer *compression_dlbuf = nullptr;
    DLOutputBuffer *prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    int            cell_index_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {0};

    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    int local_cell_cnt = 0;
    cudaStream_t first_dl_stream = nullptr;
    cuphyBatchedMemcpyHelper& batchedMemcpyHelper = slot_map->getBatchedMemcpyHelper();
    batchedMemcpyHelper.reset(); // reset for upcoming batch of updateMemcpy calls
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    start_t_1 = Time::nowNs();

    //NVLOGI_FMT(TAG,"DL Task 2 started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);
    ti.add("Wait Slot Start Task"); // Synchronize the Compression task and GPU Uplane prepare
    if(slot_map->waitFHCBDone() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitFHCBDone returned error");
        goto exit_error;
    }

    //Init size 
    for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
    {
        tx_v_cells[i].size=0;
        slot_map->tx_v_for_slot_map[i].size=0;
    }    

    if(pdctx->getmMIMO_enable()) {
        ti.add("Wait Peer Update");
        slot_map->waitPeerUpdateDone();
    }

    if(pdsch||pdcch_dl||pdcch_ul||pbch||csirs)
    {
        /////////////////////////////////////////////////////////////////////////////////////
        //// U-Plane
        /////////////////////////////////////////////////////////////////////////////////////
        ti.add("UPlane Prepare");
        PUSH_RANGE_PHYDRV((std::string("DL UPL PREP") + std::to_string(slot_map->getId())).c_str(), 3);
        for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
        {
            Cell * cell_ptr = slot_map->aggr_cell_list[i];
            auto nic_index = cell_ptr->getNicIndex();
            DLOutputBuffer * dlbuf = slot_map->aggr_dlbuf_list[i];
            if(cell_ptr == nullptr || dlbuf == nullptr) continue;

            if (!compression_dlbuf) {
                compression_dlbuf = dlbuf;
            }

            if(!prepare_tx_dlbuf_per_nic[nic_index])
            {
                prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;
            }

            if (local_cell_cnt == 0)  {first_dl_stream = cell_ptr->getDlStream();}
            local_cell_cnt += 1;

            if(pdsch && (pdctx->getUeMode()!=0))
            {
                start_tx = slot_map->getTaskTsExec(0) + (t_ns((Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) + cell_ptr->getUlUplaneTxOffsetNs()));
            }
            else
            {
                start_tx = slot_map->getTaskTsExec(0) + (t_ns((Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) - cell_ptr->getT1aMaxUpNs()));
            }
            slot_map->timings.start_t_dl_uprep[i] = Time::nowNs();

            if(slot_map->aggr_slot_info[i])
            {
                pdctx->setGpuCommsCtx();
                if(fhproxy->prepareUPlanePackets(
                    cell_ptr->getRUType(),
                    cell_ptr->getPeerId(),
                    first_dl_stream,
                    start_tx,
                    slot_oran_ind, //it_ulchannels.second.slot_oran_ind,
                    *(slot_map->aggr_slot_info[i]), // *slot_map->aggr_slot_info[i], //*slot_info,
                    dlbuf->getTxMsgContainer(),
                    dlbuf->getSize(),
                    dlbuf->getModCompressionConfig(),
                    dlbuf->getModCompressionTempConfig(),
                    dlbuf, t_ns(SYMBOL_DURATION_NS),
                    batchedMemcpyHelper))
                {
                    EXIT_L1(EXIT_FAILURE); //This should not happen
                }

                const auto& eaxc = cell_ptr->geteAxCIdsPdsch();
                auto cell_index = cell_index_per_nic[nic_index];
                ++cell_index_per_nic[nic_index];
                prb_info_per_nic[nic_index].prb_ptrs[cell_index] = dlbuf->getPrbPtrs();
                prb_info_per_nic[nic_index].num_antennas[cell_index] = eaxc.size();
                prb_info_per_nic[nic_index].max_num_prb_per_symbol[cell_index] = cell_ptr->getDLGridSize();

                for (auto ant = 0; ant < eaxc.size(); ant++) {
                    prb_info_per_nic[nic_index].eAxCMap[cell_index][eaxc[ant] & 0x1f] = ant;
                }
            }

            slot_map->timings.end_t_dl_uprep[i] = Time::nowNs();

            // NVSLOGD(TAG) << "Task DL " << task_num << "Map " << slot_map->getId() << " PDSCH " << (long int)pdsch->getId() << " Prepare symbol took " << (Time::NsToUs(Time::nowNs()-uplane_symbol_t)).count() << " us";

            struct umsg_fh_tx_msg& txmsg = dlbuf->getTxMsgContainer();

            //Check if NIC is in pre-allocated range
            if(cell_ptr->getNicIndex() >= MAX_NUM_OF_NIC_SUPPORTED) {
                NVLOGF_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"NIC index of {} out of bounds.  Max num of NICs supported is {}",
                            cell_ptr->getNicIndex(), MAX_NUM_OF_NIC_SUPPORTED);
                continue;
            }

            //Check if NIC has too many TX REQ slots
            if((slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].size >=  MAX_NUM_TX_REQ_UPLANE_GPU_COMM_PER_NIC))
            {
                NVLOGF_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"Number of tx req is more than max for NIC {} {}",
                            slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].nic_name.c_str(), cell_ptr->getNicIndex());
                continue;
            }

            slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].nic_name = cell_ptr->getNicName();
            slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].tx_v_per_nic[slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].size++] = txmsg.txrq_gpu;

        }
        // Avoid calling batched memcpy if there is no update done inside the Uplane prepare
        if((first_dl_stream != nullptr) && (batchedMemcpyHelper.getMemcpyCount() > 0)) {
            cuphyStatus_t batched_memcpy_status = batchedMemcpyHelper.launchBatchedMemcpy(first_dl_stream);
            if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr {} Map {} got launchBatchedMemcpy error", task_num, slot_map->getId());
                POP_RANGE_PHYDRV
                goto cleanup;
            }
        }
        POP_RANGE_PHYDRV

        // Ready/wait information for trigger. Only a single trigger for all cells. Note that GPU comms
        // mode does not need to set the compression output pointer since the compression kernel writes
        // directly into the packet buffer after the header
        ti.add("Uplane TX");

        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
        {
            if(slot_map->tx_v_for_slot_map[i].size)
            {
                if(!prepare_tx_dlbuf_per_nic[i])
                {
                    goto cleanup;
                }
            }
        }
        /* Internal GComm stream has been created in default DL context */
        pdctx->setGpuCommsCtx();
        PUSH_RANGE_PHYDRV((std::string("DL UPL GTX") + std::to_string(slot_map->getId())).c_str(), 3);
        slot_map->timings.start_t_dl_utx[0] = Time::nowNs();
        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
        {
            if(slot_map->tx_v_for_slot_map[i].size)
            {
                prb_info_per_nic[i].ready_flag = prepare_tx_dlbuf_per_nic[i]->getReadyFlag();
                prb_info_per_nic[i].wait_val   = 1;
                prb_info_per_nic[i].compression_stop_evt = compression_dlbuf->getCompressionStopEvt();
                prb_info_per_nic[i].comm_start_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareStartEvt();
                prb_info_per_nic[i].comm_copy_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareCopyEvt();
                prb_info_per_nic[i].comm_preprep_stop_evt = prepare_tx_dlbuf_per_nic[i]->getPrePrepareStopEvt();
                prb_info_per_nic[i].comm_stop_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareStopEvt();
                prb_info_per_nic[i].trigger_end_evt = prepare_tx_dlbuf_per_nic[i]->getTxEndEvt();
                prb_info_per_nic[i].enable_prepare_tracing = pdctx->enablePrepareTracing();
                prb_info_per_nic[i].enable_dl_cqe_tracing = pdctx->enableDlCqeTracing();
                prb_info_per_nic[i].disable_empw = pdctx->disableEmpw();
                prb_info_per_nic[i].cqe_trace_cell_mask = pdctx->get_cqe_trace_cell_mask();
                prb_info_per_nic[i].cqe_trace_slot_mask = pdctx->get_cqe_trace_slot_mask();
                if(0 != fhproxy->UserPlaneSendPacketsGpuComm(&slot_map->tx_v_for_slot_map[i], prb_info_per_nic[i]))
                {
                    // TODO Add graceful error handling for recovery
                    NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "UserPlaneSendPacketsGpuComm returned fatal error");
                    EXIT_L1(EXIT_FAILURE);
                };
            }
        }

        slot_map->timings.end_t_dl_utx[0] = Time::nowNs();
    }

    ti.add("Signal Slot Channel End");
    slot_map->addSlotChannelEnd();
    slot_map->setDlGpuCommEnd();
    POP_RANGE_PHYDRV
    pdctx->setDlCtx();

    if(pdsch != nullptr)
    {

        //Wait here till H2D copy is completed before trigerring callback
        ti.add("PDSCH TB H2D Wait");
        start_t = Time::nowNs();
        while(!pdsch_tb_h2d_done)
        {
            pdsch_tb_h2d_done=(bool)pdsch->waitEventNonBlocking(pdctx->get_event_pdsch_tb_cpy_complete((uint8_t)slot));
            if(!pdsch_tb_h2d_done && ((Time::nowNs()-start_t)>h2d_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PDSCH TB H2D copy wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),h2d_wait_th.count());
                goto cleanup;
            }
        }
        if(!slot_map->pdsch_cb_done){
            //Trigger PDSCH callback as soon as wait for H2D copy is done
            ti.add("PDSCH Callback");
            PUSH_RANGE_PHYDRV((std::string("DL CALLBACK") + std::to_string(slot_map->getId())).c_str(), 3);
            slot_map->timings.start_t_dl_callback = Time::nowNs();
            pdsch->callback(/* slot_map->aggr_slot_params, */ slot_map->getSlot3GPP());
            slot_map->timings.end_t_dl_callback = Time::nowNs();
            slot_map->pdsch_cb_done=true;
            POP_RANGE_PHYDRV
        }
    }

    ti.add("Metric Update");
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        cell_ptr->updateMetric(CellMetric::kDlSlotsTotal, 1);

        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        struct umsg_fh_tx_msg& txmsg = dlbuf->getTxMsgContainer();
        fhproxy->UpdateTxMetricsGpuComm(cell_ptr->getPeerId(), txmsg);
    }

    if (0) if(((sfn % 100) == 0) && (slot == 0))
    {
        NVLOGC_FMT(TAG,"Calling print_max_delays");
        for(uint32_t i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
        {
            if(slot_map->tx_v_for_slot_map[i].size)
            {
                fhproxy->print_max_delays(slot_map->tx_v_for_slot_map[i].nic_name);
            }
        }
    }

cleanup:
    slot_map->addSlotEndTask();

    start_t_2 = Time::nowNs();

    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " uson CPU " << (int)cpu;

    return 0;

exit_error:
    return -1;
}

int task_work_function_dl_aggr_2_ring_cpu_doorbell(Worker* worker, void* param, int first_cell, int num_cells,int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Aggr2 Ring CPU Doorbell", 13);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    DLOutputBuffer *prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    int                                                                          sfn = 0, slot = 0;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_; 
    uint32_t cpu;  
    t_ns wait_thresh_t(500000); //500us
    PreparePRBInfo prb_info_per_nic[MAX_NUM_OF_NIC_SUPPORTED];
    PacketTimingInfo packet_timing_info_per_nic[MAX_NUM_OF_NIC_SUPPORTED];
    DLOutputBuffer *compression_dlbuf = nullptr;

    //pdctx->dl_cpu_db_task.lock();

NVLOGI_FMT(TAG,"starting cpudb");
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGI_FMT(TAG,"getcpu failed for {}", __FUNCTION__);
    }

    ti.add("Wait DL Comp End"); // Synchronize end of Compression Task
    if(slot_map->waitDlCompEnd() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitDlCompEnd returned error");
        EXIT_L1(EXIT_FAILURE);
    }     

    ti.add("Get Prepare Tx DL Buf");
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        DLOutputBuffer * dlbuf = slot_map->aggr_dlbuf_list[i];
        Cell * cell_ptr = slot_map->aggr_cell_list[i];

        if(dlbuf == nullptr) continue;
        if(cell_ptr == nullptr) continue;

        auto nic_index = cell_ptr->getNicIndex();

        if (!prepare_tx_dlbuf_per_nic[nic_index])
            prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;

        if (!compression_dlbuf) {
            compression_dlbuf = dlbuf;
        }            
    }    

    pdctx->setGpuCommsCtx();
    ti.add("CPU copy and ring doorbell");
    for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
    {
        if(slot_map->tx_v_for_slot_map[i].size>0)
        {
            prb_info_per_nic[i].ready_flag = prepare_tx_dlbuf_per_nic[i]->getReadyFlag();
            prb_info_per_nic[i].wait_val   = 1;
            prb_info_per_nic[i].compression_stop_evt = compression_dlbuf->getCompressionStopEvt();
            prb_info_per_nic[i].trigger_end_evt = prepare_tx_dlbuf_per_nic[i]->getTxEndEvt();
            prb_info_per_nic[i].p_packet_mem_copy_per_symbol_dur_us = slot_map->timings.packet_mem_copy_per_symbol_dur_us;
            prb_info_per_nic[i].use_copy_kernel_for_d2h = false; //Set default copy mode to batched memcpy using copy engine. Change to true to use copy kernel for d2h.

            NVLOGI_FMT(TAG,"Done waiting for copy kernel to finish for Slot Map {}",slot_map->getId());
            if(0 != fhproxy->RingCPUDoorbell(&slot_map->tx_v_for_slot_map[i], prb_info_per_nic[i], packet_timing_info_per_nic[i]))
            {
                // TODO Add graceful error handling for recovery
                NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "RingCPUDoorbell returned fatal error");
                EXIT_L1(EXIT_FAILURE);
            }                       
        }
    }

    NVLOGI_FMT(TAG,"[DL Aggr2 CPU Door Bell Task] Slot Map {} Frame ID {} Subframe ID {} Slot ID {} [Compression kernel done time] {} [CPU send task start time] {} [PacketCopy done times] {} {} {} {} {} {} {} {} {} {} {} {} {} {} [Trigger done times] {} {} {} {} {} {} {} {} {} {} {} {} {} {} [PacketCopy launch times] {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        slot_map->getId(),packet_timing_info_per_nic[0].frame_id,packet_timing_info_per_nic[0].subframe_id,packet_timing_info_per_nic[0].slot_id,slot_map->timings.end_t_dl_compression_compl.count(),packet_timing_info_per_nic[0].cpu_send_start_timestamp,packet_timing_info_per_nic[0].pkt_copy_done_timestamp[0],
        packet_timing_info_per_nic[0].pkt_copy_done_timestamp[1],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[2],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[3],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[4],
        packet_timing_info_per_nic[0].pkt_copy_done_timestamp[5],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[6],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[7],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[8],
        packet_timing_info_per_nic[0].pkt_copy_done_timestamp[9],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[10],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[11],packet_timing_info_per_nic[0].pkt_copy_done_timestamp[12],
        packet_timing_info_per_nic[0].pkt_copy_done_timestamp[13],packet_timing_info_per_nic[0].trigger_done_timestamp[0],packet_timing_info_per_nic[0].trigger_done_timestamp[1],packet_timing_info_per_nic[0].trigger_done_timestamp[2],
        packet_timing_info_per_nic[0].trigger_done_timestamp[3],packet_timing_info_per_nic[0].trigger_done_timestamp[4],packet_timing_info_per_nic[0].trigger_done_timestamp[5],packet_timing_info_per_nic[0].trigger_done_timestamp[6],
        packet_timing_info_per_nic[0].trigger_done_timestamp[7],packet_timing_info_per_nic[0].trigger_done_timestamp[8],packet_timing_info_per_nic[0].trigger_done_timestamp[9],packet_timing_info_per_nic[0].trigger_done_timestamp[10],
        packet_timing_info_per_nic[0].trigger_done_timestamp[11],packet_timing_info_per_nic[0].trigger_done_timestamp[12],packet_timing_info_per_nic[0].trigger_done_timestamp[13],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[0],
        packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[1],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[2],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[3],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[4],
        packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[5],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[6],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[7],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[8],
        packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[9],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[10],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[11],packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[12],
        packet_timing_info_per_nic[0].pkt_copy_launch_timestamp[13]);
        
    NVLOGD_FMT(TAG,"[DL Aggr2 CPU Door Bell Task] Slot Map {} Frame ID {} Subframe ID {} Slot ID {} [PacketCopy kernel num packets] {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        slot_map->getId(),packet_timing_info_per_nic[0].frame_id,packet_timing_info_per_nic[0].subframe_id,packet_timing_info_per_nic[0].slot_id,packet_timing_info_per_nic[0].num_packets_per_symbol[0],
        packet_timing_info_per_nic[0].num_packets_per_symbol[1],packet_timing_info_per_nic[0].num_packets_per_symbol[2],packet_timing_info_per_nic[0].num_packets_per_symbol[3],packet_timing_info_per_nic[0].num_packets_per_symbol[4],
        packet_timing_info_per_nic[0].num_packets_per_symbol[5],packet_timing_info_per_nic[0].num_packets_per_symbol[6],packet_timing_info_per_nic[0].num_packets_per_symbol[7],packet_timing_info_per_nic[0].num_packets_per_symbol[8],
        packet_timing_info_per_nic[0].num_packets_per_symbol[9],packet_timing_info_per_nic[0].num_packets_per_symbol[10],packet_timing_info_per_nic[0].num_packets_per_symbol[11],packet_timing_info_per_nic[0].num_packets_per_symbol[12],
        packet_timing_info_per_nic[0].num_packets_per_symbol[13]);        

    slot_map->setDlCpuDoorBellTaskDone();
    slot_map->addSlotEndTask();
    ti.add("End Task");
    //pdctx->dl_cpu_db_task.unlock();

    return 0; 
}

int task_work_function_dl_aggr_3_buf_cleanup(Worker* worker, void* param, int first_cell, int num_cells,int num_dl_tasks)
{
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task Buff Cleanup", 13);
    ti.add("Start Task");
    int                                                                          task_num = 1, ret = 0;
    SlotMapDl*                                                                   slot_map = (SlotMapDl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_t_1, start_t_2, start_tx,start_t,channel_wait_th(pdctx->getcuphy_dl_channel_wait_th());
    t_ns end_time;
    int                                                                          sfn = 0, slot = 0;
    PhyPdschAggr* pdsch = slot_map->aggr_pdsch;
    PhyPdcchAggr* pdcch_dl = slot_map->aggr_pdcch_dl;
    PhyPdcchAggr* pdcch_ul = slot_map->aggr_pdcch_ul;
    PhyPbchAggr*  pbch = slot_map->aggr_pbch;
    PhyCsiRsAggr* csirs = slot_map->aggr_csirs;
    bool pdsch_cuphy_done=false,pdcch_dl_cuphy_done=false,pdcch_ul_cuphy_done=false,pbch_cuphy_done=false,csirs_cuphy_done=false,gpu_wait_error=false;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    int num_dlc_tasks = get_num_dlc_tasks(pdctx->getNumDLWorkers(),pdctx->gpuCommEnabledViaCpu(),pdctx->getmMIMO_enable());
    int num_tasks_to_wait = 0;
    if(pdctx->gpuCommEnabledViaCpu())
        num_tasks_to_wait = num_dl_tasks - 4 - num_dlc_tasks;
    else
        num_tasks_to_wait = num_dl_tasks - 3 - num_dlc_tasks;
    uint32_t cpu;
    struct slot_command_api::dl_slot_callbacks dl_cb;
    std::array<uint32_t,DL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
    }

    //NVLOGI_FMT(TAG,"DL Task 3(DL Buff Cleanup) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);
    start_t_1 = Time::nowNs();

    ti.add("Buffer Clearing");
    pdctx->getDlCtx()->setCtx();
#ifdef CUPHY_PTI_ENABLE_TRACING
    if (((sfn % 128) == 0) && (slot == 0))
    {
        cuphy_pti_calibrate_gpu_timer(pdctx->get_stream_timing_dl(),slot_map->getSlot3GPP().t0_);
    }
#endif

    cudaStream_t first_strm=(cudaStream_t)0x0;
    int local_cell_cnt = 0;
    int first_valid_cell = 0;
    DLOutputBuffer *first_dlbuf = nullptr;
    DLOutputBuffer *prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    CleanupDlBufInfo* h_buffers_addr = (CleanupDlBufInfo*)pdctx->getDlHBuffersAddr(slot & (DL_HELPER_MEMSET_BUFFERS_PER_CTX - 1));
    CleanupDlBufInfo* d_buffers_addr = (CleanupDlBufInfo*)pdctx->getDlDBuffersAddr(slot & (DL_HELPER_MEMSET_BUFFERS_PER_CTX - 1));
    size_t max_buffer_size = 0;

    if ((h_buffers_addr == nullptr) || (d_buffers_addr == nullptr)) {
        NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT, "nullptr for h_buffers_addr or d_buffers_addr");
        goto exit_error;
    }

    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
    {
        Cell * cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer * dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        auto nic_index = cell_ptr->getNicIndex();
        cell_idx_list[cell_count++]=cell_ptr->getIdx();

        // This is kind of a hack and should be replaced. DlOutputBuffers happens to be a good
        // place to allocate GPU memory since it has a device attached, and the host-pinned
        // memory API needs a gDev device. We just use the first dlbuf in the slot map for this
        if (!first_dlbuf) {
            first_dlbuf = dlbuf;
        }

        if (!prepare_tx_dlbuf_per_nic[nic_index])
            prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;

        h_buffers_addr[local_cell_cnt] = {(uint4*)dlbuf->getBufD(), dlbuf->getSize()};
        // The memset kernel expects each dl buf to be uint4 aligned. An error message is printed otherwise.
        if((reinterpret_cast<uintptr_t>(dlbuf->getBufD()) & 0xF) != 0) { // Raise an error if not uint4 aligned
            NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,  "DL buf. {} for cell {} is not properly aligned", reinterpret_cast<void*>((uint4*)dlbuf->getBufD()) , i);
            //FIXME should we return?
        }

        max_buffer_size = std::max(max_buffer_size, dlbuf->getSize());
        if (local_cell_cnt == 0)  { first_strm = cell_ptr->getDlStream(); first_valid_cell = i; }

        local_cell_cnt += 1;
    }

    // The assumption here is that we do not have to synchronize the buffer memset complete
    // because we use 16 DL Buffers per cell, and it would take 10 ms (including UL) for
    // the DL buffers to be reused
    if(slot_map->getNumCells()>0)
    {

        CUDA_CHECK_PHYDRIVER(cudaMemcpyAsync(d_buffers_addr, h_buffers_addr, sizeof(CleanupDlBufInfo) * slot_map->getNumCells(), cudaMemcpyHostToDevice, first_strm));
        launch_memset_kernel((void*)d_buffers_addr, slot_map->getNumCells(), max_buffer_size, first_strm);
    }

    slot_map->addSlotEndTask();

    //Make sure that all tasks (except debug thread) have run to completion
    ti.add("Wait Slot End Task");
    if(slot_map->waitSlotEndTask(num_dl_tasks) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitSlotEnd returned error");
        if(pdctx->getDlCb(dl_cb))
        {
                auto msg_id = (pdcch_ul == nullptr)? SCF_FAPI_DL_TTI_REQUEST: SCF_FAPI_UL_DCI_REQUEST;
                dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_ind,msg_id,SCF_ERROR_CODE_L1_DL_CPU_TASK_ERROR,cell_idx_list,cell_count);
        }
    }


    //Channel wait loop
    pdsch_cuphy_done = (pdsch == nullptr);
    pdcch_dl_cuphy_done = (pdcch_dl == nullptr);
    pdcch_ul_cuphy_done = (pdcch_ul == nullptr);
    pbch_cuphy_done = (pbch == nullptr);
    csirs_cuphy_done = (csirs == nullptr);

    start_t = Time::nowNs();
    slot_map->timings.start_t_dl_pdsch_compl[0]=start_t;
    slot_map->timings.start_t_dl_pdcchdl_compl[0]=start_t;
    slot_map->timings.start_t_dl_pdcchul_compl[0]=start_t;
    slot_map->timings.start_t_dl_pbch_compl[0]=start_t;
    slot_map->timings.start_t_dl_csirs_compl[0]=start_t;

    ti.add("Wait cuPHY Channel Done");
    while(!pdsch_cuphy_done || !pdcch_dl_cuphy_done || !pdcch_ul_cuphy_done || !pbch_cuphy_done || !csirs_cuphy_done)
    {
        if(!pdcch_dl_cuphy_done)
        {
            pdcch_dl_cuphy_done=(pdcch_dl->waitRunCompletionEventNonBlocking()==1);
            if(!pdcch_dl_cuphy_done && ((Time::nowNs()-start_t)>channel_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PDCCH DL wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                gpu_wait_error=true;
                goto cleanup;
            }
            else if(pdcch_dl_cuphy_done)
            {
                slot_map->timings.end_t_dl_pdcchdl_compl[0]=Time::nowNs();
            }
        }
        if(!pdcch_ul_cuphy_done)
        {
            pdcch_ul_cuphy_done=(pdcch_ul->waitRunCompletionEventNonBlocking()==1);
            if(!pdcch_ul_cuphy_done && ((Time::nowNs()-start_t)>channel_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PDCCH UL wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                gpu_wait_error=true;
                goto cleanup;
            }
            else if(pdcch_ul_cuphy_done)
            {
                slot_map->timings.end_t_dl_pdcchul_compl[0]=Time::nowNs();
            }
        }
        if(!pbch_cuphy_done)
        {
            pbch_cuphy_done=(pbch->waitRunCompletionEventNonBlocking()==1);
            if(!pbch_cuphy_done && ((Time::nowNs()-start_t)>channel_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PBCH wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                gpu_wait_error=true;
                goto cleanup;
            }
            else if(pbch_cuphy_done)
            {
                slot_map->timings.end_t_dl_pbch_compl[0]=Time::nowNs();
            }
        }
        if(!csirs_cuphy_done)
        {
            csirs_cuphy_done=(csirs->waitRunCompletionEventNonBlocking()==1);
            if(!csirs_cuphy_done && ((Time::nowNs()-start_t)>channel_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"CSI-RS wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                gpu_wait_error=true;
                goto cleanup;
            }
            else if(csirs_cuphy_done)
            {
                slot_map->timings.end_t_dl_csirs_compl[0]=Time::nowNs();
            }
        }
        if(!pdsch_cuphy_done)
        {
            pdsch_cuphy_done=(pdsch->waitRunCompletionEventNonBlocking()==1);
            if(!pdsch_cuphy_done && ((Time::nowNs()-start_t)>channel_wait_th))
            {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PDSCH wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                gpu_wait_error=true;
                goto cleanup;
            }
            else if(pdsch_cuphy_done)
            {
                slot_map->timings.end_t_dl_pdsch_compl[0]=Time::nowNs();
            }
        }
    }

    //Compression kernel and GPU Comms trigger kernel wait loops (Only if Debug worker is disabled)
    if(first_dlbuf!=nullptr && !pdctx->debug_worker_enabled()){
        if(first_dlbuf->waitCompressionStop() != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL {} Map {} got COMPRESSION DL wait error", task_num , slot_map->getId());
            gpu_wait_error=true;
            goto cleanup;
        }

        //Note: Since we have already waited on Slot Channel End, all prepare work has been submitted and we can access prepare events
        for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; ++i)
        {
            if (prepare_tx_dlbuf_per_nic[i])
            {
                int failure = non_blocking_event_wait_with_timeout(prepare_tx_dlbuf_per_nic[i]->getTxEndEvt(),channel_wait_th);
                if(failure != 0) {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"GPU Comms Trigger kernel wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),channel_wait_th.count());
                    gpu_wait_error=true;
                    goto cleanup;
                }
            }
        }
    }

cleanup:

    if(gpu_wait_error)
    {
        if(pdctx->getDlCb(dl_cb))
        {
                auto msg_id = (!pdcch_ul_cuphy_done)? SCF_FAPI_DL_TTI_REQUEST : SCF_FAPI_UL_DCI_REQUEST;
                dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context, slot_ind,msg_id,SCF_ERROR_CODE_L1_DL_GPU_ERROR,cell_idx_list,cell_count);
        }
    }

    if(pdsch && !slot_map->pdsch_cb_done){
        //Trigger PDSCH callback before Slot Map release if CB is not completed
        ti.add("PDSCH Callback");
        PUSH_RANGE_PHYDRV((std::string("DL CALLBACK") + std::to_string(slot_map->getId())).c_str(), 3);
        slot_map->timings.start_t_dl_callback = Time::nowNs();
        pdsch->callback(/* slot_map->aggr_slot_params, */ slot_map->getSlot3GPP());
        slot_map->timings.end_t_dl_callback = Time::nowNs();
        slot_map->pdsch_cb_done=true;
        POP_RANGE_PHYDRV
    }

    ti.add("Slot Map Release");
    PUSH_RANGE_PHYDRV((std::string("DL CLEANUP") + std::to_string(slot_map->getId())).c_str(), 3);
    slot_map->release(num_cells);
    POP_RANGE_PHYDRV

    start_t_2 = Time::nowNs();
    ti.add("End Task");
    // NVSLOGI(TAG) << "SFN " << sfn << "." << slot
    //             << " Task DL " << task_num << " Map " << slot_map->getId() << " DL objects " << slot_map->getNumCells()
    //             << " Started at " << start_t_1.count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsExec(0)).count() << " us "
    //             << " L2 tick at " << slot_map->getTaskTsExec(0).count()
    //             << " after " << Time::NsToUs(start_t_1 - slot_map->getTaskTsEnq()).count() << " us "
    //             << " slot cmd enqueue at " << slot_map->getTaskTsEnq().count()
    //             << " Task duration " << Time::NsToUs(start_t_2 - start_t_1).count() << " us "
    //             << " Exec time in " << Time::NsToUs(slot_map->getTaskTsExec(task_num) - start_t_1).count() << " uson CPU " << (int)cpu;

    return 0;

exit_error:
    return -1;

}

//NOTE 1 : The default path now invokes the below task which is fanned out across multiple threads/cores, task_work_function_dl_aggr_2_gpu_comm_tx task and not the "task_work_function_dl_aggr_2_gpu_comm" task. 
//NOTE 2 : The cuphyBatchedMemcpyHelper code path exercised in the below task presently will only work functionally if the batched memcpy setting is disabled owing to the applicalbe methods not called in a thread-safe manner.
int task_work_function_dl_aggr_2_gpu_comm_prepare(Worker* worker, void* param, int task_num, int first_cell, int num_tasks) {
    char name[64];
    sprintf(name, "DL Task GPU Comms Prepare %d", task_num + 1);
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, name, 14);

    ti.add("Start Task");
    SlotMapDl* slot_map = (SlotMapDl*)param;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy* fhproxy = pdctx->getFhProxy();
    t_ns start_tx;
    int sfn = slot_map->getSlot3GPP().sfn_;
    int slot = slot_map->getSlot3GPP().slot_;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    
    DLOutputBuffer* compression_dlbuf = nullptr;
    DLOutputBuffer* prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    int cell_index_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {0};
    PreparePRBInfo prb_info_per_nic[MAX_NUM_OF_NIC_SUPPORTED];
    cudaStream_t first_dl_stream = nullptr;
    int ret = 0;
    const uint8_t dlc_packing_scheme = pdctx->get_dlc_core_packing_scheme();
    cuphyBatchedMemcpyHelper& batchedMemcpyHelper = slot_map->getBatchedMemcpyHelper();
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);    

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    ti.add("Wait Slot Start Task"); // Synchronize the Compression task and GPU Uplane prepare
    if(slot_map->waitFHCBDone() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitFHCBDone returned error");
        goto exit_error;
    }

    // When DL affinity is disabled, wait for C-plane task to complete for this task_num
    // This ensures proper serialization between task_work_function_cplane and this task
    if (pdctx->getmMIMO_enable() && pdctx->get_enable_dl_core_affinity() == 0) {
        ti.add("Wait CPlane Done");
        if (slot_map->waitCplaneDoneForTask(task_num) < 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitCplaneDoneForTask returned error for task_num {}", task_num);
            goto exit_error;
        }
    }

    ti.add("UPlane Prepare");
    PUSH_RANGE_PHYDRV((std::string("DL UPL PREP") + std::to_string(slot_map->getId())).c_str(), 3);

    //Deduce first_dl_stream here
    for(int i = first_cell;i < slot_map->getNumCells(); i++)
    {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;    
        first_dl_stream = cell_ptr->getDlStream();
        break;
    }
    
    for(int i = 0; i < slot_map->getNumCells(); i++) {
        // Apply packing scheme filter
        if (dlc_packing_scheme == 0) {
            // Scheme 0 (default): strided iteration - task_num handles cells task_num, task_num+num_tasks, etc.
            if ((i % num_tasks) != task_num) continue;
        } else if (dlc_packing_scheme == 1) {
            // Scheme 1 (fixed per-cell): check if cell's dlc_core_index matches this task_num
            Cell* check_cell = slot_map->aggr_cell_list[i];
            if (check_cell == nullptr) continue;
            if (check_cell->getDlcCoreIndex() != task_num) continue;
        }
        // Note: Scheme 2 (dynamic workload-based) is not yet supported and validated during init

        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        auto nic_index = cell_ptr->getNicIndex();
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;


        // Calculate start_tx based on UE mode
        if(slot_map->aggr_pdsch && (pdctx->getUeMode()!=0)) {
            start_tx = slot_map->getTaskTsExec(0) + (t_ns((Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) + cell_ptr->getUlUplaneTxOffsetNs()));
        } else {
            start_tx = slot_map->getTaskTsExec(0) + (t_ns((Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead())) - cell_ptr->getT1aMaxUpNs()));
        }

        slot_map->timings.start_t_dl_uprep[i] = Time::nowNs();

        if(slot_map->aggr_slot_info[i]) {
            pdctx->setGpuCommsCtx();
            if(fhproxy->prepareUPlanePackets(
                cell_ptr->getRUType(),
                cell_ptr->getPeerId(),
                first_dl_stream,
                start_tx,
                slot_oran_ind,
                *(slot_map->aggr_slot_info[i]),
                dlbuf->getTxMsgContainer(),
                dlbuf->getSize(),
                dlbuf->getModCompressionConfig(),
                dlbuf->getModCompressionTempConfig(),
                dlbuf, t_ns(SYMBOL_DURATION_NS),
                batchedMemcpyHelper)) {
                EXIT_L1(EXIT_FAILURE);
            }

            // Setup PRB info
            const auto& eaxc = cell_ptr->geteAxCIdsPdsch();
            auto cell_index = cell_index_per_nic[nic_index];
            ++cell_index_per_nic[nic_index];
            prb_info_per_nic[nic_index].prb_ptrs[cell_index] = dlbuf->getPrbPtrs();
            prb_info_per_nic[nic_index].num_antennas[cell_index] = eaxc.size();
            prb_info_per_nic[nic_index].max_num_prb_per_symbol[cell_index] = cell_ptr->getDLGridSize();

            for (auto ant = 0; ant < eaxc.size(); ant++) {
                prb_info_per_nic[nic_index].eAxCMap[cell_index][eaxc[ant] & 0x1f] = ant;
            }
        }

        slot_map->timings.end_t_dl_uprep[i] = Time::nowNs();
    }

    // Execute batched memcpy if needed
    if((first_dl_stream != nullptr) && (batchedMemcpyHelper.getMemcpyCount() > 0)) {
        cuphyStatus_t batched_memcpy_status = batchedMemcpyHelper.launchBatchedMemcpy(first_dl_stream);
        if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task DL Aggr_2 {} Prepare Map {} got launchBatchedMemcpy error", task_num,slot_map->getId());
        }
    }
    POP_RANGE_PHYDRV

    ti.add("Signal UPlane Prepare End");
    slot_map->incUplanePrepDone();
    slot_map->addSlotEndTask();

    ti.add("End Task");
    return 0;
    
exit_error:
    return -1;    
}

int task_work_function_dl_aggr_2_gpu_comm_tx(Worker* worker, void* param, int first_cell, int num_cells, int num_tasks) {
    auto ctx = makeInstrumentationContextDL(param, worker);
    TaskInstrumentation ti(ctx, "DL Task GPU Comms TX", 13);
    ti.add("Start Task");
    SlotMapDl* slot_map = (SlotMapDl*)param;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    FhProxy* fhproxy = pdctx->getFhProxy();
    struct slot_command_api::dl_slot_callbacks dl_cb;
    DLOutputBuffer* compression_dlbuf = nullptr;
    DLOutputBuffer* prepare_tx_dlbuf_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {nullptr};
    PreparePRBInfo prb_info_per_nic[MAX_NUM_OF_NIC_SUPPORTED];
    int cell_index_per_nic[MAX_NUM_OF_NIC_SUPPORTED] = {0};
    int slot = slot_map->getSlot3GPP().slot_;
    int ret = 0;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    CuphyPtiSetIndexScope cuphy_pti_index_scope(slot);   

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    ti.add("Wait Uplane Prepare"); // Synchronize the UPlane prepare tasks
    if(slot_map->waitUplanePrepDone(num_tasks) < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitUplanePrepDone returned error");
        goto exit_error;
    }     

    //Init size 
    for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++)
    {
        slot_map->tx_v_for_slot_map[i].size=0;
    }         

    // Get first compression dlbuf and prepare_tx_dlbuf for each NIC
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++) {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        auto nic_index = cell_ptr->getNicIndex();
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        
        if (!compression_dlbuf) {
            compression_dlbuf = dlbuf;
        }
        if (!prepare_tx_dlbuf_per_nic[nic_index]) {
            prepare_tx_dlbuf_per_nic[nic_index] = dlbuf;
        }

        // Setup PRB info
        const auto& eaxc = cell_ptr->geteAxCIdsPdsch();
        auto cell_index = cell_index_per_nic[nic_index];
        ++cell_index_per_nic[nic_index];
        prb_info_per_nic[nic_index].prb_ptrs[cell_index] = dlbuf->getPrbPtrs();
        prb_info_per_nic[nic_index].num_antennas[cell_index] = eaxc.size();
        prb_info_per_nic[nic_index].max_num_prb_per_symbol[cell_index] = cell_ptr->getDLGridSize();

        for (auto ant = 0; ant < eaxc.size(); ant++) {
            prb_info_per_nic[nic_index].eAxCMap[cell_index][eaxc[ant] & 0x1f] = ant;
        }

        struct umsg_fh_tx_msg& txmsg = dlbuf->getTxMsgContainer();

        //Check if NIC is in pre-allocated range
        if(cell_ptr->getNicIndex() >= MAX_NUM_OF_NIC_SUPPORTED) {
            NVLOGF_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"NIC index of {} out of bounds.  Max num of NICs supported is {}",
                        cell_ptr->getNicIndex(), MAX_NUM_OF_NIC_SUPPORTED);
            continue;
        }

        //Check if NIC has too many TX REQ slots
        if((slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].size >=  MAX_NUM_TX_REQ_UPLANE_GPU_COMM_PER_NIC))
        {
            NVLOGF_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"Number of tx req is more than max for NIC {} {}",
                        slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].nic_name.c_str(), cell_ptr->getNicIndex());
            continue;
        }

        slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].nic_name = cell_ptr->getNicName();
        slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].tx_v_per_nic[slot_map->tx_v_for_slot_map[cell_ptr->getNicIndex()].size++] = txmsg.txrq_gpu;                
    }
   

    ti.add("Uplane TX");
    pdctx->setGpuCommsCtx();
    PUSH_RANGE_PHYDRV((std::string("DL UPL GTX") + std::to_string(slot_map->getId())).c_str(), 3);
    slot_map->timings.start_t_dl_utx[0] = Time::nowNs();

    for(int i = 0; i < MAX_NUM_OF_NIC_SUPPORTED; i++) {
        if(slot_map->tx_v_for_slot_map[i].size) {
            if(!prepare_tx_dlbuf_per_nic[i]) {
                goto cleanup;
            }

            prb_info_per_nic[i].ready_flag = prepare_tx_dlbuf_per_nic[i]->getReadyFlag();
            prb_info_per_nic[i].wait_val = 1;
            prb_info_per_nic[i].compression_stop_evt = compression_dlbuf->getCompressionStopEvt();
            prb_info_per_nic[i].comm_start_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareStartEvt();
            prb_info_per_nic[i].comm_copy_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareCopyEvt();
            prb_info_per_nic[i].comm_preprep_stop_evt = prepare_tx_dlbuf_per_nic[i]->getPrePrepareStopEvt();
            prb_info_per_nic[i].comm_stop_evt = prepare_tx_dlbuf_per_nic[i]->getPrepareStopEvt();
            prb_info_per_nic[i].trigger_end_evt = prepare_tx_dlbuf_per_nic[i]->getTxEndEvt();
            prb_info_per_nic[i].enable_prepare_tracing = pdctx->enablePrepareTracing();
            prb_info_per_nic[i].enable_dl_cqe_tracing = pdctx->enableDlCqeTracing();
            prb_info_per_nic[i].disable_empw = pdctx->disableEmpw();
            prb_info_per_nic[i].cqe_trace_cell_mask = pdctx->get_cqe_trace_cell_mask();
            prb_info_per_nic[i].cqe_trace_slot_mask = pdctx->get_cqe_trace_slot_mask();

            if(0 != fhproxy->UserPlaneSendPacketsGpuComm(&slot_map->tx_v_for_slot_map[i], prb_info_per_nic[i])) {
                NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "UserPlaneSendPacketsGpuComm returned fatal error");
                EXIT_L1(EXIT_FAILURE);
            }
        }
    }

    slot_map->timings.end_t_dl_utx[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Slot Channel End");
    slot_map->addSlotChannelEnd();
    slot_map->setDlGpuCommEnd();
    pdctx->setDlCtx();

    // Handle PDSCH callback if needed
    if(slot_map->aggr_pdsch != nullptr && !slot_map->pdsch_cb_done) {
        ti.add("PDSCH TB H2D Wait");
        t_ns start_t = Time::nowNs();
        t_ns h2d_wait_th(pdctx->geth2d_copy_wait_th());
        bool pdsch_tb_h2d_done = false;

        while(!pdsch_tb_h2d_done) {
            pdsch_tb_h2d_done = (bool)slot_map->aggr_pdsch->waitEventNonBlocking(pdctx->get_event_pdsch_tb_cpy_complete((uint8_t)slot_map->getSlot3GPP().slot_));
            if(!pdsch_tb_h2d_done && ((Time::nowNs()-start_t)>h2d_wait_th)) {
                NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"PDSCH TB H2D copy wait ERROR on Map {}! Wait timeout after {} ns",slot_map->getId(),h2d_wait_th.count());
                break;
            }
        }

        ti.add("PDSCH Callback");
        PUSH_RANGE_PHYDRV((std::string("DL CALLBACK") + std::to_string(slot_map->getId())).c_str(), 3);
        slot_map->timings.start_t_dl_callback = Time::nowNs();
        slot_map->aggr_pdsch->callback(slot_map->getSlot3GPP());
        slot_map->timings.end_t_dl_callback = Time::nowNs();
        slot_map->pdsch_cb_done = true;
        POP_RANGE_PHYDRV
    }

    ti.add("Metric Update");
    for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++) {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        cell_ptr->updateMetric(CellMetric::kDlSlotsTotal, 1);

        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if(cell_ptr == nullptr || dlbuf == nullptr) continue;
        struct umsg_fh_tx_msg& txmsg = dlbuf->getTxMsgContainer();
        fhproxy->UpdateTxMetricsGpuComm(cell_ptr->getPeerId(), txmsg);
    }

cleanup:
    slot_map->addSlotEndTask();

    ti.add("End Task");
    return 0;

exit_error:
    return -1;
}
