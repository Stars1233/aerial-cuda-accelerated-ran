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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 25) // "DRV.FUNC_UL"

#include "cuphydriver_api.hpp"
#include "app_config.hpp"
#include "constant.hpp"
#include "context.hpp"
#include "time.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "order_entity.hpp"
#include <unordered_map>
#include "aerial-fh-driver/oran.hpp"
#include <sched.h>
#include "task_instrumentation_v3.hpp"
#include "task_instrumentation_v3_factories.hpp"
#include "memtrace.h"
#include "nvlog_fmt.hpp"
#include "cuphy_pti.hpp"
#include "cupti_helper.hpp"
#include <optional>
#include "scf_5g_fapi.h"
#include "cuphyoam.hpp"

int task_work_function_ul_aggr_bfw(Worker* worker, void* param, int first_cell, int num_cells, int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task BFW", 13);
    ti.add("Start Task");
    int                                                                          ret = 0;
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    t_ns                                                                         start_t_1, start_t_2, start_tx;
    t_ns end_time;
    int                                                                          sfn = 0, slot = 0;
    PhyUlBfwAggr* ulbfw = slot_map->aggr_ulbfw;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"getcpu failed for {}", __FUNCTION__);
        goto exit_error;
    }

    if(ulbfw == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No ulbfw in SlotMap UL");
        goto exit_error;
    }

    NVLOGI_FMT(TAG,"UL Task (BFW) started CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);

    start_t_1 = Time::nowNs();

    /////////////////////////////////////////////////////////////////////////////////////
    //// cuPHY Setup + Run
    /////////////////////////////////////////////////////////////////////////////////////

    PUSH_RANGE_PHYDRV((std::string("UL BFW") + std::to_string(slot_map->getId())).c_str(), 2);
    slot_map->timings.start_t_ul_bfw_cuda[0] = Time::nowNs();
    if (ulbfw) {
        try
        {
            ti.add("Cuda Setup");
            if (ulbfw->setup(slot_map->aggr_cell_list))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL BFW Setup returned error for Map {}",slot_map->getId());
                ulbfw->setSetupStatus(CH_SETUP_DONE_ERROR);
                //goto exit_error;
            }
            else
                ulbfw->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            slot_map->timings.start_t_ul_bfw_run[0] = Time::nowNs();

            ti.add("Cuda Run");
            if(ulbfw->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL BFW run returned error for Map {}",slot_map->getId());
                ulbfw->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getUlCb(ul_cb))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                    ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_BFW_CVI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                }
                //goto exit_error;
            }
            else
                ulbfw->setRunStatus(CH_RUN_DONE_NO_ERROR);

            ti.add("Signal Completion");
            if(ulbfw->signalRunCompletionEvent(ulbfw->getStream(),false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ULBFW signalCompletion returned error");
                goto exit_error;
            }
            pdctx->recordUlBFWCompletion(slot);
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_ul_bfw_cuda[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    slot_map->setUlBfwEndTask();
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    start_t_2 = Time::nowNs();

    ti.add("End Task");

    return 0;

//FIXME: abort the whole DL slot in case of error
exit_error:
    return -1;
}

int task_work_function_ul_aggr_1_pucch_pusch(Worker* worker, void* param, int first_cell, int num_cells, int num_ul_tasks) {
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task PUCCH and PUSCH", 20);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    int                                                                          ret = 0, i = 0;
    PhyPuschAggr* pusch = slot_map->aggr_pusch;
    PhyPucchAggr* pucch = slot_map->aggr_pucch;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = slot_map->aggr_order_entity;
    ULInputBuffer * ulbuf_st1 = nullptr;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pucch_pusch timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pucch_pusch Task aborted for Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Determine current sfn/slot
    int sfn, slot;
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;

    

    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    //Determine PUSCH Phase1, PUSCH Phase1, and PUCCH stream
    cudaStream_t* pusch_streams = pdctx->getUlOrderStreamsPusch();
    cudaStream_t* pucch_streams = pdctx->getUlOrderStreamsPucch();
    cudaStream_t phase1_stream;
    cudaStream_t phase2_stream;
    cudaStream_t pucch_stream;
    bool serialize_pucch_pusch = pdctx->serializePucchPusch();
    if(pdctx->splitUlCudaStreamsEnabled()) //Revisit for TDD pattern, Assume DDDSUUDDDD pattern
    {
        phase1_stream = ((slot % 10 == 3) || (slot % 10 == 4))? pusch_streams[PHASE1_SPLIT_STREAM1] : pusch_streams[PHASE1_SPLIT_STREAM2];
        phase2_stream = ((slot % 10 == 3) || (slot % 10 == 4))? pusch_streams[PHASE2_SPLIT_STREAM1] : pusch_streams[PHASE2_SPLIT_STREAM2];

        pucch_stream = ((slot % 10 == 3) || (slot % 10 == 4))? pucch_streams[0] : pucch_streams[1];
    }
    else
    {
        phase1_stream = pusch_streams[PHASE1_SPLIT_STREAM1];
        phase2_stream = pusch_streams[PHASE2_SPLIT_STREAM1];

        pucch_stream = pucch_streams[0];
    }

    //Determine current cpu
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto error_next;
    }


    ////PUCCH SETUP
    if(pucch != nullptr) {
        PUSH_RANGE_PHYDRV("UL_CUPHY_PUCCH_SETUP", 1);
        slot_map->timings.start_t_ul_pucch_cuda[0] = Time::nowNs();
        try {
            ti.add("PUCCH Cuda Setup");
            if(pucch->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st1, pucch_stream))
            {
                NVLOGW_FMT(TAG,"PUCCH setup returned error for Map {}",slot_map->getId());
                pucch->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                pucch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
        POP_RANGE_PHYDRV
    }


    ////PUSCH SETUP
    if(pusch != nullptr) {
        PUSH_RANGE_PHYDRV("UL_CUPHY_PUSCH_SETUP", 1);
        slot_map->timings.start_t_ul_pusch_cuda[0] = Time::nowNs();
        try {
            ti.add("PUSCH Cuda Setup");
            if(pusch->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st1, phase1_stream, phase2_stream))
            {
                NVLOGW_FMT(TAG, "PUSCH setup returned error for Map {}",slot_map->getId());
                pusch->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
            {
                pusch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
                slot_map->setIsEarlyHarqPresent(pusch->getPuschDynParams()->pDataOut->isEarlyHarqPresent);
                slot_map->setIsFrontLoadedDmrsPresent(pusch->getPuschDynParams()->pDataOut->isFrontLoadedDmrsPresent);
            }

        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
        POP_RANGE_PHYDRV
    }


    ////Pause CPU until order kernel is launched
    ti.add("Wait for Order Launch");
    if(!oentity->getOrderLaunchedStatus()){
        if(oentity->waitOrderLaunched(1*NS_X_MS))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: waitOrderLaunched returned error for Map {}",__func__,slot_map->getId());
            goto error_next;
        }
    }

    ti.add("PUSCH Stream Wait");
    if(pusch != nullptr) {
            //Wait on Order kernel completion done event if there is no early-HARQ UEs or no front-loaded DM-RS UEs
            if(slot_map->getIsEarlyHarqPresent()==0 && slot_map->getIsFrontLoadedDmrsPresent()==0){
                oentity = slot_map->aggr_order_entity;
                pusch->waitToStartGPUEvent(oentity->getRunCompletionEvt(), phase1_stream);
            }
    }

    ////PUSCH RUN1
    if(pusch != nullptr) {
        PUSH_RANGE_PHYDRV("UL_CUPHY_PUSCH_RUN1", 1);
        slot_map->timings.start_t_ul_pusch_run[0] = Time::nowNs();
        try {

            ti.add("PUSCH Cuda Run");

            //Run call ordering used when running with PUSCH EH
            //PUSCH_RUN_EARLY_HARQ_PROC  = 1, // processing sub-slot OFDMs in early-HARQ + D2H copies of early-HARQ result
            //                                Note - runs on phase1_stream
            //PUSCH_RUN_FULL_SLOT_PROC   = 2, // processing full-slot OFDMs (aka non-early-HARQ)
            //                                Note - runs on phase1_stream
            //PUSCH_RUN_FULL_SLOT_COPY   = 3, // copying output of PUSCH_RUN_FULL_SLOT_PROC from GPU to CPU
            //                                Note - runs on phase2_stream

            //Run call ordering used when running without PUSCH EH
            //PUSCH_RUN_ALL_PHASES       = 4, // PUSCH_RUN_EARLY_HARQ_PROC + PUSCH_RUN_FULL_SLOT_PROC + PUSCH_RUN_FULL_SLOT_COPY
            //                                Note - this phase is just running all PUSCH run phases on both streams
            //                                PUSCH_RUN_EARLY_HARQ_PROC and PUSCH_RUN_FULL_SLOT_PROC still run on phase1_stream, PUSCH_RUN_FULL_SLOT_COPY still runs on phase2_stream
            if((slot_map->getIsEarlyHarqPresent()==1) || (slot_map->getIsFrontLoadedDmrsPresent()==1))
            {
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_SUB_SLOT_PROC returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
        POP_RANGE_PHYDRV
    }

    ////Pause pucch stream
    ti.add("PUCCH Stream Wait");
    if(pucch != nullptr) {
        oentity = slot_map->aggr_order_entity;
        pucch->waitToStartGPUEvent(oentity->getRunCompletionEvt(), pucch_stream);
    }

    ////PUCCH RUN
    if(pucch != nullptr) {
        PUSH_RANGE_PHYDRV("UL_CUPHY_PUCCH_RUN", 1);
        slot_map->timings.start_t_ul_pucch_run[0] = Time::nowNs();
        try {

            //If pusch eh enabled, have pucch stream wait on pusch eh
            if(serialize_pucch_pusch && pusch != nullptr)
            {
                if(slot_map->getIsEarlyHarqPresent() == 1 || slot_map->getIsFrontLoadedDmrsPresent() == 1)
                {
                    pucch->waitToStartGPUEvent(pusch->getPuschStatParams()->subSlotCompletedEvent,pucch_stream);
                }
            }

            ti.add("PUCCH Cuda Run");
            if(pucch->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUCCH run returned error for Map {}",slot_map->getId());
                pucch->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getUlCb(ul_cb))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                    ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(), SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                }
                //goto error_next;
            }
            else
                pucch->setRunStatus(CH_RUN_DONE_NO_ERROR);

            ti.add("PUCCH Signal Completion");
            if(pucch->signalRunCompletionEvent(pucch_stream,false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUCCH signalRunCompletionEvent returned error");
                goto error_next;
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
        slot_map->timings.end_t_ul_pucch_cuda[0] = Time::nowNs();
        POP_RANGE_PHYDRV
    }

    ////PUSCH RUN2
    if(pusch != nullptr) {
        PUSH_RANGE_PHYDRV("UL_CUPHY_PUSCH_RUN2", 1);
        try {

            //If pucch is enabled, have pusch stream wait on pucch
            if(serialize_pucch_pusch && pucch != nullptr) {
                pusch->waitToStartGPUEvent(pucch->getRunCompletionEvent(),phase1_stream);
            }

            if((slot_map->getIsEarlyHarqPresent()==1) || (slot_map->getIsFrontLoadedDmrsPresent()==1))
            {

                //NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Is this hit?");
#if 0
                // Potential test case 1 - periodically set the early exit flag to 1 for specific slot.
                //if ((sfn == 4) && (slot == 14)) {
                if (slot == 14) {
                   NVLOGC_FMT(TAG, "Will set slot early exit flag for sfn {} slot {} (one of EH or FL DMRS is present)", sfn, slot);
                   pusch->setWorkCancelFlag();
                }
#endif
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_PROC))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_FULL_SLOT_PROC returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(), SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);
            }
            else
            {

                //Note: PUCCH gets to run before all phases of PUSCH if running non-EH
                //This is to accomodate (d) variant where all early UCI is on PUCCH
#if 0
                // Potential test case 1 - periodically set the early exit flag to 1 for specific slot.
                //if ((sfn == 4) && (slot == 14)) {
                if (slot == 14) { // randomly selected
                   NVLOGC_FMT(TAG, "Will set slot early exit flag for sfn {} slot {}, neither EH nor FL DMRS present", sfn, slot);
                   pusch->setWorkCancelFlag();
                }
#endif
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_ALL_PHASES returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(), SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);

                ti.add("PUSCH Signal Completion");
                if(pusch->signalRunCompletionEvent(phase2_stream,false))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH signalRunCompletionEvent returned error");
                    goto error_next;
                }
            }

            if((slot_map->getIsEarlyHarqPresent()==0) && (slot_map->getIsFrontLoadedDmrsPresent()==1))
            {
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_FULL_SLOT_COPY returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);

                ti.add("Signal Completion");
                if(pusch->signalRunCompletionEvent(phase2_stream,false))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH signalRunCompletionEvent returned error");
                    goto error_next;
                }
            }

#ifdef EARLY_UCI_CUBB_CALLFLOW_TEST
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(pusch->getPuschStatParams()->subSlotCompletedEvent,phase1_stream));
#endif
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
        slot_map->timings.end_t_ul_pusch_cuda[0] = Time::nowNs();
        POP_RANGE_PHYDRV
    }

    ti.add("Signal Channel End Task");
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    
    ti.add("End Task");

    return 0;

    error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} line {}: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_pucch_pusch aborted the tasklist for an error");
    return -1;
}


int task_work_function_ul_aggr_1_pusch(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task PUSCH", 12);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    int                                                                          ret = 0, i = 0;
    t_ns                                                                         start_t_1, start_t_3, start_tx;
    int                                                                          sfn = 0, slot = 0;
    PhyPuschAggr* pusch = slot_map->aggr_pusch;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = slot_map->aggr_order_entity;
    ULInputBuffer * ulbuf_st1 = nullptr;
    t_ns t1, t2, t3;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pusch timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pusch Task aborted for Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    cudaStream_t* streams = pdctx->getUlOrderStreamsPusch();
    cudaStream_t phase1_stream;
    cudaStream_t phase2_stream;

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
        goto error_next;
    }
    //NVLOGI_FMT(TAG,"UL Task 1(PUSCH) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);

/*
    if(slot_map->waitChannelStartTask() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitChannelStartTask returned error");
        goto error_next;
    }
*/
    start_t_1 = Time::nowNs();
    if(pusch == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH Aggr is NULL");
        goto error_next;
    }

    //NVLOGC_FMT(TAG, "Size of occasion array = {}", slot_map->num_prach_occa.size());

    PUSH_RANGE_PHYDRV("UL_CUPHY_PUSCH", 1);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// cuPHY UL PUSCH Channel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(pdctx->splitUlCudaStreamsEnabled()) //Revisit for TDD pattern, Assume DDDSUUDDDD pattern
    {
        phase1_stream = ((slot % 10 == 3) || (slot % 10 == 4))? streams[PHASE1_SPLIT_STREAM1] : streams[PHASE1_SPLIT_STREAM2];
        phase2_stream = ((slot % 10 == 3) || (slot % 10 == 4))? streams[PHASE2_SPLIT_STREAM1] : streams[PHASE2_SPLIT_STREAM2];
    }
    else
    {
        phase1_stream = streams[PHASE1_SPLIT_STREAM1];
        phase2_stream = streams[PHASE2_SPLIT_STREAM1];
    }

    slot_map->timings.start_t_ul_pusch_cuda[0] = Time::nowNs();
    if(pusch != nullptr)
    {
        try {
            ti.add("Cuda Setup");
            if(pusch->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st1, phase1_stream, phase2_stream))
            {
                NVLOGW_FMT(TAG, "PUSCH setup returned error for Map {}",slot_map->getId());
                pusch->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
            {
                pusch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);
                slot_map->setIsEarlyHarqPresent(pusch->getPuschDynParams()->pDataOut->isEarlyHarqPresent);
                slot_map->setIsFrontLoadedDmrsPresent(pusch->getPuschDynParams()->pDataOut->isFrontLoadedDmrsPresent);
            }

            t1 = Time::nowNs();

            ti.add("Wait for Order Launch");
            if(!oentity->getOrderLaunchedStatus()){
                if(oentity->waitOrderLaunched(1*NS_X_MS))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: waitOrderLaunched returned error for Map {}",__func__,slot_map->getId());
                    goto error_next;
                }
            }

            ti.add("Wait To Start GPU");
            //Wait on Order kernel completion done event if there is no early-HARQ UEs or no front-loaded DM-RS UEs
            if(slot_map->getIsEarlyHarqPresent()==0 && slot_map->getIsFrontLoadedDmrsPresent()==0){
                oentity = slot_map->aggr_order_entity;
                pusch->waitToStartGPUEvent(oentity->getRunCompletionEvt(), phase1_stream);
            }

            // t2 = Time::nowNs();
            slot_map->timings.start_t_ul_pusch_run[0] = Time::nowNs();

            ti.add("Cuda Run");
            //PUSCH_RUN_EARLY_HARQ_PROC  = 1, // processing sub-slot OFDMs in early-HARQ + D2H copies of early-HARQ result
            //                                  Note - runs on phase1_stream
            //PUSCH_RUN_FULL_SLOT_PROC   = 2, // processing full-slot OFDMs (aka non-early-HARQ)
            //                                  Note - runs on phase1_stream
            //PUSCH_RUN_FULL_SLOT_COPY   = 3, // copying output of PUSCH_RUN_FULL_SLOT_PROC from GPU to CPU
            //                                  Note - runs on phase2_stream
            //PUSCH_RUN_ALL_PHASES       = 4, // PUSCH_RUN_EARLY_HARQ_PROC + PUSCH_RUN_FULL_SLOT_PROC + PUSCH_RUN_FULL_SLOT_COPY
            if((slot_map->getIsEarlyHarqPresent()==0) && (slot_map->getIsFrontLoadedDmrsPresent()==0))
            {
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase 3 returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);

                // t3 = Time::nowNs();
                ti.add("Signal Completion");
                if(pusch->signalRunCompletionEvent(phase2_stream,false))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH signalRunCompletionEvent returned error");
                    goto error_next;
                }
            }
            else
            {
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_SUB_SLOT_PROC returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);

                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_PROC))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_FULL_SLOT_PROC returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                    //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);
            }

            if((slot_map->getIsEarlyHarqPresent()==0) && (slot_map->getIsFrontLoadedDmrsPresent()==1))
            {
                if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_FULL_SLOT_COPY returned error for Map {}",slot_map->getId());
                    pusch->setRunStatus(CH_RUN_DONE_ERROR);
                    slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                    if(pdctx->getUlCb(ul_cb))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                    }
                //goto error_next;
                }
                else
                    pusch->setRunStatus(CH_RUN_DONE_NO_ERROR);

                ti.add("Signal Completion");
                if(pusch->signalRunCompletionEvent(phase2_stream,false))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH signalRunCompletionEvent returned error");
                    goto error_next;
                }
            }

#ifdef EARLY_UCI_CUBB_CALLFLOW_TEST
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(pusch->getPuschStatParams()->subSlotCompletedEvent,phase1_stream));
#endif
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_ul_pusch_cuda[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Channel End Task");
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    start_t_3 = Time::nowNs();

    ti.add("End Task");

    return 0;

    error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} line {}: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_pusch aborted the tasklist for an error");
    return -1;
}

int task_work_function_ul_aggr_1_pucch(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task PUCCH", 10);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    int                                                                          ret = 0, i = 0;
    t_ns                                                                         start_t_1, start_t_3, start_tx;
    int                                                                          sfn = 0, slot = 0;
    PhyPucchAggr* pucch = slot_map->aggr_pucch;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = slot_map->aggr_order_entity;
    ULInputBuffer * ulbuf_st1 = nullptr;
    t_ns t1, t2, t3;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    cudaStream_t* streams = pdctx->getUlOrderStreamsPucch();
    cudaStream_t stream;

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pucch timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_pucch Task aborted for Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

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
        goto error_next;
    }

    //NVLOGI_FMT(TAG,"UL Task 1(PUCCH) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);

/*
    if(slot_map->waitChannelStartTask() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitChannelStartTask returned error");
        goto error_next;
    }
*/
    start_t_1 = Time::nowNs();
    if(pucch == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUCCH Aggr is NULL");
        goto error_next;
    }

    //NVLOGC_FMT(TAG, "Size of occasion array = {}", slot_map->num_prach_occa.size());

    PUSH_RANGE_PHYDRV("UL_CUPHY_PUCCH", 1);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// cuPHY UL PUCCH Channel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(pdctx->splitUlCudaStreamsEnabled()) //Revisit for TDD pattern, Assume DDDSUUDDDD pattern
    {
        stream = ((slot % 10 == 3) || (slot % 10 == 4))? streams[0] : streams[1];
    }
    else
    {
        stream = streams[0];
    }

    slot_map->timings.start_t_ul_pucch_cuda[0] = Time::nowNs();
    if(pucch != nullptr)
    {
        try {
            ti.add("Cuda Setup");
            if(pucch->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st1, stream))
            {
                NVLOGW_FMT(TAG,"PUCCH setup returned error for Map {}",slot_map->getId());
                pucch->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                pucch->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            t1 = Time::nowNs();

            ti.add("Wait for Order Launch");
            if(!oentity->getOrderLaunchedStatus()){
                if(oentity->waitOrderLaunched(1*NS_X_MS))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: waitOrderLaunched returned error for Map {}",__func__,slot_map->getId());
                    goto error_next;
                }
            }
            ti.add("Wait To Start GPU");
            oentity = slot_map->aggr_order_entity;
            pucch->waitToStartGPUEvent(oentity->getRunCompletionEvt(), stream);

            // t2 = Time::nowNs();
            slot_map->timings.start_t_ul_pucch_run[0] = Time::nowNs();

            ti.add("Cuda Run");
            if(pucch->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUCCH run returned error for Map {}",slot_map->getId());
                pucch->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getUlCb(ul_cb))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                    ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                }
                //goto error_next;
            }
            else
                pucch->setRunStatus(CH_RUN_DONE_NO_ERROR);

            // t3 = Time::nowNs();

            ti.add("Signal Completion");
            if(pucch->signalRunCompletionEvent(stream,false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUCCH signalRunCompletionEvent returned error");
                goto error_next;
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_ul_pucch_cuda[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Channel End Task");
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    start_t_3 = Time::nowNs();

    ti.add("End Task");

    return 0;

    error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} line {}: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_pucch aborted the tasklist for an error");
    return -1;
}

int task_work_function_ul_aggr_1_prach(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task PRACH", 10);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    int                                                                          ret = 0, i = 0;
    t_ns                                                                         start_t_1, start_t_3, start_tx;
    int                                                                          sfn = 0, slot = 0;
    PhyPrachAggr* prach = slot_map->aggr_prach;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = slot_map->aggr_order_entity;
    ULInputBuffer * ulbuf_st1 = nullptr;
    t_ns t1, t2, t3;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    cudaStream_t* streams = pdctx->getUlOrderStreamsPrach();
    cudaStream_t stream;

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_prach timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_prach Task aborted for Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

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
        goto error_next;
    }

    //NVLOGI_FMT(TAG,"UL Task 1(PRACH) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);

/*
    if(slot_map->waitChannelStartTask() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitChannelStartTask returned error");
        goto error_next;
    }
*/
    if(pdctx->splitUlCudaStreamsEnabled()) //Revisit for TDD pattern, Assume DDDSUUDDDD pattern
    {
        stream = ((slot % 10 == 3) || (slot % 10 == 4))? streams[0] : streams[1];
    }
    else
    {
        stream = streams[0];
    }

    start_t_1 = Time::nowNs();
    if(prach == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PRACH Aggr is NULL");
        goto error_next;
    }

    //NVLOGC_FMT(TAG, "Size of occasion array = {}", slot_map->num_prach_occa.size());

    PUSH_RANGE_PHYDRV("UL_CUPHY_PRACH", 1);

    slot_map->timings.start_t_ul_prach_cuda[0] = Time::nowNs();
    if(prach != nullptr)
    {
        try
        {
            ti.add("Cuda Setup");
            if(prach->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st3, stream))
            {
                NVLOGW_FMT(TAG, "PRACH setup returned error for Map {}",slot_map->getId());
                prach->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                prach->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            t1 = Time::nowNs();

            ti.add("Wait for Order Launch");
            if(!oentity->getOrderLaunchedStatus()){
                if(oentity->waitOrderLaunched(1*NS_X_MS))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: waitOrderLaunched returned error for Map {}",__func__,slot_map->getId());
                    goto error_next;
                }
            }
            ti.add("Wait To Start GPU");
            oentity = slot_map->aggr_order_entity;
            prach->waitToStartGPUEvent(oentity->getRunCompletionEvt(), stream);
            // t2 = Time::nowNs();
            slot_map->timings.start_t_ul_prach_run[0] = Time::nowNs();

            ti.add("Cuda Run");
            if(prach->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PRACH run returned error for Map {}",slot_map->getId());
                prach->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getUlCb(ul_cb))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                    ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                }
                //goto error_next;
            }
            else
                prach->setRunStatus(CH_RUN_DONE_NO_ERROR);
            // t3 = Time::nowNs();

            ti.add("Signal Completion");
            if(prach->signalRunCompletionEvent(stream,false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PRACH signalRunCompletionEvent returned error");
                goto error_next;
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_ul_prach_cuda[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Channel End Task");
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    start_t_3 = Time::nowNs();

    ti.add("End Task");

    return 0;

    error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} line {}: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_prach aborted the tasklist for an error");
    return -1;
}

int task_work_function_ul_aggr_1_srs(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task SRS", 10);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    int                                                                          ret = 0, i = 0;
    t_ns                                                                         start_t_1, start_t_3, start_tx;
    int                                                                          sfn = 0, slot = 0;
    PhySrsAggr* srs = slot_map->aggr_srs;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = slot_map->aggr_order_entity;
    ULInputBuffer * ulbuf_st1 = nullptr;
    t_ns t1, t2, t3;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    int cell_count=0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();

    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_srs timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_srs Task aborted for Slot Map {}",slot_map->getId());

        ti.add("Signal Channel End Task");
        slot_map->addChannelEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto error_next;
    }

    start_t_1 = Time::nowNs();
    if(srs == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "srs Aggr is NULL");
        goto error_next;
    }



    PUSH_RANGE_PHYDRV("UL_CUPHY_SRS", 1);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// cuPHY UL SRS Channel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    slot_map->timings.start_t_ul_srs_cuda[0] = Time::nowNs();
    if(srs != nullptr)
    {
        try {
            ti.add("Cuda Setup");
            if(srs->setup(slot_map->aggr_cell_list, slot_map->aggr_ulbuf_st2))
            {
                NVLOGW_FMT(TAG, "SRS setup returned error for Map {}",slot_map->getId());
                srs->setSetupStatus(CH_SETUP_DONE_ERROR);
            }
            else
                srs->setSetupStatus(CH_SETUP_DONE_NO_ERROR);

            t1 = Time::nowNs();

            ti.add("Wait for Order Launch");
            if(!oentity->getOrderLaunchedStatusSrs()){
                if(oentity->waitOrderLaunchedSrs(1*NS_X_MS))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: waitOrderLaunched returned error for Map {}",__func__,slot_map->getId());
                    goto error_next;
                }
            }

            ti.add("Wait To Start GPU");
            oentity = slot_map->aggr_order_entity;
            if(pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE) {
                srs->waitToStartGPUEvent(oentity->getRunCompletionEvt());
            } else {
                srs->waitToStartGPUEvent(oentity->getSrsRunCompletionEvt());
	    }

            // t2 = Time::nowNs();
            slot_map->timings.start_t_ul_srs_run[0] = Time::nowNs();

            ti.add("Cuda Run");
            if(srs->run())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SRS run returned error for Map {}",slot_map->getId());
                srs->setRunStatus(CH_RUN_DONE_ERROR);
                slot_map->getCellMplaneIdxList(cell_idx_list,&cell_count);
                if(pdctx->getUlCb(ul_cb))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                    ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CH_ERROR,cell_idx_list,cell_count, false);
                }
                //goto error_next;
            }
            else
                srs->setRunStatus(CH_RUN_DONE_NO_ERROR);
            // t3 = Time::nowNs();

            ti.add("Signal Completion");
            if(srs->signalRunCompletionEvent(srs->getStream(),false))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SRS signalRunCompletionEvent returned error");
                goto error_next;
            }
        }
        PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()
    }
    slot_map->timings.end_t_ul_srs_cuda[0] = Time::nowNs();
    POP_RANGE_PHYDRV

    ti.add("Signal Channel End Task");
    slot_map->addChannelEndTask();
    slot_map->addSlotEndTask();
    start_t_3 = Time::nowNs();

    ti.add("End Task");

    return 0;

    error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} line {}: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_srs aborted the tasklist for an error");
    return -1;
}


int task_work_function_ul_aggr_1_cplane(Worker* worker, void* param,int task_num,int first_cell, int num_tasks)
{
    char name[64];
    sprintf(name, "UL Task CPlane %d", task_num + 1);
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, name, 40);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    int                                                                          ret = 0, i = 0;
    Cell * cell_ptr = nullptr;
    t_ns                                                                         start_t_1,start_tx,t_start_ul_channel_tasks;
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    uint32_t cpu;
    uint8_t send_Cplane_error=SEND_CPLANE_NO_ERROR;
    int                                                                          sfn = 0, slot = 0;
    PhyPuschAggr* pusch= slot_map->aggr_pusch;
    PhyPucchAggr* pucch= slot_map->aggr_pucch;
    PhyPrachAggr* prach= slot_map->aggr_prach;
    PhySrsAggr* srs = slot_map->aggr_srs;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cplane_tx_err_cell_idx_list={};
    int cplane_tx_err_cell_count=0;

    start_t_1 = Time::nowNs();
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto error_next;
    }

    if(pusch||pucch||prach||srs) //Run ULC-PLane Tx only if one or more of these channels are enabled in the slot map
    {
        /////////////////////////////////////////////////////////////////////////////////////
        //// C-Plane
        /////////////////////////////////////////////////////////////////////////////////////
        if (!pdctx->isCPlaneDisabled())
        {
            PUSH_RANGE_PHYDRV("UL CPlane", 1);
            try
            {

                ti.add("Wait ULBFW Completion");
                t_start_ul_channel_tasks = slot_map->getTaskTsExec(1);
                int prevSlotUlBfwCompStatus = 1;
                if(pdctx->getmMIMO_enable() && task_num < slot_map->getNumCells())
                {
                    // Determine which slot to check for previous slot UL BFW completion
                    int previous_slot = (slot_oran_ind.osfid_*2 + slot_oran_ind.oslotid_ -1 +  SLOTS_PER_FRAME)%SLOTS_PER_FRAME;

                    //Perform wait based on first cell's T1aMaxCpUlNs
                    t_ns current_time = Time::nowNs();
                    cell_ptr = slot_map->aggr_cell_list[task_num];
                    int exec_slot_ahead = cell_ptr->getSlotAhead() - 1;
                    start_tx = slot_map->getTaskTsExec(0) + t_ns(Cell::getTtiNsFromMu(cell_ptr->getMu()) * exec_slot_ahead) - t_ns(cell_ptr->getT1aMaxCpUlNs());
                    t_ns deadline_time = start_tx - t_ns(pdctx->getSendCPlane_ulbfw_backoff_th_ns());
                    prevSlotUlBfwCompStatus = pdctx->queryUlBFWCompletion(previous_slot);
                    while(!prevSlotUlBfwCompStatus)
                    {
                        current_time = Time::nowNs();
                        if(current_time >= deadline_time)
                        {
                                // UL BFW pipeline is not completed before deadline
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: Previous slot ULBFW pipeline is not complete! oframe_id_ {} osfid_ {} oslotid_ {}",
                                                                            slot_oran_ind.oframe_id_, slot_oran_ind.osfid_, slot_oran_ind.oslotid_);
                                break;
                        }
                        prevSlotUlBfwCompStatus = pdctx->queryUlBFWCompletion(previous_slot);
                    }
                }
                
                ti.add("CPlane Prepare");
                for(int i = task_num; i < slot_map->getNumCells(); i += num_tasks)
                {
                    cell_ptr = slot_map->aggr_cell_list[i];
                    if(cell_ptr == nullptr) continue;

                    // The exec time is already 1 slot ahead (see cuphydriver_api.cpp) so account for that here.
                    int exec_slot_ahead = cell_ptr->getSlotAhead() - 1;
                    start_tx = slot_map->getTaskTsExec(0) + t_ns(Cell::getTtiNsFromMu(cell_ptr->getMu()) * exec_slot_ahead) - t_ns(cell_ptr->getT1aMaxCpUlNs());
                    slot_map->timings.start_t_ul_cplane[i] = Time::nowNs();
                    uint8_t frameStruct = 0;

                    PhyPrachAggr * prach = slot_map->aggr_prach;
                    if(prach != nullptr)
                    {
                        //Note: FrameStructure is the same for all the rach[]
                        slot_command_api::prach_params* prachparams = prach->getDynParams();
                        uint8_t oran_fft    = prachparams->nfft == 1536 ? 0x0D: __builtin_ctz(prachparams->nfft);
                        //NVLOGC_FMT(TAG, "FFT = {} , mu = {}", prachparams->nfft, prachparams->mu);
                        frameStruct = oran_fft << 4 | prachparams->mu;
                    }

                    uint8_t* bfw_header = nullptr;
                    uint8_t count = 0;
                  
                    // tis facilitates nested calls to add additional subtask TIs
                    ti_subtask_info tis{}; 
                    
                    // To ease any pressure in NIC w/ C-Plane BFW packets - the TX start time is offset by 
                    // (cell_ID * (TX_WINDOW) / NUM_CELLS). 
                    uint64_t tx_cell_start_ofs_ns = pdctx->getFhProxy()->getUlcBfwEnableDividePerCell() ? 
                                   i * ((cell_ptr->getT1aMaxCpUlNs() - cell_ptr->getT1aMinCpUlNs()) / slot_map->getNumCells()) :
                                   0; 
                    if(slot_map->aggr_slot_info[i])
                    {
                        if((ret=fhproxy->prepareCPlaneInfo(
                            cell_ptr->getIdx(),
                            cell_ptr->getRUType(),
                            cell_ptr->getPeerId(),
                            cell_ptr->getDLCompMeth(),
                            start_tx,
                            tx_cell_start_ofs_ns,
                            DIRECTION_UPLINK,
                            slot_oran_ind, //it_ulchannels.second.slot_oran_ind,
                            *(slot_map->aggr_slot_info[i]), //*slot_info,
                            cell_ptr->getSection3TimeOffset(),
                            slot_map->getDynBeamIdOffset(),
                            frameStruct,
                            0,&bfw_header,
                            t_start_ul_channel_tasks,
                            prevSlotUlBfwCompStatus,
                            tis))!=SEND_CPLANE_NO_ERROR)
                        {
                            cplane_tx_err_cell_idx_list[cplane_tx_err_cell_count++]=cell_ptr->getIdx();

                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL C-plane prepare error for cell index {},error type {} Map {} Abort UL Tasks!\n",i,ret,slot_map->getId());
                            send_Cplane_error|=ret; //OR the ret code in order to prevent overwriting of the errorneous ret state if any from previous interations of the loop
                            slot_map->atom_ul_cplane_info_for_uplane_rdy_count.fetch_add(1);
                            //TODO: we should really keep track of the following:
                            //1.) cplane failures on per cell basis - abort only cells that failed
                            //2.) partial cplane send - right now we only deal with complete or partial failures
                            //How does order kernel deal with these additional edge cases? (Is data thrown away during another slot?)

                            //Leaving original dispatch for reference
                            // if(ret==SEND_CPLANE_TIMING_ERROR_NO_ABORT_TASKS)
                            // {
                            //     NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL C-plane send timing error for cell index {} error type {} Map {},Error detection too late. No UL Task Abort!",i,ret,slot_map->getId());
                            //     send_Cplane_error=ret;
                            // }
                            // else
                            // {
                            //     NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL C-plane send error for cell index {},error type {} Map {} Abort UL Tasks!\n",i,ret,slot_map->getId());
                            //     send_Cplane_error=ret;
                            //     goto error_next;
                            // }
                        }

                        if (ret == SEND_CPLANE_NO_ERROR && pdctx->getmMIMO_enable()) 
                        {
                            // First, prepare and enqueue the BFW packets into the NIC as they have a stricter deadline. 
                            int ret1 = fhproxy->sendCPlaneMMIMO (true /* isBFW */, cell_ptr->getIdx(), cell_ptr->getPeerId(), DIRECTION_UPLINK, tis);  
                            // Next, prepare and enqueue the non-BFW packets into the NIC. 
                            int ret2 = fhproxy->sendCPlaneMMIMO (false /* isBFW */, cell_ptr->getIdx(), cell_ptr->getPeerId(), DIRECTION_UPLINK, tis);  
                            if (ret1 != SEND_CPLANE_NO_ERROR || ret2 != SEND_CPLANE_NO_ERROR) 
                            {
                                cplane_tx_err_cell_idx_list[cplane_tx_err_cell_count++]=cell_ptr->getIdx();
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL C-plane send error for cell index {},error type BFW:{} nonBFW:{} Map {} Abort UL Tasks!\n",i,ret1,ret2,slot_map->getId());
                            }
                            slot_map->aggr_slot_info[i]->section_id_ready.store(true);
                            slot_map->atom_ul_cplane_info_for_uplane_rdy_count.fetch_add(1);
                        }
                        // Appends any subtask instrumentation added by the sendCPlane call.
                        ti.appendList(tis); 
                    }
                    if(pdctx->getUlCb(ul_cb))
                    {
                        if(bfw_header != nullptr)
                        {
                            ul_cb.fh_bfw_coeff_usage_done_fn(ul_cb.fh_bfw_coeff_usage_done_fn_context, bfw_header);
                        }
                    }
                    slot_map->timings.end_t_ul_cplane[i] = Time::nowNs();
                }
            }
            PHYDRIVER_CATCH_EXCEPTIONS();
            POP_RANGE_PHYDRV
        } // if (!pdctx->isCPlaneDisabled)
    }

    ti.add("Cplane error check");
    if(send_Cplane_error) {
        //Abort the rest of the pipeline on any cplane error

        if(pdctx->getUlCb(ul_cb))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
            ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_ind,SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CPLANE_TX_ERROR,cplane_tx_err_cell_idx_list,cplane_tx_err_cell_count, false);
        }
        slot_map->abortTasks();
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_cplane {} aborted the tasklist for an error", task_num);
    }

    //Increment completion counters
    ti.add("Signal completion");
    slot_map->addULCTasksComplete();
    slot_map->addSlotEndTask();

    ti.add("End Task");

    return 0;
    error_next:
        ////////////////////////////////////////////////////////////////////////
        ///// Currently we do not support pipeline recovery from CUDA/FH errors
        ////////////////////////////////////////////////////////////////////////
        NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "%s line %d: pipeline failed, exit", __func__, __LINE__);
        EXIT_L1(EXIT_FAILURE);

        return 0;
}

int task_work_function_ul_aggr_3_early_uci_ind(Worker* worker, void* param,int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task AGGR3 Early UCI IND", 10);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    t_ns                                                                         start_t;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    uint32_t cpu;
    int                                                                          sfn = 0, slot = 0,ret;
    PhyPuschAggr* pusch= slot_map->aggr_pusch;
    PhyPucchAggr* pucch= slot_map->aggr_pucch;
    struct slot_command_api::ul_slot_callbacks ul_cb;
    uint8_t isEarlyUciDetComplete=0;
    t_ns timeout_thresh_t(5 * NS_X_MS);
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());

    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto error_next;
    }

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3_early_uci_ind timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        slot_map->setEarlyUciEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3_early_uci_ind Task aborted for Slot Map {}",slot_map->getId());

        slot_map->setEarlyUciEndTask();
        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    if(slot_map->getIsEarlyHarqPresent()==1) //Task should immediately return if Early Harq is not present
    {
        //Note: currently this is needed as otherwise PUCCH seems to wait for PUSCH completion
        ti.add("PUCCH Wait");
        if(pucch != nullptr) {
            bool pucch_finished;
            bool pucch_timeout;
            start_t = Time::nowNs();
            do {
                pucch_finished = (pucch->waitRunCompletionEventNonBlocking()==1);
                pucch_timeout = (Time::nowNs() - start_t > timeout_thresh_t);
            } while(!pucch_finished && !pucch_timeout);
            if(pucch_timeout) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 Early UCI IND task waiting for PUCCH run completion event more than {} ns for Slot Map {}",
                        timeout_thresh_t.count(),slot_map->getId());
                    goto error_next;
            }
        }

        ti.add("UCI Det Completion WAIT");
        start_t = Time::nowNs();
        while(!isEarlyUciDetComplete)
        {
            isEarlyUciDetComplete=pusch->waitEventNonBlocking(pusch->getPuschStatParams()->subSlotCompletedEvent);
            if(isEarlyUciDetComplete)
            {
                if(pusch->getPreEarlyHarqWaitKernelStatus()==PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT)
                {
                    NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} PUSCH Pre Early Harq Wait kernel timeout!",sfn,slot,slot_map->getId());
                }
                NVLOGI_FMT(TAG,"Triggering Early UCI Indication Callback to L2A for Slot Map {}",slot_map->getId());
                if(pdctx->getUlCb(ul_cb))
                {
                    ti.add("Early UCI Callback");
                    ul_cb.callback_fn_early_uci(ul_cb.callback_fn_early_uci_context, slot_map->getSlot3GPP(),*pusch->getDynParams(),pusch->getPuschDynParams()->pDataOut,pusch->getPuschStatParams(), slot_map->get_t0());

                    ti.add("Run PUSCH_RUN_FULL_SLOT_COPY");
                    //PUSCH_RUN_FULL_SLOT_COPY: D2H copies all PUSCH results from GPU to CPU
                    //                          Note - runs on phase2_stream
                    if(pusch->run(cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH run phase PUSCH_RUN_FULL_SLOT_COPY returned error");
                        goto error_next;
                    }

                    cudaStream_t* streams = pdctx->getUlOrderStreamsPusch();
                    cudaStream_t phase2_stream;
                    if(pdctx->splitUlCudaStreamsEnabled()) //Revisit for TDD pattern, Assume DDDSUUDDDD pattern
                    {
                        phase2_stream = ((slot % 10 == 3) || (slot % 10 == 4))? streams[PHASE2_SPLIT_STREAM1] : streams[PHASE2_SPLIT_STREAM2];
                    }
                    else
                    {
                        phase2_stream = streams[PHASE2_SPLIT_STREAM1];
                    }

                    ti.add("Signal Completion");
                    if(pusch->signalRunCompletionEvent(phase2_stream,false))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PUSCH signalRunCompletionEvent returned error");
                        goto error_next;
                    }
                }
            }
            else if(Time::nowNs() - start_t > timeout_thresh_t)
            {
                //Set the work cancel flag to attempt pusch pipeline termination if early work termination is enabled.
                if(pusch->getSetupStatus() == CH_SETUP_DONE_NO_ERROR && pusch->getRunStatus() == CH_RUN_DONE_NO_ERROR)
                {
                    pusch->setWorkCancelFlag();
                }
                
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 Early UCI IND task waiting for UCI Detected event more than {} ns for Slot Map {}, attempting PUSCH pipeline termination only if no PUSCH setup or run errors",
                       timeout_thresh_t.count(),slot_map->getId());
                goto error_next;
            }
        }
    }

    slot_map->setEarlyUciEndTask();
    slot_map->addSlotEndTask();

    ti.add("End Task");

    return 0;
   error_next:
    return -1;
}

int task_work_function_ul_aggr_1_orderKernel(Worker* worker, void* param, int first_cell, int ok_task_num,int isSrs)
{
    char name[64];
    sprintf(name, "UL Task Order Kernel %d", ok_task_num);    
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, name, 10);
    ti.add("Start Task");
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    int                                                                          task_num = 1, ret = 0, i = 0;
    uint16_t                                                                     numPrbPuschPucch[UL_MAX_CELLS_PER_SLOT] = {0}, numPrbPrach[UL_MAX_CELLS_PER_SLOT] = {0};
    uint32_t                                                                     numPrbSrs[UL_MAX_CELLS_PER_SLOT] = {0};
    uint8_t *buf_st_1[UL_MAX_CELLS_PER_SLOT]={nullptr},*buf_st_2[UL_MAX_CELLS_PER_SLOT]={nullptr}, *buf_st_3_o0[UL_MAX_CELLS_PER_SLOT]={nullptr},*buf_st_3_o1[UL_MAX_CELLS_PER_SLOT]={nullptr},*buf_st_3_o2[UL_MAX_CELLS_PER_SLOT]={nullptr},*buf_st_3_o3[UL_MAX_CELLS_PER_SLOT]={nullptr};
    uint8_t *buf_pcap_capture[UL_MAX_CELLS_PER_SLOT]={nullptr};
    uint8_t *buf_pcap_capture_ts[UL_MAX_CELLS_PER_SLOT]={nullptr};
    std::array<t_ns,UL_MAX_CELLS_PER_SLOT> slot_start;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> ta4_min_ns,ta4_max_ns;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> ta4_min_ns_srs,ta4_max_ns_srs;
    uint8_t num_order_cells=0;
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    t_ns                                                                         start_t_1, start_t_3;
    int                                                                          sfn = 0, slot = 0;
    int num_prach_occasions = 0;
    int start_ro = 0;
    struct slot_command_api::slot_indication slot_ind = slot_map->getSlot3GPP();
    struct slot_command_api::oran_slot_ind slot_oran_ind = slot_command_api::to_oran_slot_format(slot_ind);
    Cell * cell_ptr = nullptr;
    OrderEntity * oentity = nullptr;
    ULInputBuffer * ulbuf_st1 = nullptr;
    ULInputBuffer * ulbuf_st2 = nullptr;
    ULInputBuffer * ulbuf_pcap_capture = nullptr;
    ULInputBuffer * ulbuf_pcap_capture_ts = nullptr;
    std::vector<ULInputBuffer*>& ulbuf_st3_v = slot_map->aggr_ulbuf_st3;
    ULInputBuffer * prach_ul_buf[PRACH_MAX_OCCASIONS];
    slot_command_api::slot_info_t * slot_info = nullptr;
    slot_command_api::srs_params* srs_pparms = nullptr;
    PhyPuschAggr* pusch= slot_map->aggr_pusch;
    PhyPucchAggr* pucch= slot_map->aggr_pucch;
    PhyPrachAggr* prach= slot_map->aggr_prach;
    PhySrsAggr* srs = slot_map->aggr_srs;
    std::array<uint8_t,UL_MAX_CELLS_PER_SLOT> srs_start_symbol;
    t_ns t1, t2, t3,t_start_ul_channel_tasks;
    uint32_t cpu;
    uint32_t srsMask = 0;
    uint32_t nonSrsUlMask = 0;
    uint8_t pusch_prb_non_zero=0;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT> pusch_prb_symbol_map;
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT> pucch_prb_symbol_map;
    std::array<uint32_t,ORAN_PUSCH_SYMBOLS_X_SLOT> num_order_cells_sym_mask_arr;
    std::fill(pusch_prb_symbol_map.begin(),pusch_prb_symbol_map.end(),0);
    std::fill(pucch_prb_symbol_map.begin(),pucch_prb_symbol_map.end(),0);
    std::fill(num_order_cells_sym_mask_arr.begin(),num_order_cells_sym_mask_arr.end(),0);

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_orderKernel timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());

        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_1_orderKernel Task aborted for Slot Map {}",slot_map->getId());

        slot_map->addSlotEndTask();

        ti.add("End Task");

        return 0;
    }

    start_t_1 = Time::nowNs();

    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto error_next;
    }

    //NVLOGI_FMT(TAG,"UL Task 1(Order kernel) started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);

    //NVLOGC_FMT(TAG, "Size of occasion array = {}", slot_map->num_prach_occa.size());
    if(pusch||pucch||prach||srs) //Run Order kernel only if one or more of these channels are enabled in the slot map
    {
#ifdef CUPHY_PTI_ENABLE_TRACING
        if (0) if (((sfn % 128) == 64) && (slot == 4))
        {
            pdctx->getUlCtx()->setCtx();
            cuphy_pti_calibrate_gpu_timer(pdctx->get_stream_timing_ul());
        }
#endif

        ti.add("Run Order");
        if(srs)
        {
            srs_pparms = srs->getDynParams();
        }
        for(int i = first_cell; i < slot_map->getNumCells(); i++)
        {
            cell_ptr    = slot_map->aggr_cell_list[i];
            ulbuf_st1   = slot_map->aggr_ulbuf_st1[i];
            ulbuf_st2   = slot_map->aggr_ulbuf_st2[i];
            ulbuf_pcap_capture   = slot_map->aggr_ulbuf_pcap_capture[i];
            ulbuf_pcap_capture_ts   = slot_map->aggr_ulbuf_pcap_capture_ts[i];
            buf_st_1[i]=(ulbuf_st1 == nullptr? nullptr : ulbuf_st1->getBufD());
            buf_st_2[i] = (ulbuf_st2 == nullptr? nullptr : ulbuf_st2->getBufD());
            // buf_pcap_capture[i] = (ulbuf_pcap_capture == nullptr? nullptr : ulbuf_pcap_capture->getBufD());
            buf_pcap_capture[i] = (ulbuf_pcap_capture == nullptr? nullptr : ulbuf_pcap_capture->getBufH());
            buf_pcap_capture_ts[i] = (ulbuf_pcap_capture_ts == nullptr? nullptr : ulbuf_pcap_capture_ts->getBufH());
            //(slot_map->num_prach_occa.size() > i ) ? num_prach_occasions = slot_map->num_prach_occa[i] : 0;
            num_prach_occasions = slot_map->num_prach_occa[i];
            if(ulbuf_st2 != nullptr && cell_ptr->getRUType() != SINGLE_SECT_MODE) {
                if(isSrs==1)
                {
                    srsMask |= (1 << i);
                }
            }
            if(ulbuf_st1 != nullptr) {
                if(isSrs==0)
                {
                    nonSrsUlMask |= (1 << i);
                }
            }
            if(srs_pparms != nullptr)
            {
                srs_start_symbol[i]=srs_pparms->cell_dyn_info[i].srsStartSym;
            }

            if(cell_ptr == nullptr || (ulbuf_st1 == nullptr &&  ulbuf_st2 == nullptr && ulbuf_st3_v.size() == 0))
                continue;

            //NVLOGC_FMT(TAG, "Number of occasions = {}", num_prach_occasions);
            for(int idx = 0; idx < PRACH_MAX_OCCASIONS; idx++)
            {
                if(idx < num_prach_occasions)
                {
                    prach_ul_buf[idx] = ulbuf_st3_v[idx + start_ro];
                    if(prach_ul_buf[idx] != nullptr)
                        nonSrsUlMask |= (1 << i);
                    //NVLOGC_FMT(TAG, "Addr {} = {}", idx, prach_ul_buf[idx]);
                }
                else
                    prach_ul_buf[idx] = nullptr;
            }
            start_ro += num_prach_occasions;

            buf_st_3_o0[i]=(prach_ul_buf[0] == nullptr? nullptr : prach_ul_buf[0]->getBufD());
            buf_st_3_o1[i]=(prach_ul_buf[1] == nullptr? nullptr : prach_ul_buf[1]->getBufD());
            buf_st_3_o2[i]=(prach_ul_buf[2] == nullptr? nullptr : prach_ul_buf[2]->getBufD());
            buf_st_3_o3[i]=(prach_ul_buf[3] == nullptr? nullptr : prach_ul_buf[3]->getBufD());

            if(slot_map->aggr_slot_info[i])
            {
                numPrbPuschPucch[i] = fhproxy->countPuschPucchPrbs(*(slot_map->aggr_slot_info[i]), cell_ptr->geteAxCNumPusch(), cell_ptr->geteAxCNumPucch(),&pusch_prb_symbol_map[i*ORAN_PUSCH_SYMBOLS_X_SLOT],&pucch_prb_symbol_map[i*ORAN_PUSCH_SYMBOLS_X_SLOT],&num_order_cells_sym_mask_arr[0],i,pusch_prb_non_zero,param);
                numPrbPrach[i] = fhproxy->countPrachPrbs(*(slot_map->aggr_slot_info[i]), cell_ptr->geteAxCNumPrach(), param);
                numPrbSrs[i] = fhproxy->countSrsPrbs(*(slot_map->aggr_slot_info[i]), cell_ptr->geteAxCNumSrs());
                if(cell_ptr->getRUType() == SINGLE_SECT_MODE) {
                    numPrbPuschPucch[i] -= numPrbSrs[i]; // We can't know ahead of time how many SRS prbs there will be, so we set PUSCH to be all of them, then subtract the SRS prbs from the PUSCH prbs
                    //Handles use case where slot contains PUCCH PDU followed by PUSCH PDU. In this case pusch_prb_non_zero will be 0 as PUSCH symbol map is empty
                    //so to handle this use case, we need to set pusch_prb_non_zero to 1 if there are PUSCH PRBs in the slot
                    if(numPrbPuschPucch[i] > 0 && pusch != nullptr)
                    {
                        pusch_prb_non_zero = 1;
                    }
                }

                NVLOGI_FMT(TAG, "SFN {}.{} Task UL {} Map {} Cell {} requesting {} PUSCH/PUCCH {} PRACH {} SRS PRBs", sfn, slot, task_num,slot_map->getId(),cell_ptr->getPhyId(),numPrbPuschPucch[i], numPrbPrach[i],numPrbSrs[i]);
            }
            slot_start[i]=slot_map->getTaskTsExec(0) + t_ns(AppConfig::getInstance().getTaiOffset()) + t_ns(Cell::getTtiNsFromMu(cell_ptr->getMu()) * (cell_ptr->getSlotAhead()-1)); //-1 since slot_map->getTaskTsExec(0) is already at L2A slot tick + 1
            ta4_min_ns[i]=cell_ptr->getTa4MinNs();
            ta4_max_ns[i]=cell_ptr->getTa4MaxNs();
            ta4_min_ns_srs[i]=cell_ptr->getTa4MinNsSrs();
            ta4_max_ns_srs[i]=cell_ptr->getTa4MaxNsSrs();
            num_order_cells++;
            if(pdctx->ru_health_check_enabled() && (cell_ptr->isHealthy() == false) && (nonSrsUlMask != 0))
            {
                cell_ptr->num_consecutive_unhealthy_slots++;
                NVLOGI_FMT(TAG,"SFN {}.{} Task UL {} Map {} Cell {} is unhealthy. Order kernel processing will be skipped",sfn, slot, task_num,slot_map->getId(),cell_ptr->getPhyId());
            }
            if(pdctx->ru_health_check_enabled() && (cell_ptr->isHealthySrs() == false) && (srsMask != 0))
            {
                cell_ptr->num_consecutive_unhealthy_slots_srs++;
                NVLOGI_FMT(TAG,"SFN {}.{} Task UL {} Map {} Cell {} is unhealthy. SRS Order kernel processing will be skipped",sfn, slot, task_num,slot_map->getId(),cell_ptr->getPhyId());
            }            
        }
        NVLOGD_FMT(TAG,"Slot Map ({}), num_ordered_cells({})",slot_map->getId(),num_order_cells);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////// Order Kernel
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        oentity     = slot_map->aggr_order_entity;
        slot_map->timings.start_t_ul_order_cuda = Time::nowNs();
        oentity->runOrder(slot_oran_ind,
                        numPrbPuschPucch, //FIXME: this should be numPrbPusch + numPrbPucch
                        buf_st_1,
                        buf_pcap_capture,
                        buf_pcap_capture_ts,
                        numPrbPrach,
                        buf_st_3_o0,
                        buf_st_3_o1,
                        buf_st_3_o2,
                        buf_st_3_o3,
                        pdctx->getStartSectionIdPrach(),
                        pdctx->getStartSectionIdPrach()+1,
                        pdctx->getStartSectionIdPrach()+2,
                        pdctx->getStartSectionIdPrach()+3,
                        numPrbSrs,
                        buf_st_2,
                        slot_start,
                        ta4_min_ns,
                        ta4_max_ns,
                        ta4_min_ns_srs,
                        ta4_max_ns_srs,
                        num_order_cells,
                        srsMask,
                        srs_start_symbol,
                        nonSrsUlMask,
                        ((pusch==nullptr)?NULL:pusch->getSymOrderSigDoneGpuFlag(0)),
                        pusch_prb_symbol_map,
                        num_order_cells_sym_mask_arr,pusch_prb_non_zero,(uint32_t)slot_map->getId()
        );
        slot_map->timings.end_t_ul_order_cuda = Time::nowNs();
    }

    ti.add("Unlock Next Task");
    if(slot_map->unlockNextTask(task_num, slot_map->getNumCells()) == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task {} can't activate the next one", task_num);
        goto error_next;
    }

    slot_map->addSlotEndTask();

    start_t_3 = Time::nowNs();

    ti.add("End Task");

    return 0;

error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "%s line %d: pipeline failed, exit", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_1_orderKernel aborted the tasklist for an error");
    return -1;
}

int task_work_function_ul_aggr_2(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task UL AGGR 2", 10);
    ti.add("Start Task");
    int                                                                          task_num = 2, ret_task = 0, ret_order = 0, completed_cells = 0;
    int                                                                          ret_rx = 0;
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    int                                                                          printed = 0, totMsg = 0, num_msgs = 0; //, umsg_index=0;
    t_ns                                                                         start_t_1, start_t_2, start_t_3, start_free_t, start_rx;
    t_ms                                                                         thresholdms_t(1);
    std::array<enum ul_status, UL_MAX_CELLS_PER_SLOT>                            ul_status_index = {UL_INIT};
    t_ns                                                                         ul_start_t[UL_MAX_CELLS_PER_SLOT]={};
    int                                                                          sfn = 0, slot = 0, ret = 0;
    FhProxy*                                                                     fhproxy = pdctx->getFhProxy();
    PhyPuschAggr*                                                                pusch = slot_map->aggr_pusch;
    PhyPucchAggr*                                                                pucch = slot_map->aggr_pucch;
    PhyPrachAggr*                                                                prach = slot_map->aggr_prach;
    OrderEntity*                                                                 oentity  = slot_map->aggr_order_entity;
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());

    /////////////////////////////////////////////////////////////////////////////////////
    //// Wait completion previous task
    /////////////////////////////////////////////////////////////////////////////////////
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE(TAG, "getcpu failed for %s\n", __FUNCTION__);
        goto error_next;
    }
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;

    //NVLOGI(TAG,"UL Task 2 started on CPU %u for Map %d, SFN slot (%d,%d)\n", cpu, slot_map->getId(), sfn, slot);
    start_t_1 = Time::nowNs();

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_2 timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());
        goto error_next;
    }
    
    ti.add("Wait Current Task");
    while(1)
    {

        //Check abort condition and whether we can move forward in the task
        ret_task = slot_map->checkCurrentTask(task_num, slot_map->getNumCells());
        if(ret_task == -1)
        {
            NVLOGI(TAG, "Task UL %d Map %u got SlotMapUl abort notification\n", task_num, slot_map->getId());
            goto error_next;
        }
        else if(ret_task == 1)
            break;
        else
        {
            if(printed == 0 && Time::getDifferenceNowToNs(start_t_1) > thresholdms_t)
            { //1ms
                NVLOGI(TAG, "Previous task %d Map %u is taking more than %u ms\n", task_num - 1 ,slot_map->getId(), thresholdms_t.count());
                printed = 1;
            }
        }
    }

    start_t_2 = Time::nowNs();

    ti.add("RX Data");
    PUSH_RANGE_PHYDRV("UL RX", 1);
    // while(completed_cells < slot_map->getNumCells()) // Add time constraints? E.g. // Time::getDifferenceNsToNow(slot_map->getTaskTsExec(task_num)) < t_ns(Cell::getTtiNsFromMu(MU_SUPPORTED))
    while(completed_cells < num_cells)
    {
        for(int i = first_cell; i < first_cell + num_cells && i < slot_map->getNumCells(); i++)
        {
            Cell*     cell_ptr    = slot_map->aggr_cell_list[i];           

            ////////////////////////////////////////////////////////////////////////////////////////
            //// FIXME: Temporary condition in case of PUCCH only
            ////////////////////////////////////////////////////////////////////////////////////////
            if(pucch == nullptr && pusch == nullptr && prach == nullptr)
            {
                ul_status_index[i] = UL_ORDERED;
                completed_cells++;
                continue;
            }
            ////////////////////////////////////////////////////////////////////////////////////////

            /*
            * No need to RX packets from cells connected to this PUSCH object because
            * the Order Kernel has already completed
            */
            if(ul_status_index[i] == UL_ORDERED)
            {
                // NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL2 skip cell %d\n", cell_ptr->getPhyId());
                continue;
            }

            /*
             * Intialization: lock RX queues
             * Cell will be immediately used in the switch
             */
            if(ul_status_index[i] == UL_INIT)
            {
                cell_ptr->lockRxQueue();
                ul_status_index[i] = UL_SETUP;
                ul_start_t[i] = Time::nowNs();
            }

            /* Setup: Wait to start Order Kernel */
            if(ul_status_index[i] == UL_SETUP)
            {
                start_rx = Time::nowNs();
                ret_rx = fhproxy->UserPlaneReceivePackets(cell_ptr->getPeerId());
                if(ret_rx < 0)
                {
                    NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task UL %d Map %lu cell %u rxUplane failed -1",task_num,slot_map->getId(),cell_ptr->getPhyId());
                    goto error_next;
                }

                /* Process U-plane packet for PUSCH and/or PRACH */
                if(ret_rx > 0 || fhproxy->UserPlaneCheckMsg(cell_ptr->getPeerId()))
                {
                    // Enable Order Kernel
                    oentity->enableOrder(i,ORDER_KERNEL_START);

                    // Update timers
                    slot_map->timings.start_t_ul_order   = Time::nowNs();
                    slot_map->timings.start_t_ul_rx_pkts[i] = start_rx;
                    slot_map->timings.end_t_ul_rx_pkts[i]   = Time::nowNs();
                    NVLOGI(TAG, "SFN %d.%d Task UL %d Map %lu Cell %u received %d packets during UL_SETUP state\n",sfn,slot,task_num,slot_map->getId(),cell_ptr->getPhyId(),ret_rx);
                    
                    // Move forward
                    ul_status_index[i]              = UL_START;
                    continue;
                }
                /* CPU timeout per cell: Prevent any deadlock if no U-plane packet arrives */
                else if ((Time::nowNs() - ul_start_t[i]).count() > pdctx->getUlOrderTimeoutCPU())
                {
                    // Abort Order Kernel
                    oentity->enableOrder(i,ORDER_KERNEL_ABORT);

                    // Update timers
                    slot_map->timings.start_t_ul_order   = Time::nowNs();
                    slot_map->timings.start_t_ul_rx_pkts[i] = start_rx;
                    slot_map->timings.end_t_ul_rx_pkts[i]   = Time::nowNs();

                    // Move forward
                    ul_status_index[i]              = UL_START;

                    NVLOGI(TAG, "SFN %d.%d Task UL %d Map %lu Cell %u is not receiving U-plane. Abort Order kernel\n",sfn,slot,task_num,slot_map->getId(),cell_ptr->getPhyId());
                    continue;
                }
            }

            if(ul_status_index[i] == UL_START)
            {
                /* Check Order Kernel completion */
                ret_order = oentity->checkOrderCPU(false);

                /* Order Kernel IS completed. RX other packets */
                if(ret_order == 1)
                {
                    slot_map->timings.end_t_ul_order     = Time::nowNs();

                    // Free Umsgs
                    slot_map->timings.start_t_ul_freemsg[i] = slot_map->timings.end_t_ul_order;
                    // cell_ptr->freeUmsg();
                    fhproxy->UserPlaneFreeMsg(cell_ptr->getPeerId());
                    slot_map->timings.end_t_ul_freemsg[i]   = Time::nowNs();

                    // Move forward
                    ul_status_index[i] = UL_ORDERED;
                    completed_cells++;

                    cell_ptr->unlockRxQueue();                    
                    NVLOGI(TAG, "SFN %d.%d Task UL %d Map %lu Cell %u transitioned to UL_ORDERED state\n",sfn,slot,task_num,slot_map->getId(),cell_ptr->getPhyId());

                    // NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "ORDER %ld completed at %ld\n", oentity->getId(), Time::nowNs().count());
                    // NVSLOGI(TAG)     << "Task UL " << task_num << " Map " << slot_map->getId() << " ORDER " << oentity->getId() << " COMPLETED cell " << cell_ptr->getPhyId();
                                    // << " free time is " << Time::NsToUs(slot_map->timings.end_t_ul_freemsg[i] - slot_map->timings.start_t_ul_freemsg[i]).count() << " us";

                    continue;
                }
                /* Order Kernel IS NOT completed. RX other packets */
                else if(ret_order == 0)
                {
                    ret_rx = fhproxy->UserPlaneReceivePackets(cell_ptr->getPeerId());
                    if(ret_rx < 0)
                    {
                        NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task UL %d Map %lu cell %u rxUplane failed -1",task_num,slot_map->getId(),cell_ptr->getPhyId());
                        goto error_next;
                    }

                    if(ret_rx > 0)
                    {
                        NVLOGI(TAG, "SFN %d.%d Task UL %d Map %lu Cell %u received %d packets during UL_START state\n",sfn,slot,task_num,slot_map->getId(),cell_ptr->getPhyId(),ret_rx);
                        slot_map->timings.end_t_ul_rx_pkts[i] = Time::nowNs();
                        //F13 use-case requires cleanup during the slot
                        // cell_ptr->freeUmsg();
                        fhproxy->UserPlaneFreeMsg(cell_ptr->getPeerId());
                    }
                }
                /* Internal error */
                else
                {
                    NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, " Map %lu cell %u returned checkOrder %d\n",slot_map->getId(),cell_ptr->getPhyId(),ret_order);
                    goto error_next;
                }
            }
        }
    }
    POP_RANGE_PHYDRV

    slot_map->addSlotEndTask();

    start_t_3 = Time::nowNs();

    ti.add("Unlock Next Task");
    if(slot_map->unlockNextTask(task_num, num_cells) == false)
    {
        NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task %d can't activate the next one\n", task_num );
        return -1;
    }

    ti.add("End Task");

    return 0;

error_next:
    ////////////////////////////////////////////////////////////////////////
    ///// Currently we do not support pipeline recovery from CUDA/FH errors
    ////////////////////////////////////////////////////////////////////////
    NVLOGF(TAG, AERIAL_CUPHYDRV_API_EVENT, "%s line %d: pipeline failed, exit\n", __func__, __LINE__);
    EXIT_L1(EXIT_FAILURE);

    slot_map->abortTasks();
    NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task ul_aggr_2 aborted the tasklist for an error\n");
    return -1;
}


int task_work_function_ul_aggr_3(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task UL AGGR 3", 34);
    ti.add("Start Task");


    int                                                                          task_num = 3, ret_task = 0;
    SlotMapUl*                                                                     slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    PhyPuschAggr * pusch = nullptr;
    PhyPucchAggr * pucch = nullptr;
    PhyPrachAggr * prach = nullptr;
    PhyUlBfwAggr * ulbfw = nullptr;
    PhySrsAggr * srs = nullptr;
    bool pusch_early_exit = false;

    pusch = slot_map->aggr_pusch;
    pucch = slot_map->aggr_pucch;
    prach = slot_map->aggr_prach;
    ulbfw = slot_map->aggr_ulbfw;
    srs = slot_map->aggr_srs;
    bool isSrsOnly = ((srs != nullptr) && (pusch==nullptr) && (pucch==nullptr) && (prach==nullptr));

    std::vector<ULInputBuffer*>& ulbuf_st3_v = slot_map->aggr_ulbuf_st3;
    ULInputBuffer * ulbuf_st1 = nullptr;
    ULInputBuffer * ulbuf_st2 = nullptr;

    PhyWaiter pusch_waiter(pusch);
    PhyWaiter pucch_waiter(pucch);
    PhyWaiter prach_waiter(prach);
    PhyWaiter srs_waiter(srs);

    NVLOGD_FMT(TAG,"{}:Slot ID {} pusch {} pucch {} prach {} srs {}",__func__,slot_map->getId(),(void*)pusch,(void*)pucch,(void*)prach,(void*)srs);

    bool early_uci_task_complete = (slot_map->getIsEarlyHarqPresent()!=1);//Invoke the wait (set to false) only if Early Harq is Present
    bool gpu_early_harq_timeout = false;
    bool still_waiting;
    uint8_t isWaitFullSlotComplete=0;

    t_ns timeout_thresh_t(10*NS_X_MS); //10ms wait on the UL cuPHY pipelines
    t_ns order_wait_thresh_t(2*NS_X_MS); //1ms wait on the order kernel
    t_ns start_t;
    int                                                                          printed  = 0;
    t_ms                                                                         thresholdms_t(1);
    int                                                                          sfn = 0, slot = 0, ret = 0;
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());
    int num_ul_tasks_to_wait;
    bool en_orderKernel_tb=pdctx->enableOKTb();
    int srs_task_offset = 0;

    OrderEntity* oentity = nullptr;
    if(pusch||pucch||prach||(srs && pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)) {
        oentity = slot_map->aggr_order_entity;
    }
    OrderWaiter order_waiter(oentity);

    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    std::array<uint8_t,UL_MAX_CELLS_PER_SLOT> cell_timeout_list={};
    std::array<bool,UL_MAX_CELLS_PER_SLOT> srs_order_cell_timeout_list = {false};
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_error_list={};
    uint32_t cell_error_count=0;

    int cell_count=0;
    int num_non_ulc_tasks_to_ignore = 0;
    struct slot_command_api::ul_slot_callbacks ul_cb;

    /*
     * Wait completion of Task 2 before starting with Task 3
     */
    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;
    bool log_once = false;
    bool pusch_reorder_kernel_timeout_state = false;
    bool pusch_terminating_state = false;
    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        ret_task=-1;
        goto cleanup;
    }
    NVLOGD_FMT(TAG,"UL Task 3 started on CPU {} for Map {}, SFN slot ({},{})", cpu, slot_map->getId(), sfn, slot);
    std::fill(cell_timeout_list.begin(),cell_timeout_list.end(),ORDER_KERNEL_EXIT_PRB);

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0)
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3 timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());
        ret_task=-1;
        goto cleanup;//Note: we still need to release slot map
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted())
    {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3 Task aborted for Slot Map {}",slot_map->getId());
        ret_task=-1;
        goto cleanup;//Note: we still need to release slot map
    }

    if(pdctx->cpuCommEnabled())
    {
        order_waiter.setState(WAIT_STATE_COMPLETED);
    }

    if(pdctx->get_ru_type_for_srs_proc() != SINGLE_SECT_MODE)
    {
        srs_task_offset = (srs?1:0);
    }

    if(pdctx->cpuCommEnabled())
    {
        if(pusch)
            num_non_ulc_tasks_to_ignore = 4 + srs_task_offset;
        else
            num_non_ulc_tasks_to_ignore = 3 + srs_task_offset;
    }
    else
    {
        if(pusch)
            num_non_ulc_tasks_to_ignore = 3 + srs_task_offset;
        else
            num_non_ulc_tasks_to_ignore = 2 + srs_task_offset;
    }

    if(ulbfw)
    {
        num_non_ulc_tasks_to_ignore += 1;
    }

    num_ul_tasks_to_wait = num_ul_tasks - num_non_ulc_tasks_to_ignore - num_ulc_tasks;

    if(srs && pdctx->get_ru_type_for_srs_proc() != SINGLE_SECT_MODE)
    {
        if(!isSrsOnly)
        {
            num_ul_tasks_to_wait -= 1;
        }
    }

    if(pdctx->gpuCommEnabledViaCpu())
    {
        order_wait_thresh_t = t_ns(6*NS_X_MS);
    }
    
    if(!en_orderKernel_tb)
    {
        /*The channel end waits are now needed to ensure cuda event recording for channel run completion is called before cudaEventSynchronize*/
        ti.add("Wait Channel End Task");
        if(slot_map->waitChannelEndTask(num_ul_tasks_to_wait) < 0) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitChannelEndTask returned error for Slot Map {} num_ul_tasks={} num_ulc_tasks={} num_ul_tasks_to_wait={}",slot_map->getId(),num_ul_tasks,num_ulc_tasks,num_ul_tasks_to_wait);
            ret_task=-1;
            goto cleanup;
        }
    }

    ti.add("Wait Loop");
    start_t = Time::nowNs();
    slot_map->timings.start_t_ul_pusch_compl[0] = Time::nowNs();
    slot_map->timings.start_t_ul_pucch_compl[0] = Time::nowNs();
    slot_map->timings.start_t_ul_prach_compl[0] = Time::nowNs();
    slot_map->timings.start_t_ul_srs_compl[0] = Time::nowNs();
    do {
        wait_action wa;

        if(!(pdctx->cpuCommEnabled()) && (pusch||pucch||prach))
        {
            //ORDER Wait Processing
            wa = order_waiter.checkAction(false);
            bool ok_timeout = false;
            if(wa == WAIT_ACTION_COMPLETED) {

                ti.add("Wait Order");
                
                // Add Order Kernel timing measurement
                int64_t duration = Time::nowNs().count() - slot_map->getSlot3GPP().t0_;
                pdctx->order_kernel_timing_tracker.addValue(duration, sfn, slot);

                ti.add("Order Metrics");
                for(int ii = first_cell; ii < first_cell + num_cells && ii < slot_map->getNumCells(); ii++)
                {
                    Cell*     cell_ptr    = slot_map->aggr_cell_list[ii];
                    ulbuf_st1   = slot_map->aggr_ulbuf_st1[ii];
                    cell_idx_list[cell_count++]=cell_ptr->getIdx();
                    if(cell_ptr==nullptr || (ulbuf_st1 == nullptr && ulbuf_st3_v.size() == 0))
                    {
                        NVLOGI_FMT(TAG, "ST1 buffer empty");
                        continue;
                    }
                    cell_ptr->updateMetric(CellMetric::kUlSlotsTotal, 1);
                    if(oentity->getOrderExitCondition(ii)!=ORDER_KERNEL_EXIT_PRB && cell_ptr->getPuschDynPrmIndex(slot) != -1)
                    {
                        cell_timeout_list[cell_ptr->getPuschDynPrmIndex(slot)]=oentity->getOrderExitCondition(ii);
                        if(pusch->getSetupStatus() == CH_SETUP_DONE_NO_ERROR && pusch->getRunStatus() == CH_RUN_DONE_NO_ERROR)
                        {
                            pusch_terminating_state = true;
                            pusch->setWorkCancelFlag();
                        }
                        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}.{} Slot Map {} Order kernel timeout error (exit condition {}) for cell index {} Dyn index {}! {}",
                                sfn, slot, slot_map->getId(), oentity->getOrderExitCondition(ii), ii, cell_ptr->getPuschDynPrmIndex(slot),
                                pusch_terminating_state ? "Attempting PUSCH pipeline termination" : "Not attempting PUSCH pipeline termination due to PUSCH cuPHY setup and/or channel error");
                        ok_timeout = true;
                        cell_error_list[cell_error_count++]=cell_ptr->getIdx();
                    }
                    if(pdctx->ru_health_check_enabled())
                    {
                        if(cell_ptr->isHealthy() && (oentity->getOrderExitCondition(ii) == ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT) || (oentity->getOrderExitCondition(ii) == ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT))
                        {
                            cell_ptr->num_consecutive_ok_timeout++;
                            uint32_t ok_timeout = cell_ptr->num_consecutive_ok_timeout.load();
                            if(ok_timeout >= pdctx->getAggr_obj_non_avail_th()/2)
                            {
                                NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} cell index {} Dyn index {} setting as unhealthy!",sfn,slot,slot_map->getId(),ii,cell_ptr->getPuschDynPrmIndex(slot));
                                cell_ptr->setUnhealthy();
                            }
                        }
                        else
                        {
                            cell_ptr->num_consecutive_ok_timeout = 0;
                            if(cell_ptr->isHealthy() == false)
                            {
                                if(ulbuf_st1 && cell_ptr->getPuschDynPrmIndex(slot) != -1)
                                {
                                    // If cell was marked as unhealthty, order kernel did not wait for UL traffic on this cell and exited early with ORDER_KERNEL_EXIT_PRB to avoid
                                    // impacing other cells. However, we need to mark the tbCrc for this cell as CRC_FAIL. Hence set the error condition for this cell in cell_timeout_list
                                    cell_timeout_list[cell_ptr->getPuschDynPrmIndex(slot)]=ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
                                    NVLOGI_FMT(TAG, "SFN {}.{} Slot Map {} cell index {} Dyn index {} add to cell_timeout error_code={}",
                                        sfn,slot,slot_map->getId(),ii,cell_ptr->getPuschDynPrmIndex(slot), +ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT);
                                    if(ok_timeout == false)
                                        cell_error_list[cell_error_count++]=cell_ptr->getIdx();
                                }
                                uint32_t unhealthy_slots = cell_ptr->num_consecutive_unhealthy_slots.load();
                                if(unhealthy_slots >= pdctx->get_max_ru_unhealthy_slots())
                                {
                                    NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} cell index {} Dyn index {} setting as healthy!",sfn,slot,slot_map->getId(),ii,cell_ptr->getPuschDynPrmIndex(slot));
                                    cell_ptr->setHealthy();
                                    cell_ptr->num_consecutive_unhealthy_slots = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        if(!en_orderKernel_tb)
        {
            if((!early_uci_task_complete) || (!isWaitFullSlotComplete)) 
            {
                //Early UCI Task Wait Processing
                if(!early_uci_task_complete){
                    early_uci_task_complete = slot_map->waitEarlyUciEndTaskNonBlocking();
                    if(early_uci_task_complete){
                        ti.add("Wait EarlyUCITaskEnd");
                    }
                }
                
                if(early_uci_task_complete){  
                    // check waitCompletedFullSlotEvent only if the PUSCH early-HARQ processing is enabled and early-HARQ bits are present
                    if((slot_map->getIsEarlyHarqPresent()==1) && pusch){
                        if(!isWaitFullSlotComplete){
                            isWaitFullSlotComplete=pusch->waitEventNonBlocking(pusch->getPuschStatParams()->waitCompletedFullSlotEvent);
                        }
                        
                        // waitCompletedFullSlotEvent event can assure the full-slot wait kernel is done and the corresponding status is for the current slot (not the stale status from the previous slot). 
                        if(isWaitFullSlotComplete){
                            if(pusch->getPostEarlyHarqWaitKernelStatus()==PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT) {
                                //Set force CRC error flag to true
                                NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} PUSCH Post Early Harq Wait kernel timeout!",sfn,slot,slot_map->getId());
                                gpu_early_harq_timeout = true;
                            } else {
                                gpu_early_harq_timeout = false;
                            }
                        }
                    }
                    else{
                        gpu_early_harq_timeout = false;
                    }
                }
            }


            //PUSCH Wait Processing - (only check completion if early UCI task is complete)
            if(pusch_waiter.getState() != WAIT_STATE_STARTED || (pusch_waiter.getState() == WAIT_STATE_STARTED && early_uci_task_complete)) {
                wa = pusch_waiter.checkAction();
                switch(wa) {
                    case WAIT_ACTION_STARTED:
                        ti.add("Started PUSCH");
                        break;
                    case WAIT_ACTION_COMPLETED:
                        slot_map->timings.end_t_ul_pusch_compl[0] = Time::nowNs();

                        ti.add("Validate PUSCH");
                        pusch->validate(cell_timeout_list,gpu_early_harq_timeout,slot_map->aggr_ulbuf_pcap_capture, slot_map->aggr_ulbuf_pcap_capture_ts);

                        ti.add("Callback PUSCH");
                        slot_map->timings.start_t_ul_pusch_cb[0] = Time::nowNs();
                        pusch->callback(cell_timeout_list,gpu_early_harq_timeout);
                        slot_map->timings.end_t_ul_pusch_cb[0] = Time::nowNs();

                        ti.add("Done PUSCH");
                        break;
                }
            }

            //PUCCH Wait Processing
            wa = pucch_waiter.checkAction();
            switch(wa) {
                case WAIT_ACTION_STARTED:
                    ti.add("Started PUCCH");
                    break;
                case WAIT_ACTION_COMPLETED:
                    slot_map->timings.end_t_ul_pucch_compl[0] = Time::nowNs();

                    ti.add("Validate PUCCH");
                    pucch->validate();

                    ti.add("Callback PUCCH");
                    slot_map->timings.start_t_ul_pucch_cb[0] = Time::nowNs();
                    pucch->callback();
                    slot_map->timings.end_t_ul_pucch_cb[0] = Time::nowNs();

                    ti.add("Done PUCCH");
                    break;
            }

            //PRACH Wait Processing
            wa = prach_waiter.checkAction();
            switch(wa) {
                case WAIT_ACTION_STARTED:
                    ti.add("Started PRACH");
                    break;
                case WAIT_ACTION_COMPLETED:
                    slot_map->timings.end_t_ul_prach_compl[0] = Time::nowNs();

                    ti.add("Validate PRACH");
                    prach->validate();

                    ti.add("Callback PRACH");
                    slot_map->timings.start_t_ul_prach_cb[0] = Time::nowNs();
                    prach->callback();
                    slot_map->timings.end_t_ul_prach_cb[0] = Time::nowNs();

                    ti.add("Done PRACH");
                    break;
            }

            if(pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)
            {
                //SRS Wait Processing
                wa = srs_waiter.checkAction();
                switch(wa) {
                    case WAIT_ACTION_STARTED:
                        ti.add("Started SRS");
                        break;
                    case WAIT_ACTION_COMPLETED:
                        slot_map->timings.end_t_ul_srs_compl[0] = Time::nowNs();

                        ti.add("Validate SRS");
                        srs->validate();

                        ti.add("Callback SRS");
                        slot_map->timings.start_t_ul_srs_cb[0] = Time::nowNs();
                        srs->callback(srs_order_cell_timeout_list);
                        slot_map->timings.end_t_ul_srs_cb[0] = Time::nowNs();

                        ti.add("Done SRS");
                        break;
                }
            }            

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////// PUSCH UL capture debug to capture on CRC error
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            if(pdctx->get_ul_pcap_capture_enable())
            {
                CuphyOAM *oam = CuphyOAM::getInstance();
                auto pcap_capture_cell_bitmask = oam->ul_pcap_arm_cell_bitmask.load();
                auto pcap_flush_cell_bitmask = oam->ul_pcap_flush_cell_bitmask.load();
                // If the cell is in the flush bitmask, also capture the PCAP for the cell
                uint64_t capture_cells = pcap_flush_cell_bitmask & pcap_capture_cell_bitmask;
                for (int i = 0; i < UL_MAX_CELLS_PER_SLOT; ++i)
                {
                    uint64_t cell_mask = 1ULL << i;
                    if (capture_cells & cell_mask)
                    {
                        auto& capture_info = pdctx->ul_pcap_capture_context_info.ul_pcap_capture_info[pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx];
                        capture_info.mtu = pdctx->get_ul_pcap_capture_mtu();
                        capture_info.buffer_pointer = static_cast<uint8_t*>(slot_map->aggr_ulbuf_pcap_capture[i]->getBufH());
                        capture_info.buffer_pointer_ts = static_cast<uint8_t*>(slot_map->aggr_ulbuf_pcap_capture_ts[i]->getBufH());
                        capture_info.cell_id = i;
                        capture_info.sfn = sfn;
                        capture_info.slot = slot;
                        pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx = (pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx + 1) % UL_MAX_CELLS_PER_SLOT;
                        oam->ul_pcap_flush_cell_bitmask.fetch_and(~cell_mask, std::memory_order_release);
                        NVLOGC_FMT(TAG, "SFN {}.{} Cell {} Trigger Pcap flush", sfn, slot, i);
                    }
                }
            }
        }

        if(!pusch && !pucch && !prach)
        {
            wa = order_waiter.checkAction(true);
        }

        /* NVLOGI_FMT(TAG, "Waiting states before, timer: {} state values:{} {}",
                   (Time::nowNs() - start_t).count(), order_waiter.getState(), order_waiter_srs.getState()); */

        //Determine if we need to keep running
        if(!en_orderKernel_tb)
        {
            still_waiting = (order_waiter.stillWaiting() || (!early_uci_task_complete) ||
                            pusch_waiter.stillWaiting() || pucch_waiter.stillWaiting() || prach_waiter.stillWaiting());
            if(pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)
            {
                still_waiting = still_waiting || srs_waiter.stillWaiting();
            }
        }
        else
        {
            still_waiting = order_waiter.stillWaiting();
        }

        /* NVLOGI_FMT(TAG, "Waiting states after, timer: {} state values:{} {} {} {} {} {} {}",
                    (Time::nowNs() - start_t).count(),
                    order_waiter.getState(), order_waiter_srs.getState(),
                    pusch_waiter.getState(),pucch_waiter.getState(),prach_waiter.getState(),srs_waiter.getState(),ulbfw_waiter.getState()); */
        
        //Attempt pusch pipeline termination if order kernel is still waiting more than 2ms from the start of the task 3 
        //And if early work termination is enabled.
        if((order_waiter.stillWaiting())&&(Time::nowNs() - start_t > order_wait_thresh_t) && pusch)
        {
            bool terminate_pusch = false;
            if(pusch->getSetupStatus() == CH_SETUP_DONE_NO_ERROR && pusch->getRunStatus() == CH_RUN_DONE_NO_ERROR)
            {
                terminate_pusch = true;
                pusch->setWorkCancelFlag();
            }

            // Print ERR log only when PUSCH timeout state changes or when terminating state changes
            if (pusch_reorder_kernel_timeout_state != true || pusch_terminating_state != terminate_pusch)
            {
                // Update PUSCH timeout state and terminating state
                pusch_reorder_kernel_timeout_state = true;
                pusch_terminating_state = terminate_pusch;
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: SFN {}.{} AGGR 3 task waiting for order kernel for more than {} ns for Slot Map {}. {}",
                            sfn, slot, order_wait_thresh_t.count(), slot_map->getId(),
                            pusch_terminating_state ? "Attempting PUSCH pipeline termination" : "Not attempting PUSCH pipeline termination due to PUSCH cuPHY setup and/or channel error");
            }
        }
        else
        {
            // Reset PUSCH timeout state when waiting for order kernel is done
            if (pusch_reorder_kernel_timeout_state == true)
            {
                pusch_reorder_kernel_timeout_state = false;
                NVLOGC_FMT(TAG, "SFN {}.{} AGGR 3 task PUSCH order kernel timeout waiting finished", sfn, slot);
            }
        }

//In flux..
#if 0
        if ((!pusch_early_exit) && pusch_waiter.stillWaiting() && (Time::nowNs() - start_t > timeout_thresh_t/4)) {
#if 1
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 task waiting for PUSCH for more than {} ns for Slot Map {}; will set early exit flag",
                        timeout_thresh_t.count()/4, slot_map->getId());
            pusch->setWorkCancelFlag();
#else
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 task waiting for PUSCH for more than {} ns for Slot Map {} but won't do anything",
                        timeout_thresh_t.count()/4, slot_map->getId()); // used to check if we crash
#endif

/*
Example output log (last entry for 100K slots):
- default code (no delay)
  15:45:15.680486 WRN 47581 0 [MAC.FAPI] Cell 19 | DL 1469.14 Mbps 1400 Slots | UL  196.70 Mbps  400 Slots | Prmb  700 | HARQ 9600 | SR    0 | CSI1 2400 | CSI2 2400 | SRS    0 | ERR    0 | INV    0 | Slots 98000
  15:45:16.680015 WRN 47581 0 [MAC.FAPI] Finished running 100000 slots test. slot_counter=100000 restart_interval=0


- Without early exit and with delay in cfo kernel:
  15:28:01.440486 WRN 44427 0 [MAC.FAPI] Cell 19 | DL 1469.14 Mbps 1400 Slots | UL  121.20 Mbps  323 Slots | Prmb  418 | HARQ 7716 | SR    0 | CSI1 1938 | CSI2 1938 | SRS    0 | ERR  156 | INV 1218 | Slots 98000
  15:28:02.440017 WRN 44427 0 [MAC.FAPI] Finished running 100000 slots test. slot_counter=100000 restart_interval=0

- With early exit and with delay in cfo kernel
  15:37:04.160483 WRN 45637 0 [MAC.FAPI] Cell 19 | DL 1469.14 Mbps 1400 Slots | UL  108.21 Mbps  400 Slots | Prmb  700 | HARQ 9600 | SR    0 | CSI1 2400 | CSI2 2400 | SRS    0 | ERR    0 | INV 3600 | Slots 98000
  15:37:05.160015 WRN 45637 0 [MAC.FAPI] Finished running 100000 slots test. slot_counter=100000 restart_interval=0
*/
            pusch_early_exit = true;
        }
#endif

        if(still_waiting && (Time::nowNs() - start_t > timeout_thresh_t)) {
            if(!log_once) {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 task waiting for PHY Channels more than {} ns for Slot Map {}, state values: {} {} {} {} {} {}",
                        timeout_thresh_t.count(),
                        slot_map->getId(),
                        +order_waiter.getState(),
                        early_uci_task_complete,
                        +pusch_waiter.getState(), +pucch_waiter.getState(), +prach_waiter.getState(), +srs_waiter.getState());
                log_once = true;
            }
            /* Do not exit the while(still_waiting) loop. Need to keep spinning here to avoid freeing the slot map and other resources within the slot map while L1 is in recovery.
            If this while loop does not exit before the L1 recovery ends, then L1_Recovery will trigger a EXIT_L1
            If while loop exits during the L1 recovery, then the GPU process would have completed and the slotmap is correctly freed.*/
            ENTER_L1_RECOVERY();
        }
        //std::this_thread::sleep_for(std::chrono::microseconds(20); //FIXME could consider adding a delay here to avoid doing back to back event queries for large durations (e.g., even 1ms)
    } while(still_waiting);

    cleanup:

    slot_map->addSlotEndTask();
    if(cell_error_count>0)
    {
        if(pdctx->getUlCb(ul_cb))
        {
            NVLOGI_FMT(TAG, "Calling ul_tx_error_fn {}\n",__LINE__);
            ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_MISSING_UL_IQ,cell_error_list,cell_error_count,true);
        }
    }


    //Make sure that all UL tasks have run to completion here and release the slot map if SRS is not present or FX RU
    if(!srs || pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)
    {
        ti.add("Wait Slot End Task");
        if(slot_map->waitSlotEndTask(num_ul_tasks) < 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitSlotEnd returned error");
            if(pdctx->getUlCb(ul_cb))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
                ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_ERROR_INDICATION,SCF_ERROR_CODE_L1_UL_CPU_TASK_ERROR,cell_idx_list,cell_count,false);
            }
        }

        ti.add("Slot Map Release");
        PUSH_RANGE_PHYDRV("UL CLEAN", 1);
        if(ret_task==-1)
            slot_map->release(num_cells,false);
        else
            slot_map->release(num_cells,true);
        POP_RANGE_PHYDRV
    }

    ti.add("End Task");

    return 0;
}

int task_work_function_ul_aggr_3_srs(Worker* worker, void* param, int first_cell, int num_cells,int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task UL AGGR 3 SRS", 10);
    ti.add("Start Task");

    SlotMapUl* slot_map = (SlotMapUl*)param;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    PhySrsAggr* srs = slot_map->aggr_srs;
    OrderEntity* oentity = nullptr;
    int ret_task = 0;
    bool log_once = false;
    struct slot_command_api::ul_slot_callbacks ul_cb;

    t_ns timeout_thresh_t(10*NS_X_MS); //10ms wait on the UL cuPHY pipelines
    t_ns start_t;
    int sfn = 0, slot = 0, ret = 0;
    int num_ulc_tasks = get_num_ulc_tasks(pdctx->getNumULWorkers());

    if(srs) {
        oentity = slot_map->aggr_order_entity;
    }
    OrderWaiter order_waiter_srs(oentity);
    PhyWaiter srs_waiter(srs);

    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> cell_idx_list={};
    std::array<bool,UL_MAX_CELLS_PER_SLOT> srs_order_cell_timeout_list = {false};
    int cell_count=0;

    sfn = slot_map->getSlot3GPP().sfn_;
    slot = slot_map->getSlot3GPP().slot_;

    std::optional<CuphyCuptiScopedExternalId> cuphy_cupti_scoped_external_id;
    if (pdctx->cuptiTracingEnabled()) {
        cuphy_cupti_scoped_external_id.emplace(slot_map->getSlot3GPP().t0_);
    }

    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        ret_task=-1;
        goto cleanup;
    }

    //Only run after ULC tasks have completed
    ti.add("ULC Tasks Complete Wait");
    ret = slot_map->waitULCTasksComplete(num_ulc_tasks);
    if(ret != 0) {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3_srs timeout waiting for ULC Tasks, Slot Map {}",slot_map->getId());
        ret_task=-1;
        goto cleanup;
    }

    //Check abort condition
    ti.add("Check Task Abort");
    if(slot_map->tasksAborted()) {
        NVLOGW_FMT(TAG,"task_work_function_ul_aggr_3_srs Task aborted for Slot Map {}",slot_map->getId());
        ret_task=-1;
        goto cleanup;
    }

    if(pdctx->cpuCommEnabled()) {
        order_waiter_srs.setState(WAIT_STATE_COMPLETED);
    }

    ti.add("Wait Loop");
    start_t = Time::nowNs();
    slot_map->timings.start_t_ul_srs_compl[0] = Time::nowNs();
    bool still_waiting;
    
    do {
        wait_action wa;

        if(!(pdctx->cpuCommEnabled()) && srs) {
            wa = order_waiter_srs.checkAction(true);
            if(wa == WAIT_ACTION_COMPLETED) {
                ti.add("Wait SRS Order");

                for(int ii = first_cell; ii < first_cell + num_cells && ii < slot_map->getNumCells(); ii++) {
                    Cell* cell_ptr = slot_map->aggr_cell_list[ii];
                    ULInputBuffer* ulbuf_st2 = slot_map->aggr_ulbuf_st2[ii];
                    cell_idx_list[cell_count++]=cell_ptr->getIdx();
                    if(cell_ptr==nullptr || ulbuf_st2 == nullptr)
                        continue;
                    if(oentity->getOrderSrsExitCondition(ii)!=ORDER_KERNEL_EXIT_PRB) {
                        srs_order_cell_timeout_list[cell_ptr->getSrsDynPrmIndex()] = true;
                        NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} SRS Order kernel timeout error (exit condition {}) for cell index {}! srs_cell_dyn_index {}",
                            sfn,slot,slot_map->getId(),oentity->getOrderSrsExitCondition(ii),ii, cell_ptr->getSrsDynPrmIndex());
                    }
                    if(pdctx->ru_health_check_enabled())
                    {
                        if(cell_ptr->isHealthySrs() && (oentity->getOrderSrsExitCondition(ii) == ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT) || (oentity->getOrderSrsExitCondition(ii) == ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT))
                        {
                            cell_ptr->num_consecutive_ok_timeout_srs++;
                            uint32_t ok_timeout = cell_ptr->num_consecutive_ok_timeout_srs.load();
                            if(ok_timeout >= pdctx->getAggr_obj_non_avail_th()/2)
                            {
                                NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} SRS cell index {} Dyn index {} setting as unhealthy!",sfn,slot,slot_map->getId(),ii,cell_ptr->getSrsDynPrmIndex());
                                cell_ptr->setUnhealthySrs();
                            }
                        }
                        else
                        {
                            cell_ptr->num_consecutive_ok_timeout_srs = 0;
                            if(cell_ptr->isHealthySrs() == false)
                            {
                                uint32_t unhealthy_slots = cell_ptr->num_consecutive_unhealthy_slots_srs.load();
                                if(unhealthy_slots >= pdctx->get_max_ru_unhealthy_slots())
                                {
                                    NVLOGE_FMT(TAG,AERIAL_CUPHY_API_EVENT,"SFN {}.{} Slot Map {} SRS cell index {} Dyn index {} setting as healthy!",sfn,slot,slot_map->getId(),ii,cell_ptr->getSrsDynPrmIndex());
                                    cell_ptr->setHealthySrs();
                                    cell_ptr->num_consecutive_unhealthy_slots_srs = 0;
                                }
                            }
                        }
                    }                    
                }
            }
        }

        //SRS Wait Processing
        wa = srs_waiter.checkAction();
        switch(wa) {
            case WAIT_ACTION_STARTED:
                ti.add("Started SRS");
                break;
            case WAIT_ACTION_COMPLETED:
                slot_map->timings.end_t_ul_srs_compl[0] = Time::nowNs();

                ti.add("Validate SRS");
                srs->validate();

                ti.add("Callback SRS");
                slot_map->timings.start_t_ul_srs_cb[0] = Time::nowNs();
                srs->callback(srs_order_cell_timeout_list);
                slot_map->timings.end_t_ul_srs_cb[0] = Time::nowNs();

                ti.add("Done SRS");
                break;
        }

        still_waiting = (order_waiter_srs.stillWaiting() || srs_waiter.stillWaiting());

        if(still_waiting && (Time::nowNs() - start_t > timeout_thresh_t)) {
            if(!log_once) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 SRS task waiting for PHY Channels more than {} ns for Slot Map {}, state values: {} {}",
                           timeout_thresh_t.count(),
                           slot_map->getId(),
                           +order_waiter_srs.getState(),
                           +srs_waiter.getState());
                log_once = true;
            }
            ENTER_L1_RECOVERY();
        }

        // Add 10us sleep if we need to keep waiting
        if(still_waiting) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    } while(still_waiting);

cleanup:
    slot_map->addSlotEndTask();

    ti.add("Wait Slot End Task");
    if(slot_map->waitSlotEndTask(num_ul_tasks) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "waitSlotEnd returned error");
        if(pdctx->getUlCb(ul_cb))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Calling ul_tx_error_fn {}\n",__LINE__);
            ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context, slot_map->getSlot3GPP(),SCF_FAPI_UL_TTI_REQUEST,SCF_ERROR_CODE_L1_UL_CPU_TASK_ERROR,cell_idx_list,cell_count, false);
        }
    }

    ti.add("Slot Map Release");
    PUSH_RANGE_PHYDRV("UL CLEAN", 1);
    if(ret_task==-1)
        slot_map->release(num_cells,false);
    else
        slot_map->release(num_cells,true);
    POP_RANGE_PHYDRV    

    ti.add("End Task");

    return ret_task;
}

int task_work_function_ul_aggr_3_ulbfw(Worker* worker, void* param, int first_cell, int num_cells, int num_ul_tasks)
{
    auto ctx = makeInstrumentationContextUL(param, worker);
    TaskInstrumentation ti(ctx, "UL Task UL AGGR 3 ULBFW",  13);
    ti.add("Start Task");
    int ret = 0;
    SlotMapUl* slot_map = (SlotMapUl*)param;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();
    PhyUlBfwAggr* ulbfw = slot_map->aggr_ulbfw;
    uint32_t cpu;
    t_ns start_t;
    bool log_once = false;
    bool still_waiting;
    t_ns timeout_thresh_t(2*NS_X_MS); //2ms wait on the ULBFW cuPHY pipeline
    PhyWaiter ulbfw_waiter(ulbfw);
    wait_action wa;

    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        goto cleanup;
    }

    if(ulbfw == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No ulbfw in SlotMap UL");
        goto cleanup;
    }

    NVLOGI_FMT(TAG, "UL Task 3 (ULBFW) started CPU {} for Map {}, SFN slot ({},{})", 
               cpu, slot_map->getId(), slot_map->getSlot3GPP().sfn_, slot_map->getSlot3GPP().slot_);

    ti.add("Wait ULBFW End Task");
    if(slot_map->waitUlBfwEndTask() < 0) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ULBFW task timed out for Slot Map {}", slot_map->getId());
        goto cleanup;
    }

    ti.add("Wait Loop");
    start_t = Time::nowNs();
    slot_map->timings.start_t_ul_bfw_compl[0] = Time::nowNs();
    do {
        //ULBFW Wait Processing
        wa = ulbfw_waiter.checkAction();
        switch(wa) {
            case WAIT_ACTION_STARTED:
                ti.add("Started ULBFW");
                break;
            case WAIT_ACTION_COMPLETED:
                slot_map->timings.end_t_ul_bfw_compl[0] = Time::nowNs();

                ti.add("Validate ULBFW");
                ulbfw->validate();

                ti.add("Callback ULBFW");
                slot_map->timings.start_t_ul_bfw_cb[0] = Time::nowNs();
                ulbfw->callback();
                slot_map->timings.end_t_ul_bfw_cb[0] = Time::nowNs();
                ti.add("Done ULBFW");
                break;
        }
        still_waiting = ulbfw_waiter.stillWaiting();
        if(still_waiting && (Time::nowNs() - start_t > timeout_thresh_t)) {
            if(!log_once) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: AGGR 3 ULBFW task waiting for PHY Channels more than {} ns for Slot Map {}, state values: {}",
                           timeout_thresh_t.count(),
                           slot_map->getId(),
                           +ulbfw_waiter.getState());
                log_once = true;
            }
            ENTER_L1_RECOVERY();
        }

        if(still_waiting) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }while(still_waiting);

cleanup:
    slot_map->addSlotEndTask();

    ti.add("End Task");

    return ret;
}

