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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 15) // "DRV.PDSCH"

#include "phypdsch_aggr.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "cuda_events.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "memtrace.h"

#define getName(var)  #var

// nvlog versions of the cuphy::print_pdsch_static, cuphy::print_pdsch_dynamic, and cuphy::print_pdsch_dynamic_cell_group functions
// to print cuPHY PDSCH static and dynamic parameters.
// printPdschDynPrmsAggr calls printPdschDynamicCellGroupAggr under the hood too.
void printPdschStaticParamsAggr(const cuphyPdschStatPrms_t* static_params);
void printPdschDynPrmsAggr(const cuphyPdschDynPrms_t* dynamic_params);
void printPdschDynamicCellGroupAggr(const cuphyPdschCellGrpDynPrm_t* cell_group_params);

PhyPdschAggr::PhyPdschAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx *        _mpsCtx) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    cuphyStatus_t status;
    bool          ref_check              = false;
    bool          identical_ldpc_configs = true;
    PhyDriverCtx* pdctx                  = StaticConversion<PhyDriverCtx>(pdh).get();

    mf.init(_pdh, std::string("PhyPdschAggr"), sizeof(PhyPdschAggr));
    cuphyMf.init(_pdh, std::string("cuphyPdschTx"), 0);

    channel_type = slot_command_api::channel_type::PDSCH;
    channel_name.assign("PDSCH");

    try
    {
        if(pdctx->isValidation())
            ref_check = true;
        tb_crc_data_in = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
    }
    PHYDRIVER_CATCH_THROW_EXCEPTIONS();

    cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

    gDev->synchronizeStream(s_channel);

    procModeBmsk = PDSCH_PROC_MODE_NO_GRAPHS;
    if(pdctx->getEnableDlCuphyGraphs())
        procModeBmsk = PDSCH_PROC_MODE_GRAPHS;
    if(pdctx->getPdschFallback() == 1)
        procModeBmsk |= PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK;

    tb_bytes = 0;
    tb_count = 0;
    nUes = 0;
    first_slot = 0;

    cell_id_list.clear();

    std::memset(&static_params, 0, sizeof(cuphyPdschStatPrms_t));
    static_params.pCellStatPrms = NULL;
    static_params.nCells = 0;
    static_params.pOutInfo = &cuphy_tracker;

    dyn_params.cuStream = s_channel;
    dyn_params.procModeBmsk = procModeBmsk;
    dyn_params.pTbCRCDataIn = &tb_crc_data_in;

    DataIn.pTbInput = (uint8_t**) calloc(PDSCH_MAX_CELLS_PER_CELL_GROUP, sizeof(uint8_t*));
    dyn_params.pDataIn = &DataIn;
    DataOut.pTDataTx = (cuphyTensorPrm_t*) calloc(PDSCH_MAX_CELLS_PER_CELL_GROUP, sizeof(cuphyTensorPrm_t));
    statusOut = {cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    dyn_params.pDataOut = &DataOut;
    dyn_params.pStatusInfo = &statusOut;

    static_params_cell.clear();
    cell_id_list.clear();

    handle = nullptr;
};

PhyPdschAggr::~PhyPdschAggr(){
    if(handle)
        cuphyDestroyPdschTx(handle);
    delete[] static_params.pDbg;

    free(DataOut.pTDataTx);
    free(DataIn.pTbInput);
};

int PhyPdschAggr::createPhyObj()
{
    bool identical_ldpc_configs = true;
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;

    setCtx();
    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        int tmp_cell_id = cell_ptr->getPhyId();

        // Add only active cells here
        if(tmp_cell_id == DEFAULT_PHY_CELL_ID)
            continue;

        if(static_params_cell.size() > 0)
        {
            auto it = std::find_if(
                static_params_cell.begin(), static_params_cell.end(),
                [&tmp_cell_id](cuphyCellStatPrm_t _cellStatPrm) { return (_cellStatPrm.phyCellId == tmp_cell_id); }
            );

            if(it != static_params_cell.end())
                continue;
        }

        cell_id_list.push_back(tmp_cell_id);

        cellStatPrm.phyCellId = cell_ptr->getPhyId();
        cellStatPrm.nRxAnt    = cell_ptr->getRxAnt();
        cellStatPrm.nRxAntSrs = cell_ptr->getRxAntSrs();
        cellStatPrm.nTxAnt    = cell_ptr->getTxAnt();
        cellStatPrm.nPrbUlBwp = cell_ptr->getPrbUlBwp();
        cellStatPrm.nPrbDlBwp = cell_ptr->getPrbDlBwp();
        cellStatPrm.mu        = cell_ptr->getMu();

        //Only mu == 1 supported; FIXME Eventually add a consistent way for error checking for all params.
        if(cellStatPrm.mu != 1)
        {
            throw std::runtime_error("Unsupported numerology value!");
        }

        static_params_cell.push_back(cellStatPrm);
        mf.addCpuRegularSize(sizeof(cuphyCellStatPrm_t));
    }

    static_params.nMaxCellsPerSlot      = static_params_cell.size();
    static_params.nMaxUesPerCellGroup   = PDSCH_MAX_UES_PER_CELL_GROUP;
    static_params.read_TB_CRC           = false;
    static_params.full_slot_processing  = true;
    static_params.nCells                = static_params_cell.size();
    static_params.pCellStatPrms         = static_cast<cuphyCellStatPrm_t*>(static_params_cell.data());
    static_params.enableBatchedMemcpy   = pdctx->getUseBatchedMemcpy();

    /*
     * Create cuPHY object only if the desider number of cells
     * has been activated into cuphydriver.
     */
    if(static_params_cell.size() == pdctx->getCellGroupNum())
    {
        int cuda_strm_prio = 0;
        CUDA_CHECK_PHYDRIVER(cudaStreamGetPriority(s_channel, &cuda_strm_prio));
        static_params.stream_priority = cuda_strm_prio;

        static_params.pDbg = new cuphyPdschDbgPrms_t[cellCount]; // ideally delete before
        uint8_t check_TB_size = 1; // enable optional TB size check
        for (int i = 0; i < static_params_cell.size(); i++)
            //static_params.pDbg[i] = {"./testVectors/TVnr_3900_PDSCH_gNB_CUPHY_s0p15.h5", check_TB_size, pdctx->isValidation(), identical_ldpc_configs};
            static_params.pDbg[i] = {"", check_TB_size, pdctx->isValidation(), identical_ldpc_configs};
        //  static_params.pDbg[i] = { "/path/to/TV_cuphy_V14-DS-08_slot0_MIMO2x16_PRB82_DataSyms10_qam64.h5",
        //                           check_TB_size, true, identical_ldpc_configs};
        //int cuda_strm_prio = 0;
        //cuphy::print_pdsch_static(&static_params);
        //printPdschStaticParamsAggr(&static_params); //cuphydriver nvlog version; see below
        cuphyStatus_t createStatus = cuphyCreatePdschTx(&handle, &static_params); // Currently calling PdschTx constructor with empty filename
        std::string cuphy_ch_create_name = "cuphyCreatePdschTx";

        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);

        //printf("from cuphydriver %lu \n", cuphyGetGpuMemoryFootprintPdschTx(handle));
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPdschTx(handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(static_params.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();

        gDev->synchronizeStream(s_channel);

    }
    else if(static_params_cell.size() > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    return 0;
}

slot_command_api::pdsch_params* PhyPdschAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->pdsch.get();
}

int PhyPdschAggr::setup(
    const std::vector<Cell *> &aggr_cell_list,
    const std::vector<DLOutputBuffer *> &aggr_dlbuf
)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_ptr      = pdctx->getCellById(cell_id);
    cuphyStatus_t status;
#if 1
    if(!aggr_slot_params->cgcmd->pdsch->tb_data.pTbInput[0])
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, s_channel));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
        return 0;
    }
#endif
    setCtx();

    dyn_params.procModeBmsk = procModeBmsk;
    dyn_params.procModeBmsk |= PDSCH_INTER_CELL_BATCHING;

    if(
        (pdctx->getPdschFallback() == 0)                                                ||
        (pdctx->getPdschFallback() == 1 && pdctx->getEnableDlCuphyGraphs() && first_slot < 2)  ||
        (pdctx->getPdschFallback() == 1 && !pdctx->getEnableDlCuphyGraphs() && first_slot < 1)
    )
    {
        slot_command_api::pdsch_params* pparms = getDynParams();
        dyn_params.pCellGrpDynPrm = &pparms->cell_grp_info;

        DataIn.pBufferType = aggr_slot_params->cgcmd->pdsch->tb_data.pBufferType;
        for(int index=0; index < dyn_params.pCellGrpDynPrm->nCells; index++)
            DataIn.pTbInput[index] = aggr_slot_params->cgcmd->pdsch->tb_data.pTbInput[index];

        for(int index=0; index < aggr_cell_list.size(); index++)
        {
            //no need to check for aggr_dlbuf[index] == nullptr because for DL, all cells have a corresponding
            //dlbuf attached. However, pTDataTx has to be sent only for the cells which have PDSCH and is indexed
            //by cellPrmDynIdx. Hence find if aggr_cell_list[index] has a PDSCH and assign it to pTDataTx[cellPrmDynIdx]
            uint16_t phyCellId = aggr_cell_list[index]->getPhyId();
            int cellPrmDynIdx = -1;
            for(uint32_t i=0; i < dyn_params.pCellGrpDynPrm->nCells; i++)
            {
                //NVLOGD_FMT(TAG, "{}:{} PDSCH testModel={}",__func__,__LINE__, dyn_params.pCellGrpDynPrm->pCellPrms[i].testModel); //Debug log commented out as a WAR for the large setup time upon first test run post build compile if left uncommented
                uint16_t cellPrmStatIdx = dyn_params.pCellGrpDynPrm->pCellPrms[i].cellPrmStatIdx;
                if(phyCellId == static_params.pCellStatPrms[cellPrmStatIdx].phyCellId)
                {
                    cellPrmDynIdx = dyn_params.pCellGrpDynPrm->pCellPrms[i].cellPrmDynIdx;
                    break;
                }
            }

            if(cellPrmDynIdx !=-1)
            {
                DataOut.pTDataTx[cellPrmDynIdx].desc  = aggr_dlbuf[index]->getTensor()->desc().handle();
                DataOut.pTDataTx[cellPrmDynIdx].pAddr = aggr_dlbuf[index]->getBufD();
                //NVLOGI_FMT(TAG, "PhyPdschAggr::setup - Cell {} cellPrmDynIdx {} DLBuffer {} at index {}",
                    //phyCellId,cellPrmDynIdx,aggr_dlbuf[index]->getId(),index);
            }
            /*else
                NVLOGI_FMT(TAG, "PhyPdschAggr: Cell {} has no PDSCH", phyCellId);
            */
        }

         //cuphy::print_pdsch_dynamic_cell_group(dyn_params.pCellGrpDynPrm);
         //printParametersAggr(dyn_params.pCellGrpDynPrm);
         //printPdschDynamicCellGroupAggr(dyn_params.pCellGrpDynPrm); //cuphydriver nvlog version
         //printPdschDynPrmsAggr(&dyn_params); //cuphydriver nvlog version, included printPdschDynamicCellGroupAggr too
        if(pdctx->enable_prepone_h2d_cpy)
        {
            if(pdctx->h2d_copy_thread_enable) //Wait only if copy thread is enabled
            {
                if(waitH2dCopyCudaEventRec()<0)
                {
                    const int sfn  = aggr_slot_params->si->sfn_;
                    const int slot = aggr_slot_params->si->slot_;
                    NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error! waitH2dCopyCudaEventRec timeout!", sfn, slot);
                    return -1;
                }
            }
            CUDA_CHECK(cudaStreamWaitEvent(s_channel, pdctx->get_event_pdsch_tb_cpy_complete(aggr_slot_params->si->slot_), 0));
        }
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, s_channel));
        //cuphy::print_pdsch_dynamic(&dyn_params);
        status = cuphySetupPdschTx(handle, &dyn_params, nullptr); // cuPHY should not throw any exception. Can also add try/catch if need be.
        if(status != CUPHY_STATUS_SUCCESS)
        {
            const int sfn  = aggr_slot_params->si->sfn_;
            const int slot = aggr_slot_params->si->slot_;
            if (dyn_params.pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) {
                NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB Error in cuphySetupPdschTx(): {}. Will not call cuphyRunPdschTx(). May be L2 misconfiguration. Triggered by TB {} in cell group and cellPrmStatIdx {}.", sfn, slot, cuphyGetErrorString(status), dyn_params.pStatusInfo->ueIdx, dyn_params.pStatusInfo->cellPrmStatIdx);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error in cuphySetupPdschTx(): {}. Will not call cuphyRunPdschTx(). May be L2 misconfiguration.", sfn, slot, cuphyGetErrorString(status));
            }

            {
                MemtraceDisableScope md;
                CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
            }
            return -1;
        }
        {
            MemtraceDisableScope md;
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
        }

        if(pdctx->getPdschFallback() == 1)
            first_slot++;
    }
    else
    {
        for(int index=0; index < static_params.nCells && index < aggr_dlbuf.size() && index < PDSCH_MAX_CELLS_PER_CELL_GROUP; index++)
            fbOutBuf[index] = aggr_dlbuf[index]->getBufD();

        //Still keep these two to highlight from timers that setup is not run
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, s_channel));
        status = cuphyFallbackBuffersSetupPdschTx(handle, (void**)fbOutBuf, aggr_dlbuf.size(), s_channel);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyFallbackBuffersSetupPdschTx(): {}", cuphyGetErrorString(status));
            return -1;
        }
        {
            MemtraceDisableScope md;
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
        }
    }

    return 0;
}

int PhyPdschAggr::run()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    cuphyStatus_t status;
    int ret=0;

    setCtx();
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, s_channel));
    }
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR) && (aggr_slot_params->cgcmd->pdsch->tb_data.pTbInput[0]))
    {
        status = cuphyRunPdschTx(handle, dyn_params.procModeBmsk);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyRunPdschTx(): {}", cuphyGetErrorString(status));
            ret=-1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, s_channel));
    }
    return ret;
}

int PhyPdschAggr::callback(struct slot_command_api::slot_indication si)
{
    PhyDriverCtx*                       pdctx    = StaticConversion<PhyDriverCtx>(pdh).get();
    slot_command_api::dl_slot_callbacks dl_cb;

#ifdef AERIAL_METRICS
    Cell*    cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list, &cellCount);
    if(cellCount == 0)
        return EINVAL;

    slot_command_api::pdsch_params* pparms = getDynParams();
    dyn_params.pCellGrpDynPrm              = &pparms->cell_grp_info;
    auto n_cells                           = pparms->cell_grp_info.nCells;
    auto cell_metrics                      = pparms->cell_grp_info.pCellMetrics;

    for(uint32_t cellIdx = 0; cellIdx < cellCount; cellIdx++)
    {
        auto& cell_ptr = cell_list[cellIdx];
        cell_ptr->updateMetric(CellMetric::kPdschTxBytesTotal, cell_metrics[cellIdx].tbSize);
        cell_ptr->updateMetric(CellMetric::kPdschTxTbTotal, cell_metrics[cellIdx].nTBs);
    }
#endif

    if(pdctx->getDlCb(dl_cb))
    {
        dl_cb.callback_fn(dl_cb.callback_fn_context, getDynParams());

        // for(int idx = 0; idx < cell_id_list.size(); idx++)
        // {
        //     Cell * cell_ptr = pdctx->getCellByPhyId(cell_id_list[idx]);
        //     if(cell_ptr)
        //     {
        //         NVLOGD_FMT(TAG, "After DL callback for cell {}", cell_ptr->getId());
        //         cell_ptr->updateMetric(CellMetric::kPdschTxBytesTotal, tb_bytes);
        //         cell_ptr->updateMetric(CellMetric::kPdschNrOfUesPerSlot, nUes);
        //         cell_ptr->updateMetric(CellMetric::kPdschProcessingTime, this->getGPURunTime());
        //     }
        // }
    }

    // if(
    //     (pdctx->getPdschFallback() == 0)                                                    ||
    //     (pdctx->getPdschFallback() == 1 && pdctx->getEnableDlCuphyGraphs() && cell_ptr->first_slot < 2) ||
    //     (pdctx->getPdschFallback() == 1 && !pdctx->getEnableDlCuphyGraphs() && cell_ptr->first_slot < 1)
    // )
    // {
    //     tb_count = pdsch_dyn_params.pCellGrpDynPrm->nCws;
    //     cell_ptr->updateMetric(CellMetric::kPdschTxTbTotal, tb_count);

    //     tb_bytes = 0;
    //     for(uint16_t tb_idx = 0; tb_idx < tb_count; tb_idx++)
    //     {
    //         tb_bytes += pdsch_dyn_params.pCellGrpDynPrm->pCwPrms[tb_idx].tbSize;
    //     }
    //     nUes = pdsch_dyn_params.pCellGrpDynPrm->nUes;
    // }


    return 0;
}

int PhyPdschAggr::cleanup()
{
    PhyChannel::cleanup();

    return 0;
}

void PhyPdschAggr::updatePhyCellId(uint16_t phyCellId_old,uint16_t phyCellId_new)
{
    for(uint32_t i=0; i < static_params_cell.size(); i++)
    {
        if(static_params_cell[i].phyCellId == phyCellId_old)
        {
            static_params_cell[i].phyCellId = phyCellId_new;
            break;
        }
    }
}

int PhyPdschAggr::waitH2dCopyCudaEventRec() {
    t_ns start_wait;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(getPhyDriverHandler()).get();

    start_wait = Time::nowNs();
    int slot_index = aggr_slot_params->si->slot_ % PDSCH_MAX_GPU_BUFFS;
    int tmp_i = 1;
    do{
        if(Time::getDifferenceNowToNs(start_wait).count() > (GENERIC_WAIT_THRESHOLD_NS * 2))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}.{} H2D copy CUDA event record wait is taking more than {} ns", aggr_slot_params->si->sfn_,aggr_slot_params->si->slot_,(GENERIC_WAIT_THRESHOLD_NS * 2));
            return -1;
        }
#if 0
        if(Time::getDifferenceNowToNs(start_wait).count() > (tmp_i * 500000))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "H2D copy Cuda event record is taking more than {} ns for Frame {} Slot {}", tmp_i * 500000, aggr_slot_params->si->sfn_,aggr_slot_params->si->slot_);
            tmp_i += 1;
        }
#endif

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Wait end Map {} num_active_cells = {} atomvar = {} ",
        //                 getId(), num_active_cells, atom_dl_end_threads.load());

    } while(pdctx->h2d_copy_cuda_event_rec_done[slot_index].load(std::memory_order_acquire) != true);

    pdctx->h2d_copy_cuda_event_rec_done[slot_index].store(false,std::memory_order_release);
    return 0;
}


void printPdschStaticParamsAggr(const cuphyPdschStatPrms_t* static_params)
{
    if (static_params == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 cuphyPdschStatPrms_t params pointer is null");
        return;
    }
    int n_cells = static_params->nCells;
    NVLOGI_FMT(TAG, "{} L2: PDSCH Static parameters for {} cells", __FUNCTION__, n_cells);

    // Parameters common across all cells
    NVLOGI_FMT(TAG, "{} L2: read_TB_CRC:           {:4d}", __FUNCTION__, static_params->read_TB_CRC);
    NVLOGI_FMT(TAG, "{} L2: full_slot_processing:  {:4d}", __FUNCTION__, static_params->full_slot_processing);
    NVLOGI_FMT(TAG, "{} L2: stream_priority:       {:4d}", __FUNCTION__, static_params->stream_priority);
    NVLOGI_FMT(TAG, "{} L2: nMaxCellsPerSlot:      {:4d}", __FUNCTION__, static_params->nMaxCellsPerSlot);
    NVLOGI_FMT(TAG, "{} L2: nMaxUesPerCellGroup:   {:4d}", __FUNCTION__, static_params->nMaxUesPerCellGroup);
    NVLOGI_FMT(TAG, "{} L2: nMaxCBsPerTB:          {:4d}", __FUNCTION__, static_params->nMaxCBsPerTB);
    NVLOGI_FMT(TAG, "{} L2: nMaxPrb:               {:4d}", __FUNCTION__, static_params->nMaxPrb);
    NVLOGI_FMT(TAG, "{} L2: enableBatchedMemcpy    {:4d}", __FUNCTION__, static_params->enableBatchedMemcpy);

    // Cell specific parameters
    cuphyCellStatPrm_t* cell_static_params = static_params->pCellStatPrms;
    cuphyPdschDbgPrms_t* cell_dbg_params   = static_params->pDbg;
    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: Cell {}", __FUNCTION__, cell_id);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: phyCellId:      {:4d}", __FUNCTION__, cell_static_params[cell_id].phyCellId);
        NVLOGI_FMT(TAG, "{} L2: nRxAnt:         {:4d}", __FUNCTION__, cell_static_params[cell_id].nRxAnt);
        NVLOGI_FMT(TAG, "{} L2: nTxAnt:         {:4d}", __FUNCTION__, cell_static_params[cell_id].nTxAnt);
        NVLOGI_FMT(TAG, "{} L2: nPrbUlBwp:      {:4d}", __FUNCTION__, cell_static_params[cell_id].nPrbUlBwp);
        NVLOGI_FMT(TAG, "{} L2: nPrbDlBwp:      {:4d}", __FUNCTION__, cell_static_params[cell_id].nPrbDlBwp);
        NVLOGI_FMT(TAG, "{} L2: mu:             {:4d}", __FUNCTION__, cell_static_params[cell_id].mu);
        // Debug fields
        NVLOGI_FMT(TAG, "{} L2: \nDBG:", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: pCfgFileName:             {}", __FUNCTION__, cell_dbg_params[cell_id].pCfgFileName);
        NVLOGI_FMT(TAG, "{} L2: checkTbSize:              {}", __FUNCTION__, cell_dbg_params[cell_id].checkTbSize);
        NVLOGI_FMT(TAG, "{} L2: refCheck:                 {}", __FUNCTION__, cell_dbg_params[cell_id].refCheck);
        NVLOGI_FMT(TAG, "{} L2: cfgIdenticalLdpcEncCfgs:  {}", __FUNCTION__, cell_dbg_params[cell_id].cfgIdenticalLdpcEncCfgs);
        NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);
    }
}


void printPdschDynamicCellGroupAggr(const cuphyPdschCellGrpDynPrm_t* cell_group_params)
{
    if (cell_group_params == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 cuphyPdschCellGrpDynPrm_t params pointer is null");
        return;
    }

    // Print information for all cells
    int n_cells = cell_group_params->nCells;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} cells", __FUNCTION__, n_cells);

    cuphyPdschCellDynPrm_t* pdsch_cell_dynamic_params = cell_group_params->pCellPrms;

    for (int cell_id = 0; cell_id < n_cells; cell_id++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: Cell {}", __FUNCTION__, cell_id);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        NVLOGI_FMT(TAG, "{} L2: cellPrmStatIdx:  {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: cellPrmDynIdx:   {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].cellPrmDynIdx);

        NVLOGI_FMT(TAG, "{} L2: slotNum:         {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].slotNum);
        NVLOGI_FMT(TAG, "{} L2: nCsiRsPrms:      {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].nCsiRsPrms);
        NVLOGI_FMT(TAG, "{} L2: csiRsPrmsOffset: {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].csiRsPrmsOffset);
        NVLOGI_FMT(TAG, "{} L2: testModel:       {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].testModel);

        //NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in UE group instead. If so, the values here will be 0.
        NVLOGI_FMT(TAG, "{} L2: pdschStartSym:   {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].pdschStartSym);
        NVLOGI_FMT(TAG, "{} L2: nPdschSym:       {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].nPdschSym);
        NVLOGI_FMT(TAG, "{} L2: dmrsSymLocBmsk:  {:4d}", __FUNCTION__, pdsch_cell_dynamic_params[cell_id].dmrsSymLocBmsk);
        NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);
    }

    // Print information for all UE groups
    int n_ue_groups = cell_group_params->nUeGrps;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} UE groups", __FUNCTION__, n_ue_groups);
    cuphyPdschUeGrpPrm_t* pdsch_ue_group_dynamic_params = cell_group_params->pUeGrpPrms;

    for (int ue_group_id = 0; ue_group_id <  n_ue_groups; ue_group_id++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: UE group {}", __FUNCTION__, ue_group_id);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        NVLOGI_FMT(TAG, "{} L2: resourceAlloc:   {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].resourceAlloc);
        // Only print rbBitmap in case of resource allocation type 0
        if (pdsch_ue_group_dynamic_params[ue_group_id].resourceAlloc == 0) {
            std::stringstream tmp_rb_bitmap;
            tmp_rb_bitmap << __FUNCTION__ << " L2: rbBitmap: {";
            for (int i = 0; i <  MAX_RBMASK_BYTE_SIZE; i++) {
                if (i != 0) tmp_rb_bitmap << ", ";
                tmp_rb_bitmap << std::to_string(pdsch_ue_group_dynamic_params[ue_group_id].rbBitmap[i]);
            }
            tmp_rb_bitmap << "}";
            NVLOGI_FMT(TAG, "{}", tmp_rb_bitmap.str().c_str());
        }

        NVLOGI_FMT(TAG, "{} L2: startPrb:        {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].startPrb);
        NVLOGI_FMT(TAG, "{} L2: nPrb:            {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].nPrb);

        //NB: pdschStartSym, nPdschSym and dmrsSymLocBmsk may be provided in the cell instead, and thus be common across all UE groups. If so, the values here will be 0.
        NVLOGI_FMT(TAG, "{} L2: pdschStartSym:   {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].pdschStartSym);
        NVLOGI_FMT(TAG, "{} L2: nPdschSym:       {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].nPdschSym);
        NVLOGI_FMT(TAG, "{} L2: dmrsSymLocBmsk:  {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].dmrsSymLocBmsk);

        NVLOGI_FMT(TAG, "{} L2: nUes:            {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].nUes);

        // Indices to pUePrms array
        std::stringstream tmp_ue_idxs;
        tmp_ue_idxs << __FUNCTION__ << " L2: pUePrmIdxs: {";
        for (int i = 0; i <  pdsch_ue_group_dynamic_params[ue_group_id].nUes; i++) {
            if (i != 0) tmp_ue_idxs << ", ";
            tmp_ue_idxs << std::to_string(pdsch_ue_group_dynamic_params[ue_group_id].pUePrmIdxs[i]);
        }
        tmp_ue_idxs << "}";
        NVLOGI_FMT(TAG, "{}", tmp_ue_idxs.str().c_str());

        //pDmrsDynPrm -> pointer to DMRS info
        NVLOGI_FMT(TAG, "{} L2: nDmrsCdmGrpsNoData:  {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].pDmrsDynPrm->nDmrsCdmGrpsNoData);

        //pCellPrm -> pointer to parent group's dynamic params.
        NVLOGI_FMT(TAG, "{} L2: UE group's parent static cell Idx:    {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].pCellPrm->cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: UE group's parent dynamic cell Idx:   {:4d}", __FUNCTION__, pdsch_ue_group_dynamic_params[ue_group_id].pCellPrm->cellPrmDynIdx);

        NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);
    }
    NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);

    // Print information for all UEs
    int n_ues = cell_group_params->nUes;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} UEs", __FUNCTION__, n_ues);

    cuphyPdschUePrm_t* pdsch_ue_dynamic_params = cell_group_params->pUePrms;

    for (int ue_id = 0; ue_id <  n_ues; ue_id++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: UE {}", __FUNCTION__, ue_id);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        NVLOGI_FMT(TAG, "{} L2: scid:         {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].scid);
        NVLOGI_FMT(TAG, "{} L2: dmrsScrmId:   {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].dmrsScrmId);
        NVLOGI_FMT(TAG, "{} L2: nUeLayers:    {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].nUeLayers);
        NVLOGI_FMT(TAG, "{} L2: dmrsPortBmsk:   0x{:08x}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].dmrsPortBmsk);
        NVLOGI_FMT(TAG, "{} L2: refPoint:     {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].refPoint);
        NVLOGI_FMT(TAG, "{} L2: BWPStart:     {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].BWPStart);
        NVLOGI_FMT(TAG, "{} L2: beta_drms:     {}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].beta_dmrs);
        NVLOGI_FMT(TAG, "{} L2: beta_qam:      {}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].beta_qam);
        NVLOGI_FMT(TAG, "{} L2: rnti:         {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].rnti);
        NVLOGI_FMT(TAG, "{} L2: dataScramId:  {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].dataScramId);

        NVLOGI_FMT(TAG, "{} L2: nCw:          {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].nCw);

        // Indices to pCwPrms array
        std::stringstream tmp_cw_idxs;
        tmp_cw_idxs << __FUNCTION__ << " L2: pCwIdxs: {";
        for (int i = 0; i <  pdsch_ue_dynamic_params[ue_id].nCw; i++) {
            if (i != 0) tmp_cw_idxs << ", ";
            tmp_cw_idxs << std::to_string(pdsch_ue_dynamic_params[ue_id].pCwIdxs[i]);
        }
        tmp_cw_idxs << "}";
        NVLOGI_FMT(TAG, "{}", tmp_cw_idxs.str().c_str());

        // Precoding parameters
        NVLOGI_FMT(TAG, "{} L2: enablePrcdBf: {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].enablePrcdBf);
        NVLOGI_FMT(TAG, "{} L2: pmwPrmIdx:    {:4d}", __FUNCTION__, pdsch_ue_dynamic_params[ue_id].pmwPrmIdx);

        // Add pointer to parent UE group
        bool found = false;
        int ue_group_id = 0;
        while ((!found) && (ue_group_id < n_ue_groups)) {
            if (pdsch_ue_dynamic_params[ue_id].pUeGrpPrm == &pdsch_ue_group_dynamic_params[ue_group_id]) {
              found = true;
          }
          ue_group_id += 1;
        }
        NVLOGI_FMT(TAG, "{} L2: parent UE group: {:4d}", __FUNCTION__, found ? (ue_group_id - 1): -1);
    }
    NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);

    // Print information for all CWs
    int n_cws = cell_group_params->nCws;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} CWs", __FUNCTION__, n_cws);

    cuphyPdschCwPrm_t* pdsch_cw_dynamic_params = cell_group_params->pCwPrms;

    for (int cw_id = 0; cw_id <  n_cws; cw_id++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: CW {}", __FUNCTION__, cw_id);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        NVLOGI_FMT(TAG, "{} L2: mcsTableIndex:         {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].mcsTableIndex);
        NVLOGI_FMT(TAG, "{} L2: mcsIndex:              {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].mcsIndex);
        NVLOGI_FMT(TAG, "{} L2: targetCodeRate:        {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].targetCodeRate);
        NVLOGI_FMT(TAG, "{} L2: qamModOrder:           {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].qamModOrder);
        NVLOGI_FMT(TAG, "{} L2: rv:                    {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].rv);
        NVLOGI_FMT(TAG, "{} L2: tbStartOffset:         {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].tbStartOffset);
        NVLOGI_FMT(TAG, "{} L2: tbSize:                {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].tbSize);

        NVLOGI_FMT(TAG, "{} L2: n_PRB_LBRM:            {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].n_PRB_LBRM);
        NVLOGI_FMT(TAG, "{} L2: maxLayers:             {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].maxLayers);
        NVLOGI_FMT(TAG, "{} L2: maxQm:                 {:4d}", __FUNCTION__, pdsch_cw_dynamic_params[cw_id].maxQm);

        // FIXME pointer to parent UE
        bool found = false;
        int ue_id = 0;
        while ((!found) && (ue_id < n_ues)) {
            if (pdsch_cw_dynamic_params[cw_id].pUePrm == &pdsch_ue_dynamic_params[ue_id]) {
              found = true;
          }
          ue_id += 1;
        }
        NVLOGI_FMT(TAG, "{} L2: parent UE:             {:4d}", __FUNCTION__, found ? (ue_id - 1): -1);
    }
    NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);

    // Print information for all CSI-RS
    int n_csi_rs = cell_group_params->nCsiRsPrms;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} CSI-RS params", __FUNCTION__, n_csi_rs);

    _cuphyCsirsRrcDynPrm* pdsch_csi_rs_dynamic_params = cell_group_params->pCsiRsPrms;

    for (int csirs_idx = 0; csirs_idx <  n_csi_rs; csirs_idx++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: CSI RS {}", __FUNCTION__, csirs_idx);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        NVLOGI_FMT(TAG, "{} L2: startRb:               {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].startRb);
        NVLOGI_FMT(TAG, "{} L2: nRb:                   {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].nRb);
        NVLOGI_FMT(TAG, "{} L2: freqDomain:            {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].freqDomain);
        NVLOGI_FMT(TAG, "{} L2: row:                   {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].row);
        NVLOGI_FMT(TAG, "{} L2: symbL0:                {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].symbL0);
        NVLOGI_FMT(TAG, "{} L2: symbL1:                {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].symbL1);
        NVLOGI_FMT(TAG, "{} L2: freqDensity:           {:4d}", __FUNCTION__, pdsch_csi_rs_dynamic_params[csirs_idx].freqDensity);
    }
    NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);

    // Print precoding information
    int n_precoding_matrices = cell_group_params->nPrecodingMatrices;
    NVLOGI_FMT(TAG, "{} L2: PDSCH cell group dynamic parameters for {} precoding params", __FUNCTION__, n_precoding_matrices);

    cuphyPmW_t* pdsch_pmw_dynamic_params = cell_group_params->pPmwPrms;
    for (int precoding_idx = 0; precoding_idx < n_precoding_matrices; precoding_idx++) {
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);
        NVLOGI_FMT(TAG, "{} L2: Precoding matrix {}", __FUNCTION__, precoding_idx);
        NVLOGI_FMT(TAG, "{} L2: ------------------------------------", __FUNCTION__);

        uint8_t n_ports = pdsch_pmw_dynamic_params[precoding_idx].nPorts;
        NVLOGI_FMT(TAG, "{} L2: nPorts:                 {:4d}", __FUNCTION__, n_ports);
        NVLOGI_FMT(TAG, "{} L2: matrix:", __FUNCTION__);
        //max. rows printed even if not relevant; these extra rows should contain {0, 0}
        for (int layer_idx = 0; layer_idx < MAX_DL_LAYERS_PER_TB; layer_idx++) { // Not all layers used
            std::stringstream tmp_pmw_matrix_row;
            tmp_pmw_matrix_row.precision(5);
            tmp_pmw_matrix_row << __FUNCTION__ << " L2: ";
            for (int port_idx = 0; port_idx < n_ports; port_idx++) {
                __half2 val = pdsch_pmw_dynamic_params[precoding_idx].matrix[layer_idx * n_ports + port_idx];
                tmp_pmw_matrix_row << "{" << std::fixed << (float(val.x)) << ", ";
                tmp_pmw_matrix_row << std::fixed << (float(val.y)) << "} ";
            }
            NVLOGI_FMT(TAG, "{}", tmp_pmw_matrix_row.str().c_str());
        }
        NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);
    }
    NVLOGI_FMT(TAG, "{} L2: ", __FUNCTION__);
}

void printPdschDynPrmsAggr(const cuphyPdschDynPrms_t* dynamic_params)
{
    if (dynamic_params == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 cuphyPdschDynPrms_t params pointer is null");
        return;
    }
   NVLOGI_FMT(TAG, "{} L2: PDSCH dynamic parameters",  __FUNCTION__);

    const cuphyPdschCellGrpDynPrm_t* cell_group_params = dynamic_params->pCellGrpDynPrm;
    NVLOGI_FMT(TAG, "{} L2: procModeBmsk:       {:4d}",  __FUNCTION__, dynamic_params->procModeBmsk);

    //PDSCH data input buffers
    if (dynamic_params->pDataIn == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 pDataIn is null");
        return;
    }
    NVLOGI_FMT(TAG, "{} L2: pDataIn.pBufferType:   {}",  __FUNCTION__, ((dynamic_params->pDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER"));
    std::stringstream tmp_pTbInput;
    tmp_pTbInput << __FUNCTION__ << " L2: pDataIn.pTbInput:    {";
    for (int i = 0; i <  cell_group_params->nCells; i++) {
        if (i != 0) tmp_pTbInput << ", ";
        tmp_pTbInput << reinterpret_cast<void*>(dynamic_params->pDataIn->pTbInput[i]);
    }
    tmp_pTbInput << "}";
    NVLOGI_FMT(TAG, "{}", tmp_pTbInput.str().c_str());

    //PDSCH TB-CRC
    if (dynamic_params->pTbCRCDataIn == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 pTbCRCDataIn is null");
        return;
    }

    NVLOGI_FMT(TAG, "{} L2: pTbCRCDataIn.pBufferType:   {}", __FUNCTION__, ((dynamic_params->pTbCRCDataIn->pBufferType == cuphyPdschDataIn_t::CPU_BUFFER) ? "CPU_BUFFER" : "GPU_BUFFER"));
    if (dynamic_params->pTbCRCDataIn->pTbInput == nullptr)
    {
        NVLOGI_FMT(TAG, "{} L2: pTbCRCDataIn.pTbInput:      {}", __FUNCTION__, reinterpret_cast<void*>(dynamic_params->pTbCRCDataIn->pTbInput));
    } else {
        std::stringstream tmp_pTbCRCInput;
        tmp_pTbCRCInput << __FUNCTION__ << " L2: pTbCRCDataIn.pTbInput:    {";
        for (int i = 0; i <  cell_group_params->nCells; i++) {
            if (i != 0) tmp_pTbCRCInput << ", ";
            tmp_pTbCRCInput << reinterpret_cast<void*>(dynamic_params->pTbCRCDataIn->pTbInput[i]);
        }
        tmp_pTbCRCInput << "}";
        NVLOGI_FMT(TAG, "{}", tmp_pTbCRCInput.str().c_str());
    }

    // PDSCH data output buffers
    if (dynamic_params->pDataOut == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 pDataOut is null");
        return;
    }
    std::stringstream tmp_pTDataTx_addr;
    tmp_pTDataTx_addr << __FUNCTION__ << " L2: pDataOut.pTDataTx addr.:    {"; //TODO could add descriptor info too
    for (int i = 0; i <  cell_group_params->nCells; i++) {
        if (i != 0) tmp_pTDataTx_addr << ", ";
        tmp_pTDataTx_addr << dynamic_params->pDataOut->pTDataTx[i].pAddr;
    }
    tmp_pTDataTx_addr << "}";
    NVLOGI_FMT(TAG, "{}", tmp_pTDataTx_addr.str().c_str());

    printPdschDynamicCellGroupAggr(const_cast<cuphyPdschCellGrpDynPrm_t*>(cell_group_params));
}

float PhyPdschAggr::getPdschH2DCopyTime(const uint8_t slot) {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (!pdctx->enable_prepone_h2d_cpy) {
        return 0.0f;
    }
    const uint8_t slot_index = slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS;
    return 1000.0f * GetCudaEventElapsedTime(pdctx->get_event_pdsch_tb_cpy_start(slot_index),
                                             pdctx->get_event_pdsch_tb_cpy_complete(slot_index),
                                             __func__, getId());
}
