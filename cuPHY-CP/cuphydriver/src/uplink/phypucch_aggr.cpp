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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 22) // "DRV.PUCCH"

#include "phypucch_aggr.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"

PhyPucchAggr::PhyPucchAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t*     _s_channels,
    MpsCtx * _mpsCtx) :
    PhyChannel(_pdh, _gDev,0, _s_channels[0], _mpsCtx)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    channel_type = slot_command_api::channel_type::PUCCH;
    channel_name.assign("PUCCH");

    cuphy::disable_hdf5_error_print();

    mf.init(_pdh, std::string("PhyPucchAggr"), sizeof(PhyPucchAggr)); //FIXME not updated
    cuphyMf.init(_pdh, std::string("cuphyPucchRx"), 0);

#if 0
    std::string dbgFile = std::string("PUCCH_Debug") + std::to_string(id) + std::string(".h5");
    debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(dbgFile.c_str())));
    dbg_params.pOutFileName = dbgFile.c_str();
#else
    dbg_params.enableDynApiLogging  = 0;
    dbg_params.enableStatApiLogging = 0;
    dbg_params.pOutFileName = nullptr;
#endif

    pf0OutBuffer = std::move(cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>(UL_MAX_CELLS_PER_SLOT * MAX_PUCCH_F0_UCIS));
    pf1OutBuffer = std::move(cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>(UL_MAX_CELLS_PER_SLOT* MAX_PUCCH_F1_UCIS));

    // Calculate PUCCH user buffer size based on F2/F3 UCI limits
    // With ENABLE_64C: MAX_CELLS_PER_SLOT=40, each cell supports 24 F2+F3 UCIs
    // Total: 40 * 24 = 960 (matches CUPHY_PUCCH_F2_MAX_UCI and CUPHY_PUCCH_F3_MAX_UCI)
#ifdef ENABLE_64C
    pucch_users = 960;  // MAX_CELLS_PER_SLOT (40) * 24 UCIs per cell
#else
    pucch_users = 480;  // MAX_CELLS_PER_SLOT (20) * 24 UCIs per cell
#endif
    NVLOGI_FMT(TAG, "PhyPucchAggr: Allocating buffers for pucch_users={}", pucch_users);
    
    bPucchF2OutOffsets = std::move(cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc>(pucch_users));
    bPucchF3OutOffsets = std::move(cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc>(pucch_users));
    bCrcFlags = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pucch_users));
    bRssi     = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(pucch_users));
    bSnr      = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(pucch_users));
    bInterf   = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(pucch_users));
    bTaEst    = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(pucch_users));
    bRsrp     = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(pucch_users));
    bNumCsi2Bits = std::move(cuphy::buffer<uint16_t, cuphy::pinned_alloc>(pucch_users));
    bHarqDetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pucch_users));
    bCsiP1DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pucch_users));
    bCsiP2DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pucch_users));
    
    uint32_t  nUciPayloadBytes = MAX_N_PRBS_SUPPORTED * MAX_N_BBU_LAYERS_PUCCH_SUPPORTED * OFDM_SYMBOLS_PER_SLOT * CUPHY_N_TONES_PER_PRB * CUPHY_QAM_256;
    bUciPayloads = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciPayloadBytes));

    procModeBmsk = PUCCH_PROC_MODE_FULL_SLOT;

    DataOut.pF0UcisOut = pf0OutBuffer.addr();
    DataOut.pF1UcisOut = pf1OutBuffer.addr();
    DataOut.pUciPayloads = bUciPayloads.addr();
    DataOut.pCrcFlags = bCrcFlags.addr();
    DataOut.pRssi = bRssi.addr();
    DataOut.pSinr = bSnr.addr();
    DataOut.pInterf = bInterf.addr();
    DataOut.pTaEst = bTaEst.addr();
    DataOut.pRsrp = bRsrp.addr();
    DataOut.pNumCsi2Bits = bNumCsi2Bits.addr();
    DataOut.HarqDetectionStatus = bHarqDetectionStatus.addr();
    DataOut.CsiP1DetectionStatus = bCsiP1DetectionStatus.addr();
    DataOut.CsiP2DetectionStatus = bCsiP2DetectionStatus.addr();
    DataOut.pPucchF2OutOffsets = bPucchF2OutOffsets.addr();
    DataOut.pPucchF3OutOffsets = bPucchF3OutOffsets.addr();

    clearUciFlags();

    DataIn.pTDataRx = (cuphyTensorPrm_t*) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyTensorPrm_t));
    for(int idx = 0; idx < UL_MAX_CELLS_PER_SLOT; idx++)
    {
        pucch_data_rx_desc[idx] = {CUPHY_C_16F, static_cast<int>(ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB), static_cast<int>(OFDM_SYMBOLS_PER_SLOT), static_cast<int>(MAX_AP_PER_SLOT), cuphy::tensor_flags::align_tight};
        DataIn.pTDataRx[idx].desc = pucch_data_rx_desc[idx].handle();
        DataIn.pTDataRx[idx].pAddr = nullptr;
    }

    dyn_params.cuStream       = s_channel;
    dyn_params.procModeBmsk   = procModeBmsk;
    dyn_params.pDataIn        = &DataIn;
    dyn_params.pDataOut       = &DataOut;
    dyn_params.cpuCopyOn = 1;
    
    statusOut = {cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    dyn_params.pStatusOut = &statusOut;

    handle = nullptr;

    launch_kernel_warmup(s_channel);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);

    gDev->synchronizeStream(s_channel);

    batch_handle = nullptr;
};

PhyPucchAggr::~PhyPucchAggr()
{
    if(handle)
        cuphyDestroyPucchRx(handle);

#if 0
    dbg_params.pOutFileName = nullptr;
    debugFile.reset();
#endif
};

slot_command_api::pucch_params* PhyPucchAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->pucch.get();
}

int PhyPucchAggr::createPhyObj() 
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    setCtx();

    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    pucchCellStatPrmsVec.reserve(cellCount);
    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        int tmp_cell_id = cell_ptr->getPhyId();

        // Add only the new cells here
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

        // set static cell paramaters
        cellStatPrm.phyCellId = cell_ptr->getPhyId();
        cellStatPrm.nRxAnt    = cell_ptr->getRxAnt();
        cellStatPrm.nRxAntSrs = cell_ptr->getRxAntSrs();
        cellStatPrm.nTxAnt    = cell_ptr->getTxAnt();
        cellStatPrm.nPrbUlBwp = cell_ptr->getPrbUlBwp();
        cellStatPrm.nPrbDlBwp = cell_ptr->getPrbDlBwp();
        cellStatPrm.mu        = cell_ptr->getMu();

        pucchCellStatPrmsVec.push_back(*(cell_ptr->getPhyStatic()->pPucchCellStatPrms));
        cellStatPrm.pPucchCellStatPrms = &(pucchCellStatPrmsVec[pucchCellStatPrmsVec.size()-1]);
        static_params_cell.push_back(cellStatPrm);
    }

    static_params.nMaxCells            = static_params_cell.size();
    static_params.nMaxCellsPerSlot     = static_params_cell.size();
    static_params.pCellStatPrms        = static_cast<cuphyCellStatPrm_t*>(static_params_cell.data());
    static_params.pDbg                 = &dbg_params;
    static_params.polarDcdrListSz      = pdctx->getPuxchPolarDcdrListSz();
    static_params.enableUlRxBf         = pdctx->getmMIMO_enable();
    static_params.enableBatchedMemcpy  = pdctx->getUseBatchedMemcpy();
    static_params.pOutInfo             = &cuphy_tracker;

    /*
     * Create cuPHY object only if the desider number of cells
     * has been activated into cuphydriver.
     */
    if(static_params_cell.size() == pdctx->getCellGroupNum())
    {
        cuphyStatus_t createStatus = cuphyCreatePucchRx(&handle, &static_params, s_channel);
        std::string cuphy_ch_create_name = "cuphyCreatePucchRx";            
        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPucchRx(handle);
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

void PhyPucchAggr::print_pucch_params(const slot_command_api::pucch_params * pucch_params) {
    NVLOGI_FMT(TAG, "================== PUCCH Parameters ==================");
    NVLOGI_FMT(TAG, "{}: Number of cells: {}", __FUNCTION__, pucch_params->grp_dyn_pars.nCells);
    
    //Cell level parameters
    for(uint16_t i = 0; i < pucch_params->grp_dyn_pars.nCells; i++) {
        const auto& cell = pucch_params->grp_dyn_pars.pCellPrms[i];
        NVLOGI_FMT(TAG, "{}: Cell[{}]: cellPrmStatIdx={}, cellPrmDynIdx={}, slotNum={}, pucchHoppingId={}", 
                   __FUNCTION__, i, cell.cellPrmStatIdx, cell.cellPrmDynIdx, cell.slotNum, cell.pucchHoppingId);
    }
    
    //PUCCH format 0 UCI parameters
    if(pucch_params->grp_dyn_pars.nF0Ucis > 0) {
        NVLOGI_FMT(TAG, "{}: Format 0 UCIs: {}", __FUNCTION__, pucch_params->grp_dyn_pars.nF0Ucis);
        for(uint16_t i = 0; i < pucch_params->grp_dyn_pars.nF0Ucis; i++) {
            const auto& uci = pucch_params->grp_dyn_pars.pF0UciPrms[i];
            NVLOGI_FMT(TAG, "{}: F0[{}]: RNTI={}, startPrb={}, bwpStart={}, startSym={}, nSym={}, cellPrmDynIdx={}, initialCyclicShift={}, srFlag={}, bitLenHarq={}, freqHopFlag={}, secondHopPrb={}, groupHopFlag={}, sequenceHopFlag={}", 
                       __FUNCTION__, i, uci.rnti, uci.startPrb, uci.bwpStart, uci.startSym, uci.nSym, uci.cellPrmDynIdx, uci.initialCyclicShift, uci.srFlag, uci.bitLenHarq, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag);
        }
    }
    
    //PUCCH format 1 UCI parameters
    if(pucch_params->grp_dyn_pars.nF1Ucis > 0) {
        NVLOGI_FMT(TAG, "{}: Format 1 UCIs: {}", __FUNCTION__, pucch_params->grp_dyn_pars.nF1Ucis);
        for(uint16_t i = 0; i < pucch_params->grp_dyn_pars.nF1Ucis; i++) {
            const auto& uci = pucch_params->grp_dyn_pars.pF1UciPrms[i];
            NVLOGI_FMT(TAG, "{}: F1[{}]: RNTI={}, startPrb={}, bwpStart={}, startSym={}, nSym={}, cellPrmDynIdx={}, initialCyclicShift={}, timeDomainOccIdx={}, srFlag={}, bitLenHarq={}, freqHopFlag={}, secondHopPrb={}, groupHopFlag={}, sequenceHopFlag={}", 
                       __FUNCTION__, i, uci.rnti, uci.startPrb, uci.bwpStart, uci.startSym, uci.nSym, uci.cellPrmDynIdx, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag);
        }
    }
    
    //PUCCH format 2 UCI parameters
    if(pucch_params->grp_dyn_pars.nF2Ucis > 0) {
        NVLOGI_FMT(TAG, "{}: Format 2 UCIs: {}", __FUNCTION__, pucch_params->grp_dyn_pars.nF2Ucis);
        for(uint16_t i = 0; i < pucch_params->grp_dyn_pars.nF2Ucis; i++) {
            const auto& uci = pucch_params->grp_dyn_pars.pF2UciPrms[i];
            NVLOGI_FMT(TAG, "{}: F2[{}]: RNTI={}, startPrb={}, prbSize={}, bwpStart={}, startSym={}, nSym={}, cellPrmDynIdx={}, bitLenHarq={}, bitLenCsiPart1={}, freqHopFlag={}, secondHopPrb={}, dataScramblingId={}, DmrsScramblingId={}, nBitsCsi2={}", 
                       __FUNCTION__, i, uci.rnti, uci.startPrb, uci.prbSize, uci.bwpStart, uci.startSym, uci.nSym, uci.cellPrmDynIdx, uci.bitLenHarq, uci.bitLenCsiPart1, uci.freqHopFlag, uci.secondHopPrb, uci.dataScramblingId, uci.DmrsScramblingId, uci.nBitsCsi2);
        }
    }
    
    //PUCCH format 3 UCI parameters
    if(pucch_params->grp_dyn_pars.nF3Ucis > 0) {
        NVLOGI_FMT(TAG, "{}: Format 3 UCIs: {}", __FUNCTION__, pucch_params->grp_dyn_pars.nF3Ucis);
        for(uint16_t i = 0; i < pucch_params->grp_dyn_pars.nF3Ucis; i++) {
            const auto& uci = pucch_params->grp_dyn_pars.pF3UciPrms[i];
            NVLOGI_FMT(TAG, "{}: F3[{}]: RNTI={}, startPrb={}, prbSize={}, bwpStart={}, startSym={}, nSym={}, cellPrmDynIdx={}, bitLenHarq={}, bitLenCsiPart1={}, freqHopFlag={}, secondHopPrb={}, groupHopFlag={}, sequenceHopFlag={}, dataScramblingId={}, AddDmrsFlag={}, nBitsCsi2={}", 
                       __FUNCTION__, i, uci.rnti, uci.startPrb, uci.prbSize, uci.bwpStart, uci.startSym, uci.nSym, uci.cellPrmDynIdx, uci.bitLenHarq, uci.bitLenCsiPart1, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.dataScramblingId, uci.AddDmrsFlag, uci.nBitsCsi2);
        }
    }
    
    NVLOGI_FMT(TAG, "=======================================================");
}

int PhyPucchAggr::setup(
    const std::vector<Cell *>& aggr_cell_list,
    const std::vector<ULInputBuffer *>& aggr_ulbuf_st1,
    cudaStream_t stream
)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    t_ns          t1    = Time::nowNs();
    //TBD: consider a list of buffers, one per cell. Here support is limited to a single cell
    slot_command_api::oran_slot_ind oran_ind     = getOranAggrSlotIndication();

    setCtx();

    slot_command_api::pucch_params* pparms = getDynParams();
    if(pparms == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error getting Dynamic PUCCH params from Slot Command");
        return -1;
    }
    dyn_params.pCellGrpDynPrm = &pparms->grp_dyn_pars;
    //print_pucch_params(pparms);
   
    for(int idx = 0; idx < aggr_ulbuf_st1.size(); idx++)
    {
        int count = -1;
        if(aggr_ulbuf_st1[idx] != nullptr)
        {
            auto phyCellId = aggr_cell_list[idx]->getPhyId();
            for(uint32_t dyn_idx=0; dyn_idx < dyn_params.pCellGrpDynPrm->nCells; dyn_idx++)
            {
                if(static_params_cell[dyn_params.pCellGrpDynPrm->pCellPrms[dyn_idx].cellPrmStatIdx].phyCellId == phyCellId)
                {
                    count = dyn_params.pCellGrpDynPrm->pCellPrms[dyn_idx].cellPrmDynIdx;
                    break;
                }
            }
            if(count != -1)
            {
                DataIn.pTDataRx[count].pAddr = aggr_ulbuf_st1[idx]->getBufD();
                //NVLOGI_FMT(TAG, "PhyPucchAggr::setup - Cell {} cellPrmDynIdx {} ULBuffer {} at index {}",
                    //phyCellId,count,aggr_ulbuf_st1[idx]->getId(),idx);
            }
            //else
                //NVLOGI_FMT(TAG, "PhyPucchAggr::setup - Cell {} has no PUCCH",phyCellId);
        }
    }

    // Set processing mode bitmask
    procModeBmsk = pdctx->getEnableUlCuphyGraphs() ? PUCCH_PROC_MODE_FULL_SLOT_GRAPHS : PUCCH_PROC_MODE_FULL_SLOT;
    dyn_params.procModeBmsk = procModeBmsk;
    dyn_params.pDbg = &dbg_params;

    printParameters(&static_params, dyn_params.pCellGrpDynPrm);

    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    dyn_params.cuStream = stream;

    t_ns t2     = Time::nowNs();
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, dyn_params.cuStream));
    }

    cuphyStatus_t status;
    if(pdctx->getUseGreenContexts() == 0)
    {
        status = cuphySetupPucchRx(handle, &dyn_params, batch_handle);
    }
    else
    {
        MemtraceDisableScope md;
        status = cuphySetupPucchRx(handle, &dyn_params, batch_handle);
    }
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphySetupPucchRx returned error {}", status);
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, dyn_params.cuStream));
        return -1;
    }

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, dyn_params.cuStream));
    }

    return 0;
}

int PhyPucchAggr::run()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret=0;
    
    setCtx();

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, dyn_params.cuStream));
    }
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR))
    {
        auto runStatus = cuphyRunPucchRx(handle, procModeBmsk);
        if (runStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyRunPucchRx returned error {}", runStatus);
            ret=-1;
        }
    }
    t_ns t3 = Time::nowNs();
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, dyn_params.cuStream));
    }

    return ret;
}

int PhyPucchAggr::validate()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret = 0;

    if(pdctx->isValidation())
    {
    }

    return ret;
}

int PhyPucchAggr::callback()
{
    PhyDriverCtx*                       pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    slot_command_api::ul_slot_callbacks ul_cb;

    if(pdctx->getUlCb(ul_cb))
    {
        NVLOGI_FMT(TAG, "Calling PUCCH UL callback for cell {}", cell_id);
        auto pucch = getDynParams();
        ul_cb.uci_cb_fn2(ul_cb.uci_cb_fn2_context, *(aggr_slot_params->si), *pucch, DataOut);
    }
    
    //cell_ptr->updateMetric(CellMetric::kPucchProcessingTime, this->getGPURunTime());

    return 0;
}

int PhyPucchAggr::clearUciFlags()
{
    /*
    if(aggr_slot_params){
        struct slot_command_api::slot_indication* si = aggr_slot_params->si;
        NVLOGD_FMT(TAG,"PhyPucchAggr::{} SFN {}.{}",__func__,si->sfn_,si->slot_);        
    }*/
    memset(bHarqDetectionStatus.addr(), 2, sizeof(uint8_t) * pucch_users);
    memset(bCsiP1DetectionStatus.addr(), 2, sizeof(uint8_t) * pucch_users);
    memset(bCsiP2DetectionStatus.addr(), 2, sizeof(uint8_t) * pucch_users);
    memset(bCrcFlags.addr(), 1, sizeof(uint8_t) * pucch_users);    
    
    return 1;
}

void PhyPucchAggr::updatePhyCellId(uint16_t phyCellId_old,uint16_t phyCellId_new)
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

void printPucchUciPrm(uint16_t nUcis, cuphyPucchUciPrm_t *pUciPrms)
{
    for (uint i = 0; i < nUcis; i++)
    {
        cuphyPucchUciPrm_t& uci = pUciPrms[i];
        NVLOGI_FMT(TAG, "printParameters cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
            i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
        NVLOGI_FMT(TAG, "printParameters cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
            i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        NVLOGI_FMT(TAG, "printParameters cuphyPucchUciPrm_t[{}] bwpStart={}", i, uci.bwpStart);
    }
}

void PhyPucchAggr::printParameters(const cuphyPucchStatPrms_t* stat, const cuphyPucchCellGrpDynPrm_t* l2)
{
#if 0
    if (stat != nullptr && stat->pCellStatPrms != nullptr)
    {
        cuphyCellStatPrm_t* st = stat->pCellStatPrms;
        NVLOGI_FMT(TAG, "{} cuphyCellStatPrm_t: phyCellId ={} nTxAnt={} nRxAnt={} nPrbUlBwp={}, nPrbDlBwp={} mu={}",
            __FUNCTION__, st->phyCellId, st->nTxAnt, st->nRxAnt, st->nPrbUlBwp,
            st->nPrbDlBwp, st->mu);
    }

    if (l2 != nullptr)
    {
        NVLOGI_FMT(TAG, "{} nCells={} nF0Ucis={} nF1Ucis={} nF2Ucis={} nF3Ucis={} nF4Ucis={}", __FUNCTION__, l2->nCells, l2->nF0Ucis, l2->nF1Ucis, l2->nF2Ucis, l2->nF3Ucis, l2->nF4Ucis);
        for (uint i=0 ; i < l2->nCells && (l2->pCellPrms !=nullptr); i++)
        {
            cuphyPucchCellDynPrm_t& cellPrm =  l2->pCellPrms[i];
            NVLOGI_FMT(TAG, "{} cuphyPucchCellDynPrm_t[{}] cellPrmDynIdx={} cellPrmStatIdx={} pucchHoppingId={} slotNum={}",
                __FUNCTION__, i, cellPrm.cellPrmDynIdx, cellPrm.cellPrmStatIdx, cellPrm.pucchHoppingId, cellPrm.slotNum);
        }

        printPucchUciPrm(l2->nF0Ucis, l2->pF0UciPrms);
        printPucchUciPrm(l2->nF1Ucis, l2->pF1UciPrms);
        printPucchUciPrm(l2->nF2Ucis, l2->pF2UciPrms);
        printPucchUciPrm(l2->nF3Ucis, l2->pF3UciPrms);
        printPucchUciPrm(l2->nF4Ucis, l2->pF4UciPrms);

/*
        for (uint i = 0; i <l2->nF0Ucis; i++)
        {
            cuphyPucchUciPrm_t& uci = l2->pF0UciPrms[i];
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
                __FUNCTION__, i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
                __FUNCTION__, i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        }

        for (uint i = 0; i <l2->nF1Ucis; i++)
        {
            cuphyPucchUciPrm_t& uci = l2->pF1UciPrms[i];
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
                __FUNCTION__, i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
                __FUNCTION__, i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        }

        for (uint i = 0; i <l2->nF2Ucis; i++)
        {
            cuphyPucchUciPrm_t& uci = l2->pF2UciPrms[i];
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
                __FUNCTION__, i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
                __FUNCTION__, i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        }

        for (uint i = 0; i <l2->nF3Ucis; i++)
        {
            cuphyPucchUciPrm_t& uci = l2->pF3UciPrms[i];
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
                __FUNCTION__, i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
                __FUNCTION__, i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        }

        for (uint i = 0; i <l2->nF4Ucis; i++)
        {
            cuphyPucchUciPrm_t& uci = l2->pF4UciPrms[i];
            NVLOGI_FMT(TAG, "{} bwpStart={} bwpSize={}", __FUNCTION__, uci.bwpStart, uci.bwpSize);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] uciOutputIdx={} formatType={} rnti={} multiSlotTxIndicator={} pi2Bpsk={} startPrb={} startSym={} nSym={}",
                __FUNCTION__, i, uci.uciOutputIdx, uci.formatType, uci.rnti, uci.multiSlotTxIndicator, uci.pi2Bpsk, uci.startPrb, uci.startSym, uci.nSym);
            NVLOGI_FMT(TAG, "{} cuphyPucchUciPrm_t[{}] freqHopFlag={} secondHopPrb={} groupHopFlag={} sequenceHopFlag={} initialCyclicShift={} timeDomainOccIdx={} srFlag={} bitLenHarq={}",
                __FUNCTION__, i, uci.freqHopFlag, uci.secondHopPrb, uci.groupHopFlag, uci.sequenceHopFlag, uci.initialCyclicShift, uci.timeDomainOccIdx, uci.srFlag, uci.bitLenHarq);
        }
*/
    }
#endif
}
