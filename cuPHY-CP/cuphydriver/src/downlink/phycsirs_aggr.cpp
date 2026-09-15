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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 12) // "DRV.CSIRS"

#include "phychannel.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "phycsirs_aggr.hpp"
#define getName(var)  #var

void printParameters(const cuphyCsirsRrcDynPrm_t* l2)
{
#if 0
    NVLOGI_FMT(TAG, "------------------------- CSI_RS PARAMS ----------------------------");
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->startRb), l2->startRb);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->nRb), l2->nRb);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->freqDomain), l2->freqDomain);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->row), l2->row);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->symbL0), l2->symbL0);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->symbL1), l2->symbL1);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->freqDensity), l2->freqDensity);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->scrambId), l2->scrambId);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->idxSlotInFrame), l2->idxSlotInFrame);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->csiType), l2->csiType);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10d}", __FUNCTION__, getName(l2->cdmType), l2->cdmType);
    NVLOGI_FMT(TAG, "{} L2: {:<25s}:{:10f}", __FUNCTION__, getName(l2->beta), l2->beta);
    NVLOGI_FMT(TAG, "--------------------------------------------------------------------");
#endif
}

void printParameters(cuphyCsirsDynPrms_t dyn_params)
{
    NVLOGI_FMT(TAG, "------------------------- CSI_RS PARAMS ----------------------------");
    NVLOGI_FMT(TAG, "{} L2: nCells:  {:4d}", __FUNCTION__, dyn_params.nCells);
    
    for(uint32_t i=0; i < dyn_params.nCells; i++)
    {
        NVLOGI_FMT(TAG, "{} L2: cell-id[{}].rrcParamsOffset = {} ", __FUNCTION__, i, dyn_params.pCellParam[i].rrcParamsOffset);
        NVLOGI_FMT(TAG, "{} L2: cell-id[{}].nRrcParams = {} ", __FUNCTION__, i, dyn_params.pCellParam[i].nRrcParams);
    }
}

PhyCsiRsAggr::PhyCsiRsAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx * _mpsCtx
    ) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    cuphyStatus_t status;
    PhyDriverCtx* pdctx                  = StaticConversion<PhyDriverCtx>(pdh).get();

    mf.init(_pdh, std::string("PhyCsiRs"), sizeof(PhyCsiRsAggr));
    cuphyMf.init(_pdh, std::string("cuphyCsirsTx"), 0);

    channel_type = slot_command_api::channel_type::CSI_RS;
    channel_name.assign("CSI_RS");

    // Added temporarily to simplify cuPHY API graph mode addition.
    // Setting to zero ensures the processing mode is streams unless changed.
    std::memset(&dyn_params, 0, sizeof(cuphyCsirsDynPrms_t));

    try
    {
        handle = std::make_unique<cuphyCsirsTxHndl_t>();
        mf.addCpuRegularSize(sizeof(cuphyCsirsTxHndl_t));

        if(pdctx->isValidation())
            ref_check = true;

    }
    PHYDRIVER_CATCH_THROW_EXCEPTIONS();

    cell_id_list.clear();

    std::memset(&stat_params, 0, sizeof(cuphyCsirsStatPrms_t));
    stat_params.pCellStatPrms = NULL;
    stat_params.nCells = 0;
    stat_params.pOutInfo = &cuphy_tracker;

    static_params_cell.clear();

    dyn_params_num = 0;
    data_out.pTDataTx = nullptr;
};
int PhyCsiRsAggr::createPhyObj()
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

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
    stat_params.nMaxCellsPerSlot = static_params_cell.size();
    stat_params.nCells = static_params_cell.size();
    stat_params.pCellStatPrms = static_cast<cuphyCellStatPrm_t*>(static_params_cell.data());

    if(stat_params.nMaxCellsPerSlot > PDSCH_MAX_CELLS_PER_CELL_GROUP)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " CSI_RS is max cells per slot = {},  But max cell per group is = {} ", stat_params.nMaxCellsPerSlot, PDSCH_MAX_CELLS_PER_CELL_GROUP);
        return -1;
    }
    if(stat_params.nMaxCellsPerSlot == pdctx->getCellGroupNum())
    {
        data_out.pTDataTx = (cuphyTensorPrm_t*) calloc(stat_params.nMaxCellsPerSlot, sizeof(cuphyTensorPrm_t));
        cuphyStatus_t  status = cuphyCreateCsirsTx(handle.get(), &stat_params);
        std::string cuphy_ch_create_name = "cuphyCreateCsirsTx";            
        checkPhyChannelObjCreationError(status,cuphy_ch_create_name);
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerCsirsTx(*handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(stat_params.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();
    }
    

    return 0;
}
PhyCsiRsAggr::~PhyCsiRsAggr(){
    if(data_out.pTDataTx)
    {
        free(data_out.pTDataTx);
    }

    if(*handle)    
    {
        cuphyStatus_t status = cuphyDestroyCsirsTx(*handle);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyDestroyCsirsTx = {}", cuphyGetErrorString(status));
            // PHYDRIVER_THROW_EXCEPTIONS(-1, "cuphyDestroyCsirsTx");
        }
    }
};

slot_command_api::csirs_params* PhyCsiRsAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->csirs.get();
}

int PhyCsiRsAggr::setup(const std::vector<DLOutputBuffer *>& aggr_dlbuf, const std::vector<Cell *>& aggr_cell_list)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    cuphyStatus_t status;

    setCtx();

    slot_command_api::csirs_params* pparms = getDynParams();
    if(pparms == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error getting Dynamic CSI_RS params from Slot Command");
        return -1;
    }
    dyn_params_num = pparms->nCsirsRrcDynPrm;

    dyn_params.cuStream = s_channel;
    dyn_params.pRrcDynPrm = pparms->csirsList;
    dyn_params.nCells = pparms->nCells;
    dyn_params.pCellParam = pparms->cellInfo;
    dyn_params.pDataOut = &data_out;
    dyn_params.procModeBmsk = pdctx->getEnableDlCuphyGraphs() ? CSIRS_PROC_MODE_GRAPHS : CSIRS_PROC_MODE_STREAMS;

    auto pm_group = getPmGroup();
    if (pm_group != nullptr) {
        dyn_params.nPrecodingMatrices = pm_group->nPmCsirs;
        dyn_params.pPmwParams = pm_group->csirs_list.data();
    } else {
        dyn_params.nPrecodingMatrices = 0;
        dyn_params.pPmwParams = nullptr;
    }

    int slotBufferIdx = -1 ;
    /*for(auto& csirs_pdu: pparms->csirsList)
    {
        printParameters(&csirs_pdu);
    }*/
    for(int idx = 0; idx < aggr_dlbuf.size(); idx++)
    {
        bool found_buf = false;
        int cell_idx = 0;
        for(cell_idx = 0; cell_idx < pparms->nCells; cell_idx++ )
        {
            if(pparms->phy_cell_index_list[cell_idx] == aggr_cell_list[idx]->getPhyId())
            {
                found_buf = true;
                break;
            }
        }
        if(found_buf)
        {
            dyn_params.pCellParam[cell_idx].slotBufferIdx = ++slotBufferIdx;
            dyn_params.pDataOut->pTDataTx[slotBufferIdx].desc = aggr_dlbuf[idx]->getTensor()->desc().handle();
            dyn_params.pDataOut->pTDataTx[slotBufferIdx].pAddr = aggr_dlbuf[idx]->getBufD();
        }
        /*else
        {
            NVLOGI_FMT(TAG, "PhyCsirsAggr: Cell {} has no PDSCH", aggr_cell_list[idx]->getPhyId());
        }*/
    }
    // printParameters(dyn_params);
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_setup, s_channel));
    }
    if(dyn_params_num > 0)
    {
        status = cuphySetupCsirsTx(*handle, &dyn_params);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            const int sfn  = aggr_slot_params->si->sfn_;
            const int slot = aggr_slot_params->si->slot_;
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error in cuphySetupCsirsTx(): {}. Will not call cuphyRunCsirsTx(). May be L2 misconfiguration.", sfn, slot, cuphyGetErrorString(status));
            MemtraceDisableScope md;
            CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
            return -1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
    }

    return 0;
}

int PhyCsiRsAggr::run()
{
    cuphyStatus_t status;
    int ret=0;

    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_run, s_channel));
    }
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR) && (dyn_params_num > 0))
    {
        status = cuphyRunCsirsTx(*handle);
        if (status != CUPHY_STATUS_SUCCESS) {
            std::cerr << "Error! cuphyRunCsirsTx(): " << cuphyGetErrorString(status) << std::endl;
            ret=-1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_run, s_channel));
    }

    return ret;
}

int PhyCsiRsAggr::callback()
{
    PhyDriverCtx*                       pdctx    = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell*                               cell_ptr = pdctx->getCellById(cell_id);
    slot_command_api::dl_slot_callbacks dl_cb;
#if 0
    if(pdctx->getDlCb(dl_cb))
    {
        NVLOGD_FMT(TAG, "Calling DL callback for cell {}", cell_id);
        auto pdsch = getDynParams();
        dl_cb.callback_fn(pdsch->cell_index, current_slot_params->si);
    }

    uint16_t tb_count = dyn_params.pCellGrpDynPrm->nCws;
    cell_ptr->updateMetric(CellMetric::kPdschTxTbTotal, tb_count);

    uint64_t tb_bytes = 0;
    for(uint16_t tb_idx = 0; tb_idx < tb_count; tb_idx++)
    {
        tb_bytes += dyn_params.pCellGrpDynPrm->pCwPrms[tb_idx].tbSize;
    }

    cell_ptr->updateMetric(CellMetric::kPdschTxBytesTotal, tb_bytes);
    cell_ptr->updateMetric(CellMetric::kPdschNrOfUesPerSlot, dyn_params.pCellGrpDynPrm->nUes);
    cell_ptr->updateMetric(CellMetric::kPdschProcessingTime, this->getGPURunTime());
#endif
    return 0;
}

static int checkReference(float2 * expTensorData, __half2 * outTensorData, uint32_t n_p, uint32_t n_t, uint32_t n_f)
{
    float err_threshold = 0.001;
    int  err_cnt = 0;

    NVLOGC_FMT(TAG, "\nComparing matlab and gpu test vectors: ");
    NVLOGC_FMT(TAG, "\nTensor dimension: {} x {} x {}", n_f, n_t, n_p);
    NVLOGC_FMT(TAG, "\n-------------------------------------");

    for(int k = 0; k < n_p; k++)  {
        for(int i = 0; i < n_t; i++)  {
            for(int j = 0; j < n_f; j++)  {
                __half2        outData      = outTensorData[k * n_t * n_f + i * n_f + j]; // from GPU
                float2         outDataFloat;
                outDataFloat.x = float (outData.x);
                outDataFloat.y = float (outData.y);
                float2         expDataFloat = expTensorData[k * n_t * n_f + i * n_f + j]; // from matlab
                float err_x = abs(outDataFloat.x - expDataFloat.x);
                float err_y = abs(outDataFloat.y - expDataFloat.y);
                if (err_x > err_threshold || err_y > err_threshold)
                {
                    // NVLOGC_FMT(TAG, "tfSignal mismatch at p = {:2d} t = {:2d} f = {:4d}, matlab = ({:6.3f}, {:6.3f}) vs gpu = ({:6.3f}, {:6.3f})",
                    //     k, i, j, expDataFloat.x, expDataFloat.y, outDataFloat.x, outDataFloat.y);
                    err_cnt++;
                }
                // else if(abs(expDataFloat.x) > err_threshold || abs(expDataFloat.y) > err_threshold)
                // {
                //     NVLOGC_FMT(TAG, "tfSignal match at p = {:2d} t = {:2d} f = {:4d}, matlab = ({:6.3f}, {:6.3f}) vs gpu = ({:6.3f}, {:6.3f})",
                //         k, i, j, expDataFloat.x, expDataFloat.y, outDataFloat.x, outDataFloat.y);
                // }
            }
        }
    }
    if (err_cnt == 0) {
        NVLOGC_FMT(TAG, "====> Test PASS");
        return 0;
    }
    else {
        NVLOGC_FMT(TAG, "====> Test FAIL. Found {} mismatched symbols", err_cnt);
        return 1;
    }
}

int PhyCsiRsAggr::validate() {
    int ret = 0;
    if(ref_check == true)
    {
        cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc> h_csirs_out_tensor(tx_tensor->layout());
        h_csirs_out_tensor.convert(*tx_tensor, s_channel);
        ret = checkReference(ref_output.addr(),
                                h_csirs_out_tensor.addr(),
                                h_csirs_out_tensor.layout().dimensions()[2] /* Number of ports */,
                                h_csirs_out_tensor.layout().dimensions()[1] /* time domain */,
                                h_csirs_out_tensor.layout().dimensions()[0] /* freq. domain */
                            );
    }

    return ret;
}

uint16_t PhyCsiRsAggr::getMinPrb()
{
    uint16_t min_prb = UINT16_MAX;
    auto params = getDynParams();
    auto& nzpcsirsList = params->csirsList;
    for(int idx = 0; idx < params->nCsirsRrcDynPrm; idx++)
    {
        min_prb = std::min(nzpcsirsList[idx].startRb, min_prb);
    }
    return min_prb;
}

uint16_t PhyCsiRsAggr::getMaxPrb()
{
    uint16_t max_prb = UINT16_MAX;
    auto params = getDynParams();
    auto& nzpcsirsList = params->csirsList;
    for(int idx = 0; idx < params->nCsirsRrcDynPrm; idx++)
    {
        max_prb = std::min(static_cast<uint16_t>(nzpcsirsList[idx].startRb + nzpcsirsList[idx].nRb), max_prb);
    }
    return max_prb;
}
