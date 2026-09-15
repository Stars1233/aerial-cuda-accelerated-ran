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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 14) // "DRV.PDCCH_DL"

#include "phypdcch_aggr.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuphy.h"
#include "cuphy_internal.h"
#include "memtrace.h"

using namespace std;
using namespace cuphy;
using namespace hdf5hpp;

PhyPdcchAggr::PhyPdcchAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx*          _mpsCtx,
    slot_command_api::channel_type ch) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    cuphyStatus_t status;
    bool          ref_check              = false;
    bool          identical_ldpc_configs = true;
    PhyDriverCtx* pdctx                  = StaticConversion<PhyDriverCtx>(pdh).get();

    mf.init(_pdh, std::string("PhyPdcchAggr"), sizeof(PhyPdcchAggr));
    cuphyMf.init(_pdh, std::string("cuphyPdcchTx"), 0);

    channel_type = ch;
    if (channel_type == slot_command_api::channel_type::PDCCH_DL)
        channel_name.assign("PDCCH_DL");
    else if (channel_type == slot_command_api::channel_type::PDCCH_UL)
        channel_name.assign("PDCCH_UL");
    else {
        channel_name.assign("ERR_CHANNEL");
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Error channel type: {}", __FUNCTION__, +channel_type);
    }

    std::memset(&static_params, 0, sizeof(cuphyPdcchStatPrms_t));
    static_params.pOutInfo = &cuphy_tracker;
    static_params.kernelSelOption = PDCCH_ALL;
    static_params.delayUs = 0;

    handle = nullptr;
};

PhyPdcchAggr::~PhyPdcchAggr(){
    if(handle) {
        cuphyDestroyPdcchTx(handle);
        free(DataIn.pDciInput);
        free(DataOut.pTDataTx);
    }
};

int PhyPdcchAggr::createPhyObj()
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

    /*
     * Create cuPHY object only if the desider number of cells
     * has been activated into cuphydriver.
     */
    if(static_params_cell.size() == pdctx->getCellGroupNum())
    {
        setCtx();

        static_params.nMaxCellsPerSlot = static_params_cell.size();
        auto status = cuphyCreatePdcchTx(&handle, &static_params);
        std::string cuphy_ch_create_name = "cuphyCreatePdcchTx";            
        checkPhyChannelObjCreationError(status,cuphy_ch_create_name);
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPdcchTx(handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(static_params.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();

        dyn_params.cuStream = s_channel;
        dyn_params.procModeBmsk = pdctx->getEnableDlCuphyGraphs() ? PDCCH_PROC_MODE_GRAPHS : PDCCH_PROC_MODE_STREAMS;
        DataIn.pDciInput = (uint8_t*) calloc(static_params_cell.size() * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL * CUPHY_PDCCH_MAX_DCIS_PER_CORESET * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, sizeof(uint8_t));
        DataIn.pBufferType = cuphyPdcchDataIn_t::CPU_BUFFER;
        dyn_params.pDataIn = &DataIn;
        DataOut.pTDataTx = (cuphyTensorPrm_t*) calloc(static_params_cell.size(), sizeof(cuphyTensorPrm_t));
        dyn_params.pDataOut = &DataOut;

        gDev->synchronizeStream(s_channel);
    }
    else if(static_params_cell.size() > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    return 0;
}


slot_command_api::pdcch_group_params* PhyPdcchAggr::getDynParams()
{
    if(channel_type == slot_command_api::channel_type::PDCCH_DL || channel_type == slot_command_api::channel_type::PDCCH_UL)
        return aggr_slot_params->cgcmd->pdcch.get();
    else {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Error channel type: {}", __FUNCTION__, +channel_type);
        return nullptr;
    }
}

int PhyPdcchAggr::setup(const std::vector<Cell *> &aggr_cell_list, const std::vector<DLOutputBuffer *> &aggr_dlbuf)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    cuphyStatus_t status;
    int dci_dyn_idx = 0;
    int dci_in_idx = 0;

    slot_command_api::pdcch_group_params* params = getDynParams();
    if (params == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error! No dyn PDCCH found");
        return -EINVAL;
    }

    setCtx();

    dyn_params.nDci = params->csets_group.nDcis;
    dyn_params.pDciPrms = params->csets_group.dcis.data();
    dyn_params.nCoresets = params->csets_group.nCoresets;
    //NVLOGI_FMT(TAG, "PhyPdcchAggr::setup - nCoresets {} nDci {}",dyn_params.nCoresets,dyn_params.nDci);

    for(int idx_cs = 0; idx_cs < dyn_params.nCoresets; idx_cs++) {
        for(int idx_buf = 0; idx_buf < aggr_dlbuf.size(); idx_buf++) {
            if(aggr_cell_list[idx_buf]->getPhyId() == params->phy_cell_index_list[idx_cs]) {
                params->csets_group.csets[idx_cs].slotBufferIdx = idx_buf;
                //NVLOGI_FMT(TAG, "PhyPdcchAggr::setup - slotBufferIdx,idx_buf {} idx_cs {} PCI {} DLBuffer {} ",
                    //idx_buf, idx_cs, aggr_cell_list[idx_buf]->getPhyId(), aggr_dlbuf[idx_buf]->getId());
                NVLOGD_FMT(TAG, "{}:{} PDCCH testModel={}",__func__,__LINE__, params->csets_group.csets[idx_cs].testModel);
                break;
            }
        }
    }

    dyn_params.pCoresetDynPrm = params->csets_group.csets.data();
    dyn_params.nCells = aggr_cell_list.size();

    // DataIn.pDciInput = params->csets_group.payloads.data();
    for(dci_dyn_idx = 0, dci_in_idx = 0; dci_dyn_idx < params->csets_group.nDcis; dci_dyn_idx++) {
        memcpy(&(DataIn.pDciInput[dci_in_idx]), &(params->csets_group.payloads[dci_dyn_idx]), CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES);
        dci_in_idx += CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES;
    }

    for(int idx=0; idx < static_params.nMaxCellsPerSlot && idx < aggr_dlbuf.size(); idx++) {
        dyn_params.pDataOut->pTDataTx[idx].desc  = aggr_dlbuf[idx]->getTensor()->desc().handle();
        dyn_params.pDataOut->pTDataTx[idx].pAddr = aggr_dlbuf[idx]->getBufD();
    }

    auto pm_group = getPmGroup();
    if (pm_group != nullptr) {
        dyn_params.nPrecodingMatrices = pm_group->nPmPdcch;
        dyn_params.pPmwParams = pm_group->pdcch_list.data();
    } else {
        dyn_params.nPrecodingMatrices = 0;
        dyn_params.pPmwParams = nullptr;
    }

    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_setup, s_channel));
    }

    status = cuphySetupPdcchTx(handle, &dyn_params);
    if (status != CUPHY_STATUS_SUCCESS) {
        const int sfn  = aggr_slot_params->si->sfn_;
        const int slot = aggr_slot_params->si->slot_;
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error in cuphySetupPdcchTx(): {}. Will not call cuphyRunPdcchTx(). May be L2 misconfiguration.", sfn, slot, cuphyGetErrorString(status));
        CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
        return -1;
    }

    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
    }

    return 0;
}

int PhyPdcchAggr::run()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    cuphyStatus_t status;
    int ret=0;

    setCtx();

    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_run, s_channel));
    }
    if ((getSetupStatus() == CH_SETUP_DONE_NO_ERROR)) {
        status = cuphyRunPdcchTx(handle, 0);
        if (status != CUPHY_STATUS_SUCCESS) {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyRunPdcchTx(): {}", cuphyGetErrorString(status));
            ret=-1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_run, s_channel));
    }

    NVLOGD_FMT(TAG, "{} run finished", channel_name.c_str());
    return ret;
}

int PhyPdcchAggr::callback()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    slot_command_api::dl_slot_callbacks dl_cb;

    /*
     * Do we need PDCCH DL callback specific?
     * if yes, need to extend this callback mechanism
     */
    if(pdctx->getDlCb(dl_cb))
    {
        // NVSLOGD(TAG) << "Calling DL callback for cell " << cell_ptr->getPhyId();
        // auto pdsch = getDynParams();
        // dl_cb.callback_fn(pdsch->cell_index, current_slot_params->si);
    }

    return 0;
}
