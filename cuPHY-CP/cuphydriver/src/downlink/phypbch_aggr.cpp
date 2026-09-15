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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 13) // "DRV.PBCH"
#include "phypbch_aggr.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "cuphy.h"
#include "cuphy_internal.h"


template <typename T, int N>
inline void printArray(T* arr)
{
    NVLOGC_FMT(TAG, "[ ");
    for(int i = 0; i < N; i++)
    {
        NVLOGC_FMT(TAG, "0x{:x} ", +arr[i]);
    }
    NVLOGC_FMT(TAG, "]");
}

PhyPbchAggr::PhyPbchAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx * _mpsCtx) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    
    mf.init(_pdh, std::string("PhyPbchAggr"), sizeof(PhyPbchAggr));
    cuphyMf.init(_pdh, std::string("cuphySsbTx"), 0);

    channel_type        = slot_command_api::channel_type::PBCH;
    channel_name.assign("PBCH");

    ssbDynPrms.cuStream = s_channel;
    ssbDynPrms.nCells = 0;
    ssbDynPrms.nSSBlocks = 0;
    ssbStatParams.nMaxCellsPerSlot = 0;
    ssbStatParams.pOutInfo = &cuphy_tracker;
    ssbStatParams.pDbgPrms = nullptr;
    handle = nullptr;
    DataOut.pTDataTx = nullptr;

}

PhyPbchAggr::~PhyPbchAggr(){
    if(DataOut.pTDataTx)
    {
        free(DataOut.pTDataTx);
    }
    if(handle) {
        cuphyDestroySsbTx(handle);
        //free(DataIn.pMibInput); // shouldn't be calloc'ed in the first place, as ptr reassigned from group_params
    }
};

int PhyPbchAggr::createPhyObj() 
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
        
        if(cell_id_list.size() > 0)
        {
            auto it = std::find_if(
                cell_id_list.begin(), cell_id_list.end(),
                [&tmp_cell_id](uint16_t cell_id) { return (cell_id == tmp_cell_id); }
            );

            if(it != cell_id_list.end())
                continue;
        }
        cell_id_list.push_back(tmp_cell_id);
    }
    ssbStatParams.nMaxCellsPerSlot = cell_id_list.size();
    if(ssbStatParams.nMaxCellsPerSlot > PDSCH_MAX_CELLS_PER_CELL_GROUP)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " PBCH is max cells per slot = {},  But max cell per group is = {} ", ssbStatParams.nMaxCellsPerSlot, PDSCH_MAX_CELLS_PER_CELL_GROUP);
        return -1;
    }
    if(ssbStatParams.nMaxCellsPerSlot == pdctx->getCellGroupNum())
    {
        setCtx();

        //DataIn.pMibInput = (uint32_t *)calloc(CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT * ssbStatParams.nMaxCellsPerSlot, sizeof(uint32_t));
        DataIn.pBufferType = cuphySsbDataIn_t::CPU_BUFFER;
        ssbDynPrms.pDataIn = &DataIn;
        DataOut.pTDataTx = (cuphyTensorPrm_t*) calloc(ssbStatParams.nMaxCellsPerSlot, sizeof(cuphyTensorPrm_t));
        ssbDynPrms.pDataOut = &DataOut;
        cuphyStatus_t createStatus = cuphyCreateSsbTx(&handle, &ssbStatParams);
        std::string cuphy_ch_create_name = "cuphyCreateSsbTx";            
        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerSsbTx(handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(ssbStatParams.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();
        gDev->synchronizeStream(s_channel);
    }
    else if(ssbStatParams.nMaxCellsPerSlot > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    return 0;
}
int PhyPbchAggr::setup(const std::vector<DLOutputBuffer *>& aggr_dlbuf, const std::vector<Cell *>& aggr_cell_list)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    setCtx();
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_setup, s_channel));
    }

    slot_command_api::pbch_group_params* group_params = getDynParams();
    uint8_t last_slot_buff_idx = 0;
    uint32_t mib_index = 0;
    uint16_t first_NID = 0;
    

    
    for (uint i = 0; i < group_params->nSsbBlocks; i++)
    {
        auto& pbch_dyn_data = group_params->pbch_dyn_block_params[i];
        auto& cell_index = pbch_dyn_data.cell_index;
        auto& pbch_per_cell_data = group_params->pbch_dyn_cell_params[cell_index];
        last_slot_buff_idx = pbch_per_cell_data.slotBufferIdx;
        bool found_buf = false;
        for(int idx = 0; idx < aggr_dlbuf.size(); idx++)
        {
            if(group_params->phy_cell_index_list[last_slot_buff_idx] == aggr_cell_list[idx]->getPhyId())
            {
                ssbDynPrms.pDataOut->pTDataTx[last_slot_buff_idx].desc = aggr_dlbuf[idx]->getTensor()->desc().handle();
                ssbDynPrms.pDataOut->pTDataTx[last_slot_buff_idx].pAddr = aggr_dlbuf[idx]->getBufD();
                NVLOGD_FMT(TAG, "NID = {}, SS block index = {} Output_buf = {} ",  pbch_per_cell_data.NID, pbch_dyn_data.blockIndex, ssbDynPrms.pDataOut->pTDataTx[last_slot_buff_idx].pAddr);
                found_buf = true;
                break;
            }
        }
        if(!found_buf)
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,"NID = {}, SS block index = {} Output_buf not found",pbch_per_cell_data.NID, pbch_dyn_data.blockIndex);
    }

    ssbDynPrms.procModeBmsk = pdctx->getEnableDlCuphyGraphs() ? SSB_PROC_MODE_GRAPHS : SSB_PROC_MODE_STREAMS;
    ssbDynPrms.nCells = group_params->ncells;
    ssbDynPrms.pPerCellSsbDynParams = group_params->pbch_dyn_cell_params.data();
    ssbDynPrms.nSSBlocks = group_params->nSsbBlocks;
    ssbDynPrms.pPerSsBlockParams = group_params->pbch_dyn_block_params.data();
    DataIn.pMibInput = group_params->pbch_dyn_mib_data.data();
    auto pm_group = getPmGroup();
    if (pm_group != nullptr) {
        ssbDynPrms.nPrecodingMatrices = pm_group->nPmPbch;
        ssbDynPrms.pPmwParams = pm_group->ssb_list.data();
    } else {
        ssbDynPrms.nPrecodingMatrices = 0;
        ssbDynPrms.pPmwParams = nullptr;
    }

    auto status = cuphySetupSsbTx(handle, &ssbDynPrms);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        const int sfn  = aggr_slot_params->si->sfn_;
        const int slot = aggr_slot_params->si->slot_;
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error in cuphySetupSsbTx(): {}. Will not call cuphyRunSsbTx(). May be L2 misconfiguration.", sfn, slot, cuphyGetErrorString(status));
        {
            MemtraceDisableScope md;
            CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
        }
        return -1;
    }

    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_setup, s_channel));
    }
    return 0;
}

int PhyPbchAggr::run()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int ret=0;
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(start_run, s_channel));
    }
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR))
    {
        cuphyStatus_t status = cuphyRunSsbTx(handle, 0);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyRunSsbTx(): {}", cuphyGetErrorString(status));
            ret=-1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_DRIVER_CHECK(cuEventRecord(end_run, s_channel));
    }
    return ret;
}

int PhyPbchAggr::callback()
{
    return 0;
}

slot_command_api::pbch_group_params* PhyPbchAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->pbch.get();
}



void printParameters(PhyDriverCtx* pdctx,slot_command_api::SSTxParams* l2)
{
#if 0
    if (l2 == nullptr || tv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Either L2 params and or TV params is null");
        return;
    }
    NVLOGC_FMT(TAG, "{} SS Block TX pipeline Parameters: ", __FUNCTION__);
    NVLOGC_FMT(TAG, "\n-------------------------------");
    NVLOGC_FMT(TAG, "{} L2 nHF:        {:4d}", __FUNCTION__, l2->nHF );
    NVLOGC_FMT(TAG, "{} L2 NID:        {:4d}", __FUNCTION__, l2->NID );
    NVLOGC_FMT(TAG, "{} L2 Lmax:       {:4d}", __FUNCTION__, l2->Lmax);
    NVLOGC_FMT(TAG, "{} L2 blockIndex: {:4d}", __FUNCTION__, l2->blockIndex);
    NVLOGC_FMT(TAG, "{} L2 f0:         {:4d}", __FUNCTION__, l2->f0);
    NVLOGC_FMT(TAG, "{} L2 t0:         {:4d}", __FUNCTION__, l2->t0);
    NVLOGC_FMT(TAG, "{} L2 k_SSB:      {:4d}", __FUNCTION__, l2->k_SSB);
    NVLOGC_FMT(TAG, "{} L2 SFN:        {:4d}", __FUNCTION__, l2->SFN);
    NVLOGC_FMT(TAG, "{} L2 nF:         {:4d}", __FUNCTION__, l2->nF);
    NVLOGC_FMT(TAG, "{} L2 nT:         {:4d}", __FUNCTION__, l2->nT);
#endif
}

