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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 34) // "DRV.ULBFW"

#include "phychannel.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "phyulbfw_aggr.hpp"
#define getName(var)  #var

//#define ULBFW_H5DUMP


PhyUlBfwAggr::PhyUlBfwAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx * _mpsCtx
    ) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    cuphyStatus_t status;
    PhyDriverCtx* pdctx                  = StaticConversion<PhyDriverCtx>(pdh).get();

    mf.init(_pdh, std::string("PhyUlBfw"), sizeof(PhyUlBfwAggr));
    cuphyMf.init(_pdh, std::string("cuphyUlBfwTx"), 0);

    channel_type = slot_command_api::channel_type::BFW; //TODO :Assign DLBFW equivalent
    channel_name.assign("UL_BFW");//TODO :Assign DLBFW equivalent

    data_in.pChEstInfo = (cuphySrsChEstBuffInfo_t*) calloc(slot_command_api::MAX_SRS_CHEST_BUFFERS, sizeof(cuphySrsChEstBuffInfo_t));
    for(int i = 0; i < slot_command_api::MAX_SRS_CHEST_BUFFERS; i++)
    {
        #if 0
        ulBfwChEstBuffInfo[i] = std::move(cuphy::tensor_device(nullptr, CUPHY_C_32F, CV_NUM_PRBG, 
                                                        CV_NUM_GNB_ANT,
                                                        CV_NUM_UE_LAYER, 
                                                        cuphy::tensor_flags::align_tight));
        data_in.pChEstInfo[i].tChEstBuffer.desc = ulBfwChEstBuffInfo[i].desc().handle();
        #endif
        data_in.pChEstInfo[i].tChEstBuffer.pAddr = nullptr;
    }    

    std::memset(&dyn_params, 0, sizeof(cuphyBfwDynPrms_t));

    try
    {
        handle = std::make_unique<cuphyBfwTxHndl_t>();
        mf.addCpuRegularSize(sizeof(cuphyBfwTxHndl_t));

        if(pdctx->isValidation())
            ref_check = true;

    }
    PHYDRIVER_CATCH_THROW_EXCEPTIONS();

};
int PhyUlBfwAggr::createPhyObj()
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    setCtx();

    //TODO : Change the constant values for BFW once available
    stat_params.lambda = 0;
    stat_params.nMaxGnbAnt = CV_NUM_GNB_ANT;
    stat_params.nMaxPrbGrps = slot_command_api::MAX_NUM_PRGS_DBF;
    stat_params.nMaxTotalLayers = slot_command_api::MAX_PUSCH_UE_GROUPS * CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP;
    stat_params.nMaxUeGrps = slot_command_api::MAX_PUSCH_UE_GROUPS;
    stat_params.compressBitwidth = 9;
    stat_params.beta = pdctx->get_bfw_beta_prescaler();
    stat_params.bfwPowerNormAlg_selector = pdctx->get_bfw_power_normalization_alg_selector();
    stat_params.useKernelCopy = pdctx->gpuCommEnabledViaCpu() ? 1 : 0;    
    NVLOGD_FMT(TAG, "bfwPowerNormAlg_selector: {}", stat_params.bfwPowerNormAlg_selector);

    stat_params.pDbg = &ulBfwDbgPrms;
    stat_params.pStatDbg = &ulBfwStatDbgPrms;
    stat_params.pOutInfo = &cuphy_tracker;

#ifdef ULBFW_H5DUMP
    //std::string dbg_output_file = std::string("ULBFW_Debug") + std::to_string(id) + std::string(".h5");
    //debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(dbg_output_file.c_str())));
    //stat_params.pDbg->pOutFileName             = dbg_output_file.c_str();
    std::string stat_dbg_output_file = std::string("ULBFW_Stat_Debug") + std::to_string(id) + std::string(".h5");
    debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(stat_dbg_output_file.c_str())));
    stat_params.pStatDbg->pOutFileName             = stat_dbg_output_file.c_str();
    stat_params.pStatDbg->enableApiLogging = 0;
#else
    stat_params.pDbg->pOutFileName = nullptr;
    stat_params.pStatDbg->pOutFileName = nullptr;
    stat_params.pStatDbg->enableApiLogging = 0;
#endif

    stat_params.enableBatchedMemcpy = pdctx->getUseBatchedMemcpy();

    cuphyStatus_t  status = cuphyCreateBfwTx(handle.get(), &stat_params, s_channel);
    std::string cuphy_ch_create_name = "cuphyCreateBfwTx";            
    checkPhyChannelObjCreationError(status,cuphy_ch_create_name);

    //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPrachRx(handle);
    pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(stat_params.pOutInfo->pMemoryFootprint);
    //pCuphyTracker->printMemoryFootprint();

    return 0;
}

PhyUlBfwAggr::~PhyUlBfwAggr(){
    
    cuphyStatus_t status = cuphyDestroyBfwTx(*(handle.get()));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyDestroyBfwTx = {}", cuphyGetErrorString(status));
        // PHYDRIVER_THROW_EXCEPTIONS(-1, "cuphyDestroyBfwTx");
    }
};

//TODO :Create equivalent for BFW once available
slot_command_api::bfw_params* PhyUlBfwAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->bfw.get();
}

int PhyUlBfwAggr::setup(std::vector<Cell *>& aggr_cell_list)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    cuphyStatus_t status;

    setCtx();

    slot_command_api::bfw_params* pparms = getDynParams();
    if(pparms == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error getting Dynamic DLBFW params from Slot Command");
        return -1;
    }


    dyn_params.pDynPrm = &pparms->bfw_dyn_info;
    dyn_params.cuStream = s_channel;
    NVLOGD_FMT(TAG, "UL prevUeGrpChEstInfoBufIdx={}", pparms->prevUeGrpChEstInfoBufIdx);
    for(int i = 0; i < pparms->prevUeGrpChEstInfoBufIdx; i++)
    {
        data_in.pChEstInfo[i].tChEstBuffer.desc =pparms->dataIn.pChEstInfo[i].tChEstBuffer.desc;
        data_in.pChEstInfo[i].tChEstBuffer.pAddr=pparms->dataIn.pChEstInfo[i].tChEstBuffer.pAddr;
        data_in.pChEstInfo[i].startPrbGrp       =pparms->dataIn.pChEstInfo[i].startPrbGrp;
        data_in.pChEstInfo[i].srsPrbGrpSize     =pparms->dataIn.pChEstInfo[i].srsPrbGrpSize;
        data_in.pChEstInfo[i].startValidPrg     =pparms->dataIn.pChEstInfo[i].startValidPrg;
        data_in.pChEstInfo[i].nValidPrg         =pparms->dataIn.pChEstInfo[i].nValidPrg;
    }
    dyn_params.pDataIn = &data_in;
    dyn_params.pDataOut = &pparms->dataOutH;
    dyn_params.pDynDbg = &ulBfwDynDbgPrms;
    dyn_params.pDynDbg->enableApiLogging=0;
    dyn_params.procModeBmsk=pdctx->getEnableDlCuphyGraphs() ? BFW_PROC_MODE_WITH_GRAPH : BFW_PROC_MODE_NO_GRAPH;
    //TODO : Populate Input and Output buffers
    
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, s_channel));
    //if(dyn_params_num > 0)
    {
        status = cuphySetupBfwTx(*handle, &dyn_params);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "{}:cuphySetupBfwTx(): {}",__FUNCTION__, cuphyGetErrorString(status));
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
            return -1;
        }
    }
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG, "UL PhyBfwAggr{} SFN {}.{} setup", this_id, si->sfn_, si->slot_);

    return 0;
}

int PhyUlBfwAggr::run()
{
    cuphyStatus_t status;
    int ret=0;
    
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, s_channel));
    //if(dyn_params_num > 0)
    if(getSetupStatus() == CH_SETUP_DONE_NO_ERROR)
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        uint64_t procModeBmsk=pdctx->getEnableDlCuphyGraphs() ? BFW_PROC_MODE_WITH_GRAPH : BFW_PROC_MODE_NO_GRAPH;
        status = cuphyRunBfwTx(*handle,procModeBmsk);
        if (status != CUPHY_STATUS_SUCCESS) {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Error! cuphyRunBfwTx(): {}", cuphyGetErrorString(status));
            ret=-1;
        }
    }
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, s_channel));
    }
    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG, "UL PhyBfwAggr{} SFN {}.{} run", this_id, si->sfn_, si->slot_);

    return ret;
}

int PhyUlBfwAggr::callback()
{
    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG,"ULBFW pipeline {} SFN {}.{} callback entered",this_id, si->sfn_, si->slot_);
    return 0;
}


int PhyUlBfwAggr::validate() {
    int ret = 0;
#ifdef ULBFW_H5DUMP
    bool triggerH5Dump = true;
    if(triggerH5Dump)
    {
        NVLOGC_FMT(TAG, "SFN {}.{} Generating H5 Debug ULBFW file {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
        auto& stream = s_channel;
        cudaStreamSynchronize(stream);
        cuphyStatus_t debugStatus = cuphyWriteDbgBufSynchBfw(*(handle.get()), stream);
        if(debugStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyWriteDbgBufSynchBfw returned error {}", debugStatus);
            return -1;
        }
        cudaStreamSynchronize(stream);
        debugFileH.get()->close();
        debugFileH.reset();
        EXIT_L1(EXIT_FAILURE);
    }
#endif    
    return ret;
}

