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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 21) // "DRV.PRACH"

//#define PRACH_H5DUMP

#include "phyprach_aggr.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuda_events.hpp"
#include <unordered_map>

static void print_prach_static_params(const cuphyPrachStatPrms_t * prach_stat_params)
{
    #if 1
    NVLOGI_FMT(TAG, "\nPRACH Static Parameters:");
    NVLOGI_FMT(TAG, "---------------------------------------------------------------");
    NVLOGI_FMT(TAG, "Number of Cells             : {}", prach_stat_params->nMaxCells);
    for(int i = 0 ; i < prach_stat_params->nMaxCells; i++)
    {
        NVLOGI_FMT(TAG, "Cell Index              : {}", i);
        NVLOGI_FMT(TAG, "Occasion Start index    : {}", prach_stat_params->pCellPrms[i].occaStartIdx);
        NVLOGI_FMT(TAG, "No. of FDM occasions    : {}", prach_stat_params->pCellPrms[i].nFdmOccasions);
        NVLOGI_FMT(TAG, "Number of Antennas      : {}", prach_stat_params->pCellPrms[i].N_ant);
        NVLOGI_FMT(TAG, "FR                      : {}", prach_stat_params->pCellPrms[i].FR);
        NVLOGI_FMT(TAG, "duplex                  : {}", prach_stat_params->pCellPrms[i].duplex);
        NVLOGI_FMT(TAG, "mu                      : {}", prach_stat_params->pCellPrms[i].mu);
        NVLOGI_FMT(TAG, "Configuration Index     : {}", prach_stat_params->pCellPrms[i].configurationIndex);
        NVLOGI_FMT(TAG, "Restricted Set          : {}", prach_stat_params->pCellPrms[i].restrictedSet);
        NVLOGI_FMT(TAG, "---------------------------------------------------------------");
    }
    NVLOGI_FMT(TAG, "No. of Occasions per pipeline    : {}", prach_stat_params->nMaxOccaProc);
    for(int i = 0; i < prach_stat_params->nMaxOccaProc; i++)
    { 
        NVLOGI_FMT(TAG, "Cell Static index                : {}", prach_stat_params->pOccaPrms[i].cellPrmStatIdx);
        NVLOGI_FMT(TAG, "Root Sequence Index              : {}", prach_stat_params->pOccaPrms[i].prachRootSequenceIndex);
        NVLOGI_FMT(TAG, "Zero Corr. zone config           : {}", prach_stat_params->pOccaPrms[i].prachZeroCorrConf);
        NVLOGI_FMT(TAG, "---------------------------------------------------------------");
    }
#endif
}

static void print_prach_dynamic_params(const cuphyPrachOccaDynPrms_t * prach_dyn_params, const uint16_t numOccasions)
{
    #if 1
    NVLOGI_FMT(TAG, "\nPRACH Dynamic Parameters:");
    for(int i = 0; i < numOccasions; i++)
    {
        NVLOGI_FMT(TAG, "---------------------------------------------------------------");
        NVLOGI_FMT(TAG, "Occasion Param Stat Index        : {}", prach_dyn_params[i].occaPrmStatIdx);
        NVLOGI_FMT(TAG, "Occasion Param Dyn Index         : {}", prach_dyn_params[i].occaPrmDynIdx);
        NVLOGI_FMT(TAG, "Force threshold                  : {}", prach_dyn_params[i].force_thr0);
        NVLOGI_FMT(TAG, "---------------------------------------------------------------");
    }
    #endif
}

PhyPrachAggr::PhyPrachAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t*     _s_channels,
    MpsCtx * _mpsCtx):
    PhyChannel(_pdh, _gDev, 0, _s_channels[0], _mpsCtx),
    m_batchedMemcpyHelper(4 /* prach D2H copies*/, batchedMemcpySrcHint::srcIsDevice, batchedMemcpyDstHint::dstIsHost, (PRACH_USE_BATCHED_MEMCPY == 1) && StaticConversion<PhyDriverCtx>(pdh).get()->getUseBatchedMemcpy())
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    setCtx();

    mf.init(_pdh, std::string("PhyPrach"), sizeof(PhyPrachAggr));
    cuphyMf.init(_pdh, std::string("cuphyPrachRx"), 0);

    channel_type = slot_command_api::channel_type::PRACH;
    channel_name.assign("PRACH");

    cuphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing
				    
#if 0
    dyn_dbg_params.enableApiLogging  = 1;
    stat_dbg_params.enableApiLogging = 1;
#else
    dyn_dbg_params.enableApiLogging  = 0;
    stat_dbg_params.enableApiLogging = 0;
    stat_dbg_params.pOutFileName = nullptr;
#endif

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////// Init PRACH
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    try
    {
        //Note: Assumption Max RBs and repetitions are used for allocation 
        uint16_t nRA_RB = ORAN_PRACH_PRB;
        uint32_t N_rep = ORAN_PRACH_REPETITIONS;      /*!< number of preamble repetition */
        cell_id_list.clear();
        pInput.pTDataRx = (cuphyTensorPrm_t*) calloc(PRACH_MAX_OCCASIONS_AGGR, sizeof(cuphyTensorPrm_t));

        for(int idx = 0; idx < PRACH_MAX_OCCASIONS_AGGR; idx++)
        {
            prach_data_rx_desc[idx] = {CUPHY_C_16F, static_cast<int>(nRA_RB * CUPHY_N_TONES_PER_PRB * N_rep), cuphy::tensor_flags::align_tight};
            pInput.pTDataRx[idx].desc = prach_data_rx_desc[idx].handle();
            pInput.pTDataRx[idx].pAddr = nullptr;
        }

        // Allocate output buffers
        gpu_num_detectedPrmb    = std::move(cuphy::tensor_device(
                                    CUPHY_R_32U,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addGpuRegularSize(gpu_num_detectedPrmb.desc().get_size_in_bytes());

        cpu_num_detectedPrmb    = cuphy::tensor_pinned(
                                    CUPHY_R_32U,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight);
        mf.addCpuPinnedSize(cpu_num_detectedPrmb.desc().get_size_in_bytes());
        gpu_prmbIndex_estimates = std::move(cuphy::tensor_device(
                                    CUPHY_R_32U,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addGpuRegularSize(gpu_prmbIndex_estimates.desc().get_size_in_bytes());
        cpu_prmbIndex_estimates = cuphy::tensor_pinned(
                                    CUPHY_R_32U,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight);
        mf.addCpuPinnedSize(cpu_prmbIndex_estimates.desc().get_size_in_bytes());
        gpu_prmbDelay_estimates = std::move(cuphy::tensor_device(
                                    CUPHY_R_32F,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addGpuRegularSize(gpu_prmbDelay_estimates.desc().get_size_in_bytes());
        cpu_prmbDelay_estimates = cuphy::tensor_pinned(
                                    CUPHY_R_32F,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight);
        mf.addCpuPinnedSize(cpu_prmbDelay_estimates.desc().get_size_in_bytes());
       
        gpu_prmbPower_estimates = std::move(cuphy::tensor_device(
                                    CUPHY_R_32F,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addGpuRegularSize(gpu_prmbPower_estimates.desc().get_size_in_bytes());

        cpu_prmbPower_estimates = cuphy::tensor_pinned(
                                    CUPHY_R_32F,
                                    PRACH_MAX_NUM_PREAMBLES,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight);
        mf.addCpuPinnedSize(cpu_prmbPower_estimates.desc().get_size_in_bytes());

        ant_rssi = std::move(cuphy::tensor_pinned(
                                    CUPHY_R_32F,
                                    MAX_N_ANTENNAS_SUPPORTED,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addCpuPinnedSize(ant_rssi.desc().get_size_in_bytes());

        rssi = std::move(cuphy::tensor_pinned(
                                    CUPHY_R_32F,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addCpuPinnedSize(rssi.desc().get_size_in_bytes());

        interference = std::move(cuphy::tensor_pinned(
                                    CUPHY_R_32F,
                                    PRACH_MAX_OCCASIONS_AGGR,
                                    cuphy::tensor_flags::align_tight));
        mf.addCpuPinnedSize(interference.desc().get_size_in_bytes());

        pDynParams.cuStream = s_channel;
        pDynParams.pDataOut = &pOutput;

        pDynParams.pDataOut->numDetectedPrmb.desc = gpu_num_detectedPrmb.desc().handle();
        pDynParams.pDataOut->numDetectedPrmb.pAddr = gpu_num_detectedPrmb.addr();

        pDynParams.pDataOut->prmbIndexEstimates.desc = gpu_prmbIndex_estimates.desc().handle();
        pDynParams.pDataOut->prmbIndexEstimates.pAddr = gpu_prmbIndex_estimates.addr();

        pDynParams.pDataOut->prmbDelayEstimates.desc = gpu_prmbDelay_estimates.desc().handle();
        pDynParams.pDataOut->prmbDelayEstimates.pAddr = gpu_prmbDelay_estimates.addr();

        pDynParams.pDataOut->prmbPowerEstimates.desc = gpu_prmbPower_estimates.desc().handle();
        pDynParams.pDataOut->prmbPowerEstimates.pAddr = gpu_prmbPower_estimates.addr();

        pDynParams.pDataOut->rssi.desc  = rssi.desc().handle();
        pDynParams.pDataOut->rssi.pAddr = rssi.addr();
        pDynParams.pDataOut->antRssi.desc  = ant_rssi.desc().handle();
        pDynParams.pDataOut->antRssi.pAddr = ant_rssi.addr();
        pDynParams.pDataOut->interference.desc  = interference.desc().handle();
        pDynParams.pDataOut->interference.pAddr = interference.addr();

        //Note: pAddr for input data needs to be set properly in setup
        pDynParams.pDataIn = &pInput;
        pDynParams.procModeBmsk = 0;
	    pDynParams.pDbg = &dyn_dbg_params;
        
        statusOut = {cuphyPrachStatusType_t::CUPHY_PRACH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
        pDynParams.pStatusOut = &statusOut;
    }
    PHYDRIVER_CATCH_THROW_EXCEPTIONS();

    cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

    //CRC errors number
    prach_crc_errors_h.reset(new host_buf(1 * sizeof(uint32_t), gDev));
    prach_crc_errors_h->clear();
    mf.addCpuPinnedSize(sizeof(uint32_t));

    launch_kernel_warmup(s_channel);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);

    gDev->synchronizeStream(s_channel);

    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_copy));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_copy));

    handle = nullptr;
    handle_temp = nullptr;
};

PhyPrachAggr::~PhyPrachAggr()
{
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_copy));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_copy));
    if(handle)
        cuphyDestroyPrachRx(handle);
    free(pInput.pTDataRx);
};

int PhyPrachAggr::createPhyObj() 
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    int index = 0;
    uint32_t cellCount = 0;

    setCtx();

    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    //NVLOGC_FMT(TAG, "number of cells: {}", cell_list.size());
    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        int tmp_cell_id = cell_ptr->getPhyId();

        // Add only active cells here
        if(tmp_cell_id == DEFAULT_PHY_CELL_ID)
            continue;
        
        //change for multi-cell (Not needed in case of PRACH: Check again)
        auto it = std::find(cell_id_list.begin(), cell_id_list.end(), tmp_cell_id);
        if(it == cell_id_list.end())
        {
            cell_id_list.push_back(tmp_cell_id);
        }
        else
            continue;
   
        index = prachCellStatVec.size();
        if(prachCellStatIndex.insert(std::pair<cell_id_t,uint32_t>(cell_ptr->getId(),index)).second == false)
        {           
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} insert error in prachCellStatIndex", cell_id);
            return -1;
        }
        else
            NVLOGI_FMT(TAG, "prachCellStatIndex: cell-id={} cellPrmStatIdx={} occaStartIdx={}",cell_ptr->getId(),index,prachOccaStatVec.size());

        auto cell_stat_prms = cell_ptr->getPrachCellStatConfig();
        cell_stat_prms->occaStartIdx = prachOccaStatVec.size();
        prachCellStatVec.push_back(*(cell_ptr->getPrachCellStatConfig()));

        cell_ptr->setPrachOccaPrmStatIdx(prachOccaStatVec.size());
        prach_occa_stat_params = *(cell_ptr->getPrachOccaStatVec());
        for(uint32_t i = 0; i < prach_occa_stat_params.size(); i++)
            prach_occa_stat_params[i].cellPrmStatIdx = index;
        prachOccaStatVec.insert(prachOccaStatVec.end(), prach_occa_stat_params.begin(), prach_occa_stat_params.end());
    }

    prach_params_static.nMaxCells = prachCellStatVec.size();
    prach_params_static.pCellPrms = static_cast<cuphyPrachCellStatPrms_t*>(prachCellStatVec.data());
    prach_params_static.pOccaPrms = static_cast<cuphyPrachOccaStatPrms_t*>(prachOccaStatVec.data());
    prach_params_static.nMaxOccaProc = prachOccaStatVec.size();
    prach_params_static.enableUlRxBf = pdctx->getmMIMO_enable();
    prach_params_static.pOutInfo = &cuphy_tracker;
    prach_params_static.pDbg = &stat_dbg_params;
    
    if(prachCellStatVec.size() == pdctx->getCellGroupNum())
    {

#ifdef PRACH_H5DUMP
        std::string dbg_output_file = std::string("PRACH_Debug") + std::to_string(id) + std::string(".h5");
        debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(dbg_output_file.c_str())));
        prach_params_static.pDbg->pOutFileName             = dbg_output_file.c_str();
        prach_params_static.pDbg->enableApiLogging=1;
#else
        prach_params_static.pDbg->pOutFileName = nullptr;
        prach_params_static.pDbg->enableApiLogging=0;
#endif    
        //print_prach_static_params(&prach_params_static);
        cuphyStatus_t createStatus = cuphyCreatePrachRx(&handle, &prach_params_static);

        std::string cuphy_ch_create_name = "cuphyCreatePrachRx";            
        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPrachRx(handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(prach_params_static.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();

        gDev->synchronizeStream(s_channel);
    }
    else if(prachCellStatVec.size() > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    return 0;
}

int PhyPrachAggr::deleteTempPhyObj() 
{
    if(handle_temp)
        cuphyDestroyPrachRx(handle_temp);
    return 0;
}

int PhyPrachAggr::createNewPhyObj() 
{
    setCtx();
    //print_prach_static_params(&prach_params_static);
    cuphyStatus_t createStatus = cuphyCreatePrachRx(&handle_temp, &prach_params_static);

    if(createStatus != CUPHY_STATUS_SUCCESS)
    {
        switch(createStatus)
        {
            case CUPHY_STATUS_INVALID_ARGUMENT:
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,"cuphyCreatePrachRx returned CUPHY_STATUS_INVALID_ARGUMENT");
                break;
            case CUPHY_STATUS_ALLOC_FAILED:
                 NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "cuphyCreatePrachRx returned CUPHY_STATUS_ALLOC_FAILED");
                break;
            case CUPHY_STATUS_NOT_SUPPORTED:
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "cuphyCreatePrachRx returned CUPHY_STATUS_NOT_SUPPORTED");
                break;
            default:
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "cuphyCreatePrachRx returned CUPHY_STATUS_UNKNOWN");
                break;
        }
        return -1;
    }

    gDev->synchronizeStream(s_channel);
    return 0;
}
void PhyPrachAggr::changePhyObj()
{
    cuphyPrachRxHndl_t temp = handle;
    handle = handle_temp;
    handle_temp = temp;
}

slot_command_api::prach_params* PhyPrachAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->prach.get();
}

int PhyPrachAggr::setup(
    const std::vector<Cell *>& aggr_cell_list,
    const std::vector<ULInputBuffer *>& ulbuf_st3_v,
    cudaStream_t stream
)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    t_ns          t1    = Time::nowNs();

    slot_command_api::oran_slot_ind oran_ind     = getOranAggrSlotIndication();
    
    setCtx();

    slot_command_api::prach_params* pparms = getDynParams();
    if(pparms == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error getting Dynamic PRACH params from Slot Command");
        return -1;
    }
    
    if(aggr_cell_list.size() == 0)
        return -1;

    if(ulbuf_st3_v.size() == 0)
        return -1;

    pDynParams.pOccaPrms = pparms->rach.data();
    pDynParams.nOccaProc = pparms->nOccasion;
    //print_prach_dynamic_params(pDynParams.pOccaPrms, pDynParams.nOccaProc);

    // Buffer-to-cell mapping: Reconciles two different orderings
    // 1. Buffers (ulbuf_st3_v): Allocated sequentially per aggr_cell_list (includes all UL cells, not just PRACH)
    // 2. PRACH PDUs: Arrive from L2 in pparms->phy_cell_index_list order (may differ from aggr_cell_list)
    // Map stores {start_buffer_idx, num_occasions} per phy_cell_id
    
    cell_buffer_info_size = 0;
    
    int buffer_idx = 0;
    for(const Cell* cell : aggr_cell_list) 
    {
        const int phy_cell_id = cell->getPhyId();
        const size_t num_occasions = cell->getPrachOccaSize();
        
        if(num_occasions > 0) 
        {
            cell_buffer_info[cell_buffer_info_size++] = {phy_cell_id, {buffer_idx, num_occasions}};
            NVLOGD_FMT(TAG, "{}: PHY Cell ID={} has {} PRACH Occasions, buffer indices [{}-{})",
            __FUNCTION__, phy_cell_id, num_occasions, buffer_idx, buffer_idx + num_occasions - 1);
            buffer_idx += num_occasions;
        }
    }

    // Reorder PRACH PDUs (L2 order) and map to corresponding buffers
    int previous_cell_id = -1;
    int occasion_counter = 0;
    
    for(int idx = 0; idx < pDynParams.nOccaProc; idx++)
    {
        const int phy_cell_id = pparms->phy_cell_index_list[idx];
        
        if(phy_cell_id != previous_cell_id) {
            occasion_counter = 0;
            previous_cell_id = phy_cell_id;
        }
        
        const int occasion_num = occasion_counter++;
        
        const auto buffer_it = std::find_if(cell_buffer_info.begin(), cell_buffer_info.begin() + cell_buffer_info_size,
            [phy_cell_id](const auto& pair) { return pair.first == phy_cell_id; });
        
        if(buffer_it == cell_buffer_info.begin() + cell_buffer_info_size || occasion_num >= static_cast<int>(buffer_it->second.second)) 
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Cannot find buffer index for PHY Cell ID={} Occasion={}", __FUNCTION__, phy_cell_id, occasion_num);
            return -1;
        }
        
        const int actual_buffer_idx = buffer_it->second.first + occasion_num;
        
        pDynParams.pOccaPrms[idx].occaPrmDynIdx = idx;
        pDynParams.pDataIn->pTDataRx[idx].pAddr = ulbuf_st3_v[actual_buffer_idx]->getBufD();
        
        NVLOGD_FMT(TAG, "{}: Occasion={} PHY Cell ID={} Occasion Number={} -> Buffer Index={}", __FUNCTION__, idx, phy_cell_id, occasion_num, actual_buffer_idx);
    }

    // Set processing mode bitmask
    pDynParams.procModeBmsk = pdctx->getEnableUlCuphyGraphs() ? PRACH_PROC_MODE_WITH_GRAPH : PRACH_PROC_MODE_NO_GRAPH;
    pDynParams.cuStream = stream;

    t_ns t2     = Time::nowNs();
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, pDynParams.cuStream));
    }
    cuphyStatus_t status = cuphySetupPrachRx(handle, &pDynParams);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphySetupPrachRx returned error {}", cuphyGetErrorString(status));
        {
            MemtraceDisableScope md;
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, pDynParams.cuStream));
        }
        return -1;
    }
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, pDynParams.cuStream));
    }

    return 0;
}

int PhyPrachAggr::run()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret=0;
    
    setCtx();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PRACH Run
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, pDynParams.cuStream));
    }
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR))
    {
        cuphyStatus_t status =  cuphyRunPrachRx(handle);

        if (status != CUPHY_STATUS_SUCCESS) {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyRunPrachRx returned error {}", cuphyGetErrorString(status));
            ret=-1;
        }    
    }
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, pDynParams.cuStream));
    }
    t_ns t3 = Time::nowNs();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PRACH Output Copy from GPU to CPU
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_copy, pDynParams.cuStream));
    }

    // Perform the 4 D2H copies using batched async memcpy, if enabled.
    // Each updateMemcpy call updates the parameters of a copy or fallback to a default async. memcpy if batched memcpy not enabled.
    m_batchedMemcpyHelper.reset(); // reset for upcoming batch of updateMemcpy calls
    m_batchedMemcpyHelper.updateMemcpy(cpu_num_detectedPrmb.addr(),  gpu_num_detectedPrmb.addr(),
                 PRACH_MAX_OCCASIONS_AGGR * sizeof(uint32_t), cudaMemcpyDeviceToHost, pDynParams.cuStream);

    m_batchedMemcpyHelper.updateMemcpy(cpu_prmbIndex_estimates.addr(), gpu_prmbIndex_estimates.addr(),
                 PRACH_MAX_NUM_PREAMBLES * PRACH_MAX_OCCASIONS_AGGR *  sizeof(uint32_t), cudaMemcpyDeviceToHost, pDynParams.cuStream);

    m_batchedMemcpyHelper.updateMemcpy(cpu_prmbDelay_estimates.addr(), gpu_prmbDelay_estimates.addr(),
                 PRACH_MAX_NUM_PREAMBLES * PRACH_MAX_OCCASIONS_AGGR * sizeof(float), cudaMemcpyDeviceToHost, pDynParams.cuStream);

    m_batchedMemcpyHelper.updateMemcpy(cpu_prmbPower_estimates.addr(), gpu_prmbPower_estimates.addr(),
                 PRACH_MAX_NUM_PREAMBLES * PRACH_MAX_OCCASIONS_AGGR * sizeof(float), cudaMemcpyDeviceToHost, pDynParams.cuStream);

    // Launch the batched memcpy if enabled. No-op otherwise.
    cuphyStatus_t batched_memcpy_status = m_batchedMemcpyHelper.launchBatchedMemcpy(pDynParams.cuStream);
    if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "Launching batched memcpy for PRACH returned an error");
        ret=-1;
    }

    // Left old commented out code: copy kernel instead of memcpy
    // launch_kernel_copy(pDynParams.cuStream, (uint8_t*) gpu_num_detectedPrmb.addr(), (uint8_t*) cpu_num_detectedPrmb.addr(), sizeof(uint32_t));
    // launch_kernel_copy(pDynParams.cuStream, (uint8_t*) gpu_prmbIndex_estimates.addr(), (uint8_t*) cpu_prmbIndex_estimates.addr(), PRACH_MAX_NUM_PREAMBLES * sizeof(uint32_t));
    // launch_kernel_copy(pDynParams.cuStream, (uint8_t*) gpu_prmbDelay_estimates.addr(), (uint8_t*) cpu_prmbDelay_estimates.addr(), PRACH_MAX_NUM_PREAMBLES * sizeof(uint32_t));
    // launch_kernel_copy(pDynParams.cuStream, (uint8_t*) gpu_prmbPower_estimates.addr(), (uint8_t*) cpu_prmbPower_estimates.addr(), PRACH_MAX_NUM_PREAMBLES * sizeof(uint32_t));

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_copy, pDynParams.cuStream));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PRACH notify completion
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Better to comment out this memset when debugging received input
    // CUDA_CHECK_PHYDRIVER(cudaMemsetAsync(prach_data_rx.addr(), 0, prach_data_rx.desc().get_size_in_bytes(), s_channel));
    return ret;
}

uint32_t PhyPrachAggr::getCellStatVecSize()
{
    return prachCellStatVec.size();
}

int PhyPrachAggr::updateConfig(cell_id_t cell_id, cell_phy_info& cell_pinfo)
{

    /*for(auto it = prachCellStatIndex.begin(); it != prachCellStatIndex.end(); it++)
    {
        NVLOGD_FMT(TAG,"prachCellStatIndex key {} T {}",it->first,it->second);
    }*/

    cell_id_t temp_cell_id = cell_id;
    auto it = prachCellStatIndex.find(temp_cell_id);
    if(it == prachCellStatIndex.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "cell_id {} not found in prachCellStatIndex", cell_id);
        return -1;
    }
    cuphyPrachStatPrms_t const prach_stat_params{
        .pOutInfo = nullptr,
        .nMaxCells = 1,
        .pCellPrms = &cell_pinfo.prachStatParams,
        .pOccaPrms = cell_pinfo.prach_configs.data(),
        .nMaxOccaProc = static_cast<uint16_t>(cell_pinfo.prach_configs.size()),
    };
    auto status = cuphyValidatePrachParams(&prach_stat_params);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "cuphyValidatePrachParams returned error {}", cuphyGetErrorString(status));
        return -1;
    }
    uint32_t cellPrmStatIdx = it->second;

    if(prachCellStatVec[cellPrmStatIdx].configurationIndex != cell_pinfo.prachStatParams.configurationIndex)
    {
        NVLOGI_FMT(TAG,"Cell stat idx = {} changing PRACH configIdx from {} to {}",cellPrmStatIdx,
            prachCellStatVec[cellPrmStatIdx].configurationIndex,cell_pinfo.prachStatParams.configurationIndex);

        prachCellStatVec[cellPrmStatIdx].configurationIndex = cell_pinfo.prachStatParams.configurationIndex;
    }

    if( prachCellStatVec[cellPrmStatIdx].restrictedSet != cell_pinfo.prachStatParams.restrictedSet)
    {
        NVLOGI_FMT(TAG,"Cell stat idx = {} changing restrictedSet from {} to {}",cellPrmStatIdx,
            prachCellStatVec[cellPrmStatIdx].restrictedSet, cell_pinfo.prachStatParams.restrictedSet);

        prachCellStatVec[cellPrmStatIdx].restrictedSet = cell_pinfo.prachStatParams.restrictedSet;
    }


    uint32_t occStartIdx = prachCellStatVec[cellPrmStatIdx].occaStartIdx;
    if(prachCellStatVec[cellPrmStatIdx].nFdmOccasions >= cell_pinfo.prachStatParams.nFdmOccasions)
    {
        NVLOGI_FMT(TAG, "prach aggr : update config - nFdmOccasions in new config <= existing config : {}",
            cell_pinfo.prachStatParams.nFdmOccasions);
        for(uint32_t j = 0; j < cell_pinfo.prachStatParams.nFdmOccasions; j++)
        {
            prachOccaStatVec[occStartIdx + j].prachRootSequenceIndex = cell_pinfo.prach_configs[j].prachRootSequenceIndex;
            prachOccaStatVec[occStartIdx + j].prachZeroCorrConf = cell_pinfo.prach_configs[j].prachZeroCorrConf;
        }
        for(uint32_t j = cell_pinfo.prachStatParams.nFdmOccasions; j < prachCellStatVec[cellPrmStatIdx].nFdmOccasions; j++)
            prachOccaStatVec[occStartIdx + j].cellPrmStatIdx =
            prachOccaStatVec[occStartIdx + j].prachRootSequenceIndex =
            prachOccaStatVec[occStartIdx + j].prachZeroCorrConf = 0;

        prach_occa_stat_params = cell_pinfo.prach_configs;
        for(uint32_t j = 0; j < prach_occa_stat_params.size(); j++)
            prach_occa_stat_params[j].cellPrmStatIdx = cellPrmStatIdx;

        prachCellStatVec[cellPrmStatIdx].nFdmOccasions = cell_pinfo.prachStatParams.nFdmOccasions;
    }
    else
    {
        for(uint32_t j = occStartIdx; j < occStartIdx+prachCellStatVec[cellPrmStatIdx].nFdmOccasions; j++)
            prachOccaStatVec[j].prachRootSequenceIndex = prachOccaStatVec[j].prachZeroCorrConf = 
                prachOccaStatVec[j].cellPrmStatIdx = 0;

        //prachOccaStatVec.erase(prachOccaStatVec.begin()+occStartIdx,prachOccaStatVec.begin()+occStartIdx+
            //prachCellStatVec[cellPrmStatIdx].nFdmOccasions);

        NVLOGI_FMT(TAG, "prach aggr: update config - zero out {} entries at {} for cell_id {}",
            prachCellStatVec[cellPrmStatIdx].nFdmOccasions,occStartIdx,cell_id);

        prachCellStatVec[cellPrmStatIdx].occaStartIdx = prachOccaStatVec.size();
        PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* cell_ptr = pdctx->getCellById(cell_id);
        cell_ptr->setPrachOccaPrmStatIdx(prachOccaStatVec.size());
        prachCellStatVec[cellPrmStatIdx].nFdmOccasions = cell_pinfo.prachStatParams.nFdmOccasions;
        prach_occa_stat_params = cell_pinfo.prach_configs;
        for(uint32_t j = 0; j < prach_occa_stat_params.size(); j++)
            prach_occa_stat_params[j].cellPrmStatIdx = cellPrmStatIdx;
        NVLOGI_FMT(TAG, "PhyPrachAggr::updateConfig : cell-id={} cellPrmStatIdx={} occaStartIdx={}",cell_id,cellPrmStatIdx,prachOccaStatVec.size());
        prachOccaStatVec.insert(prachOccaStatVec.end(), prach_occa_stat_params.begin(), prach_occa_stat_params.end());
        prach_params_static.pOccaPrms = static_cast<cuphyPrachOccaStatPrms_t*>(prachOccaStatVec.data());
        prach_params_static.nMaxOccaProc = prachOccaStatVec.size();

        NVLOGI_FMT(TAG, "prach aggr: update config - added {} occasions at {} for cell_id {}. size {}",
            prachCellStatVec[cellPrmStatIdx].nFdmOccasions,
            prachCellStatVec[cellPrmStatIdx].occaStartIdx,cell_id,
            prachOccaStatVec.size());
    }
    return 0;
}
#if 0
int PhyPrachAggr::wait(int wait_ns)
{
    t_ns          threshold_t(wait_ns), start_t = Time::nowNs();
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(!isActive())
        return -1;

    while(ACCESS_ONCE(*((uint32_t*)prach_completed_h->addr())) == 0)
    // while(cudaEventQuery(end_copy) != cudaSuccess)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: PRACH Object {} Cell {} waiting for GPU task more than {} ns",
                    getId(), cell_id, wait_ns);
            return -1;
        }
    }

    // Enable only in debug mode when needed
    #if 0
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "============ PRACH input buffer bytes {} ============ ", static_cast<int>(buf_sz));
    for(int b=0; b < static_cast<int>(buf_sz); b++) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{:02X}", buf_h[b]);
    }
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "============================================================");
    #endif

    return 0;
}
#endif

#if 0
int PhyPrachAggr::reference_comparison()
{
    int numPrmb = ((uint32_t*)cpu_num_detectedPrmb.addr())[0];
    if (numPrmb != 1) {
         std::cout << "Preamble NOT detected ========> Test FAIL\n\n";
         return -1;
    }

    // Compare results between GPU and Matlab output
    std::cout << "---------------------------------------------------------------\n";
    std::cout << "Comparing test vectors ... \n";
    int index_est_mismatch = 0;
    int delay_est_mismatch = 0;
    int power_est_mismatch = 0;
    for (int prmbCounter = 0; prmbCounter < numPrmb; prmbCounter++) {
	    int matlab_prmbIndex_val = matlab_prmbIndex_estimates({prmbCounter});
	    int cpu_prmbIndex_val = ((uint32_t*)cpu_prmbIndex_estimates.addr())[prmbCounter];
	    NVLOGC_FMT(TAG, "prmbIndex - matlab = {:6d} vs. gpu = {:6d}", matlab_prmbIndex_val, cpu_prmbIndex_val);
        if (matlab_prmbIndex_val != cpu_prmbIndex_val) {
		    index_est_mismatch += 1;
        }
	    float matlab_prmbDelay_val = matlab_prmbDelay_estimates({prmbCounter})*1e6;
	    float cpu_prmbDelay_val = ((uint32_t*)cpu_prmbDelay_estimates.addr())[prmbCounter]*1e6;
        NVLOGC_FMT(TAG, "prmbDelay - matlab = {:6.3f} vs. gpu = {:6.3f}", matlab_prmbDelay_val, cpu_prmbDelay_val);
        if (!compare_approx(matlab_prmbDelay_val, cpu_prmbDelay_val, 0.001f)) {
		    delay_est_mismatch += 1;
	    }
        float matlab_prmbPower_val = matlab_prmbPower_estimates({prmbCounter});
	    float cpu_prmbPower_val = ((uint32_t*)cpu_prmbPower_estimates.addr())[prmbCounter];
        NVLOGC_FMT(TAG, "prmbPower - matlab = {:6.3f} vs. gpu = {:6.3f}", matlab_prmbPower_val, cpu_prmbPower_val);
        if (!compare_approx(matlab_prmbPower_val, cpu_prmbPower_val, 0.001f)) {
		    power_est_mismatch += 1;
	    }
	}

    if (index_est_mismatch+delay_est_mismatch+power_est_mismatch == 0) {
        return 0;
    }

    return -1;
}
#endif

int PhyPrachAggr::validate()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret = 0;

    setCtx();

    if(pdctx->isValidation())
    {
        //ret = reference_comparison();
    }

#ifdef PRACH_H5DUMP
        bool triggerH5Dump = true;
        if(triggerH5Dump)
        {
            NVLOGC_FMT(TAG, "SFN {}.{} Generating H5 Debug PRACH file {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
            auto& stream = s_channel;
            cudaStreamSynchronize(stream);
            cuphyStatus_t debugStatus = cuphyWriteDbgBufSynchPrach(handle, stream);
            if(debugStatus != CUPHY_STATUS_SUCCESS)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyWriteDbgBufSynchPrach returned error {}", debugStatus);
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

int PhyPrachAggr::callback()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    
    slot_command_api::ul_slot_callbacks ul_cb;

    uint32_t total_preambles_detected = 0;
    uint32_t start_ro = 0;
    uint32_t cell_index = 0;

    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    setCtx();

    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    NVLOGD_FMT(TAG, "CRC errors {}", ((uint32_t*)(prach_crc_errors_h->addr()))[0]);
    
    Cell* cell_ptr = pdctx->getCellById(cell_id);
    //rach_occasion = cell_ptr->getPrachOccaSize();
    if(pdctx->getUlCb(ul_cb))
    {
        NVLOGI_FMT(TAG, "Calling PRACH UL callback");

        auto prach = getDynParams();

        #if 0
        for(int cell_idx = 0; cell_idx < cell_id_list.size(); cell_idx++)
        {
            Cell * cell_ptr = pdctx->getCellByPhyId(cell_id_list[cell_idx]);
            NVLOGI_FMT(TAG, "Cell Index = {}", cell_id_list[cell_idx]);
            rach_occasion = cell_ptr->getPrachOccaSize();
            if(rach_occasion > 0)
            {
                for(int ro_idx = 0; ro_idx < rach_occasion; ro_idx++)
                {
                    uint32_t preambles_detected = ((uint32_t*)cpu_num_detectedPrmb.addr())[ro_idx + start_ro];

                    NVLOGI_FMT(TAG, "RO {} SFN {:03d}.{:02d} Preambles num detected {}",
                            ro_idx, aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_,
                            preambles_detected);

                    for(int i = 0; i < preambles_detected; ++i)
                    {
                        NVLOGI_FMT(TAG, "SFN {:03d}.{:02d} \t #{} prmbIndex {} prmbDelay {} prmbPower {}",
                        aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, i,
                        ((uint32_t*)cpu_prmbIndex_estimates.addr())[i + ro_idx * PRACH_MAX_NUM_PREAMBLES],
                        ((float*)cpu_prmbDelay_estimates.addr())[i + ro_idx * PRACH_MAX_NUM_PREAMBLES], 10 * std::log10(((float*)cpu_prmbPower_estimates.addr())[i  + ro_idx * PRACH_MAX_NUM_PREAMBLES]));
                    }
                    auto numAnt = prachCellStatVec[cell_idx].N_ant;
                    for(int antIndex = 0; antIndex < numAnt; ++antIndex) {
                        NVLOGI_FMT(TAG, " rssi per antenna[{}]= {:6.4f}",  antIndex, static_cast<float*>(ant_rssi.addr())[ro_idx * MAX_N_ANTENNAS_SUPPORTED + antIndex]);
                    }
                }
            }
            start_ro += rach_occasion;
        }
        #endif
        
        ul_cb.prach_cb_fn(
            ul_cb.prach_cb_context,
            *(aggr_slot_params->si), *prach, (uint32_t*)cpu_num_detectedPrmb.addr(),
            cpu_prmbIndex_estimates.addr(), cpu_prmbDelay_estimates.addr(), cpu_prmbPower_estimates.addr(),
            (void *)ant_rssi.addr(), (void *)rssi.addr(), (void *)interference.addr());
    }

    return 0;
}

float PhyPrachAggr::getGPUCopyTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_copy, end_copy, __func__);
}

//Note: https://godbolt.org/z/exshhq3bf
uint16_t PhyPrachAggr::numPrbcConversionTable(uint32_t L_RA, uint32_t delta_f_RA, uint32_t delta_f) {
    return std::ceil((L_RA/12.0) * static_cast<float>(delta_f_RA)/delta_f);
}

int PhyPrachAggr::cleanup()
{
    PhyChannel::cleanup();
    return 0;
}
