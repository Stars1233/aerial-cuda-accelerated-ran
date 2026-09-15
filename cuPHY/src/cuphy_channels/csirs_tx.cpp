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

#include "csirs_tx.hpp"
#include "cuphy_internal.h"
#include "utils.cuh"

#define CHECK_CONFIG 1 // runtime check enabled

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyCreateCsirsTx(cuphyCsirsTxHndl_t* pCsirsTxHndl, cuphyCsirsStatPrms_t const* pStatPrms)
{
    if((pCsirsTxHndl == nullptr) || (pStatPrms == nullptr))
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyCreateCsirsTx called with cuphyCsirsTxHndl_t ptr ({:p}) or cuphyCsirsStatPrms_t ptr ({:p}) nullptr.",
                  static_cast<void*>(pCsirsTxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        cuphyStatus_t constructorStatus = CUPHY_STATUS_SUCCESS;
        CsirsTx* new_pipeline = new CsirsTx(pStatPrms, &constructorStatus);
        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        
        if(constructorStatus != CUPHY_STATUS_SUCCESS)
        {
            delete new_pipeline;
            return constructorStatus;
        }
        
        *pCsirsTxHndl = new_pipeline;

        return CUPHY_STATUS_SUCCESS;
    });
}

#if 0
const void* cuphyGetMemoryFootprintTrackerCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl)
{
    if(csirsTxHndl == nullptr)
    {
        return nullptr;
    }
    CsirsTx* pipeline_ptr  = static_cast<CsirsTx*>(csirsTxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* CsirsTx::getMemoryTracker()
{
    return &memory_footprint;
}

cuphyStatus_t CUPHYWINAPI cuphySetupCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl, cuphyCsirsDynPrms_t* pDynPrms)
{
    if((pDynPrms == nullptr) ||  (csirsTxHndl == nullptr))
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphySetupCsirsTx called with nullptr for cuphyCsirsTxHndl_t ({:p}) or for cuphyCsirsDynPrms_t ptr ({:p}).",
                   static_cast<void*>(csirsTxHndl), static_cast<void*>(pDynPrms));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphySetupCsirsTx", 1);
        CsirsTx* pipeline_ptr  = static_cast<CsirsTx*>(csirsTxHndl);
        pDynPrms->chan_graph = pipeline_ptr->GetGraph();
        pipeline_ptr->dynamic_params = pDynPrms;
        cuphyStatus_t status = pipeline_ptr->expandParameters(pDynPrms, pDynPrms->cuStream);
        pipeline_ptr->setKernelParams();

        POP_RANGE
        return status;
    }, CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl)
{
    if(csirsTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyRunCsirsTx error: cuphyCsirsTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        CsirsTx* pipeline_ptr  = static_cast<CsirsTx*>(csirsTxHndl);
        cuphyStatus_t status = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl)
{
    if(csirsTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyDestroyCsirsTx error: cuphyCsirsTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    CsirsTx* pipeline_ptr  = static_cast<CsirsTx*>(csirsTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

CsirsTx::CsirsTx(cuphyCsirsStatPrms_t const* pStatPrms, cuphyStatus_t* pStatus) :
    static_params(pStatPrms),
    /* following parameters are properly updated on every setup() */
    dynamic_params(nullptr),
    m_genScramblingArgs{0},
    m_genTfSignalArgs{0},
    m_cudaGraphModeEnabled(true),
    numParams(1),
    numCells(1)
{

    //Memset all launch configs structs to 0; they will be properly updated on every setup() as needed
    memset(&m_genCsirsScramblingLaunchCfg, 0, sizeof(cuphyGenScramblingLaunchCfg_t));
    memset(&m_genCsirsTfSignalLaunchCfg, 0, sizeof(cuphyGenCsirsTfSignalLaunchCfg_t));

    if(pStatus == nullptr)
    {
        return;
    }
    // Check static configuration parameters
    cuphyStatus_t& configStatus = *pStatus;
    configStatus = checkStaticConfig(pStatPrms);
    
    if (configStatus != CUPHY_STATUS_SUCCESS)
    {
        // Early return on configuration error
        return;
    }

    maxCellsPerSlot = pStatPrms->nMaxCellsPerSlot;
    pStatPrms->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    // Set kernel functions to nullptr to ensure call to cudaFuncGetSymbol happens only once.
    // func of m_genCsirsTfSignalLaunchCfg not set as it's templated
    m_genCsirsScramblingLaunchCfg.kernelNodeParamsDriver.func = nullptr;

    // Allocate buffer for scrambling sequence computed on the GPU
    d_goldSeq = make_unique_device<uint8_t>(CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot * OFDM_SYMBOLS_PER_SLOT * round_up_to_next<int>(Ng, 32) / 8, &memory_footprint);

    // Allocate workspace buffers and update relevant host/device pointers
    workspace_offsets[0] = 0; // unchanged
    updateWorkspaceOffsets(maxCellsPerSlot, CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot);
    h_workspace = make_unique_pinned<uint8_t>(workspace_offsets[N_CSIRS_COMPONENTS]);
    d_workspace = make_unique_device<uint8_t>(workspace_offsets[N_CSIRS_COMPONENTS], &memory_footprint);
    updateWorkspacePtrs();

    // params pointers unchanged as they point to the beginning of the workspace buffer
    h_params = (CsirsParams*)(h_workspace.get() + workspace_offsets[0]);
    d_params = (CsirsParams*)(d_workspace.get() + workspace_offsets[0]);

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        memory_footprint.printMemoryFootprint(this, "CSIRS");
    }
}

void CsirsTx::updateWorkspaceOffsets(int cells, int nRrcParams)
{
    int max_alignment = alignof(__half2*); // for the output tensor addr.
    //workspace_offsets[0] = 0;
    workspace_offsets[1] = workspace_offsets[0] + round_up_to_next<int>(nRrcParams * sizeof(CsirsParams), max_alignment); //Alignment of offsets following CSI-RS params
    workspace_offsets[2] = workspace_offsets[1] + round_up_to_next<int>((nRrcParams + 1) * sizeof(uint32_t), max_alignment);  // Alignment of tensor addr. after offsets
    workspace_offsets[3] = workspace_offsets[2] + round_up_to_next<int>(cells * sizeof(__half2*), max_alignment); // Alignment of tensor REs after tensor addr.
    workspace_offsets[4] = workspace_offsets[3] + round_up_to_next<int>(cells * sizeof(uint16_t), max_alignment);
    workspace_offsets[5] = workspace_offsets[4] + round_up_to_next<int>(nRrcParams * sizeof(cuphyCsirsPmWOneLayer_t), max_alignment);
}

void CsirsTx::updateWorkspacePtrs()
{
    // The *_params pointers are updated once during CSIRS-TX creation and not here

    h_offsets = (uint32_t*)(h_workspace.get() + workspace_offsets[1]);
    d_offsets = (uint32_t*)(d_workspace.get() + workspace_offsets[1]);

    h_cell_tensor_addr = (__half2**)(h_workspace.get() + workspace_offsets[2]);
    d_cell_tensor_addr = (__half2**)(d_workspace.get() + workspace_offsets[2]);

    h_cell_tensor_REs = (uint16_t*)(h_workspace.get() + workspace_offsets[3]);
    d_cell_tensor_REs = (uint16_t*)(d_workspace.get() + workspace_offsets[3]);

    h_pmw_params = (cuphyCsirsPmWOneLayer_t*)(h_workspace.get() + workspace_offsets[4]);
    d_pmw_params = (cuphyCsirsPmWOneLayer_t*)(d_workspace.get() + workspace_offsets[4]);
}

CsirsTx::~CsirsTx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
}

cuphyStatus_t CsirsTx::checkConfig(CsirsParams* params, int numCsirsParams)
{
    for(int i = 0; i < numCsirsParams; ++i)
    {
        if (params[i].startRb + params[i].nRb > MAX_N_PRBS_SUPPORTED)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported startRb/nRb");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if (params[i].csiType == cuphyCsiType_t::ZP_CSI_RS)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Should not call CSI-RS channel with zero power CSI Type.");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if (params[i].li[0] > OFDM_SYMBOLS_PER_SLOT || params[i].li[1] > OFDM_SYMBOLS_PER_SLOT)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported time domain location.");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CsirsTx::checkStaticConfig(cuphyCsirsStatPrms_t const* pStatPrms)
{
    if (pStatPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "{} called with nullptr pStatPrms", __func__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->nMaxCellsPerSlot == 0)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "nMaxCellsPerSlot must be greater than 0");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->nCells == 0)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "nCells must be greater than 0");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->pCellStatPrms == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "pCellStatPrms is nullptr");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pStatPrms->nCells > pStatPrms->nMaxCellsPerSlot)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "nCells {} must be less than or equal to nMaxCellsPerSlot {}", pStatPrms->nCells, pStatPrms->nMaxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    for(int i = 0; i < pStatPrms->nCells; i++)
    {
        if(pStatPrms->pCellStatPrms[i].nTxAnt == 0 || pStatPrms->pCellStatPrms[i].nTxAnt > CUPHY_CSIRS_MAX_ANTENNA_PORTS)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported number of transmit antennas {} in cell {} must be in [1, {}]", 
                      pStatPrms->pCellStatPrms[i].nTxAnt, i, CUPHY_CSIRS_MAX_ANTENNA_PORTS);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if(pStatPrms->pCellStatPrms[i].nPrbDlBwp == 0 || pStatPrms->pCellStatPrms[i].nPrbDlBwp > MAX_N_PRBS_SUPPORTED)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported nPrbDlBwp {} in cell {} must be in [1, {}]", 
                      pStatPrms->pCellStatPrms[i].nPrbDlBwp, i, MAX_N_PRBS_SUPPORTED);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

template <fmtlog::LogLevel log_level>
void CsirsTx::printCsirsConfig(const cuphyCsirsStatPrms_t& cell_static_params, const cuphyCsirsDynPrms_t& dyn_params)
{
    NVLOG_FMT(log_level, NVLOG_CSIRS, "CSIRS pipeline parameters: ");

    int total_rrc_params = 0;
    for(int i = 0; i < dyn_params.nCells; i++)
    {
        const cuphyCsirsCellDynPrm_t& params = dyn_params.pCellParam[i];
        NVLOG_FMT(log_level, NVLOG_CSIRS, "------------------------------------------------------");
        NVLOG_FMT(log_level, NVLOG_CSIRS, "pCellParam Index:  {:5d}", i);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "rrcParamsOffset:   {:5d}", params.rrcParamsOffset);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "nRrcParams:        {:5d}", params.nRrcParams);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "slotBufferIdx:     {:5d}", params.slotBufferIdx);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "cellPrmStatIdx:    {:5d}", params.cellPrmStatIdx);
        // only the nPrbDlBwp field from the static parameters is currently used, so print only that
        NVLOG_FMT(log_level, NVLOG_CSIRS, "cell's {} nPrbDlBwp:{:5d} (from static parameters)", params.cellPrmStatIdx, cell_static_params.pCellStatPrms[params.cellPrmStatIdx].nPrbDlBwp);
        total_rrc_params += params.nRrcParams;
    }

    NVLOG_FMT(log_level, NVLOG_CSIRS, "CSIRS TX pipeline with {} precoded CSIRS RRC", dyn_params.nPrecodingMatrices);
    for(int i = 0; i < total_rrc_params; ++i)
    {
        const cuphyCsirsRrcDynPrm_t& params = dyn_params.pRrcDynPrm[i];
        NVLOG_FMT(log_level, NVLOG_CSIRS, "------------------------------------------------------");
        NVLOG_FMT(log_level, NVLOG_CSIRS, "pRrcDynPrm Index: {:5d}", i);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "startRb:          {:5d}", params.startRb);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "nRb:              {:5d}", params.nRb);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "csiType:          {:5d}", params.csiType);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "row:              {:5d}", params.row);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "freqDomain:       {:5d}", params.freqDomain);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "symbL0:           {:5d}", params.symbL0);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "symbL1:           {:5d}", params.symbL1);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "cdmType:          {:5d}", params.cdmType);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "freqDensity:      {:5d}", params.freqDensity);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "scrambId:         {:5d}", params.scrambId);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "beta:             {:.2f}", params.beta);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "idxSlotInFrame:   {:5d}", params.idxSlotInFrame);
        NVLOG_FMT(log_level, NVLOG_CSIRS, "enablePrcdBf:     {:5d}", params.enablePrcdBf);
        if(params.enablePrcdBf)
        {
            NVLOG_FMT(log_level, NVLOG_CSIRS, "pmwPrmIdx:        {:5d}", params.pmwPrmIdx);
            NVLOG_FMT(log_level, NVLOG_CSIRS, "nPorts :          {:5d}", dyn_params.pPmwParams[params.pmwPrmIdx].nPorts);
            std::stringstream matrix_row;
            matrix_row.precision(5);
            matrix_row << "Precoding Matrix:      ";
            for(int idx = 0; idx < dyn_params.pPmwParams[params.pmwPrmIdx].nPorts; idx++)
            {
                if (idx != 0) matrix_row << ", ";
                matrix_row << std::fixed << "{" << (float)dyn_params.pPmwParams[params.pmwPrmIdx].matrix[idx].x << ", " <<  (float)dyn_params.pPmwParams[params.pmwPrmIdx].matrix[idx].y << "}";
            }
            NVLOG_FMT(log_level, NVLOG_CSIRS, "{}", matrix_row.str());
        }
    }
}

cuphyStatus_t CsirsTx::expandParameters(cuphyCsirsDynPrms_t* dyn_params,
                                        cudaStream_t         cuda_strm)
{
    if(dyn_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "expandParameters() got cuphyCsirsDynPrms_t nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    //printCsirsConfig<fmtlog::DBG>(*static_params, *dyn_params);
    numCells = dyn_params->nCells;

    if (numCells > maxCellsPerSlot) {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Number of cells passed in CSI-RS setup {} > nMaxCellsPerSlot from static parameters {}.", numCells, maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    numParams= 0;
    for (int j = 0; j < numCells; j++)
    {
        numParams += dyn_params->pCellParam[j].nRrcParams;
    }
    updateWorkspaceOffsets(numCells, numParams);
    updateWorkspacePtrs();

    if(numParams > (CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot))
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Number of parameters passed in CSI-RS setup ({}) is more than {}.", numParams, CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(dyn_params->nPrecodingMatrices > CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot) [[unlikely]]
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "{}: nPrecodingMatrices ({}) exceeds allocated capacity ({}).", __FUNCTION__, dyn_params->nPrecodingMatrices, CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(dyn_params->nPrecodingMatrices>0)
    {
        memcpy(h_pmw_params, dyn_params->pPmwParams, sizeof(cuphyPmWOneLayer_t) * dyn_params->nPrecodingMatrices);
    }

    for (int j = 0; j < numCells; j++)
    {
        uint16_t cell_index = dyn_params->pCellParam[j].slotBufferIdx; // assumption cell_index in [0, nCells) range.
        h_cell_tensor_addr[cell_index] = (__half2*)dyn_params->pDataOut->pTDataTx[cell_index].pAddr;

#if 0
        int num_dims;
        int dims[3];
        CUPHY_CHECK(cuphyGetTensorDescriptor(dyn_params->pDataOut->pTDataTx[cell_index].desc, 3, nullptr, &num_dims, dims, nullptr));
        h_cell_tensor_REs[cell_index] = dims[0];
#else
        uint16_t static_cell_index = dyn_params->pCellParam[j].cellPrmStatIdx; // should be in [0, nCells) range.
        h_cell_tensor_REs[cell_index] = static_params->pCellStatPrms[static_cell_index].nPrbDlBwp * CUPHY_N_TONES_PER_PRB;
#endif

        for(int i = dyn_params->pCellParam[j].rrcParamsOffset; i < dyn_params->pCellParam[j].rrcParamsOffset + dyn_params->pCellParam[j].nRrcParams; ++i)
        {
            if(dyn_params->pRrcDynPrm[i].row > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH)
            {
                NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Row in CSI-RS parameter is more than {}.", CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            CsirsParams* h_param = &(h_params[i]);
            // copy and expand config params to CSIRS params
            h_param->startRb = dyn_params->pRrcDynPrm[i].startRb;
            h_param->nRb = dyn_params->pRrcDynPrm[i].nRb;
            h_param->csiType = dyn_params->pRrcDynPrm[i].csiType;
            h_param->row = dyn_params->pRrcDynPrm[i].row;
            h_param->li[0] = dyn_params->pRrcDynPrm[i].symbL0;
            h_param->li[1] = dyn_params->pRrcDynPrm[i].symbL1;
            h_param->cdmType = dyn_params->pRrcDynPrm[i].cdmType;
            h_param->scrambId = dyn_params->pRrcDynPrm[i].scrambId;
            h_param->idxSlotInFrame = dyn_params->pRrcDynPrm[i].idxSlotInFrame;
            h_param->beta = dyn_params->pRrcDynPrm[i].beta;
            h_param->enablePrcdBf = dyn_params->pRrcDynPrm[i].enablePrcdBf;
            h_param->pmwPrmIdx = dyn_params->pRrcDynPrm[i].pmwPrmIdx;

            //update cell_index; needed to access the addr. of this cell's output tensor and its number of PRBs (freq. dimension)
            h_param->cell_index = cell_index;

            if ((h_param->csiType == cuphyCsiType_t::TRS) &&  \
                ((h_param->row != 1) || (dyn_params->pRrcDynPrm[i].freqDensity != 3)))
            {
                NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Invalid TRS config. Should have row = 1 and frequency density = 3.");
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            switch (h_param->cdmType)
            {
                case cuphyCdmType_t::NO_CDM:
                    h_param->seqIndexCount = 1;
                    break;
                case cuphyCdmType_t::CDM2_FD:
                    h_param->seqIndexCount = 2;
                    break;
                case cuphyCdmType_t::CDM4_FD2_TD2:
                    h_param->seqIndexCount = 4;
                    break;
                case cuphyCdmType_t::CDM8_FD2_TD4:
                    h_param->seqIndexCount = 8;
                    break;
                default:
                {
                    NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unknown cdmType");
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
            }

            h_param->rho = 0.0f;
            h_param->genEvenRB = 0;
            switch (dyn_params->pRrcDynPrm[i].freqDensity)
            {
                case 0:
                    h_param->rho = 0.5f;
                    h_param->genEvenRB = 1;
                    break;
                case 1:
                    h_param->rho = 0.5f;
                    h_param->genEvenRB = 0;
                    break;
                case 2:
                    h_param->rho = 1;
                    break;
                case 3:
                    h_param->rho = 3;
                    break;
                default:
                {
                    NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unknown freqDensity value");
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
            }

            uint8_t numPorts = csirsRowDataNumPorts[h_param->row - 1];
            if(numPorts == 1)
            {
                h_param->alpha = h_param->rho;
            }
            else
            {
                h_param->alpha = 2 * h_param->rho;
            }

            if(h_param->csiType == cuphyCsiType_t::ZP_CSI_RS)
            {
                h_param->beta = 0;
            }

            h_param->nTxAnt = static_params->pCellStatPrms[static_cell_index].nTxAnt;

            // fill ki
            uint16_t freqDomain = dyn_params->pRrcDynPrm[i].freqDomain;
            for(int i = 0; i < CUPHY_CSIRS_MAX_KI_INDEX_LENGTH && freqDomain > 0; ++i)
            {
                // rightmost(least significant) bit
                uint8_t ki = log2((freqDomain & (freqDomain - 1)) ^ freqDomain);
                switch (h_param->row)
                {
                    case 1:
                    case 2:
                        h_param->ki[i] = ki;
                        break;
                    case 4:
                        h_param->ki[i] = 4 * ki;
                        break;
                    default:
                        h_param->ki[i] = 2 * ki;
                }

                freqDomain = freqDomain & (freqDomain - 1);
            }
        }
    }

#if CHECK_CONFIG
    cuphyStatus_t check_config_status = checkConfig(h_params, numParams);
    if (check_config_status != CUPHY_STATUS_SUCCESS) {
        return check_config_status;
    }
#endif

    // create offset array used to define starting thread for each parameter set
    int offset = 0;
    for(int j = 0; j < numParams; ++j)
    {
        h_offsets[j] = offset;
        CsirsParams* h_param = &(h_params[j]);
        //int orig_numElements = h_param->nRb * rowData.lenKBarLBar * rowData.lenKPrime * rowData.lenLPrime;
        int numElements = h_param->nRb * ((h_param->row == 1) ? 3 : csirsRowDataNumPorts[h_param->row - 1]);

        // The lenKBarLBar * lenKPrime * lenLPrime has following possible values {1, 2, 3, 4, 8, 12, 16, 24, 32}
        // nRb can be 1 to 273
        // Max. numElements per parameter is 8736
        offset +=  (numElements + 31) & ~31; // aligned by warp size, so that two parameter sets are never part of same warp
    }

    // initialize last parameter with total threads which are needed on GPU
    h_offsets[numParams] = offset;

    CUDA_CHECK(cudaMemcpyAsync(d_workspace.get(), h_workspace.get(), workspace_offsets[N_CSIRS_COMPONENTS], cudaMemcpyHostToDevice, cuda_strm));
    return CUPHY_STATUS_SUCCESS;
}

void CsirsTx::createGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating.
    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. If the number of kernel parameters changes, the calls below should be updated.
    void* arg;
    void* kernelParams[9] = {&arg, &arg, &arg, &arg, &arg, &arg, &arg, &arg, &arg}; //use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode3ParamsDriver, 3, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode8ParamsDriver, 8, &kernelParams[0]));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genCsirsScramblingNode, m_graph, nullptr, 0, &m_emptyNode3ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genCsirsTfSignalNode, m_graph, &m_genCsirsScramblingNode, 1, &m_emptyNode8ParamsDriver));
    // The 2nd arg of the first kernel is int but using void* (whose size is greater), so the code is more generalizable.
    // Same for the second kernel whose 3rd and 5th arguments are int and uint32_t respectively.
}

void CsirsTx::updateGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_CSIRS, cuGraphExecKernelNodeSetParams(m_graphExec, m_genCsirsScramblingNode, &(m_genCsirsScramblingLaunchCfg.kernelNodeParamsDriver)));
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_CSIRS, cuGraphExecKernelNodeSetParams(m_graphExec, m_genCsirsTfSignalNode, &(m_genCsirsTfSignalLaunchCfg.kernelNodeParamsDriver)));
}

void CsirsTx::setKernelParams()
{
    m_genCsirsTfSignalLaunchCfg.totalNumThreadsLB = h_offsets[numParams]; // set lower bound of total number of threads for CSI-RS computation
    cuphyCsirsKernelSelect(&m_genCsirsScramblingLaunchCfg, &m_genCsirsTfSignalLaunchCfg, numParams);
    //---------------------------------------------
    // set kernel args for genScramblingKernel()
    m_genScramblingArgs[0] = d_params;
    m_genScramblingArgs[1] = &numParams;
    m_genScramblingArgs[2] = d_goldSeq.get();

    m_genCsirsScramblingLaunchCfg.kernelArgs[0] = &m_genScramblingArgs[0];
    m_genCsirsScramblingLaunchCfg.kernelArgs[1] = m_genScramblingArgs[1];   // &numParams
    m_genCsirsScramblingLaunchCfg.kernelArgs[2] = &m_genScramblingArgs[2];

    m_genCsirsScramblingLaunchCfg.kernelNodeParamsDriver.kernelParams = m_genCsirsScramblingLaunchCfg.kernelArgs;

    // set kernel args for genCsirsTfSignalKernel()
    m_genTfSignalArgs[0] = d_cell_tensor_addr;
    m_genTfSignalArgs[1] = d_params;
    m_genTfSignalArgs[2] = &numParams;
    m_genTfSignalArgs[3] = d_offsets;
    m_genTfSignalArgs[4] = &h_offsets[numParams];
    m_genTfSignalArgs[5] = d_goldSeq.get();
    m_genTfSignalArgs[6] = d_cell_tensor_REs;
    m_genTfSignalArgs[7] = d_pmw_params;

    m_genCsirsTfSignalLaunchCfg.kernelArgs[0] = &m_genTfSignalArgs[0];
    m_genCsirsTfSignalLaunchCfg.kernelArgs[1] = &m_genTfSignalArgs[1];
    m_genCsirsTfSignalLaunchCfg.kernelArgs[2] = m_genTfSignalArgs[2];   // &numParams
    m_genCsirsTfSignalLaunchCfg.kernelArgs[3] = &m_genTfSignalArgs[3];
    m_genCsirsTfSignalLaunchCfg.kernelArgs[4] = m_genTfSignalArgs[4];   // &h_offsets[numParams]
    m_genCsirsTfSignalLaunchCfg.kernelArgs[5] = &m_genTfSignalArgs[5];
    m_genCsirsTfSignalLaunchCfg.kernelArgs[6] = &m_genTfSignalArgs[6];
    m_genCsirsTfSignalLaunchCfg.kernelArgs[7] = &m_genTfSignalArgs[7];

    m_genCsirsTfSignalLaunchCfg.kernelNodeParamsDriver.kernelParams = m_genCsirsTfSignalLaunchCfg.kernelArgs;
    //---------------------------------------------

    //executable graph setup
    m_cudaGraphModeEnabled = (dynamic_params->procModeBmsk & CSIRS_PROC_MODE_GRAPHS) ? true : false;
    if(m_cudaGraphModeEnabled)
    {
        updateGraph();
    }

}

// Generate CSIRS symbols and map them to subcarriers
cuphyStatus_t CsirsTx::run(const cudaStream_t& cuda_strm)
{
    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult e = cuGraphLaunch(m_graphExec, cuda_strm);
        if (e != CUDA_SUCCESS) {
            const char* pErrStr;
            CUresult str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Invalid graph launch for CSI-RS ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }
    else
    {
        CUresult e = launch_kernel(m_genCsirsScramblingLaunchCfg.kernelNodeParamsDriver, cuda_strm);
        if(e != CUDA_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Invalid argument for genScramblingKernel launch.");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }

        e = launch_kernel(m_genCsirsTfSignalLaunchCfg.kernelNodeParamsDriver, cuda_strm);
        if(e != CUDA_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Invalid argument for genCsirsTfSignalKernel launch.");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

