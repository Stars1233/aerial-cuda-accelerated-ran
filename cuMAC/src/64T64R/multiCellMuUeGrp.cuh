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

#pragma once

#include "api.h"
#include "cumac.h"

// cuMAC namespace
namespace cumac {

// dynamic descriptor for CUDA kernels
typedef struct mcUeGrpDynDescr{
    //----------------- input buffers (common for both DL and UL) ----------------- 
    uint16_t**  sortedUeList        = nullptr;
    uint8_t*    cellAssocActUe      = nullptr;
    int8_t*     newDataActUe        = nullptr;
    float*      wbSinr              = nullptr;
    float*      srsWbSnr            = nullptr;
    cuComplex** srsEstChan          = nullptr;
    int32_t**   srsUeMap            = nullptr;
    int8_t*     riActUe             = nullptr; 
    uint8_t*    muMimoInd           = nullptr;
    float*      avgRatesActUe       = nullptr;
    int16_t*    allocSolLastTx      = nullptr; 
    uint8_t*    layerSelSolLastTx   = nullptr; 
    uint32_t*   bufferSize          = nullptr; // buffer sizes of all active UEs in cell group
    uint32_t*   currSlotIdxPerCell  = nullptr; // current slot index for each cell in the coordinated cell group
    uint32_t*   lastSchdSlotActUe   = nullptr; // last scheduled slot index for each active UE in the coordinated cells
    
    //----------------- output buffers (common for both DL and UL) ----------------- 
    uint16_t*   setSchdUePerCellTTI = nullptr;
    int16_t*    allocSol            = nullptr;
    uint8_t*    layerSelSol         = nullptr;
    uint8_t*    nSCID               = nullptr;
    uint16_t*   ueOrderInGrp        = nullptr;
    multiCellMuGrpList* muGrpList   = nullptr;

    //----------------- parameters (common for both DL and UL) -----------------
    uint8_t     muGrpUpdate; // trigger for performing MU-MIMO UE grouping in the current TTI    
    uint8_t     semiStatFreqAlloc; // indication for whether or not to enable semi-static subband allocation for SU UEs/MU UEGs
    uint16_t    numUeForGrpPerCell; // number of UEs considered for MU-MIMO UE grouping per TTI per cell 
    uint8_t     numUeSchdPerCellTTI; // total number of SU-MIMO UEs and MU-MIMO UE groups scheduled per TTI per cell 
    uint16_t    nMaxActUePerCell; // maximum number of active UEs per cell. 
    uint8_t     nMaxUePerGrpUl; // maximum number of UEs per UEG for UL
    uint8_t     nMaxUePerGrpDl; // maximium number of UEs per UEG for DL
    uint8_t     nMaxLayerPerGrpUl; // maximium number of layers per UEG for UL
    uint8_t     nMaxLayerPerGrpDl; // maximium number of layers per UEG for DL
    uint8_t     nMaxLayerPerUeSuUl; // maximium number of layers per UE for SU-MIMO UL
    uint8_t     nMaxLayerPerUeSuDl; // maximium number of layers per UE for SU-MIMO DL
    uint8_t     nMaxLayerPerUeMuUl; // maximium number of layers per UE for MU-MIMO UL
    uint8_t     nMaxLayerPerUeMuDl; // maximium number of layers per UE for MU-MIMO DL  
    uint8_t     nMaxUegPerCellDl; // maximum number of UEGs per cell for DL 
    uint8_t     nMaxUegPerCellUl; // maximum number of UEGs per cell for UL
    uint16_t    nActiveUe;
    uint16_t    nCell;
    uint16_t    nPrbGrp; // the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
    uint8_t     nBsAnt; // Each RU’s number of TX & RX antenna ports. Value: 64
    uint8_t     nUeAnt; // Each active UE’s number of TX & RX antenna ports. Value: 2, 4
    float       W; // Frequency bandwidth (Hz) of a PRG.
    float       zfCoeff; // Scalar coefficient used for regularizing the zero-forcing beamformer.
    float       betaCoeff;
    float       chanCorrThr; // threshold on the channel vector correlation value for UE grouping
    float       srsSnrThr;
    uint8_t     allocType; // PRB allocation type. Currently only support 1: consecutive type-1 allocation.
    float       muGrpSrsSnrMaxGap; // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG
    float       muGrpSrsSnrSplitThr; // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately
} mcUeGrpDynDescr_t;

class multiCellMuUeGrp {
public:
    // default constructor
    multiCellMuUeGrp(cumacCellGrpPrms* cellGrpPrms);

    // desctructor
    ~multiCellMuUeGrp();

    multiCellMuUeGrp(multiCellMuUeGrp const&)            = delete;
    multiCellMuUeGrp& operator=(multiCellMuUeGrp const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cudaStream_t                strm); // requires externel synchronization

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);

    // parameter/data buffer logging function for debugging purpose
    void debugLog(); // for debugging only, printing out dynamic descriptor parameters

private:
    // MU-MIMO UE grouping mode
    uint8_t          ueGrpMode;
    // 0: dynamic UE grouping per TTI
    // 1: flag-triggered UE grouping (controlled by the muUeGrpTrigger flag in cumacCellGrpPrms)

    // indicator for DL/UL
    uint8_t          DL; // 1 for DL, 0 for UL

    // indicator for HARQ
    uint8_t          enableHarq;

    // indicator for PRB allocation type
    uint8_t          allocType; // 0 for type-0, 1 for type-1 

    // dynamic descriptors
    std::unique_ptr<mcUeGrpDynDescr_t> pCpuDynDesc;
    mcUeGrpDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;
    
    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect();
};

typedef struct multiCellMuUeGrp*            mcMuUeGrpHndl_t;

static __global__ void multiCellMuUeGrpKernel_dl_dynPerTTI(mcUeGrpDynDescr_t* pDynDescr);
static __global__ void multiCellMuUeGrpKernel_dl_semiStatic(mcUeGrpDynDescr_t* pDynDescr); 
static __global__ void multiCellMuUeGrpKernel_ul_dynPerTTI(mcUeGrpDynDescr_t* pDynDescr);
}