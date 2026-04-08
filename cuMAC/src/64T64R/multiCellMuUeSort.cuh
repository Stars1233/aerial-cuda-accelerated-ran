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
typedef struct mcUeSortDynDescr{
    //----------------- input buffers (common for both DL and UL) ----------------- 
    uint8_t*    cellAssocActUe = nullptr;
    float*      wbSinr = nullptr;
    cuComplex** srsEstChan = nullptr;
    int32_t**   srsUeMap = nullptr;
    float*      avgRatesActUe = nullptr;
    int8_t*     newDataActUe = nullptr;
    float*      srsWbSnr = nullptr;
    uint32_t*   bufferSize = nullptr; // buffer sizes of all active UEs in cell group
        
    //----------------- output buffers (common for both DL and UL) ----------------- 
    uint8_t*    muMimoInd = nullptr;
    uint16_t**  sortedUeList = nullptr;

    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nMaxActUePerCell; // Maximum number of active UEs per cell. 
    uint16_t    nActiveUe; // Total number of active UEs in the coordinated cell group.
    uint16_t    nCell; // Total number of coordinated cells. 
    uint16_t    nPrbGrp; // the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
    uint8_t     nBsAnt; // Each RU’s number of TX & RX antenna ports. Value: 64
    uint8_t     nUeAnt; // Each active UE’s number of TX & RX antenna ports. Value: 2, 4
    float       W; // Frequency bandwidth (Hz) of a PRG.
    float       betaCoeff; // Coefficient for adjusting the cell-edge UEs' performance in multi-cell scheduling.
    float       muCoeff; // Coefficient for prioritizing UEs selected for MU-MIMO transmissions.
    float       srsSnrThr; // Threshold on SRS reported SNR in dB for determining the feasibility of MU-MIMO transmission. Value: a real number in unit of dB. Default value is 5.0 (dB).
} mcUeSortDynDescr_t;

class multiCellMuUeSort {
public:
    // default constructor
    multiCellMuUeSort(cumacCellGrpPrms* cellGrpPrms);

    // desctructor
    ~multiCellMuUeSort();

    multiCellMuUeSort(multiCellMuUeSort const&)            = delete;
    multiCellMuUeSort& operator=(multiCellMuUeSort const&) = delete;

    // setup() function for per-TTI scheduling
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cudaStream_t                strm); // requires externel synchronization

    // run() function for per-TTI scheduling
    void run(cudaStream_t strm);

    // logging function for debugging 
    void debugLog(); 

private:
    // indicator for HARQ
    uint8_t          enableHarq;

    // dynamic descriptors
    std::unique_ptr<mcUeSortDynDescr_t> pCpuDynDesc;
    mcUeSortDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect();
};

typedef struct multiCellMuUeSort*           mcMuUeSortHndl_t;

static __global__ void multiCellMuUeSortKernel(mcUeSortDynDescr_t* pDynDescr);

// HARQ enabled kernel
static __global__ void multiCellMuUeSortKernel_harq(mcUeSortDynDescr_t* pDynDescr);
}