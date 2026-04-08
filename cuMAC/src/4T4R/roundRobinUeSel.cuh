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
typedef struct rrUeSelDynDescr {
    //----------------- input buffers (common for both DL and UL) ----------------- 
    uint16_t*   cellId; // IDs of coordinated cells
    uint8_t*    cellAssocActUe;
    uint16_t*   prioWeightActUe;
    uint32_t*   bufferSize; // buffer sizes of all active UEs in cell group
    
    // HARQ related buffers
    int8_t*     newDataActUe; 

    //----------------- output buffers (common for both DL and UL) ----------------- 
    uint16_t*   setSchdUePerCellTTI; // set of global IDs of the schedule UEs per cell per TTI

    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nCell;
    uint16_t    nActiveUe; // number of active UEs for all coordinated cells
    uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
    uint16_t    prioWeightStep;
} rrUeSelDynDescr_t;

class multiCellRRUeSel {
public:
    // constructor 
    multiCellRRUeSel(cumacCellGrpPrms* cellGrpPrms);

    // destructor
    ~multiCellRRUeSel();

    multiCellRRUeSel(multiCellRRUeSel const&)            = delete;
    multiCellRRUeSel& operator=(multiCellRRUeSel const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
               cumacSchdSol*             schdSol,
               cumacCellGrpPrms*         cellGrpPrms,
               cudaStream_t              strm); // requires externel synchronization     

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);

private:
    // indicator for HARQ
    uint8_t enableHarq;

    // dynamic descriptors
    std::unique_ptr<rrUeSelDynDescr_t> pCpuDynDesc;
    rrUeSelDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect();
};

typedef struct multiCellRRUeSel*            mcRRUeSelHndl_t;

static __global__ void multiCellRRUeSelKernel(rrUeSelDynDescr_t* pDynDescr);
static __global__ void multiCellRRUeSelKernel_harq(rrUeSelDynDescr_t* pDynDescr);
}