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
typedef struct rrDynDescr {
    //----------------- input buffers (common for both DL and UL) ----------------- 
    uint16_t*   cellId; // IDs of coordinated cells
    uint8_t*    cellAssoc;
    uint16_t*   setSchdUePerCellTTI; // set of global IDs of the schedule UEs per cell per TTI
    uint16_t*   prioWeightActUe;
    
    // HARQ related buffers
    int8_t*     newDataActUe; 
    int16_t*    allocSolLastTx; 

    //----------------- output buffers (common for both DL and UL) ----------------- 
    int16_t*    allocSol; // -1 indicates unallocated

    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nUe;
    uint16_t    nCell;
    uint16_t    nPrbGrp;
    uint16_t    prioWeightStep;
} rrDynDescr_t;

class multiCellRRScheduler {
public:
    // constructor
    multiCellRRScheduler(cumacCellGrpPrms* cellGrpPrms);

    // destructor
    ~multiCellRRScheduler();

    multiCellRRScheduler(multiCellRRScheduler const&)            = delete;
    multiCellRRScheduler& operator=(multiCellRRScheduler const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
               cumacSchdSol*             schdSol,
               cumacCellGrpPrms*         cellGrpPrms,
               cudaStream_t              strm); // requires externel synchronization     

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);

private:
    // allocate type
    uint8_t          allocType;

    // indicator for HARQ
    uint8_t          enableHarq;

    // dynamic descriptors
    std::unique_ptr<rrDynDescr_t> pCpuDynDesc;
    rrDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect();
};

typedef struct multiCellRRScheduler*        mcRRSchdHndl_t;

static __global__ void roundRobinSchedulerKernel_type1(rrDynDescr_t* pDynDescr);

static __global__ void roundRobinSchedulerKernel_type1_harq(rrDynDescr_t* pDynDescr);
}