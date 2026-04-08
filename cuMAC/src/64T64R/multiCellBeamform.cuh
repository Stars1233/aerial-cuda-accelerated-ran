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
typedef struct mcBeamformDynDescr{
    //----------------- input buffers (common for both DL and UL) ----------------- 
    cuComplex** srsEstChan          = nullptr;
    int32_t**   srsUeMap            = nullptr;
    uint16_t*   setSchdUePerCellTTI = nullptr;
    int16_t*    allocSol            = nullptr;
    uint8_t*    layerSelSol         = nullptr;
    uint16_t*   ueOrderInGrp        = nullptr;

    //----------------- output buffers (common for both DL and UL) ----------------- 
    cuComplex*  prdMat = nullptr;
    float*      beamformGainCurrTx = nullptr;
    float*      bfGainPrgCurrTx = nullptr;  

    // synchronization variables    
    int*        numCompleteBlk;

    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nCell;
    uint8_t     nBsAnt;
    uint8_t     nUeAnt;
    uint16_t    nPrbGrp;
    uint16_t    numUeForGrpPerCell;
    float       zfCoeff;
    uint8_t     bfPowAllocScheme;
} mcBeamformDynDescr_t;

class multiCellBeamform {
public:
    // default constructor
    multiCellBeamform(cumacCellGrpPrms* cellGrpPrms);

    // destructor
    ~multiCellBeamform();

    multiCellBeamform(multiCellBeamform const&)            = delete;
    multiCellBeamform& operator=(multiCellBeamform const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cudaStream_t                strm); // requires external synchronization

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);    

private:
    // indicator for DL/UL
    uint8_t         DL; // 1 for DL, 0 for UL

    // indicator for PRB allocation type
    uint8_t         allocType; // 0 for type-0, 1 for type-1 

    // CPU buffer allocation
    void*           cpuDataBuff;

    // GPU buffer allocation
    uint8_t*        gpuDataBuff;

    // dynamic descriptors
    mcBeamformDynDescr_t* pCpuDynDesc;
    mcBeamformDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;
    
    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    int* numCompleteBlk_d; // variable in GPU global memory to indicate the number of thread blocks that have completed compute job
    int* numCompleteBlk_h; // storing zero value in CPU memory for initializing numCompleteBlk_d per setup call

    void kernelSelect();
};

typedef struct multiCellBeamform*           mcBeamformHndl_t;

static __global__ void multiCellBeamformKernel_rzf_dl(mcBeamformDynDescr_t* pDynDescr);
}