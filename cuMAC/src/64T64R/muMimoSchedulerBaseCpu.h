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

// dynamic descriptor for CPU kernels
typedef struct muSchBaseDynDescrCpu {
    //----------------- input buffers (common for both DL and UL) ----------------- 
    cuComplex** srsEstChan          = nullptr;
    int32_t**   srsUeMap            = nullptr;
    float*      sinVal              = nullptr;
    float*      ueTxPow             = nullptr;
    float*      bsTxPow             = nullptr;
    uint8_t*    cellAssocActUe      = nullptr;
    float*      avgRatesActUe       = nullptr;
    float*      noiseVarActUe       = nullptr;
    // HARQ related buffers
    int8_t*     newDataActUe        = nullptr; 
    int16_t*    allocSolLastTx      = nullptr; 
    uint8_t*    layerSelSolLastTx   = nullptr; 

    //----------------- output buffers (common for both DL and UL) ----------------- 
    uint16_t*   setSchdUePerCellTTI = nullptr;
    int16_t*    allocSol            = nullptr;
    uint8_t*    layerSelSol         = nullptr;
    uint8_t*    nSCID               = nullptr;

    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nCell;
    uint16_t    nPrbGrp;
    uint16_t    nActiveUe;
    uint8_t     nBsAnt;
    uint8_t     nUeAnt;
    uint16_t    numUeForGrpPerCell;
    uint8_t     numUeSchdPerCellTTI;
    float       W;
    float       zfCoeff;
    float       betaCoeff;
    float       sinValThr;
    uint8_t     nMaxUePerGrpUl;
    uint8_t     nMaxUePerGrpDl;
    uint8_t     nMaxLayerPerGrpUl;
    uint8_t     nMaxLayerPerGrpDl;
    uint8_t     nMaxLayerPerUeSuUl;
    uint8_t     nMaxLayerPerUeSuDl; 
    uint8_t     nMaxLayerPerUeMuUl;
    uint8_t     nMaxLayerPerUeMuDl;
} muSchBaseDynDescrCpu_t;

class muMimoSchedulerBaseCpu {
public:
    muMimoSchedulerBaseCpu();
    ~muMimoSchedulerBaseCpu();
    muMimoSchedulerBaseCpu(muMimoSchedulerBaseCpu const&)            = delete;
    muMimoSchedulerBaseCpu& operator=(muMimoSchedulerBaseCpu const&) = delete;

    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               uint8_t                     in_enableHarq,
               uint8_t                     in_dl = 1 /* default DL scheduling*/);
    // in_enableHarq: 0 - HARQ disabled, 1 - HARQ enabled
    // in_dl: 0 - for UL MU-MIMO scheduling, 1 - for DL MU-MIMO scheduling 
    // baseline UL MU-MIMO scheduler not available yet

    void run();

private:
    // dynamic descriptor
    std::unique_ptr<muSchBaseDynDescrCpu_t> pCpuDynDesc;

    // indicator for UL/DL scheduling
    uint8_t          dl;

    // allocate type
    uint8_t          allocType;

    // indicator for HARQ
    uint8_t          enableHarq;

    // CPU matrix operation algorithms
    std::unique_ptr<cpuMatAlg> matAlg;

    // scheduler kernel
    void muSchBaseCpu_type1_dl();
    void muSchBaseCpu_type1_dl_harq();
};

typedef struct muMimoSchedulerBaseCpu*      muMimoSchBaseHndl_t;   
}