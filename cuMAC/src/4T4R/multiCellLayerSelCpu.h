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
typedef struct mcLayerSelDynDescrCpu {
    //----------------- input buffers (common for both DL and UL) ----------------- 
    int16_t*    allocSol; // PRB allocation solution
    float*      sinVal; // array of singular values
    // ** passing sinVal_asim for Aerial Sim
    uint16_t*   setSchdUePerCellTTI; // global UE IDs of scheduled UEs
    uint8_t*    cellAssoc; // only for type-0 allocation
    int8_t*     riActUe;

    //----------------- Aerial Sim specific data buffers -----------------
    cuComplex** srsEstChan;

    // HARQ related buffers
    int8_t*     newDataActUe;
    uint8_t*    layerSelSolLastTx; 

    //----------------- output buffers (common for both DL and UL) ----------------- 
    uint8_t*    layerSelSol; // layer selection solution for the selected UEs per TTI in the coordinated cells
        
    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nUe;
    uint16_t    nPrbGrp;
    uint8_t     nUeAnt;
    uint16_t    nCell;
    uint8_t     nTxAnt;
    uint8_t     nRxAnt;

    // parameters not included in the API data structures
    // (set up internally in cuMAC setup() function)
    uint16_t    nUePerBlk;
    float       sinValThr;
    float       corrThr;
} mcLayerSelDynDescrCpu_t;

class multiCellLayerSelCpu {
public:
    // default constructor
    multiCellLayerSelCpu(cumacCellGrpPrms* cellGrpPrms);

    // constructor for Aerial Sim
    multiCellLayerSelCpu(cumacCellGrpPrms* cellGrpPrms, uint8_t in_Asim); // in_Asim: indicator for Aerial Sim

    // desctructor
    ~multiCellLayerSelCpu();

    multiCellLayerSelCpu(multiCellLayerSelCpu const&)            = delete;
    multiCellLayerSelCpu& operator=(multiCellLayerSelCpu const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms); // requires externel synchronization

    // run() function for per-TTI algorithm execution
    void run();

private:
    // indicator for DL/UL
    uint8_t          DL; // 1 for DL, 0 for UL

    // allocate type
    uint8_t          allocType;

    // precoding type: 0 - no precoding, 1 - SVD precoding
    uint8_t          precodingScheme;

    // indicator for HARQ
    uint8_t          enableHarq;

    // Aerial Sim indicator
    uint8_t          Asim;

    // dynamic descriptor
    std::unique_ptr<mcLayerSelDynDescrCpu_t> pDynDescr;

    void mcLayerSelKernel_type0();
    void mcLayerSelKernel_type1();
    void mcLayerSelKernel_type1_harq();

    // Aerial Sim
    void mcLayerSelKernel_type1_cfr();
    void mcLayerSelKernel_type1_cfr_harq();
};

typedef struct multiCellLayerSelCpu*        mcLayerSelCpuHndl_t;
}