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

typedef std::pair<float, uint16_t> pfMetricUeSel;

// dynamic descriptor for CPU kernels
typedef struct mcUeSelDynDescrCpu {
    //----------------- input buffers ----------------- 
    uint8_t*    numUeSchdPerCellTTIArr; // array of the numbers of UEs scheduled per TTI for each cell
    float*      wbSinr;
    uint16_t*   cellId; // IDs of coordinated cells
    uint8_t*    cellAssocActUe;
    float*      avgRatesActUe; // average data rates for all active UEs in the coordinated cells 
    uint32_t*   bufferSize; // buffer sizes of all active UEs in cell group
    
    // HARQ related buffers
    int8_t*     newDataActUe; 

    //----------------- output buffers ----------------- 
    uint16_t*   setSchdUePerCellTTI; // set of global IDs of the schedule UEs per cell per TTI

    //----------------- parameters -----------------
    uint16_t    nCell;
    uint16_t    nActiveUe; // number of active UEs for all coordinated cells
    uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
    uint8_t     nUeAnt;
    float       W; // frequency bandwidth (Hz) of a PRB group
    float       betaCoeff; // coefficient for improving cell edge UEs' performance in multi-cell scheduling
} mcUeSelDynDescrCpu_t;

class multiCellUeSelectionCpu {
public:
    // constructor 
    multiCellUeSelectionCpu(cumacCellGrpPrms* cellGrpPrms);

    // destructor
    ~multiCellUeSelectionCpu();

    multiCellUeSelectionCpu(multiCellUeSelectionCpu const&)            = delete;
    multiCellUeSelectionCpu& operator=(multiCellUeSelectionCpu const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms); 
    
    // run() function for per-TTI algorithm execution
    void run();

    // parameter/data buffer logging function for debugging purpose
    void debugLog();

private:
    // indicator for heterogeneous UE selction across cells
    uint8_t heteroUeSelCells;

    // indicator for HARQ
    uint8_t enableHarq;

    // dynamic descriptor
    std::unique_ptr<mcUeSelDynDescrCpu_t> pCpuDynDesc;

    void multiCellUeSelCpu();
    void multiCellUeSelCpu_hetero();
};

typedef struct multiCellUeSelectionCpu*     mcUeSelCpuHndl_t;
}