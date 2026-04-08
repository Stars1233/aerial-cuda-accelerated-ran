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

typedef std::pair<float, uint16_t> prioUeSel;

// dynamic descriptor for CPU kernels
typedef struct rrUeSelDynDescrCpu {
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
} rrUeSelDynDescrCpu_t;

class roundRobinUeSelCpu {
public:
    // default constructor
    roundRobinUeSelCpu(cumacCellGrpPrms* cellGrpPrms);
    // uint16_t nActiveUe is the (maximum) total number of active UEs in all coordinated cells

    // destructor
    ~roundRobinUeSelCpu();
    roundRobinUeSelCpu(roundRobinUeSelCpu const&)            = delete;
    roundRobinUeSelCpu& operator=(roundRobinUeSelCpu const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
               cumacSchdSol*             schdSol,
               cumacCellGrpPrms*         cellGrpPrms); // requires externel synchronization     

    // run() function for per-TTI algorithm execution
    void run();

    // parameter/data buffer logging function for debugging purpose
    void debugLog();

private:
    // indicator for HARQ
    uint8_t enableHarq;

    // dynamic descriptor
    std::unique_ptr<rrUeSelDynDescrCpu_t> pCpuDynDesc;

    void rrUeSelCpu();

    // HARQ enabled function
    void rrUeSelCpu_harq();
};

typedef struct roundRobinUeSelCpu*          rrUeSelCpuHndl_t;
}