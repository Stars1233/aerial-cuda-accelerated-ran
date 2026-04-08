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
typedef struct rrDynDescrCpu {
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
} rrDynDescrCpu_t;

class roundRobinSchedulerCpu {
public:
    roundRobinSchedulerCpu(cumacCellGrpPrms* cellGrpPrms);
    ~roundRobinSchedulerCpu();
    roundRobinSchedulerCpu(roundRobinSchedulerCpu const&)            = delete;
    roundRobinSchedulerCpu& operator=(roundRobinSchedulerCpu const&) = delete;

    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms);

    void run();

private:
    // dynamic descriptor
    std::unique_ptr<rrDynDescrCpu_t> pCpuDynDesc;

    // allocate type
    uint8_t          allocType;

    // indicator for HARQ
    uint8_t          enableHarq;

    // scheduler kernel
    void roundRobinSchedulerCpu_type0();
    void roundRobinSchedulerCpu_type1();

    void roundRobinSchedulerCpu_type1_harq();
};

typedef struct roundRobinSchedulerCpu*      rrSchdCpuHndl_t;
}