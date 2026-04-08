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
typedef struct mcsSelDynDescrCpu {
    //----------------- input buffers (common for both DL and UL) ----------------- 
    int8_t*     cqiActUe; 
    int16_t*    allocSol; // -1 indicates unallocated
    int8_t*     tbErrLast; // per-UE indicator for TB decoding error of the last transmission: 0 - decoded correctly, 1 - decoding error, -1 - not scheduled for the last TTI
    ollaParam*  ollaParamArr;
    float*      wbSinr; // global wideband SINR array
    uint16_t*   setSchdUePerCellTTI; // global UE IDs of scheduled UEs
    float*      beamformGainLastTx;
    float*      beamformGainCurrTx;
    // HARQ related buffers
    int8_t*     newDataActUe;
    int16_t*    mcsSelSolLastTx; 

    //----------------- output buffers ----------------- 
    int16_t*    mcsSelSol; // MCS selection solution

    // ***********************************
    //----------------- parameters (common for both DL and UL) -----------------
    uint16_t    nUe;
    uint16_t    nCell;
    uint16_t    nPrbGrp;
    uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
    uint8_t     nBsAnt;
    float       mcsSelSinrCapThr;
    uint8_t     mcsSelLutType;
} mcsSelDynDescrCpu_t;

class mcsSelectionLUTCpu {
public:
    // constructor
    mcsSelectionLUTCpu(cumacCellGrpPrms*   cellGrpPrms);

    // destructor
    ~mcsSelectionLUTCpu(){};

    mcsSelectionLUTCpu(mcsSelectionLUTCpu const&)            = delete;
    mcsSelectionLUTCpu& operator=(mcsSelectionLUTCpu const&) = delete;

    // setup() function for per-TTI scheduling
    void setup(cumacCellGrpUeStatus*        cellGrpUeStatus,
               cumacSchdSol*                schdSol,
               cumacCellGrpPrms*            cellGrpPrms);

    // run() function for per-TTI scheduling
    void run();

    // for debugging
    void debugLog();

private:
    // CPU matrix operation algorithms
    std::unique_ptr<cpuMatAlg> matAlg;

    // indicator for UE reported CQI based MCS selection
    uint8_t          CQI; // 1 for UE reported CQI based, 0 otherwise

    // allocate type
    uint8_t          allocType;

    // indicator for HARQ
    uint8_t          enableHarq;

    // system parameters
    uint16_t         m_nUe; // the (maximum) total number of active UEs in all coordinated cells

    // OLLA parameters
    std::vector<ollaParam> ollaParamArr;

    // dynamic descriptor
    std::unique_ptr<mcsSelDynDescrCpu_t> pDynDescr;

    // MCS selection kernel for type-1 PRB allocation
    void mcsSelSinrRepKernel_type1_wbSinr();
    void mcsSelSinrRepKernel_type0_wbSinr();
};

typedef struct mcsSelectionLUTCpu*          mcsSelLUTCpuHndl_t;
}
