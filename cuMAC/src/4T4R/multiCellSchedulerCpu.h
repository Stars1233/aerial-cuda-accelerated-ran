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

typedef std::pair<float, uint32_t> pfMetric;

// dynamic descriptor for CPU kernels
typedef struct mcDynDescrCpu {
    // buffers
    uint16_t*   cellId; // IDs of coordinated cells
    uint8_t*    cellAssoc;
    int16_t*    allocSol; // -1 indicates unallocated
    cuComplex*  estH_fr; 
    cuComplex*  prdMat;
    float*      avgRates;
    float*      sinVal; // singular values associated with each precoder matrix, per cell-UE link, per PRG, per layer (maximum number of layers is equal to the number of UE antennas) for the selected/scheduled UEs 
    float*      postEqSinr; // global post-eq SINR array
    uint16_t*   setSchdUePerCellTTI; // global UE IDs of scheduled UEs

    // parameters
    uint16_t    nUe;
    uint16_t    nCell;
    uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
    uint16_t    totNumCell; // number of all cells in the network. (not needed if channel buffer only contains channels within coordinated cells)
    uint16_t    nPrbGrp;
    uint8_t     nBsAnt;
    uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
    float       W;
    float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
    float       betaCoeff; // coefficient for improving cell edge UEs' performance in multi-cell scheduling
} mcDynDescrCpu_t;

class multiCellSchedulerCpu {
public:
    multiCellSchedulerCpu(cumacCellGrpPrms* cellGrpPrms);
    ~multiCellSchedulerCpu();
    multiCellSchedulerCpu(multiCellSchedulerCpu const&)            = delete;
    multiCellSchedulerCpu& operator=(multiCellSchedulerCpu const&) = delete;

    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cumacSimParam*              simParam,
               uint8_t                     in_columnMajor); 

    void run();

    void debugLog(); // for debugging only, printing out dynamic descriptor parameters

private:
    // indicator for DL/UL
    uint8_t          DL; // 1 for DL, 0 for UL

    // CPU matrix operation algorithms
    std::unique_ptr<cpuMatAlg> matAlg;

    // dynamic descriptor
    std::unique_ptr<mcDynDescrCpu_t> pCpuDynDesc;

    // precoder type
    uint8_t          precodingScheme;

    // allocate type
    uint8_t          allocType;

    // column-major or row-major channel matrix access: 0 - row major, 1 - column major
    uint8_t          columnMajor;

    // Aerial Sim indicator
    uint8_t          Asim;

    // record of computed PF metrics
    std::vector<pfMetric> pfRecord;

    // DL: 
    // type-0 allocation functions
    void multiCellSchedulerCpu_noPrdMmse();
    void multiCellSchedulerCpu_svdMmse();
    
    // type-1 allocation functions
    // multi-cell scheduler for Type-1 allocation (consecutive RB allocation) with no precoding/SVD precoding and MMSE-IRC equalizer
    // column-major channel access
    void multiCellSchedulerCpu_type1_NoPrdMmse_cm();
    void multiCellSchedulerCpu_type1_svdPrdMmse_cm();

    // row-major channel access
    void multiCellSchedulerCpu_type1_NoPrdMmse_rm();

    // Aerial Sim
    void multiCellSchedulerCpu_Asim_type1_svdPrdMmseIrc_rm();
    
    // Aerial Sim - HARQ
    void multiCellSchedulerCpu_Asim_type1_svdPrdMmseIrc_rm_harq();

    // UL:
    // multi-cell scheduler for Type-1 allocation (consecutive RB allocation)
    void multiCellSchedulerCpu_type1_svdPrdMmse_UL();
    void multiCellSchedulerCpu_type1_svdPrdMmse_harq_UL();
};

typedef struct multiCellSchedulerCpu*       mcSchdCpuHndl_t;
}
