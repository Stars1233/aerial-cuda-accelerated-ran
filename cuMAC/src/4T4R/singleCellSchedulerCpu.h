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
typedef struct scDynDescrCpu {
    uint16_t    cellId; // index of the cell that calling the single-cell scheduler
    uint8_t*    cellAssoc;
    int16_t*    allocSol; // type 0 allocate: -1 indicates unallocated
    cuComplex** estH_fr_perUeBuffer;
    cuComplex*  prdMat;
    float*      avgRates;

    // parameters
    uint16_t    nUe;
    uint16_t    nCell;
    uint16_t    nPrbGrp;
    uint8_t     nBsAnt;
    uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
    float       W;
    float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
} scDynDescrCpu_t;

class singleCellSchedulerCpu {
public:
    singleCellSchedulerCpu();
    ~singleCellSchedulerCpu();
    singleCellSchedulerCpu(singleCellSchedulerCpu const&)            = delete;
    singleCellSchedulerCpu& operator=(singleCellSchedulerCpu const&) = delete;

    void setup(uint16_t                    cellId, // index of the cell that calling the single-cell scheduler
               cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cumacSimParam*              simParam); // requires externel synchronization

    void run();

private:
    // CPU matrix operation algorithms
    cpuMatAlg*       matAlg;

    // dynamic descriptor 
    scDynDescrCpu_t* pCpuDynDesc;

    // precoder type
    uint8_t          precodingScheme;

    // allocate type
    uint8_t          allocType;

    // allocate type-0 kernels
    void singleCellSchedulerCpu_noPrdMmse();
    void singleCellSchedulerCpu_svdMmse();

    // allocate type-1 kernels
    void singleCellSchedulerCpu_type1_NoPrdMmse();

};

typedef struct singleCellSchedulerCpu*      scSchdCpuHndl_t;
}