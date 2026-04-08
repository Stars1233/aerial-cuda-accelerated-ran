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
 typedef struct scDynDescr {
   uint16_t    cellId; // index of the cell that calling the single-cell scheduler
   uint8_t*    cellAssoc;
   int16_t*    allocSol; // -1 indicates unallocated
   cuComplex** estH_fr_perUeBuffer;
   cuComplex*  prdMat;
   float*      avgRates;
   float*      pfMetricArr; // for storing computed PF metrices
   uint16_t*   pfIdArr; // for storing indices (indicating PRB and UE indecies) of computed PF metrices

   //----------------- parameters -----------------
   uint16_t    nUe;
   uint16_t    nCell;
   uint16_t    nPrbGrp;
   uint8_t     nBsAnt;
   uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
   float       W;
   float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
   uint16_t    nMaxSchdUePerRnd; // maximum number of UEs per cell that can be scheduled per round 
} scDynDescr_t;

 class singleCellScheduler {
 public:
    singleCellScheduler();
    ~singleCellScheduler();
    singleCellScheduler(singleCellScheduler const&)            = delete;
    singleCellScheduler& operator=(singleCellScheduler const&) = delete;

    void setup(uint16_t                    cellId, // index of the cell that calling the single-cell scheduler
               cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               cumacSimParam*              simParam,
               cudaStream_t                strm); // requires externel synchronization     

    void run(cudaStream_t strm);
    
 private:
    // allocate type: 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
    uint8_t allocType;

    // precoding type: 0 - no precoding, 1 - SVD precoding
    uint8_t precodingScheme;

    // dynamic descriptors
    scDynDescr_t* pCpuDynDesc;
    scDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    launchCfg_t* pLaunchCfg;
    void*        kernelFunc;

    void kernelSelect();
 };

 typedef struct singleCellScheduler*         scSchdHndl_t;

 // single-cell scheduler kernel for Type-0 allocation with no precoding and MMSE equalizer
 static __global__ void singleCellSchedulerKernel_noPrdMmse(scDynDescr_t* pDynDescr);

 // single-cell scheduler kernel for Type-0 allocation with SVD precoding and MMSE equalizer
 static __global__ void singleCellSchedulerKernel_svdMmse(scDynDescr_t* pDynDescr);

 // single-cell scheduler kernel for Type-1 allocation (consecutive RB allocation) with no precoding and MMSE equalizer
 static __global__ void singleCellSchedulerKernel_type1_NoPrdMmse(scDynDescr_t* pDynDescr);
}