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
typedef struct caDynDescr {
    //----------------- input buffers ----------------- 
    cuComplex*  estH_fr; 

    //----------------- output buffers ----------------- 
    uint8_t*    cellAssoc;

    //----------------- parameters -----------------
    uint16_t    nUe;
    uint16_t    totNumCell; // total number of cells in the network including both coordinated and interfering cells
    uint16_t    nPrbGrp;
    uint8_t     nBsAnt;
    uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
} caDynDescr_t;

template <typename Chann_T> 
class cellAssociation{
public:
    cellAssociation();
    ~cellAssociation();
    cellAssociation(cellAssociation const&)            = delete;
    cellAssociation& operator=(cellAssociation const&) = delete;

    //void setup(cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm); // requires externel synchronization 
    void setup(cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, cudaStream_t strm);  
    void run(cudaStream_t strm); // run cell association, results store in GPU
    uint8_t * getCellAssociaResGpu(); // get the GPU pointer for cell association results

private:
    int m_numThrdPerBlk; // number of threads per block
    int m_numThrdBlk; // number of threadblacks
    // dynamic descriptors
    caDynDescr_t* m_pCpuDynDesc;
    caDynDescr_t* m_pGpuDynDesc;

    // launch configuration structure
    launchCfg_t*    m_pLaunchCfg;
};

template class cellAssociation<cuComplex>;
template class cellAssociation<__half2>;

// cell association kernel, parallelizable over nPrbGrp * totNumCell
__global__ void cellAssocParaPrbgCellKernel(caDynDescr_t* pDynDescr);

// cell association kernel, parallelizable over totNumCell
__global__ void cellAssocParaCellKernel(caDynDescr_t* pDynDescr);

// // Source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//     if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//     if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//     if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }
}