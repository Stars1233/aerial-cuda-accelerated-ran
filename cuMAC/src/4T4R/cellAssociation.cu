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

#include "cumac.h"

// cuMAC namespace
namespace cumac {

template <typename Chann_T>
cellAssociation<Chann_T>::cellAssociation()
{
    // dynamic descriptors
    m_pCpuDynDesc = new caDynDescr_t;
    CUDA_CHECK_ERR(cudaMalloc((void **)&m_pGpuDynDesc, sizeof(caDynDescr_t)));

    // Kernel launch config
    m_pLaunchCfg = new launchCfg_t;
}

template <typename Chann_T>
void cellAssociation<Chann_T>::setup(cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam, cudaStream_t strm)
{
    m_pCpuDynDesc -> estH_fr    = cellGrpPrms -> estH_fr;
    m_pCpuDynDesc -> cellAssoc  = cellGrpPrms -> cellAssoc;
    m_pCpuDynDesc -> nUe        = cellGrpPrms -> nUe;
    // FIXME: currently assuming nCell = totNumCell
    // cellGrpPrms -> nCell       :   number of cooperate cells
    // simParam    -> totNumCell  :   number of total cells (cooperate + interference)
    m_pCpuDynDesc -> totNumCell = simParam -> totNumCell;
    m_pCpuDynDesc -> nPrbGrp    = cellGrpPrms -> nPrbGrp;
    m_pCpuDynDesc -> nBsAnt     = cellGrpPrms -> nBsAnt;
    m_pCpuDynDesc -> nUeAnt     = cellGrpPrms -> nUeAnt;
    CUDA_CHECK_ERR(cudaMemcpyAsync(m_pGpuDynDesc, m_pCpuDynDesc, sizeof(caDynDescr_t), cudaMemcpyHostToDevice, strm));

    // config kernel launch params
        // if (totNumCell or totNumCell * nPrbGrp) <= 1024, parallelize over nPrbGrp * totNumCell, using cellAssocParaPrbgCellKernel, blockDim = {nPrbGrp * totNumCell, 1, 1}, diff shared memory size
        // else if totNumCell <= 1024, parallelize  totNumCell, using cellAssocParaCellKernel, blockDim = {totNumCell, 1, 1}, diff shared memory size
        // else: Error, does not support over 1024 cells

    // config same kernel parameters: kernel input m_pGpuDynDesc, blockDim = {..., 1, 1}, gridDim = {nUe, 1, 1}, 
    m_pLaunchCfg->kernelArgs[0] = &m_pGpuDynDesc;
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pLaunchCfg->kernelNodeParamsDriver;
    kernelNodeParamsDriver.kernelParams   = &(m_pLaunchCfg->kernelArgs[0]);
    kernelNodeParamsDriver.blockDimY  = 1;
    kernelNodeParamsDriver.blockDimZ  = 1;
    kernelNodeParamsDriver.gridDimX   = m_pCpuDynDesc -> nUe;
    kernelNodeParamsDriver.gridDimY   = 1;
    kernelNodeParamsDriver.gridDimZ   = 1;
    kernelNodeParamsDriver.extra      = nullptr;

    // config different kernel parameters: func, blockDimX, sharedMemBytes
    if(((m_pCpuDynDesc -> nPrbGrp) * (m_pCpuDynDesc -> totNumCell)) <= 1024) // parallelizable over nPrbGrp * totNumCell
    {
        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_pLaunchCfg->kernelNodeParamsDriver.func, reinterpret_cast<void*>(cellAssocParaPrbgCellKernel)));
        kernelNodeParamsDriver.blockDimX = (m_pCpuDynDesc -> nPrbGrp) * (m_pCpuDynDesc -> totNumCell);
        kernelNodeParamsDriver.sharedMemBytes = sizeof(float)*(kernelNodeParamsDriver.blockDimX) + sizeof(uint16_t)*(m_pCpuDynDesc -> totNumCell);
    }
    else if(m_pCpuDynDesc -> totNumCell <= 1024) // parallelizable over totNumCell
    {
        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&m_pLaunchCfg->kernelNodeParamsDriver.func, reinterpret_cast<void*>(cellAssocParaCellKernel)));
        kernelNodeParamsDriver.blockDimX = m_pCpuDynDesc -> totNumCell;
        kernelNodeParamsDriver.sharedMemBytes = sizeof(float)*(kernelNodeParamsDriver.blockDimX) + sizeof(uint16_t)*(m_pCpuDynDesc -> totNumCell);
    }
    else // error, does not supported totNumCell
    {
        printf("Error: Max supported totalNumCell is 1024 but totNumCell = %d!", m_pCpuDynDesc -> totNumCell);
        exit(1);
    }
}

template <typename Chann_T>
cellAssociation<Chann_T>::~cellAssociation()
{
    delete m_pCpuDynDesc;
    CUDA_CHECK_ERR(cudaFree(m_pGpuDynDesc));
    delete m_pLaunchCfg;
}

template <typename Chann_T>
void cellAssociation<Chann_T>::run(cudaStream_t strm)
{
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_pLaunchCfg->kernelNodeParamsDriver;
    
    #ifdef CELLASSOCIATION_KERNEL_TIME_MEASURE_
    cudaEvent_t start, stop;
    CUDA_CHECK_ERR(cudaEventCreate(&start));
    CUDA_CHECK_ERR(cudaEventCreate(&stop));
    float milliseconds = 0;

    CUDA_CHECK_ERR(cudaEventRecord(start));
    for (int exeIdx = 0; exeIdx < numRunSchKnlTimeMsr; exeIdx++) {
    #endif
    CUDA_CHECK_RES(cuLaunchKernel(kernelNodeParamsDriver.func,
                                  kernelNodeParamsDriver.gridDimX,
                                  kernelNodeParamsDriver.gridDimY, 
                                  kernelNodeParamsDriver.gridDimZ,
                                  kernelNodeParamsDriver.blockDimX, 
                                  kernelNodeParamsDriver.blockDimY, 
                                  kernelNodeParamsDriver.blockDimZ,
                                  kernelNodeParamsDriver.sharedMemBytes,
                                  strm,
                                  kernelNodeParamsDriver.kernelParams,
                                  kernelNodeParamsDriver.extra));  
    #ifdef CELLASSOCIATION_KERNEL_TIME_MEASURE_
    }
    CUDA_CHECK_ERR(cudaEventRecord(stop));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("cell association on GPU elapsed time: = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif  
}

template <typename Chann_T>
uint8_t * cellAssociation<Chann_T>::getCellAssociaResGpu()
{ 
    return  m_pCpuDynDesc->cellAssoc; 
}

__global__ void cellAssocParaPrbgCellKernel(caDynDescr_t* pDynDescr)
{
    // cell parameters
    uint16_t nUe            =  pDynDescr -> nUe;
    uint16_t totNumCell     =  pDynDescr -> totNumCell;
    uint16_t nPrbGrp        =  pDynDescr -> nPrbGrp;
    uint8_t nBsAnt          =  pDynDescr -> nBsAnt;
    uint8_t nUeAnt          =  pDynDescr -> nUeAnt;
    auto * estH_fr          =  pDynDescr -> estH_fr;
    uint8_t * cellAssocRes  =  pDynDescr -> cellAssoc;

    extern __shared__ float sharedMemPtr[];
    float * assocMetric = sharedMemPtr; // store the metric per UE
    uint16_t * cellAssocIdx = (uint16_t*)&sharedMemPtr[totNumCell * nPrbGrp]; // store cell index for comparison

    uint16_t ueIdx = blockIdx.x;
    uint16_t prbgIdx = (threadIdx.x / totNumCell);
    uint16_t cellIdx = threadIdx.x - prbgIdx * totNumCell;
    // blockIdx.x = ueIdx, threadIdx.x = prbgIdx * totNumCell + cellIdx, [cell0_prb0, cell1_prb0, ... ]

    assocMetric[threadIdx.x] = 0.0f;

    int chanOffsetAnt = (prbgIdx * nUe * totNumCell + ueIdx * totNumCell + cellIdx ) * nBsAnt * nUeAnt;

    // obtain channel and sum over Tx and Rx
    for (uint8_t txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) 
    {
        for (uint8_t rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) 
        {
            float temp_gain = estH_fr[chanOffsetAnt].x * estH_fr[chanOffsetAnt].x + estH_fr[chanOffsetAnt].y * estH_fr[chanOffsetAnt].y;
            // alternative: calculate the expected data rate per cell
            // float temp_gain = log2(1+(estH_fr[chanOffsetAnt].x * estH_fr[chanOffsetAnt].x + estH_fr[chanOffsetAnt].y * estH_fr[chanOffsetAnt].y)/noise_sigma2);
            // calculate the sum of channel gain
            assocMetric [threadIdx.x] += temp_gain;
            chanOffsetAnt ++; // NOTE: assume row-major channel
        }
    }
    __syncthreads();

    // parallel reduction for sum over Prbg
    uint16_t h = nPrbGrp;
    uint16_t s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(threadIdx.x < (h-s) * totNumCell)
        {
            assocMetric[threadIdx.x] += assocMetric[threadIdx.x + s * totNumCell];
        }
        h = s; s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(threadIdx.x < totNumCell) // using threads [0, totNumCell - 1], where threadIdx.x = cellIdx
    {
        assocMetric[threadIdx.x] += assocMetric[threadIdx.x + totNumCell];
        cellAssocRes[cellIdx * nUe + ueIdx] = 0;

        // initialize cellAssocIdx with cellIdx
        cellAssocIdx[cellIdx] = cellIdx;
    }
    __syncthreads();

    // use the best channel gain for cell associateion, parallel reduction for find max, using threads [0, totNumCell - 1]
    h = totNumCell;
    s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(threadIdx.x < h-s)
        {
            if(assocMetric[threadIdx.x] < assocMetric[threadIdx.x + s])
            {
                assocMetric[threadIdx.x] = assocMetric[threadIdx.x + s];
                cellAssocIdx[threadIdx.x] = cellAssocIdx[threadIdx.x + s];
            }
        }
        h = s; 
        s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(threadIdx.x == 0) // Set cell association
    {
        if(assocMetric[0] < assocMetric[1])
        {
            cellAssocRes[cellAssocIdx[1] * nUe + ueIdx] = 1; 
        }
        else
        {
            cellAssocRes[cellAssocIdx[0] * nUe + ueIdx] = 1;
        }
    }
}

__global__ void cellAssocParaCellKernel(caDynDescr_t* pDynDescr)
{
    // cell parameters
    uint16_t nUe            =  pDynDescr -> nUe;
    uint16_t totNumCell     =  pDynDescr -> totNumCell;
    uint16_t nPrbGrp        =  pDynDescr -> nPrbGrp;
    uint8_t nBsAnt          =  pDynDescr -> nBsAnt;
    uint8_t nUeAnt          =  pDynDescr -> nUeAnt;
    auto * estH_fr          =  pDynDescr -> estH_fr;
    uint8_t * cellAssocRes  =  pDynDescr -> cellAssoc;

    extern __shared__ float sharedMemPtr[];
    float * assocMetric = sharedMemPtr; // store the metric per UE
    uint16_t * cellAssocIdx = (uint16_t*)&sharedMemPtr[totNumCell]; // store cell index for comparison

    uint16_t ueIdx = blockIdx.x;
    uint16_t cellIdx = threadIdx.x;

    assocMetric[cellIdx] = 0.0f;
    // obtain channel and sum over Prbg, Tx and Rx
    for(uint16_t prbgIdx = 0; prbgIdx < nPrbGrp; prbgIdx++)
    {
        int chanOffsetAnt = (prbgIdx * nUe * totNumCell + ueIdx * totNumCell + cellIdx ) * nBsAnt * nUeAnt;
        
        for (uint8_t txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) 
        {
            for (uint8_t rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) 
            {
                float temp_gain = estH_fr[chanOffsetAnt].x * estH_fr[chanOffsetAnt].x + estH_fr[chanOffsetAnt].y * estH_fr[chanOffsetAnt].y;
                // alternative: calculate the expected data rate per cell
                // float temp_gain = log2(1+(estH_fr[chanOffsetAnt].x * estH_fr[chanOffsetAnt].x + estH_fr[chanOffsetAnt].y * estH_fr[chanOffsetAnt].y)/noise_sigma2);
                // calculate the sum of channel gain
                assocMetric [cellIdx] += temp_gain; 
                chanOffsetAnt ++;  // NOTE: assume row-major channel
            }
        }
    }

    // initialize cellAssocIdx with cellIdx and cellAssocRes with all 0; using all threads since blockDim.x = totNumCell
    cellAssocIdx[cellIdx] = cellIdx;
    cellAssocRes[cellIdx * nUe + ueIdx] = 0;
    __syncthreads();

    // use the best channel gain for cell associateion, parallel reduction for find max
    uint16_t h = totNumCell;
    uint16_t s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(threadIdx.x < h-s)
        {
            if(assocMetric[threadIdx.x] < assocMetric[threadIdx.x + s])
            {
                assocMetric[threadIdx.x] = assocMetric[threadIdx.x + s];
                cellAssocIdx[threadIdx.x] = cellAssocIdx[threadIdx.x + s];
            }
        }
        h = s; 
        s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        if(assocMetric[0] < assocMetric[1]) // Set cell association
        {
            cellAssocRes[cellAssocIdx[1] * nUe + ueIdx] = 1; 
        }
        else
        {
            cellAssocRes[cellAssocIdx[0] * nUe + ueIdx] = 1;
        }
    }
}
}