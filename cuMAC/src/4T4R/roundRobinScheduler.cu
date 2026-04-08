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

 // #define SCHEDULER_KERNEL_TIME_MEASURE_ 
 #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
 constexpr uint16_t numRunSchKnlTimeMsr 1000;
 #endif

 #define dir 0 // controls direction of comparator sorts

 inline __device__ int smallestPow2(int k)
 {
    if (k == 1) {
       return 2;
    } else {
       return 1 << (32-__clz(k-1));
    }
 }

 inline __device__ void bitonicSort(float* valueArr, uint16_t* idArr, uint16_t n)
 {
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              float t;
              uint16_t t_id;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == d) {
                  t = valueArr[pos];
                  valueArr[pos] = valueArr[pos + stride];
                  valueArr[pos + stride] = t;
                  t_id = idArr[pos];
                  idArr[pos] = idArr[pos + stride];
                  idArr[pos + stride] = t_id;
              }
           }
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           float t;
           uint16_t t_id;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == dir) {
               t = valueArr[pos];
               valueArr[pos] = valueArr[pos + stride];
               valueArr[pos + stride] = t;
             
               t_id = idArr[pos];
               idArr[pos] = idArr[pos + stride];
               idArr[pos + stride] = t_id;
           }
        }
    }

    __syncthreads(); 
 }

 multiCellRRScheduler::multiCellRRScheduler(cumacCellGrpPrms* cellGrpPrms)
 {
    pCpuDynDesc = std::make_unique<rrDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(rrDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    enableHarq = cellGrpPrms->harqEnabledInd;
 }

 multiCellRRScheduler::~multiCellRRScheduler()
 {
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
 }

 static __global__ void roundRobinSchedulerKernel_type1(rrDynDescr_t* pDynDescr)
 {
    uint16_t cIdx = blockIdx.x;
    uint16_t cellIdx = pDynDescr->cellId[cIdx];

    __shared__ uint16_t ueIds[maxNumSchdUePerCellTTI_];

    for (int eIdx = threadIdx.x; eIdx < maxNumSchdUePerCellTTI_; eIdx += blockDim.x) {
        ueIds[eIdx] = 0xFFFF;
    }

    __shared__ int nAssocUeFound;
    if (threadIdx.x == 0) {
        nAssocUeFound = 0;
    }
    __syncthreads();

    for (int uIdx = threadIdx.x; uIdx < pDynDescr->nUe; uIdx += blockDim.x) {
        if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
            int storeIdx = atomicAdd(&nAssocUeFound, 1);
            ueIds[storeIdx] = uIdx; // selected UE index
        }
    }
    __syncthreads();

    if (nAssocUeFound == 0) {
        return;
    }

    if (threadIdx.x == 0) {
        uint16_t numAllocRbgPerUe = floor(static_cast<float>(pDynDescr->nPrbGrp)/nAssocUeFound);

        uint16_t numRemainingRbg = pDynDescr->nPrbGrp - numAllocRbgPerUe*nAssocUeFound;

        uint16_t startRbgAlloc = 0;

        for (uint16_t tempUeIdx = 0; tempUeIdx < nAssocUeFound; tempUeIdx++) {
            uint16_t ueIdx = ueIds[tempUeIdx];

            if (numRemainingRbg > 0) {
                pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numAllocRbgPerUe + 1);
                startRbgAlloc += numAllocRbgPerUe + 1;
                numRemainingRbg--;

                pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                if (numAllocRbgPerUe > 0) {
                    pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                    pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numAllocRbgPerUe);
                    startRbgAlloc += numAllocRbgPerUe;
                    pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0;
                } else {
                    pDynDescr->allocSol[2*ueIdx]   = -1;
                    pDynDescr->allocSol[2*ueIdx+1] = -1;

                    uint32_t tempPrio = pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] + pDynDescr->prioWeightStep;
                    tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                    pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = static_cast<uint16_t>(tempPrio);
                }
            }
        }
    }
 }

 static __global__ void roundRobinSchedulerKernel_type1_harq(rrDynDescr_t* pDynDescr)
 {
    uint16_t cIdx = blockIdx.x;
    uint16_t cellIdx = pDynDescr->cellId[cIdx];

    __shared__ uint16_t assocUeIdxNewTx[maxNumSchdUePerCellTTI_];
    __shared__ uint16_t assocUeIdxReTx[maxNumSchdUePerCellTTI_];
    __shared__ uint16_t numResvdPrgReTx[maxNumSchdUePerCellTTI_];

    for (int eIdx = threadIdx.x; eIdx < maxNumSchdUePerCellTTI_; eIdx += blockDim.x) {
        assocUeIdxNewTx[eIdx]   = 0xFFFF;
        assocUeIdxReTx[eIdx]    = 0xFFFF;
        numResvdPrgReTx[eIdx]   = 0xFFFF;
    }

    __shared__ int numAssocUeNewTx;
    __shared__ int numAssocUeReTx;
    if (threadIdx.x == 0) {
        numAssocUeNewTx = 0;
        numAssocUeReTx  = 0;
    }
    __syncthreads();

    for (int uIdx = threadIdx.x; uIdx < pDynDescr->nUe; uIdx += blockDim.x) {
        if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
            if (pDynDescr->newDataActUe[pDynDescr->setSchdUePerCellTTI[uIdx]] == 1) { // the UE is scheduled for new transmission
                int storeIdx = atomicAdd(&numAssocUeNewTx, 1);
                assocUeIdxNewTx[storeIdx] = uIdx; // selected UE index
            } else { // the UE is scheduled for re-transmission
                int storeIdx = atomicAdd(&numAssocUeReTx, 1);
                assocUeIdxReTx[storeIdx] = uIdx; // selected UE index

                uint16_t tempResvdPrg = pDynDescr->allocSolLastTx[uIdx*2+1] - pDynDescr->allocSolLastTx[uIdx*2];

                numResvdPrgReTx[storeIdx] = tempResvdPrg;
            }
        }
    }
    __syncthreads();

    if (numAssocUeNewTx == 0 && numAssocUeReTx == 0) {
        return;
    }

    if (threadIdx.x == 0) {
        uint16_t numRemainingPrg = pDynDescr->nPrbGrp;

        uint16_t startRbgAlloc = 0;
        for (uint16_t uid = 0; uid < numAssocUeReTx; uid++) {
            uint16_t ueIdx = assocUeIdxReTx[uid];
            if (numRemainingPrg >= numResvdPrgReTx[uid]) {
                pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numResvdPrgReTx[uid]);
                startRbgAlloc += numResvdPrgReTx[uid];
                numRemainingPrg -= numResvdPrgReTx[uid];

                pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                pDynDescr->allocSol[2*ueIdx]   = -1;
                pDynDescr->allocSol[2*ueIdx+1] = -1;

                pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0xFFFF;
            }
        }

        uint16_t numAllocRbgPerUe = floor(static_cast<float>(numRemainingPrg)/numAssocUeNewTx);
        
        uint16_t numRemainingRbg = numRemainingPrg - numAllocRbgPerUe*numAssocUeNewTx;

        for (uint16_t uid = 0; uid < numAssocUeNewTx; uid++) {
            uint16_t ueIdx = assocUeIdxNewTx[uid];

            if (numRemainingRbg > 0) {
                pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numAllocRbgPerUe + 1);
                startRbgAlloc += numAllocRbgPerUe + 1;
                numRemainingRbg--;

                pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                if (numAllocRbgPerUe > 0) {
                    pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                    pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numAllocRbgPerUe);
                    startRbgAlloc += numAllocRbgPerUe;
                    pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = 0;
                } else {
                    pDynDescr->allocSol[2*ueIdx]   = -1;
                    pDynDescr->allocSol[2*ueIdx+1] = -1;

                    uint32_t tempPrio = pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] + pDynDescr->prioWeightStep;
                    tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                    pDynDescr->prioWeightActUe[pDynDescr->setSchdUePerCellTTI[ueIdx]] = static_cast<uint16_t>(tempPrio);
                }
            }
        }
    }
 }

 void multiCellRRScheduler::kernelSelect()
 {
    if (allocType == 0) { // type-0 PRB allocation
        printf("Error: GPU RR scheduler is only supported for type-1 PRB allocation.\n");
        return;
    }

    void* kernelFunc;

    if (enableHarq == 1) { // HARQ enabled
        kernelFunc = reinterpret_cast<void*>(roundRobinSchedulerKernel_type1_harq);
    } else {
        kernelFunc = reinterpret_cast<void*>(roundRobinSchedulerKernel_type1);
    }

    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

    // launch geometry
    gridDim  = {numThrdBlk, 1, 1};
    blockDim = {numThrdPerBlk, 1, 1};
  
    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver   = pLaunchCfg->kernelNodeParamsDriver;
  
    kernelNodeParamsDriver.blockDimX                  = blockDim.x;
    kernelNodeParamsDriver.blockDimY                  = blockDim.y;
    kernelNodeParamsDriver.blockDimZ                  = blockDim.z;
  
    kernelNodeParamsDriver.gridDimX                   = gridDim.x;
    kernelNodeParamsDriver.gridDimY                   = gridDim.y;
    kernelNodeParamsDriver.gridDimZ                   = gridDim.z;
  
    kernelNodeParamsDriver.extra                      = nullptr;
    kernelNodeParamsDriver.sharedMemBytes             = 0;
}

void multiCellRRScheduler::setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
                                 cumacSchdSol*             schdSol,
                                 cumacCellGrpPrms*         cellGrpPrms,
                                 cudaStream_t              strm)
{
    pCpuDynDesc->cellId                 = cellGrpPrms->cellId;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->cellAssoc              = cellGrpPrms->cellAssoc;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->prioWeightActUe        = cellGrpUeStatus->prioWeightActUe;
    pCpuDynDesc->nUe                    = cellGrpPrms->nUe; // total number of UEs across all coordinated cells
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell; // number of coordinated cells
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->prioWeightStep         = cellGrpPrms->prioWeightStep;
    allocType                           = cellGrpPrms->allocType;

    if (enableHarq == 1) { // HARQ enabled
        pCpuDynDesc->newDataActUe        = cellGrpUeStatus->newDataActUe;
        pCpuDynDesc->allocSolLastTx      = cellGrpUeStatus->allocSolLastTx;
    } else { // HARQ disabled
        pCpuDynDesc->newDataActUe        = nullptr;
        pCpuDynDesc->allocSolLastTx      = nullptr;
    }

    numThrdPerBlk = 1024;
    numThrdBlk    = cellGrpPrms->nCell;

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(rrDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel 
    kernelSelect();

    pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void multiCellRRScheduler::run(cudaStream_t strm)
{
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;
 
    #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
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
    #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
    }
    CUDA_CHECK_ERR(cudaEventRecord(stop));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Multi-cell Round-Robin PRG allocation ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif    
  }
}