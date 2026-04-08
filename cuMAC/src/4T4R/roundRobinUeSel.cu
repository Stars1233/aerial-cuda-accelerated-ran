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

 multiCellRRUeSel::multiCellRRUeSel(cumacCellGrpPrms* cellGrpPrms)
 {
    pCpuDynDesc = std::make_unique<rrUeSelDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(rrUeSelDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    enableHarq = cellGrpPrms->harqEnabledInd;
 }

 multiCellRRUeSel::~multiCellRRUeSel()
 {
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
 }

 static __global__ void multiCellRRUeSelKernel(rrUeSelDynDescr_t* pDynDescr)
 {
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];

   __shared__ float    prioWeight[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      prioWeight[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx] == 1) {
         if (pDynDescr->bufferSize != nullptr && pDynDescr->bufferSize[uIdx] == 0) {
            continue;
         }

         int storeIdx = atomicAdd(&nAssocUeFound, 1);
         prioWeight[storeIdx] = static_cast<float>(pDynDescr->prioWeightActUe[uIdx]);
         ueIds[storeIdx] = uIdx; // global UE index
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);

      bitonicSort(prioWeight, ueIds, pow2N); // internal synchronization
   }

   if (threadIdx.x < pDynDescr->numUeSchdPerCellTTI) {
      pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeSchdPerCellTTI + threadIdx.x] = ueIds[threadIdx.x];
   } 

   for (int uIdx = threadIdx.x + pDynDescr->numUeSchdPerCellTTI; uIdx < nAssocUeFound; uIdx += blockDim.x) {
      if (ueIds[uIdx] != 0xFFFF) {
         uint32_t tempPrio = pDynDescr->prioWeightActUe[ueIds[uIdx]] + pDynDescr->prioWeightStep;
         tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
         pDynDescr->prioWeightActUe[ueIds[uIdx]] = static_cast<uint16_t>(tempPrio);
      }
   }
 }

 static __global__ void multiCellRRUeSelKernel_harq(rrUeSelDynDescr_t* pDynDescr)
 {
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];

   __shared__ float    prioWeight[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      prioWeight[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx] == 1) {
         int storeIdx = atomicAdd(&nAssocUeFound, 1);

         if (pDynDescr->newDataActUe[uIdx] == 1) { // new transmission
            if (pDynDescr->bufferSize != nullptr && pDynDescr->bufferSize[uIdx] == 0) {
               continue;
            }

            prioWeight[storeIdx] = static_cast<float>(pDynDescr->prioWeightActUe[uIdx]);
            ueIds[storeIdx] = uIdx; // global UE index
         } else { // re-transmission
            prioWeight[storeIdx] = static_cast<float>(0xFFFF); // assign highest priority for re-tx
            ueIds[storeIdx] = uIdx; // global UE index
         }
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);

      bitonicSort(prioWeight, ueIds, pow2N); // internal synchronization
   }

   if (threadIdx.x < pDynDescr->numUeSchdPerCellTTI) {
      pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeSchdPerCellTTI + threadIdx.x] = ueIds[threadIdx.x];
   } 

   for (int uIdx = threadIdx.x + pDynDescr->numUeSchdPerCellTTI; uIdx <nAssocUeFound; uIdx += blockDim.x) {
      if (ueIds[uIdx] != 0xFFFF) {
         uint32_t tempPrio = pDynDescr->prioWeightActUe[ueIds[uIdx]] + pDynDescr->prioWeightStep;
         tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
         pDynDescr->prioWeightActUe[ueIds[uIdx]] = static_cast<uint16_t>(tempPrio);
      }
   }
 }

 void multiCellRRUeSel::kernelSelect()
 {
    void* kernelFunc;

    if (enableHarq == 1) { // HARQ enabled
        kernelFunc = reinterpret_cast<void*>(multiCellRRUeSelKernel_harq);
    } else {
        kernelFunc = reinterpret_cast<void*>(multiCellRRUeSelKernel);
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

 void multiCellRRUeSel::setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
                              cumacSchdSol*             schdSol,
                              cumacCellGrpPrms*         cellGrpPrms,
                              cudaStream_t              strm)
 {
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->cellId                 = cellGrpPrms->cellId;
    pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
    pCpuDynDesc->prioWeightActUe        = cellGrpUeStatus->prioWeightActUe;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe; 
    pCpuDynDesc->numUeSchdPerCellTTI    = cellGrpPrms->numUeSchdPerCellTTI;
    pCpuDynDesc->prioWeightStep         = cellGrpPrms->prioWeightStep;

    if (enableHarq == 1) { // HARQ is enabled
        pCpuDynDesc->newDataActUe        = cellGrpUeStatus->newDataActUe; 
    } else {
        pCpuDynDesc->newDataActUe        = nullptr;
    }

    if (cellGrpUeStatus->bufferSize != nullptr) {
        pCpuDynDesc->bufferSize = cellGrpUeStatus->bufferSize;
    } else {
        pCpuDynDesc->bufferSize = nullptr;
    }

    numThrdPerBlk = 1024;
    numThrdBlk    = cellGrpPrms->nCell;

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(rrUeSelDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel 
    kernelSelect();

    pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 }

 void multiCellRRUeSel::run(cudaStream_t strm)
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
    printf("Multi-cell Round-Robin UE selection ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif    
  }
}