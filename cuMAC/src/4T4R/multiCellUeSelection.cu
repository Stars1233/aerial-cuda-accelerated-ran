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

 multiCellUeSelection::multiCellUeSelection(cumacCellGrpPrms* cellGrpPrms)
 {
    pCpuDynDesc = std::make_unique<mcUeSelDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcUeSelDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    enableHarq = cellGrpPrms->harqEnabledInd;
 }

 multiCellUeSelection::~multiCellUeSelection()
 {
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
 }

 static __global__ void multiCellUeSelKernel(mcUeSelDynDescr_t* pDynDescr)
 {
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];

   __shared__ float avgRate[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      avgRate[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFoundLeft;
   __shared__ int nAssocUeFoundRight;
   if (threadIdx.x == 0) {
      nAssocUeFoundLeft = 0;
      nAssocUeFoundRight = 0;
   }
   __syncthreads();

   const float W = pDynDescr->W;
   const float betaCoeff = pDynDescr->betaCoeff;
   const int nActiveUe = pDynDescr->nActiveUe;
   const int nUeAnt = pDynDescr->nUeAnt;
   const int cellUeOffset = (int)cellIdx * nActiveUe;
   for (int uIdx = threadIdx.x; uIdx < nActiveUe; uIdx += blockDim.x) {
      bool isCandidate = false;
      float candidateRate = 0.0f;

      if (pDynDescr->cellAssocActUe[cellUeOffset + uIdx]) {
         if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[uIdx] == 1) { // new transmission
            if (pDynDescr->bufferSize == nullptr || pDynDescr->bufferSize[uIdx] != 0) {
               float dataRate = 0.0f;
               const int ueAntOffset = uIdx * nUeAnt;
               for (int j = 0; j < nUeAnt; j++) {
                  dataRate += log2f(1.0f + pDynDescr->wbSinr[ueAntOffset + j]);
               }
               dataRate *= W;
               constexpr float kEps = 1e-6f;
               candidateRate = powf(dataRate, betaCoeff) / fmaxf(pDynDescr->avgRatesActUe[uIdx], kEps);
               isCandidate = true;
            }
         } else { // re-transmission
            candidateRate = std::numeric_limits<float>::max();
            isCandidate = true;
         }
      }

      if (isCandidate) {
         int storeIdx = 0;
         if (uIdx & 0x1) {
            int rightOff = atomicAdd(&nAssocUeFoundRight, 1);
            storeIdx = maxNumActUePerCell_ - 1 - rightOff;
         } else {
            storeIdx = atomicAdd(&nAssocUeFoundLeft, 1);
         }
         avgRate[storeIdx] = candidateRate;
         ueIds[storeIdx] = uIdx; // global UE index
      }
   }
   __syncthreads();

   int nLeft = nAssocUeFoundLeft;
   int nRight = nAssocUeFoundRight;
   int rightStart = maxNumActUePerCell_ - nRight;
   for (int base = 0; base < nRight; base += blockDim.x) {
      int i = base + threadIdx.x;
      float tmpRate;
      uint16_t tmpId;
      if (i < nRight) {
         tmpRate = avgRate[rightStart + i];
         tmpId = ueIds[rightStart + i];
      }
      __syncthreads();
      if (i < nRight) {
         avgRate[nLeft + i] = tmpRate;
         ueIds[nLeft + i] = tmpId;
      }
      __syncthreads();
   }

   int nAssocUeFound = nLeft + nRight;

   // Clear stale right-group duplicates that fall outside the compacted range
   for (int base = 0; base < nRight; base += blockDim.x) {
      int i = base + threadIdx.x;
      if (i < nRight) {
         int srcIdx = rightStart + i;
         if (srcIdx >= nAssocUeFound) {
            avgRate[srcIdx] = -1.0f;
            ueIds[srcIdx] = 0xFFFF;
         }
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);

      bitonicSort(avgRate, ueIds, pow2N); // internal synchronization
   }

   if (threadIdx.x < pDynDescr->numUeSchdPerCellTTI) {
      pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeSchdPerCellTTI + threadIdx.x] = ueIds[threadIdx.x];
   }
/*
   if (blockIdx.x == 0 && threadIdx.x ==0) {
      for (int idx = 0; idx < maxNumActUePerCell_; idx++) {
         printf("idx = %d, UE global ID = %d, avgRate = %f\n", idx, ueIds[idx], avgRate[idx]);
      }
   }
*/
 }

 static __global__ void multiCellUeSelKernel_hetero(mcUeSelDynDescr_t* pDynDescr)
 {
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];

   __shared__ float avgRate[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      avgRate[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFoundLeft;
   __shared__ int nAssocUeFoundRight;
   if (threadIdx.x == 0) {
      nAssocUeFoundLeft = 0;
      nAssocUeFoundRight = 0;
   }
   __syncthreads();

   const float W = pDynDescr->W;
   const float betaCoeff = pDynDescr->betaCoeff;
   const int nActiveUe = pDynDescr->nActiveUe;
   const int nUeAnt = pDynDescr->nUeAnt;
   const int cellUeOffset = (int)cellIdx * nActiveUe;
   for (int uIdx = threadIdx.x; uIdx < nActiveUe; uIdx += blockDim.x) {
      bool isCandidate = false;
      float candidateRate = 0.0f;

      if (pDynDescr->cellAssocActUe[cellUeOffset + uIdx]) {
         if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[uIdx] == 1) { // new transmission
            if (pDynDescr->bufferSize == nullptr || pDynDescr->bufferSize[uIdx] != 0) {
               float dataRate = 0.0f;
               const int ueAntOffset = uIdx * nUeAnt;
               for (int j = 0; j < nUeAnt; j++) {
                  dataRate += log2f(1.0f + pDynDescr->wbSinr[ueAntOffset + j]);
               }
               dataRate *= W;
               constexpr float kEps = 1e-6f;
               candidateRate = powf(dataRate, betaCoeff) / fmaxf(pDynDescr->avgRatesActUe[uIdx], kEps);
               isCandidate = true;
            }
         } else { // re-transmission
            candidateRate = std::numeric_limits<float>::max();
            isCandidate = true;
         }
      }

      if (isCandidate) {
         int storeIdx = 0;
         if (uIdx & 0x1) {
            int rightOff = atomicAdd(&nAssocUeFoundRight, 1);
            storeIdx = maxNumActUePerCell_ - 1 - rightOff;
         } else {
            storeIdx = atomicAdd(&nAssocUeFoundLeft, 1);
         }
         avgRate[storeIdx] = candidateRate;
         ueIds[storeIdx] = uIdx; // global UE index
      }
   }
   __syncthreads();


   int nLeft = nAssocUeFoundLeft;
   int nRight = nAssocUeFoundRight;
   int rightStart = maxNumActUePerCell_ - nRight;
   for (int base = 0; base < nRight; base += blockDim.x) {
      int i = base + threadIdx.x;
      float tmpRate;
      uint16_t tmpId;
      if (i < nRight) {
         tmpRate = avgRate[rightStart + i];
         tmpId = ueIds[rightStart + i];
      }
      __syncthreads();
      if (i < nRight) {
         avgRate[nLeft + i] = tmpRate;
         ueIds[nLeft + i] = tmpId;
      }
      __syncthreads();
   }

   int nAssocUeFound = nLeft + nRight;

   // Clear stale right-group duplicates that fall outside the compacted range
   for (int base = 0; base < nRight; base += blockDim.x) {
      int i = base + threadIdx.x;
      if (i < nRight) {
         int srcIdx = rightStart + i;
         if (srcIdx >= nAssocUeFound) {
            avgRate[srcIdx] = -1.0f;
            ueIds[srcIdx] = 0xFFFF;
         }
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);

      bitonicSort(avgRate, ueIds, pow2N); // internal synchronization
   }

   if (threadIdx.x < pDynDescr->numUeSchdPerCellTTI) {
      if (threadIdx.x < pDynDescr->numUeSchdPerCellTTIArr[cIdx]) {
         pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeSchdPerCellTTI + threadIdx.x] = ueIds[threadIdx.x];
      } else {
         pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeSchdPerCellTTI + threadIdx.x] = 0xFFFF;
      }
   }
      
/*
   if (blockIdx.x == 0 && threadIdx.x ==0) {
      for (int idx = 0; idx < maxNumActUePerCell_; idx++) {
         printf("idx = %d, UE global ID = %d, avgRate = %f\n", idx, ueIds[idx], avgRate[idx]);
      }
   }
*/
 }

 void multiCellUeSelection::kernelSelect()
 {
   void* kernelFunc;

   if (heteroUeSelCells == 1) { // heterogeneous UE selection across cells
      kernelFunc = reinterpret_cast<void*>(multiCellUeSelKernel_hetero);
   } else { // homogeneous UE selection across cells
      kernelFunc = reinterpret_cast<void*>(multiCellUeSelKernel);
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

 void multiCellUeSelection::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                  cumacSchdSol*               schdSol,
                                  cumacCellGrpPrms*           cellGrpPrms,
                                  cudaStream_t                strm)
 {
   pCpuDynDesc->cellId                 = cellGrpPrms->cellId;
   pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
   pCpuDynDesc->wbSinr                 = cellGrpPrms->wbSinr;
   pCpuDynDesc->avgRatesActUe          = cellGrpUeStatus->avgRatesActUe;
   pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
   pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe; 
   pCpuDynDesc->numUeSchdPerCellTTI    = cellGrpPrms->numUeSchdPerCellTTI;
   pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
   pCpuDynDesc->W                      = cellGrpPrms->W;
   pCpuDynDesc->betaCoeff              = cellGrpPrms->betaCoeff;

   if (cellGrpPrms->numUeSchdPerCellTTIArr != nullptr) { // heterogeneous UE selection across cells
      heteroUeSelCells = 1;
      pCpuDynDesc->numUeSchdPerCellTTIArr = cellGrpPrms->numUeSchdPerCellTTIArr;
   } else { // homogeneous UE selection across cells
      heteroUeSelCells = 0;
      pCpuDynDesc->numUeSchdPerCellTTIArr = nullptr;
   }
   
   if (cellGrpUeStatus->bufferSize != nullptr) {
      pCpuDynDesc->bufferSize = cellGrpUeStatus->bufferSize;
   } else {
      pCpuDynDesc->bufferSize = nullptr;
   }

   if (enableHarq == 1) { // HARQ is enabled
      pCpuDynDesc->newDataActUe        = cellGrpUeStatus->newDataActUe; 
   } else {
      pCpuDynDesc->newDataActUe        = nullptr;
   }

   numThrdPerBlk = 1024;
   numThrdBlk    = cellGrpPrms->nCell;

   CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcUeSelDynDescr_t), cudaMemcpyHostToDevice, strm));

   // select kernel 
   kernelSelect();

   pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 }

 void multiCellUeSelection::run(cudaStream_t strm)
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
   printf("Multi-cell UE selection ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
   #endif    
 }

 void multiCellUeSelection::debugLog()
 {

 }
}

