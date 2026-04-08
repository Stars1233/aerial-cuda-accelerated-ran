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
constexpr uint16_t numRunSchKnlTimeMsr = 1000;
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

multiCellMuUeSort::multiCellMuUeSort(cumacCellGrpPrms* cellGrpPrms)
{
   enableHarq                          = cellGrpPrms->harqEnabledInd;
      
   // allocate memory for dynamic descriptors
   pCpuDynDesc = std::make_unique<mcUeSortDynDescr_t>();
   CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcUeSortDynDescr_t)));

   pLaunchCfg = std::make_unique<launchCfg_t>();
}

multiCellMuUeSort::~multiCellMuUeSort()
{
   CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
}

static __global__ void multiCellMuUeSortKernel(mcUeSortDynDescr_t* pDynDescr)
{
   uint16_t cellIdx = blockIdx.x;
 
   __shared__ float weights[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];
 
   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      weights[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }
 
   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();
 
   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx]) {
         int storeIdx = atomicAdd(&nAssocUeFound, 1);
         uint8_t tempMuMimoInd = 0;
         if (pDynDescr->srsWbSnr[uIdx] >= pDynDescr->srsSnrThr) {
            tempMuMimoInd = 1;
         }
         pDynDescr->muMimoInd[uIdx] = tempMuMimoInd;

         ueIds[storeIdx] = uIdx; // global UE index

         if (pDynDescr->bufferSize != nullptr && pDynDescr->bufferSize[uIdx] == 0) {
            continue;
         }
 
         float dataRate = 0;
         for (int j = 0; j < pDynDescr->nUeAnt; j++) {
            dataRate += pDynDescr->W*log2f(1.0 + pDynDescr->wbSinr[uIdx*pDynDescr->nUeAnt + j]);
         }
         if (tempMuMimoInd == 1) { // feasible for MU-MIMO transmission
            weights[storeIdx] = pow(dataRate, pDynDescr->betaCoeff)* pDynDescr->muCoeff /pDynDescr->avgRatesActUe[uIdx];
         } else { // not feasible for MU-MIMO transmission
            weights[storeIdx] = pow(dataRate, pDynDescr->betaCoeff)/pDynDescr->avgRatesActUe[uIdx];
         }
      }
   }
   __syncthreads();
 
   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);
 
      bitonicSort(weights, ueIds, pow2N); // internal synchronization
   }

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nMaxActUePerCell; uIdx += blockDim.x) {
      pDynDescr->sortedUeList[cellIdx][uIdx] = ueIds[uIdx];
   }
}

static __global__ void multiCellMuUeSortKernel_harq(mcUeSortDynDescr_t* pDynDescr)
{
   uint16_t cellIdx = blockIdx.x;
 
   __shared__ float weights[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];
 
   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      weights[eIdx] = -1.0;
      ueIds[eIdx] = 0xFFFF;
   }
 
   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();
 
   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx]) {
         int storeIdx = atomicAdd(&nAssocUeFound, 1);

         uint8_t tempMuMimoInd = 0;
         if (pDynDescr->newDataActUe[uIdx] == 1) { // new data/initial transmission
            if (pDynDescr->srsWbSnr[uIdx] >= pDynDescr->srsSnrThr) {
               tempMuMimoInd = 1;
            }
         } // HARQ re-tx is always assigned to SU-MIMO
          
         pDynDescr->muMimoInd[uIdx] = tempMuMimoInd;

         ueIds[storeIdx] = uIdx; // global UE index
 
         if (pDynDescr->newDataActUe[uIdx] == 1) { // new transmission
            if (pDynDescr->bufferSize != nullptr && pDynDescr->bufferSize[uIdx] == 0) {
               continue;
            }

            float dataRate = 0;
            for (int j = 0; j < pDynDescr->nUeAnt; j++) {
               dataRate += pDynDescr->W*log2f(1.0 + pDynDescr->wbSinr[uIdx*pDynDescr->nUeAnt + j]);
            }
            if (tempMuMimoInd == 1) { // feasible for MU-MIMO transmission
               weights[storeIdx] = pow(dataRate, pDynDescr->betaCoeff)* pDynDescr->muCoeff /pDynDescr->avgRatesActUe[uIdx];
            } else { // not feasible for MU-MIMO transmission
               weights[storeIdx] = pow(dataRate, pDynDescr->betaCoeff)/pDynDescr->avgRatesActUe[uIdx];
            }
         } else { // re-transmission
            weights[storeIdx] = std::numeric_limits<float>::max();
         }
      }
   }
   __syncthreads();
 
   if (nAssocUeFound > 0) {
      uint16_t pow2N = smallestPow2(nAssocUeFound);
 
      bitonicSort(weights, ueIds, pow2N); // internal synchronization
   }

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nMaxActUePerCell; uIdx += blockDim.x) {
      pDynDescr->sortedUeList[cellIdx][uIdx] = ueIds[uIdx];
   }
}

void multiCellMuUeSort::kernelSelect()
{
   void* kernelFunc;

   if (enableHarq == 1) { // HARQ enabled
      kernelFunc = reinterpret_cast<void*>(multiCellMuUeSortKernel_harq);
   } else {
      kernelFunc = reinterpret_cast<void*>(multiCellMuUeSortKernel);
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

void multiCellMuUeSort::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                              cumacSchdSol*               schdSol,
                              cumacCellGrpPrms*           cellGrpPrms,
                              cudaStream_t                strm)
{
   pCpuDynDesc->nMaxActUePerCell       = cellGrpPrms->nMaxActUePerCell;
   pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe;
   pCpuDynDesc->nCell                  = cellGrpPrms->nCell; 
   pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp; 
   pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt; 
   pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
   pCpuDynDesc->W                      = cellGrpPrms->W;
   pCpuDynDesc->betaCoeff              = cellGrpPrms->betaCoeff;
   pCpuDynDesc->muCoeff                = cellGrpPrms->muCoeff;
   pCpuDynDesc->srsSnrThr              = cellGrpPrms->srsSnrThr;
   pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
   pCpuDynDesc->wbSinr                 = cellGrpPrms->wbSinr;
   pCpuDynDesc->srsWbSnr               = cellGrpPrms->srsWbSnr;
   pCpuDynDesc->srsUeMap               = cellGrpPrms->srsUeMap;
   pCpuDynDesc->avgRatesActUe          = cellGrpUeStatus->avgRatesActUe;
   if (enableHarq == 1) { // HARQ is enabled
      pCpuDynDesc->newDataActUe       = cellGrpUeStatus->newDataActUe; 
   } else {
      pCpuDynDesc->newDataActUe       = nullptr;
   }
   if (cellGrpUeStatus->bufferSize != nullptr) {
      pCpuDynDesc->bufferSize = cellGrpUeStatus->bufferSize;
   } else {
      pCpuDynDesc->bufferSize = nullptr;
   }
   pCpuDynDesc->muMimoInd              = schdSol->muMimoInd;
   pCpuDynDesc->sortedUeList           = schdSol->sortedUeList;

   numThrdPerBlk = 1024;
   numThrdBlk    = cellGrpPrms->nCell;

   CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcUeSortDynDescr_t), cudaMemcpyHostToDevice, strm));

   // select kernel 
   kernelSelect();
 
   pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void multiCellMuUeSort::run(cudaStream_t strm)
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
   printf("Multi-cell MU-MIMO UE sorting ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
   #endif 
}

void multiCellMuUeSort::debugLog()
{

}
}