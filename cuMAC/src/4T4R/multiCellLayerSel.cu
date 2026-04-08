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

 multiCellLayerSel::multiCellLayerSel(cumacCellGrpPrms* cellGrpPrms)
 {
    Asim = 0;

    // allocate memory for dynamic descriptors
    pCpuDynDesc = std::make_unique<mcLayerSelDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcLayerSelDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    DL = cellGrpPrms->dlSchInd;

    enableHarq = cellGrpPrms->harqEnabledInd;
 }

 multiCellLayerSel::multiCellLayerSel(cumacCellGrpPrms* cellGrpPrms, uint8_t in_Asim)
 {
    Asim = in_Asim;

    // allocate memory for dynamic descriptors
    pCpuDynDesc = std::make_unique<mcLayerSelDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcLayerSelDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    DL = cellGrpPrms->dlSchInd;

    enableHarq = cellGrpPrms->harqEnabledInd;
 }

 multiCellLayerSel::~multiCellLayerSel()
 {
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
 }

 static __global__ void mcLayerSelKernel_type0(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t globalUidx = 0xFFFF;

   __shared__ uint16_t assocCellIdx[1024];

   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      for (uint16_t cIdx = prgIdx; cIdx < pDynDescr->nCell; cIdx += pDynDescr->nPrbGrp) {
         if (pDynDescr->cellAssoc[cIdx*pDynDescr->nUe + uIdx] == 1) {
            assocCellIdx[blockLocalUidx] = cIdx;
         }
      }
   }

   __shared__ uint8_t numLayers[1024];
   numLayers[threadIdx.x] = 0xFF;
   __syncthreads(); 

   if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
      if (pDynDescr->allocSol[prgIdx*pDynDescr->nCell + assocCellIdx[blockLocalUidx]] == uIdx) {
         int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
         float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;
         numLayers[threadIdx.x] = 1;
         if (maxSinValThr > 0) {
            for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
               if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                  numLayers[threadIdx.x] = lIdx + 1;
                  break;
               }
            }
         }
      }
   }
   __syncthreads(); 

   // parallel reduction to find the minimum number of layers across all allocated PRGs for each UE
   uint16_t h = pDynDescr->nPrbGrp;
   uint16_t s = ceilf(h*0.5f);
   uint8_t temp;
#pragma unroll
   while(s > 1) {
      if(prgIdx < (h - s)) {
         if (numLayers[threadIdx.x] > numLayers[threadIdx.x + s]) {
            temp = numLayers[threadIdx.x];
            numLayers[threadIdx.x] = numLayers[threadIdx.x + s];
            numLayers[threadIdx.x + s] = temp;
         }
      }
      h = s; 
      s = ceilf(h*0.5f);

      __syncthreads();
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0) {
      if (numLayers[threadIdx.x] < numLayers[threadIdx.x+1]) {
         pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x];
      } else {
         pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x+1];
      }
   }
 }

 static __global__ void mcLayerSelKernel_type1(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t globalUidx = 0xFFFF;
   uint16_t numAllocPrg = 0;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
      numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
   }

   __shared__ uint16_t numLayers[1024];
   numLayers[threadIdx.x] = 0;
   if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
      if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
         int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
         float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;
         numLayers[threadIdx.x] = 1;

         if (maxSinValThr > 0) {
            for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
               if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                  numLayers[threadIdx.x] = lIdx + 1;
                  break;
               }
            }
         }
      }
   }
   __syncthreads(); 

   // parallel reduction to find the average number of layers across all allocated PRGs for each UE
   uint16_t h = pDynDescr->nPrbGrp;
   uint16_t s = ceilf(h*0.5f);
#pragma unroll
   while(s > 1) {
      if(prgIdx < (h - s)) {
         numLayers[threadIdx.x] += numLayers[threadIdx.x + s];
      }
      h = s; 
      s = ceilf(h*0.5f);

      __syncthreads();
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0) {
      numLayers[threadIdx.x] += numLayers[threadIdx.x+1];
      if (numAllocPrg > 0) {
         pDynDescr->layerSelSol[uIdx] = floor(static_cast<float>(numLayers[threadIdx.x])/numAllocPrg);
      } else {
         pDynDescr->layerSelSol[uIdx] = 0xFF;
      }
   }
 }

 static __global__ void mcLayerSelKernel_ri(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t globalUidx = 0xFFFF;
   uint16_t numAllocPrg = 0;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
      numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0 && globalUidx < 0xFFFF) {
      int8_t riVal = pDynDescr->riActUe[globalUidx];
      if (riVal >= 1 && riVal <= pDynDescr->nUeAnt) { // valid RI
         if (numAllocPrg > 0) {
            pDynDescr->layerSelSol[uIdx] = static_cast<uint8_t>(riVal);
         } else {
            pDynDescr->layerSelSol[uIdx] = 0xFF;
         }
      } else {
         if (numAllocPrg > 0) {
            pDynDescr->layerSelSol[uIdx] = 1;
         } else {
            pDynDescr->layerSelSol[uIdx] = 0xFF;
         }
      }
   }
 }
/*
 static __global__ void mcLayerSelKernel_type1_cfr(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx       = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx                 = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx               = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t nBsUeAntPrd          = pDynDescr->nTxAnt*pDynDescr->nRxAnt;
   uint16_t nPrgBsUeAntPrd       = pDynDescr->nPrbGrp*nBsUeAntPrd;
   uint16_t globalUidx           = 0xFFFF;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
   }

   __shared__ uint8_t numLayers[1024];
   numLayers[threadIdx.x] = 0xFF;
   if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
      uint8_t cIdx = 0;
      for (uint8_t cellIdx = 0; cellIdx < pDynDescr->nCell; cellIdx++) {
         if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
            cIdx = cellIdx;
            break;
         }
      }

      if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
         uint32_t hMatStart = uIdx*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd;
         float numerator = 0;
         float norm1 = 0;
         float norm2 = 0;
         for (uint8_t aIdx = 0; aIdx < pDynDescr->nRxAnt; aIdx++) {
            cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][hMatStart + aIdx*pDynDescr->nTxAnt];
            cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][hMatStart + aIdx*pDynDescr->nTxAnt + 1];
            numerator += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
            norm1 += tmp1.x*tmp1.x + tmp1.y*tmp1.y;
            norm2 += tmp2.x*tmp2.x + tmp2.y*tmp2.y;
         }
         float denominator = sqrtf(norm1*norm2);
         numerator = fabsf(numerator/denominator);

         if (numerator < pDynDescr->corrThr) {
            numLayers[threadIdx.x] = 2;
         } else {
            numLayers[threadIdx.x] = 1;
         }
      }
   }
   __syncthreads(); 

   // parallel reduction to find the minimum number of layers across all allocated PRGs for each UE
   uint16_t h = pDynDescr->nPrbGrp;
   uint16_t s = ceilf(h*0.5f);
   uint8_t temp;
#pragma unroll
   while(s > 1) {
      if(prgIdx < (h - s)) {
         if (numLayers[threadIdx.x] > numLayers[threadIdx.x + s]) {
            temp = numLayers[threadIdx.x];
            numLayers[threadIdx.x] = numLayers[threadIdx.x + s];
            numLayers[threadIdx.x + s] = temp;
         }
      }
      h = s; 
      s = ceilf(h*0.5f);

      __syncthreads();
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0) {
      if (numLayers[threadIdx.x] < numLayers[threadIdx.x+1]) {
         pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x];
      } else {
         pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x+1];
      }
   }  
 }*/

 static __global__ void mcLayerSelKernel_type1_harq(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t globalUidx = 0xFFFF;
   uint16_t numAllocPrg = 0;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
      numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
   }

   __shared__ uint16_t numLayers[1024];
   numLayers[threadIdx.x] = 0;
   if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
      if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
         int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
         float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;
         numLayers[threadIdx.x] = 1;

         if (maxSinValThr > 0) {
            for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
               if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                  numLayers[threadIdx.x] = lIdx + 1;
                  break;
               }
            }
         }
      }
   }
   __syncthreads(); 

   // parallel reduction to find the average number of layers across all allocated PRGs for each UE
   uint16_t h = pDynDescr->nPrbGrp;
   uint16_t s = ceilf(h*0.5f);

   #pragma unroll
   while(s > 1) {
      if(prgIdx < (h - s)) {
         numLayers[threadIdx.x] += numLayers[threadIdx.x + s];
      }
      h = s; 
      s = ceilf(h*0.5f);

      __syncthreads();
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0 && globalUidx < 0xFFFF) {
      if (pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
         numLayers[threadIdx.x] += numLayers[threadIdx.x+1];

         if (numAllocPrg > 0) {
            pDynDescr->layerSelSol[uIdx] = floor(static_cast<float>(numLayers[threadIdx.x])/numAllocPrg);
         } else {
            pDynDescr->layerSelSol[uIdx] = 0xFF;
         }
      } else { // the UE is scheduled for re-transmission
         pDynDescr->layerSelSol[uIdx] = pDynDescr->layerSelSolLastTx[uIdx];
      }
   }
 }

 static __global__ void mcLayerSelKernel_ri_harq(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t globalUidx = 0xFFFF;
   uint16_t numAllocPrg = 0;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
      numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0 && globalUidx < 0xFFFF) {
      if (pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
         int8_t riVal = pDynDescr->riActUe[globalUidx];
         if (riVal >= 1 && riVal <= pDynDescr->nUeAnt) { // valid RI
            if (numAllocPrg > 0) {
               pDynDescr->layerSelSol[uIdx] = static_cast<uint8_t>(riVal);
            } else {
               pDynDescr->layerSelSol[uIdx] = 0xFF;
            }
         } else { // valid RI
            if (numAllocPrg > 0) {
               pDynDescr->layerSelSol[uIdx] = 1;
            } else {
               pDynDescr->layerSelSol[uIdx] = 0xFF;
            }
            
         }
      } else { // the UE is scheduled for re-transmission
         pDynDescr->layerSelSol[uIdx] = pDynDescr->layerSelSolLastTx[uIdx];
      }
   }
 }
/*
 static __global__ void mcLayerSelKernel_type1_cfr_harq(mcLayerSelDynDescr_t* pDynDescr)
 {
   uint16_t blockLocalUidx       = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
   uint16_t uIdx                 = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
   uint16_t prgIdx               = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
   uint16_t nBsUeAntPrd          = pDynDescr->nTxAnt*pDynDescr->nRxAnt;
   uint16_t nPrgBsUeAntPrd       = pDynDescr->nPrbGrp*nBsUeAntPrd;
   uint16_t globalUidx           = 0xFFFF;
   if (uIdx < pDynDescr->nUe) {
      globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
   }

   __shared__ uint8_t numLayers[1024];
   numLayers[threadIdx.x] = 0xFF;
   if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
      uint8_t cIdx = 0;
      for (uint8_t cellIdx = 0; cellIdx < pDynDescr->nCell; cellIdx++) {
         if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
            cIdx = cellIdx;
            break;
         }
      }

      if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
         uint32_t hMatStart = uIdx*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd;
         float numerator = 0;
         float norm1 = 0;
         float norm2 = 0;
         for (uint8_t aIdx = 0; aIdx < pDynDescr->nRxAnt; aIdx++) {
            cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][hMatStart + aIdx*pDynDescr->nTxAnt];
            cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][hMatStart + aIdx*pDynDescr->nTxAnt + 1];
            numerator += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
            norm1 += tmp1.x*tmp1.x + tmp1.y*tmp1.y;
            norm2 += tmp2.x*tmp2.x + tmp2.y*tmp2.y;
         }
         float denominator = sqrtf(norm1*norm2);
         numerator = fabsf(numerator/denominator);

         if (numerator < pDynDescr->corrThr) {
            numLayers[threadIdx.x] = 2;
         } else {
            numLayers[threadIdx.x] = 1;
         }
      }
   }
   __syncthreads(); 

   // parallel reduction to find the minimum number of layers across all allocated PRGs for each UE
   uint16_t h = pDynDescr->nPrbGrp;
   uint16_t s = ceilf(h*0.5f);
   uint8_t temp;
#pragma unroll
   while(s > 1) {
      if(prgIdx < (h - s)) {
         if (numLayers[threadIdx.x] > numLayers[threadIdx.x + s]) {
            temp = numLayers[threadIdx.x];
            numLayers[threadIdx.x] = numLayers[threadIdx.x + s];
            numLayers[threadIdx.x + s] = temp;
         }
      }
      h = s; 
      s = ceilf(h*0.5f);

      __syncthreads();
   }

   if (uIdx < pDynDescr->nUe && prgIdx == 0 && globalUidx < 0xFFFF) {
      if (pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
         if (numLayers[threadIdx.x] < numLayers[threadIdx.x+1]) {
            pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x];
         } else {
            pDynDescr->layerSelSol[uIdx] = numLayers[threadIdx.x+1];
         }
      } else { // the UE is scheduled for re-transmission
         pDynDescr->layerSelSol[uIdx] = pDynDescr->layerSelSolLastTx[uIdx];
      }
   }
 }*/

 void multiCellLayerSel::kernelSelect()
 {
    void* kernelFunc;

    if (Asim == 1) { // Aerial Sim
        if (allocType == 0) { 
            printf("Error: For Aerial Sim, GPU layer selection is only supported for type-1 allocation\n");
            return;
        }
        
        if (enableHarq == 1) { // HARQ enabled
            //if (precodingScheme == 0) { // no precoding
            //   kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type1_cfr_harq);
            //} else { // SVD precoding
               if (RI == 1) { // RI based
                  kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_ri_harq);
               } else {
                  kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type1_harq);
               }
            //}
        } else {
            //if (precodingScheme == 0) { // no precoding
            //   kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type1_cfr);
            //} else { // SVD precoding
            if (RI == 1) { // RI based
               kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_ri);
            } else {
               kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type1);
            }
            //}
        }
    } else {
        if (enableHarq == 1) { // HARQ enabled
            printf("Error: GPU layer selection is not supported for HARQ\n");
            return;
        }

        if (allocType == 0) { // type-0 allocation
            kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type0);
        } else { // type-1 allocation
            kernelFunc = reinterpret_cast<void*>(mcLayerSelKernel_type1);
        }
    }

    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
 }

 void multiCellLayerSel::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                               cumacSchdSol*               schdSol,
                               cumacCellGrpPrms*           cellGrpPrms,
                               uint8_t                     in_ri,
                               cudaStream_t                strm)
 {
    pCpuDynDesc->nUe                    = cellGrpPrms->nUe;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->sinValThr              = cellGrpPrms->sinValThr;
    allocType                           = cellGrpPrms->allocType;
    precodingScheme                     = cellGrpPrms->precodingScheme;
   
    RI = in_ri;

    if (RI == 1) { // RI based
       pCpuDynDesc->riActUe = cellGrpUeStatus->riActUe;
    } else {
       pCpuDynDesc->riActUe = nullptr;
    }

    if (Asim == 1) { // for Aerial Sim
        pCpuDynDesc->sinVal             = cellGrpPrms->sinVal_asim;
        pCpuDynDesc->srsEstChan       = cellGrpPrms->srsEstChan;
        pCpuDynDesc->corrThr            = cellGrpPrms->corrThr;
        pCpuDynDesc->cellAssoc          = cellGrpPrms->cellAssoc;
    } else {
        pCpuDynDesc->sinVal             = cellGrpPrms->sinVal;
        pCpuDynDesc->srsEstChan       = nullptr;
        if (allocType == 0) { // for type-0 allocation
           pCpuDynDesc->cellAssoc       = cellGrpPrms->cellAssoc;
        } else {
           pCpuDynDesc->cellAssoc       = nullptr;
        }
    }
    
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->layerSelSol            = schdSol->layerSelSol;
    
    if (DL == 1) { // DL
      pCpuDynDesc->nTxAnt = cellGrpPrms->nBsAnt;
      pCpuDynDesc->nRxAnt = cellGrpPrms->nUeAnt;
    } else { // UL  
      pCpuDynDesc->nTxAnt = cellGrpPrms->nUeAnt;
      pCpuDynDesc->nRxAnt = cellGrpPrms->nBsAnt;
    }

    if (enableHarq == 1) { // HARQ enabled
      pCpuDynDesc->newDataActUe       = cellGrpUeStatus->newDataActUe;
      pCpuDynDesc->layerSelSolLastTx  = cellGrpUeStatus->layerSelSolLastTx;
    } else {
      pCpuDynDesc->newDataActUe       = nullptr;
      pCpuDynDesc->layerSelSolLastTx  = nullptr;
    }

    // launch geometry
    int numUePerBlk = floor(1024.0/cellGrpPrms->nPrbGrp);
    pCpuDynDesc->nUePerBlk = numUePerBlk;
    blockDim  = {static_cast<uint16_t>(numUePerBlk*cellGrpPrms->nPrbGrp), 1, 1};
    gridDim = {static_cast<uint16_t>(ceil(static_cast<float>(cellGrpPrms->nUe)/numUePerBlk)), 1, 1};

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcLayerSelDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect();

    pLaunchCfg->kernelArgs[0] = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 }           

 void multiCellLayerSel::run(cudaStream_t strm)
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
    printf("Multi-cell layer selection ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
#endif  
 }

 void multiCellLayerSel::debugLog()
 {

 }
}