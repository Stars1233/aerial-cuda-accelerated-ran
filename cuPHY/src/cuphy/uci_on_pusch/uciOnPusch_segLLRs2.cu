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

#include "uciOnPusch_segLLRs2.hpp"
#include "descrambling.cuh"
#include <cassert>

//#define DEBUG_PRINT

using namespace cuphy_i;

constexpr int LLRS_BCO = 4; // to reduce bank conflicts //FixMe: tune it
constexpr int THREAD_STRIDE_HALFS = ((MAX_BITS_PER_RE + LLRS_BCO + 7) & ~7);  // multiple of 8

__global__ void
//__launch_bounds__(168, 8) // 168 = 12 * 14 is maximum CTA size //disabled, since occupancy is not the limiting factor here, but using LB may result in register spills
uciOnPuschSegLLRs2Kernel(uciOnPuschSegLLRs2DynDescr_t* pDesc)
{
   const uint16_t csi2Idx  = blockIdx.y;

   // Indices of RE being processed:
   const uint16_t prbIdx     = blockIdx.x;
   const uint16_t reIdx      = N_SC_PER_PRB * prbIdx + threadIdx.x;
   const uint8_t  symIdx     = threadIdx.y;

   // Indicies of Ue and UeGrp
   uint16_t ueGrpIdx = pDesc->csi2ToUserMapArray[csi2Idx].ueGrpIdx;
   uint16_t ueIdx    = pDesc->csi2ToUserMapArray[csi2Idx].ueIdx;

   // Allocation size:
   uint16_t  nPrb = pDesc->pUeGrpPrmsGpu[ueGrpIdx].nPrb;

   // User parameters:
   uint32_t  G_csi2       = pDesc->pUePrmsGpu[ueIdx].G_csi2;
  // uint32_t  G_harq      = pDesc->pUePrmsGpu[ueIdx].G_harq;
   uint16_t  nBitsCsi2    = pDesc->pUePrmsGpu[ueIdx].nBitsCsi2;
   uint8_t   nBitsPerQam  = pDesc->pUePrmsGpu[ueIdx].Qm;
   uint8_t   nLayers      = pDesc->pUePrmsGpu[ueIdx].Nl;
   uint32_t  cinit        = pDesc->pUePrmsGpu[ueIdx].cinit;
   uint32_t* pLayerMap    = pDesc->pUePrmsGpu[ueIdx].layer_map_array;
   uint8_t  isDataPresent = pDesc->pUePrmsGpu[ueIdx].isDataPresent;

   // HARQ and CSI-P2 allocation:
   harqAndCsi1RePrms_t& harqAndCsi1RePrm =  pDesc->harqAndCsi1RePrmsArray[csi2Idx];
   reGrid_t& rvdHarqReGrid               =  harqAndCsi1RePrm.rvdHarqReGrids[symIdx];
   reGrid_t& harqReGrid                  =  harqAndCsi1RePrm.harqReGrids[symIdx];
   reGrid_t& csi1ReGrid                  =  harqAndCsi1RePrm.csi1ReGrids[symIdx];
   uint16_t* pNumUnassignedResInSymbol   =  harqAndCsi1RePrm.nUnassignedResInSymbol;
   uint32_t* pNumUnassignedBitsInSymbol  =  harqAndCsi1RePrm.nUnassignedBitsInSymbol;
   uint8_t   nBitsPerRe                  =  harqAndCsi1RePrm.nBitsPerRe;
   uint32_t  nBitsPerSym                 =  harqAndCsi1RePrm.nBitsPerSym;
   bool      harqPunctFlag               =  harqAndCsi1RePrm.harqPunctFlag;
   bool*     pDmrsFlag                   =  harqAndCsi1RePrm.dmrsFlag;
   uint8_t   nSym                        =  harqAndCsi1RePrm.nSym;
   uint32_t* pDescramOffsets             =  harqAndCsi1RePrm.descramOffsets;

   // Input Buffer
   tensor_ref_any<CUPHY_R_16F>& tEqOutLLRs  =  pDesc->tEqOutLLRs[ueGrpIdx];

   // Output buffers
   __half* pSchLLRs  = pDesc->pUePrmsGpu[ueIdx].d_schAndCsi2LLRs;
   __half* pCsi2LLRs = pDesc->pUePrmsGpu[ueIdx].d_schAndCsi2LLRs + pDesc->pUePrmsGpu[ueIdx].G;

   // shared memory assignments
   __shared__ extern __half sh_buff[];
   __shared__ bool sh_dmrsFlag[MAX_ND_SUPPORTED];
   __shared__ uint32_t sh_numUnassignedBitsInSymbol[MAX_ND_SUPPORTED];
   __shared__ uint16_t sh_numUnassignedResInSymbol[MAX_ND_SUPPORTED];
   // Preload pLayerMap values so the per-layer IMAD computing gBase no longer
   // waits on a global LDG for each iteration. MAX_N_LAYERS_PUSCH is 8 so this is cheap.
   __shared__ uint32_t sh_pLayerMap[MAX_N_LAYERS_PUSCH];
   // Narrow reGrid preload: only the .nRes fields of csi1ReGrid/rvdHarqReGrid
   // are used as control-flow gates below (~465 stall samples combined on
   // the full-struct LDGs). Preloading just nRes (2 B × 14 × 2 arrays = 56 B
   // SMEM) lets those branches source from LDS instead of LDG. Full
   // reGrid_t preload was tried and regressed due to extended barrier wait;
   // this narrower version adds only 2 LDGs per thread (tid < 14) to the pre-sync region.
   __shared__ uint16_t sh_csi1NRes    [MAX_ND_SUPPORTED];
   __shared__ uint16_t sh_rvdHarqNRes [MAX_ND_SUPPORTED];

   int tid =  threadIdx.x + blockDim.x * threadIdx.y; //blockDim.x = 12, dataSymIdx = threadIdx.y

    // round up to 8 halfs (16B) for per-thread stride
   __half* reLLRs  = sh_buff + THREAD_STRIDE_HALFS * tid;

   // Hoist tEqOutLLRs.addr()/layout stride reads here so the global loads
   // issue before __syncthreads and their latency overlaps with the
   // cooperative preload + sync wait. By the time s0/s1/idx0 are consumed at
   // the s0==1 branch below, the data is already back — no long_sb there.
   auto tEqOutLLRsAdr = tEqOutLLRs.addr();
   auto layout        = tEqOutLLRs.layout();
   int  s0            = layout.strides[0];
   int  s1            = layout.strides[1];
   int  s2            = layout.strides[2];
   int  s3            = layout.strides[3];

   if (tid < nSym) {
      sh_dmrsFlag[tid] = pDmrsFlag[tid];
      sh_numUnassignedBitsInSymbol[tid] = pNumUnassignedBitsInSymbol[tid];
      sh_numUnassignedResInSymbol[tid]  = pNumUnassignedResInSymbol[tid];
      sh_csi1NRes[tid]    = harqAndCsi1RePrm.csi1ReGrids[tid].nRes;
      sh_rvdHarqNRes[tid] = harqAndCsi1RePrm.rvdHarqReGrids[tid].nRes;
   }
   if (tid < MAX_N_LAYERS_PUSCH) {
      sh_pLayerMap[tid] = pLayerMap[tid];
   }
   __syncthreads();

   // check for early exit
   if((symIdx >= nSym) || (prbIdx >= nPrb)){
      return;
   }

#if __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
   // For sm_120 and sm_121, we have limited occupancy due to shared memory size and
   // max thread count per SM, so we can afford to a few extra registers to early-load
   // values needed later. On sm_90, this would push us above 40 registers/thread,
   // which limits occupancy and decreases performance.
   const bool dmrsFlagSymIdx = sh_dmrsFlag[symIdx];
   if(dmrsFlagSymIdx && ((reIdx % 2 == 0) || (isDataPresent == 0))){
#else
   if(sh_dmrsFlag[symIdx] && ((reIdx % 2 == 0) || (isDataPresent == 0))){
#endif
      return;
   }

#if __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
   const uint32_t descramOffset = pDescramOffsets[symIdx];
#endif

   // Loads RE LLRs (s0/s1/s2/s3 and tEqOutLLRsAdr were hoisted pre-sync above)
   int  idx0          = s2 * reIdx + s3 * symIdx;

   if (s0 == 1) {
      // Async: issue cp.async.ca.shared.global copies now, commit the group,
      // and let the per-sym loop below overlap with the outstanding LLR
      // fetch. The wait_group happens just before the descramble reads.
      copyLLRsVec<true /*Async*/>(tEqOutLLRsAdr, s1, idx0, reLLRs, nBitsPerQam, nLayers, sh_pLayerMap);
   } else {
      for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
      {
            int idx1 = sh_pLayerMap[layerIdx] * s1 + idx0;
            for(uint8_t bitIdx = 0; bitIdx < nBitsPerQam; ++bitIdx)
            {
                  reLLRs[bitIdx + layerIdx * nBitsPerQam] = tEqOutLLRsAdr[bitIdx * s0 + idx1];
            }
      }
   }

   // Assign resources to CSI-P2 and SCH for symbols < symIdx
   reGrid_t csi2ReGrid;
   uint32_t nAssignedCsi2RmBits = 0;
   uint32_t schRmBufferOffset   = 0;
   uint16_t nUnassignedResNext  = sh_numUnassignedResInSymbol[0];

   for(int i = 0; i < symIdx; ++i)
   {
      uint32_t nUnassignedBits = sh_numUnassignedBitsInSymbol[i];
      uint16_t nUnassignedRes  = nUnassignedResNext;
      nUnassignedResNext       = sh_numUnassignedResInSymbol[i + 1];

      if((nAssignedCsi2RmBits < G_csi2) && (nUnassignedRes > 0))
      {
         assignReGrid(csi2ReGrid, 
                      nUnassignedRes, 
                      nUnassignedBits,
                      nAssignedCsi2RmBits,
                      G_csi2,
                      nBitsPerRe);

         updateNumUnassigned(csi2ReGrid,
                             nUnassignedRes,
                             nUnassignedBits,
                             nBitsPerRe); 
#ifdef DEBUG_PRINT
          if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (12 == threadIdx.y) && (0 == threadIdx.z))
          {
                for(int i = 0; i < 30; ++i)
                {
                   printf("\n dataSymIdx = %d, csi2ReGrid.rmBufferOffset = %d, G_csi2 = %d, nAssignedCsi2RmBits = %d", i, csi2ReGrid.rmBufferOffset, G_csi2, nAssignedCsi2RmBits);
                   __half LLR = pCsi2LLRs[i];
                   printf("\n csi2LLR[%d] = %f", i, static_cast<float>(LLR));
                }
          }
#endif
      }

      // remaining bits assigned to SCH
      if(sh_dmrsFlag[i])
      {
         schRmBufferOffset += nBitsPerSym >> 1;
      }else
      {
         schRmBufferOffset += nUnassignedBits;
      }
   }

   // Wait for the async LLR copy to drain before any access to reLLRs below.
   // The sym loop above doesn't touch reLLRs, so its cycles overlap with the
   // outstanding cp.async group.
   //
   // This wait is effectively a no-op with no previous async copy (i.e.,
   // when s0 != 1). Predicating the wait with a conditional resulted in a
   // performance regression, as did moving the wait lower below the gold
   // sequence generation. Thus, at least on RTX PRO 4500, this was the optimal
   // placement for the wait. Different GPUs may have different optimal wait points.
   cpLLRsVecWait<true /*Async*/>();

   // RE descrambling sequence
   uint32_t reDescSeq = 0;
#if __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
   if(dmrsFlagSymIdx){
#else
   const uint32_t descramOffset = pDescramOffsets[symIdx];
   const bool dmrsFlagSymIdx = sh_dmrsFlag[symIdx];
   if(dmrsFlagSymIdx){
#endif
      reDescSeq = descrambling::gold32n(cinit, descramOffset + reIdx/2*nBitsPerRe);
   }else{
      reDescSeq = descrambling::gold32n(cinit, descramOffset + reIdx*nBitsPerRe);
   }

   // If DMRS symbol, descramble and store the LLR
   if(dmrsFlagSymIdx){
      __half* pRmLLRBuff = pSchLLRs + schRmBufferOffset + reIdx / 2 * nBitsPerRe;
      descramAndStoreLLRs(reDescSeq, 0, pRmLLRBuff, nBitsPerRe, reLLRs);
      return;
   }

   // Assign resources to CSI-P2 for symIdx
   bool csi2AssignedToSymFlag = false;
   if(nAssignedCsi2RmBits < G_csi2)
   {
      csi2AssignedToSymFlag = true;
      assignReGrid(csi2ReGrid, 
                  sh_numUnassignedResInSymbol[symIdx],
                  sh_numUnassignedBitsInSymbol[symIdx],
                  nAssignedCsi2RmBits,
                  G_csi2,
                  nBitsPerRe);
   }

   
   uint16_t virtualReIdx        = reIdx;
   bool     rvdHarqReFlag       = false;
   bool     assignFlag          = false;
   uint16_t cumltNumAssignedRes = 0;
   bool     thisReIsPunctFlag   = false;

   if(sh_rvdHarqNRes[symIdx] > 0)
   {
      // check if RE reserved to HARQ. Compute the number of REs
      // reserved to HARQ < reIdx
      uint16_t cumltNumRvdRes = 0;
      checkIfReAssigned(rvdHarqReGrid,
                        reIdx,
                        rvdHarqReFlag,
                        cumltNumRvdRes);

      if(rvdHarqReFlag)
      {
         // If reserved, check if RE assigend to HARQ. Compute the number of
         // REs assigned to HARQ < reIdx.
         checkIfReAssigned(harqReGrid,
                           cumltNumRvdRes,
                           assignFlag,
                           cumltNumAssignedRes);
         if(assignFlag)
         {
            if(harqPunctFlag)
            {
               thisReIsPunctFlag = true;
            }else
            {
               return;
            }
         }
      }else
      {
         // Otherwise update virtualReIdx (RE index after all RVD HARQ REs removed)
         virtualReIdx = virtualReIdx - cumltNumRvdRes;
      }
   }

   if((sh_csi1NRes[symIdx] > 0) && (!rvdHarqReFlag))
   {
      // Check if RE assigend to CSI-P1. Compute the number of CSI-P1 REs
      // assigned < reIdx.
      checkIfReAssigned(csi1ReGrid,
                        virtualReIdx,
                        assignFlag,
                        cumltNumAssignedRes);
      if(assignFlag)
      {
         return;
      }else
      {
         // otherwise update virtualReIdx
         if(harqPunctFlag)
         {
            virtualReIdx = reIdx - cumltNumAssignedRes;        // RE index after all CSI-P1 REs removed (HARQ puncturing means REs assigned to both HARQ and SCH/CSI-P2)
         }else
         {
            virtualReIdx = virtualReIdx - cumltNumAssignedRes; // RE index after all HARQ + CSI-P1 REs removed
         }
      }
   }else
   {
      if(harqPunctFlag) 
      {
         virtualReIdx = reIdx; 
      }
   }

   // If punctured the bits belong to HARQ, set RE LLRs to zero
   if(thisReIsPunctFlag) 
   {
      for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
      {
         for(uint8_t bitIdx = 0; bitIdx < nBitsPerQam; ++bitIdx)
         {
            reLLRs[bitIdx + layerIdx * nBitsPerQam] = 0;
         }
      }
   }

   // Check if RE assigned to CSI-P2. If not than it is assigned to SCH.
   bool     asignCsi2Flag           = false;
   uint16_t cumltNumAssignedCsi2Res = 0;
   if(csi2AssignedToSymFlag) 
   {
      checkIfReAssigned(csi2ReGrid,
                        virtualReIdx,
                        asignCsi2Flag,
                        cumltNumAssignedCsi2Res);

      if(asignCsi2Flag){

         // determine if 1-bit spx scrambling used
         bool oneBitSpxFlag = false;
         if((nBitsCsi2 == 1) && (nBitsPerQam > 1)){
            oneBitSpxFlag = true;
         }

         // descramble and store the LLRs
         __half* pRmLLRBuff = pCsi2LLRs + csi2ReGrid.rmBufferOffset + cumltNumAssignedCsi2Res * nBitsPerRe;
         descramAndStoreLLRs(reDescSeq, oneBitSpxFlag, pRmLLRBuff, nBitsPerRe, reLLRs);
         return;
      }else{
         virtualReIdx = virtualReIdx - cumltNumAssignedCsi2Res; // RE index after all HARQ + CSI-P1 + CSI-P2 REs removed (if no HARQ puncturing)
                                                                // RE index after all CSI-P1 + CSI-P2 REs removed (if HARQ puncturing)
      }
   }

   // RE assigned to SCH 
   if(isDataPresent == 1)
   {
      __half* pRmBuffer = pSchLLRs + schRmBufferOffset + virtualReIdx * nBitsPerRe;
      descramAndStoreLLRs(reDescSeq, 0, pRmBuffer, nBitsPerRe, reLLRs);
   }

#ifdef DEBUG_PRINT
   __syncthreads();

   if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
   {
       for(int i = 0; i < 30; ++i)
       {
          __half LLR = pCsi2LLRs[i];
          printf("\n csi2LLR[%d] = %f", i, static_cast<float>(LLR));
       }
   }
#endif
}

void kernelSelect(uint16_t nCsi2Ues, csi2ToUserMap_t* pCsi2ToUserMapCpu, cuphyPuschRxUeGrpPrms_t*  pUeGrpPrmsCpu, cuphyUciOnPuschSegLLRs2LaunchCfg_t* pLaunchCfg)
{
   // determine max number of PRBs and OFDM symbols
   uint16_t MAX_N_PRBS      = 0;
   uint8_t  MAX_N_DATA_SYMS = 0;

   for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
   {
      uint16_t ueGrpIdx = pCsi2ToUserMapCpu[csi2Idx].ueGrpIdx;

      if(pUeGrpPrmsCpu[ueGrpIdx].nPrb > MAX_N_PRBS)
         MAX_N_PRBS = pUeGrpPrmsCpu[ueGrpIdx].nPrb;

      //if(pUeGrpPrmsCpu[ueGrpIdx].nDataSym > MAX_N_DATA_SYMS)
         //MAX_N_DATA_SYMS = pUeGrpPrmsCpu[ueGrpIdx].nDataSym;
      uint8_t nDmrsCdmGrpsNoData = pUeGrpPrmsCpu[ueGrpIdx].nDmrsCdmGrpsNoData;
      uint8_t nDataSym           = pUeGrpPrmsCpu[ueGrpIdx].nDataSym;
      uint8_t nDmrsSyms          = pUeGrpPrmsCpu[ueGrpIdx].nDmrsSyms;
      if(nDmrsCdmGrpsNoData==1)
      {
          nDataSym += nDmrsSyms;
      }
      if(nDataSym > MAX_N_DATA_SYMS)
         MAX_N_DATA_SYMS = nDataSym;
   }
   //printf("MAX_N_DATA_SYMS[%d]\n", MAX_N_DATA_SYMS);
   // launch geometry
   dim3 blockDim(N_SC_PER_PRB, MAX_N_DATA_SYMS); // One thread block covers one entire PRB: 12 subcarriers x MAX_N_DATA_SYMS
   dim3 gridDim(MAX_N_PRBS, nCsi2Ues);

   // The kernel's cooperative `tid < nSym` SMEM preloads
   // (sh_dmrsFlag, sh_numUnassigned*, sh_csi1NRes, sh_rvdHarqNRes) require
   // the block to contain at least `nSym` threads. Per-UE nSym is bounded
   // above by MAX_N_DATA_SYMS (computed in the loop above), so this check
   // guards future changes to blockDim that might shrink it below that bound.
   assert(blockDim.x * blockDim.y >= MAX_N_DATA_SYMS &&
          "uciOnPuschSegLLRs2Kernel: blockDim.x * blockDim.y must be >= MAX_N_DATA_SYMS "
          "(per-symbol cooperative preloads rely on tid covering [0, nSym))");

   // kernel (only one kernel option for now)
   void* kernelFunc = reinterpret_cast<void*>(uciOnPuschSegLLRs2Kernel);
  {MemtraceDisableScope md;cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);}

   // populate kernel parameters
   CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

   kernelNodeParamsDriver.blockDimX = blockDim.x;
   kernelNodeParamsDriver.blockDimY = blockDim.y;
   kernelNodeParamsDriver.blockDimZ = blockDim.z;

   kernelNodeParamsDriver.gridDimX = gridDim.x;
   kernelNodeParamsDriver.gridDimY = gridDim.y;
   kernelNodeParamsDriver.gridDimZ = gridDim.z;

   kernelNodeParamsDriver.extra          = nullptr;

   uint32_t dynShMem = blockDim.x * blockDim.y * THREAD_STRIDE_HALFS * sizeof(__half); // for reLLRs; LLRS_BCO for bank conflict

   kernelNodeParamsDriver.sharedMemBytes = (dynShMem + 15) & ~15; // ensure it is multiple of 16
}

void computeCsi1AndHarqReGrids(harqAndCsi1RePrms_t& harqAndCsiRePrms, PerTbParams& uePrms, cuphyPuschRxUeGrpPrms_t& ueGrpPrms)
{
   // RE allocation parameters:
   uint32_t* pNumUnassignedBitsInSymbol = harqAndCsiRePrms.nUnassignedBitsInSymbol;
   uint16_t* pNumUnassignedResInSymbol  = harqAndCsiRePrms.nUnassignedResInSymbol;
   reGrid_t* pRvdHarqReGrids            = harqAndCsiRePrms.rvdHarqReGrids;
   reGrid_t* pHarqReGrids               = harqAndCsiRePrms.harqReGrids;
   reGrid_t* pCsi1ReGrids               = harqAndCsiRePrms.csi1ReGrids;
   uint8_t&  nSym                       = harqAndCsiRePrms.nSym;
   bool*     pDmrsFlag                  = harqAndCsiRePrms.dmrsFlag;
   uint32_t* pDescramOffsets            = harqAndCsiRePrms.descramOffsets;

 
   // User parameters:
   uint32_t G_harq_rvd  = uePrms.G_harq_rvd;
   uint32_t G_harq      = uePrms.G_harq;
   uint32_t G_csi1      = uePrms.G_csi1;
   uint32_t nLayers     = uePrms.Nl;
   uint32_t nBitsPerQam = uePrms.Qm;
   uint8_t  nBitsPerRe  = nLayers * nBitsPerQam;
   uint16_t nBitsHarq   = uePrms.nBitsHarq;
   
   // User group parameters:
   uint8_t  nDataSym           =  ueGrpPrms.nDataSym;
   uint8_t  nDmrsSyms          =  ueGrpPrms.nDmrsSyms;
   uint8_t* pDataSymLoc        =  ueGrpPrms.dataSymLoc;
   uint8_t* pDmrsSymLoc        =  ueGrpPrms.dmrsSymLoc;
   uint16_t nPrb               =  ueGrpPrms.nPrb;
   uint8_t  nDmrsCdmGrpsNoData = ueGrpPrms.nDmrsCdmGrpsNoData;

   // Determine number of symbols containing SCH and/or UCI
   if(nDmrsCdmGrpsNoData == 1){
      nSym = nDataSym + nDmrsSyms;
   }else{
      nSym = nDataSym;
   }

   // Determine first symbol in slot containing SCH and/or UCI
   uint8_t startSymInSlot = 0;
   startSymInSlot = std::min(pDataSymLoc[0], pDmrsSymLoc[0]); // work for both nDmrsCdmGrpsNoData = 1 and nDmrsCdmGrpsNoData = 2 

   // Determine which symbols contain DMRS
   for(int symIdx = 0; symIdx < nSym; ++symIdx){
      pDmrsFlag[symIdx] = false;
   }

   if(nDmrsCdmGrpsNoData == 1)
   {
      for(int i = 0; i < nDmrsSyms; ++i)
      {
         uint8_t dmrsSymIdx_within_schAndUciSymbols    = pDmrsSymLoc[i] - startSymInSlot;
         pDmrsFlag[dmrsSymIdx_within_schAndUciSymbols] = true;
      }
   }

   // Determine maxLength
   uint8_t maxLength = 1;
   if(nDmrsSyms > 1){
      if(pDmrsSymLoc[1] == (pDmrsSymLoc[0] + 1)){
         maxLength = 2;
      }
   }

   // Determine starting HARQ symbol
   uint8_t startSymHarq = pDmrsSymLoc[0] - startSymInSlot;
   if(nDmrsCdmGrpsNoData == 1){
      startSymHarq = pDmrsSymLoc[0] - startSymInSlot + maxLength;
   }

   // Determine if HARQ puncturing used
   if((0 < nBitsHarq) && (nBitsHarq < 3)){
      harqAndCsiRePrms.harqPunctFlag = true;
   }else if(nBitsHarq == 0){
      harqAndCsiRePrms.harqPunctFlag = true;
   }else{
      harqAndCsiRePrms.harqPunctFlag = false;
      G_harq_rvd                     = G_harq;
   }

   // Determine starting CSI data symbol (always 0 now)
   uint8_t startSymCsi1 = 0;

   // Initialize number of unassigned Res and bits
   uint16_t nResPerSym  = nPrb * N_SC_PER_PRB;
   uint32_t nBitsPerSym = nPrb * N_SC_PER_PRB * nBitsPerRe;
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      if(pDmrsFlag[symIdx]){
         pNumUnassignedResInSymbol[symIdx]  = 0;
         pNumUnassignedBitsInSymbol[symIdx] = 0;
      }else{
         pNumUnassignedResInSymbol[symIdx]  = nResPerSym;
         pNumUnassignedBitsInSymbol[symIdx] = nBitsPerSym;
      }
   }

   // Initialize grid values
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      pRvdHarqReGrids[symIdx] = std::move(reGrid_t());
      pHarqReGrids[symIdx]    = std::move(reGrid_t());
      pCsi1ReGrids[symIdx]    = std::move(reGrid_t());
   }

   // Determine RE grids reserved for HARQ
   uint32_t nAssignedHarqRvdRmBits = 0;
   for(int symIdx = startSymHarq; symIdx < nSym; ++symIdx)
   {
      if(nAssignedHarqRvdRmBits >= G_harq_rvd)
         break;
      if(pNumUnassignedResInSymbol[symIdx]  == 0)
         continue;

      assignReGrid(pRvdHarqReGrids[symIdx], 
                   pNumUnassignedResInSymbol[symIdx], 
                   pNumUnassignedBitsInSymbol[symIdx],
                   nAssignedHarqRvdRmBits,
                   G_harq_rvd,
                   nBitsPerRe);

      updateNumUnassigned(pRvdHarqReGrids[symIdx],
                          pNumUnassignedResInSymbol[symIdx],
                          pNumUnassignedBitsInSymbol[symIdx],
                          nBitsPerRe);
   }

   // HARQ rateMatched bits assigned to sub-grid of reserved HARQ grid
   uint32_t nAssignedHarqRmBits = 0;
   for(int symIdx = startSymHarq; symIdx < nSym; ++symIdx)
   {
      if(nAssignedHarqRmBits >= G_harq)
         break;

      uint32_t nRvdRes  = pRvdHarqReGrids[symIdx].nRes;
      uint32_t nRvdBits = nRvdRes * nBitsPerRe; 
      if(nRvdRes == 0)
         continue;

      assignReGrid(pHarqReGrids[symIdx], 
                   nRvdRes, 
                   nRvdBits,
                   nAssignedHarqRmBits,
                   G_harq,
                   nBitsPerRe);
   }

   // Determine CSI-P1 RE grid
   uint32_t nAssignedCsi1RmBits = 0;
   for(int symIdx = startSymCsi1; symIdx < nSym; ++symIdx)
   {
      if(nAssignedCsi1RmBits >= G_csi1)
         break;
      if(pNumUnassignedResInSymbol[symIdx]  == 0)
         continue;

      assignReGrid(pCsi1ReGrids[symIdx], 
                  pNumUnassignedResInSymbol[symIdx], 
                  pNumUnassignedBitsInSymbol[symIdx],
                  nAssignedCsi1RmBits,
                  G_csi1,
                  nBitsPerRe);
   }

   // Update number of bits avaliable for assignment to SCH and CSI-P2
   for(int symIdx = 0; symIdx < nSym; ++symIdx)
   {
      if(harqAndCsiRePrms.harqPunctFlag) // If HARQ puncturing enabled, REs assigned to HARQ are also "assigned" to SCH or CSI-P2.
      {
         pNumUnassignedResInSymbol[symIdx]  = nResPerSym;
         pNumUnassignedBitsInSymbol[symIdx] = nBitsPerSym;
      }
      
      updateNumUnassigned(pCsi1ReGrids[symIdx],
                           pNumUnassignedResInSymbol[symIdx],
                           pNumUnassignedBitsInSymbol[symIdx],
                           nBitsPerRe);
   }

   // Determine descrambling offsets
   uint32_t totalNumSchAndUciBits = 0;
   for(int symIdx = 0; symIdx < nSym; ++symIdx) 
   {
      pDescramOffsets[symIdx] = totalNumSchAndUciBits;
      if(pDmrsFlag[symIdx])
      {
         totalNumSchAndUciBits += nBitsPerSym >> 1;
      }else
      {
         totalNumSchAndUciBits += nBitsPerSym;
      }
   }

   // Wrap:
   harqAndCsiRePrms.nBitsPerRe   =  nBitsPerRe;
   harqAndCsiRePrms.nBitsPerSym  =  nBitsPerSym;
   harqAndCsiRePrms.nResPerSym   =  nResPerSym;
}


void  uciOnPuschSegLLRs2::setup( uint16_t                             nCsi2Ues,
                                 uint16_t*                            pCsi2UeIdxs,
                                 PerTbParams*                         pTbPrmsCpu,
                                 PerTbParams*                         pTbPrmsGpu,
                                 uint16_t                             nUeGrps,
                                 cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
                                 cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu,
                                 cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsGpu,               
                                 uciOnPuschSegLLRs2DynDescr_t*        pCpuDynDesc,
                                 void*                                pGpuDynDesc,
                                 uint8_t                              enableCpuToGpuDescrAsyncCpy,
                                 cuphyUciOnPuschSegLLRs2LaunchCfg_t*  pLaunchCfg,
                                 cudaStream_t                         strm)
{
   // pipeline parameters
   pCpuDynDesc->pUePrmsGpu    = pTbPrmsGpu;
   pCpuDynDesc->pUeGrpPrmsGpu = pUeGrpPrmsGpu;

   // input buffer addrs
   for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
   {
      auto                     tensorDesc       = static_cast<tensor_desc&> (*pTensorPrmsEqOutLLRs[ueGrpIdx].desc);
      const tensor_layout_any& tEqOutLLRsLayout = tensorDesc.layout();
      void*                    tEqOutLLRsAddr   = pTensorPrmsEqOutLLRs[ueGrpIdx].pAddr;

      pCpuDynDesc->tEqOutLLRs[ueGrpIdx] = std::move(tensor_ref_any<CUPHY_R_16F>(tEqOutLLRsAddr, tEqOutLLRsLayout.dimensions.begin(), tEqOutLLRsLayout.strides.begin()));
   }

   for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
   {
      uint16_t ueIdx    = pCsi2UeIdxs[csi2Idx];
      int16_t  ueGrpIdx = pTbPrmsCpu[ueIdx].userGroupIndex;

      // set indicies
      pCpuDynDesc->csi2ToUserMapArray[csi2Idx].ueIdx    = ueIdx;
      pCpuDynDesc->csi2ToUserMapArray[csi2Idx].ueGrpIdx = ueGrpIdx;

      // compute CSI-P1 and HARQ resource element grids for the user
     computeCsi1AndHarqReGrids(pCpuDynDesc->harqAndCsi1RePrmsArray[csi2Idx], pTbPrmsCpu[ueIdx], pUeGrpPrmsCpu[ueGrpIdx]);
   }

   // save pointer to GPU descriptor
   uciOnPuschSegLLRs2KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<uciOnPuschSegLLRs2DynDescr_t*>(pGpuDynDesc);

   // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
      cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(uciOnPuschSegLLRs2DynDescr_t), cudaMemcpyHostToDevice, strm);
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nCsi2Ues, pCpuDynDesc->csi2ToUserMapArray, pUeGrpPrmsCpu, pLaunchCfg);

   pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);

#ifdef DEBUG_PRINT
    reGrid_t* pHarqRvdReGrid = pCpuDynDesc->harqAndCsi1RePrmsArray[0].rvdHarqReGrids;
    for(int symIdx = 0; symIdx < 13; ++symIdx)
    {
       NVLOGI_FMT(NVLOG_PUSCH, "{}: symIdx {} reGrid_harq_rvd.nRes {} reGrid_harq_rvd.ReStride {} reGrid_harq_rvd.rmBufferOffset {}", __FUNCTION__, symIdx, pHarqRvdReGrid[symIdx].nRes, pHarqRvdReGrid[symIdx].ReStride, pHarqRvdReGrid[symIdx].rmBufferOffset);
    }
   
    reGrid_t* pHarqReGrid = pCpuDynDesc->harqAndCsi1RePrmsArray[0].harqReGrids;
    for(int symIdx = 0; symIdx < 13; ++symIdx)
    {
       NVLOGI_FMT(NVLOG_PUSCH, "{}: symIdx {} reGrid_harq.nRes {} reGrid_harq.ReStride {} reGrid_harq.rmBufferOffset {}", __FUNCTION__, symIdx, pHarqReGrid[symIdx].nRes, pHarqReGrid[symIdx].ReStride, pHarqReGrid[symIdx].rmBufferOffset);
    }

    reGrid_t* pCsi1ReGrid = pCpuDynDesc->harqAndCsi1RePrmsArray[0].csi1ReGrids;
    for(int symIdx = 0; symIdx < 13; ++symIdx)
    {
       NVLOGI_FMT(NVLOG_PUSCH, "{}: symIdx {} reGrid_csi1.nRes {} reGrid_csi1.ReStride {} reGrid_csi1.rmBufferOffset {}", __FUNCTION__, symIdx, pCsi1ReGrid[symIdx].nRes, pCsi1ReGrid[symIdx].ReStride, pCsi1ReGrid[symIdx].rmBufferOffset);
    }
#endif
}


void uciOnPuschSegLLRs2::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(uciOnPuschSegLLRs2DynDescr_t);
   dynDescrAlignBytes = alignof(uciOnPuschSegLLRs2DynDescr_t);
}