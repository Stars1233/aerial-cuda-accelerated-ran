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

multiCellLayerSelCpu::multiCellLayerSelCpu(cumacCellGrpPrms* cellGrpPrms)
{
    Asim = 0;

    pDynDescr = std::make_unique<mcLayerSelDynDescrCpu_t>();

    DL = cellGrpPrms->dlSchInd;

    enableHarq = cellGrpPrms->harqEnabledInd;
}

multiCellLayerSelCpu::multiCellLayerSelCpu(cumacCellGrpPrms* cellGrpPrms, uint8_t in_Asim)
{
    Asim = in_Asim;

    pDynDescr = std::make_unique<mcLayerSelDynDescrCpu_t>();

    DL = cellGrpPrms->dlSchInd;

    enableHarq = cellGrpPrms->harqEnabledInd;
}

multiCellLayerSelCpu::~multiCellLayerSelCpu() {}

void multiCellLayerSelCpu::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                 cumacSchdSol*               schdSol,
                                 cumacCellGrpPrms*           cellGrpPrms)
{
    pDynDescr->nUe                    = cellGrpPrms->nUe;
    pDynDescr->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pDynDescr->nCell                  = cellGrpPrms->nCell;
    pDynDescr->nUeAnt                 = cellGrpPrms->nUeAnt;
    pDynDescr->sinValThr              = cellGrpPrms->sinValThr;
    allocType                         = cellGrpPrms->allocType;
    precodingScheme                   = cellGrpPrms->precodingScheme;
    
    if (Asim == 1) { // for Aerial Sim
        pDynDescr->sinVal             = cellGrpPrms->sinVal_asim;
        pDynDescr->srsEstChan       = cellGrpPrms->srsEstChan;
        pDynDescr->corrThr            = cellGrpPrms->corrThr;
        pDynDescr->cellAssoc          = cellGrpPrms->cellAssoc;
    } else {
        pDynDescr->sinVal             = cellGrpPrms->sinVal;
        pDynDescr->srsEstChan       = nullptr;
        if (allocType == 0) { // for type-0 allocation
           pDynDescr->cellAssoc       = cellGrpPrms->cellAssoc;
        } else {
           pDynDescr->cellAssoc       = nullptr;
        }
    }

    pDynDescr->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pDynDescr->allocSol               = schdSol->allocSol;
    pDynDescr->layerSelSol            = schdSol->layerSelSol;

    if (DL == 1) { // DL
      pDynDescr->nTxAnt = cellGrpPrms->nBsAnt;
      pDynDescr->nRxAnt = cellGrpPrms->nUeAnt;
    } else { // UL  
      pDynDescr->nTxAnt = cellGrpPrms->nUeAnt;
      pDynDescr->nRxAnt = cellGrpPrms->nBsAnt;
    }

    if (enableHarq == 1) { // HARQ enabled
      pDynDescr->newDataActUe       = cellGrpUeStatus->newDataActUe;
      pDynDescr->layerSelSolLastTx  = cellGrpUeStatus->layerSelSolLastTx;
    } else {
      pDynDescr->newDataActUe       = nullptr;
      pDynDescr->layerSelSolLastTx  = nullptr;
    }
}

void multiCellLayerSelCpu::run()
{
    if (Asim == 1) { // Aerial Sim
        if (allocType == 0) {
            printf("Error: For Aerial Sim, CPU layer selection is only supported for type-1 allocation\n");
            return;
        }
        
        if (enableHarq == 1) { // HARQ enabled
            if (precodingScheme == 0) { // no precoding
               mcLayerSelKernel_type1_cfr_harq();
            } else { // SVD precoding
               mcLayerSelKernel_type1_harq();
            }
        } else {
            if (precodingScheme == 0) { // no precoding
               mcLayerSelKernel_type1_cfr();
            } else { // SVD precoding
               mcLayerSelKernel_type1();
            }
        }
     } else {
        if (enableHarq == 1) { // HARQ enabled
            printf("Error: CPU layer selection is not supported for HARQ\n");
            return;
        }

        if (allocType == 0) { // type-0 allocation
            mcLayerSelKernel_type0();
        } else { // type-1 allocation
            mcLayerSelKernel_type1();
        }
    }
}

void multiCellLayerSelCpu::mcLayerSelKernel_type0()
{
   for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
      uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      if (globalUidx == 0xFFFF) {
         continue;
      }

      uint16_t assocCellIdx;

      for (uint16_t cIdx = 0; cIdx < pDynDescr->nCell; cIdx++) {
         if (pDynDescr->cellAssoc[cIdx*pDynDescr->nUe + uIdx] == 1) {
            assocCellIdx = cIdx;
            break;
         }
      }

      uint8_t numLayers = 0xFF;

      for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
         if (pDynDescr->allocSol[prgIdx*pDynDescr->nCell + assocCellIdx] == uIdx) {
            int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
            float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;

            uint8_t tempNumLayers = 1;

            if (maxSinValThr > 0) {
               for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
                  if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                     tempNumLayers = lIdx + 1;
                     break;
                  }
               }
            }

            if (tempNumLayers < numLayers) {
               numLayers = tempNumLayers;
            }
         }
      }

      pDynDescr->layerSelSol[uIdx] = numLayers;
   }
}

void multiCellLayerSelCpu::mcLayerSelKernel_type1()
{
   for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
      uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      if (globalUidx == 0xFFFF) {
         pDynDescr->layerSelSol[uIdx] = 0xFF;
         continue;
      }

      uint16_t numLayers = 0;

      uint16_t numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];

      for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
         if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
            int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
            float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;

            uint8_t tempNumLayers = 1;

            if (maxSinValThr > 0) {
               for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
                  if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                     tempNumLayers = lIdx + 1;
                     break;
                  }
               }
            }

            numLayers += tempNumLayers;
         }
      }

      if (numAllocPrg > 0) {
         pDynDescr->layerSelSol[uIdx] = floor(static_cast<float>(numLayers)/numAllocPrg);
      } else {
         pDynDescr->layerSelSol[uIdx] = 0xFF;
      }
   }
}

void multiCellLayerSelCpu::mcLayerSelKernel_type1_cfr()
{
   uint16_t nBsUeAntPrd          = pDynDescr->nTxAnt*pDynDescr->nRxAnt;
   uint16_t nPrgBsUeAntPrd       = pDynDescr->nPrbGrp*nBsUeAntPrd;

   for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
      uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      if (globalUidx == 0xFFFF) {
         continue;
      }

      uint8_t cIdx = 0;
      for (uint8_t cellIdx = 0; cellIdx < pDynDescr->nCell; cellIdx++) {
         if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
            cIdx = cellIdx;
            break;
         }
      }

      uint8_t numLayers = 0xFF;

      for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
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

            uint8_t tempNumLayers;

            if (numerator < pDynDescr->corrThr) {
               tempNumLayers = 2;
            } else {
               tempNumLayers = 1;
            }

            if (tempNumLayers < numLayers) {
               numLayers = tempNumLayers;
            }
         }
      }

      pDynDescr->layerSelSol[uIdx] = numLayers;
   }
}

void multiCellLayerSelCpu::mcLayerSelKernel_type1_harq()
{
   for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
      uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      if (globalUidx == 0xFFFF) {
         continue;
      }

      if (pDynDescr->newDataActUe[globalUidx] != 1) { // NOT scheduled for new transmission
         pDynDescr->layerSelSol[uIdx] = pDynDescr->layerSelSolLastTx[uIdx];
      } else {
         uint16_t numLayers = 0;

         uint16_t numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];

         for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
            if (prgIdx >= pDynDescr->allocSol[2*uIdx] && prgIdx < pDynDescr->allocSol[2*uIdx+1]) {
               int indexTemp = uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt;
               float maxSinValThr = pDynDescr->sinVal[indexTemp]*pDynDescr->sinValThr;

               uint8_t tempNumLayers = 1;

               if (maxSinValThr > 0) {
                  for (int lIdx = pDynDescr->nUeAnt - 1; lIdx >= 1; lIdx--) {
                     if (pDynDescr->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                        tempNumLayers = lIdx + 1;
                        break;
                     }
                  }
               }

               numLayers += tempNumLayers;
            }
         }
         if (numAllocPrg > 0) {
            pDynDescr->layerSelSol[uIdx] = floor(static_cast<float>(numLayers)/numAllocPrg);
         } else {
            pDynDescr->layerSelSol[uIdx] = 0xFF;
         }
      }
   }
}

void multiCellLayerSelCpu::mcLayerSelKernel_type1_cfr_harq()
{
   uint16_t nBsUeAntPrd          = pDynDescr->nTxAnt*pDynDescr->nRxAnt;
   uint16_t nPrgBsUeAntPrd       = pDynDescr->nPrbGrp*nBsUeAntPrd;
   
   for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
      uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

      if (globalUidx == 0xFFFF) {
         continue;
      }

      if (pDynDescr->newDataActUe[globalUidx] != 1) { // NOT scheduled for new transmission
         pDynDescr->layerSelSol[uIdx] = pDynDescr->layerSelSolLastTx[uIdx];
      } else {
         uint8_t cIdx = 0;
         for (uint8_t cellIdx = 0; cellIdx < pDynDescr->nCell; cellIdx++) {
            if (pDynDescr->cellAssoc[cellIdx*pDynDescr->nUe + uIdx] == 1) {
               cIdx = cellIdx;
               break;
            }
         }

         uint8_t numLayers = 0xFF;

         for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
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

               uint8_t tempNumLayers;

               if (numerator < pDynDescr->corrThr) {
                  tempNumLayers = 2;
               } else {
                  tempNumLayers = 1;
               }

               if (tempNumLayers < numLayers) {
                  numLayers = tempNumLayers;
               }
            }
         }

         pDynDescr->layerSelSol[uIdx] = numLayers;
      }
   }
}
}