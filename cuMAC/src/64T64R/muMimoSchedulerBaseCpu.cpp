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

muMimoSchedulerBaseCpu::muMimoSchedulerBaseCpu()
{
    pCpuDynDesc = std::make_unique<muSchBaseDynDescrCpu_t>();
    matAlg      = std::make_unique<cpuMatAlg>();
}

void muMimoSchedulerBaseCpu::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                   cumacSchdSol*               schdSol,
                                   cumacCellGrpPrms*           cellGrpPrms,
                                   uint8_t                     in_enableHarq,
                                   uint8_t                     in_dl)
{
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->numUeForGrpPerCell     = cellGrpPrms->numUeForGrpPerCell;
    pCpuDynDesc->numUeSchdPerCellTTI    = cellGrpPrms->numUeSchdPerCellTTI; // maximum number of UEs scheduled per cell per TTI
    pCpuDynDesc->W                      = cellGrpPrms->W;
    pCpuDynDesc->zfCoeff                = cellGrpPrms->zfCoeff;
    pCpuDynDesc->betaCoeff              = cellGrpPrms->betaCoeff;
    pCpuDynDesc->sinValThr              = cellGrpPrms->sinValThr;
    pCpuDynDesc->nMaxUePerGrpUl         = cellGrpPrms->nMaxUePerGrpUl;
    pCpuDynDesc->nMaxUePerGrpDl         = cellGrpPrms->nMaxUePerGrpDl;
    pCpuDynDesc->nMaxLayerPerGrpUl      = cellGrpPrms->nMaxLayerPerGrpUl;
    pCpuDynDesc->nMaxLayerPerGrpDl      = cellGrpPrms->nMaxLayerPerGrpDl;
    pCpuDynDesc->nMaxLayerPerUeSuDl     = cellGrpPrms->nMaxLayerPerUeSuDl;
    pCpuDynDesc->srsEstChan             = cellGrpPrms->srsEstChan;
    pCpuDynDesc->srsUeMap               = cellGrpPrms->srsUeMap;
    pCpuDynDesc->sinVal                 = cellGrpPrms->sinVal;
    pCpuDynDesc->ueTxPow                = cellGrpUeStatus->ueTxPow;
    pCpuDynDesc->bsTxPow                = cellGrpPrms->bsTxPow;
    pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
    pCpuDynDesc->avgRatesActUe          = cellGrpUeStatus->avgRatesActUe;
    pCpuDynDesc->noiseVarActUe          = cellGrpUeStatus->noiseVarActUe;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->layerSelSol            = schdSol->layerSelSol;
    pCpuDynDesc->nSCID                  = schdSol->nSCID;
    dl                                  = in_dl;
    enableHarq                          = in_enableHarq;
    allocType                           = cellGrpPrms->allocType;

    // sanity check
    if (dl == 0) {
        throw std::runtime_error("Error: baseline MU-MIMO scheduler does not support UL yet");
    }

    if (allocType == 0) {
        throw std::runtime_error("Error: baseline MU-MIMO scheduler does not support type-0 PRB allocation yet");
    }

    if (pCpuDynDesc->numUeForGrpPerCell < pCpuDynDesc->numUeSchdPerCellTTI) {
        throw std::runtime_error("Error: for baseline MU-MIMO scheduler, numUeSchdPerCellTTI should be no greater than numUeForGrpPerCell");
    }

    if (enableHarq == 1) { // HARQ is enabled
        pCpuDynDesc->newDataActUe       = cellGrpUeStatus->newDataActUe; 
        pCpuDynDesc->allocSolLastTx     = cellGrpUeStatus->allocSolLastTx; 
        pCpuDynDesc->layerSelSolLastTx  = cellGrpUeStatus->layerSelSolLastTx; 
    } else {
        pCpuDynDesc->newDataActUe       = nullptr;
        pCpuDynDesc->allocSolLastTx     = nullptr;
        pCpuDynDesc->layerSelSolLastTx  = nullptr;
    }
}

void muMimoSchedulerBaseCpu::run()
{
    if (enableHarq == 1) { // HARQ is enabled
        muSchBaseCpu_type1_dl_harq();
    } else { // HARQ is disabled
        muSchBaseCpu_type1_dl();
    }
}

void muMimoSchedulerBaseCpu::muSchBaseCpu_type1_dl()
{
    std::srand(unsigned(std::time(0)));

    uint32_t nBsUeAntPrd = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nPrgBsUeAntPrd = pCpuDynDesc->nPrbGrp*nBsUeAntPrd;

    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) { // loop through all cells
        // initialize UE selection solution
        for (int uIdx = 0; uIdx < pCpuDynDesc->numUeForGrpPerCell; uIdx++) {
            pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + uIdx] = 0xFFFF;
        }

        // determine associated UEs
        std::vector<uint16_t> assocUeId;
        std::vector<uint16_t> anchorUe;
        std::vector<uint16_t> subbands;
        
        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cIdx*pCpuDynDesc->nActiveUe + uIdx] == 1) {
                assocUeId.push_back(uIdx);
            }
        }

        // layer selection
        for (int idx = 0; idx < assocUeId.size(); idx++) {
            uint16_t globalUidx = assocUeId[idx];
            uint16_t numLayers = 0;

            for (int prgIdx = 0; prgIdx < pCpuDynDesc->nPrbGrp; prgIdx++) {
               int indexTemp = globalUidx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + prgIdx*pCpuDynDesc->nUeAnt;
               float maxSinValThr = pCpuDynDesc->sinVal[indexTemp]*pCpuDynDesc->sinValThr;

               uint8_t tempNumLayers = 1;

               if (maxSinValThr > 0) {
                  for (int lIdx = pCpuDynDesc->nUeAnt - 1; lIdx >= 1; lIdx--) {
                     if (pCpuDynDesc->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                        tempNumLayers = lIdx + 1;
                        break;
                     }
                  }
               }

               numLayers += tempNumLayers;
            }

            pCpuDynDesc->layerSelSol[globalUidx] = floor(static_cast<float>(numLayers)/pCpuDynDesc->nPrbGrp);
            pCpuDynDesc->layerSelSol[globalUidx] = pCpuDynDesc->layerSelSol[globalUidx] > pCpuDynDesc->nMaxLayerPerUeSuDl ? pCpuDynDesc->nMaxLayerPerUeSuDl : pCpuDynDesc->layerSelSol[globalUidx];
        }

        // calculating PF metrics
        std::vector<std::vector<float>> pfMetrics;

        for (int idx = 0; idx < assocUeId.size(); idx++) {
            int uIdx = assocUeId[idx];

            std::vector<float> pfMetricUe;

            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                // calculate PF for UE uIdx on PRG rbgIdx
                // calculate RZF predocder (row-major matrix access)
                std::vector<cuComplex> rzfPrd_rm(pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt);
                std::vector<cuComplex> gramMat_rm(pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt);
                std::vector<cuComplex> invGramMat_rm(pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt);
                
                for (int i = 0; i < pCpuDynDesc->nUeAnt; i++) {
                    for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                        if (i == j) {
                            gramMat_rm[i*pCpuDynDesc->nUeAnt + j].x = pCpuDynDesc->zfCoeff;
                            gramMat_rm[i*pCpuDynDesc->nUeAnt + j].y = 0;
                        } else {
                            gramMat_rm[i*pCpuDynDesc->nUeAnt + j].x = 0;
                            gramMat_rm[i*pCpuDynDesc->nUeAnt + j].y = 0;
                        }
                    }
                }

                // H*H^H + alpha*I
                matAlg->matMultiplication_aaHplusb_rm(&pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][uIdx]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, gramMat_rm.data());
                // (H*H^H + alpha*I)^-1
                matAlg->matInverse_rm(gramMat_rm.data(), pCpuDynDesc->nUeAnt, invGramMat_rm.data());
                // un-normalized RZF precoder = H^H*(H*H^H + alpha*I)^-1
                matAlg->matMultiplication_aHb_rm(&pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][uIdx]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, invGramMat_rm.data(), pCpuDynDesc->nUeAnt, rzfPrd_rm.data());

                // calculate normalizatino factor
                float trace = 0;
                for (int i = 0; i < pCpuDynDesc->nBsAnt; i++) {
                    for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                        float x = rzfPrd_rm[i*pCpuDynDesc->nUeAnt + j].x;
                        float y = rzfPrd_rm[i*pCpuDynDesc->nUeAnt + j].y;
                        trace += x*x + y*y;
                    }
                }

                float lambda = pCpuDynDesc->bsTxPow[cIdx]/trace;
                float snr = lambda/pCpuDynDesc->noiseVarActUe[uIdx];
                float insRate = pCpuDynDesc->nUeAnt*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));

                pfMetricUe.push_back(pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[uIdx]);
            }

            pfMetrics.push_back(pfMetricUe);
        }

        // get a copy
        std::vector<std::vector<float>> pfCopy(pfMetrics);

        // RME SU-MIMO PRG allocation
        int schdUeCounter = 0;
        std::vector<int> prgAllocUeId(pCpuDynDesc->nPrbGrp, -1);

        while(1) {
            if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                break;
            }

            std::vector<float> maxPfArr;
            std::vector<int> maxPfPrgIdxArr;

            for (int uIdx = 0; uIdx < assocUeId.size(); uIdx++) {
                auto maxIt = std::max_element(pfMetrics[uIdx].begin(), pfMetrics[uIdx].end());
                int index = std::distance(pfMetrics[uIdx].begin(), maxIt);
                maxPfArr.push_back(*maxIt);
                maxPfPrgIdxArr.push_back(index);
            }

            auto maxIt = std::max_element(maxPfArr.begin(), maxPfArr.end());
            if (*maxIt == -1.0) {
                break;
            }

            int maxUeId = std::distance(maxPfArr.begin(), maxIt);
            int maxPrgId = maxPfPrgIdxArr[maxUeId];

            pCpuDynDesc->allocSol[2*assocUeId[maxUeId]] = static_cast<int16_t>(maxPrgId);
            pCpuDynDesc->allocSol[2*assocUeId[maxUeId] + 1] = static_cast<int16_t>(maxPrgId) + 1;

            pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + schdUeCounter] = assocUeId[maxUeId];
            schdUeCounter++;
            anchorUe.push_back(assocUeId[maxUeId]);

            // left
            for (int prgIdx = maxPrgId-1; prgIdx >= 0; prgIdx--) {
                if (prgAllocUeId[prgIdx] != -1) {
                    break;
                }

                bool alloc = true;

                for (int uIdx = 0; uIdx < assocUeId.size(); uIdx++) {
                    if (uIdx == maxUeId) {
                        continue;
                    } 

                    if (pfMetrics[uIdx][prgIdx] > pfMetrics[maxUeId][prgIdx]) {
                        alloc = false;
                        break;
                    }
                }

                if (alloc) {
                    pCpuDynDesc->allocSol[2*assocUeId[maxUeId]] = static_cast<int16_t>(prgIdx);
                } else {
                    break;
                }
            }

            // right
            for (int prgIdx = maxPrgId + 1; prgIdx < pCpuDynDesc->nPrbGrp; prgIdx++) {
                if (prgAllocUeId[prgIdx] != -1) {
                    break;
                }

                bool alloc = true;

                for (int uIdx = 0; uIdx < assocUeId.size(); uIdx++) {
                    if (uIdx == maxUeId) {
                        continue;
                    } 

                    if (pfMetrics[uIdx][prgIdx] > pfMetrics[maxUeId][prgIdx]) {
                        alloc = false;
                        break;
                    }
                }

                if (alloc) {
                    pCpuDynDesc->allocSol[2*assocUeId[maxUeId] + 1] = static_cast<int16_t>(prgIdx) + 1;
                } else {
                    break;
                }
            }

            // remove rows and columns
            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                pfMetrics[maxUeId][rbgIdx] = -1.0;
            }

            for (int rbgIdx = pCpuDynDesc->allocSol[2*assocUeId[maxUeId]]; rbgIdx < pCpuDynDesc->allocSol[2*assocUeId[maxUeId] + 1]; rbgIdx++) {
                for (int uIdx = 0; uIdx < assocUeId.size(); uIdx++) {
                    pfMetrics[uIdx][rbgIdx] = -1.0;
                }

                prgAllocUeId[rbgIdx] = maxUeId;
            }
        }

        // fill gaps
        while(1) {
            bool alloc = false; // PRG allocated

            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                if (prgAllocUeId[rbgIdx] != -1) {
                    continue;
                }

                float leftPf = -1.0;
                float rightPf = -1.0;
                if (rbgIdx > 0 && prgAllocUeId[rbgIdx - 1] != -1) {
                    leftPf = 0;
                    for (int rIdx = pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx - 1]]]; rIdx < pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx - 1]] + 1]; rIdx++) {
                        leftPf += pfCopy[prgAllocUeId[rbgIdx - 1]][rIdx];
                    }
                }

                if (rbgIdx < (pCpuDynDesc->nPrbGrp-1) && prgAllocUeId[rbgIdx + 1] != -1) {
                    rightPf = 0;
                    for (int rIdx = pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx + 1]]]; rIdx < pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx + 1]] + 1]; rIdx++) {
                        rightPf += pfCopy[prgAllocUeId[rbgIdx + 1]][rIdx];
                    }
                }

                if (leftPf != -1.0 || rightPf != -1.0) {
                    if (leftPf > rightPf) {
                        pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx - 1]] + 1] = static_cast<int16_t>(rbgIdx) + 1;
                        prgAllocUeId[rbgIdx] = prgAllocUeId[rbgIdx - 1];
                    } else {
                        pCpuDynDesc->allocSol[2*assocUeId[prgAllocUeId[rbgIdx + 1]]] = static_cast<int16_t>(rbgIdx);
                        prgAllocUeId[rbgIdx] = prgAllocUeId[rbgIdx + 1];
                    }

                    alloc = true;
                }
            }

            if (!alloc) {
                break;
            }
        }

        // multi-user scheduling (iterative candidate UE group generation)
        // directly continue to the next cell if the number of scheduled SU-MIMO UEs reaches numUeSchdPerCellTTI
        if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
            continue;
        }

        // define constants
        size_t num_elements = static_cast<size_t>(pCpuDynDesc->nBsAnt);

        // step 1: determine subbands assigned to anchor UEs, and initialize the list of available UEs
        std::vector<uint16_t> avlbUe(assocUeId); // get a copy of all UEs associated to cell cIdx
        // remove anchor UEs
        avlbUe.erase(std::remove_if(
                     avlbUe.begin(), avlbUe.end(),
                     [&anchorUe](uint16_t value) {
                        // Check if the value is in vector a
                        return std::find(anchorUe.begin(), anchorUe.end(), value) != anchorUe.end();
                     }), avlbUe.end());

        // step 2: iterative UR group generation
        std::vector<int> subbandId(anchorUe.size());
        for (int i = 0; i < anchorUe.size(); i++) {
            subbandId[i] = i;
        }

        std::random_shuffle(subbandId.begin(), subbandId.end());

        for (int i = 0; i < subbandId.size(); i++) {
            if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                break;
            }

            int sbIdx = subbandId[i];

            int prgStart = pCpuDynDesc->allocSol[2*anchorUe[sbIdx]];
            int prgEndPlusOne = pCpuDynDesc->allocSol[2*anchorUe[sbIdx] + 1];

            float maxPf = 0;
            for (int rbgIdx = prgStart; rbgIdx < prgEndPlusOne; rbgIdx++) {
                maxPf += pfCopy[prgAllocUeId[rbgIdx]][rbgIdx];
            }

            std::vector<uint16_t> bestUeg;
            std::vector<uint16_t> bestUegLayers;
            uint8_t numLayersAdded = pCpuDynDesc->layerSelSol[anchorUe[sbIdx]];
            bestUeg.push_back(anchorUe[sbIdx]);
            bestUegLayers.push_back(pCpuDynDesc->layerSelSol[anchorUe[sbIdx]]);

            for (int uIdx = 0; uIdx < (pCpuDynDesc->nMaxUePerGrpDl - 1); uIdx++) {
                if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                    break;
                }

                float maxPfItr = -1.0;
                uint16_t bestCandiUe;

                for (uint16_t candiUeIdx = 0; candiUeIdx < avlbUe.size(); candiUeIdx++) {
                    uint8_t numLayersCombined = numLayersAdded + pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]];
                    if (numLayersCombined > pCpuDynDesc->nMaxLayerPerGrpDl) {
                        continue;
                    }

                    float pfTemp = 0;

                    for (int rbgIdx = prgStart; rbgIdx < prgEndPlusOne; rbgIdx++) {
                        std::vector<cuComplex> stackedChannMat;
                    
                        for (int extUeIdx = 0; extUeIdx < bestUeg.size(); extUeIdx++) {
                            for (int lIdx = 0; lIdx < bestUegLayers[extUeIdx]; lIdx++) {
                                cuComplex* channPtr = &pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][bestUeg[extUeIdx]]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd + lIdx*pCpuDynDesc->nBsAnt];
                                stackedChannMat.insert(stackedChannMat.end(), channPtr, channPtr + num_elements);
                            }
                        }

                        for (int lIdx = 0; lIdx < pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]]; lIdx++) {
                            cuComplex* channPtr = &pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][avlbUe[candiUeIdx]]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd + lIdx*pCpuDynDesc->nBsAnt];
                            stackedChannMat.insert(stackedChannMat.end(), channPtr, channPtr + num_elements);
                        }

                        // calculate RZF predocder (row-major matrix access)
                        std::vector<cuComplex> rzfPrd_rm(pCpuDynDesc->nBsAnt*numLayersCombined);
                        std::vector<cuComplex> gramMat_rm(numLayersCombined*numLayersCombined);
                        std::vector<cuComplex> invGramMat_rm(numLayersCombined*numLayersCombined);
                
                        for (int i = 0; i < numLayersCombined; i++) {
                            for (int j = 0; j < numLayersCombined; j++) {
                                if (i == j) {
                                    gramMat_rm[i*numLayersCombined + j].x = pCpuDynDesc->zfCoeff;
                                    gramMat_rm[i*numLayersCombined + j].y = 0;
                                } else {
                                    gramMat_rm[i*numLayersCombined + j].x = 0;
                                    gramMat_rm[i*numLayersCombined + j].y = 0;
                                }
                            }
                        }

                        // H*H^H + alpha*I
                        matAlg->matMultiplication_aaHplusb_rm(stackedChannMat.data(), numLayersCombined, pCpuDynDesc->nBsAnt, gramMat_rm.data());
                        // (H*H^H + alpha*I)^-1
                        matAlg->matInverse_rm(gramMat_rm.data(), numLayersCombined, invGramMat_rm.data());
                        // un-normalized RZF precoder = H^H*(H*H^H + alpha*I)^-1
                        matAlg->matMultiplication_aHb_rm(stackedChannMat.data(), numLayersCombined, pCpuDynDesc->nBsAnt, invGramMat_rm.data(), numLayersCombined, rzfPrd_rm.data());

                        // calculate normalizatino factor
                        float trace = 0;
                        for (int i = 0; i < pCpuDynDesc->nBsAnt; i++) {
                            for (int j = 0; j < numLayersCombined; j++) {
                                float x = rzfPrd_rm[i*numLayersCombined + j].x;
                                float y = rzfPrd_rm[i*numLayersCombined + j].y;
                                trace += x*x + y*y;
                            }
                        }

                        float lambda = pCpuDynDesc->bsTxPow[cIdx]/trace;
                        for (int ueIdx = 0; ueIdx < bestUeg.size(); ueIdx++) {
                            float snr = lambda/pCpuDynDesc->noiseVarActUe[bestUeg[ueIdx]];
                            float insRate = bestUegLayers[ueIdx]*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));
                            pfTemp += pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[bestUeg[ueIdx]];
                        }

                        float snr = lambda/pCpuDynDesc->noiseVarActUe[avlbUe[candiUeIdx]];
                        float insRate = pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]]*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));
                        pfTemp += pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[avlbUe[candiUeIdx]];
                    }

                    if (pfTemp > maxPfItr) {
                        maxPfItr = pfTemp;
                        bestCandiUe = avlbUe[candiUeIdx];
                    }
                }

                if (maxPfItr >= maxPf) {
                    bestUeg.push_back(bestCandiUe);
                    bestUegLayers.push_back(pCpuDynDesc->layerSelSol[bestCandiUe]);
                    pCpuDynDesc->allocSol[2*bestCandiUe] = static_cast<int16_t>(prgStart);
                    pCpuDynDesc->allocSol[2*bestCandiUe + 1] = static_cast<int16_t>(prgEndPlusOne);

                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + schdUeCounter] = bestCandiUe;
                    schdUeCounter++;

                    numLayersAdded += pCpuDynDesc->layerSelSol[bestCandiUe];
                    avlbUe.erase(std::remove(avlbUe.begin(), avlbUe.end(), bestCandiUe), avlbUe.end());
                } else {
                    break;
                }
            }

            uint8_t layerCounter = 0;
            for (int uIdx = 0; uIdx < bestUeg.size(); uIdx++) {
                // nSCID allocation
                for (int lIdx = 0; lIdx < pCpuDynDesc->nUeAnt; lIdx++) {
                    if (lIdx < bestUegLayers[uIdx]) {
                        pCpuDynDesc->nSCID[bestUeg[uIdx]*pCpuDynDesc->nUeAnt + lIdx] = layerCounter >= totNumPdschDmrsPort_;

                        layerCounter++;
                    } else {
                        pCpuDynDesc->nSCID[bestUeg[uIdx]*pCpuDynDesc->nUeAnt + lIdx] = 0xFF;
                    }
                }
            }
        }
    }
}

void muMimoSchedulerBaseCpu::muSchBaseCpu_type1_dl_harq()
{
    std::srand(unsigned(std::time(0)));

    uint32_t nBsUeAntPrd = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nPrgBsUeAntPrd = pCpuDynDesc->nPrbGrp*nBsUeAntPrd;

    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) { // loop through all cells
        // initialize UE selection solution
        for (int uIdx = 0; uIdx < pCpuDynDesc->numUeForGrpPerCell; uIdx++) {
            pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + uIdx] = 0xFFFF;
        }

        // determine associated UEs
        std::vector<uint16_t> assocUeId; // active UE IDs associated to cell cIdx
        std::vector<uint16_t> newTxUeId; // active UE IDs for new transmissions
        std::vector<uint16_t> reTxUeId; // active UE IDs for re-transmissions
        std::vector<uint16_t> anchorUe; // active UE IDs of MU-MIMO anchor UEs
        std::vector<uint16_t> subbands;
        
        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cIdx*pCpuDynDesc->nActiveUe + uIdx] == 1) {
                assocUeId.push_back(uIdx);
            }
        }

        // layer selection
        for (int idx = 0; idx < assocUeId.size(); idx++) {
            uint16_t globalUidx = assocUeId[idx];

            if (pCpuDynDesc->newDataActUe[globalUidx] != 1) { // re-transmission
                reTxUeId.push_back(globalUidx);

                // non-adaptive layer selection for HARQ re-TX
                pCpuDynDesc->layerSelSol[globalUidx] = pCpuDynDesc->layerSelSolLastTx[globalUidx];
            } else { // new transmission
                newTxUeId.push_back(globalUidx);

                // layer selection for new transmission
                uint16_t numLayers = 0;

                for (int prgIdx = 0; prgIdx < pCpuDynDesc->nPrbGrp; prgIdx++) {
                    int indexTemp = globalUidx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + prgIdx*pCpuDynDesc->nUeAnt;
                    float maxSinValThr = pCpuDynDesc->sinVal[indexTemp]*pCpuDynDesc->sinValThr;

                    uint8_t tempNumLayers = 1;

                    if (maxSinValThr > 0) {
                        for (int lIdx = pCpuDynDesc->nUeAnt - 1; lIdx >= 1; lIdx--) {
                            if (pCpuDynDesc->sinVal[indexTemp + lIdx] >= maxSinValThr) {
                                tempNumLayers = lIdx + 1;
                                break;
                            }
                        }
                    }

                    numLayers += tempNumLayers;
                }

                pCpuDynDesc->layerSelSol[globalUidx] = floor(static_cast<float>(numLayers)/pCpuDynDesc->nPrbGrp);
                pCpuDynDesc->layerSelSol[globalUidx] = pCpuDynDesc->layerSelSol[globalUidx] > pCpuDynDesc->nMaxLayerPerUeSuDl ? pCpuDynDesc->nMaxLayerPerUeSuDl : pCpuDynDesc->layerSelSol[globalUidx];
            }
        }

        // PRG allocation 
        int schdUeCounter = 0; // counter for the number of scheduled UEs
        std::vector<int> prgAllocUeId(pCpuDynDesc->nPrbGrp, -1); // for storing 0-based local IDs for new-TX UEs. -1 indicates an unallocated PRG

        // PRG allocation for re-TX UEs
        std::random_shuffle(reTxUeId.begin(), reTxUeId.end());

        int numRemainingPrg = pCpuDynDesc->nPrbGrp;
        int startRbgAlloc = 0;

        for (int idx = 0; idx < reTxUeId.size(); idx++) {
            int ueId = reTxUeId[idx];

            if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                pCpuDynDesc->allocSol[2*ueId]       = -1;
                pCpuDynDesc->allocSol[2*ueId + 1]   = -1;
                continue;
            }

            int numPrgRsv = pCpuDynDesc->allocSolLastTx[2*ueId + 1] - pCpuDynDesc->allocSolLastTx[2*ueId];

            if (numRemainingPrg >= numPrgRsv) {
                pCpuDynDesc->allocSol[2*ueId]   = startRbgAlloc;
                pCpuDynDesc->allocSol[2*ueId+1] = startRbgAlloc + numPrgRsv;
                startRbgAlloc += numPrgRsv;
                numRemainingPrg -= numPrgRsv;

                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + schdUeCounter] = static_cast<uint16_t>(ueId);
                schdUeCounter++;

                for (int prgIdx = pCpuDynDesc->allocSol[2*ueId]; prgIdx < pCpuDynDesc->allocSol[2*ueId + 1]; prgIdx++) {
                    prgAllocUeId[prgIdx] = INT_MAX;
                }
            } else {
                pCpuDynDesc->allocSol[2*ueId]       = -1;
                pCpuDynDesc->allocSol[2*ueId + 1]   = -1;
            }
        }

        // continue to the next cell if the number of scheduled UEs reaches numUeSchdPerCellTTI or there's no available PRG for allocation
        if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI || numRemainingPrg == 0) {
            continue;
        }

        // calculating PF metrics for new-TX UEs
        std::vector<std::vector<float>> pfMetrics;

        for (int idx = 0; idx < newTxUeId.size(); idx++) {
            int uIdx = newTxUeId[idx];

            std::vector<float> pfMetricUe;

            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                if (prgAllocUeId[rbgIdx] != -1) { // PRG allocated to a re-TX UE
                    pfMetricUe.push_back(-1.0);
                } else {
                    // calculate PF for UE uIdx on PRG rbgIdx
                    // calculate RZF predocder (row-major matrix access)
                    std::vector<cuComplex> rzfPrd_rm(pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt);
                    std::vector<cuComplex> gramMat_rm(pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt);
                    std::vector<cuComplex> invGramMat_rm(pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt);
                
                    for (int i = 0; i < pCpuDynDesc->nUeAnt; i++) {
                        for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                            if (i == j) {
                                gramMat_rm[i*pCpuDynDesc->nUeAnt + j].x = pCpuDynDesc->zfCoeff;
                                gramMat_rm[i*pCpuDynDesc->nUeAnt + j].y = 0;
                            } else {
                                gramMat_rm[i*pCpuDynDesc->nUeAnt + j].x = 0;
                                gramMat_rm[i*pCpuDynDesc->nUeAnt + j].y = 0;
                            }
                        }
                    }

                    // H*H^H + alpha*I
                    matAlg->matMultiplication_aaHplusb_rm(&pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][uIdx]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, gramMat_rm.data());
                    // (H*H^H + alpha*I)^-1
                    matAlg->matInverse_rm(gramMat_rm.data(), pCpuDynDesc->nUeAnt, invGramMat_rm.data());
                    // un-normalized RZF precoder = H^H*(H*H^H + alpha*I)^-1
                    matAlg->matMultiplication_aHb_rm(&pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][uIdx]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, invGramMat_rm.data(), pCpuDynDesc->nUeAnt, rzfPrd_rm.data());

                    // calculate normalizatino factor
                    float trace = 0;
                    for (int i = 0; i < pCpuDynDesc->nBsAnt; i++) {
                        for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                            float x = rzfPrd_rm[i*pCpuDynDesc->nUeAnt + j].x;
                            float y = rzfPrd_rm[i*pCpuDynDesc->nUeAnt + j].y;
                            trace += x*x + y*y;
                        }
                    }

                    float lambda = pCpuDynDesc->bsTxPow[cIdx]/trace;
                    float snr = lambda/pCpuDynDesc->noiseVarActUe[uIdx];
                    float insRate = pCpuDynDesc->nUeAnt*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));

                    pfMetricUe.push_back(pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[uIdx]);
                }
            }

            pfMetrics.push_back(pfMetricUe);
        }

        // get a copy
        std::vector<std::vector<float>> pfCopy(pfMetrics);

        // RME SU-MIMO PRG allocation for new-TX UEs
        while(1) {
            if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                break;
            }

            std::vector<float> maxPfArr;
            std::vector<int> maxPfPrgIdxArr;

            for (int uIdx = 0; uIdx < newTxUeId.size(); uIdx++) {
                auto maxIt = std::max_element(pfMetrics[uIdx].begin(), pfMetrics[uIdx].end());
                int index = std::distance(pfMetrics[uIdx].begin(), maxIt);
                maxPfArr.push_back(*maxIt);
                maxPfPrgIdxArr.push_back(index);
            }

            auto maxIt = std::max_element(maxPfArr.begin(), maxPfArr.end());
            if (*maxIt == -1.0) {
                break;
            }

            int maxUeId = std::distance(maxPfArr.begin(), maxIt); // 0-based local ID for new-TX UEs
            int maxPrgId = maxPfPrgIdxArr[maxUeId];

            pCpuDynDesc->allocSol[2*newTxUeId[maxUeId]] = static_cast<int16_t>(maxPrgId);
            pCpuDynDesc->allocSol[2*newTxUeId[maxUeId] + 1] = static_cast<int16_t>(maxPrgId) + 1;

            pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + schdUeCounter] = newTxUeId[maxUeId];
            schdUeCounter++;
            anchorUe.push_back(newTxUeId[maxUeId]);

            // left
            for (int prgIdx = maxPrgId-1; prgIdx >= 0; prgIdx--) {
                if (prgAllocUeId[prgIdx] != -1) {
                    break;
                }

                bool alloc = true;

                for (int uIdx = 0; uIdx < newTxUeId.size(); uIdx++) {
                    if (uIdx == maxUeId) {
                        continue;
                    } 

                    if (pfMetrics[uIdx][prgIdx] > pfMetrics[maxUeId][prgIdx]) {
                        alloc = false;
                        break;
                    }
                }

                if (alloc) {
                    pCpuDynDesc->allocSol[2*newTxUeId[maxUeId]] = static_cast<int16_t>(prgIdx);
                } else {
                    break;
                }
            }

            // right
            for (int prgIdx = maxPrgId + 1; prgIdx < pCpuDynDesc->nPrbGrp; prgIdx++) {
                if (prgAllocUeId[prgIdx] != -1) {
                    break;
                }

                bool alloc = true;

                for (int uIdx = 0; uIdx < newTxUeId.size(); uIdx++) {
                    if (uIdx == maxUeId) {
                        continue;
                    } 

                    if (pfMetrics[uIdx][prgIdx] > pfMetrics[maxUeId][prgIdx]) {
                        alloc = false;
                        break;
                    }
                }

                if (alloc) {
                    pCpuDynDesc->allocSol[2*newTxUeId[maxUeId] + 1] = static_cast<int16_t>(prgIdx) + 1;
                } else {
                    break;
                }
            }

            // remove rows and columns
            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                pfMetrics[maxUeId][rbgIdx] = -1.0;
            }

            for (int rbgIdx = pCpuDynDesc->allocSol[2*newTxUeId[maxUeId]]; rbgIdx < pCpuDynDesc->allocSol[2*newTxUeId[maxUeId] + 1]; rbgIdx++) {
                for (int uIdx = 0; uIdx < newTxUeId.size(); uIdx++) {
                    pfMetrics[uIdx][rbgIdx] = -1.0;
                }

                prgAllocUeId[rbgIdx] = maxUeId;
            }
        }

        // fill gaps
        while(1) {
            bool alloc = false; // PRG allocated

            for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
                if (prgAllocUeId[rbgIdx] != -1) {
                    continue;
                }

                float leftPf = -1.0;
                float rightPf = -1.0;
                if (rbgIdx > 0 && prgAllocUeId[rbgIdx - 1] != -1 && prgAllocUeId[rbgIdx - 1] != INT_MAX) {
                    leftPf = 0;
                    for (int rIdx = pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx - 1]]]; rIdx < pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx - 1]] + 1]; rIdx++) {
                        leftPf += pfCopy[prgAllocUeId[rbgIdx - 1]][rIdx];
                    }
                }

                if (rbgIdx < (pCpuDynDesc->nPrbGrp-1) && prgAllocUeId[rbgIdx + 1] != -1 && prgAllocUeId[rbgIdx + 1] != INT_MAX) {
                    rightPf = 0;
                    for (int rIdx = pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx + 1]]]; rIdx < pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx + 1]] + 1]; rIdx++) {
                        rightPf += pfCopy[prgAllocUeId[rbgIdx + 1]][rIdx];
                    }
                }

                if (leftPf != -1.0 || rightPf != -1.0) {
                    if (leftPf > rightPf) {
                        pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx - 1]] + 1] = static_cast<int16_t>(rbgIdx) + 1;
                        prgAllocUeId[rbgIdx] = prgAllocUeId[rbgIdx - 1];
                    } else {
                        pCpuDynDesc->allocSol[2*newTxUeId[prgAllocUeId[rbgIdx + 1]]] = static_cast<int16_t>(rbgIdx);
                        prgAllocUeId[rbgIdx] = prgAllocUeId[rbgIdx + 1];
                    }

                    alloc = true;
                }
            }

            if (!alloc) {
                break;
            }
        }

        // multi-user scheduling (iterative candidate UE group generation)
        // directly continue to the next cell if the number of scheduled SU-MIMO UEs reaches numUeSchdPerCellTTI
        if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
            continue;
        }

        // define constants
        size_t num_elements = static_cast<size_t>(pCpuDynDesc->nBsAnt);

        // step 1: initialize the list of available UEs
        std::vector<uint16_t> avlbUe(newTxUeId); // get a copy of all UEs associated to cell cIdx
        // remove anchor UEs
        avlbUe.erase(std::remove_if(
                     avlbUe.begin(), avlbUe.end(),
                     [&anchorUe](uint16_t value) {
                        // Check if the value is in vector anchorUe
                        return std::find(anchorUe.begin(), anchorUe.end(), value) != anchorUe.end();
                     }), avlbUe.end());

        // step 2: iterative UR group generation
        std::vector<int> subbandId(anchorUe.size());
        for (int i = 0; i < anchorUe.size(); i++) {
            subbandId[i] = i;
        }

        std::random_shuffle(subbandId.begin(), subbandId.end());

        for (int i = 0; i < subbandId.size(); i++) {
            if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                break;
            }

            int sbIdx = subbandId[i];

            int prgStart = pCpuDynDesc->allocSol[2*anchorUe[sbIdx]];
            int prgEndPlusOne = pCpuDynDesc->allocSol[2*anchorUe[sbIdx] + 1];

            float maxPf = 0;
            for (int rbgIdx = prgStart; rbgIdx < prgEndPlusOne; rbgIdx++) {
                maxPf += pfCopy[prgAllocUeId[rbgIdx]][rbgIdx];
            }

            std::vector<uint16_t> bestUeg;
            std::vector<uint16_t> bestUegLayers;
            uint8_t numLayersAdded = pCpuDynDesc->layerSelSol[anchorUe[sbIdx]];
            bestUeg.push_back(anchorUe[sbIdx]);
            bestUegLayers.push_back(pCpuDynDesc->layerSelSol[anchorUe[sbIdx]]);

            for (int uIdx = 0; uIdx < (pCpuDynDesc->nMaxUePerGrpDl - 1); uIdx++) {
                if (schdUeCounter == pCpuDynDesc->numUeSchdPerCellTTI) {
                    break;
                }

                float maxPfItr = -1.0;
                uint16_t bestCandiUe;

                for (uint16_t candiUeIdx = 0; candiUeIdx < avlbUe.size(); candiUeIdx++) {
                    uint8_t numLayersCombined = numLayersAdded + pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]];
                    if (numLayersCombined > pCpuDynDesc->nMaxLayerPerGrpDl) {
                        continue;
                    }

                    float pfTemp = 0;

                    for (int rbgIdx = prgStart; rbgIdx < prgEndPlusOne; rbgIdx++) {
                        std::vector<cuComplex> stackedChannMat;
                    
                        for (int extUeIdx = 0; extUeIdx < bestUeg.size(); extUeIdx++) {
                            for (int lIdx = 0; lIdx < bestUegLayers[extUeIdx]; lIdx++) {
                                cuComplex* channPtr = &pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][bestUeg[extUeIdx]]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd + lIdx*pCpuDynDesc->nBsAnt];
                                stackedChannMat.insert(stackedChannMat.end(), channPtr, channPtr + num_elements);
                            }
                        }

                        for (int lIdx = 0; lIdx < pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]]; lIdx++) {
                            cuComplex* channPtr = &pCpuDynDesc->srsEstChan[cIdx][pCpuDynDesc->srsUeMap[cIdx][avlbUe[candiUeIdx]]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd + lIdx*pCpuDynDesc->nBsAnt];
                            stackedChannMat.insert(stackedChannMat.end(), channPtr, channPtr + num_elements);
                        }

                        // calculate RZF predocder (row-major matrix access)
                        std::vector<cuComplex> rzfPrd_rm(pCpuDynDesc->nBsAnt*numLayersCombined);
                        std::vector<cuComplex> gramMat_rm(numLayersCombined*numLayersCombined);
                        std::vector<cuComplex> invGramMat_rm(numLayersCombined*numLayersCombined);
                
                        for (int i = 0; i < numLayersCombined; i++) {
                            for (int j = 0; j < numLayersCombined; j++) {
                                if (i == j) {
                                    gramMat_rm[i*numLayersCombined + j].x = pCpuDynDesc->zfCoeff;
                                    gramMat_rm[i*numLayersCombined + j].y = 0;
                                } else {
                                    gramMat_rm[i*numLayersCombined + j].x = 0;
                                    gramMat_rm[i*numLayersCombined + j].y = 0;
                                }
                            }
                        }

                        // H*H^H + alpha*I
                        matAlg->matMultiplication_aaHplusb_rm(stackedChannMat.data(), numLayersCombined, pCpuDynDesc->nBsAnt, gramMat_rm.data());
                        // (H*H^H + alpha*I)^-1
                        matAlg->matInverse_rm(gramMat_rm.data(), numLayersCombined, invGramMat_rm.data());
                        // un-normalized RZF precoder = H^H*(H*H^H + alpha*I)^-1
                        matAlg->matMultiplication_aHb_rm(stackedChannMat.data(), numLayersCombined, pCpuDynDesc->nBsAnt, invGramMat_rm.data(), numLayersCombined, rzfPrd_rm.data());

                        // calculate normalizatino factor
                        float trace = 0;
                        for (int i = 0; i < pCpuDynDesc->nBsAnt; i++) {
                            for (int j = 0; j < numLayersCombined; j++) {
                                float x = rzfPrd_rm[i*numLayersCombined + j].x;
                                float y = rzfPrd_rm[i*numLayersCombined + j].y;
                                trace += x*x + y*y;
                            }
                        }

                        float lambda = pCpuDynDesc->bsTxPow[cIdx]/trace;
                        for (int ueIdx = 0; ueIdx < bestUeg.size(); ueIdx++) {
                            float snr = lambda/pCpuDynDesc->noiseVarActUe[bestUeg[ueIdx]];
                            float insRate = bestUegLayers[ueIdx]*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));
                            pfTemp += pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[bestUeg[ueIdx]];
                        }

                        float snr = lambda/pCpuDynDesc->noiseVarActUe[avlbUe[candiUeIdx]];
                        float insRate = pCpuDynDesc->layerSelSol[avlbUe[candiUeIdx]]*pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + snr)));
                        pfTemp += pow(insRate, pCpuDynDesc->betaCoeff)/pCpuDynDesc->avgRatesActUe[avlbUe[candiUeIdx]];
                    }

                    if (pfTemp > maxPfItr) {
                        maxPfItr = pfTemp;
                        bestCandiUe = avlbUe[candiUeIdx];
                    }
                }

                if (maxPfItr >= maxPf) {
                    bestUeg.push_back(bestCandiUe);
                    bestUegLayers.push_back(pCpuDynDesc->layerSelSol[bestCandiUe]);
                    pCpuDynDesc->allocSol[2*bestCandiUe] = static_cast<int16_t>(prgStart);
                    pCpuDynDesc->allocSol[2*bestCandiUe + 1] = static_cast<int16_t>(prgEndPlusOne);

                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeForGrpPerCell + schdUeCounter] = bestCandiUe;
                    schdUeCounter++;

                    numLayersAdded += pCpuDynDesc->layerSelSol[bestCandiUe];
                    avlbUe.erase(std::remove(avlbUe.begin(), avlbUe.end(), bestCandiUe), avlbUe.end());
                } else {
                    break;
                }
            }

            uint8_t layerCounter = 0;
            for (int uIdx = 0; uIdx < bestUeg.size(); uIdx++) {
                // nSCID allocation
                for (int lIdx = 0; lIdx < pCpuDynDesc->nUeAnt; lIdx++) {
                    if (lIdx < bestUegLayers[uIdx]) {
                        pCpuDynDesc->nSCID[bestUeg[uIdx]*pCpuDynDesc->nUeAnt + lIdx] = layerCounter >= totNumPdschDmrsPort_;

                        layerCounter++;
                    } else {
                        pCpuDynDesc->nSCID[bestUeg[uIdx]*pCpuDynDesc->nUeAnt + lIdx] = 0xFF;
                    }
                }
            }
        }
    }
}
}