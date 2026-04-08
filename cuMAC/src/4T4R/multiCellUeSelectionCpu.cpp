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

multiCellUeSelectionCpu::multiCellUeSelectionCpu(cumacCellGrpPrms* cellGrpPrms)
{
    pCpuDynDesc = std::make_unique<mcUeSelDynDescrCpu_t>();

    enableHarq = cellGrpPrms->harqEnabledInd;
}

multiCellUeSelectionCpu::~multiCellUeSelectionCpu() {}

void multiCellUeSelectionCpu::multiCellUeSelCpu()
{
    constexpr float kEps = 1e-6f;

    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        uint16_t cellIdx = pCpuDynDesc->cellId[cIdx];
        std::vector<pfMetricUeSel> pf;

        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cellIdx*pCpuDynDesc->nActiveUe + uIdx]) {
                if (pCpuDynDesc->newDataActUe == nullptr || pCpuDynDesc->newDataActUe[uIdx] == 1) { // new transmission
                    if (pCpuDynDesc->bufferSize != nullptr && pCpuDynDesc->bufferSize[uIdx] == 0) {
                        continue;
                    }

                    pfMetricUeSel pfTemp;

                    float dataRate = 0.0f;
                    for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                        dataRate += log2f(1.0f + pCpuDynDesc->wbSinr[uIdx*pCpuDynDesc->nUeAnt + j]);
                    }
                    dataRate *= pCpuDynDesc->W;
                    pfTemp.first = powf(dataRate, pCpuDynDesc->betaCoeff) / fmaxf(pCpuDynDesc->avgRatesActUe[uIdx], kEps);
                    pfTemp.second = uIdx;

                    pf.push_back(pfTemp);
                } else { // re-transmission
                    pfMetricUeSel pfTemp;

                    pfTemp.first = std::numeric_limits<float>::max();
                    pfTemp.second = uIdx;

                    pf.push_back(pfTemp);
                }
            } 
        }

        std::sort(pf.begin(), pf.end(), [](pfMetricUeSel a, pfMetricUeSel b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        if (pf.size() < pCpuDynDesc->numUeSchdPerCellTTI) { // number of active UEs < required number of scheduled UEs per cell per TTI
            for (int selUeIdx = 0; selUeIdx < pf.size(); selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = pf[selUeIdx].second;
            }

            for (int selUeIdx = pf.size(); selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
            }
        } else {
            for (int selUeIdx = 0; selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = pf[selUeIdx].second;
            }
        }
    }
}

void multiCellUeSelectionCpu::multiCellUeSelCpu_hetero()
{
    constexpr float kEps = 1e-6f;

    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        uint16_t cellIdx = pCpuDynDesc->cellId[cIdx];
        std::vector<pfMetricUeSel> pf;

        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cellIdx*pCpuDynDesc->nActiveUe + uIdx]) {
                if (pCpuDynDesc->newDataActUe == nullptr || pCpuDynDesc->newDataActUe[uIdx] == 1) { // new transmission
                    if (pCpuDynDesc->bufferSize != nullptr && pCpuDynDesc->bufferSize[uIdx] == 0) {
                        continue;
                    }

                    pfMetricUeSel pfTemp;

                    float dataRate = 0.0f;
                    for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                        dataRate += log2f(1.0f + pCpuDynDesc->wbSinr[uIdx*pCpuDynDesc->nUeAnt + j]);
                    }
                    dataRate *= pCpuDynDesc->W;
                    pfTemp.first = powf(dataRate, pCpuDynDesc->betaCoeff) / fmaxf(pCpuDynDesc->avgRatesActUe[uIdx], kEps);
                    pfTemp.second = uIdx;

                    pf.push_back(pfTemp);
                } else { // re-transmission
                    pfMetricUeSel pfTemp;

                    pfTemp.first = std::numeric_limits<float>::max();
                    pfTemp.second = uIdx;

                    pf.push_back(pfTemp);
                }
            } 
        }

        std::sort(pf.begin(), pf.end(), [](pfMetricUeSel a, pfMetricUeSel b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        if (pf.size() < pCpuDynDesc->numUeSchdPerCellTTI) { // number of active UEs < required number of scheduled UEs per cell per TTI
            for (int selUeIdx = 0; selUeIdx < pf.size(); selUeIdx++) {
                if (selUeIdx < pCpuDynDesc->numUeSchdPerCellTTIArr[cIdx]) {
                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = pf[selUeIdx].second;
                } else {
                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
                }
            }

            for (int selUeIdx = pf.size(); selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
            }
        } else {
            for (int selUeIdx = 0; selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                if (selUeIdx < pCpuDynDesc->numUeSchdPerCellTTIArr[cIdx]) {
                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = pf[selUeIdx].second;
                } else {
                    pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
                }
            }
        }
    }
}

void multiCellUeSelectionCpu::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                    cumacSchdSol*               schdSol,
                                    cumacCellGrpPrms*           cellGrpPrms)
{
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
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

    if (cellGrpPrms->numUeSchdPerCellTTIArr) { // heterogeneous UE selection across cells
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
}

void multiCellUeSelectionCpu::run()
{
    if (heteroUeSelCells == 1) { // heterogeneous UE selection across cells
        multiCellUeSelCpu_hetero();
    } else { // homogeneous UE selection across cells
        multiCellUeSelCpu();
    }
}
}