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

roundRobinUeSelCpu::roundRobinUeSelCpu(cumacCellGrpPrms* cellGrpPrms)
{
    pCpuDynDesc = std::make_unique<rrUeSelDynDescrCpu_t>();
    enableHarq = cellGrpPrms->harqEnabledInd;
}

roundRobinUeSelCpu::~roundRobinUeSelCpu() {}

void roundRobinUeSelCpu::setup(cumacCellGrpUeStatus*     cellGrpUeStatus,
                               cumacSchdSol*             schdSol,
                               cumacCellGrpPrms*         cellGrpPrms)
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
}

void roundRobinUeSelCpu::run()
{
    if (enableHarq == 1) { // HARQ is enabled
        rrUeSelCpu_harq();
    } else { // HARQ is disabled
        rrUeSelCpu();
    }
}

void roundRobinUeSelCpu::rrUeSelCpu()
{
    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        uint16_t cellIdx = pCpuDynDesc->cellId[cIdx];
        std::vector<prioUeSel> prioVec;

        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cellIdx*pCpuDynDesc->nActiveUe + uIdx] == 1) {
                if (pCpuDynDesc->bufferSize != nullptr && pCpuDynDesc->bufferSize[uIdx] == 0) {
                    continue;
                }

                prioUeSel prioTemp;
                prioTemp.first = static_cast<float>(pCpuDynDesc->prioWeightActUe[uIdx]);
                prioTemp.second = uIdx;
                prioVec.push_back(prioTemp);
            } 
        }

        std::sort(prioVec.begin(), prioVec.end(), [](prioUeSel a, prioUeSel b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        if (prioVec.size() < pCpuDynDesc->numUeSchdPerCellTTI) { // number of active UEs < required number of scheduled UEs per cell per TTI
            for (int selUeIdx = 0; selUeIdx < prioVec.size(); selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = prioVec[selUeIdx].second;
            }

            for (int selUeIdx = prioVec.size(); selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
            }
        } else {
            for (int selUeIdx = 0; selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = prioVec[selUeIdx].second;
            }

            for (int uid = pCpuDynDesc->numUeSchdPerCellTTI; uid < prioVec.size(); uid++) {
                uint32_t tempPrio = pCpuDynDesc->prioWeightActUe[prioVec[uid].second] + pCpuDynDesc->prioWeightStep;
                tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                pCpuDynDesc->prioWeightActUe[prioVec[uid].second] = static_cast<uint16_t>(tempPrio);
            }
        }
    }
}

void roundRobinUeSelCpu::rrUeSelCpu_harq()
{
    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        uint16_t cellIdx = pCpuDynDesc->cellId[cIdx];
        std::vector<prioUeSel> prioVec;

        for (int uIdx = 0; uIdx < pCpuDynDesc->nActiveUe; uIdx++) {
            if (pCpuDynDesc->cellAssocActUe[cellIdx*pCpuDynDesc->nActiveUe + uIdx] == 1) {
                if (pCpuDynDesc->newDataActUe[uIdx] == 1) { // new transmission
                    if (pCpuDynDesc->bufferSize != nullptr && pCpuDynDesc->bufferSize[uIdx] == 0) {
                        continue;
                    }

                    prioUeSel prioTemp;

                    prioTemp.first = static_cast<float>(pCpuDynDesc->prioWeightActUe[uIdx]);
                    prioTemp.second = uIdx;

                    prioVec.push_back(prioTemp);
                } else { // re-transmission
                    prioUeSel prioTemp;

                    prioTemp.first = static_cast<float>(0xFFFF); // assign highest priority for re-tx
                    prioTemp.second = uIdx;
                    
                    prioVec.push_back(prioTemp);
                }
            } 
        }

        std::sort(prioVec.begin(), prioVec.end(), [](prioUeSel a, prioUeSel b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        if (prioVec.size() < pCpuDynDesc->numUeSchdPerCellTTI) { // number of active UEs < required number of scheduled UEs per cell per TTI
            for (int selUeIdx = 0; selUeIdx < prioVec.size(); selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = prioVec[selUeIdx].second;
            }

            for (int selUeIdx = prioVec.size(); selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = 0xFFFF;
            }
        } else {
            for (int selUeIdx = 0; selUeIdx < pCpuDynDesc->numUeSchdPerCellTTI; selUeIdx++) {
                pCpuDynDesc->setSchdUePerCellTTI[cIdx*pCpuDynDesc->numUeSchdPerCellTTI + selUeIdx] = prioVec[selUeIdx].second;
            }

            for (int uid = pCpuDynDesc->numUeSchdPerCellTTI; uid < prioVec.size(); uid++) {
                uint32_t tempPrio = pCpuDynDesc->prioWeightActUe[prioVec[uid].second] + pCpuDynDesc->prioWeightStep;
                tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                pCpuDynDesc->prioWeightActUe[prioVec[uid].second] = static_cast<uint16_t>(tempPrio);
            }
        }
    }
}

void roundRobinUeSelCpu::debugLog()
{

}
}