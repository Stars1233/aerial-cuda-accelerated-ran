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

roundRobinSchedulerCpu::roundRobinSchedulerCpu(cumacCellGrpPrms* cellGrpPrms)
{
    pCpuDynDesc = std::make_unique<rrDynDescrCpu_t>();
    enableHarq = cellGrpPrms->harqEnabledInd;
}

roundRobinSchedulerCpu::~roundRobinSchedulerCpu() {}

void roundRobinSchedulerCpu::roundRobinSchedulerCpu_type0()
{
    std::srand(unsigned(std::time(0)));

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network
        int numAssocUe = 0;

        std::vector<int> assocUeIdx;

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx]) {
                numAssocUe++;
                assocUeIdx.push_back(ueIdx);
            }
        }
        
        if (numAssocUe == 0) {
            continue;
        }

        int numAllocRbgPerUe = floor(static_cast<float>(pCpuDynDesc->nPrbGrp)/numAssocUe);

        int numRemainingRgb = pCpuDynDesc->nPrbGrp - numAllocRbgPerUe*numAssocUe;

        std::random_shuffle(assocUeIdx.begin(), assocUeIdx.end());

        int startRbgAlloc = 0;

        for (int tempUeIdx = 0; tempUeIdx < numAssocUe; tempUeIdx++) {
            int ueIdx = assocUeIdx[tempUeIdx];

            if (numRemainingRgb > 0) {
                int endRbgIdx = startRbgAlloc + numAllocRbgPerUe + 1;
                for (int rbgIdx = startRbgAlloc; rbgIdx < endRbgIdx; rbgIdx++) {
                    pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + cIdx] = ueIdx;
                }
                startRbgAlloc += numAllocRbgPerUe + 1;
                numRemainingRgb--;

                pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                int endRbgIdx = startRbgAlloc + numAllocRbgPerUe;
                for (int rbgIdx = startRbgAlloc; rbgIdx < endRbgIdx; rbgIdx++) {
                    pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + cIdx] = ueIdx;
                }
                startRbgAlloc += numAllocRbgPerUe;

                if (numAllocRbgPerUe > 0) {
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
                } else {
                    uint32_t tempPrio = pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] + pCpuDynDesc->prioWeightStep;
                    tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = static_cast<uint16_t>(tempPrio);
                }
            }
        }
    }
}

void roundRobinSchedulerCpu::roundRobinSchedulerCpu_type1()
{
    std::srand(unsigned(std::time(0)));

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network
        int numAssocUe = 0;

        std::vector<int> assocUeIdx;

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx]) {
                numAssocUe++;
                assocUeIdx.push_back(ueIdx);
            }
        }

        if (numAssocUe == 0) {
            continue;
        }
        
        int numAllocRbgPerUe = floor(static_cast<float>(pCpuDynDesc->nPrbGrp)/numAssocUe);

        int numRemainingRgb = pCpuDynDesc->nPrbGrp - numAllocRbgPerUe*numAssocUe;

        std::random_shuffle(assocUeIdx.begin(), assocUeIdx.end());

        int startRbgAlloc = 0;

        for (int tempUeIdx = 0; tempUeIdx < numAssocUe; tempUeIdx++) {
            int ueIdx = assocUeIdx[tempUeIdx];
            
            if (numRemainingRgb > 0) {
                pCpuDynDesc->allocSol[2*ueIdx]   = startRbgAlloc;
                pCpuDynDesc->allocSol[2*ueIdx+1] = startRbgAlloc + numAllocRbgPerUe + 1;
                startRbgAlloc += numAllocRbgPerUe + 1;
                numRemainingRgb--;

                pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                if (numAllocRbgPerUe > 0) {
                    pCpuDynDesc->allocSol[2*ueIdx]   = startRbgAlloc;
                    pCpuDynDesc->allocSol[2*ueIdx+1] = startRbgAlloc + numAllocRbgPerUe;
                    startRbgAlloc += numAllocRbgPerUe;
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
                } else {
                    pCpuDynDesc->allocSol[2*ueIdx]   = -1;
                    pCpuDynDesc->allocSol[2*ueIdx+1] = -1;

                    uint32_t tempPrio = pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] + pCpuDynDesc->prioWeightStep;
                    tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = static_cast<uint16_t>(tempPrio);
                }
            }
        }
    }
}

void roundRobinSchedulerCpu::roundRobinSchedulerCpu_type1_harq()
{
    std::srand(unsigned(std::time(0)));

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network
        int numAssocUeNewTx = 0;
        int numAssocUeReTx  = 0;
        
        std::vector<int> assocUeIdxNewTx;
        std::vector<int> assocUeIdxReTx;
        std::vector<int> numResvdPrgReTx;

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx] == 1) {
                if (pCpuDynDesc->newDataActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] == 1) { // the UE is scheduled for new transmission
                    numAssocUeNewTx++;
                    assocUeIdxNewTx.push_back(ueIdx);
                } else { // the UE is scheduled for re-transmission
                    numAssocUeReTx++;
                    assocUeIdxReTx.push_back(ueIdx);
                    int tempResvdPrg = pCpuDynDesc->allocSolLastTx[ueIdx*2+1] - pCpuDynDesc->allocSolLastTx[ueIdx*2];

                    numResvdPrgReTx.push_back(tempResvdPrg);
                }
            }
        }

        if (numAssocUeNewTx == 0 && numAssocUeReTx == 0) {
            continue;
        }

        int numRemainingPrg = pCpuDynDesc->nPrbGrp;

        int startRbgAlloc = 0;
        for (int uid = 0; uid < numAssocUeReTx; uid++) {
            int ueIdx = assocUeIdxReTx[uid];
            if (numRemainingPrg >= numResvdPrgReTx[ueIdx]) {
                pCpuDynDesc->allocSol[2*ueIdx]   = startRbgAlloc;
                pCpuDynDesc->allocSol[2*ueIdx+1] = startRbgAlloc + numResvdPrgReTx[ueIdx];
                startRbgAlloc += numResvdPrgReTx[ueIdx];
                numRemainingPrg -= numResvdPrgReTx[ueIdx];

                pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                pCpuDynDesc->allocSol[2*ueIdx]   = -1;
                pCpuDynDesc->allocSol[2*ueIdx+1] = -1;

                pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0xFFFF;
            }
        }

        int numAllocRbgPerUe = floor(static_cast<float>(numRemainingPrg)/numAssocUeNewTx);
        
        int numRemainingRgb = numRemainingPrg - numAllocRbgPerUe*numAssocUeNewTx;

        std::random_shuffle(assocUeIdxNewTx.begin(), assocUeIdxNewTx.end());

        for (int uid = 0; uid < numAssocUeNewTx; uid++) {
            int ueIdx = assocUeIdxNewTx[uid];

            if (numRemainingRgb > 0) {
                pCpuDynDesc->allocSol[2*ueIdx]   = startRbgAlloc;
                pCpuDynDesc->allocSol[2*ueIdx+1] = startRbgAlloc + numAllocRbgPerUe + 1;
                startRbgAlloc += numAllocRbgPerUe + 1;
                numRemainingRgb--;

                pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
            } else {
                if (numAllocRbgPerUe > 0) {
                    pCpuDynDesc->allocSol[2*ueIdx]   = startRbgAlloc;
                    pCpuDynDesc->allocSol[2*ueIdx+1] = startRbgAlloc + numAllocRbgPerUe;
                    startRbgAlloc += numAllocRbgPerUe;
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = 0;
                } else {
                    pCpuDynDesc->allocSol[2*ueIdx]   = -1;
                    pCpuDynDesc->allocSol[2*ueIdx+1] = -1;

                    uint32_t tempPrio = pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] + pCpuDynDesc->prioWeightStep;
                    tempPrio = tempPrio > 0x0000FFFF ? 0x0000FFFF : tempPrio;
                    pCpuDynDesc->prioWeightActUe[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]] = static_cast<uint16_t>(tempPrio);
                }
            }
        }
    }
}

void roundRobinSchedulerCpu::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                   cumacSchdSol*               schdSol,
                                   cumacCellGrpPrms*           cellGrpPrms)
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
}

 void roundRobinSchedulerCpu::run()
 {
    if (allocType == 1) { // type-1 allocation
        if (enableHarq == 1) { // HARQ enabled
            roundRobinSchedulerCpu_type1_harq();
        } else { // HARQ disabled
            roundRobinSchedulerCpu_type1();
        }
    } else { // type-0 allocation
        if (enableHarq == 1) { // HARQ enabled
            printf("Error: For type-0 allocation, CPU RR scheduler is only supported for HARQ disabled.\n");
            return;
        } else { // HARQ disabled
            roundRobinSchedulerCpu_type0();
        }
    }
 }
}
