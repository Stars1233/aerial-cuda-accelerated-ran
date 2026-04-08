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

mcsSelectionLUTCpu::mcsSelectionLUTCpu(cumacCellGrpPrms*   cellGrpPrms)
{
    if (cellGrpPrms->nBsAnt != 4) {
        throw std::runtime_error("Error: CPU LUT-based MCS selection is only supported for 4T4R configuration");
    }

    m_nUe = cellGrpPrms->nActiveUe;
    CQI = cellGrpPrms->mcsSelCqi;
    allocType = cellGrpPrms->allocType;
    enableHarq = cellGrpPrms->harqEnabledInd;

    if (allocType == 0 && cellGrpPrms->nBsAnt != 4) {
        throw std::runtime_error("Error: CPU LUT-based MCS selection for type-0 PRB allocation is only supported for 4T4R configuration");
    }

    if (CQI == 1) {
        throw std::runtime_error("Error: CPU LUT-based MCS selection is only supported for wideband SINR-based MCS selection");
    }
    
    pDynDescr = std::make_unique<mcsSelDynDescrCpu_t>();
    matAlg = std::make_unique<cpuMatAlg>();
    ollaParamArr = std::vector<ollaParam>(m_nUe);

    for (int ueIdx = 0; ueIdx < m_nUe; ueIdx++) {
        ollaParamArr[ueIdx].delta_ini   = 0.0;
        ollaParamArr[ueIdx].delta       = 0.0;
        ollaParamArr[ueIdx].delta_up    = 1.0;
        ollaParamArr[ueIdx].delta_down  = 1.0*cellGrpPrms->blerTargetActUe[ueIdx];
    }
}

void mcsSelectionLUTCpu::setup(cumacCellGrpUeStatus*  cellGrpUeStatus,
                               cumacSchdSol*          schdSol,
                               cumacCellGrpPrms*      cellGrpPrms)
{
    pDynDescr->nUe                    = cellGrpPrms->nUe; 
    pDynDescr->nCell                  = cellGrpPrms->nCell;
    pDynDescr->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pDynDescr->nUeAnt                 = cellGrpPrms->nUeAnt;
    pDynDescr->nBsAnt                 = cellGrpPrms->nBsAnt;
    pDynDescr->mcsSelLutType          = cellGrpPrms->mcsSelLutType;
    pDynDescr->mcsSelSinrCapThr       = cellGrpPrms->mcsSelSinrCapThr;
    pDynDescr->ollaParamArr           = ollaParamArr.data();
    pDynDescr->wbSinr                 = cellGrpPrms->wbSinr;
    pDynDescr->tbErrLast              = cellGrpUeStatus->tbErrLast;
    pDynDescr->beamformGainLastTx     = cellGrpUeStatus->beamformGainLastTx;  
    pDynDescr->beamformGainCurrTx     = cellGrpUeStatus->beamformGainCurrTx;
    pDynDescr->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pDynDescr->allocSol               = schdSol->allocSol;
    pDynDescr->mcsSelSol              = schdSol->mcsSelSol;

    if (CQI == 1) {
        pDynDescr->cqiActUe           = cellGrpUeStatus->cqiActUe;
    } else {
        pDynDescr->cqiActUe           = nullptr;
    }

    if (enableHarq == 1) { // HARQ enabled
        pDynDescr->newDataActUe       = cellGrpUeStatus->newDataActUe;
        pDynDescr->mcsSelSolLastTx    = cellGrpUeStatus->mcsSelSolLastTx;
    } else {
        pDynDescr->newDataActUe       = nullptr;
        pDynDescr->mcsSelSolLastTx    = nullptr;
    }
}

void mcsSelectionLUTCpu::run()
{
    if (allocType == 0) {
        mcsSelSinrRepKernel_type0_wbSinr();
    } else if (allocType == 1) {    
        mcsSelSinrRepKernel_type1_wbSinr();
    } else {
        throw std::runtime_error("Error: invalid PRB allocation type");
    }
}

void mcsSelectionLUTCpu::mcsSelSinrRepKernel_type0_wbSinr()
{
    for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
        uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

        if (globalUidx == 0xFFFF) {
            continue;
        }

        uint16_t numAllocPrg = 0; // number of allocated PRGs to the considered UE
        for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
            for (int cIdx = 0; cIdx < pDynDescr->nCell; cIdx++) {
                if (pDynDescr->allocSol[prgIdx*pDynDescr->nCell + cIdx] == uIdx) {
                    numAllocPrg++;
                }
            }
        }

        int8_t   tbErrLast = -1;

        if (numAllocPrg > 0) { // being scheduled for the current slot
            tbErrLast = pDynDescr->tbErrLast[uIdx];
            if (tbErrLast == 1) {
                pDynDescr->ollaParamArr[globalUidx].delta += pDynDescr->ollaParamArr[globalUidx].delta_up;
            } else if (tbErrLast == 0) {
                pDynDescr->ollaParamArr[globalUidx].delta -= pDynDescr->ollaParamArr[globalUidx].delta_down;
            }

            if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
                double dAvgSinrDB = 10.0*log10(static_cast<double>(pDynDescr->wbSinr[globalUidx*pDynDescr->nUeAnt]));
                float avgSinrDB = static_cast<float>(dAvgSinrDB);

                avgSinrDB = avgSinrDB > pDynDescr->mcsSelSinrCapThr ? pDynDescr->mcsSelSinrCapThr : avgSinrDB;

                avgSinrDB -= pDynDescr->ollaParamArr[globalUidx].delta;

                uint8_t selectedMcs = 0;
                for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                    if (pDynDescr->mcsSelLutType == 0) {
                        if (avgSinrDB >= L1T1B024[mcsIdx]) {
                            selectedMcs = mcsIdx;
                            break;
                        }
                    } else if (pDynDescr->mcsSelLutType == 1) {
                        if (avgSinrDB >= L1T1B050PRGS01_GTC25[mcsIdx]) {
                            selectedMcs = mcsIdx;
                            break;
                        }
                    }
                }

                pDynDescr->mcsSelSol[uIdx] = selectedMcs;
            } else { // the UE is scheduled for re-transmission
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            }
        } else {
            pDynDescr->mcsSelSol[uIdx] = -1;
        }
    }
}

void mcsSelectionLUTCpu::mcsSelSinrRepKernel_type1_wbSinr()
{
    for (int uIdx = 0; uIdx < pDynDescr->nUe; uIdx++) {
        uint16_t globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];

        if (globalUidx == 0xFFFF) {
            continue;
        }

        uint16_t numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx]; // number of allocated PRGs to the considered UE

        int8_t   tbErrLast = -1;

        if (numAllocPrg > 0) { // being scheduled for the current slot
            tbErrLast = pDynDescr->tbErrLast[uIdx];
            if (tbErrLast == 1) {
                pDynDescr->ollaParamArr[globalUidx].delta += pDynDescr->ollaParamArr[globalUidx].delta_up;
            } else if (tbErrLast == 0) {
                pDynDescr->ollaParamArr[globalUidx].delta -= pDynDescr->ollaParamArr[globalUidx].delta_down;
            }

            if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
                double dAvgSinrDB = 10.0*log10(static_cast<double>(pDynDescr->wbSinr[globalUidx*pDynDescr->nUeAnt]));
                float avgSinrDB = static_cast<float>(dAvgSinrDB);

                avgSinrDB = avgSinrDB > pDynDescr->mcsSelSinrCapThr ? pDynDescr->mcsSelSinrCapThr : avgSinrDB;

                avgSinrDB -= pDynDescr->ollaParamArr[globalUidx].delta;

                uint8_t selectedMcs = 0;
                for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                    if (pDynDescr->mcsSelLutType == 0) {
                        if (avgSinrDB >= L1T1B024[mcsIdx]) {
                            selectedMcs = mcsIdx;
                            break;
                        }
                    } else if (pDynDescr->mcsSelLutType == 1) {
                        if (avgSinrDB >= L1T1B050PRGS01_GTC25[mcsIdx]) {
                            selectedMcs = mcsIdx;
                            break;
                        }
                    }
                }

                pDynDescr->mcsSelSol[uIdx] = selectedMcs;
            } else { // the UE is scheduled for re-transmission
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            }
        } else {
            pDynDescr->mcsSelSol[uIdx] = -1;
        }
    }
}

void mcsSelectionLUTCpu::debugLog()
{
    printf("CPU delta: ");
    for (int ueIdx = 0; ueIdx < m_nUe; ueIdx++) {
        printf("%f ", pDynDescr->ollaParamArr[ueIdx].delta);
    }
    printf("\n");
}
}