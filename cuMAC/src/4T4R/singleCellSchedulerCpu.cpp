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

singleCellSchedulerCpu::singleCellSchedulerCpu()
{
    matAlg      = new cpuMatAlg;
    pCpuDynDesc = new scDynDescrCpu_t;
}

singleCellSchedulerCpu::~singleCellSchedulerCpu()
{
    delete matAlg;
    delete pCpuDynDesc;
}

void singleCellSchedulerCpu::singleCellSchedulerCpu_noPrdMmse()
{
    uint16_t nBsAntSqrd                 = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint16_t nBsUeAntPrd                = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    cuComplex* CMat                     = new cuComplex[nBsAntSqrd]; 
    cuComplex* CInvMat                  = new cuComplex[nBsAntSqrd];

    for (uint16_t rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
        float   maxv = 0;
        int16_t maxi = -1;
        for (uint16_t ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[pCpuDynDesc->cellId*pCpuDynDesc->nUe + ueIdx])
                    continue;
        
            matAlg->matMultiplication_aHa(&pCpuDynDesc->estH_fr_perUeBuffer[ueIdx][rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);

            for (uint16_t eIdx = 0; eIdx < pCpuDynDesc->nBsAnt; eIdx++) {
                CMat[eIdx*pCpuDynDesc->nBsAnt+eIdx].x += pCpuDynDesc->sigmaSqrd;
            }

            matAlg->matInverse(CMat, pCpuDynDesc->nBsAnt, CInvMat);
            //matAlg->matInverseEigen(CMat, pCpuDynDesc->nBsAnt, CInvMat);

            // evaluate PF metric for each UE
            float dataRate = 0;

            for (uint16_t j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                dataRate += pCpuDynDesc->W*log2f(1.0/pCpuDynDesc->sigmaSqrd/CInvMat[j*pCpuDynDesc->nBsAnt+j].x);
            }
            float pfMetric = dataRate / pCpuDynDesc->avgRates[ueIdx];

            if (pfMetric > maxv) {
                maxv = pfMetric;
                maxi = ueIdx;
            }
        }
        pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + pCpuDynDesc->cellId] = maxi;
        // printf("CPU: cellId = %d, rbgIdx = %d, maxv = %f, pDynDescr->allocSol = %d\n", pCpuDynDesc->cellId, rbgIdx, maxv, pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + pCpuDynDesc->cellId]);
    }

    delete CMat;
    delete CInvMat;
}

void singleCellSchedulerCpu::singleCellSchedulerCpu_svdMmse()
{
    uint16_t nBsAntSqrd                 = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint16_t nBsUeAntPrd                = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsAntSqrd             = pCpuDynDesc->nCell*nBsAntSqrd;
    cuComplex* CMat                     = new cuComplex[nBsAntSqrd]; 
    cuComplex* CInvMat                  = new cuComplex[nBsAntSqrd];
    cuComplex* HVMat                    = new cuComplex[nBsUeAntPrd];

    for (uint16_t rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
        float   maxv = 0;
        int16_t maxi = -1;
        for (uint16_t ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[pCpuDynDesc->cellId*pCpuDynDesc->nUe + ueIdx])
                    continue;

            uint32_t vMatStart = rbgIdx*pCpuDynDesc->nUe*nCellBsAntSqrd + ueIdx*nCellBsAntSqrd + pCpuDynDesc->cellId*nBsAntSqrd;
            matAlg->matMultiplication_ab(&pCpuDynDesc->estH_fr_perUeBuffer[ueIdx][rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, &pCpuDynDesc->prdMat[vMatStart], pCpuDynDesc->nBsAnt, HVMat);
        
            matAlg->matMultiplication_aHa(HVMat, pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);

            for (uint16_t eIdx = 0; eIdx < pCpuDynDesc->nBsAnt; eIdx++) {
                CMat[eIdx*pCpuDynDesc->nBsAnt+eIdx].x += pCpuDynDesc->sigmaSqrd;
            }

            matAlg->matInverse(CMat, pCpuDynDesc->nBsAnt, CInvMat);
            //matAlg->matInverseEigen(CMat, pCpuDynDesc->nBsAnt, CInvMat);

            // evaluate PF metric for each UE
            float dataRate = 0;

            for (uint16_t j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                dataRate += pCpuDynDesc->W*log2f(1.0/pCpuDynDesc->sigmaSqrd/CInvMat[j*pCpuDynDesc->nBsAnt+j].x);
            }
            float pfMetric = dataRate / pCpuDynDesc->avgRates[ueIdx];

            if (pfMetric > maxv) {
                maxv = pfMetric;
                maxi = ueIdx;
            }  
        }
        pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + pCpuDynDesc->cellId] = maxi;
        // printf("CPU: cellId = %d, rbgIdx = %d, maxv = %f, pDynDescr->allocSol = %d\n", pCpuDynDesc->cellId, rbgIdx, maxv, pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->nCell + pCpuDynDesc->cellId]);
    }

    delete CMat;
    delete CInvMat;
    delete HVMat;
}

void singleCellSchedulerCpu::singleCellSchedulerCpu_type1_NoPrdMmse()
{
    uint16_t nBsAntSqrd                 = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint16_t nBsUeAntPrd                = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    cuComplex* CMat                     = new cuComplex[nBsAntSqrd]; 
    cuComplex* CInvMat                  = new cuComplex[nBsAntSqrd];

    std::vector<pfMetric> pf;

    for (uint32_t rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
        for (uint32_t ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[pCpuDynDesc->cellId*pCpuDynDesc->nUe + ueIdx])
                    continue;

            pfMetric pfTemp;
            pfTemp.second = ueIdx*static_cast<uint32_t>(pCpuDynDesc->nPrbGrp) + rbgIdx;

            matAlg->matMultiplication_aHa(&pCpuDynDesc->estH_fr_perUeBuffer[ueIdx][rbgIdx*nBsUeAntPrd], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);

            for (uint16_t eIdx = 0; eIdx < pCpuDynDesc->nBsAnt; eIdx++) {
                CMat[eIdx*pCpuDynDesc->nBsAnt+eIdx].x += pCpuDynDesc->sigmaSqrd;
            }

            matAlg->matInverse(CMat, pCpuDynDesc->nBsAnt, CInvMat);
            //matAlg->matInverseEigen(CMat, pCpuDynDesc->nBsAnt, CInvMat);

            // evaluate PF metric for each UE
            float dataRate = 0;

            for (uint16_t j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                dataRate += pCpuDynDesc->W*log2f(1.0/pCpuDynDesc->sigmaSqrd/CInvMat[j*pCpuDynDesc->nBsAnt+j].x);
            }
            pfTemp.first = dataRate / pCpuDynDesc->avgRates[ueIdx];
            pf.push_back(pfTemp);
        }  
    }

    // sort all computed PF metrices across all PRB groups and UEs
    std::sort(pf.begin(), pf.end(), [](pfMetric a, pfMetric b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

    // consecutive allocate algorithm
    // initialize schedule solution arrays
    for (uint32_t ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
        if (!pCpuDynDesc->cellAssoc[pCpuDynDesc->cellId*pCpuDynDesc->nUe + ueIdx])
            continue;

        pCpuDynDesc->allocSol[ueIdx*2] = -1;
        pCpuDynDesc->allocSol[ueIdx*2+1] = -1;
    }

    uint8_t* S  = new uint8_t[pCpuDynDesc->nPrbGrp];
    for (uint32_t rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
        S[rbgIdx] = 1;
    }

    uint16_t nAllocated = 0;
    uint16_t k = 0;

    while(nAllocated < pCpuDynDesc->nPrbGrp) {
        if (k == pf.size())
            break;

        uint16_t c = pf[k].second%pCpuDynDesc->nPrbGrp;

        if (S[c] == 0) {
            k++;
            continue;
        }

        uint16_t i = pf[k].second/pCpuDynDesc->nPrbGrp;

        if (pCpuDynDesc->allocSol[2*i] == -1) {
            pCpuDynDesc->allocSol[2*i] = c;
            pCpuDynDesc->allocSol[2*i+1] = c + 1;
            S[c] = 0;
            nAllocated++;
            pf.erase(pf.begin()+k);
            k = 0;
        } else if (c == (pCpuDynDesc->allocSol[2*i] - 1)) {
            pCpuDynDesc->allocSol[2*i] = c;
            S[c] = 0;
            nAllocated++;
            pf.erase(pf.begin()+k);
            k = 0;
        } else if (c == pCpuDynDesc->allocSol[2*i+1]) {
            pCpuDynDesc->allocSol[2*i+1] = c + 1;
            S[c] = 0;
            nAllocated++;
            pf.erase(pf.begin()+k);
            k = 0;
        } else {
            k++;
        }            
    }

    delete S;
    delete CMat;
    delete CInvMat;
}

void singleCellSchedulerCpu::setup(uint16_t                    cellId, // index of the cell that calling the single-cell scheduler
                                   cumacCellGrpUeStatus*       cellGrpUeStatus,
                                   cumacSchdSol*               schdSol,
                                   cumacCellGrpPrms*           cellGrpPrms,
                                   cumacSimParam*              simParam)
{
    pCpuDynDesc->cellId                 = cellId; // index of the cell that calling the single-cell scheduler
    pCpuDynDesc->avgRates               = cellGrpUeStatus->avgRates;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->cellAssoc              = cellGrpPrms->cellAssoc;
    pCpuDynDesc->estH_fr_perUeBuffer    = cellGrpPrms->estH_fr_perUeBuffer;
    pCpuDynDesc->prdMat                 = cellGrpPrms->prdMat;
    pCpuDynDesc->nUe                    = cellGrpPrms->nUe;
    pCpuDynDesc->nCell                  = simParam->totNumCell; // total number of cells in the network
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->W                      = cellGrpPrms->W;
    pCpuDynDesc->sigmaSqrd              = cellGrpPrms->sigmaSqrd;
    precodingScheme                     = cellGrpPrms->precodingScheme;
    allocType                           = cellGrpPrms->allocType;
}

void singleCellSchedulerCpu::run()
{
#ifdef CPU_SCHEDULER_TIME_MEASURE_
    auto start = std::chrono::steady_clock::now();
    for (int idx = 0; idx < numRunSchKnlTimeMsr; idx++) {
#endif
    switch (precodingScheme) {
        case 0: // no precoding
            if (allocType) {
                singleCellSchedulerCpu_type1_NoPrdMmse();
            } else {
                singleCellSchedulerCpu_noPrdMmse();
            }
            break;
        case 1: // SVD precoding
            if (allocType) {
                // to-do
            } else {
                singleCellSchedulerCpu_svdMmse();
            }
            break;
        default: // default no precoding with allocate type 0 (non-consecutive)
            singleCellSchedulerCpu_noPrdMmse();
            break;
    }
#ifdef CPU_SCHEDULER_TIME_MEASURE_
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("CPU single-cell PF scheduler elapsed time: %f ms\n", 1000.0*elapsed_seconds.count()/static_cast<float>(numRunSchKnlTimeMsr));
#endif
}
}