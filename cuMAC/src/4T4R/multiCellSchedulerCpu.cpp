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

// #define CPU_SCHEDULER_TIME_MEASURE_ 
#ifdef CPU_SCHEDULER_TIME_MEASURE_
#define numRunSchKnlTimeMsr    1000
#endif

multiCellSchedulerCpu::multiCellSchedulerCpu(cumacCellGrpPrms* cellGrpPrms)
{
     matAlg      = std::make_unique<cpuMatAlg>();
     pCpuDynDesc = std::make_unique<mcDynDescrCpu_t>();

     // set default to column-major channel matrix access
     columnMajor = 1;

     DL = cellGrpPrms->dlSchInd;
}

multiCellSchedulerCpu::~multiCellSchedulerCpu() {}

void multiCellSchedulerCpu::multiCellSchedulerCpu_noPrdMmse()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;

    cuComplex* CMat             = new cuComplex[nUeAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nUeAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsUeAntPrd];  
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network
        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            float   maxv = 0;
            int16_t maxi = -1;

            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                // if UE is associated with the considered cell
                // compute C matrix
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nUeAnt; rowIdx++) {
                    for (int colIdx = 0; colIdx < pCpuDynDesc->nUeAnt; colIdx++) {
                        if (rowIdx == colIdx) {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = pCpuDynDesc->sigmaSqrd;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        } else {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = 0;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        }
                        
                    }
                }
                uint32_t hTemp = rbgIdx*nUeCellBsUeAntPrd+ ueIdx*nCellBsUeAntPrd;
                for (int lIdx = 0; lIdx < pCpuDynDesc->nCell; lIdx++) {
                    int l = pCpuDynDesc->cellId[lIdx];
                    if (l == cIdx) {
                        continue;
                    }
                    uint32_t hInterfMatStart = hTemp + l*nBsUeAntPrd;
                    matAlg->matMultiplication_aaHplusb(&pCpuDynDesc->estH_fr[hInterfMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);
                }
                // compute inverse of C matrix, C^-1, and store result in CInvMat
                matAlg->matInverse(CMat, pCpuDynDesc->nUeAnt, CInvMat);

                // compute H^H*C^-1
                uint32_t hMatStart = hTemp + cIdx*nBsUeAntPrd;
                matAlg->matMultiplication_aHb(&pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CInvMat, pCpuDynDesc->nUeAnt, DMat);

                // compute H^H*C^-1*H + I
                matAlg->matMultiplication_ab(DMat, pCpuDynDesc->nBsAnt, pCpuDynDesc->nUeAnt, &pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nBsAnt, EMat);
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nBsAnt; rowIdx++) {
                    EMat[rowIdx*pCpuDynDesc->nBsAnt+rowIdx].x += 1.0;
                }

                // compute (H^H*C^-1*H + I)^-1
                matAlg->matInverse(EMat, pCpuDynDesc->nBsAnt, EInvMat);
                
                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = 1.0/EInvMat[j*pCpuDynDesc->nBsAnt+j].x;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp - 1.0;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(sinrTemp)));
                }
                float pfMetric = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];
                if (pfMetric > maxv) {
                    maxv = pfMetric;
                    maxi = ueIdx;
                }  
            }
            pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->totNumCell + cIdx] = maxi;
        }
    }

    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
}

void multiCellSchedulerCpu::multiCellSchedulerCpu_svdMmse()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;
    uint32_t nCellBsAntSqrd     = pCpuDynDesc->totNumCell*nBsAntSqrd;
    uint32_t nUeCellBsAntSqrd   = pCpuDynDesc->nUe*nCellBsAntSqrd;

    cuComplex* BMat             = new cuComplex[nBsUeAntPrd];
    cuComplex* CMat             = new cuComplex[nUeAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nUeAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsUeAntPrd];  
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];


    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            float   maxv = 0;
            int16_t maxi = -1;

            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                // if UE is associated with the considered cell
                // compute C matrix
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nUeAnt; rowIdx++) {
                    for (int colIdx = 0; colIdx < pCpuDynDesc->nUeAnt; colIdx++) {
                        if (rowIdx == colIdx) {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = pCpuDynDesc->sigmaSqrd;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        } else {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = 0;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        }
                        
                    }
                }
                uint32_t hTemp = rbgIdx*nUeCellBsUeAntPrd+ ueIdx*nCellBsUeAntPrd;
                for (int lIdx = 0; lIdx < pCpuDynDesc->nCell; lIdx++) {
                    int l = pCpuDynDesc->cellId[lIdx];

                    if (l == cIdx) {
                        continue;
                    }
                    uint32_t hInterfMatStart = hTemp + l*nBsUeAntPrd;
                    matAlg->matMultiplication_aaHplusb(&pCpuDynDesc->estH_fr[hInterfMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);
                }
                // compute inverse of C matrix, C^-1, and store result in CInvMat
                matAlg->matInverse(CMat, pCpuDynDesc->nUeAnt, CInvMat);

                // compute H*V
                uint32_t hMatStart = hTemp + cIdx*nBsUeAntPrd;
                uint32_t vMatStart = ueIdx*pCpuDynDesc->nPrbGrp*nBsAntSqrd + rbgIdx*nBsAntSqrd;
                matAlg->matMultiplication_ab(&pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, &pCpuDynDesc->prdMat[vMatStart], pCpuDynDesc->nBsAnt, BMat);
                
                // compute (H*V)^H*C^-1
                matAlg->matMultiplication_aHb(BMat, pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CInvMat, pCpuDynDesc->nUeAnt, DMat);

                // compute (H*V)^H*C^-1*(H*V) + I
                matAlg->matMultiplication_ab(DMat, pCpuDynDesc->nBsAnt, pCpuDynDesc->nUeAnt, BMat, pCpuDynDesc->nBsAnt, EMat);
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nBsAnt; rowIdx++) {
                    EMat[rowIdx*pCpuDynDesc->nBsAnt+rowIdx].x += 1.0;
                }

                // compute ((H*V)^H*C^-1*(H*V) + I)^-1
                matAlg->matInverse(EMat, pCpuDynDesc->nBsAnt, EInvMat);
                
                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = 1.0/EInvMat[j*pCpuDynDesc->nBsAnt+j].x;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp - 1.0;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(sinrTemp)));
                }
                float pfMetric = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];

                if (pfMetric > maxv) {
                    maxv = pfMetric;
                    maxi = ueIdx;
                }  
            }
            pCpuDynDesc->allocSol[rbgIdx*pCpuDynDesc->totNumCell + cIdx] = maxi;
        }
    }
    
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
}

void multiCellSchedulerCpu::multiCellSchedulerCpu_type1_svdPrdMmse_cm()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;

    cuComplex* BMat             = new cuComplex[nBsUeAntPrd];
    cuComplex* CMat             = new cuComplex[nUeAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nUeAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsUeAntPrd];  
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];

    uint8_t* S                  = new uint8_t[pCpuDynDesc->nPrbGrp];

    // initialize scheduling solution
    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                continue;

            pCpuDynDesc->allocSol[ueIdx*2] = -1;
            pCpuDynDesc->allocSol[ueIdx*2+1] = -1;
        }
    }

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        std::vector<pfMetric> pf;

        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                pfMetric pfTemp;
                pfTemp.second = ueIdx*static_cast<uint32_t>(pCpuDynDesc->nPrbGrp) + rbgIdx;    

                // if UE is associated with the considered cell
                // compute C matrix
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nUeAnt; rowIdx++) {
                    for (int colIdx = 0; colIdx < pCpuDynDesc->nUeAnt; colIdx++) {
                        if (rowIdx == colIdx) {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = pCpuDynDesc->sigmaSqrd;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        } else {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = 0;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        }
                        
                    }
                }
                uint32_t hTemp = rbgIdx*nUeCellBsUeAntPrd+ ueIdx*nCellBsUeAntPrd;
                for (int lIdx = 0; lIdx < pCpuDynDesc->nCell; lIdx++) {
                    int l = pCpuDynDesc->cellId[lIdx];

                    if (l == cIdx) {
                        continue;
                    }
                    uint32_t hInterfMatStart = hTemp + l*nBsUeAntPrd;
                    matAlg->matMultiplication_aaHplusb(&pCpuDynDesc->estH_fr[hInterfMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);
                }
                // compute inverse of C matrix, C^-1, and store result in CInvMat
                matAlg->matInverse(CMat, pCpuDynDesc->nUeAnt, CInvMat);

                // compute H*V
                uint32_t hMatStart = hTemp + cIdx*nBsUeAntPrd;
                uint32_t vMatStart = ueIdx*pCpuDynDesc->nPrbGrp*nBsAntSqrd + rbgIdx*nBsAntSqrd;
                matAlg->matMultiplication_ab(&pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, &pCpuDynDesc->prdMat[vMatStart], pCpuDynDesc->nBsAnt, BMat);
                
                // compute (H*V)^H*C^-1
                matAlg->matMultiplication_aHb(BMat, pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CInvMat, pCpuDynDesc->nUeAnt, DMat);

                // compute (H*V)^H*C^-1*(H*V) + I
                matAlg->matMultiplication_ab(DMat, pCpuDynDesc->nBsAnt, pCpuDynDesc->nUeAnt, BMat, pCpuDynDesc->nBsAnt, EMat);

                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nBsAnt; rowIdx++) {
                    EMat[rowIdx*pCpuDynDesc->nBsAnt+rowIdx].x += 1.0;
                }

                // compute ((H*V)^H*C^-1*(H*V) + I)^-1
                matAlg->matInverse(EMat, pCpuDynDesc->nBsAnt, EInvMat);
                
                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = 1.0/EInvMat[j*pCpuDynDesc->nBsAnt+j].x;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp - 1.0;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(sinrTemp)));
                }

                pfTemp.first = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];
                pf.push_back(pfTemp);
            }
        }

        // sort all computed PF metrices across all PRB groups and UEs
        std::sort(pf.begin(), pf.end(), [](pfMetric a, pfMetric b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        std::copy (pf.begin(), pf.end(), pfRecord.begin()+ cellIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI);

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
    }

    delete BMat;
    delete S;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
}

void multiCellSchedulerCpu::multiCellSchedulerCpu_type1_NoPrdMmse_cm()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;

    cuComplex* CMat             = new cuComplex[nUeAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nUeAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsUeAntPrd];  
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];

    uint8_t* S                  = new uint8_t[pCpuDynDesc->nPrbGrp];

    // initialize scheduling solution
    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                continue;

            pCpuDynDesc->allocSol[ueIdx*2] = -1;
            pCpuDynDesc->allocSol[ueIdx*2+1] = -1;
        }
    }

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        std::vector<pfMetric> pf;

        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                pfMetric pfTemp;
                pfTemp.second = ueIdx*static_cast<uint32_t>(pCpuDynDesc->nPrbGrp) + rbgIdx;    

                // if UE is associated with the considered cell
                // compute C matrix
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nUeAnt; rowIdx++) {
                    for (int colIdx = 0; colIdx < pCpuDynDesc->nUeAnt; colIdx++) {
                        if (rowIdx == colIdx) {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = pCpuDynDesc->sigmaSqrd;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        } else {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = 0;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        }
                        
                    }
                }
                uint32_t hTemp = rbgIdx*nUeCellBsUeAntPrd+ ueIdx*nCellBsUeAntPrd;
                for (int lIdx = 0; lIdx < pCpuDynDesc->nCell; lIdx++) {
                    int l = pCpuDynDesc->cellId[lIdx];

                    if (l == cIdx) {
                        continue;
                    }
                    uint32_t hInterfMatStart = hTemp + l*nBsUeAntPrd;
                    matAlg->matMultiplication_aaHplusb(&pCpuDynDesc->estH_fr[hInterfMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);
                }
                // compute inverse of C matrix, C^-1, and store result in CInvMat
                matAlg->matInverse(CMat, pCpuDynDesc->nUeAnt, CInvMat);

                // compute H^H*C^-1
                uint32_t hMatStart = hTemp + cIdx*nBsUeAntPrd;

                matAlg->matMultiplication_aHb(&pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CInvMat, pCpuDynDesc->nUeAnt, DMat);

                // compute H^H*C^-1*H + I
                matAlg->matMultiplication_ab(DMat, pCpuDynDesc->nBsAnt, pCpuDynDesc->nUeAnt, &pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nBsAnt, EMat);
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nBsAnt; rowIdx++) {
                    EMat[rowIdx*pCpuDynDesc->nBsAnt+rowIdx].x += 1.0;
                }

                // compute (H^H*C^-1*H + I)^-1
                matAlg->matInverse(EMat, pCpuDynDesc->nBsAnt, EInvMat);
                
                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = 1.0/EInvMat[j*pCpuDynDesc->nBsAnt+j].x;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp - 1.0;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(sinrTemp)));
                }

                pfTemp.first = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];
                pf.push_back(pfTemp);
            }
        }

        // sort all computed PF metrices across all PRB groups and UEs
        std::sort(pf.begin(), pf.end(), [](pfMetric a, pfMetric b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        std::copy (pf.begin(), pf.end(), pfRecord.begin()+ cellIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI);

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
    }

    delete S;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
}

void multiCellSchedulerCpu::multiCellSchedulerCpu_type1_NoPrdMmse_rm()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;

    cuComplex* CMat             = new cuComplex[nUeAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nUeAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsUeAntPrd];  
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];

    uint8_t* S                  = new uint8_t[pCpuDynDesc->nPrbGrp];

    // initialize scheduling solution
    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                continue;

            pCpuDynDesc->allocSol[ueIdx*2] = -1;
            pCpuDynDesc->allocSol[ueIdx*2+1] = -1;
        }
    }

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        std::vector<pfMetric> pf;

        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                pfMetric pfTemp;
                pfTemp.second = ueIdx*static_cast<uint32_t>(pCpuDynDesc->nPrbGrp) + rbgIdx;    

                // if UE is associated with the considered cell
                // compute C matrix
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nUeAnt; rowIdx++) {
                    for (int colIdx = 0; colIdx < pCpuDynDesc->nUeAnt; colIdx++) {
                        if (rowIdx == colIdx) {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = pCpuDynDesc->sigmaSqrd;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        } else {
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].x = 0;
                            CMat[colIdx*pCpuDynDesc->nUeAnt + rowIdx].y = 0;
                        }
                        
                    }
                }
                uint32_t hTemp = rbgIdx*nUeCellBsUeAntPrd+ ueIdx*nCellBsUeAntPrd;
                for (int lIdx = 0; lIdx < pCpuDynDesc->nCell; lIdx++) {
                    int l = pCpuDynDesc->cellId[lIdx];

                    if (l == cIdx) {
                        continue;
                    }
                    uint32_t hInterfMatStart = hTemp + l*nBsUeAntPrd;
                    matAlg->matMultiplication_aaHplusb_rm(&pCpuDynDesc->estH_fr[hInterfMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CMat);
                }
                // compute inverse of C matrix, C^-1, and store result in CInvMat
                matAlg->matInverse_rm(CMat, pCpuDynDesc->nUeAnt, CInvMat);

                // compute H^H*C^-1
                uint32_t hMatStart = hTemp + cIdx*nBsUeAntPrd;

                matAlg->matMultiplication_aHb_rm(&pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nUeAnt, pCpuDynDesc->nBsAnt, CInvMat, pCpuDynDesc->nUeAnt, DMat);

                // compute H^H*C^-1*H + I
                matAlg->matMultiplication_ab_rm(DMat, pCpuDynDesc->nBsAnt, pCpuDynDesc->nUeAnt, &pCpuDynDesc->estH_fr[hMatStart], pCpuDynDesc->nBsAnt, EMat);
                for (int rowIdx = 0; rowIdx < pCpuDynDesc->nBsAnt; rowIdx++) {
                    EMat[rowIdx*pCpuDynDesc->nBsAnt+rowIdx].x += 1.0;
                }

                // compute (H^H*C^-1*H + I)^-1
                matAlg->matInverse_rm(EMat, pCpuDynDesc->nBsAnt, EInvMat);
                
                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = 1.0/EInvMat[j*pCpuDynDesc->nBsAnt+j].x;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp - 1.0;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(sinrTemp)));
                }

                pfTemp.first = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];
                pf.push_back(pfTemp);
            }
        }

        // sort all computed PF metrices across all PRB groups and UEs
        std::sort(pf.begin(), pf.end(), [](pfMetric a, pfMetric b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        std::copy (pf.begin(), pf.end(), pfRecord.begin()+ cellIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI);

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
    }

    delete S;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
}

//---------------------------- UL functions ----------------------------------
void multiCellSchedulerCpu::multiCellSchedulerCpu_type1_svdPrdMmse_UL()
{
    uint32_t nUeAntSqrd         = pCpuDynDesc->nUeAnt*pCpuDynDesc->nUeAnt;
    uint32_t nBsAntSqrd         = pCpuDynDesc->nBsAnt*pCpuDynDesc->nBsAnt;
    uint32_t nBsUeAntPrd        = pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
    uint32_t nCellBsUeAntPrd    = pCpuDynDesc->totNumCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd  = pCpuDynDesc->nUe*nCellBsUeAntPrd;

    uint8_t* S                  = new uint8_t[pCpuDynDesc->nPrbGrp];

    // initialize scheduling solution
    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
            if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                continue;

            pCpuDynDesc->allocSol[ueIdx*2] = -1;
            pCpuDynDesc->allocSol[ueIdx*2+1] = -1;
        }
    }

    for (int cellIdx = 0; cellIdx < pCpuDynDesc->nCell; cellIdx++) {
        int cIdx = pCpuDynDesc->cellId[cellIdx]; // real cell ID among all cells in the network

        std::vector<pfMetric> pf;

        for (int rbgIdx = 0; rbgIdx < pCpuDynDesc->nPrbGrp; rbgIdx++) {
            for (int ueIdx = 0; ueIdx < pCpuDynDesc->nUe; ueIdx++) {
                if (!pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + ueIdx])
                    continue;

                pfMetric pfTemp;
                pfTemp.second = ueIdx*static_cast<uint32_t>(pCpuDynDesc->nPrbGrp) + rbgIdx;    

                // if UE is associated with the considered cell
                

                // compute data rate
                float dataRate = 0;
                for (int j = 0; j < pCpuDynDesc->nUeAnt; j++) {
                    float sinrTemp = pow(pCpuDynDesc->sinVal[ueIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j], 2.0)/pCpuDynDesc->sigmaSqrd;
                    pCpuDynDesc->postEqSinr[pCpuDynDesc->setSchdUePerCellTTI[ueIdx]*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt + rbgIdx*pCpuDynDesc->nUeAnt + j] = sinrTemp;
                    dataRate += pCpuDynDesc->W*static_cast<float>(log2(static_cast<double>(1.0 + sinrTemp)));
                }

                pfTemp.first = pow(dataRate, pCpuDynDesc->betaCoeff) / pCpuDynDesc->avgRates[ueIdx];
                pf.push_back(pfTemp);
            }
        }

        // sort all computed PF metrices across all PRB groups and UEs
        std::sort(pf.begin(), pf.end(), [](pfMetric a, pfMetric b)
                                  {
                                      return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                                  });

        std::copy (pf.begin(), pf.end(), pfRecord.begin()+ cellIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI);

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
    }

    delete S;
}

void multiCellSchedulerCpu::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                                  cumacSchdSol*               schdSol,
                                  cumacCellGrpPrms*           cellGrpPrms,
                                  cumacSimParam*              simParam,
                                  uint8_t                     in_columnMajor)
{
    pCpuDynDesc->cellId                 = cellGrpPrms->cellId;
    pCpuDynDesc->avgRates               = cellGrpUeStatus->avgRates;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->cellAssoc              = cellGrpPrms->cellAssoc;
    pCpuDynDesc->nUe                    = cellGrpPrms->nUe; // total number of UEs across all coordinated cells
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell; // number of coordinated cells
    pCpuDynDesc->totNumCell             = simParam->totNumCell; // number of all cells in the network. (not needed if channel buffer only contains channels within coordinated cells)
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->W                      = cellGrpPrms->W;
    pCpuDynDesc->numUeSchdPerCellTTI    = cellGrpPrms->numUeSchdPerCellTTI;
    pCpuDynDesc->sigmaSqrd              = cellGrpPrms->sigmaSqrd;
    pCpuDynDesc->betaCoeff              = cellGrpPrms->betaCoeff;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->postEqSinr             = cellGrpPrms->postEqSinr;   
    precodingScheme                     = cellGrpPrms->precodingScheme;
    allocType                           = cellGrpPrms->allocType;
    columnMajor                         = in_columnMajor;

    if (DL == 1) { // DL
        pCpuDynDesc->estH_fr            = cellGrpPrms->estH_fr;
        pCpuDynDesc->prdMat             = cellGrpPrms->prdMat;
        pCpuDynDesc->sinVal             = nullptr;
    } else { // UL
        pCpuDynDesc->estH_fr            = nullptr;
        pCpuDynDesc->prdMat             = nullptr;
        pCpuDynDesc->sinVal             = cellGrpPrms->sinVal;
    }

    pfRecord.reserve(pCpuDynDesc->nCell*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI);
}

void multiCellSchedulerCpu::debugLog()
{
    printf("********************************************\n");
    printf("** CPU multi-cell PF scheduler parameters:\n\n");
    printf("nUe: %d\n", pCpuDynDesc->nUe);
    printf("nCell: %d\n", pCpuDynDesc->nCell);
    printf("totNumCell: %d\n", pCpuDynDesc->totNumCell);
    printf("nPrbGrp: %d\n", pCpuDynDesc->nPrbGrp);
    printf("nBsAnt: %d\n", pCpuDynDesc->nBsAnt);
    printf("nUeAnt: %d\n", pCpuDynDesc->nUeAnt);
    printf("W: %f\n", pCpuDynDesc->W);
    printf("sigmaSqrd: %4.3e\n", pCpuDynDesc->sigmaSqrd);
    printf("allocType: %d\n", allocType);
    printf("precodingScheme: %d\n", precodingScheme);
    printf("betaCoeff: %f\n", pCpuDynDesc->betaCoeff);
    printf("columnMajor: %d\n", columnMajor);

    printf("cellId: ");
    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        printf("%d ", pCpuDynDesc->cellId[cIdx]);
    }
    printf("\n");

    printf("avgRates: ");
    for (int uIdx = 0; uIdx < pCpuDynDesc->nUe; uIdx++) {
        printf("%4.3e ", pCpuDynDesc->avgRates[uIdx]);
    }
    printf("\n");

    int* numAssocUePerCell = new int[pCpuDynDesc->nCell];

    printf("cellAssoc: \n");
    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        numAssocUePerCell[cIdx] = 0;
        printf("cell %d: ", pCpuDynDesc->cellId[cIdx]);
        for (int uIdx = 0; uIdx < pCpuDynDesc->nUe; uIdx++) {
            if (pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + uIdx]) {
                numAssocUePerCell[cIdx]++;
                printf("%d ", uIdx);
            }
        }
        printf("\n");
    }
    printf("\n");

    printf("estH_fr: \n");
    for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
        printf("cell %d, ", pCpuDynDesc->cellId[cIdx]);
        for (int uIdx = 0; uIdx < pCpuDynDesc->nUe; uIdx++) {
            if (pCpuDynDesc->cellAssoc[cIdx*pCpuDynDesc->nUe + uIdx]) {
                printf("UE %d, PRG 0:\n", uIdx);
                if (columnMajor) { // column major
                    for (int rxAnt = 0; rxAnt < pCpuDynDesc->nUeAnt; rxAnt++) {
                        for (int txAnt = 0 ; txAnt < pCpuDynDesc->nBsAnt; txAnt++) {
                            int index = 0;
                            index += uIdx*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += pCpuDynDesc->cellId[cIdx]*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += txAnt*pCpuDynDesc->nUeAnt;
                            index += rxAnt;
                            printf("(%4.3e, %4.3e) ", pCpuDynDesc->estH_fr[index].x, pCpuDynDesc->estH_fr[index].y);
                        }
                        printf("\n");
                    }
                    
                } else { // row major
                    for (int rxAnt = 0; rxAnt < pCpuDynDesc->nUeAnt; rxAnt++) {
                        for (int txAnt = 0 ; txAnt < pCpuDynDesc->nBsAnt; txAnt++) {
                            int index = 0;
                            index += uIdx*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += pCpuDynDesc->cellId[cIdx]*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += rxAnt*pCpuDynDesc->nBsAnt;
                            index += txAnt;
                            printf("(%4.3e, %4.3e) ", pCpuDynDesc->estH_fr[index].x, pCpuDynDesc->estH_fr[index].y);
                        }
                        printf("\n");
                    }
                }
/*
                printf("UE %d, PRG %d:\n", uIdx, pCpuDynDesc->nPrbGrp-1);
                if (columnMajor) { // column major
                    for (int rxAnt = 0; rxAnt < pCpuDynDesc->nUeAnt; rxAnt++) {
                        for (int txAnt = 0 ; txAnt < pCpuDynDesc->nBsAnt; txAnt++) {
                            int index = (pCpuDynDesc->nPrbGrp-1)*pCpuDynDesc->nUe*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += uIdx*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += pCpuDynDesc->cellId[cIdx]*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += txAnt*pCpuDynDesc->nUeAnt;
                            index += rxAnt;
                            printf("(%4.3e, %4.3e) ", pCpuDynDesc->estH_fr[index].x, pCpuDynDesc->estH_fr[index].y);
                        }
                        printf("\n");
                    }
                    
                } else { // row major
                    for (int rxAnt = 0; rxAnt < pCpuDynDesc->nUeAnt; rxAnt++) {
                        for (int txAnt = 0 ; txAnt < pCpuDynDesc->nBsAnt; txAnt++) {
                            int index = (pCpuDynDesc->nPrbGrp-1)*pCpuDynDesc->nUe*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += uIdx*pCpuDynDesc->totNumCell*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += pCpuDynDesc->cellId[cIdx]*pCpuDynDesc->nBsAnt*pCpuDynDesc->nUeAnt;
                            index += rxAnt*pCpuDynDesc->nBsAnt;
                            index += txAnt;
                            printf("(%4.3e, %4.3e) ", pCpuDynDesc->estH_fr[index].x, pCpuDynDesc->estH_fr[index].y);
                        }
                        printf("\n");
                    }
                }
*/
                break;
            }
        }
    }
/*
    if (allocType) {
        printf("Sorted PF metrics: \n");
        for (int cIdx = 0; cIdx < pCpuDynDesc->nCell; cIdx++) {
            uint32_t pfStartCell = cIdx*pCpuDynDesc->nPrbGrp*pCpuDynDesc->numUeSchdPerCellTTI;
            printf("cell %d: ", cIdx);
            for (int idx = 0; idx < pCpuDynDesc->nPrbGrp*numAssocUePerCell[cIdx]; idx++) {
                printf("(%4.3e, %d) ", pfRecord[pfStartCell+idx].first, pfRecord[pfStartCell+idx].second);
            }
            printf("\n");
        }
        printf("\n");
    }
*/
    printf("** End of logging \n");
    printf("********************************************\n");
}

void multiCellSchedulerCpu::run()
{
#ifdef CPU_SCHEDULER_TIME_MEASURE_
    auto start = std::chrono::steady_clock::now();
    for (int idx = 0; idx < numRunSchKnlTimeMsr; idx++) {
#endif
    if (DL == 1) { // for DL
        switch (precodingScheme) {
            case 0: // no precoding
                if (allocType) {
                    if (columnMajor) {
                        multiCellSchedulerCpu_type1_NoPrdMmse_cm();
                    } else {
                        multiCellSchedulerCpu_type1_NoPrdMmse_rm();
                    }
                } else {
                    if (columnMajor) {
                        multiCellSchedulerCpu_noPrdMmse();
                    } else {
                        printf("Error: DL CPU multi-cell scheduler - row-major channel matrix access for type-0 allocation not supported\n");
                        return;
                    }
                }
                break;
            case 1: // SVD precoding
                if (allocType) {
                    if (columnMajor) {
                        multiCellSchedulerCpu_type1_svdPrdMmse_cm();
                    } else {
                        printf("Error: DL CPU multi-cell scheduler - row-major channel matrix access for type-1 allocation with SVD precoder not supported\n");
                        return;
                    }
                } else {
                    if (columnMajor) {
                        multiCellSchedulerCpu_svdMmse();
                    } else {
                        printf("Error: DL CPU multi-cell scheduler - row-major channel matrix access for type-0 allocation not supported\n");
                        return;
                    }
                }
                break;
            default: // default no precoding with allocate type 0 (non-consecutive)
                multiCellSchedulerCpu_noPrdMmse();
                break;
        }
    } else {  // for UL
        if (precodingScheme == 0) {
            printf("Error: For UL, CPU multi-cell scheduler is only supported for SVD precoding.\n");
            return;
        }

        if (allocType == 0) {
            printf("Error: For UL, CPU multi-cell scheduler is only supported for type-1 PRB allocation.\n");
            return;
        }

        multiCellSchedulerCpu_type1_svdPrdMmse_UL();
    }
#ifdef CPU_SCHEDULER_TIME_MEASURE_
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("CPU multi-cell PF scheduler elapsed time: %f ms\n", 1000.0*elapsed_seconds.count()/static_cast<float>(numRunSchKnlTimeMsr));
#endif
}
}