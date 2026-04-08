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
constexpr uint16_t numRunSchKnlTimeMsr = 1000;
#endif

static __device__ __constant__ float L1T1B024GPU[28] = {-4.57, -3.02, -0.6, 1.26, 2.87, 4.89, 5.9, 6.67, 7.72, 8.41, 9.36, 10.56, 11.65, 12.64, 13.23, 14.25, 15.23, 16.14, 16.72, 17.79, 19.19, 19.65, 20.95, 21.75, 22.6, 23.91, 24.38, 25.99};
static __device__ __constant__ float L1T1B050PRGS01_GTC25GPU[28] = {-4.3, -2.44, -0.46, 1.24, 2.94, 4.89, 5.72, 6.84, 7.67, 8.67, 9.23, 10.48, 11.49, 12.37, 13.42, 14.43, 15.04, 16.0, 16.98, 18.03, 19.24, 19.88, 20.74, 21.71, 22.94, 23.5, 24.98, 26.08};
static __device__ __constant__ float minSinrCqiGPU[15] = {-5.7456, -2.7627, 1.2773, 5.1605, 6.9945, 8.6601, 10.9361, 12.6731, 14.3437, 16.3933, 18.0984, 19.8376, 22.2465, 24.0897, 26.2837};

mcsSelectionLUT::mcsSelectionLUT(cumacCellGrpPrms*   cellGrpPrms, 
                                        cudaStream_t        strm)
{
    m_nUe       = cellGrpPrms->nActiveUe;
    enableHarq  = cellGrpPrms->harqEnabledInd;
    allocType   = cellGrpPrms->allocType;
    CQI         = cellGrpPrms->mcsSelCqi;

    if (allocType == 0 && cellGrpPrms->nBsAnt != 4) {
        throw std::runtime_error("Error: LUT-based MCS selection for type-0 PRB allocation is only supported for 4TR SU-MIMO");
    }

    pCpuDynDesc = std::make_unique<mcsSelDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcsSelDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    CUDA_CHECK_ERR(cudaMalloc((void **)&ollaParamArr, sizeof(ollaParam)*m_nUe));

    std::vector<float> blerTargetActUe(m_nUe);
    CUDA_CHECK_ERR(cudaMemcpy(blerTargetActUe.data(), cellGrpPrms->blerTargetActUe, sizeof(float)*m_nUe, cudaMemcpyDeviceToHost));

    ollaParamArrCpuInit = std::vector<ollaParam>(m_nUe);
    for (int ueIdx = 0; ueIdx < m_nUe; ueIdx++) {
        ollaParamArrCpuInit[ueIdx].delta_ini   = 0.0;
        ollaParamArrCpuInit[ueIdx].delta       = 0.0;
        ollaParamArrCpuInit[ueIdx].delta_up    = 1.0;
        ollaParamArrCpuInit[ueIdx].delta_down  = 1.0*blerTargetActUe[ueIdx];
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync(ollaParamArr, ollaParamArrCpuInit.data(), sizeof(ollaParam)*m_nUe, cudaMemcpyHostToDevice, strm));
}

mcsSelectionLUT::~mcsSelectionLUT()
{
    CUDA_CHECK_ERR(cudaFree(ollaParamArr));
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
}

// ***************** Type-0 PRB allocation *****************
static __global__ void mcsSelSinrRepKernel_type0_cqi(mcsSelDynDescr_t* pDynDescr)
{
    uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
    uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
    uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
    uint16_t globalUidx = 0xFFFF;
    
    __shared__ int numAllocPrg[1024];
    __shared__ int16_t allocSol[maxNumCoorCells_*maxNumPrgPerCell_];

    numAllocPrg[threadIdx.x] = 0;
    for (int idx = threadIdx.x; idx < pDynDescr->nCell*pDynDescr->nPrbGrp; idx += blockDim.x) {
        allocSol[idx] = pDynDescr->allocSol[idx];
    }
    __syncthreads();

    if (uIdx < pDynDescr->nUe) {
        globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
    }

    if (globalUidx < 0xFFFF) {
        for (int cIdx = 0; cIdx < pDynDescr->nCell; cIdx++) {
            if (allocSol[prgIdx*pDynDescr->nCell + cIdx] == uIdx) {
                int temp = atomicAdd(&(numAllocPrg[blockLocalUidx]), 1);
            }
        }
    }
    __syncthreads();

    int8_t   tbErrLast = -1;

    // load per-UE, per-PRG, per-layer post-eq SINRs into shared memory
    if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
        if (numAllocPrg[blockLocalUidx] > 0) { // being scheduled for the current slot
            // update OLLA parameters
            if (prgIdx == 0) {
                tbErrLast = pDynDescr->nBsAnt == 4 ? pDynDescr->tbErrLast[uIdx] : pDynDescr->tbErrLast[globalUidx];

                if (tbErrLast == 1) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta += pDynDescr->ollaParamArrGPU[globalUidx].delta_up;
                } else if (tbErrLast == 0) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta -= pDynDescr->ollaParamArrGPU[globalUidx].delta_down;
                }
            }
        }
    }
    __syncthreads(); 

    if (numAllocPrg[blockLocalUidx] > 0 && prgIdx == 0) { // using threads [0, totNumCell - 1], where threadIdx.x = cellIdx
        if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
            int8_t cqiVal = pDynDescr->cqiActUe[globalUidx];

            if (cqiVal < 0) {
                if (pDynDescr->nBsAnt == 4) { // 4TR
                    pDynDescr->mcsSelSol[uIdx] = 0;
                } else { // 64TR
                    pDynDescr->mcsSelSol[globalUidx] = 0;
                }
            } else {
                float avgSinrDB;
                if (cqiVal == 0) {
                    avgSinrDB = minSinrCqiGPU[0];
                } else if (cqiVal == 15) {
                    avgSinrDB = minSinrCqiGPU[14];
                } else {
                    avgSinrDB = (minSinrCqiGPU[cqiVal-1] + minSinrCqiGPU[cqiVal])/2.0;
                }
                avgSinrDB -= pDynDescr->ollaParamArrGPU[globalUidx].delta;
            
                uint8_t selectedMcs = 0;
                for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                    if (avgSinrDB >= L1T1B024GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                }

                if (pDynDescr->nBsAnt == 4) { // 4TR
                    pDynDescr->mcsSelSol[uIdx] = selectedMcs;   
                } else {
                    pDynDescr->mcsSelSol[globalUidx] = selectedMcs;
                }
            }
        } else { // the UE is scheduled for re-transmission
            if (pDynDescr->nBsAnt == 4) { // 4TR
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            } else {
                pDynDescr->mcsSelSol[globalUidx] = pDynDescr->mcsSelSolLastTx[globalUidx];
            }
        }
    } else if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF && prgIdx == 0) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            pDynDescr->mcsSelSol[uIdx] = -1;
        } else {
            pDynDescr->mcsSelSol[globalUidx] = -1;
        }
    }
}

static __global__ void mcsSelSinrRepKernel_type0_wbSinr(mcsSelDynDescr_t* pDynDescr)
{
    uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
    uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
    uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
    uint16_t globalUidx = 0xFFFF;

    __shared__ int numAllocPrg[1024];
    __shared__ int16_t allocSol[maxNumCoorCells_*maxNumPrgPerCell_];

    numAllocPrg[threadIdx.x] = 0;
    for (int idx = threadIdx.x; idx < pDynDescr->nCell*pDynDescr->nPrbGrp; idx += blockDim.x) {
        allocSol[idx] = pDynDescr->allocSol[idx];
    }
    __syncthreads();

    if (uIdx < pDynDescr->nUe) {
        globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
    }

    if (globalUidx < 0xFFFF) {
        for (int cIdx = 0; cIdx < pDynDescr->nCell; cIdx++) {
            if (allocSol[prgIdx*pDynDescr->nCell + cIdx] == uIdx) {
                int temp = atomicAdd(&(numAllocPrg[blockLocalUidx]), 1);
            }
        }
    }
    __syncthreads();

    int8_t   tbErrLast = -1;

    // load per-UE, per-PRG, per-layer post-eq SINRs into shared memory
    if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
        if (numAllocPrg[blockLocalUidx] > 0) { // being scheduled for the current slot
            // update OLLA parameters
            if (prgIdx == 0) {
                tbErrLast = pDynDescr->nBsAnt == 4 ? pDynDescr->tbErrLast[uIdx] : pDynDescr->tbErrLast[globalUidx];
                if (tbErrLast == 1) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta += pDynDescr->ollaParamArrGPU[globalUidx].delta_up;
                } else if (tbErrLast == 0) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta -= pDynDescr->ollaParamArrGPU[globalUidx].delta_down;
                }
            }
        }
    }
    __syncthreads(); 

    if (numAllocPrg[blockLocalUidx] > 0 && prgIdx == 0) { 
        if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
            double dAvgSinrDB = 10.0*log10(static_cast<double>(pDynDescr->wbSinr[globalUidx*pDynDescr->nUeAnt]));
            float avgSinrDB = static_cast<float>(dAvgSinrDB);

            avgSinrDB = avgSinrDB > pDynDescr->mcsSelSinrCapThr ? pDynDescr->mcsSelSinrCapThr : avgSinrDB;

            if (pDynDescr->beamformGainLastTx != nullptr && pDynDescr->beamformGainCurrTx != nullptr) {
                if (pDynDescr->beamformGainLastTx[globalUidx] != -100.0 && pDynDescr->beamformGainCurrTx[globalUidx] != -100.0) {
                    avgSinrDB += pDynDescr->beamformGainCurrTx[globalUidx] - pDynDescr->beamformGainLastTx[globalUidx];
                }
                pDynDescr->beamformGainLastTx[globalUidx] = pDynDescr->beamformGainCurrTx[globalUidx];
            }
            
            avgSinrDB -= pDynDescr->ollaParamArrGPU[globalUidx].delta;

            uint8_t selectedMcs = 0;
            for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                if (pDynDescr->mcsSelLutType == 0) {
                    if (avgSinrDB >= L1T1B024GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                } else if (pDynDescr->mcsSelLutType == 1) {
                    if (avgSinrDB >= L1T1B050PRGS01_GTC25GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                }
            }

            if (pDynDescr->nBsAnt == 4) { // 4TR SU-MIMO
                pDynDescr->mcsSelSol[uIdx] = selectedMcs;
            } else { // 64TR MU-MIMO
                pDynDescr->mcsSelSol[globalUidx] = selectedMcs;
            }
        } else { // the UE is scheduled for re-transmission
            if (pDynDescr->beamformGainLastTx != nullptr && pDynDescr->beamformGainCurrTx != nullptr) {
                pDynDescr->beamformGainLastTx[globalUidx] = pDynDescr->beamformGainCurrTx[globalUidx];
            }
                
            if (pDynDescr->nBsAnt == 4) { // 4TR
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            } else {
                pDynDescr->mcsSelSol[globalUidx] = pDynDescr->mcsSelSolLastTx[globalUidx];
            }
        }
    } else if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF && prgIdx == 0) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            pDynDescr->mcsSelSol[uIdx] = -1;
        } else { // 64TR
            pDynDescr->mcsSelSol[globalUidx] = -1;
        }
    }    
}


// ***************** Type-1 PRB allocation *****************
static __global__ void mcsSelSinrRepKernel_type1_cqi(mcsSelDynDescr_t* pDynDescr)
{
    uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
    uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
    uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
    uint16_t globalUidx = 0xFFFF;
    if (uIdx < pDynDescr->nUe) {
        globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
    }
    uint16_t numAllocPrg = 0; // number of allocated PRGs to the considered UE

    int8_t   tbErrLast = -1;

    // load per-UE, per-PRG, per-layer post-eq SINRs into shared memory
    if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
        } else {
            numAllocPrg = pDynDescr->allocSol[2*globalUidx+1] - pDynDescr->allocSol[2*globalUidx];
        }

        if (numAllocPrg > 0) { // being scheduled for the current slot
            // update OLLA parameters
            if (prgIdx == 0) {
                tbErrLast = pDynDescr->nBsAnt == 4 ? pDynDescr->tbErrLast[uIdx] : pDynDescr->tbErrLast[globalUidx];

                if (tbErrLast == 1) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta += pDynDescr->ollaParamArrGPU[globalUidx].delta_up;
                } else if (tbErrLast == 0) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta -= pDynDescr->ollaParamArrGPU[globalUidx].delta_down;
                }
            }
        }
    }
    __syncthreads(); 

    if (numAllocPrg > 0 && prgIdx == 0) { // using threads [0, totNumCell - 1], where threadIdx.x = cellIdx
        if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
            int8_t cqiVal = pDynDescr->cqiActUe[globalUidx];

            if (cqiVal < 0) {
                if (pDynDescr->nBsAnt == 4) { // 4TR
                    pDynDescr->mcsSelSol[uIdx] = 0;
                } else { // 64TR
                    pDynDescr->mcsSelSol[globalUidx] = 0;
                }
            } else {
                float avgSinrDB;
                if (cqiVal == 0) {
                    avgSinrDB = minSinrCqiGPU[0];
                } else if (cqiVal == 15) {
                    avgSinrDB = minSinrCqiGPU[14];
                } else {
                    avgSinrDB = (minSinrCqiGPU[cqiVal-1] + minSinrCqiGPU[cqiVal])/2.0;
                }
                avgSinrDB -= pDynDescr->ollaParamArrGPU[globalUidx].delta;
            
                uint8_t selectedMcs = 0;
                for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                    if (avgSinrDB >= L1T1B024GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                }

                if (pDynDescr->nBsAnt == 4) { // 4TR
                    pDynDescr->mcsSelSol[uIdx] = selectedMcs;   
                } else {
                    pDynDescr->mcsSelSol[globalUidx] = selectedMcs;
                }
            }
        } else { // the UE is scheduled for re-transmission
            if (pDynDescr->nBsAnt == 4) { // 4TR
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            } else {
                pDynDescr->mcsSelSol[globalUidx] = pDynDescr->mcsSelSolLastTx[globalUidx];
            }
        }
    } else if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF && prgIdx == 0) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            pDynDescr->mcsSelSol[uIdx] = -1;
        } else {
            pDynDescr->mcsSelSol[globalUidx] = -1;
        }
    }
}

static __global__ void mcsSelSinrRepKernel_type1_wbSinr(mcsSelDynDescr_t* pDynDescr)
{
    uint16_t blockLocalUidx = floor(static_cast<float>(threadIdx.x)/pDynDescr->nPrbGrp);
    uint16_t uIdx = blockIdx.x*pDynDescr->nUePerBlk + blockLocalUidx;
    uint16_t prgIdx = threadIdx.x - blockLocalUidx*pDynDescr->nPrbGrp;
    uint16_t globalUidx = 0xFFFF;

    if (uIdx < pDynDescr->nUe) {
        globalUidx = pDynDescr->setSchdUePerCellTTI[uIdx];
    }
    uint16_t numAllocPrg = 0; // number of allocated PRGs to the considered UE

    int8_t   tbErrLast = -1;

    // load per-UE, per-PRG, per-layer post-eq SINRs into shared memory
    if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            numAllocPrg = pDynDescr->allocSol[2*uIdx+1] - pDynDescr->allocSol[2*uIdx];
        } else {
            numAllocPrg = pDynDescr->allocSol[2*globalUidx+1] - pDynDescr->allocSol[2*globalUidx];
        }

        if (numAllocPrg > 0) { // being scheduled for the current slot
            // update OLLA parameters
            if (prgIdx == 0) {
                tbErrLast = pDynDescr->nBsAnt == 4 ? pDynDescr->tbErrLast[uIdx] : pDynDescr->tbErrLast[globalUidx];
                if (tbErrLast == 1) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta += pDynDescr->ollaParamArrGPU[globalUidx].delta_up;
                } else if (tbErrLast == 0) {
                    pDynDescr->ollaParamArrGPU[globalUidx].delta -= pDynDescr->ollaParamArrGPU[globalUidx].delta_down;
                }
            }
        }
    }
    __syncthreads(); 

    if (numAllocPrg > 0 && prgIdx == 0) { 
        if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[globalUidx] == 1) { // the UE is scheduled for new transmission
            double dAvgSinrDB = 10.0*log10(static_cast<double>(pDynDescr->wbSinr[globalUidx*pDynDescr->nUeAnt]));
            float avgSinrDB = static_cast<float>(dAvgSinrDB);

            avgSinrDB = avgSinrDB > pDynDescr->mcsSelSinrCapThr ? pDynDescr->mcsSelSinrCapThr : avgSinrDB;

            if (pDynDescr->beamformGainLastTx != nullptr && pDynDescr->beamformGainCurrTx != nullptr) {
                if (pDynDescr->beamformGainLastTx[globalUidx] != -100.0 && pDynDescr->beamformGainCurrTx[globalUidx] != -100.0) {
                    avgSinrDB += pDynDescr->beamformGainCurrTx[globalUidx] - pDynDescr->beamformGainLastTx[globalUidx];
                }
                pDynDescr->beamformGainLastTx[globalUidx] = pDynDescr->beamformGainCurrTx[globalUidx];
            }
            
            avgSinrDB -= pDynDescr->ollaParamArrGPU[globalUidx].delta;

            uint8_t selectedMcs = 0;
            for (int mcsIdx = 27; mcsIdx >= 0; mcsIdx--) {
                if (pDynDescr->mcsSelLutType == 0) {
                    if (avgSinrDB >= L1T1B024GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                } else if (pDynDescr->mcsSelLutType == 1) {
                    if (avgSinrDB >= L1T1B050PRGS01_GTC25GPU[mcsIdx]) {
                        selectedMcs = mcsIdx;
                        break;
                    }
                }
            }

            if (pDynDescr->nBsAnt == 4) { // 4TR SU-MIMO
                pDynDescr->mcsSelSol[uIdx] = selectedMcs;
            } else { // 64TR MU-MIMO
                pDynDescr->mcsSelSol[globalUidx] = selectedMcs;
            }
        } else { // the UE is scheduled for re-transmission
            if (pDynDescr->beamformGainLastTx != nullptr && pDynDescr->beamformGainCurrTx != nullptr) {
                pDynDescr->beamformGainLastTx[globalUidx] = pDynDescr->beamformGainCurrTx[globalUidx];
            }
                
            if (pDynDescr->nBsAnt == 4) { // 4TR
                pDynDescr->mcsSelSol[uIdx] = pDynDescr->mcsSelSolLastTx[uIdx];
            } else {
                pDynDescr->mcsSelSol[globalUidx] = pDynDescr->mcsSelSolLastTx[globalUidx];
            }
        }
    } else if (uIdx < pDynDescr->nUe && globalUidx < 0xFFFF && prgIdx == 0) {
        if (pDynDescr->nBsAnt == 4) { // 4TR
            pDynDescr->mcsSelSol[uIdx] = -1;
        } else { // 64TR
            pDynDescr->mcsSelSol[globalUidx] = -1;
        }
    }
}

void mcsSelectionLUT::kernelSelect()
{
    void* kernelFunc;
    if (allocType == 0) {
        if (CQI == 1) { // UE reported wideband CQI based
            kernelFunc = reinterpret_cast<void*>(mcsSelSinrRepKernel_type0_cqi);
        } else { // wideband post-eq SINR based
            kernelFunc = reinterpret_cast<void*>(mcsSelSinrRepKernel_type0_wbSinr);
        }
    } else if (allocType == 1) {
        if (CQI == 1) { // UE reported wideband CQI based
            kernelFunc = reinterpret_cast<void*>(mcsSelSinrRepKernel_type1_cqi);
        } else { // wideband post-eq SINR based
            kernelFunc = reinterpret_cast<void*>(mcsSelSinrRepKernel_type1_wbSinr);
        }
    } else {
        throw std::runtime_error("Error: invalid PRB allocation type");
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

void mcsSelectionLUT::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                            cumacSchdSol*               schdSol,
                            cumacCellGrpPrms*           cellGrpPrms,
                            cudaStream_t                strm)
{
    pCpuDynDesc->nUe = cellGrpPrms->nBsAnt == 4 ? cellGrpPrms->nUe : (cellGrpPrms->nCell*cellGrpPrms->numUeForGrpPerCell);
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->mcsSelSinrCapThr       = cellGrpPrms->mcsSelSinrCapThr;
    pCpuDynDesc->mcsSelLutType          = cellGrpPrms->mcsSelLutType;
    pCpuDynDesc->ollaParamArrGPU        = ollaParamArr;
    pCpuDynDesc->wbSinr                 = cellGrpPrms->wbSinr;
    pCpuDynDesc->beamformGainLastTx     = cellGrpUeStatus->beamformGainLastTx;  
    pCpuDynDesc->beamformGainCurrTx     = cellGrpUeStatus->beamformGainCurrTx;
    pCpuDynDesc->tbErrLast              = cellGrpUeStatus->tbErrLast;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->mcsSelSol              = schdSol->mcsSelSol;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    
    if (CQI == 1) { // CQI based
        pCpuDynDesc->cqiActUe           = cellGrpUeStatus->cqiActUe;
    } else {
        pCpuDynDesc->cqiActUe           = nullptr;
    }

    if (enableHarq == 1) { // HARQ enabled
        pCpuDynDesc->newDataActUe           = cellGrpUeStatus->newDataActUe;
        pCpuDynDesc->mcsSelSolLastTx        = cellGrpUeStatus->mcsSelSolLastTx;
    } else {
        pCpuDynDesc->newDataActUe           = nullptr;
        pCpuDynDesc->mcsSelSolLastTx        = nullptr;
    }

    uint16_t numUePerBlk = floor(1024.0/cellGrpPrms->nPrbGrp);
    pCpuDynDesc->nUePerBlk = numUePerBlk;

    // launch geometry
    blockDim = {static_cast<uint16_t>(numUePerBlk*cellGrpPrms->nPrbGrp), 1, 1};
    gridDim  = {static_cast<uint16_t>(ceil(static_cast<float>(pCpuDynDesc->nUe)/numUePerBlk)), 1, 1};
    
    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcsSelDynDescr_t), cudaMemcpyHostToDevice, strm));
    
    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect();

    pLaunchCfg->kernelArgs[0] = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void mcsSelectionLUT::run(cudaStream_t strm)
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
    printf("Multi-cell MCS selection ext time = %f microseconds\n", milliseconds*1000.0/static_cast<float>(numRunSchKnlTimeMsr));
#endif  
}

void mcsSelectionLUT::debugLog()
{
    std::vector<ollaParam> ollaParamArrCpu(m_nUe);
    CUDA_CHECK_ERR(cudaMemcpy(ollaParamArrCpu.data(), pCpuDynDesc->ollaParamArrGPU, m_nUe*sizeof(ollaParam), cudaMemcpyDeviceToHost));

    printf("GPU delta: ");
    for (int ueIdx = 0; ueIdx < m_nUe; ueIdx++) {
        printf("%f ", ollaParamArrCpu[ueIdx].delta);
    }
    printf("\n");
}
}