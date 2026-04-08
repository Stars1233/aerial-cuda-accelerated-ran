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

#define dir 0

namespace cg = cooperative_groups;

singleCellScheduler::singleCellScheduler()
{
    // allocate memory for dynamic descriptors
    pCpuDynDesc = new scDynDescr_t;
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(scDynDescr_t)));

    pLaunchCfg = new launchCfg_t;
}

singleCellScheduler::~singleCellScheduler()
{
    delete pCpuDynDesc;
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));

    delete pLaunchCfg;
}

inline __device__ void bitonicSort(float* valueArr, uint16_t* idArr, uint16_t n)
  {
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              float t;
              int t_id;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == d) {
                  t = valueArr[pos];
                  valueArr[pos] = valueArr[pos + stride];
                  valueArr[pos + stride] = t;
                  t_id = idArr[pos];
                  idArr[pos] = idArr[pos + stride];
                  idArr[pos + stride] = t_id;
              }
           }
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           float t;
           int t_id;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == dir) {
               t = valueArr[pos];
               valueArr[pos] = valueArr[pos + stride];
               valueArr[pos + stride] = t;
             
               t_id = idArr[pos];
               idArr[pos] = idArr[pos + stride];
               idArr[pos + stride] = t_id;
           }
        }
    }

    __syncthreads(); 
 }

static __global__ void singleCellSchedulerKernel_noPrdMmse(scDynDescr_t* pDynDescr) {
    uint16_t nBsAntSqrd             = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t nBsUeAntPrd            = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint16_t assocUeIdxInBlk        = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk    = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    __shared__ cuComplex  CMat[1024];
    __shared__ cuComplex  CInvMat[1024];
    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    __shared__ uint16_t   CMatIdxTemp[1024];
    __shared__ uint16_t   CMatIdx[1024];
    __shared__ int16_t    ueIdxArr[1024];
    __shared__ float      pfMetric[1024];

    colIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
    rowIdx[threadIdx.x] = eIdx - colIdx[threadIdx.x]*pDynDescr->nBsAnt;
  
    CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
    CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;
  
    ueIdxArr[threadIdx.x] = -1;
    pfMetric[threadIdx.x] = 0;

    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    __shared__ int nAssocUeFound;
    bool cnt = true;

    while (cnt) {
        if (threadIdx.x == 0)
            nAssocUeFound = 0;
        __syncthreads();

        uIdx = -1;

        int16_t assocUeRank = -1;
        for (int j = 0; j<pDynDescr->nUe; j++) {
            if (pDynDescr->cellAssoc[pDynDescr->cellId*pDynDescr->nUe + j]) {
                assocUeRank += 1;
                if (assocUeRank == realAssocUeIdxInBlk) {
                    uIdx = j;
                    if (eIdx == 0) {
                        atomicAdd(&nAssocUeFound, 1);
                    }
                    break;
                }
            }
        }
        __syncthreads();

        if (nAssocUeFound < pDynDescr->nMaxSchdUePerRnd) {
            cnt = false;
        }

        // compute H^H*H + sigmaSqrd*I
        // use CMat for storing H^H*H + sigmaSqrd*I
        if (uIdx >= 0) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_perUeBuffer[uIdx][blockIdx.x*nBsUeAntPrd+rowIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
                cuComplex tmp2 = pDynDescr->estH_fr_perUeBuffer[uIdx][blockIdx.x*nBsUeAntPrd+colIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
                CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }

            if (colIdx[threadIdx.x] == rowIdx[threadIdx.x]) {
                CMat[CMatIdx[threadIdx.x]].x += pDynDescr->sigmaSqrd;
                CInvMat[CMatIdx[threadIdx.x]].x = 1.0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            } else {
                CInvMat[CMatIdx[threadIdx.x]].x = 0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute (H^H*H + sigmaSqrd*I)^-1
        // use CInvMat for storing (H^H*H + sigmaSqrd*I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]+col_i*pDynDescr->nBsAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    CInvMat[CMatIdx[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        // evaluate PF metric for each UE
        if (uIdx >= 0 && eIdx == 0) {
            // compute data rate
            float dataRate = 0;
            for (int j = 0; j < pDynDescr->nUeAnt; j++) {
                dataRate += pDynDescr->W*log2f(1.0/pDynDescr->sigmaSqrd/CInvMat[CMatIdxTemp[threadIdx.x]+j*pDynDescr->nBsAnt+j].x);
            }
            pfMetric[realAssocUeIdxInBlk] = dataRate/pDynDescr->avgRates[uIdx];
            ueIdxArr[realAssocUeIdxInBlk] = uIdx;
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxSchdUePerRnd;
        }
        __syncthreads(); 
    }

    // select UE
    if (threadIdx.x == 0) {
        float   maxv = 0;
        int16_t maxi = -1;
        for (int j = 0; j<pDynDescr->nUe; j++) {
            if (ueIdxArr[j] < 0)
                break;
            if (pfMetric[j] > maxv) {
                maxv = pfMetric[j];
                maxi = ueIdxArr[j];
            }
        }

        pDynDescr->allocSol[blockIdx.x*pDynDescr->nCell + pDynDescr->cellId] = maxi;
        // printf("cellId = %d, pDynDescr->allocSol = %d\n", pDynDescr->cellId, pDynDescr->allocSol[blockIdx.x*pDynDescr->nCell + pDynDescr->cellId]);
    }
}

static __global__ void singleCellSchedulerKernel_svdMmse(scDynDescr_t* pDynDescr) {
    uint16_t nBsAntSqrd             = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t nBsUeAntPrd            = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint32_t nCellBsAntSqrd         = pDynDescr->nCell*nBsAntSqrd;
    uint16_t assocUeIdxInBlk        = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk    = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    __shared__ cuComplex  CMat[1024];
    __shared__ cuComplex  DMat[1024];
    __shared__ cuComplex  CInvMat[1024];
    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    __shared__ uint8_t    colIdx2[1024];
    __shared__ uint8_t    rowIdx2[1024];
    __shared__ uint16_t   CMatIdxTemp[1024];
    __shared__ uint16_t   CMatIdx[1024];
    __shared__ uint16_t   CMatIdxTemp2[1024];
    __shared__ uint16_t   CMatIdx2[1024];
    __shared__ int16_t    ueIdxArr[1024];
    __shared__ float      pfMetric[1024];

    colIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
    rowIdx[threadIdx.x] = eIdx - colIdx[threadIdx.x]*pDynDescr->nBsAnt;

    colIdx2[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nUeAnt);
    rowIdx2[threadIdx.x] = eIdx - colIdx2[threadIdx.x]*pDynDescr->nUeAnt;
  
    CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
    CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;

    CMatIdxTemp2[threadIdx.x] = assocUeIdxInBlk*nBsUeAntPrd;
    CMatIdx2[threadIdx.x] = CMatIdxTemp2[threadIdx.x] + eIdx;
  
    ueIdxArr[threadIdx.x] = -1;
    pfMetric[threadIdx.x] = 0;

    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    __shared__ int nAssocUeFound;
    bool cnt = true;

    while (cnt) {
        if (threadIdx.x == 0)
            nAssocUeFound = 0;
        __syncthreads();

        uIdx = -1;

        int16_t assocUeRank = -1;
        for (int j = 0; j<pDynDescr->nUe; j++) {
            if (pDynDescr->cellAssoc[pDynDescr->cellId*pDynDescr->nUe + j]) {
                assocUeRank += 1;
                if (assocUeRank == realAssocUeIdxInBlk) {
                    uIdx = j;
                    if (eIdx == 0) {
                        atomicAdd(&nAssocUeFound, 1);
                    }
                    break;
                }
            }
        }
        __syncthreads();

        if (nAssocUeFound < pDynDescr->nMaxSchdUePerRnd) {
            cnt = false;
        }

        // compute H*V
        // use DMat for storing H*V
        uint32_t vMatStart = blockIdx.x*pDynDescr->nUe*nCellBsAntSqrd + uIdx*nCellBsAntSqrd + pDynDescr->cellId*nBsAntSqrd;

        if (uIdx >= 0 && eIdx < nBsUeAntPrd) {
            DMat[CMatIdx2[threadIdx.x]].x = 0;
            DMat[CMatIdx2[threadIdx.x]].y = 0;

            for (int j = 0; j<pDynDescr->nBsAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_perUeBuffer[uIdx][blockIdx.x*nBsUeAntPrd+j*pDynDescr->nUeAnt+rowIdx2[threadIdx.x]];
                cuComplex tmp2 = pDynDescr->prdMat[vMatStart + colIdx2[threadIdx.x]*pDynDescr->nBsAnt +j];
#ifdef SCSCHEDULER_DEBUG_
            //printf("tmp1.x = %f, tmp1.y = %f; tmp2.x = %f, tmp2.y = %f\n", tmp1.x, tmp1.y, tmp2.x, tmp2.y);
#endif
                DMat[CMatIdx2[threadIdx.x]].x += tmp1.x*tmp2.x - tmp1.y*tmp2.y;
                DMat[CMatIdx2[threadIdx.x]].y += tmp1.x*tmp2.y + tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        // compute (H*V)^H*(H*V) + sigmaSqrd*I
        // use CMat for storing (H*V)^H*(H*V) + sigmaSqrd*I
        if (uIdx >= 0) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = DMat[CMatIdxTemp2[threadIdx.x]+rowIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
                cuComplex tmp2 = DMat[CMatIdxTemp2[threadIdx.x]+colIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
#ifdef SCSCHEDULER_DEBUG_
            // printf("tmp1.x = %f, tmp1.y = %f; tmp2.x = %f, tmp2.y = %f\n", tmp1.x, tmp1.y, tmp2.x, tmp2.y);
#endif            
                CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }

            if (colIdx[threadIdx.x] == rowIdx[threadIdx.x]) {
                CMat[CMatIdx[threadIdx.x]].x += pDynDescr->sigmaSqrd;
                CInvMat[CMatIdx[threadIdx.x]].x = 1.0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            } else {
                CInvMat[CMatIdx[threadIdx.x]].x = 0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute ((H*V)^H*(H*V) + sigmaSqrd*I)^-1
        // use CInvMat for storing ((H*V)^H*(H*V) + sigmaSqrd*I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]+col_i*pDynDescr->nBsAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    CInvMat[CMatIdx[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        // evaluate PF metric for each UE
        if (uIdx >= 0 && eIdx == 0) {
            // compute data rate
            float dataRate = 0;
            for (int j = 0; j < pDynDescr->nUeAnt; j++) {
                dataRate += pDynDescr->W*log2f(1.0/pDynDescr->sigmaSqrd/CInvMat[CMatIdxTemp[threadIdx.x]+j*pDynDescr->nBsAnt+j].x);
#ifdef SCSCHEDULER_DEBUG_
            //printf("CInvMat.x = %f, CInvMat.y = %f\n", CInvMat[CMatIdxTemp[threadIdx.x]+j*pDynDescr->nBsAnt+j].x, CInvMat[CMatIdxTemp[threadIdx.x]+j*pDynDescr->nBsAnt+j].y);
#endif
            }
            pfMetric[realAssocUeIdxInBlk] = dataRate/pDynDescr->avgRates[uIdx];
            ueIdxArr[realAssocUeIdxInBlk] = uIdx;
#ifdef SCSCHEDULER_DEBUG_
        // printf("uIdx = %d, dataRate = %f, pfMetric[assocUeIdxInBlk] = %f\n", uIdx, dataRate, pfMetric[assocUeIdxInBlk]);
#endif
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxSchdUePerRnd;
        }
        __syncthreads(); 
    }

    // select UE
    if (threadIdx.x == 0) {
        float   maxv = 0;
        int16_t maxi = -1;
        for (int j = 0; j<pDynDescr->nUe; j++) {
            if (ueIdxArr[j] < 0)
                break;
            if (pfMetric[j] > maxv) {
                maxv = pfMetric[j];
                maxi = ueIdxArr[j];
            }
        }

        pDynDescr->allocSol[blockIdx.x*pDynDescr->nCell + pDynDescr->cellId] = maxi;
        // printf("GPU: cellId = %d, rbgIdx = %d, maxv = %f, pDynDescr->allocSol = %d\n", pDynDescr->cellId, blockIdx.x, maxv, pDynDescr->allocSol[blockIdx.x*pDynDescr->nCell + pDynDescr->cellId]);
    }
}

// to-do: remove dependency of cooperative groups (max number of blocks is limited by the number of SMs)
static __global__ void singleCellSchedulerKernel_type1_NoPrdMmse(scDynDescr_t* pDynDescr)
{
    cg::grid_group g = cg::this_grid(); 

    uint16_t nBsAntSqrd             = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t nBsUeAntPrd            = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint16_t assocUeIdxInBlk        = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk    = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    __shared__ cuComplex  CMat[1024];
    __shared__ cuComplex  CInvMat[1024];
    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    __shared__ uint16_t   CMatIdxTemp[1024];
    __shared__ uint16_t   CMatIdx[1024];
    __shared__ uint16_t   ueIdxArr[1024];
    __shared__ float      pfMetric[1024];

    colIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
    rowIdx[threadIdx.x] = eIdx - colIdx[threadIdx.x]*pDynDescr->nBsAnt;
  
    CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
    CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;
  
    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    __shared__ int nAssocUeFound;
    uint16_t totNumAssocUeFound = 0;

    bool cnt = true;

    while (cnt) {
        if (threadIdx.x == 0)
            nAssocUeFound = 0;
        __syncthreads();

        uIdx = -1;

        int16_t assocUeRank = -1;
        for (int j = 0; j<pDynDescr->nUe; j++) {
            if (pDynDescr->cellAssoc[pDynDescr->cellId*pDynDescr->nUe + j]) {
                assocUeRank += 1;
                if (assocUeRank == realAssocUeIdxInBlk) {
                    uIdx = j;
                    if (eIdx == 0) {
                        atomicAdd(&nAssocUeFound, 1);
                    }
                    break;
                }
            }
        }
        __syncthreads();

        if (nAssocUeFound < pDynDescr->nMaxSchdUePerRnd) {
            cnt = false;
        }

        // compute H^H*H + sigmaSqrd*I
        // use CMat for storing H^H*H + sigmaSqrd*I
        if (uIdx >= 0) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_perUeBuffer[uIdx][blockIdx.x*nBsUeAntPrd+rowIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
                cuComplex tmp2 = pDynDescr->estH_fr_perUeBuffer[uIdx][blockIdx.x*nBsUeAntPrd+colIdx[threadIdx.x]*pDynDescr->nUeAnt+j];
                CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }

            if (colIdx[threadIdx.x] == rowIdx[threadIdx.x]) {
                CMat[CMatIdx[threadIdx.x]].x += pDynDescr->sigmaSqrd;
                CInvMat[CMatIdx[threadIdx.x]].x = 1.0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            } else {
                CInvMat[CMatIdx[threadIdx.x]].x = 0;
                CInvMat[CMatIdx[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute (H^H*H + sigmaSqrd*I)^-1
        // use CInvMat for storing (H^H*H + sigmaSqrd*I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]+col_i*pDynDescr->nBsAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nBsAnt];
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    CInvMat[CMatIdx[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        // evaluate PF metric for each UE
        if (uIdx >= 0 && eIdx == 0) {
            // compute data rate
            float dataRate = 0;
            for (int j = 0; j < pDynDescr->nUeAnt; j++) {
                dataRate += pDynDescr->W*log2f(1.0/pDynDescr->sigmaSqrd/CInvMat[CMatIdxTemp[threadIdx.x]+j*pDynDescr->nBsAnt+j].x);
            }
            pfMetric[realAssocUeIdxInBlk] = dataRate/pDynDescr->avgRates[uIdx];
            ueIdxArr[realAssocUeIdxInBlk] = pDynDescr->nPrbGrp*uIdx + blockIdx.x;
            if (blockIdx.x == 0) {
                pDynDescr->allocSol[uIdx*2] = -1;
                pDynDescr->allocSol[uIdx*2+1] = -1;
            }
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxSchdUePerRnd;
        }
        totNumAssocUeFound += nAssocUeFound;
        __syncthreads(); 
    }

    // transfer computed PF metrics from shared memory to global memory
    uint16_t nRound = (totNumAssocUeFound - 1)/blockDim.x + 1;

    for (int r = 0; r<nRound; r++) {
        uint32_t entry = r*blockDim.x + threadIdx.x;
        if (entry < totNumAssocUeFound) {
            pDynDescr->pfMetricArr[blockIdx.x*totNumAssocUeFound + entry] = pfMetric[entry];
            pDynDescr->pfIdArr[blockIdx.x*totNumAssocUeFound + entry] = ueIdxArr[entry];
        }
    }
    // synchronize all thread blocks
    cg::this_grid().sync();

    if (totNumAssocUeFound == 0)
        return;

    // perform consecutive PRB allocation by a single thread block
    if (blockIdx.x == 0) {
        // first sort computed PF metrices across all PRBs and all UEs
        uint32_t pow2N = 2;
        uint32_t pfSize = totNumAssocUeFound*pDynDescr->nPrbGrp;
        while(pow2N<pfSize) {
            pow2N = pow2N << 1;
        }

        nRound = (pow2N - pfSize - 1)/blockDim.x + 1;
        for (int r = 0; r< nRound; r++) {
            uint32_t entry = pfSize + r*blockDim.x + threadIdx.x;
            if (entry < pow2N) {
                pDynDescr->pfMetricArr[entry] = 0;
                pDynDescr->pfIdArr[entry] = 0;
            }
        }

        // initialize S array
        // Reuse colIdx array in shared memory for S array
        if (threadIdx.x < pDynDescr->nPrbGrp) {
            colIdx[threadIdx.x] = 1;
        } 
        
        bitonicSort(pDynDescr->pfMetricArr, pDynDescr->pfIdArr, pow2N); // internal synchronization

        // sequential riding peaks algorithm
        if (threadIdx.x == 0) {
            uint16_t nAllocated = 0;
            uint16_t k = 0;

            while(nAllocated < pDynDescr->nPrbGrp) {
                if (k == pfSize)
                    break;

                if (pDynDescr->pfMetricArr[k] == 0) {
                    k++;
                    continue;
                }

                uint16_t c = pDynDescr->pfIdArr[k]%pDynDescr->nPrbGrp;
        
                if (colIdx[c] == 0) {
                    pDynDescr->pfMetricArr[k] = 0;
                    k++;
                    continue;
                }
        
                uint16_t i = pDynDescr->pfIdArr[k]/pDynDescr->nPrbGrp;
        
                if (pDynDescr->allocSol[2*i] == -1) {
                    pDynDescr->allocSol[2*i] = c;
                    pDynDescr->allocSol[2*i+1] = c + 1;
                    colIdx[c] = 0;
                    pDynDescr->pfMetricArr[k] = 0;
                    nAllocated++;
                    k = 0;
                } else if (c == (pDynDescr->allocSol[2*i] - 1)) {
                    pDynDescr->allocSol[2*i] = c;
                    colIdx[c] = 0;
                    pDynDescr->pfMetricArr[k] = 0;
                    nAllocated++;
                    k = 0;
                } else if (c == pDynDescr->allocSol[2*i+1]) {
                    pDynDescr->allocSol[2*i+1] = c + 1;
                    colIdx[c] = 0;
                    pDynDescr->pfMetricArr[k] = 0;
                    nAllocated++;
                    k = 0;
                } else {
                    k++;
                }            
            }
        }
    }
}


void singleCellScheduler::kernelSelect()
{
    // choose kernel based on precoder and equalizer types
    switch (precodingScheme) {
        case 0: // no precoding
            if (allocType) {
                kernelFunc = reinterpret_cast<void*>(singleCellSchedulerKernel_type1_NoPrdMmse);
            } else {
                kernelFunc = reinterpret_cast<void*>(singleCellSchedulerKernel_noPrdMmse);
            }
            break;
        case 1: // SVD precoding
            if (allocType) {
                printf("Error: Kernel function not available\n");
            } else {
                kernelFunc = reinterpret_cast<void*>(singleCellSchedulerKernel_svdMmse);
            }
            break;
        default: // default no precoding with allocate type 0 (non-consecutive)
            kernelFunc = reinterpret_cast<void*>(singleCellSchedulerKernel_noPrdMmse);
            break;
    }
    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

    // launch geometry
    gridDim  = {numThrdBlk, 1, 1};
    blockDim = {numThrdPerBlk, 1, 1};

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

void singleCellScheduler::setup(uint16_t                    cellId,
                                cumacCellGrpUeStatus*       cellGrpUeStatus,
                                cumacSchdSol*               schdSol,
                                cumacCellGrpPrms*           cellGrpPrms,
                                cumacSimParam*              simParam,
                                cudaStream_t                strm)
{
    pCpuDynDesc->cellId                 = cellId; // index of the cell that calling the single-cell scheduler
    pCpuDynDesc->avgRates               = cellGrpUeStatus->avgRates;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->pfMetricArr            = schdSol->pfMetricArr;
    pCpuDynDesc->pfIdArr                = schdSol->pfIdArr;
    pCpuDynDesc->cellAssoc              = cellGrpPrms->cellAssoc;
    pCpuDynDesc->estH_fr_perUeBuffer    = cellGrpPrms->estH_fr_perUeBuffer;
    pCpuDynDesc->prdMat                 = cellGrpPrms->prdMat;
    pCpuDynDesc->nUe                    = cellGrpPrms->nUe; // total number of UEs
    pCpuDynDesc->nCell                  = simParam->totNumCell; // number of coordinated cells
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->W                      = cellGrpPrms->W;
    pCpuDynDesc->sigmaSqrd              = cellGrpPrms->sigmaSqrd;
    pCpuDynDesc->nMaxSchdUePerRnd       = floor(1024.0/(cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt));
    allocType                           = cellGrpPrms->allocType;
    precodingScheme                     = cellGrpPrms->precodingScheme;

    numThrdPerBlk = cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt*pCpuDynDesc->nMaxSchdUePerRnd;
    numThrdBlk    = cellGrpPrms->nPrbGrp;

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(scDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect();

    pLaunchCfg->kernelArgs[0] = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
}

void singleCellScheduler::run(cudaStream_t strm)
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
    if (allocType) {
        CUDA_CHECK_ERR(cudaLaunchCooperativeKernel(kernelFunc, gridDim, blockDim, kernelNodeParamsDriver.kernelParams));
    } else {
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
    }
    #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
    }
    CUDA_CHECK_ERR(cudaEventRecord(stop));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("milliseconds = %f\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif   
}
}