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

#define dir 0 // controls direction of comparator sorts

template <typename T>
inline __device__ void bitonicSortId(T* valueArr, uint16_t n)
{
   for (int size = 2; size < n; size*=2) {
       int d=dir^((threadIdx.x & (size / 2)) != 0);
      
       for (int stride = size / 2; stride > 0; stride/=2) {
          __syncthreads(); 

          if(threadIdx.x<n/2) {
             int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

             if ((valueArr[pos] > valueArr[pos + stride]) == d) {
                 T t = valueArr[pos];
                 valueArr[pos] = valueArr[pos + stride];
                 valueArr[pos + stride] = t;
             }
          }
       }
   }
   
   for (int stride = n / 2; stride > 0; stride/=2) {
       __syncthreads(); 
       if(threadIdx.x<n/2) {
          int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

          if ((valueArr[pos] > valueArr[pos + stride]) == dir) {
              T t = valueArr[pos];
              valueArr[pos] = valueArr[pos + stride];
              valueArr[pos + stride] = t;
          }
       }
   }

   __syncthreads(); 
}

template <typename T1, typename T2>
inline __device__ void bitonicSort(T1* valueArr, T2* idArr, uint16_t n)
{
   for (int size = 2; size < n; size*=2) {
       int d=dir^((threadIdx.x & (size / 2)) != 0);
      
       for (int stride = size / 2; stride > 0; stride/=2) {
          __syncthreads(); 

          if(threadIdx.x<n/2) {
             int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

             T1 t;
             T2 t_id;

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

          T1 t;
          T2 t_id;

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

multiCellMuUeGrp::multiCellMuUeGrp(cumacCellGrpPrms* cellGrpPrms)
{
    ueGrpMode   = cellGrpPrms->ueGrpMode;
    allocType   = cellGrpPrms->allocType;
    DL          = cellGrpPrms->dlSchInd;
    enableHarq  = cellGrpPrms->harqEnabledInd;
    
    // sanity check
    if (allocType == 0) {
        throw std::runtime_error("Error: cuMAC MU-MIMO scheduler does not support type-0 allocation yet");
    }

    // allocate memory for dynamic descriptors
    pCpuDynDesc = std::make_unique<mcUeGrpDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcUeGrpDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();
}

multiCellMuUeGrp::~multiCellMuUeGrp()
{
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
}

//---------------------------- DL kernels ----------------------------------
static __global__ void multiCellMuUeGrpKernel_dl_semiStatic(mcUeGrpDynDescr_t* pDynDescr)
{
    uint16_t cIdx = blockIdx.x;

    uint32_t nBsUeAntPrd = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint32_t nPrgBsUeAntPrd = pDynDescr->nPrbGrp*nBsUeAntPrd;

    __shared__ int          muMimoUe[maxNumActUePerCell_];
    __shared__ int          suMimoUe[maxNumActUePerCell_];   
    __shared__ uint8_t      selInUeGrp[maxNumActUePerCell_];

    __shared__ float        srsSnrInGrp[maxNumUegPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueGrpUeIdx[maxNumUegPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueGrpLayerIdx[maxNumUegPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueGrpUe[maxNumUegPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueLayers[maxNumUegPerCell_*maxNumLayerPerGrpDL_];

    __shared__ uint16_t     layersSchdPerGrp[maxNumUegPerCell_];
    __shared__ uint16_t     layersSchdSuUe[maxNumActUePerCell_];

    __shared__ uint16_t     reTxUeg[maxNumUegPerCell_];
    __shared__ uint16_t     reTxUegNumPrg[maxNumUegPerCell_];

    __shared__ float        uegPf[maxNumUegPerCell_];
    __shared__ uint16_t     uegIdArr[maxNumUegPerCell_];

    __shared__ uint8_t      subbandAllocInd[maxNumUeGrpSchdPerCell_];

    __shared__ cuComplex    innerProduct[maxNumBsAnt_];
    __shared__ float        h1norm[maxNumBsAnt_];
    __shared__ float        h2norm[maxNumBsAnt_];

    __shared__ int nMuUeFound;  
    __shared__ int nSuUeFound;
    __shared__ int numMuUeGrp;
    __shared__ uint8_t numUeInGrp;
    __shared__ bool orthogonal;
    __shared__ bool findAllLayers;
    __shared__ bool dmrsPortUsedUp;
    __shared__ uint8_t nScidCurr;
    __shared__ int numUegReTx;

    if (threadIdx.x < maxNumUeGrpSchdPerCell_) {
        subbandAllocInd[threadIdx.x] = 0;   
    }

    if (threadIdx.x < pDynDescr->numUeForGrpPerCell) {
        pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + threadIdx.x] = 0xFFFF;
    }

    if (threadIdx.x < maxNumActUePerCell_) { // assumption'
        muMimoUe[threadIdx.x] = -1;
        suMimoUe[threadIdx.x] = -1; 
        selInUeGrp[threadIdx.x] = 0;
        layersSchdSuUe[threadIdx.x] = 0;    
    }

    for (int idx = threadIdx.x; idx < (maxNumUegPerCell_*maxNumLayerPerGrpDL_); idx += blockDim.x) {
        srsSnrInGrp[idx] = -1000.0;
        ueGrpUeIdx[idx] = 0xFFFF;
        ueGrpLayerIdx[idx] = 0xFFFF;
        ueGrpUe[idx] = 0xFFFF;
        ueLayers[idx] = 0;
    }

    if (threadIdx.x < maxNumUegPerCell_) {
        layersSchdPerGrp[threadIdx.x] = 0;
        reTxUeg[threadIdx.x] = 0xFFFF;
        reTxUegNumPrg[threadIdx.x] = 0xFFFF;    
        uegPf[threadIdx.x] = 0;
        uegIdArr[threadIdx.x] = threadIdx.x;
    }

    if (threadIdx.x == 0) {
       nMuUeFound = 0;
       nSuUeFound = 0;  
       numMuUeGrp = 0;  
       numUegReTx = 0;
    }
    __syncthreads();

    if (pDynDescr->muGrpUpdate == 1) {  // UE grouping is triggered for this TTI
        for (int idx = threadIdx.x; idx < (maxNumUegPerCell_*maxNumLayerPerGrpDL_); idx += blockDim.x) {
            pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + idx] = 0xFFFF;
        }
        if (threadIdx.x < maxNumUegPerCell_) {
            pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + threadIdx.x] = 0;
        }   
        
        for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) {
            if (pDynDescr->cellAssocActUe[cIdx*pDynDescr->nActiveUe + uIdx] == 1) {
                if (pDynDescr->srsWbSnr[uIdx] >= pDynDescr->srsSnrThr) {
                    int storeIdx = atomicAdd(&nMuUeFound, 1);
                    muMimoUe[storeIdx] = uIdx;
                    pDynDescr->muMimoInd[uIdx] = 1;
                } else {
                    int storeIdx = atomicAdd(&nSuUeFound, 1);
                    suMimoUe[storeIdx] = uIdx;  
                    pDynDescr->muMimoInd[uIdx] = 0; 
                }
            }
        }
        __syncthreads();

        bitonicSortId<int>(muMimoUe, maxNumActUePerCell_);
        bitonicSortId<int>(suMimoUe, maxNumActUePerCell_);   

        uint16_t prgIdx = pDynDescr->nPrbGrp/2;
       
        // MU-MIMO UE grouping  
        for (uint16_t uegIdx = 0; uegIdx < maxNumUegPerCell_; uegIdx++) {
            __syncthreads();
            if (threadIdx.x == 0) {
                nScidCurr = 0;
                findAllLayers = false;
                numUeInGrp = 0;
            }
            __syncthreads();

            for (uint16_t currMuUeIdx = 0; currMuUeIdx < nMuUeFound; currMuUeIdx++) {
                if (selInUeGrp[currMuUeIdx] == 1) { // UE already selected for a UE group
                    continue;
                }

                bool srsSnrGapMet = true;
                float thisUeSrsSnr = pDynDescr->srsWbSnr[muMimoUe[currMuUeIdx]];

                if (pDynDescr->muGrpSrsSnrSplitThr != -100.0 && numUeInGrp > 0) { 
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_] <= pDynDescr->muGrpSrsSnrSplitThr) {
                        if (thisUeSrsSnr > pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }
                    } else {
                        if (thisUeSrsSnr <= pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }   
                    }
                }
                
                for (uint16_t addedUeIdx = 0; addedUeIdx < maxNumLayerPerGrpDL_; addedUeIdx++) {
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + addedUeIdx] != -1000.0) {
                        if (abs(thisUeSrsSnr - srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + addedUeIdx]) > pDynDescr->muGrpSrsSnrMaxGap) {
                            srsSnrGapMet = false;
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if (!srsSnrGapMet) {
                    continue;
                }

                if (threadIdx.x == 0) {
                    dmrsPortUsedUp = false;
                }

                for (uint16_t layerIdx = 0; layerIdx < pDynDescr->nMaxLayerPerUeMuDl; layerIdx++) {
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        orthogonal = true;
                    }
                    __syncthreads();

                    // traverse all existing channel vectors for the current UE group
                    for (uint16_t lIdx = 0; lIdx < layersSchdPerGrp[uegIdx]; lIdx++) {
                        // determine orthogonality
                        if (threadIdx.x < pDynDescr->nBsAnt) {
                            cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][muMimoUe[currMuUeIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + layerIdx*pDynDescr->nBsAnt + threadIdx.x];
                            cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][ueGrpUeIdx[uegIdx*maxNumLayerPerGrpDL_ + lIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpDL_ + lIdx]*pDynDescr->nBsAnt + threadIdx.x];

                            innerProduct[threadIdx.x].x = tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                            innerProduct[threadIdx.x].y = tmp1.x*tmp2.y - tmp2.x*tmp1.y;
                            h1norm[threadIdx.x] = tmp1.x*tmp1.x + tmp1.y*tmp1.y;
                            h2norm[threadIdx.x] = tmp2.x*tmp2.x + tmp2.y*tmp2.y;
                        }
                        __syncthreads();

                        // parallel reduction to calculate average SINR per UE
                        uint16_t h = pDynDescr->nBsAnt;
                        uint16_t s = ceilf(h*0.5f);
                        #pragma unroll
                        while(s > 1) {
                            if(threadIdx.x < (h - s)) {
                                innerProduct[threadIdx.x].x += innerProduct[threadIdx.x + s].x;
                                innerProduct[threadIdx.x].y += innerProduct[threadIdx.x + s].y;
                                h1norm[threadIdx.x] += h1norm[threadIdx.x + s];
                                h2norm[threadIdx.x] += h2norm[threadIdx.x + s];
                            }
                            h = s; 
                            s = ceilf(h*0.5f);

                            __syncthreads();
                        }

                        if (threadIdx.x == 0) {
                            innerProduct[0].x += innerProduct[1].x;
                            innerProduct[0].y += innerProduct[1].y;

                            h1norm[0] += h1norm[1];
                            h2norm[0] += h2norm[1];

                            float corrVal = h1norm[0]*h2norm[0] == 0 ? std::numeric_limits<float>::max() : sqrt((innerProduct[0].x*innerProduct[0].x + innerProduct[0].y*innerProduct[0].y)/(h1norm[0]*h2norm[0]));
                            if (corrVal > pDynDescr->chanCorrThr) {
                                orthogonal = false;
                            }
                        }
                        __syncthreads();

                        if (!orthogonal) {
                            break;
                        }
                    }
                    
                    if (orthogonal) {
                        if (threadIdx.x == 0) {
                            ueGrpUeIdx[uegIdx*maxNumLayerPerGrpDL_ + layersSchdPerGrp[uegIdx]] = muMimoUe[currMuUeIdx];
                            ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpDL_ + layersSchdPerGrp[uegIdx]] = layerIdx;
                            layersSchdPerGrp[uegIdx]++;

                            if (selInUeGrp[currMuUeIdx] == 0) {
                                ueGrpUe[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] = muMimoUe[currMuUeIdx];
                                selInUeGrp[currMuUeIdx] = 1;
                                pDynDescr->nSCID[muMimoUe[currMuUeIdx]] = nScidCurr;
                            }
                            ueLayers[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp]++;

                            if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] == -1000.0) {
                                srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] = thisUeSrsSnr;
                            }

                            if (layersSchdPerGrp[uegIdx] == totNumPdschDmrsPort_) {
                                dmrsPortUsedUp = true;
                                nScidCurr = 1 - nScidCurr;  
                            }
                            
                            if (layersSchdPerGrp[uegIdx] == pDynDescr->nMaxLayerPerGrpDl) {
                                findAllLayers = true;
                            }
                        }
                    } else {
                        break;
                    }
                    __syncthreads();

                    if (findAllLayers || dmrsPortUsedUp) {
                        break;
                    }
                }

                if (threadIdx.x == 0) {
                    if (selInUeGrp[currMuUeIdx] == 1) {
                        numUeInGrp++;
                    }
                }
                __syncthreads();

                if (findAllLayers || numUeInGrp == pDynDescr->nMaxUePerGrpDl) {
                    break;
                }
            }

            if (threadIdx.x == 0) {
                if (numUeInGrp > 0) {
                    for (int uIdx = 0; uIdx < numUeInGrp; uIdx++) {
                        uint16_t actUeIdx = ueGrpUe[uegIdx*maxNumLayerPerGrpDL_ + uIdx];
                        pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + numMuUeGrp*maxNumLayerPerGrpDL_ + uIdx] = actUeIdx;
                        pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + numMuUeGrp] = numUeInGrp;
                        pDynDescr->layerSelSol[actUeIdx] = ueLayers[uegIdx*maxNumLayerPerGrpDL_ + uIdx];
                        pDynDescr->ueOrderInGrp[actUeIdx] = uIdx;
                    }
                    numMuUeGrp++;
                }
            }   

            if (numUeInGrp == 0) {
                break;
            }
        }
        __syncthreads();

        // SU-MIMO UEs  
        if (pDynDescr->riActUe != nullptr) { // RI-based SU-MIMO scheduling
            if (threadIdx.x < nSuUeFound) {
                int8_t riVal = pDynDescr->riActUe[suMimoUe[threadIdx.x]];
                if (riVal >= 1 && riVal <= pDynDescr->nUeAnt) { // valid RI
                    layersSchdSuUe[threadIdx.x] = riVal > pDynDescr->nMaxLayerPerUeSuDl ? pDynDescr->nMaxLayerPerUeSuDl : riVal;
                } else {
                    layersSchdSuUe[threadIdx.x] = 1;
                }
            }
        } else { // SRS-based SU-MIMO scheduling
            for (uint16_t suUeId = 0; suUeId < nSuUeFound; suUeId++) {
                int32_t chanPos =  pDynDescr->srsUeMap[cIdx][suMimoUe[suUeId]];
                if (chanPos == -1) { // SRS channel estimates not available for the current SU-MIMO UE
                    layersSchdSuUe[suUeId] = 1; // default to single-layer transmission when SRS channel estimates are not available
                } else {
                    for (uint16_t layerIdx = 0; layerIdx < pDynDescr->nMaxLayerPerUeSuDl; layerIdx++) {
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            orthogonal = true;
                        }
                        __syncthreads();
        
                        // traverse all existing channel vectors for the current UE group
                        for (uint16_t lIdx = 0; lIdx < layersSchdSuUe[suUeId]; lIdx++) {
                            // determine orthogonality
                            float corrVal = 0;
                            for (uint16_t prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
                                if (threadIdx.x < pDynDescr->nBsAnt) {
                                    cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][chanPos*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + layerIdx*pDynDescr->nBsAnt + threadIdx.x];
                                    cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][chanPos*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + lIdx*pDynDescr->nBsAnt + threadIdx.x];
        
                                    innerProduct[threadIdx.x].x = tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                                    innerProduct[threadIdx.x].y = tmp1.x*tmp2.y - tmp2.x*tmp1.y;
                                    h1norm[threadIdx.x] = tmp1.x*tmp1.x + tmp1.y*tmp1.y;
                                    h2norm[threadIdx.x] = tmp2.x*tmp2.x + tmp2.y*tmp2.y;
                                }
                                __syncthreads();
        
                                // parallel reduction to calculate average SINR per UE
                                uint16_t h = pDynDescr->nBsAnt;
                                uint16_t s = ceilf(h*0.5f);
                            #pragma unroll
                                while(s > 1) {
                                    if(threadIdx.x < (h - s)) {
                                        innerProduct[threadIdx.x].x += innerProduct[threadIdx.x + s].x;
                                        innerProduct[threadIdx.x].y += innerProduct[threadIdx.x + s].y;
                                        h1norm[threadIdx.x] += h1norm[threadIdx.x + s];
                                        h2norm[threadIdx.x] += h2norm[threadIdx.x + s];
                                    }
                                    h = s; 
                                    s = ceilf(h*0.5f);
        
                                    __syncthreads();
                                }
        
                                if (threadIdx.x == 0) {
                                    innerProduct[0].x += innerProduct[1].x;
                                    innerProduct[0].y += innerProduct[1].y;
        
                                    h1norm[0] += h1norm[1];
                                    h2norm[0] += h2norm[1];
        
                                    if (corrVal < std::numeric_limits<float>::max()) {
                                        corrVal = h1norm[0]*h2norm[0] == 0 ? std::numeric_limits<float>::max() : (corrVal + sqrt((innerProduct[0].x*innerProduct[0].x + innerProduct[0].y*innerProduct[0].y)/(h1norm[0]*h2norm[0])));
                                    }
                                }
        
                                __syncthreads();
                            }
                            if (threadIdx.x == 0) {
                                corrVal /= static_cast<float>(pDynDescr->nPrbGrp);
                                if (corrVal > pDynDescr->chanCorrThr) {
                                    orthogonal = false;
                                }
                            }
                            __syncthreads();
        
                            if (!orthogonal) {
                                break;
                            }
                        }
                        
                        if (orthogonal) {
                            if (threadIdx.x == 0) {
                                layersSchdSuUe[suUeId]++;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (threadIdx.x < nSuUeFound) {
            pDynDescr->nSCID[suMimoUe[threadIdx.x]] = 0;
            pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + (threadIdx.x + numMuUeGrp)*maxNumLayerPerGrpDL_] = suMimoUe[threadIdx.x];
            pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + (threadIdx.x + numMuUeGrp)] = 1;  
            pDynDescr->layerSelSol[suMimoUe[threadIdx.x]] = layersSchdSuUe[threadIdx.x];
            pDynDescr->ueOrderInGrp[suMimoUe[threadIdx.x]] = 0;
        }
        __syncthreads();

        if (pDynDescr->semiStatFreqAlloc == 1) { // semi-static subband allocation
            if (threadIdx.x < maxNumUegPerCell_) {
                if (pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + threadIdx.x*maxNumLayerPerGrpDL_] != 0xFFFF) {
                    pDynDescr->muGrpList->subbandId[cIdx*maxNumUegPerCell_ + threadIdx.x] = threadIdx.x % pDynDescr->nMaxUegPerCellDl; // % maximum number of subbands
                } else {
                    pDynDescr->muGrpList->subbandId[cIdx*maxNumUegPerCell_ + threadIdx.x] = -1; 
                }
            }
            __syncthreads();
        }
    } 

    // UE selection and PRG allocation
    if (threadIdx.x < maxNumUegPerCell_) {
        if (pDynDescr->semiStatFreqAlloc == 1) { // semi-static subband allocation
            for (int uIdx = 0; uIdx < maxNumLayerPerGrpDL_; uIdx++) {
                uint16_t ueIdx = pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + threadIdx.x*maxNumLayerPerGrpDL_ + uIdx];
                
                if (ueIdx != 0xFFFF) {
                    float dataRate = pDynDescr->W*log2f(1.0 + pDynDescr->wbSinr[ueIdx*pDynDescr->nUeAnt]);
                    uegPf[threadIdx.x] += pow(dataRate, pDynDescr->betaCoeff)/pDynDescr->avgRatesActUe[ueIdx];
                } else {
                    break;
                }
            }
        } else { // dynamic subband allocation  
            for (int uIdx = 0; uIdx < maxNumLayerPerGrpDL_; uIdx++) {
                uint16_t ueIdx = pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + threadIdx.x*maxNumLayerPerGrpDL_ + uIdx];
                
                if (ueIdx != 0xFFFF) {
                    if (pDynDescr->newDataActUe[ueIdx] == 0) {
                        int storeIdx = atomicAdd(&numUegReTx, 1);
                        reTxUeg[storeIdx] = threadIdx.x;
                        reTxUegNumPrg[storeIdx] = pDynDescr->allocSolLastTx[ueIdx*2+1] - pDynDescr->allocSolLastTx[ueIdx*2];
                        uegPf[threadIdx.x] = 0;
                        break;
                    } else { // new transmission
                        float dataRate = pDynDescr->W*log2f(1.0 + pDynDescr->wbSinr[ueIdx*pDynDescr->nUeAnt]);
                        uegPf[threadIdx.x] += pow(dataRate, pDynDescr->betaCoeff)/pDynDescr->avgRatesActUe[ueIdx];
                    }
                } else {
                    break;
                }
            }
        }
    }
    __syncthreads();

    bitonicSort<float, uint16_t>(uegPf, uegIdArr, maxNumUegPerCell_); // internal synchronization

    if (pDynDescr->semiStatFreqAlloc == 1) { // semi-static subband allocation
        if (threadIdx.x == 0) {
            uint16_t numPrgPerSubbband = floor(static_cast<float>(pDynDescr->nPrbGrp)/pDynDescr->nMaxUegPerCellDl);
            uint16_t numRemPrg = pDynDescr->nPrbGrp - numPrgPerSubbband*pDynDescr->nMaxUegPerCellDl;

            uint16_t selUeIdx = 0;
            uint16_t allocSubband = 0;
            for (int uegIdx = 0; uegIdx < maxNumUegPerCell_; uegIdx++) {
                if (uegPf[uegIdx] > 0 && selUeIdx < pDynDescr->numUeSchdPerCellTTI && allocSubband < pDynDescr->nMaxUegPerCellDl) {
                    uint16_t uegId = uegIdArr[uegIdx];
                    int16_t  subbandId = pDynDescr->muGrpList->subbandId[cIdx*maxNumUegPerCell_ + uegId];
                    if (subbandId != -1 && subbandAllocInd[subbandId] == 0) {
                        uint16_t tempNumUeInUeg = pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + uegId];
                        if (tempNumUeInUeg <= (pDynDescr->numUeSchdPerCellTTI - selUeIdx)) {
                            int16_t prgStart;
                            int16_t prgEnd;
                            if (subbandId < numRemPrg) {
                                prgStart = static_cast<int16_t>(subbandId*numPrgPerSubbband + subbandId);
                                prgEnd = static_cast<int16_t>(prgStart + numPrgPerSubbband + 1);
                            } else {
                                prgStart = static_cast<int16_t>(subbandId*numPrgPerSubbband + numRemPrg);
                                prgEnd = static_cast<int16_t>(prgStart + numPrgPerSubbband);  
                            }

                            for (int uIdx = 0; uIdx < tempNumUeInUeg; uIdx++) {
                                uint16_t ueId = pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + uegId*maxNumLayerPerGrpDL_ + uIdx];
                                pDynDescr->allocSol[2*ueId]   = prgStart;
                                pDynDescr->allocSol[2*ueId+1] = prgEnd;
                                pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueId;
                                selUeIdx++;
                            }

                            subbandAllocInd[subbandId] = 1;
                            allocSubband++;
                        }
                    }
                } else {
                    break;
                }
            }
        }
    } else { // dynamic subband allocation  
        __shared__ uint8_t schdNewTxUeg[maxNumUeGrpSchdPerCell_];
        __shared__ uint8_t numUeSchdNewTxUeg[maxNumUeGrpSchdPerCell_];
        __shared__ uint16_t numPrgPerGrp[maxNumUeGrpSchdPerCell_];

        if (threadIdx.x == 0) {
            uint16_t selUeIdx = 0;
            uint16_t numRemainingPrg = pDynDescr->nPrbGrp;

            uint16_t startRbgAlloc = 0;
            for (int reUegIdx = 0; reUegIdx < numUegReTx; reUegIdx++) {
                if ((numRemainingPrg >= reTxUegNumPrg[reUegIdx]) && (pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + reTxUeg[reUegIdx]] <= (pDynDescr->numUeSchdPerCellTTI - selUeIdx))) {
                    for (int uIdx = 0; uIdx < maxNumLayerPerGrpDL_; uIdx++) {
                        uint16_t ueIdx = pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + reTxUeg[reUegIdx]*maxNumLayerPerGrpDL_ + uIdx];
                        if (ueIdx != 0xFFFF) {
                            pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                            pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + reTxUegNumPrg[reUegIdx]);
                            pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueIdx;
                            selUeIdx++;
                        } else {
                            break;
                        }
                    }
                    numRemainingPrg -= reTxUegNumPrg[reUegIdx];
                    startRbgAlloc += reTxUegNumPrg[reUegIdx];
                }
            }

            // top-K new-TX UEG selection
            uint8_t numSchdNewTxUeg = 0;
            uint8_t numSchdNewTxUe = 0;
            for (int uegIdx = 0; uegIdx < maxNumUegPerCell_; uegIdx++) {
                if (uegPf[uegIdx] > 0) {
                    uint16_t uegId = uegIdArr[uegIdx];
                    uint16_t tempNumUeInUeg = pDynDescr->muGrpList->numUeInGrp[cIdx*maxNumUegPerCell_ + uegId];
                    if ((numSchdNewTxUe + tempNumUeInUeg) <= (pDynDescr->numUeSchdPerCellTTI - selUeIdx)) {
                        schdNewTxUeg[numSchdNewTxUeg] = uegId;  
                        numUeSchdNewTxUeg[numSchdNewTxUeg] = tempNumUeInUeg;
                        numSchdNewTxUeg++;
                        numSchdNewTxUe += tempNumUeInUeg;
                        if (numSchdNewTxUeg == pDynDescr->nMaxUegPerCellDl) {
                            break;
                        }
                    } 
                } else {
                    break;
                }
            }

            uint16_t numAllocRbgPerUe = 0;
            if (numSchdNewTxUe > 0) {
                numAllocRbgPerUe = floor(static_cast<float>(numRemainingPrg)/numSchdNewTxUe);
            }
        
            uint16_t numRemainingRbg = numRemainingPrg - numAllocRbgPerUe*numSchdNewTxUe;

            for (int idx = 0; idx < numSchdNewTxUeg; idx++) {
                if (numRemainingRbg >= numUeSchdNewTxUeg[idx]) {
                    numPrgPerGrp[idx] = (numAllocRbgPerUe + 1)*numUeSchdNewTxUeg[idx];
                    numRemainingRbg -= numUeSchdNewTxUeg[idx];
                } else if (numRemainingRbg > 0) {
                    numPrgPerGrp[idx] = numAllocRbgPerUe*numUeSchdNewTxUeg[idx] + numRemainingRbg;
                    numRemainingRbg = 0;
                } else {
                    numPrgPerGrp[idx] = numAllocRbgPerUe*numUeSchdNewTxUeg[idx];
                }
            }

            for (int uegIdx = 0; uegIdx < numSchdNewTxUeg; uegIdx++) {
                uint16_t uegId = schdNewTxUeg[uegIdx];
                uint16_t tempNumUeInUeg = numUeSchdNewTxUeg[uegIdx];

                if (numPrgPerGrp[uegIdx] == 0) {
                    continue;
                }

                for (int uIdx = 0; uIdx < tempNumUeInUeg; uIdx++) {
                    uint16_t ueId = pDynDescr->muGrpList->ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + uegId*maxNumLayerPerGrpDL_ + uIdx];
                    pDynDescr->allocSol[2*ueId]   = static_cast<int16_t>(startRbgAlloc);
                    pDynDescr->allocSol[2*ueId+1] = static_cast<int16_t>(startRbgAlloc + numPrgPerGrp[uegIdx]);
                    pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueId;
                    selUeIdx++;
                }
                startRbgAlloc += numPrgPerGrp[uegIdx];  
            } 
        }
    }
}

static __global__ void multiCellMuUeGrpKernel_dl_dynPerTTI(mcUeGrpDynDescr_t* pDynDescr)
{
    uint16_t cIdx = blockIdx.x;
    uint32_t nBsUeAntPrd = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint32_t nPrgBsUeAntPrd = pDynDescr->nPrbGrp*nBsUeAntPrd;

    __shared__ float        srsSnrInGrp[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_];
    __shared__ int          newTxUe[maxNumUeForGrpPerCell_];
    __shared__ int          reTxUe[maxNumUeForGrpPerCell_];
    __shared__ uint16_t     numResvdPrgReTx[maxNumUeForGrpPerCell_];
    __shared__ uint16_t     ueGrpUeIdx[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueGrpLayerIdx[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     layersSchdPerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t     ueGrpUe[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t     ueLayers[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_];
    __shared__ uint8_t      selInUeGrp[maxNumUeForGrpPerCell_];
    __shared__ cuComplex    innerProduct[maxNumBsAnt_];
    __shared__ float        h1norm[maxNumBsAnt_];
    __shared__ float        h2norm[maxNumBsAnt_];

    __shared__ uint16_t numUeNewTx;
    __shared__ uint16_t numUeReTx;
    __shared__ uint8_t numUeInGrp;
    __shared__ bool orthogonal;
    __shared__ bool findAllLayers;
    __shared__ bool dmrsPortUsedUp;
    __shared__ uint8_t nScidCurr;

    __shared__ uint8_t muMimoInd[maxNumUeForGrpPerCell_];

    if (threadIdx.x < maxNumUeForGrpPerCell_) {
        newTxUe[threadIdx.x] = -1;
        reTxUe[threadIdx.x] = -1;
        numResvdPrgReTx[threadIdx.x] = 0xFFFF;
        selInUeGrp[threadIdx.x] = 0;
        muMimoInd[threadIdx.x] = 0xFF;
    }
    __syncthreads();

    if (threadIdx.x < pDynDescr->numUeForGrpPerCell) {
        pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + threadIdx.x] = 0xFFFF;
        uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][threadIdx.x];
        
        if (ueIdx != 0xFFFF) {
            pDynDescr->allocSol[2*ueIdx]        = -1;
            pDynDescr->allocSol[2*ueIdx+1]      = -1;
            pDynDescr->layerSelSol[ueIdx]       = 0xFF;
        }
    }

    if (threadIdx.x == 0) {
        numUeNewTx = 0;
        numUeReTx = 0;
        for (int idx = 0; idx < pDynDescr->numUeForGrpPerCell; idx++) {
            uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][idx];
            if (ueIdx == 0xFFFF) {
                break;
            }
            if (pDynDescr->bufferSize == nullptr || pDynDescr->bufferSize[ueIdx] > 0) {
                if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[ueIdx] == 1) { // new-TX UE
                    newTxUe[numUeNewTx] = ueIdx;
                    numUeNewTx++;
                } else { // re-TX UE
                    reTxUe[numUeReTx] = ueIdx;
                    numResvdPrgReTx[numUeReTx] = pDynDescr->allocSolLastTx[ueIdx*2+1] - pDynDescr->allocSolLastTx[ueIdx*2];
                    numUeReTx++;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < numUeNewTx) {
        muMimoInd[threadIdx.x] = pDynDescr->muMimoInd[newTxUe[threadIdx.x]];
    }

    if (threadIdx.x < maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpDL_) {
        ueGrpUeIdx[threadIdx.x] = 0xFFFF;
        ueGrpUe[threadIdx.x] = 0xFFFF;
        ueLayers[threadIdx.x] = 0;
        ueGrpLayerIdx[threadIdx.x] = 0xFFFF;
        srsSnrInGrp[threadIdx.x] = -1000.0;
    }

    if (threadIdx.x < maxNumUeGrpSchdPerCell_) {
        layersSchdPerGrp[threadIdx.x] = 0;
    }

    uint16_t prgIdx = pDynDescr->nPrbGrp/2;

    if (numUeNewTx > 0) { // new-TX UE(s) exists
        for (uint16_t uegIdx = 0; uegIdx < maxNumUeGrpSchdPerCell_; uegIdx++) {
            __syncthreads();
            if (threadIdx.x == 0) {
                nScidCurr = 0;
                findAllLayers = false;
                numUeInGrp = 0;
            }
            __syncthreads();

            for (uint16_t currUeIdx = 0; currUeIdx < numUeNewTx; currUeIdx++) {
                if (selInUeGrp[currUeIdx] == 1) { // UE already selected for a UE group
                    continue;
                }

                if (numUeInGrp > 0 && muMimoInd[currUeIdx] == 0) {
                    continue;
                }

                bool srsSnrGapMet = true;
                float thisUeSrsSnr = pDynDescr->srsWbSnr[newTxUe[currUeIdx]];

                if (pDynDescr->muGrpSrsSnrSplitThr != -100.0 && numUeInGrp > 0) { 
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_] <= pDynDescr->muGrpSrsSnrSplitThr) {
                        if (thisUeSrsSnr > pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }
                    } else {
                        if (thisUeSrsSnr <= pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }   
                    }
                }
                
                for (uint16_t addedUeIdx = 0; addedUeIdx < maxNumLayerPerGrpDL_; addedUeIdx++) {
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + addedUeIdx] != -1000.0) {
                        if (abs(thisUeSrsSnr - srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + addedUeIdx]) > pDynDescr->muGrpSrsSnrMaxGap) {
                            srsSnrGapMet = false;
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if (!srsSnrGapMet) {
                    continue;
                }

                if (threadIdx.x == 0) {
                    dmrsPortUsedUp = false;
                }

                if (thisUeSrsSnr == -100.0) {
                    if (pDynDescr->riActUe != nullptr) { // RI-based SU-MIMO scheduling
                        int8_t ri = pDynDescr->riActUe[newTxUe[currUeIdx]];
                        if (ri >= 1 && ri <= pDynDescr->nUeAnt) {
                            if (threadIdx.x == 0) {
                                ueGrpUe[uegIdx*maxNumLayerPerGrpDL_] = newTxUe[currUeIdx];
                                selInUeGrp[currUeIdx] = 1;
                                pDynDescr->nSCID[newTxUe[currUeIdx]] = nScidCurr;
                                layersSchdPerGrp[uegIdx] = ri > pDynDescr->nMaxLayerPerUeSuDl ? pDynDescr->nMaxLayerPerUeSuDl : ri;
                                ueLayers[uegIdx*maxNumLayerPerGrpDL_] = layersSchdPerGrp[uegIdx];
                            }
                            break;
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else {
                    for (uint16_t layerIdx = 0; layerIdx < pDynDescr->nMaxLayerPerUeMuDl; layerIdx++) {
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            orthogonal = true;
                        }
                        __syncthreads();
    
                        // traverse all existing channel vectors for the current UE group
                        for (uint16_t lIdx = 0; lIdx < layersSchdPerGrp[uegIdx]; lIdx++) {
                            // determine orthogonality
                            if (threadIdx.x < pDynDescr->nBsAnt) {
                                cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][newTxUe[currUeIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + layerIdx*pDynDescr->nBsAnt + threadIdx.x];
                                cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][ueGrpUeIdx[uegIdx*maxNumLayerPerGrpDL_ + lIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpDL_ + lIdx]*pDynDescr->nBsAnt + threadIdx.x];
    
                                innerProduct[threadIdx.x].x = tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                                innerProduct[threadIdx.x].y = tmp1.x*tmp2.y - tmp2.x*tmp1.y;
                                h1norm[threadIdx.x] = tmp1.x*tmp1.x + tmp1.y*tmp1.y;
                                h2norm[threadIdx.x] = tmp2.x*tmp2.x + tmp2.y*tmp2.y;
                            }
                            __syncthreads();
    
                            // parallel reduction to calculate average SINR per UE
                            uint16_t h = pDynDescr->nBsAnt;
                            uint16_t s = ceilf(h*0.5f);
                            #pragma unroll
                            while(s > 1) {
                                if(threadIdx.x < (h - s)) {
                                    innerProduct[threadIdx.x].x += innerProduct[threadIdx.x + s].x;
                                    innerProduct[threadIdx.x].y += innerProduct[threadIdx.x + s].y;
                                    h1norm[threadIdx.x] += h1norm[threadIdx.x + s];
                                    h2norm[threadIdx.x] += h2norm[threadIdx.x + s];
                                }
                                h = s; 
                                s = ceilf(h*0.5f);
    
                                __syncthreads();
                            }
    
                            if (threadIdx.x == 0) {
                                innerProduct[0].x += innerProduct[1].x;
                                innerProduct[0].y += innerProduct[1].y;
    
                                h1norm[0] += h1norm[1];
                                h2norm[0] += h2norm[1];
    
                                float corrVal = h1norm[0]*h2norm[0] == 0 ? std::numeric_limits<float>::max() : (sqrt((innerProduct[0].x*innerProduct[0].x + innerProduct[0].y*innerProduct[0].y)/(h1norm[0]*h2norm[0])));
                                if (corrVal > pDynDescr->chanCorrThr) {
                                    orthogonal = false;
                                }
                            }
                            __syncthreads();
    
                            if (!orthogonal) {
                                break;
                            }
                        }
                        
                        if (orthogonal) {
                            if (threadIdx.x == 0) {
                                ueGrpUeIdx[uegIdx*maxNumLayerPerGrpDL_ + layersSchdPerGrp[uegIdx]] = newTxUe[currUeIdx];
                                ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpDL_ + layersSchdPerGrp[uegIdx]] = layerIdx;
                                layersSchdPerGrp[uegIdx]++;
    
                                if (selInUeGrp[currUeIdx] == 0) {
                                    ueGrpUe[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] = newTxUe[currUeIdx];
                                    selInUeGrp[currUeIdx] = 1;
                                    pDynDescr->nSCID[newTxUe[currUeIdx]] = nScidCurr;
                                }
                                ueLayers[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp]++;
    
                                if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] == -1000.0) {
                                    srsSnrInGrp[uegIdx*maxNumLayerPerGrpDL_ + numUeInGrp] = thisUeSrsSnr;
                                }
    
                                if (layersSchdPerGrp[uegIdx] == totNumPdschDmrsPort_) {
                                    dmrsPortUsedUp = true;
                                    nScidCurr = 1 - nScidCurr;  
                                }
                                
                                if (layersSchdPerGrp[uegIdx] == pDynDescr->nMaxLayerPerGrpDl) {
                                    findAllLayers = true;
                                }
                            }
                        } else {
                            break;
                        }
                        __syncthreads();
    
                        if (findAllLayers || dmrsPortUsedUp) {
                            break;
                        }
                    }
    
                    if (threadIdx.x == 0) {
                        if (selInUeGrp[currUeIdx] == 1) {
                            numUeInGrp++;
                        }
                    }
                    __syncthreads();
    
                    if (muMimoInd[currUeIdx] == 0) {
                        break;
                    }
    
                    if (findAllLayers || numUeInGrp == pDynDescr->nMaxUePerGrpDl || numUeInGrp == pDynDescr->numUeSchdPerCellTTI) {
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    __shared__ uint16_t numUePerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t numPrgPerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t selUeGrp[maxNumUeGrpSchdPerCell_];

    if (threadIdx.x < maxNumUeGrpSchdPerCell_) {
        numUePerGrp[threadIdx.x] = pDynDescr->nMaxUePerGrpDl;
        for (int idx = 0; idx < pDynDescr->nMaxUePerGrpDl; idx++) {
            if (ueGrpUe[threadIdx.x*maxNumLayerPerGrpDL_ + idx] == 0xFFFF) {
                numUePerGrp[threadIdx.x] = idx;
                break;
            }
        }
    } 
    __syncthreads();

    if (threadIdx.x == 0) {
        // first allocate PRBGs for SU-MIMO UEs with retransmission   
        uint16_t selUeIdx = 0;
        uint16_t numRemainingPrg = pDynDescr->nPrbGrp;

        uint16_t startRbgAlloc = 0;
        for (uint16_t uid = 0; uid < numUeReTx; uid++) {
            uint16_t ueIdx = reTxUe[uid];
            if (numRemainingPrg >= numResvdPrgReTx[uid]) {
                pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numResvdPrgReTx[uid]);
                pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueIdx;
                pDynDescr->layerSelSol[ueIdx] = pDynDescr->layerSelSolLastTx[ueIdx];
                pDynDescr->ueOrderInGrp[ueIdx] = 0; 
                selUeIdx++;
                startRbgAlloc += numResvdPrgReTx[uid];
                numRemainingPrg -= numResvdPrgReTx[uid];

                if (selUeIdx == pDynDescr->numUeSchdPerCellTTI) {
                    break;
                }   
            }
        }

        // top-K UE group selection
        uint16_t totNumUeInUeGrp = 0;
        uint16_t selUeGrpIdx = 0;
        for (int idx = 0; idx < maxNumUeGrpSchdPerCell_; idx++) {
            if (layersSchdPerGrp[idx] == 0) {
                break;
            }

            if (totNumUeInUeGrp + numUePerGrp[idx] <= (pDynDescr->numUeSchdPerCellTTI - selUeIdx)) {
                selUeGrp[selUeGrpIdx] = idx;
                totNumUeInUeGrp += numUePerGrp[idx];
                selUeGrpIdx++;
                if (selUeGrpIdx == pDynDescr->nMaxUegPerCellDl) {
                    break;
                }
            }
        }

        uint16_t totSchd = selUeGrpIdx;

        uint16_t numAllocRbgPerUe = 0;
        if (totNumUeInUeGrp > 0) {
            numAllocRbgPerUe = floor(static_cast<float>(numRemainingPrg)/totNumUeInUeGrp);
        }
        
        uint16_t numRemainingRbg = numRemainingPrg - numAllocRbgPerUe*totNumUeInUeGrp;

        for (int idx = 0; idx < totSchd; idx++) {
            if (numRemainingRbg >= numUePerGrp[selUeGrp[idx]]) {
                numPrgPerGrp[idx] = (numAllocRbgPerUe + 1)*numUePerGrp[selUeGrp[idx]];
                numRemainingRbg -= numUePerGrp[selUeGrp[idx]];
            } else if (numRemainingRbg > 0) {
                numPrgPerGrp[idx] = numAllocRbgPerUe*numUePerGrp[selUeGrp[idx]] + numRemainingRbg;
                numRemainingRbg = 0;
            } else {
                numPrgPerGrp[idx] = numAllocRbgPerUe*numUePerGrp[selUeGrp[idx]];
            }
        }

        for (uint16_t tempUeIdx = 0; tempUeIdx < totSchd; tempUeIdx++) {
            uint16_t ueIdx;
            uint16_t uegId = selUeGrp[tempUeIdx];
            for (uint16_t idx = 0; idx < pDynDescr->nMaxUePerGrpDl; idx++) {
                ueIdx = ueGrpUe[uegId*maxNumLayerPerGrpDL_ + idx];
                    
                if (ueIdx != 0xFFFF) {
                    if (numPrgPerGrp[tempUeIdx] > 0) {
                        pDynDescr->allocSol[2*ueIdx]        = static_cast<int16_t>(startRbgAlloc);
                        pDynDescr->allocSol[2*ueIdx+1]      = static_cast<int16_t>(startRbgAlloc + numPrgPerGrp[tempUeIdx]);
                        pDynDescr->layerSelSol[ueIdx]       = ueLayers[uegId*maxNumLayerPerGrpDL_ + idx];
                        pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueIdx;
                        pDynDescr->ueOrderInGrp[ueIdx]      = idx;
                        selUeIdx++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            startRbgAlloc += numPrgPerGrp[tempUeIdx];
        }
    }
}

//---------------------------- UL kernels ----------------------------------
static __global__ void multiCellMuUeGrpKernel_ul_dynPerTTI(mcUeGrpDynDescr_t* pDynDescr)
{
    uint16_t cIdx = blockIdx.x;
    uint32_t nBsUeAntPrd = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint32_t nPrgBsUeAntPrd = pDynDescr->nPrbGrp*nBsUeAntPrd;

    __shared__ float        srsSnrInGrp[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_];
    __shared__ int          newTxUe[maxNumUeForGrpPerCell_];
    __shared__ int          reTxUe[maxNumUeForGrpPerCell_];
    __shared__ uint16_t     numResvdPrgReTx[maxNumUeForGrpPerCell_];
    __shared__ uint16_t     ueGrpUeIdx[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_];
    __shared__ uint16_t     ueGrpLayerIdx[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_];
    __shared__ uint16_t     layersSchdPerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t     ueGrpUe[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_];
    __shared__ uint16_t     ueLayers[maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_];
    __shared__ uint8_t      selInUeGrp[maxNumUeForGrpPerCell_];
    __shared__ cuComplex    innerProduct[maxNumBsAnt_];
    __shared__ float        h1norm[maxNumBsAnt_];
    __shared__ float        h2norm[maxNumBsAnt_];

    __shared__ uint16_t numUeNewTx;
    __shared__ uint16_t numUeReTx;
    __shared__ uint8_t numUeInGrp;
    __shared__ bool orthogonal;
    __shared__ bool findAllLayers;
    __shared__ bool dmrsPortUsedUp;
    __shared__ uint8_t nScidCurr;

    __shared__ uint8_t muMimoInd[maxNumUeForGrpPerCell_];

    if (threadIdx.x < maxNumUeForGrpPerCell_) {
        newTxUe[threadIdx.x] = -1;
        reTxUe[threadIdx.x] = -1;
        numResvdPrgReTx[threadIdx.x] = 0xFFFF;
        selInUeGrp[threadIdx.x] = 0;
        muMimoInd[threadIdx.x] = 0xFF;
    }
    __syncthreads();

    if (threadIdx.x < pDynDescr->numUeForGrpPerCell) {
        pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + threadIdx.x] = 0xFFFF;
        uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][threadIdx.x];
        
        if (ueIdx != 0xFFFF) {
            pDynDescr->allocSol[2*ueIdx]        = -1;
            pDynDescr->allocSol[2*ueIdx+1]      = -1;
            pDynDescr->layerSelSol[ueIdx]       = 0xFF;
        }
    }

    if (threadIdx.x == 0) {
        numUeNewTx = 0;
        numUeReTx = 0;
        for (int idx = 0; idx < pDynDescr->numUeForGrpPerCell; idx++) {
            uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][idx];
            if (ueIdx == 0xFFFF) {
                break;
            }
            if (pDynDescr->bufferSize == nullptr || pDynDescr->bufferSize[ueIdx] > 0) {
                if (pDynDescr->newDataActUe == nullptr || pDynDescr->newDataActUe[ueIdx] == 1) { // new-TX UE
                    newTxUe[numUeNewTx] = ueIdx;
                    numUeNewTx++;
                } else { // re-TX UE
                    reTxUe[numUeReTx] = ueIdx;
                    numResvdPrgReTx[numUeReTx] = pDynDescr->allocSolLastTx[ueIdx*2+1] - pDynDescr->allocSolLastTx[ueIdx*2];
                    numUeReTx++;
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < numUeNewTx) {
        muMimoInd[threadIdx.x] = pDynDescr->muMimoInd[newTxUe[threadIdx.x]];
    }

    if (threadIdx.x < maxNumUeGrpSchdPerCell_*maxNumLayerPerGrpUL_) {
        ueGrpUeIdx[threadIdx.x] = 0xFFFF;
        ueGrpUe[threadIdx.x] = 0xFFFF;
        ueLayers[threadIdx.x] = 0;
        ueGrpLayerIdx[threadIdx.x] = 0xFFFF;
        srsSnrInGrp[threadIdx.x] = -1000.0;
    }

    if (threadIdx.x < maxNumUeGrpSchdPerCell_) {
        layersSchdPerGrp[threadIdx.x] = 0;
    }

    uint16_t prgIdx = pDynDescr->nPrbGrp/2;

    if (numUeNewTx > 0) { // new-TX UE(s) exists
        for (uint16_t uegIdx = 0; uegIdx < maxNumUeGrpSchdPerCell_; uegIdx++) {
            __syncthreads();
            if (threadIdx.x == 0) {
                nScidCurr = 0;
                findAllLayers = false;
                numUeInGrp = 0;
            }
            __syncthreads();

            for (uint16_t currUeIdx = 0; currUeIdx < numUeNewTx; currUeIdx++) {
                if (selInUeGrp[currUeIdx] == 1) { // UE already selected for a UE group
                    continue;
                }

                if (numUeInGrp > 0 && muMimoInd[currUeIdx] == 0) {
                    continue;
                }

                bool srsSnrGapMet = true;
                float thisUeSrsSnr = pDynDescr->srsWbSnr[newTxUe[currUeIdx]];

                if (pDynDescr->muGrpSrsSnrSplitThr != -100.0 && numUeInGrp > 0) { 
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpUL_] <= pDynDescr->muGrpSrsSnrSplitThr) {
                        if (thisUeSrsSnr > pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }
                    } else {
                        if (thisUeSrsSnr <= pDynDescr->muGrpSrsSnrSplitThr) {
                            continue;
                        }   
                    }
                }
                
                for (uint16_t addedUeIdx = 0; addedUeIdx < maxNumLayerPerGrpUL_; addedUeIdx++) {
                    if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpUL_ + addedUeIdx] != -1000.0) {
                        if (abs(thisUeSrsSnr - srsSnrInGrp[uegIdx*maxNumLayerPerGrpUL_ + addedUeIdx]) > pDynDescr->muGrpSrsSnrMaxGap) {
                            srsSnrGapMet = false;
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if (!srsSnrGapMet) {
                    continue;
                }

                if (threadIdx.x == 0) {
                    dmrsPortUsedUp = false;
                }

                if (thisUeSrsSnr == -100.0) {
                    if (pDynDescr->riActUe != nullptr) { // RI-based SU-MIMO scheduling
                        int8_t ri = pDynDescr->riActUe[newTxUe[currUeIdx]];
                        if (ri >= 1 && ri <= pDynDescr->nUeAnt) {
                            if (threadIdx.x == 0) {
                                ueGrpUe[uegIdx*maxNumLayerPerGrpUL_] = newTxUe[currUeIdx];
                                selInUeGrp[currUeIdx] = 1;
                                pDynDescr->nSCID[newTxUe[currUeIdx]] = nScidCurr;
                                layersSchdPerGrp[uegIdx] = ri > pDynDescr->nMaxLayerPerUeSuUl ? pDynDescr->nMaxLayerPerUeSuUl : ri;
                                ueLayers[uegIdx*maxNumLayerPerGrpUL_] = layersSchdPerGrp[uegIdx];
                            }
                            break;
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else {
                    for (uint16_t layerIdx = 0; layerIdx < pDynDescr->nMaxLayerPerUeMuUl; layerIdx++) {
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            orthogonal = true;
                        }
                        __syncthreads();
    
                        // traverse all existing channel vectors for the current UE group
                        for (uint16_t lIdx = 0; lIdx < layersSchdPerGrp[uegIdx]; lIdx++) {
                            // determine orthogonality
                            if (threadIdx.x < pDynDescr->nBsAnt) {
                                cuComplex tmp1 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][newTxUe[currUeIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + layerIdx*pDynDescr->nBsAnt + threadIdx.x];
                                cuComplex tmp2 = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][ueGrpUeIdx[uegIdx*maxNumLayerPerGrpUL_ + lIdx]]*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpUL_ + lIdx]*pDynDescr->nBsAnt + threadIdx.x];
    
                                innerProduct[threadIdx.x].x = tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                                innerProduct[threadIdx.x].y = tmp1.x*tmp2.y - tmp2.x*tmp1.y;
                                h1norm[threadIdx.x] = tmp1.x*tmp1.x + tmp1.y*tmp1.y;
                                h2norm[threadIdx.x] = tmp2.x*tmp2.x + tmp2.y*tmp2.y;
                            }
                            __syncthreads();
    
                            // parallel reduction to calculate average SINR per UE
                            uint16_t h = pDynDescr->nBsAnt;
                            uint16_t s = ceilf(h*0.5f);
                            #pragma unroll
                            while(s > 1) {
                                if(threadIdx.x < (h - s)) {
                                    innerProduct[threadIdx.x].x += innerProduct[threadIdx.x + s].x;
                                    innerProduct[threadIdx.x].y += innerProduct[threadIdx.x + s].y;
                                    h1norm[threadIdx.x] += h1norm[threadIdx.x + s];
                                    h2norm[threadIdx.x] += h2norm[threadIdx.x + s];
                                }
                                h = s; 
                                s = ceilf(h*0.5f);
    
                                __syncthreads();
                            }
    
                            if (threadIdx.x == 0) {
                                innerProduct[0].x += innerProduct[1].x;
                                innerProduct[0].y += innerProduct[1].y;
    
                                h1norm[0] += h1norm[1];
                                h2norm[0] += h2norm[1];
    
                                float corrVal = h1norm[0]*h2norm[0] == 0 ? std::numeric_limits<float>::max() : (sqrt((innerProduct[0].x*innerProduct[0].x + innerProduct[0].y*innerProduct[0].y)/(h1norm[0]*h2norm[0])));
                                if (corrVal > pDynDescr->chanCorrThr) {
                                    orthogonal = false;
                                }
                            }
                            __syncthreads();
    
                            if (!orthogonal) {
                                break;
                            }
                        }
                        
                        if (orthogonal) {
                            if (threadIdx.x == 0) {
                                ueGrpUeIdx[uegIdx*maxNumLayerPerGrpUL_ + layersSchdPerGrp[uegIdx]] = newTxUe[currUeIdx];
                                ueGrpLayerIdx[uegIdx*maxNumLayerPerGrpUL_ + layersSchdPerGrp[uegIdx]] = layerIdx;
                                layersSchdPerGrp[uegIdx]++;
    
                                if (selInUeGrp[currUeIdx] == 0) {
                                    ueGrpUe[uegIdx*maxNumLayerPerGrpUL_ + numUeInGrp] = newTxUe[currUeIdx];
                                    selInUeGrp[currUeIdx] = 1;
                                    pDynDescr->nSCID[newTxUe[currUeIdx]] = nScidCurr;
                                }
                                ueLayers[uegIdx*maxNumLayerPerGrpUL_ + numUeInGrp]++;
    
                                if (srsSnrInGrp[uegIdx*maxNumLayerPerGrpUL_ + numUeInGrp] == -1000.0) {
                                    srsSnrInGrp[uegIdx*maxNumLayerPerGrpUL_ + numUeInGrp] = thisUeSrsSnr;
                                }
    
                                if (layersSchdPerGrp[uegIdx] == totNumPuschDmrsPort_) {
                                    dmrsPortUsedUp = true;
                                    nScidCurr = 1 - nScidCurr;  
                                }
                                
                                if (layersSchdPerGrp[uegIdx] == pDynDescr->nMaxLayerPerGrpUl) {
                                    findAllLayers = true;
                                }
                            }
                        } else {
                            break;
                        }
                        __syncthreads();
    
                        if (findAllLayers || dmrsPortUsedUp) {
                            break;
                        }
                    }
    
                    if (threadIdx.x == 0) {
                        if (selInUeGrp[currUeIdx] == 1) {
                            numUeInGrp++;
                        }
                    }
                    __syncthreads();
    
                    if (muMimoInd[currUeIdx] == 0) {
                        break;
                    }
    
                    if (findAllLayers || numUeInGrp == pDynDescr->nMaxUePerGrpUl || numUeInGrp == pDynDescr->numUeSchdPerCellTTI) {
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    __shared__ uint16_t numUePerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t numPrgPerGrp[maxNumUeGrpSchdPerCell_];
    __shared__ uint16_t selUeGrp[maxNumUeGrpSchdPerCell_];

    if (threadIdx.x < maxNumUeGrpSchdPerCell_) {
        numUePerGrp[threadIdx.x] = pDynDescr->nMaxUePerGrpUl;
        for (int idx = 0; idx < pDynDescr->nMaxUePerGrpUl; idx++) {
            if (ueGrpUe[threadIdx.x*maxNumLayerPerGrpUL_ + idx] == 0xFFFF) {
                numUePerGrp[threadIdx.x] = idx;
                break;
            }
        }
    } 
    __syncthreads();

    if (threadIdx.x == 0) {
        // first allocate PRBGs for SU-MIMO UEs with retransmission   
        uint16_t selUeIdx = 0;
        uint16_t numRemainingPrg = pDynDescr->nPrbGrp;

        uint16_t startRbgAlloc = 0;
        for (uint16_t uid = 0; uid < numUeReTx; uid++) {
            uint16_t ueIdx = reTxUe[uid];
            if (numRemainingPrg >= numResvdPrgReTx[uid]) {
                pDynDescr->allocSol[2*ueIdx]   = static_cast<int16_t>(startRbgAlloc);
                pDynDescr->allocSol[2*ueIdx+1] = static_cast<int16_t>(startRbgAlloc + numResvdPrgReTx[uid]);
                pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueIdx;
                pDynDescr->layerSelSol[ueIdx] = pDynDescr->layerSelSolLastTx[ueIdx];
                pDynDescr->ueOrderInGrp[ueIdx] = 0; 
                selUeIdx++;
                startRbgAlloc += numResvdPrgReTx[uid];
                numRemainingPrg -= numResvdPrgReTx[uid];

                if (selUeIdx == pDynDescr->numUeSchdPerCellTTI) {
                    break;
                }   
            }
        }

        // top-K UE group selection
        uint16_t totNumUeInUeGrp = 0;
        uint16_t selUeGrpIdx = 0;
        for (int idx = 0; idx < maxNumUeGrpSchdPerCell_; idx++) {
            if (layersSchdPerGrp[idx] == 0) {
                break;
            }

            if (totNumUeInUeGrp + numUePerGrp[idx] <= (pDynDescr->numUeSchdPerCellTTI - selUeIdx)) {
                selUeGrp[selUeGrpIdx] = idx;
                totNumUeInUeGrp += numUePerGrp[idx];
                selUeGrpIdx++;
                if (selUeGrpIdx == pDynDescr->nMaxUegPerCellUl) {
                    break;
                }
            }
        }

        uint16_t totSchd = selUeGrpIdx;

        uint16_t numAllocRbgPerUe = 0;
        if (totNumUeInUeGrp > 0) {
            numAllocRbgPerUe = floor(static_cast<float>(numRemainingPrg)/totNumUeInUeGrp);
        }
        
        uint16_t numRemainingRbg = numRemainingPrg - numAllocRbgPerUe*totNumUeInUeGrp;

        for (int idx = 0; idx < totSchd; idx++) {
            if (numRemainingRbg >= numUePerGrp[selUeGrp[idx]]) {
                numPrgPerGrp[idx] = (numAllocRbgPerUe + 1)*numUePerGrp[selUeGrp[idx]];
                numRemainingRbg -= numUePerGrp[selUeGrp[idx]];
            } else if (numRemainingRbg > 0) {
                numPrgPerGrp[idx] = numAllocRbgPerUe*numUePerGrp[selUeGrp[idx]] + numRemainingRbg;
                numRemainingRbg = 0;
            } else {
                numPrgPerGrp[idx] = numAllocRbgPerUe*numUePerGrp[selUeGrp[idx]];
            }
        }

        for (uint16_t tempUeIdx = 0; tempUeIdx < totSchd; tempUeIdx++) {
            uint16_t ueIdx;
            uint16_t uegId = selUeGrp[tempUeIdx];
            for (uint16_t idx = 0; idx < pDynDescr->nMaxUePerGrpUl; idx++) {
                ueIdx = ueGrpUe[uegId*maxNumLayerPerGrpUL_ + idx];
                    
                if (ueIdx != 0xFFFF) {
                    if (numPrgPerGrp[tempUeIdx] > 0) {
                        pDynDescr->allocSol[2*ueIdx]        = static_cast<int16_t>(startRbgAlloc);
                        pDynDescr->allocSol[2*ueIdx+1]      = static_cast<int16_t>(startRbgAlloc + numPrgPerGrp[tempUeIdx]);
                        pDynDescr->layerSelSol[ueIdx]       = ueLayers[uegId*maxNumLayerPerGrpUL_ + idx];
                        pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + selUeIdx] = ueIdx;
                        pDynDescr->ueOrderInGrp[ueIdx]      = idx;
                        selUeIdx++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            startRbgAlloc += numPrgPerGrp[tempUeIdx];
        }
    }
}

void multiCellMuUeGrp::kernelSelect()
{
    void* kernelFunc;

    if (ueGrpMode == 0) { // 0: dynamic UE grouping per TTI
        if (DL == 1) { // DL
            kernelFunc = reinterpret_cast<void*>(multiCellMuUeGrpKernel_dl_dynPerTTI);
        } else { // UL
            kernelFunc = reinterpret_cast<void*>(multiCellMuUeGrpKernel_ul_dynPerTTI);
        }
    } else { // 1: flag-triggered UE grouping (controlled by the muGrpUpdate flag in cumacCellGrpPrms)
        if (DL == 1) { // DL
            kernelFunc = reinterpret_cast<void*>(multiCellMuUeGrpKernel_dl_semiStatic);
        } else { // UL
            throw std::runtime_error("Error: cuMAC flag-triggered MU-MIMO UE grouping is not supported for UL");
        }
    }

    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));
  
    // launch geometry
    gridDim  = {numThrdBlk, 1, 1};
    blockDim = {numThrdPerBlk, 1, 1};
  
    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver   = pLaunchCfg->kernelNodeParamsDriver;
  
    kernelNodeParamsDriver.blockDimX                  = blockDim.x;
    kernelNodeParamsDriver.blockDimY                  = blockDim.y;
    kernelNodeParamsDriver.blockDimZ                  = blockDim.z;
  
    kernelNodeParamsDriver.gridDimX                   = gridDim.x;
    kernelNodeParamsDriver.gridDimY                   = gridDim.y;
    kernelNodeParamsDriver.gridDimZ                   = gridDim.z;
  
    kernelNodeParamsDriver.extra                      = nullptr;
    kernelNodeParamsDriver.sharedMemBytes             = 0;  
}

void multiCellMuUeGrp::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                             cumacSchdSol*               schdSol,
                             cumacCellGrpPrms*           cellGrpPrms,
                             cudaStream_t                strm)
{
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->numUeForGrpPerCell     = cellGrpPrms->numUeForGrpPerCell; // number of UEs considered for MU-MIMO UE grouping per TTI per cell
    pCpuDynDesc->numUeSchdPerCellTTI    = cellGrpPrms->numUeSchdPerCellTTI; // total number of SU-MIMO UEs and MU-MIMO UE groups scheduled per TTI per cell 
    pCpuDynDesc->nMaxLayerPerUeSuUl     = cellGrpPrms->nMaxLayerPerUeSuUl;
    pCpuDynDesc->nMaxLayerPerUeSuDl     = cellGrpPrms->nMaxLayerPerUeSuDl;
    pCpuDynDesc->nMaxLayerPerUeMuUl     = cellGrpPrms->nMaxLayerPerUeMuUl;
    pCpuDynDesc->nMaxLayerPerUeMuDl     = cellGrpPrms->nMaxLayerPerUeMuDl; 
    pCpuDynDesc->nMaxUePerGrpUl         = cellGrpPrms->nMaxUePerGrpUl;
    pCpuDynDesc->nMaxUePerGrpDl         = cellGrpPrms->nMaxUePerGrpDl;
    pCpuDynDesc->nMaxLayerPerGrpUl      = cellGrpPrms->nMaxLayerPerGrpUl;
    pCpuDynDesc->nMaxLayerPerGrpDl      = cellGrpPrms->nMaxLayerPerGrpDl;
    pCpuDynDesc->nMaxUegPerCellDl       = cellGrpPrms->nMaxUegPerCellDl;
    pCpuDynDesc->nMaxUegPerCellUl       = cellGrpPrms->nMaxUegPerCellUl;
    pCpuDynDesc->muGrpSrsSnrMaxGap      = cellGrpPrms->muGrpSrsSnrMaxGap;
    pCpuDynDesc->muGrpSrsSnrSplitThr    = cellGrpPrms->muGrpSrsSnrSplitThr; 
    pCpuDynDesc->chanCorrThr            = cellGrpPrms->chanCorrThr;
    pCpuDynDesc->srsSnrThr              = cellGrpPrms->srsSnrThr;
    pCpuDynDesc->muGrpUpdate            = cellGrpPrms->muGrpUpdate;  
    pCpuDynDesc->semiStatFreqAlloc      = cellGrpPrms->semiStatFreqAlloc;  
    pCpuDynDesc->W                      = cellGrpPrms->W;
    pCpuDynDesc->betaCoeff              = cellGrpPrms->betaCoeff;
    pCpuDynDesc->currSlotIdxPerCell     = cellGrpPrms->currSlotIdxPerCell;
    pCpuDynDesc->srsEstChan             = cellGrpPrms->srsEstChan;
    pCpuDynDesc->srsUeMap               = cellGrpPrms->srsUeMap;
    pCpuDynDesc->srsWbSnr               = cellGrpPrms->srsWbSnr;
    pCpuDynDesc->wbSinr                 = cellGrpPrms->wbSinr;
    pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
    pCpuDynDesc->avgRatesActUe          = cellGrpUeStatus->avgRatesActUe;
    pCpuDynDesc->riActUe                = cellGrpUeStatus->riActUe;
    pCpuDynDesc->lastSchdSlotActUe      = cellGrpUeStatus->lastSchdSlotActUe; 
    if (enableHarq == 1) { // HARQ is enabled
        pCpuDynDesc->newDataActUe       = cellGrpUeStatus->newDataActUe; 
        pCpuDynDesc->allocSolLastTx     = cellGrpUeStatus->allocSolLastTx; 
        pCpuDynDesc->layerSelSolLastTx  = cellGrpUeStatus->layerSelSolLastTx; 
    } else {
        pCpuDynDesc->newDataActUe       = nullptr;
        pCpuDynDesc->allocSolLastTx     = nullptr;
        pCpuDynDesc->layerSelSolLastTx  = nullptr;
    }

    if (cellGrpUeStatus->bufferSize != nullptr) {
        pCpuDynDesc->bufferSize = cellGrpUeStatus->bufferSize;
    } else {
        pCpuDynDesc->bufferSize = nullptr;
    }
    pCpuDynDesc->sortedUeList           = schdSol->sortedUeList;  
    pCpuDynDesc->muMimoInd              = schdSol->muMimoInd;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->layerSelSol            = schdSol->layerSelSol;
    pCpuDynDesc->nSCID                  = schdSol->nSCID;
    pCpuDynDesc->ueOrderInGrp           = schdSol->ueOrderInGrp;    
    pCpuDynDesc->muGrpList              = schdSol->muGrpList;   
    
    if (pCpuDynDesc->numUeForGrpPerCell > maxNumUeForGrpPerCell_) {
        throw std::runtime_error("Error: cuMAC MU-MIMO scheduler - number of UEs considered for MU-MIMO grouping exceeds the maximum supported value");
    }

    if (pCpuDynDesc->numUeSchdPerCellTTI > maxNumUeGrpSchdPerCell_) {
        throw std::runtime_error("Error: cuMAC MU-MIMO scheduler - number of scheduled UEs + UE groups exceeds the maximum supported value");
    }

    numThrdPerBlk = 1024;
    numThrdBlk = pCpuDynDesc->nCell;

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcUeGrpDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel 
    kernelSelect();
 
    pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void multiCellMuUeGrp::run(cudaStream_t strm)
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
    printf("Multi-cell MU-MIMO UE grouping ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif 
}

void multiCellMuUeGrp::debugLog()
{

}
}