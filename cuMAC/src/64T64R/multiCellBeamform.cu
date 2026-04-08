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

multiCellBeamform::multiCellBeamform(cumacCellGrpPrms* cellGrpPrms)
{
    DL = cellGrpPrms->dlSchInd;
    allocType = cellGrpPrms->allocType;

    // sanity check
    if (cellGrpPrms->nBsAnt == 4) {
        throw std::runtime_error("Error: cuMAC multi-cell MIMO beamforming is not supported for 4T4R");
    }
    if (allocType == 0) {
        throw std::runtime_error("Error: cuMAC multi-cell MIMO beamforming is not supported for type-0 allocation");
    }
    if (DL == 0) {
        throw std::runtime_error("Error: cuMAC multi-cell MIMO beamforming is only supported for DL");
    }
    if (cellGrpPrms->bfPowAllocScheme != 0 && cellGrpPrms->bfPowAllocScheme != 1) {
        throw std::runtime_error("Error: cuMAC multi-cell MIMO beamforming is not supported for power allocation scheme other than 0 or 1");
    }   

    cpuDataBuff = std::malloc(sizeof(mcBeamformDynDescr_t) + sizeof(int));
    pCpuDynDesc = reinterpret_cast<mcBeamformDynDescr_t*>(cpuDataBuff);
    numCompleteBlk_h = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(cpuDataBuff) + sizeof(mcBeamformDynDescr_t));
    numCompleteBlk_h[0] = 0;
    
    CUDA_CHECK_ERR(cudaMalloc((void **)&gpuDataBuff, sizeof(mcBeamformDynDescr_t) + sizeof(int)));
    pGpuDynDesc = reinterpret_cast<mcBeamformDynDescr_t*>(gpuDataBuff);
    numCompleteBlk_d = reinterpret_cast<int*>(gpuDataBuff + sizeof(mcBeamformDynDescr_t));

    pLaunchCfg = std::make_unique<launchCfg_t>();
}

multiCellBeamform::~multiCellBeamform()
{
    std::free(cpuDataBuff);
    CUDA_CHECK_ERR(cudaFree(gpuDataBuff));
}

void multiCellBeamform::kernelSelect()
{
    void* kernelFunc = reinterpret_cast<void*>(multiCellBeamformKernel_rzf_dl);
              
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

void multiCellBeamform::setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
                              cumacSchdSol*               schdSol,
                              cumacCellGrpPrms*           cellGrpPrms,
                              cudaStream_t                strm)
{
    pCpuDynDesc->srsEstChan             = cellGrpPrms->srsEstChan;
    pCpuDynDesc->srsUeMap               = cellGrpPrms->srsUeMap;
    pCpuDynDesc->setSchdUePerCellTTI    = schdSol->setSchdUePerCellTTI;
    pCpuDynDesc->allocSol               = schdSol->allocSol;
    pCpuDynDesc->layerSelSol            = schdSol->layerSelSol;
    pCpuDynDesc->prdMat                 = cellGrpPrms->prdMat;
    pCpuDynDesc->beamformGainCurrTx     = cellGrpUeStatus->beamformGainCurrTx;
    pCpuDynDesc->bfGainPrgCurrTx        = cellGrpUeStatus->bfGainPrgCurrTx;
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->numUeForGrpPerCell     = cellGrpPrms->numUeForGrpPerCell;
    pCpuDynDesc->zfCoeff                = cellGrpPrms->zfCoeff;
    pCpuDynDesc->bfPowAllocScheme       = cellGrpPrms->bfPowAllocScheme;    
    pCpuDynDesc->ueOrderInGrp           = schdSol->ueOrderInGrp; 
    pCpuDynDesc->numCompleteBlk         = numCompleteBlk_d; 

    numThrdPerBlk = pCpuDynDesc->nBsAnt*maxNumLayerPerGrpDL_;
    numThrdBlk = pCpuDynDesc->nCell*pCpuDynDesc->nPrbGrp;

    CUDA_CHECK_ERR(cudaMemcpyAsync(gpuDataBuff, cpuDataBuff, sizeof(mcBeamformDynDescr_t) + sizeof(int), cudaMemcpyHostToDevice, strm));

    // select kernel 
    kernelSelect();
 
    pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}   

void multiCellBeamform::run(cudaStream_t strm)
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
    printf("Multi-cell MIMO beamforming ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif 
}

static __global__ void multiCellBeamformKernel_rzf_dl(mcBeamformDynDescr_t* pDynDescr)
{
    uint16_t cIdx               = floor(static_cast<float>(blockIdx.x)/pDynDescr->nPrbGrp);
    uint16_t rbgIdx             = blockIdx.x - cIdx*pDynDescr->nPrbGrp;
    uint32_t nBsUeAntPrd        = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint32_t nPrgBsUeAntPrd     = pDynDescr->nPrbGrp*nBsUeAntPrd;

    __shared__ cuComplex  stackedChann[maxNumBsAnt_*maxNumLayerPerGrpDL_];
    __shared__ cuComplex  regularizedGramMat[maxNumLayerPerGrpDL_*maxNumLayerPerGrpDL_];
    __shared__ cuComplex  invMat[maxNumLayerPerGrpDL_*maxNumLayerPerGrpDL_];
    __shared__ cuComplex  wRzf[maxNumBsAnt_*maxNumLayerPerGrpDL_];
    __shared__ uint16_t   grpUeId[maxNumLayerPerGrpDL_];
    __shared__ uint16_t   numSelLayers[maxNumLayerPerGrpDL_];
    __shared__ float      beamformGainPerLayer[maxNumLayerPerGrpDL_];
    __shared__ float      bfGainPrg[1024];
    __shared__ uint16_t   ueOrderInGrp[maxNumLayerPerGrpDL_];
    __shared__ uint16_t   layerStart[maxNumLayerPerGrpDL_]; 
    __shared__ float      fNormElem[1024];

    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    
    cuComplex c_coeff;
    cuComplex c_inv_coeff;
    cuComplex d_coeff;
    float     d_multp;
    cuComplex p_coeff;
    cuComplex p_inv_coeff;
    cuComplex l_coeff;

    __shared__ int numGrpUe;
    __shared__ int nLayersGrp;
    __shared__ int beamformFeasible;

    if (threadIdx.x == 0) {
        numGrpUe = 0;
        nLayersGrp = 0;
        beamformFeasible = 1;
    }
    if (threadIdx.x < maxNumLayerPerGrpDL_) {
        grpUeId[threadIdx.x] = 0xFFFF;
        numSelLayers[threadIdx.x] = 0xFFFF;
        ueOrderInGrp[threadIdx.x] = 0xFFFF; 
        layerStart[threadIdx.x] = 0xFFFF;   
        beamformGainPerLayer[threadIdx.x] = 0.0;    
    }

    for (int idx = threadIdx.x; idx < (pDynDescr->nBsAnt*maxNumLayerPerGrpDL_); idx += blockDim.x) {
        int prdMatIdx = cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + idx;
        pDynDescr->prdMat[prdMatIdx].x = 0;
        pDynDescr->prdMat[prdMatIdx].y = 0; 
    }
    __syncthreads();
    
    // determine UEs to be beamformed
    if (threadIdx.x < pDynDescr->numUeForGrpPerCell) {
        uint16_t ueId = pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + threadIdx.x];
        if (ueId != 0xFFFF) {
            if (rbgIdx >= pDynDescr->allocSol[2*ueId] && rbgIdx < pDynDescr->allocSol[2*ueId+1]) {
                int foundUeIdx = atomicAdd(&numGrpUe, 1);
                grpUeId[foundUeIdx] = ueId;
                numSelLayers[foundUeIdx] = pDynDescr->layerSelSol[ueId];
                ueOrderInGrp[foundUeIdx] = pDynDescr->ueOrderInGrp[ueId];   
            }
            // flush beamforming gain per PRG per Tx antenna port
            pDynDescr->bfGainPrgCurrTx[ueId*pDynDescr->nPrbGrp + rbgIdx] = 0;
        }
    }
    __syncthreads();

    // assemble stacked channel matrix
    int tempNumLayersGrp = 0;
    if (threadIdx.x < pDynDescr->nBsAnt) {
        for (int uIdx = 0; uIdx < numGrpUe; uIdx++) {
            uint16_t ueId;
            uint16_t numLayersThisUe;
            for (int tIdx = 0; tIdx < numGrpUe; tIdx++) {
                if (ueOrderInGrp[tIdx] == uIdx) {
                    ueId = grpUeId[tIdx];
                    numLayersThisUe = numSelLayers[tIdx];
                    if (threadIdx.x == 0) {
                        layerStart[tIdx] = tempNumLayersGrp;    
                    }
                    break;
                }
            }

            if (pDynDescr->srsUeMap[cIdx][ueId] == -1) { // UE is SU-MIMO and does not have SRS channel estimates
                if (threadIdx.x == 0) {
                    beamformFeasible = 0;   
                }
            } else {
                for (int lIdx = 0; lIdx < numLayersThisUe; lIdx++) {
                    stackedChann[(tempNumLayersGrp + lIdx)*pDynDescr->nBsAnt + threadIdx.x] = pDynDescr->srsEstChan[cIdx][pDynDescr->srsUeMap[cIdx][ueId]*nPrgBsUeAntPrd + rbgIdx*nBsUeAntPrd + lIdx*pDynDescr->nBsAnt + threadIdx.x];
                }
                
            }
            tempNumLayersGrp += numLayersThisUe;
        }
    }
    
    if (threadIdx.x == 0) {
        nLayersGrp = tempNumLayersGrp;  
    }
    __syncthreads();
    
    if (nLayersGrp > 0 && beamformFeasible == 1) {
        rowIdx[threadIdx.x] = floor(static_cast<float>(threadIdx.x)/nLayersGrp);
        colIdx[threadIdx.x] = threadIdx.x - rowIdx[threadIdx.x]*nLayersGrp;
        __syncthreads(); 

        // compute regularized Gram matrix HH'
        if (threadIdx.x < nLayersGrp*nLayersGrp) {
            regularizedGramMat[threadIdx.x].x = 0.0;
            regularizedGramMat[threadIdx.x].y = 0.0;
            for (int idx = 0; idx < pDynDescr->nBsAnt; idx++) {
                cuComplex tmp1 = stackedChann[rowIdx[threadIdx.x]*pDynDescr->nBsAnt + idx];
                cuComplex tmp2 = stackedChann[colIdx[threadIdx.x]*pDynDescr->nBsAnt + idx];
                regularizedGramMat[threadIdx.x].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                regularizedGramMat[threadIdx.x].y += tmp2.x*tmp1.y - tmp1.x*tmp2.y;
            }

            if (colIdx[threadIdx.x] == rowIdx[threadIdx.x]) {
                regularizedGramMat[threadIdx.x].x += pDynDescr->zfCoeff;
                invMat[threadIdx.x].x = 1.0;
                invMat[threadIdx.x].y = 0;
            } else {
                invMat[threadIdx.x].x = 0;
                invMat[threadIdx.x].y = 0;
            }
        }
        __syncthreads();

        // compute inverse of regularized Gram matrix (HH')^-1
        for (int col_i=0; col_i < nLayersGrp; col_i++) {
            if (threadIdx.x < nLayersGrp*nLayersGrp) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = regularizedGramMat[col_i*nLayersGrp+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = regularizedGramMat[threadIdx.x];
                    c_inv_coeff = invMat[threadIdx.x];
                }
            }
            __syncthreads(); 
            if (threadIdx.x < nLayersGrp*nLayersGrp) {
                if (rowIdx[threadIdx.x] == col_i) {
                    regularizedGramMat[threadIdx.x].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    regularizedGramMat[threadIdx.x].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    invMat[threadIdx.x].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    invMat[threadIdx.x].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = regularizedGramMat[rowIdx[threadIdx.x]*nLayersGrp+col_i];
                }
            }
            __syncthreads(); 

            if (threadIdx.x < nLayersGrp*nLayersGrp) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = regularizedGramMat[col_i*nLayersGrp+colIdx[threadIdx.x]];
                    p_inv_coeff = invMat[col_i*nLayersGrp+colIdx[threadIdx.x]];
                    c_coeff = regularizedGramMat[threadIdx.x];
                    c_inv_coeff = invMat[threadIdx.x];

                    regularizedGramMat[threadIdx.x].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    regularizedGramMat[threadIdx.x].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    invMat[threadIdx.x].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    invMat[threadIdx.x].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads();
        }

        // compute wRzf H'(HH')^-1
        if (threadIdx.x < pDynDescr->nBsAnt*nLayersGrp) {
            wRzf[threadIdx.x].x = 0.0;
            wRzf[threadIdx.x].y = 0.0;
            for (int idx = 0; idx < nLayersGrp; idx++) {
                cuComplex tmp1 = stackedChann[idx*pDynDescr->nBsAnt + rowIdx[threadIdx.x]];
                cuComplex tmp2 = invMat[idx*nLayersGrp + colIdx[threadIdx.x]];
                wRzf[threadIdx.x].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                wRzf[threadIdx.x].y += tmp1.x*tmp2.y - tmp1.y*tmp2.x;
            }
        }   
        __syncthreads();

        if (pDynDescr->bfPowAllocScheme == 0) { 
            if (threadIdx.x < pDynDescr->nBsAnt*nLayersGrp) {
                cuComplex tmp1 = wRzf[threadIdx.x];
                fNormElem[threadIdx.x] = tmp1.x*tmp1.x + tmp1.y*tmp1.y;
            }
            __syncthreads();

            uint16_t h = pDynDescr->nBsAnt*nLayersGrp;
            uint16_t s = ceilf(h*0.5f);
            #pragma unroll
            while(s > 1) {
                if(threadIdx.x < (h - s)) {
                    fNormElem[threadIdx.x] += fNormElem[threadIdx.x + s];
                }
                h = s; 
                s = ceilf(h*0.5f);

                __syncthreads();
            }

            if (threadIdx.x == 0) {
                fNormElem[0] += fNormElem[1];
            }
            __syncthreads();

            if (threadIdx.x < nLayersGrp) {
                beamformGainPerLayer[threadIdx.x] = 1.0/sqrt(fNormElem[0]);
            }
        } else if (pDynDescr->bfPowAllocScheme == 1) {
            if (threadIdx.x < nLayersGrp) {
                float normPerLayer = 0;
                for (int aIdx = 0; aIdx < pDynDescr->nBsAnt; aIdx++) {
                    cuComplex tmp1 = wRzf[aIdx*nLayersGrp + threadIdx.x];
                    normPerLayer += tmp1.x*tmp1.x + tmp1.y*tmp1.y;
                }
                beamformGainPerLayer[threadIdx.x] = 1.0/sqrt(static_cast<float>(nLayersGrp) * normPerLayer);
            }
        }
        __syncthreads();

        // compute final beamforming weights    
        if (threadIdx.x < pDynDescr->nBsAnt*nLayersGrp) {
            pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].x = wRzf[threadIdx.x].x * beamformGainPerLayer[colIdx[threadIdx.x]];   
            pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].y = wRzf[threadIdx.x].y * beamformGainPerLayer[colIdx[threadIdx.x]];   
        }

        // determine beamforming gain per UE
        if (threadIdx.x < numGrpUe) {
            uint16_t ueId =  grpUeId[threadIdx.x];

            float bfGain = 0;
            for (int idx = 0; idx < numSelLayers[threadIdx.x]; idx++) {
                bfGain += 20.0*log10(beamformGainPerLayer[layerStart[threadIdx.x] + idx]);
            }
            pDynDescr->bfGainPrgCurrTx[ueId*pDynDescr->nPrbGrp + rbgIdx] = bfGain/numSelLayers[threadIdx.x];
        }
        __syncthreads();
    } else if (nLayersGrp > 0 && beamformFeasible == 0) { // SU-MIMO but SRS channel estimates are not available  
        rowIdx[threadIdx.x] = floor(static_cast<float>(threadIdx.x)/nLayersGrp);
        colIdx[threadIdx.x] = threadIdx.x - rowIdx[threadIdx.x]*nLayersGrp;
        __syncthreads(); 

        if (threadIdx.x < pDynDescr->nBsAnt*nLayersGrp) {
            if (rowIdx[threadIdx.x] == colIdx[threadIdx.x]) {
                pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].x = 1.0/sqrt(static_cast<float>(nLayersGrp));   
                pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].y = 0;   
            } else {
                pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].x = 0;
                pDynDescr->prdMat[cIdx*pDynDescr->nPrbGrp*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rbgIdx*pDynDescr->nBsAnt*maxNumLayerPerGrpDL_ + 
                        rowIdx[threadIdx.x]*maxNumLayerPerGrpDL_ + 
                        colIdx[threadIdx.x]].y = 0;    
            }
        }

        if (threadIdx.x < numGrpUe) {
            uint16_t ueId =  grpUeId[threadIdx.x];
            pDynDescr->bfGainPrgCurrTx[ueId*pDynDescr->nPrbGrp + rbgIdx] = -100.0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(pDynDescr->numCompleteBlk, 1);
    }
    
    if (rbgIdx == 0) {
        if (threadIdx.x == 0) {
            unsigned int ns = 8;
            while (atomicCAS(pDynDescr->numCompleteBlk, gridDim.x, gridDim.x) < gridDim.x) {
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
            }
        }
    }
    __syncthreads();

    if (rbgIdx == 0) {
        for (int uIdx = 0; uIdx < pDynDescr->numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = pDynDescr->setSchdUePerCellTTI[cIdx*pDynDescr->numUeForGrpPerCell + uIdx];
            if (ueId != 0xFFFF) {
                for (int idx = threadIdx.x; idx < pDynDescr->nPrbGrp; idx += blockDim.x) {
                    bfGainPrg[idx] = pDynDescr->bfGainPrgCurrTx[ueId*pDynDescr->nPrbGrp + idx];
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    float avgBfGain = 0;
                    int16_t prgStart = pDynDescr->allocSol[2*ueId];
                    int16_t prgEnd = pDynDescr->allocSol[2*ueId + 1];
                    uint16_t numAllocPrg = prgEnd - prgStart;
                    for (int idx = prgStart; idx < prgEnd; idx++) {
                        avgBfGain += bfGainPrg[idx];
                    }
                    
                    pDynDescr->beamformGainCurrTx[ueId] = avgBfGain/numAllocPrg;    
                }
                __syncthreads();
            } else {
                break;
            }
        }
    }
    // verification
}
}