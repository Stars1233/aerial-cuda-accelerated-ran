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

 #include "multiCellSinrCal.cuh"

 // cuMAC namespace
 namespace cumac {

 multiCellSinrCal::multiCellSinrCal(cumacCellGrpPrms* cellGrpPrms)
 {
    pCpuDynDesc = std::make_unique<mcSinrCalDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcSinrCalDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();

    DL = cellGrpPrms->dlSchInd;

    // set default to column-major channel matrix access
    columnMajor = 1;

    // set default to no precoding
    precodingScheme = 0;

    nActiveUe = cellGrpPrms->nActiveUe;
    nPrbGrp = cellGrpPrms->nPrbGrp;
    nUeAnt = cellGrpPrms->nUeAnt;
 }
 
 multiCellSinrCal::~multiCellSinrCal()
 {
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
 }

 //---------------------------- DL kernels ----------------------------------
 static __global__ void multiCellSinrCalKernel_noPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr) 
 {
    // determine indices
    uint16_t rbgIdx               = floor(static_cast<float>(blockIdx.x)/pDynDescr->nCell);
    uint16_t cIdx                 = pDynDescr->cellId[blockIdx.x - rbgIdx*pDynDescr->nCell];
    uint16_t nBsAntSqrd           = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t nUeAntSqrd           = pDynDescr->nUeAnt*pDynDescr->nUeAnt;
    uint16_t nBsUeAntPrd          = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint16_t nCellBsUeAntPrd      = pDynDescr->nCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd    = pDynDescr->nActiveUe*nCellBsUeAntPrd;
    uint16_t assocUeIdxInBlk      = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk  = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    // setup data arrays in shared memory
    __shared__ cuComplex  DInvMat[1024];
    __shared__ cuComplex  CMat[1024];
    __shared__ cuComplex  CInvMat[1024];
    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    __shared__ uint16_t   CMatIdxTemp[1024];
    __shared__ uint16_t   CMatIdx[1024];
    __shared__ uint8_t    colIdx2[1024];
    __shared__ uint8_t    rowIdx2[1024];
    __shared__ uint16_t   CMatIdxTemp2[1024];
    __shared__ uint16_t   CMatIdx2[1024];
    __shared__ uint16_t   CMatIdxTemp3[1024];
    __shared__ uint16_t   CMatIdx3[1024];

    colIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nUeAnt);
    rowIdx[threadIdx.x] = eIdx - colIdx[threadIdx.x]*pDynDescr->nUeAnt;
    CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nUeAntSqrd;
    CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;

    colIdx2[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
    rowIdx2[threadIdx.x] = eIdx - colIdx2[threadIdx.x]*pDynDescr->nBsAnt;
    CMatIdxTemp2[threadIdx.x] = assocUeIdxInBlk*nBsUeAntPrd;
    CMatIdx2[threadIdx.x] = CMatIdxTemp2[threadIdx.x] + eIdx;

    CMatIdxTemp3[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
    CMatIdx3[threadIdx.x] = CMatIdxTemp3[threadIdx.x] + eIdx;

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
        for (int j = 0; j<pDynDescr->nActiveUe; j++) {
            if (pDynDescr->cellAssocActUe[cIdx*pDynDescr->nActiveUe + j]) {
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

        if (nAssocUeFound < pDynDescr->nMaxUeSinrCalPerRnd) {
            cnt = false;
        }

        // compute C matrix
        if (uIdx >= 0 && eIdx < nUeAntSqrd) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (uint16_t lIdx = 0; lIdx < pDynDescr->nCell; lIdx++) {
                uint16_t l = pDynDescr->cellId[lIdx];

                if (l == cIdx) {
                    continue;
                }
                uint32_t hInterfMatStart = rbgIdx*nUeCellBsUeAntPrd+ uIdx*nCellBsUeAntPrd + l*nBsUeAntPrd;
          
                for (int j = 0; j<pDynDescr->nBsAnt; j++) {
                    cuComplex tmp1 = pDynDescr->estH_fr_actUe[hInterfMatStart + rowIdx[threadIdx.x] + j*pDynDescr->nUeAnt];
                    cuComplex tmp2 = pDynDescr->estH_fr_actUe[hInterfMatStart + colIdx[threadIdx.x] + j*pDynDescr->nUeAnt];
                    CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                    CMat[CMatIdx[threadIdx.x]].y += tmp2.x*tmp1.y - tmp1.x*tmp2.y;
                }
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

        // compute inverse of C matrix
        // use CInvMat for storing inverse of C matrix
        for (int col_i=0; col_i < pDynDescr->nUeAnt; col_i++) {
            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nUeAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]+col_i*pDynDescr->nUeAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nUeAnt];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nUeAnt];
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

        // compute H^H*C^-1
        // reuse CMat for storing H^H*C^-1
        uint32_t hMatStart = rbgIdx*nUeCellBsUeAntPrd + uIdx*nCellBsUeAntPrd + cIdx*nBsUeAntPrd;
        if (uIdx >= 0 && eIdx < nBsUeAntPrd) {
            CMat[CMatIdx2[threadIdx.x]].x = 0;
            CMat[CMatIdx2[threadIdx.x]].y = 0;
        
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_actUe[hMatStart+rowIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                cuComplex tmp2 = CInvMat[CMatIdxTemp[threadIdx.x]+colIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                CMat[CMatIdx2[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx2[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        // compute H^H*C^-1*H + I
        // reuse CInvMat for storing H^H*C^-1*H + I
        if (uIdx >= 0) {
            CInvMat[CMatIdx3[threadIdx.x]].x = 0;
            CInvMat[CMatIdx3[threadIdx.x]].y = 0;
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = CMat[CMatIdxTemp2[threadIdx.x]+j*pDynDescr->nBsAnt+rowIdx2[threadIdx.x]];
                cuComplex tmp2 = pDynDescr->estH_fr_actUe[hMatStart+colIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                CInvMat[CMatIdx3[threadIdx.x]].x += tmp1.x*tmp2.x - tmp1.y*tmp2.y;
                CInvMat[CMatIdx3[threadIdx.x]].y += tmp1.x*tmp2.y + tmp2.x*tmp1.y;
            }

            if (colIdx2[threadIdx.x] == rowIdx2[threadIdx.x]) {
                CInvMat[CMatIdx3[threadIdx.x]].x += 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].x = 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            } else {
                DInvMat[CMatIdx3[threadIdx.x]].x = 0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute (H^H*C^-1*H + I)^-1
        // use DInvMat for storing (H^H*C^-1*H + I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] == col_i) {
                    d_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+rowIdx2[threadIdx.x]+col_i*pDynDescr->nBsAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] != col_i) {
                    p_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i+colIdx2[threadIdx.x]*pDynDescr->nBsAnt];
                    p_inv_coeff = DInvMat[CMatIdxTemp3[threadIdx.x]+col_i+colIdx2[threadIdx.x]*pDynDescr->nBsAnt];
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    DInvMat[CMatIdx3[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        if (uIdx >= 0 && eIdx < pDynDescr->nUeAnt) {
            pDynDescr->postEqSinr[uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + rbgIdx*pDynDescr->nUeAnt + eIdx] = 
                                                            1.0/DInvMat[CMatIdxTemp3[threadIdx.x]+eIdx*pDynDescr->nBsAnt+eIdx].x - 1.0;
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxUeSinrCalPerRnd;
        }
        __syncthreads(); 
    }
 }

 static __global__ void multiCellSinrCalKernel_svdPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr)
 {
    // determine indices
    uint16_t rbgIdx               = floor(static_cast<float>(blockIdx.x)/pDynDescr->nCell);
    uint16_t cIdx                 = pDynDescr->cellId[blockIdx.x - rbgIdx*pDynDescr->nCell];
    uint16_t nBsAntSqrd           = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t nUeAntSqrd           = pDynDescr->nUeAnt*pDynDescr->nUeAnt;
    uint16_t nBsUeAntPrd          = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
    uint16_t nCellBsUeAntPrd      = pDynDescr->nCell*nBsUeAntPrd;
    uint32_t nUeCellBsUeAntPrd    = pDynDescr->nActiveUe*nCellBsUeAntPrd;
    uint16_t assocUeIdxInBlk      = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk  = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    // setup data arrays in shared memory
    __shared__ cuComplex  DInvMat[1024];
    __shared__ cuComplex  CMat[1024];
    __shared__ cuComplex  CInvMat[1024];
    __shared__ uint8_t    colIdx[1024];
    __shared__ uint8_t    rowIdx[1024];
    __shared__ uint16_t   CMatIdxTemp[1024];
    __shared__ uint16_t   CMatIdx[1024];
    __shared__ uint8_t    colIdx2[1024];
    __shared__ uint8_t    rowIdx2[1024];
    __shared__ uint16_t   CMatIdxTemp2[1024];
    __shared__ uint16_t   CMatIdx2[1024];
    __shared__ uint16_t   CMatIdxTemp3[1024];
    __shared__ uint16_t   CMatIdx3[1024];

    colIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nUeAnt);
    rowIdx[threadIdx.x] = eIdx - colIdx[threadIdx.x]*pDynDescr->nUeAnt;
    CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nUeAntSqrd;
    CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;

    colIdx2[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
    rowIdx2[threadIdx.x] = eIdx - colIdx2[threadIdx.x]*pDynDescr->nBsAnt;
    CMatIdxTemp2[threadIdx.x] = assocUeIdxInBlk*nBsUeAntPrd;
    CMatIdx2[threadIdx.x] = CMatIdxTemp2[threadIdx.x] + eIdx;

    CMatIdxTemp3[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
    CMatIdx3[threadIdx.x] = CMatIdxTemp3[threadIdx.x] + eIdx;

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
        for (int j = 0; j<pDynDescr->nActiveUe; j++) {
            if (pDynDescr->cellAssocActUe[cIdx*pDynDescr->nActiveUe + j]) {
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

        if (nAssocUeFound < pDynDescr->nMaxUeSinrCalPerRnd) {
            cnt = false;
        }

        // compute C matrix
        if (uIdx >= 0 && eIdx < nUeAntSqrd) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (uint16_t lIdx = 0; lIdx < pDynDescr->nCell; lIdx++) {
                uint16_t l = pDynDescr->cellId[lIdx];

                if (l == cIdx) {
                    continue;
                }
                uint32_t hInterfMatStart = rbgIdx*nUeCellBsUeAntPrd+ uIdx*nCellBsUeAntPrd + l*nBsUeAntPrd;
          
                for (int j = 0; j<pDynDescr->nBsAnt; j++) {
                    cuComplex tmp1 = pDynDescr->estH_fr_actUe[hInterfMatStart + rowIdx[threadIdx.x] + j*pDynDescr->nUeAnt];
                    cuComplex tmp2 = pDynDescr->estH_fr_actUe[hInterfMatStart + colIdx[threadIdx.x] + j*pDynDescr->nUeAnt];
                    CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                    CMat[CMatIdx[threadIdx.x]].y += tmp2.x*tmp1.y - tmp1.x*tmp2.y;
                }
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

        // compute inverse of C matrix
        // use CInvMat for storing inverse of C matrix
        for (int col_i=0; col_i < pDynDescr->nUeAnt; col_i++) {
            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nUeAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]+col_i*pDynDescr->nUeAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nUeAnt];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i+colIdx[threadIdx.x]*pDynDescr->nUeAnt];
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

        // compute H*V
        // reuse DInvMat for storing H*V
        uint32_t hMatStart = rbgIdx*nUeCellBsUeAntPrd + uIdx*nCellBsUeAntPrd + cIdx*nBsUeAntPrd;
        uint32_t vMatStart = uIdx*pDynDescr->nPrbGrp*nBsAntSqrd + rbgIdx*nBsAntSqrd;

        if (uIdx >= 0 && eIdx < nBsUeAntPrd) {
            DInvMat[CMatIdx2[threadIdx.x]].x = 0;
            DInvMat[CMatIdx2[threadIdx.x]].y = 0;

            for (int j = 0; j<pDynDescr->nBsAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_actUe[hMatStart + j*pDynDescr->nUeAnt + rowIdx[threadIdx.x]];
                cuComplex tmp2 = pDynDescr->prdMat_actUe[vMatStart + colIdx[threadIdx.x]*pDynDescr->nBsAnt +j];

                DInvMat[CMatIdx2[threadIdx.x]].x += tmp1.x*tmp2.x - tmp1.y*tmp2.y;
                DInvMat[CMatIdx2[threadIdx.x]].y += tmp1.x*tmp2.y + tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        // compute (H*V)^H*C^-1
        // reuse CMat for storing (H*V)^H*C^-1
        if (uIdx >= 0 && eIdx < nBsUeAntPrd) {
            CMat[CMatIdx2[threadIdx.x]].x = 0;
            CMat[CMatIdx2[threadIdx.x]].y = 0;
        
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = DInvMat[CMatIdxTemp2[threadIdx.x]+rowIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                cuComplex tmp2 = CInvMat[CMatIdxTemp[threadIdx.x]+colIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                CMat[CMatIdx2[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx2[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        // compute (H*V)^H*C^-1*(H*V) + I
        // reuse CInvMat for storing (H*V)^H*C^-1*(H*V) + I
        if (uIdx >= 0) {
            CInvMat[CMatIdx3[threadIdx.x]].x = 0;
            CInvMat[CMatIdx3[threadIdx.x]].y = 0;

            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = CMat[CMatIdxTemp2[threadIdx.x]+j*pDynDescr->nBsAnt+rowIdx2[threadIdx.x]];
                cuComplex tmp2 = DInvMat[CMatIdxTemp2[threadIdx.x]+colIdx2[threadIdx.x]*pDynDescr->nUeAnt+j];
                CInvMat[CMatIdx3[threadIdx.x]].x += tmp1.x*tmp2.x - tmp1.y*tmp2.y;
                CInvMat[CMatIdx3[threadIdx.x]].y += tmp1.x*tmp2.y + tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        if (uIdx >= 0) {
            if (colIdx2[threadIdx.x] == rowIdx2[threadIdx.x]) {
                CInvMat[CMatIdx3[threadIdx.x]].x += 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].x = 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            } else {
                DInvMat[CMatIdx3[threadIdx.x]].x = 0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute ((H*V)^H*C^-1*(H*V) + I)^-1
        // use DInvMat for storing ((H*V)^H*C^-1*(H*V) + I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] == col_i) {
                    d_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+rowIdx2[threadIdx.x]+col_i*pDynDescr->nBsAnt];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] != col_i) {
                    p_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i+colIdx2[threadIdx.x]*pDynDescr->nBsAnt];
                    p_inv_coeff = DInvMat[CMatIdxTemp3[threadIdx.x]+col_i+colIdx2[threadIdx.x]*pDynDescr->nBsAnt];
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    DInvMat[CMatIdx3[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        if (uIdx >= 0 && eIdx < pDynDescr->nUeAnt) {
            pDynDescr->postEqSinr[uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + rbgIdx*pDynDescr->nUeAnt + eIdx] = 
                                                            1.0/DInvMat[CMatIdxTemp3[threadIdx.x]+eIdx*pDynDescr->nBsAnt+eIdx].x - 1.0;
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxUeSinrCalPerRnd;
        }
        __syncthreads(); 
    }
 }

 static __global__ void multiCellSinrCalKernel_noPrdMmseIrc_rm(mcSinrCalDynDescr_t* pDynDescr)
 {
  // determine indices
  uint16_t rbgIdx               = floor(static_cast<float>(blockIdx.x)/pDynDescr->nCell);
  uint16_t cIdx                 = pDynDescr->cellId[blockIdx.x - rbgIdx*pDynDescr->nCell];
  uint16_t nBsAntSqrd           = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
  uint16_t nUeAntSqrd           = pDynDescr->nUeAnt*pDynDescr->nUeAnt;
  uint16_t nBsUeAntPrd          = pDynDescr->nBsAnt*pDynDescr->nUeAnt;
  uint16_t nCellBsUeAntPrd      = pDynDescr->nCell*nBsUeAntPrd;
  uint32_t nUeCellBsUeAntPrd    = pDynDescr->nActiveUe*nCellBsUeAntPrd;
  uint16_t assocUeIdxInBlk      = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
  uint16_t realAssocUeIdxInBlk  = assocUeIdxInBlk;
  int16_t  uIdx;
  uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

  // setup data arrays in shared memory
  __shared__ cuComplex  DInvMat[1024];
  __shared__ cuComplex  CMat[1024];
  __shared__ cuComplex  CInvMat[1024];
  __shared__ uint8_t    colIdx[1024];
  __shared__ uint8_t    rowIdx[1024];
  __shared__ uint16_t   CMatIdxTemp[1024];
  __shared__ uint16_t   CMatIdx[1024];
  __shared__ uint8_t    colIdx2[1024];
  __shared__ uint8_t    rowIdx2[1024];
  __shared__ uint16_t   CMatIdxTemp2[1024];
  __shared__ uint16_t   CMatIdx2[1024];
  __shared__ uint16_t   CMatIdxTemp3[1024];
  __shared__ uint16_t   CMatIdx3[1024];

  rowIdx[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nUeAnt);
  colIdx[threadIdx.x] = eIdx - rowIdx[threadIdx.x]*pDynDescr->nUeAnt;
  CMatIdxTemp[threadIdx.x] = assocUeIdxInBlk*nUeAntSqrd;
  CMatIdx[threadIdx.x] = CMatIdxTemp[threadIdx.x] + eIdx;

  rowIdx2[threadIdx.x] = floor(static_cast<float>(eIdx)/pDynDescr->nBsAnt);
  colIdx2[threadIdx.x] = eIdx - rowIdx2[threadIdx.x]*pDynDescr->nBsAnt;
  CMatIdxTemp2[threadIdx.x] = assocUeIdxInBlk*nBsUeAntPrd;
  CMatIdx2[threadIdx.x] = CMatIdxTemp2[threadIdx.x] + eIdx;

  CMatIdxTemp3[threadIdx.x] = assocUeIdxInBlk*nBsAntSqrd;
  CMatIdx3[threadIdx.x] = CMatIdxTemp3[threadIdx.x] + eIdx;

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
        for (int j = 0; j<pDynDescr->nActiveUe; j++) {
            if (pDynDescr->cellAssocActUe[cIdx*pDynDescr->nActiveUe + j]) {
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

        if (nAssocUeFound < pDynDescr->nMaxUeSinrCalPerRnd) {
            cnt = false;
        }

        // compute C matrix
        if (uIdx >= 0 && eIdx < nUeAntSqrd) {
            CMat[CMatIdx[threadIdx.x]].x = 0;
            CMat[CMatIdx[threadIdx.x]].y = 0;
            for (uint16_t lIdx = 0; lIdx < pDynDescr->nCell; lIdx++) {
                uint16_t l = pDynDescr->cellId[lIdx];

                if (l == cIdx) {
                    continue;
                }
                uint32_t hInterfMatStart = rbgIdx*nUeCellBsUeAntPrd+ uIdx*nCellBsUeAntPrd + l*nBsUeAntPrd;
          
                for (int j = 0; j<pDynDescr->nBsAnt; j++) {
                    cuComplex tmp1 = pDynDescr->estH_fr_actUe[hInterfMatStart + rowIdx[threadIdx.x]*pDynDescr->nBsAnt + j];
                    cuComplex tmp2 = pDynDescr->estH_fr_actUe[hInterfMatStart + colIdx[threadIdx.x]*pDynDescr->nBsAnt + j];
                    CMat[CMatIdx[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                    CMat[CMatIdx[threadIdx.x]].y += tmp2.x*tmp1.y - tmp1.x*tmp2.y;
                }
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

        // compute inverse of C matrix
        // use CInvMat for storing inverse of C matrix
        for (int col_i=0; col_i < pDynDescr->nUeAnt; col_i++) {
            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] == col_i) {
                    d_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nUeAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CMat[CMatIdx[threadIdx.x]];
                    c_inv_coeff = CInvMat[CMatIdx[threadIdx.x]];

                    CMat[CMatIdx[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CMat[CMatIdx[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CMat[CMatIdxTemp[threadIdx.x]+rowIdx[threadIdx.x]*pDynDescr->nUeAnt+col_i];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0 && eIdx < nUeAntSqrd) {
                if (rowIdx[threadIdx.x] != col_i) {
                    p_coeff = CMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nUeAnt+colIdx[threadIdx.x]];
                    p_inv_coeff = CInvMat[CMatIdxTemp[threadIdx.x]+col_i*pDynDescr->nUeAnt+colIdx[threadIdx.x]];
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

        // compute H^H*C^-1
        // reuse CMat for storing H^H*C^-1
        uint32_t hMatStart = rbgIdx*nUeCellBsUeAntPrd + uIdx*nCellBsUeAntPrd + cIdx*nBsUeAntPrd;
        if (uIdx >= 0 && eIdx < nBsUeAntPrd) {
            CMat[CMatIdx2[threadIdx.x]].x = 0;
            CMat[CMatIdx2[threadIdx.x]].y = 0;
        
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = pDynDescr->estH_fr_actUe[hMatStart+rowIdx2[threadIdx.x]+j*pDynDescr->nBsAnt];
                cuComplex tmp2 = CInvMat[CMatIdxTemp[threadIdx.x]+colIdx2[threadIdx.x]+j*pDynDescr->nUeAnt];
                CMat[CMatIdx2[threadIdx.x]].x += tmp1.x*tmp2.x + tmp1.y*tmp2.y;
                CMat[CMatIdx2[threadIdx.x]].y += tmp1.x*tmp2.y - tmp2.x*tmp1.y;
            }
        }
        __syncthreads();

        // compute H^H*C^-1*H + I
        // reuse CInvMat for storing H^H*C^-1*H + I
        if (uIdx >= 0) {
            CInvMat[CMatIdx3[threadIdx.x]].x = 0;
            CInvMat[CMatIdx3[threadIdx.x]].y = 0;
            for (int j = 0; j<pDynDescr->nUeAnt; j++) {
                cuComplex tmp1 = CMat[CMatIdxTemp2[threadIdx.x]+j+rowIdx2[threadIdx.x]*pDynDescr->nUeAnt];
                cuComplex tmp2 = pDynDescr->estH_fr_actUe[hMatStart+colIdx2[threadIdx.x]+j*pDynDescr->nBsAnt];
                CInvMat[CMatIdx3[threadIdx.x]].x += tmp1.x*tmp2.x - tmp1.y*tmp2.y;
                CInvMat[CMatIdx3[threadIdx.x]].y += tmp1.x*tmp2.y + tmp2.x*tmp1.y;
            }

            if (colIdx2[threadIdx.x] == rowIdx2[threadIdx.x]) {
                CInvMat[CMatIdx3[threadIdx.x]].x += 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].x = 1.0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            } else {
                DInvMat[CMatIdx3[threadIdx.x]].x = 0;
                DInvMat[CMatIdx3[threadIdx.x]].y = 0;
            }
        }
        __syncthreads();

        // compute (H^H*C^-1*H + I)^-1
        // use DInvMat for storing (H^H*C^-1*H + I)^-1
        for (int col_i=0; col_i < pDynDescr->nBsAnt; col_i++) {
            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] == col_i) {
                    d_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i*pDynDescr->nBsAnt+col_i];
                    d_multp = 1.0/(d_coeff.x*d_coeff.x + d_coeff.y*d_coeff.y);
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_coeff.x*d_coeff.x + c_coeff.y*d_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_coeff.y*d_coeff.x - c_coeff.x*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].x = d_multp * (c_inv_coeff.x*d_coeff.x + c_inv_coeff.y*d_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = d_multp * (c_inv_coeff.y*d_coeff.x - c_inv_coeff.x*d_coeff.y);
                } else {
                    l_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+rowIdx2[threadIdx.x]*pDynDescr->nBsAnt+col_i];
                }
            }
            __syncthreads(); 

            if (uIdx >= 0) {
                if (rowIdx2[threadIdx.x] != col_i) {
                    p_coeff = CInvMat[CMatIdxTemp3[threadIdx.x]+col_i*pDynDescr->nBsAnt+colIdx2[threadIdx.x]];
                    p_inv_coeff = DInvMat[CMatIdxTemp3[threadIdx.x]+col_i*pDynDescr->nBsAnt+colIdx2[threadIdx.x]];
                    c_coeff = CInvMat[CMatIdx3[threadIdx.x]];
                    c_inv_coeff = DInvMat[CMatIdx3[threadIdx.x]];

                    CInvMat[CMatIdx3[threadIdx.x]].x = c_coeff.x - (l_coeff.x*p_coeff.x - l_coeff.y*p_coeff.y);
                    CInvMat[CMatIdx3[threadIdx.x]].y = c_coeff.y - (l_coeff.x*p_coeff.y + l_coeff.y*p_coeff.x);
                    DInvMat[CMatIdx3[threadIdx.x]].x = c_inv_coeff.x - (l_coeff.x*p_inv_coeff.x - l_coeff.y*p_inv_coeff.y);
                    DInvMat[CMatIdx3[threadIdx.x]].y = c_inv_coeff.y - (l_coeff.x*p_inv_coeff.y + l_coeff.y*p_inv_coeff.x);  
                }
            }
            __syncthreads(); 
        }

        if (uIdx >= 0 && eIdx < pDynDescr->nUeAnt) {
            pDynDescr->postEqSinr[uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + rbgIdx*pDynDescr->nUeAnt + eIdx] = 
                                                            1.0/DInvMat[CMatIdxTemp3[threadIdx.x]+eIdx*pDynDescr->nBsAnt+eIdx].x - 1.0;
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxUeSinrCalPerRnd;
        }
        __syncthreads(); 
  }
 }

 //---------------------------- UL kernels ----------------------------------
 static __global__ void multiCellSinrCalKernel_svdPrdMmse_UL(mcSinrCalDynDescr_t* pDynDescr)
 {
    // determine indices
    uint16_t rbgIdx               = floor(static_cast<float>(blockIdx.x)/pDynDescr->nCell);
    uint16_t cIdx                 = pDynDescr->cellId[blockIdx.x - rbgIdx*pDynDescr->nCell];
    uint16_t nBsAntSqrd           = pDynDescr->nBsAnt*pDynDescr->nBsAnt;
    uint16_t assocUeIdxInBlk      = floor(static_cast<float>(threadIdx.x)/nBsAntSqrd);
    uint16_t realAssocUeIdxInBlk  = assocUeIdxInBlk;
    int16_t  uIdx;
    uint16_t eIdx = threadIdx.x - assocUeIdxInBlk*nBsAntSqrd; // entry index in matrix

    __shared__ int nAssocUeFound;
    bool cnt = true;

    while (cnt) {
        if (threadIdx.x == 0)
            nAssocUeFound = 0;
        __syncthreads();

        uIdx = -1;

        int16_t assocUeRank = -1;
        for (int j = 0; j<pDynDescr->nActiveUe; j++) {
            if (pDynDescr->cellAssocActUe[cIdx*pDynDescr->nActiveUe + j]) {
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

        if (nAssocUeFound < pDynDescr->nMaxUeSinrCalPerRnd) {
            cnt = false;
        }

        if (uIdx >= 0 && eIdx < pDynDescr->nUeAnt) {
            pDynDescr->postEqSinr[uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + rbgIdx*pDynDescr->nUeAnt + eIdx] = 
                    pow(pDynDescr->sinVal_actUe[uIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + rbgIdx*pDynDescr->nUeAnt + eIdx], 2.0)/pDynDescr->sigmaSqrd;
        }
        if (cnt){
            realAssocUeIdxInBlk += pDynDescr->nMaxUeSinrCalPerRnd;
        }
        __syncthreads(); 
    }
 }
 
 //---------------------------- wideband SINR calculation kernel ----------------------------------
 static __global__ void multiCellWideBandSinrCalKernel(mcSinrCalDynDescr_t* pDynDescr)
 {
    int globalUeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (globalUeIdx < pDynDescr->nActiveUe) {
        for (int j = 0; j < pDynDescr->nUeAnt; j++) {
            float avgSinr = 0;
            for (int prgIdx = 0; prgIdx < pDynDescr->nPrbGrp; prgIdx++) {
                avgSinr += pDynDescr->postEqSinr[globalUeIdx*pDynDescr->nPrbGrp*pDynDescr->nUeAnt + prgIdx*pDynDescr->nUeAnt + j];
            }
            pDynDescr->wbSinr[globalUeIdx*pDynDescr->nUeAnt + j] = avgSinr/pDynDescr->nPrbGrp;
        }
    }
 }

 void multiCellSinrCal::kernelSelect()
 {
    void* kernelFunc;
    if (DL == 1) { // for DL
        switch (precodingScheme) {
            case 0: // no precoding
                if (columnMajor) {
                    kernelFunc = reinterpret_cast<void*>(multiCellSinrCalKernel_noPrdMmseIrc_cm);
                } else {
                    kernelFunc = reinterpret_cast<void*>(multiCellSinrCalKernel_noPrdMmseIrc_rm);
                }
                      
                break;
            case 1: // SVD precoding
                if (columnMajor) {
                    kernelFunc = reinterpret_cast<void*>(multiCellSinrCalKernel_svdPrdMmseIrc_cm);
                } else {
                    printf("Error: SINR calculation for SVD precoding is not supported for row-major channel access.\n");
                    return;
                }

                break;
            default:
                printf("Error: SINR calculation is not supported for the given precoding scheme.\n");
                return;
                break;
        } 
    } else { // for UL
        if (precodingScheme == 0) {
            printf("Error: For UL, SINR calculation is only supported for SVD precoding.\n");
            return;
        }

        kernelFunc = reinterpret_cast<void*>(multiCellSinrCalKernel_svdPrdMmse_UL);
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

 void multiCellSinrCal::setup(cumacCellGrpPrms*           cellGrpPrms,
                              uint8_t                     in_columnMajor,
                              cudaStream_t                strm)
 {
    pCpuDynDesc->cellAssocActUe         = cellGrpPrms->cellAssocActUe;
    pCpuDynDesc->cellId                 = cellGrpPrms->cellId;  
    pCpuDynDesc->postEqSinr             = cellGrpPrms->postEqSinr;
    pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe; // total number of active UEs
    pCpuDynDesc->nCell                  = cellGrpPrms->nCell; // number of coordinated cells
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nBsAnt                 = cellGrpPrms->nBsAnt;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;
    pCpuDynDesc->sigmaSqrd              = cellGrpPrms->sigmaSqrd;
    pCpuDynDesc->nMaxUeSinrCalPerRnd    = floor(1024.0/(cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt));
    precodingScheme                     = cellGrpPrms->precodingScheme;
    columnMajor                         = in_columnMajor;
    pCpuDynDesc->sinVal_actUe           = cellGrpPrms->sinVal_actUe;
    pCpuDynDesc->srsEstChan             = nullptr; 
    pCpuDynDesc->prdMat_asim            = nullptr; 
    pCpuDynDesc->detMat_asim            = nullptr; 
    pCpuDynDesc->sinVal_asim            = nullptr; 

    if (DL == 1) { // for DL
        pCpuDynDesc->estH_fr_actUe      = cellGrpPrms->estH_fr_actUe;
        pCpuDynDesc->prdMat_actUe       = cellGrpPrms->prdMat_actUe;
        pCpuDynDesc->detMat_actUe       = cellGrpPrms->detMat_actUe;
    } else { // for UL
        pCpuDynDesc->estH_fr_actUe      = nullptr; 
        pCpuDynDesc->prdMat_actUe       = nullptr; 
        pCpuDynDesc->detMat_actUe       = nullptr; 
    }

    numThrdPerBlk = cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt*pCpuDynDesc->nMaxUeSinrCalPerRnd;
    numThrdBlk    = cellGrpPrms->nPrbGrp*cellGrpPrms->nCell;
    
    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcSinrCalDynDescr_t), cudaMemcpyHostToDevice, strm));

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect();

    pLaunchCfg->kernelArgs[0] = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 }

 void multiCellSinrCal::setup_wbSinr(cumacCellGrpPrms*           cellGrpPrms,
                                     cudaStream_t                strm)
 {
    pCpuDynDesc->postEqSinr             = cellGrpPrms->postEqSinr;
    pCpuDynDesc->wbSinr                 = cellGrpPrms->wbSinr;
    pCpuDynDesc->nActiveUe              = cellGrpPrms->nActiveUe;
    pCpuDynDesc->nPrbGrp                = cellGrpPrms->nPrbGrp;
    pCpuDynDesc->nUeAnt                 = cellGrpPrms->nUeAnt;

    numThrdPerBlk = 1024;
    numThrdBlk    = static_cast<uint16_t>(ceil(static_cast<float>(cellGrpPrms->nActiveUe)/static_cast<float>(numThrdPerBlk)));

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(mcSinrCalDynDescr_t), cudaMemcpyHostToDevice, strm));

    void* kernelFunc = reinterpret_cast<void*>(multiCellWideBandSinrCalKernel);

    CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));
  
    // launch geometry
    gridDim  = {numThrdBlk, 1, 1};
    blockDim = {numThrdPerBlk, 1, 1};
  
    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;
  
    kernelNodeParamsDriver.blockDimX                = blockDim.x;
    kernelNodeParamsDriver.blockDimY                = blockDim.y;
    kernelNodeParamsDriver.blockDimZ                = blockDim.z;
  
    kernelNodeParamsDriver.gridDimX                 = gridDim.x;
    kernelNodeParamsDriver.gridDimY                 = gridDim.y;
    kernelNodeParamsDriver.gridDimZ                 = gridDim.z;
  
    kernelNodeParamsDriver.extra                    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes           = 0;
 } 
 
 void multiCellSinrCal::run(cudaStream_t strm)
 {
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

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

 void multiCellSinrCal::debugLog()
 {
    float* postEqSinr = new float[pCpuDynDesc->nActiveUe*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt];
    CUDA_CHECK_ERR(cudaMemcpy(postEqSinr, pCpuDynDesc->postEqSinr, pCpuDynDesc->nActiveUe*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream file;
    file.open("sinr.txt", std::fstream::out);

    file << "postEqSinr = [";
    for (int idx = 0; idx < pCpuDynDesc->nActiveUe*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt - 1; idx++) {
        file << 10.0*log10(postEqSinr[idx]) << " ";
    }
    file << 10.0*log10(postEqSinr[pCpuDynDesc->nActiveUe*pCpuDynDesc->nPrbGrp*pCpuDynDesc->nUeAnt - 1]) << "];\n";

    file.close();

    delete postEqSinr;
 }
}