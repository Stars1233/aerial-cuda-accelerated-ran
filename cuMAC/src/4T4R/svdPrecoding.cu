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

 svdPrecoding::svdPrecoding(cumacCellGrpPrms* cellGrpPrms)
 {
    tol            = 1.e-7;  // TODO: hardcoded parameter
    maxSweeps      = 15;     // TODO: hardcoded parameter
    econ           = 0;
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
    // --- Set the computation tolerance, since the default tolerance is machine precision
    // cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
    // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

    DL  = cellGrpPrms->dlSchInd;
 }

 svdPrecoding::~svdPrecoding()
 {
    if (solver_handle) CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    if (gesvdj_params) CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
 }

 void svdPrecoding::destroy()
 {
    if (d_work)  CUDA_CHECK_ERR(cudaFree(d_work));
    if (devInfo) CUDA_CHECK_ERR(cudaFree(devInfo));
    if (d_hCpy)  CUDA_CHECK_ERR(cudaFree(d_hCpy));
 }

 void svdPrecoding::setup(cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm)
 {
   if (DL == 1) { // DL
      M   = cellGrpPrms->nUeAnt;
      N   = cellGrpPrms->nBsAnt;
   } else { // UL
      M   = cellGrpPrms->nBsAnt;
      N   = cellGrpPrms->nUeAnt;
   }
   
   lda = M;
   ldu = M;
   ldv = N;
   nChanMat = cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp;

   CUDA_CHECK_ERR(cudaMalloc(&d_hCpy, M * N * nChanMat * sizeof(cuComplex)));
   CUDA_CHECK_ERR(cudaMalloc(&devInfo, nChanMat * sizeof(int)));
   
   CUSOLVER_CHECK(cusolverDnSetStream(solver_handle, strm));

   CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
         solver_handle,
         CUSOLVER_EIG_MODE_VECTOR,                   // --- Compute the singular vectors or not
         M,                                          // --- Nubmer of rows of A, 0 <= M
         N,                                          // --- Number of columns of A, 0 <= N 
         cellGrpPrms->estH_fr_actUe_prd,             // --- M x N
         lda,                                        // --- Leading dimension of A
         cellGrpPrms->sinVal_actUe,                  // --- Square matrix of size min(M, N) x min(M, N)
         cellGrpPrms->detMat_actUe,                  // --- M x M if econ = 0, M x min(M, N) if econ = 1
         ldu,                                        // --- Leading dimension of U, ldu >= max(1, M)
         cellGrpPrms->prdMat_actUe,                  // --- N x N if econ = 0, N x min(M,N) if econ = 1
         ldv,                                        // --- Leading dimension of V, ldv >= max(1, N)
         &work_size,
         gesvdj_params,
         nChanMat));

   CUDA_CHECK_ERR(cudaMalloc(&d_work, work_size*sizeof(cuComplex)));
 }


 void svdPrecoding::run(cumacCellGrpPrms* cellGrpPrms)
 {
   CUDA_CHECK_ERR(cudaMemcpy(d_hCpy, cellGrpPrms->estH_fr_actUe_prd, M * N * nChanMat * sizeof(cuComplex), cudaMemcpyDeviceToDevice));

   CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
         solver_handle,
         CUSOLVER_EIG_MODE_VECTOR,                   // --- Compute the singular vectors or not
         M,                                          // --- Number of rows of A, 0 <= M
         N,                                          // --- Number of columns of A, 0 <= N 
         d_hCpy,                                     // --- M x N
         lda,                                        // --- Leading dimension of A
         cellGrpPrms->sinVal_actUe,                  // --- Square matrix of size min(M, N) x min(M, N)
         cellGrpPrms->detMat_actUe,                  // --- M x M if econ = 0, M x min(M, N) if econ = 1
         ldu,                                        // --- Leading dimension of U, ldu >= max(1, M)
         cellGrpPrms->prdMat_actUe,                  // --- N x N if econ = 0, N x min(M, N) if econ = 1
         ldv,                                        // --- Leading dimension of V, ldv >= max(1, N)
         d_work,
         work_size,
         devInfo,
         gesvdj_params,
         nChanMat));
 }
}