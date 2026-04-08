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

#include "api.h"
#include "cumac.h"

using namespace std;

#define CUSOLVER_CHECK(call)                                                   \
{                                                                              \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSolver error in file '%s' in line %i.\n",           \
            __FILE__, __LINE__);                                           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// #define SVD_RESULTS_PRINT_
#define SVD_TIME_MEASURE_
#define SVD_VALIDATE_

#define svdGpuDeviceIdx           0 // index of GPU devive to use

#define svdNumCoorCellConst       16
#define svdNumBsAntConst          16
#define svdNumUeAntConst          4
#define svdNumPrbGrpsConst        25
#define svdNumUePerCellConst      10
#define svdTotNumUesConst         svdNumCoorCellConst*svdNumUePerCellConst
#define svdResultErrTolerance     1e-4

int main() {

    // set GPU device
    unsigned my_dev = svdGpuDeviceIdx;
    CUDA_CHECK_ERR(cudaSetDevice(my_dev));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);

    // create CUDA stream
    cudaStream_t cuStrmMain;
    CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

#ifdef SVD_TIME_MEASURE_
    cudaEvent_t start, stop;
    CUDA_CHECK_ERR(cudaEventCreate(&start));
    CUDA_CHECK_ERR(cudaEventCreate(&stop));
    float milliseconds = 0;
#endif

    const int M   = svdNumUeAntConst;
    const int N   = svdNumBsAntConst; 
    const int lda = M;
    const int ldu = M;
    const int ldv = N;
    const int nChanMat = svdNumPrbGrpsConst*svdNumCoorCellConst*svdNumUePerCellConst; // total number of channel matrices
    // const int nChanMat = svdNumPrbGrpsConst*svdNumCoorCellConst*svdTotNumUesConst;

    // --- Setting the host random channel matrices
    cuComplex* h_A = (cuComplex*)malloc(lda * N * nChanMat * sizeof(cuComplex));
    for (int k = 0; k < nChanMat; k++)
        for (int i = 0; i < lda; i++){
            for (int j = 0; j < N; j++){
                h_A[k * lda * N + j * lda + i].x = distribution(generator);
                h_A[k * lda * N + j * lda + i].y = distribution(generator);
            }
        }

    // --- Setting the device matrix and moving the host matrix to the device
    cuComplex* d_A;
    CUDA_CHECK_ERR(cudaMalloc(&d_A, lda * N * nChanMat * sizeof(cuComplex)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(d_A, h_A, lda * N * nChanMat * sizeof(cuComplex), cudaMemcpyHostToDevice, cuStrmMain));
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // --- host side SVD results space
    int*       devInfo_h    = (int*)malloc(nChanMat * sizeof(int)); /* host copy of error devInfo */
    float*     h_S          = (float*)malloc((M<N?M:N) * nChanMat * sizeof(float));
    cuComplex* h_U          = (cuComplex*)malloc(ldu * ldu * nChanMat * sizeof(cuComplex));
    cuComplex* h_V          = (cuComplex*)malloc(ldv * ldv * nChanMat * sizeof(cuComplex));

    // --- device side SVD workspace and matrices
    
    int *devInfo;       
    float *d_S;         
    cuComplex* d_U;   
    cuComplex* d_V;
    CUDA_CHECK_ERR(cudaMalloc(&devInfo, nChanMat * sizeof(int)));
    CUDA_CHECK_ERR(cudaMalloc(&d_S, (M<N?M:N) * nChanMat * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc(&d_U, ldu * ldu * nChanMat * sizeof(cuComplex)));
    CUDA_CHECK_ERR(cudaMalloc(&d_V, ldv * ldv * nChanMat * sizeof(cuComplex)));

    // --- CUDA solver initialization
    // --- Parameters configuration of Jacobi-based SVD
    const double tol            = 1.e-7;
    const int    maxSweeps      = 15;
    const int    econ           = 0; // --- econ = 1 for economy size 

    cusolverDnHandle_t solver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
    CUSOLVER_CHECK(cusolverDnSetStream(solver_handle, cuStrmMain));
    
    // --- Configuration of gesvdj
    gesvdjInfo_t gesvdj_params;
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    // --- Set the computation tolerance, since the default tolerance is machine precision
    // cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

    // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

    int work_size = 0;
    cuComplex *d_work; /* devie workspace for gesvdj */
    // --- Query the SVD workspace
    CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
        solver_handle,
        CUSOLVER_EIG_MODE_VECTOR,                   // --- Compute the singular vectors or not
        M,                                          // --- Nubmer of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        d_A,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        ldu,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M,N) if econ = 1
        ldv,                                        // --- Leading dimension of V, ldv >= max(1, N)
        &work_size,
        gesvdj_params,
        nChanMat));

    CUDA_CHECK_ERR(cudaMalloc(&d_work, work_size*sizeof(cuComplex)));

    // --- Compute SVD
#ifdef SVD_TIME_MEASURE_
    CUDA_CHECK_ERR(cudaEventRecord(start, cuStrmMain));
#endif
    CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
        solver_handle,
        CUSOLVER_EIG_MODE_VECTOR,                   // --- Compute the singular vectors or not
        M,                                          // --- Number of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        d_A,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        ldu,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M, N) if econ = 1
        ldv,                                        // --- Leading dimension of V, ldv >= max(1, N)
        d_work,
        work_size,
        devInfo,
        gesvdj_params,
        nChanMat));
#ifdef SVD_TIME_MEASURE_
    CUDA_CHECK_ERR(cudaEventRecord(stop, cuStrmMain));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("milliseconds = %f\n", milliseconds);
#endif

    CUDA_CHECK_ERR(cudaMemcpyAsync(devInfo_h, devInfo, sizeof(int) * nChanMat, cudaMemcpyDeviceToHost, cuStrmMain));
    CUDA_CHECK_ERR(cudaMemcpyAsync(h_S, d_S, sizeof(float) * (M<N?M:N) * nChanMat, cudaMemcpyDeviceToHost, cuStrmMain));
    CUDA_CHECK_ERR(cudaMemcpyAsync(h_U, d_U, sizeof(cuComplex) * ldu * ldu * nChanMat, cudaMemcpyDeviceToHost, cuStrmMain));
    CUDA_CHECK_ERR(cudaMemcpyAsync(h_V, d_V, sizeof(cuComplex) * ldv * ldv * nChanMat, cudaMemcpyDeviceToHost, cuStrmMain));
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    if (0 == devInfo_h[0]){
        // printf("gesvdj converges \n");
    }
    else if (0 > devInfo_h[0]){
        printf("%d-th parameter is wrong \n", -devInfo_h[0]);
        exit(1);
    }
    else{
        printf("WARNING: devInfo_h[0] = %d : gesvdj does not converge \n", devInfo_h[0]);
        exit(1);
    }

#ifdef SVD_VALIDATE_
    // when M != N, data in d_A seems to be corrupted
    //cudaMemcpyAsync(h_A, d_A, lda * N * nChanMat * sizeof(cuComplex), cudaMemcpyDeviceToHost, cuStrmMain);
    //cudaStreamSynchronize(cuStrmMain);

    bool same = true;
    for (int matIdx = 0; matIdx < nChanMat; matIdx++) {
        cuComplex* A_SVD = (cuComplex *)malloc(lda * N * sizeof(cuComplex));
        for (int rIdx = 0; rIdx < lda; rIdx++) {
            for (int cIdx = 0; cIdx < N; cIdx++) {
                A_SVD[rIdx + cIdx*lda].x = 0;
                A_SVD[rIdx + cIdx*lda].y = 0;
                for (int i = 0; i < (M<N?M:N); i++) {
                    A_SVD[rIdx + cIdx*lda].x += h_S[matIdx*(M<N?M:N) + i]*(h_U[matIdx*ldu*ldu + rIdx + i*ldu].x*h_V[matIdx*ldv*ldv + cIdx + i*ldv].x + h_U[matIdx*ldu*ldu + rIdx + i*ldu].y*h_V[matIdx*ldv*ldv + cIdx + i*ldv].y);
                    A_SVD[rIdx + cIdx*lda].y += h_S[matIdx*(M<N?M:N) + i]*(-h_U[matIdx*ldu*ldu + rIdx + i*ldu].x*h_V[matIdx*ldv*ldv + cIdx + i*ldv].y + h_V[matIdx*ldv*ldv + cIdx + i*ldv].x*h_U[matIdx*ldu*ldu + rIdx + i*ldu].y);
                }

                if (abs(A_SVD[rIdx + cIdx*lda].x-h_A[lda*N*matIdx + rIdx + cIdx*lda].x) > svdResultErrTolerance || abs(A_SVD[rIdx + cIdx*lda].y-h_A[lda*N*matIdx + rIdx + cIdx*lda].y) > svdResultErrTolerance) {
                    same = false;
                    printf("want: x = %f, y = %f\n", h_A[lda*N*matIdx + rIdx + cIdx*lda].x, h_A[lda*N*matIdx + rIdx + cIdx*lda].y);
                    printf("got: x = %f, y = %f\n", A_SVD[rIdx + cIdx*lda].x, A_SVD[rIdx + cIdx*lda].y);
                    printf("delta_x = %f, delta_y = %f\n", h_A[lda*N*matIdx + rIdx + cIdx*lda].x-A_SVD[rIdx + cIdx*lda].x, h_A[lda*N*matIdx + rIdx + cIdx*lda].y-A_SVD[rIdx + cIdx*lda].y);
                    break;
                }
            }
            if (!same) {
                break;
            }
        }

        if (!same) {
            printf("Error: SVD results do not match\n");
            break;
        }
    }
    if (same) {
        printf("Success: all SVD results match\n");
    }
     
#endif

#ifdef SVD_RESULTS_PRINT_
    printf("SINGULAR VALUES \n");
    printf("_______________ \n");
    int nPrint = 5;
    for (int k = 0; k < nPrint; k++) {
        for (int p = 0; p < N; p++)
            printf("Matrix nr. %d; SV nr. %d; Value = %f\n", k, p, h_S[k * N + p]);
        printf("\n");
    }
/*
    printf("SINGULAR VECTORS U \n");
    printf("__________________ \n");
    for (int k = 0; k < nPrint; k++) {
        for (int q = 0; q < (1 - econ) * M + econ * std::min(M, N); q++)
            for (int p = 0; p < M; p++)
                printf("Matrix nr. %d; U nr. %d; Value = %f\n", k, p, h_U[((1 - econ) * M + econ * std::min(M, N)) * M * k + q * M + p]);
        printf("\n");
    }

    printf("SINGULAR VECTORS V \n");
    printf("__________________ \n");
    for (int k = 0; k < nPrint; k++) {
        for (int q = 0; q < (1 - econ) * N + econ * std::min(M, N); q++)
            for (int p = 0; p < N; p++)
                printf("Matrix nr. %d; V nr. %d; Value = %f\n", k, p, h_V[((1 - econ) * N + econ * std::min(M, N)) * N * k + q * N + p]);
        printf("\n");
    }
*/
#endif

    

    // --- Free resources
    if (d_work) CUDA_CHECK_ERR(cudaFree(d_work));
    if (d_A) CUDA_CHECK_ERR(cudaFree(d_A));
    if (d_S) CUDA_CHECK_ERR(cudaFree(d_S));
    if (d_U) CUDA_CHECK_ERR(cudaFree(d_U));
    if (d_V) CUDA_CHECK_ERR(cudaFree(d_V));
    if (devInfo) CUDA_CHECK_ERR(cudaFree(devInfo));
    
    if (solver_handle) CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    if (gesvdj_params) CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));

    CUDA_CHECK_ERR(cudaDeviceReset());

    return 0;
}