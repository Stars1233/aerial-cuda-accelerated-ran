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

#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "../../src/api.h"
#include <cuda_runtime.h>
#include <mma.h>

#define row_d    4096
#define mid_d    4096
#define col_d    4096  

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}


// All the data in the matrices are stored in a column-major order,
// which is the consistent with most of the cuBLAS GEMM APIs.
// For matrix A of shape M x N, the leading dimension is M.
// For matrix A that is transposed and is of shape N x M,
// the leading dimension is N.
// Matrix A: M x K, or K x N (if transposed).
// Matrix B: K x M, or M x K (if transposed).
// Matrix C: M x N.
// WMMA_FRAG_LAYOUT_A: nvcuda::wmma::row_major if A is
// transposed, otherwise nvcuda::wmma::col_major.
// WMMA_FRAG_LAYOUT_B: nvcuda::wmma::row_major if B is
// transposed, otherwise nvcuda::wmma::col_major.

template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_complex_a_col_major_b_col_major(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose, bool is_B_transpose)
{
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_A>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_B>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        c_frag{};

    // Cr = Ar*Br - Ai*Bi
    // Ci = Ar*Bi + Ai*Br

    // calculate Ar*Br
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
            : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
            : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
            : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
            : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const* matrix_mma_a_mptr{A_real + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B_real + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n) {
        T2* matrix_mma_c_mptr{C_real + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = acc_frag.x[i];
        }
    }

    // calculate Ai*Bi
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
            : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
            : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
            : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
            : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const* matrix_mma_a_mptr{A_imag + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B_imag + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n) {
        T2* matrix_mma_c_mptr{C_real + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] =  c_frag.x[i] - acc_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }

    // calculate Ar*Bi
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
            : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
            : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
            : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
            : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const* matrix_mma_a_mptr{A_real + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B_imag + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n) {
        T2* matrix_mma_c_mptr{C_imag + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = acc_frag.x[i];
        }
    }

    // calculate Ai*Br
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
            : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
            : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
            : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
            : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const* matrix_mma_a_mptr{A_imag + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B_real + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n) {
        T2* matrix_mma_c_mptr{C_imag + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = c_frag.x[i] + acc_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}


template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
    T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose,
    bool is_B_transpose, float alpha, float beta)
{
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_A>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_B>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        c_frag{};

    // Make sure the accumulator starts from 0.
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        // Determine the first element of the mma matrices on the linear memory.
        // Matrix A mma matrix
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
                                                           : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
                                                           : ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
                                                           : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
                                                           : warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            // Determine the memory address of the first element of the mma
            // matrices. Notice that all the matrices are assumed to be
            // column-major. Therefore, the indexing is different from the
            // row-major indexing that we commonly see.
            T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha.
    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
    {
        T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);
        // Let the compiler figure out how to do the elementwise operation.
        // Such elementwise operation can be scaling, accumulation,
        // quantization, etc.
        // https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#id40
        // Be careful when dealing with the integer types.
        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}

template <typename T1, typename T2>
void launch_wmma_mm_complex(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t m, uint32_t n,
    uint32_t k, bool is_A_transpose, bool is_B_transpose,
    cudaStream_t stream)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // Block size of 128x4 means we have 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    // Each block has 4x4 warps, totalling 4x4x32 threads.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up.
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_complex_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A_real, A_imag, B_real, B_imag, C_real, C_imag, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose);
    } else if ((is_A_transpose) && (!is_B_transpose)) { // C = A^T * B
        wmma_gemm_complex_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A_real, A_imag, B_real, B_imag, C_real, C_imag, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose);
    } else if ((!is_A_transpose) && (is_B_transpose)) { // C = A * B^T
        wmma_gemm_complex_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A_real, A_imag, B_real, B_imag, C_real, C_imag, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose);
    } else { // C = A^T * B^T
        wmma_gemm_complex_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A_real, A_imag, B_real, B_imag, C_real, C_imag, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose);
    }
    CHECK_LAST_CUDA_ERROR();
}

template <typename T1, typename T2>
void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // Block size of 128x4 means we have 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    // Each block has 4x4 warps, totalling 4x4x32 threads.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up.
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B^T
    else
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    CHECK_LAST_CUDA_ERROR();
}

template <typename T1, typename T2>
void mm_complex_a_col_major_b_col_major(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag,  uint32_t m,
    uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose, bool is_B_transpose)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            // Cr = Ar*Br - Ai*Bi
            // Compute Ar*Br[mi, ni] for determining Cr[mi, ni]
            T2 accum_ArBr{0};
            // compute -Ai*Bi[mi, ni] for determining Cr[mi, ni]
            T2 accum_negAiBi{0};

            // Ci = Ar*Bi + Ai*Br
            // compute Ar*Bi[mi, ni] for dermining Ci[mi, ni]
            T2 accum_ArBi{0};
            // compute Ai*Br[mi, ni] for dermining Ci[mi, ni]
            T2 accum_AiBr{0};

            if ((!is_A_transpose) && (!is_B_transpose)) {
                for (uint32_t ki{0}; ki < k; ++ki) {
                    accum_ArBr += A_real[ki * lda + mi] * B_real[ni * ldb + ki];
                    accum_negAiBi += -A_imag[ki * lda + mi] * B_imag[ni * ldb + ki];
                    accum_ArBi += A_real[ki * lda + mi] * B_imag[ni * ldb + ki];
                    accum_AiBr += A_imag[ki * lda + mi] * B_real[ni * ldb + ki];
                }
            } else if ((is_A_transpose) && (!is_B_transpose)) {
                for (uint32_t ki{0}; ki < k; ++ki) {
                    accum_ArBr += A_real[mi * lda + ki] * B_real[ni * ldb + ki];
                    accum_negAiBi += -A_imag[mi * lda + ki] * B_imag[ni * ldb + ki];
                    accum_ArBi += A_real[mi * lda + ki] * B_imag[ni * ldb + ki];
                    accum_AiBr += A_imag[mi * lda + ki] * B_real[ni * ldb + ki];
                }
            } else if ((!is_A_transpose) && (is_B_transpose)) {
                for (uint32_t ki{0}; ki < k; ++ki) {
                    accum_ArBr += A_real[ki * lda + mi] * B_real[ki * ldb + ni];
                    accum_negAiBi += -A_imag[ki * lda + mi] * B_imag[ki * ldb + ni];
                    accum_ArBi += A_real[ki * lda + mi] * B_imag[ki * ldb + ni];
                    accum_AiBr += A_imag[ki * lda + mi] * B_real[ki * ldb + ni];
                }
            } else {
                for (uint32_t ki{0}; ki < k; ++ki) {
                    accum_ArBr += A_real[mi * lda + ki] * B_real[ki * ldb + ni];
                    accum_negAiBi += -A_imag[mi * lda + ki] * B_imag[ki * ldb + ni];
                    accum_ArBi += A_real[mi * lda + ki] * B_imag[ki * ldb + ni];
                    accum_AiBr += A_imag[mi * lda + ki] * B_real[ki * ldb + ni];
                }
            }

            C_real[ni * ldc + mi] = accum_ArBr + accum_negAiBi;
            C_imag[ni * ldc + mi] = accum_ArBi + accum_AiBr;
        }
    }
}    

// A and B are column-major matrices.
template <typename T1, typename T2>
void mm_a_col_major_b_col_major(T1 const* A, T1 const* B, T2* C, uint32_t m,
                                uint32_t n, uint32_t k, uint32_t lda,
                                uint32_t ldb, uint32_t ldc, bool is_A_transpose,
                                bool is_B_transpose)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            // Compute C[mi, ni]
            T2 accum{0};
            // C = A * B
            if ((!is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ki, ni]
                    accum += A[ki * lda + mi] * B[ni * ldb + ki];
                }
            }
            // C = A^T * B
            else if ((is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ki, ni]
                    accum += A[mi * lda + ki] * B[ni * ldb + ki];
                }
            }
            // C = A * B^T
            else if ((!is_A_transpose) && (is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ni, ki]
                    accum += A[ki * lda + mi] * B[ki * ldb + ni];
                }
            }
            // C = A^T * B^T
            else
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ni, ki]
                    accum += A[mi * lda + ki] * B[ki * ldb + ni];
                }
            }
            C[ni * ldc + mi] = accum;
        }
    }
}

template <typename T1, typename T2>
void launch_mm_complex(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t m, uint32_t n,
    uint32_t k, bool is_A_transpose, bool is_B_transpose)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    mm_complex_a_col_major_b_col_major(A_real, A_imag, B_real, B_imag, C_real, C_imag, m, n, k, lda, ldb, ldc, is_A_transpose,
        is_B_transpose);
}


template <typename T1, typename T2>
void launch_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
               uint32_t k, bool is_A_transpose, bool is_B_transpose)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    mm_a_col_major_b_col_major(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose,
                               is_B_transpose);
}

template <typename T1, typename T2>
void partitionRealImag(T1 const* arr, T2* arrReal, T2* arrImag, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        arrReal[i] = arr[i].x;
        arrImag[i] = arr[i].y;
    }
}

void fill_random_float2_values(cuComplex* arr, size_t n,
                               std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i].x = uniform_dist(e);
        arr[i].y = uniform_dist(e);
    }
}

void fill_random_float_values(float* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int8_values(int8_t* arr, size_t n,
                             std::default_random_engine& e)
{
    std::uniform_int_distribution<int8_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void fill_random_int32_values(int32_t* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_int_distribution<int32_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void float2ToHalf2(__half2* half2_arr, cuComplex const* float2_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half2_arr[i].x = __float2half(float2_arr[i].x);
        half2_arr[i].y = __float2half(float2_arr[i].y);
    }
}

void float2half(__half* half_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
    }
}

template <typename T>
float get_avg_abs_diff_ratio_complex(T const* arr_1_real, T const* arr_1_imag, T const* arr_2_real, T const* arr_2_imag, size_t n)
{
    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < n; ++i)
    {
        sum_abs_diff_ratio += std::abs(static_cast<float>(arr_1_real[i]) -
                                       static_cast<float>(arr_2_real[i])) /
                              std::abs(static_cast<float>(arr_1_real[i]) +
                                       static_cast<float>(arr_2_real[i]));
        sum_abs_diff_ratio += std::abs(static_cast<float>(arr_1_imag[i]) -
                                       static_cast<float>(arr_2_imag[i])) /
                              std::abs(static_cast<float>(arr_1_imag[i]) +
                                       static_cast<float>(arr_2_imag[i]));                               
    }
    return sum_abs_diff_ratio / n / 2.0;
}

template <typename T>
float get_avg_abs_diff_ratio(T const* arr_1, T const* arr_2, size_t n)
{
    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < n; ++i)
    {
        sum_abs_diff_ratio += std::abs(static_cast<float>(arr_1[i]) -
                                       static_cast<float>(arr_2[i])) /
                              std::abs(static_cast<float>(arr_1[i]) +
                                       static_cast<float>(arr_2[i]));
    }
    return sum_abs_diff_ratio / n;
}

template <typename T>
bool array_equal(T const* arr_1, T const* arr_2, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (arr_1[i] != arr_2[i])
        {
            return false;
        }
    }
    return true;
}

void print_test_header(bool is_A_transpose, bool is_B_transpose)
{
    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A * B" << std::endl;
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A^T * B" << std::endl;
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        std::cout << "C = A * B^T" << std::endl;
    }
    // C = A^T * B^T
    else
    {
        std::cout << "C = A^T * B^T" << std::endl;
    }
}

int main()
{
    cudaSetDevice(7);

    constexpr int num_repeats{1};
    constexpr int num_warmups{1};

    uint32_t const matrix_size_m{row_d};
    uint32_t const matrix_size_n{col_d};
    uint32_t const matrix_size_k{mid_d};
    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
 
    std::vector<cuComplex> matrix_a_float2(matrix_size_m * matrix_size_k);
    std::vector<cuComplex> matrix_b_float2(matrix_size_k * matrix_size_n);
    std::vector<__half2>   matrix_a_half2(matrix_size_m * matrix_size_k);
    std::vector<__half2>   matrix_b_half2(matrix_size_k * matrix_size_n);
    std::vector<cuComplex> matrix_c_float2(matrix_size_m * matrix_size_n);
    std::vector<cuComplex> matrix_c_float2_ref(matrix_size_m * matrix_size_n);

    // partitioned real and image matrices
    std::vector<float>     matrix_a_float_real(matrix_size_m * matrix_size_k);
    std::vector<float>     matrix_a_float_imag(matrix_size_m * matrix_size_k);
    std::vector<float>     matrix_b_float_real(matrix_size_k * matrix_size_n);
    std::vector<float>     matrix_b_float_imag(matrix_size_k * matrix_size_n);
    std::vector<__half>    matrix_a_half_real(matrix_size_m * matrix_size_k);
    std::vector<__half>    matrix_a_half_imag(matrix_size_m * matrix_size_k);
    std::vector<__half>    matrix_b_half_real(matrix_size_k * matrix_size_n);
    std::vector<__half>    matrix_b_half_imag(matrix_size_k * matrix_size_n);
    std::vector<float>     matrix_c_float_real(matrix_size_m * matrix_size_n);
    std::vector<float>     matrix_c_float_imag(matrix_size_m * matrix_size_n);
    std::vector<float>     matrix_c_float_ref_real(matrix_size_m * matrix_size_n);
    std::vector<float>     matrix_c_float_ref_imag(matrix_size_m * matrix_size_n);


    cuComplex* h_matrix_a_float2{matrix_a_float2.data()};
    cuComplex* h_matrix_b_float2{matrix_b_float2.data()};
    __half2*   h_matrix_a_half2{matrix_a_half2.data()};
    __half2*   h_matrix_b_half2{matrix_b_half2.data()};
    cuComplex* h_matrix_c_float2{matrix_c_float2.data()};
    cuComplex* h_matrix_c_float2_ref{matrix_c_float2_ref.data()};
    float*     h_matrix_a_float_real{matrix_a_float_real.data()};
    float*     h_matrix_a_float_imag{matrix_a_float_imag.data()};
    float*     h_matrix_b_float_real{matrix_b_float_real.data()};
    float*     h_matrix_b_float_imag{matrix_b_float_imag.data()};
    __half*    h_matrix_a_half_real{matrix_a_half_real.data()};
    __half*    h_matrix_a_half_imag{matrix_a_half_imag.data()};
    __half*    h_matrix_b_half_real{matrix_b_half_real.data()};
    __half*    h_matrix_b_half_imag{matrix_b_half_imag.data()};
    float*     h_matrix_c_float_real{matrix_c_float_real.data()};
    float*     h_matrix_c_float_imag{matrix_c_float_imag.data()};
    float*     h_matrix_c_float_ref_real{matrix_c_float_ref_real.data()};
    float*     h_matrix_c_float_ref_imag{matrix_c_float_ref_imag.data()};


    fill_random_float2_values(h_matrix_a_float2, matrix_a_float2.size(),
                              random_engine);
    fill_random_float2_values(h_matrix_b_float2, matrix_b_float2.size(),
                              random_engine);
    fill_random_float2_values(h_matrix_c_float2, matrix_c_float2.size(),
                              random_engine);    
    fill_random_float2_values(h_matrix_c_float2_ref, matrix_c_float2_ref.size(),
                              random_engine);        
    
    float2ToHalf2(h_matrix_a_half2, h_matrix_a_float2, matrix_a_float2.size());                          
    float2ToHalf2(h_matrix_b_half2, h_matrix_b_float2, matrix_b_float2.size());

    partitionRealImag(h_matrix_a_float2, h_matrix_a_float_real, h_matrix_a_float_imag, matrix_a_float2.size());
    partitionRealImag(h_matrix_b_float2, h_matrix_b_float_real, h_matrix_b_float_imag, matrix_b_float2.size());
    partitionRealImag(h_matrix_c_float2_ref, h_matrix_c_float_ref_real, h_matrix_c_float_ref_imag, matrix_c_float2_ref.size());
    partitionRealImag(h_matrix_a_half2, h_matrix_a_half_real, h_matrix_a_half_imag, matrix_a_half2.size());
    partitionRealImag(h_matrix_b_half2, h_matrix_b_half_real, h_matrix_b_half_imag, matrix_b_half2.size());
    partitionRealImag(h_matrix_c_float2, h_matrix_c_float_real, h_matrix_c_float_imag, matrix_c_float2.size());

    __half*     d_matrix_a_half_real;
    __half*     d_matrix_a_half_imag;
    __half*     d_matrix_b_half_real;
    __half*     d_matrix_b_half_imag;
    float*      d_matrix_c_float_real;
    float*      d_matrix_c_float_imag;
    __half2*    d_matrix_a_half2;
    __half2*    d_matrix_b_half2;
    cuComplex*  d_matrix_c_float2;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half_real, matrix_size_m * matrix_size_k * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half_imag, matrix_size_m * matrix_size_k * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half_real, matrix_size_k * matrix_size_n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half_imag, matrix_size_k * matrix_size_n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float_real, matrix_size_m * matrix_size_n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float_imag, matrix_size_m * matrix_size_n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half2, matrix_size_m * matrix_size_k * sizeof(__half2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half2, matrix_size_k * matrix_size_n * sizeof(__half2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float2, matrix_size_m * matrix_size_n * sizeof(cuComplex)));
        
    // Copy data from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half_real, h_matrix_a_half_real, matrix_size_m * matrix_size_k * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half_imag, h_matrix_a_half_imag, matrix_size_m * matrix_size_k * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half_real, h_matrix_b_half_real, matrix_size_k * matrix_size_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half_imag, h_matrix_b_half_imag, matrix_size_k * matrix_size_n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half2, h_matrix_a_half2, matrix_size_m * matrix_size_k * sizeof(__half2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half2, h_matrix_b_half2, matrix_size_k * matrix_size_n * sizeof(__half2), cudaMemcpyHostToDevice));
    


    bool is_A_transpose = false;
    bool is_B_transpose = false;

    print_test_header(is_A_transpose, is_B_transpose);
    launch_mm_complex(h_matrix_a_float_real, h_matrix_a_float_imag, h_matrix_b_float_real, h_matrix_b_float_imag, h_matrix_c_float_ref_real, h_matrix_c_float_ref_imag, matrix_size_m, matrix_size_n,
        matrix_size_k, is_A_transpose, is_B_transpose);
    // Compute matrix multiplication reference output using CUDA WMMA.
    launch_wmma_mm_complex(d_matrix_a_half_real, d_matrix_a_half_imag, d_matrix_b_half_real, d_matrix_b_half_imag, d_matrix_c_float_real, d_matrix_c_float_imag,
        matrix_size_m, matrix_size_n, matrix_size_k, is_A_transpose, is_B_transpose, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float_real, d_matrix_c_float_real,
        matrix_c_float2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float_imag, d_matrix_c_float_imag,
        matrix_c_float2.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float const avg_abs_diff_ratio{get_avg_abs_diff_ratio_complex(
            h_matrix_c_float_real, h_matrix_c_float_imag, h_matrix_c_float_ref_real, h_matrix_c_float_ref_imag,
            matrix_c_float2.size())};
    if (avg_abs_diff_ratio > 0.01) {
        std::cout << "Got high average absolute diff ratio: " << avg_abs_diff_ratio << std::endl;
    }

    // Performance measurement.
    std::function<void(cudaStream_t)> const function_hmma{std::bind(
        launch_wmma_mm_complex<__half, float>, d_matrix_a_half_real, d_matrix_a_half_imag, d_matrix_b_half_real, d_matrix_b_half_imag, d_matrix_c_float_real, d_matrix_c_float_imag, matrix_size_m, matrix_size_n, matrix_size_k,
        is_A_transpose, is_B_transpose, std::placeholders::_1)};
    float const latency_hmma{measure_performance(
        function_hmma, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "HMMA Latency: " << latency_hmma << " ms" << std::endl;
    
    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half_real));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half_imag));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half_real));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half_imag));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float_real));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float_imag));
////////////////////////////////////////////////////////////////////////////////
/*
    // HMMA
    std::cout << "FP16 HMMA" << std::endl;
    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c_float(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_float_reference(matrix_size_m * matrix_size_n);

    float* h_matrix_a_float{matrix_a_float.data()};
    float* h_matrix_b_float{matrix_b_float.data()};
    __half* h_matrix_a_half{matrix_a_half.data()};
    __half* h_matrix_b_half{matrix_b_half.data()};
    float* h_matrix_c_float{matrix_c_float.data()};
    float* h_matrix_c_float_reference{matrix_c_float_reference.data()};

    fill_random_float_values(h_matrix_a_float, matrix_a_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_b_float, matrix_b_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float, matrix_c_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float_reference,
                             matrix_c_float_reference.size(), random_engine);


    float2half(h_matrix_a_half, h_matrix_a_float, matrix_a_float.size());
    float2half(h_matrix_b_half, h_matrix_b_float, matrix_b_float.size());

    half *d_matrix_a_half, *d_matrix_b_half;
    float* d_matrix_c_float;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half,
                                matrix_size_m * matrix_size_k * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half,
                                matrix_size_k * matrix_size_n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float,
                                matrix_size_m * matrix_size_n * sizeof(float)));

    // Copy data from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half, h_matrix_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half, h_matrix_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CPU.
            launch_mm(h_matrix_a_float, h_matrix_b_float,
                      h_matrix_c_float_reference, matrix_size_m, matrix_size_n,
                      matrix_size_k, is_A_transpose, is_B_transpose);
            // Compute matrix multiplication reference output using CUDA WMMA.
            launch_wmma_mm(d_matrix_a_half, d_matrix_b_half, d_matrix_c_float,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float, d_matrix_c_float,
                                        matrix_c_float.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));

            float const avg_abs_diff_ratio{get_avg_abs_diff_ratio(
                h_matrix_c_float, h_matrix_c_float_reference,
                matrix_c_float.size())};
            if (avg_abs_diff_ratio > 0.01)
            {
                std::cout << "Got high average absolute diff ratio: "
                          << avg_abs_diff_ratio << std::endl;
            }

            // Performance measurement.
            std::function<void(cudaStream_t)> const function_hmma{std::bind(
                launch_wmma_mm<__half, float>, d_matrix_a_half, d_matrix_b_half,
                d_matrix_c_float, matrix_size_m, matrix_size_n, matrix_size_k,
                is_A_transpose, is_B_transpose, std::placeholders::_1)};
            float const latency_hmma{measure_performance(
                function_hmma, stream, num_repeats, num_warmups)};
            std::cout << std::fixed << std::setprecision(3)
                      << "HMMA Latency: " << latency_hmma << " ms" << std::endl;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float));
*/
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}