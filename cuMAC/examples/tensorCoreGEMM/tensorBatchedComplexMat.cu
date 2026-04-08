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

#define nMatrix   4096 // multiples of 16

#define matDim   64 // 16, 32, 64

// A and B must be square matrices of the same size
#define A_row_d    matDim
#define A_col_d    matDim
#define B_col_d    matDim 

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
// WMMA_FRAG_LAYOUT_A: nvcuda::wmma::row_major if A is
// transposed, otherwise nvcuda::wmma::col_major.
// WMMA_FRAG_LAYOUT_B: nvcuda::wmma::row_major if B is
// transposed, otherwise nvcuda::wmma::col_major.

template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_complex_a_col_major_b_col_major(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t AMatrixRow,
    uint32_t AMatrixCol, uint32_t BMatrixCol)
{
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t AMatStart = blockIdx.x*AMatrixRow*AMatrixCol;
    uint32_t BMatStart = blockIdx.x*AMatrixCol*BMatrixCol;
    uint32_t CMatStart = blockIdx.x*AMatrixRow*BMatrixCol;

    uint32_t const warpM{threadIdx.x / warpSize};
    uint32_t const warpN{threadIdx.y};

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
    for (uint32_t ki{0}; ki < AMatrixCol; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{ki};
        uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < AMatrixRow &&
            matrix_mma_a_col_idx < AMatrixCol &&
            matrix_mma_b_row_idx < AMatrixCol &&
            matrix_mma_b_col_idx < BMatrixCol)
        {
            T1 const* matrix_mma_a_mptr{A_real + AMatStart + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * AMatrixRow};
            T1 const* matrix_mma_b_mptr{B_real + BMatStart + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * AMatrixCol};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, AMatrixRow);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, AMatrixCol);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < AMatrixRow && matrix_mma_c_col_idx < BMatrixCol) {
        T2* matrix_mma_c_mptr{C_real + CMatStart + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * AMatrixRow};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, AMatrixRow,
                                       nvcuda::wmma::mem_col_major);

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = acc_frag.x[i];
        }
    }

    // calculate Ai*Bi
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < AMatrixCol; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{ki};
        uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < AMatrixRow &&
            matrix_mma_a_col_idx < AMatrixCol &&
            matrix_mma_b_row_idx < AMatrixCol &&
            matrix_mma_b_col_idx < BMatrixCol)
        {
            T1 const* matrix_mma_a_mptr{A_imag + AMatStart + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * AMatrixRow};
            T1 const* matrix_mma_b_mptr{B_imag + BMatStart + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * AMatrixCol};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, AMatrixRow);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, AMatrixCol);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < AMatrixRow && matrix_mma_c_col_idx < BMatrixCol) {
        T2* matrix_mma_c_mptr{C_real + CMatStart + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * AMatrixRow};

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] =  c_frag.x[i] - acc_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, AMatrixRow,
                                        nvcuda::wmma::mem_col_major);
    }

    // calculate Ar*Bi
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < AMatrixCol; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{ki};
        uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < AMatrixRow &&
            matrix_mma_a_col_idx < AMatrixCol &&
            matrix_mma_b_row_idx < AMatrixCol &&
            matrix_mma_b_col_idx < BMatrixCol)
        {
            T1 const* matrix_mma_a_mptr{A_real + AMatStart + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * AMatrixRow};
            T1 const* matrix_mma_b_mptr{B_imag + BMatStart + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * AMatrixCol};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, AMatrixRow);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, AMatrixCol);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < AMatrixRow && matrix_mma_c_col_idx < BMatrixCol) {
        T2* matrix_mma_c_mptr{C_imag + CMatStart + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * AMatrixRow};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, AMatrixRow,
                                       nvcuda::wmma::mem_col_major);

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = acc_frag.x[i];
        }
    }

    // calculate Ai*Br
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // Loop over K.
    for (uint32_t ki{0}; ki < AMatrixCol; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{ki};
        // Matrix B mma matrix
        uint32_t const matrix_mma_b_row_idx{ki};
        uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};

        // Bounds checking
        if (matrix_mma_a_row_idx < AMatrixRow &&
            matrix_mma_a_col_idx < AMatrixCol &&
            matrix_mma_b_row_idx < AMatrixCol &&
            matrix_mma_b_col_idx < BMatrixCol)
        {
            T1 const* matrix_mma_a_mptr{A_imag + AMatStart + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * AMatrixRow};
            T1 const* matrix_mma_b_mptr{B_real + BMatStart + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * AMatrixCol};
            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, AMatrixRow);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, AMatrixCol);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (matrix_mma_c_row_idx < AMatrixRow && matrix_mma_c_col_idx < BMatrixCol) {
        T2* matrix_mma_c_mptr{C_imag + CMatStart + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * AMatrixRow};

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = c_frag.x[i] + acc_frag.x[i];
        }
        // Store the output
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, AMatrixRow,
                                        nvcuda::wmma::mem_col_major);
    }
}


template <typename T1, typename T2>
void launch_batched_wmma_mm_complex(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t numMatrix, uint32_t AMatrixRow,
    uint32_t AMatrixCol, uint32_t BMatrixCol, cudaStream_t stream)
{
    // Assume there is no padding in our data.
    // assume both A and B are square matrices of the same size (16, 32, 64)
    assert(AMatrixRow == AMatrixCol && AMatrixCol == BMatrixCol);
    assert(AMatrixRow == 16 || AMatrixRow == 32 || AMatrixRow == 64);

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    int const num_warps_x = AMatrixRow/16;
    int const num_warps_y = AMatrixRow/16;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up.
    gridDim.x = numMatrix;
    gridDim.y = 1;

    // C = A * B
    wmma_gemm_complex_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
        <<<gridDim, blockDim, 0, stream>>>(A_real, A_imag, B_real, B_imag, C_real, C_imag, AMatrixRow,
                                           AMatrixCol, BMatrixCol);

    CHECK_LAST_CUDA_ERROR();
}

template <typename T1, typename T2>
void mm_complex_a_col_major_b_col_major(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t numMatrix, uint32_t AMatrixRow,
    uint32_t AMatrixCol, uint32_t BMatrixCol)
{
    // C matrix row number = AMatrixRow, column number = BMatrixCol
    for (uint32_t matIdx{0}; matIdx < numMatrix; ++matIdx) {
        uint32_t AMatStart = matIdx*AMatrixRow*AMatrixCol;
        uint32_t BMatStart = matIdx*AMatrixCol*BMatrixCol;
        uint32_t CMatStart = matIdx*AMatrixRow*BMatrixCol;
        for (uint32_t colIdx{0}; colIdx < BMatrixCol; ++colIdx) {
            for (uint32_t rowIdx{0}; rowIdx < AMatrixRow; ++rowIdx) {
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
                for (uint32_t ki{0}; ki < AMatrixCol; ++ki) {
                    accum_ArBr += A_real[AMatStart + ki * AMatrixRow + rowIdx] * B_real[BMatStart + colIdx * AMatrixCol + ki];
                    accum_negAiBi += -A_imag[AMatStart + ki * AMatrixRow + rowIdx] * B_imag[BMatStart + colIdx * AMatrixCol + ki];
                    accum_ArBi += A_real[AMatStart + ki * AMatrixRow + rowIdx] * B_imag[BMatStart + colIdx * AMatrixCol + ki];
                    accum_AiBr += A_imag[AMatStart + ki * AMatrixRow + rowIdx] * B_real[BMatStart + colIdx * AMatrixCol + ki];
                }
           
                C_real[CMatStart + colIdx * AMatrixRow + rowIdx] = accum_ArBr + accum_negAiBi;
                C_imag[CMatStart + colIdx * AMatrixRow + rowIdx] = accum_ArBi + accum_AiBr;
            }
        }
    }
}    

template <typename T1, typename T2>
void launch_batched_mm_complex(T1 const* A_real, T1 const* A_imag, T1 const* B_real, T1 const* B_imag, T2* C_real, T2* C_imag, uint32_t numMatrix, uint32_t AMatrixRow,
    uint32_t AMatrixCol, uint32_t BMatrixCol)
{
    // Assume there is no padding in our data.
    mm_complex_a_col_major_b_col_major(A_real, A_imag, B_real, B_imag, C_real, C_imag, numMatrix, AMatrixRow,
         AMatrixCol, BMatrixCol);
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

void float2ToHalf2(__half2* half2_arr, cuComplex const* float2_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half2_arr[i].x = __float2half(float2_arr[i].x);
        half2_arr[i].y = __float2half(float2_arr[i].y);
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

int main()
{
    cudaSetDevice(7);

    constexpr int num_repeats{1};
    constexpr int num_warmups{1};

    uint32_t const A_matrix_size_row{A_row_d};
    uint32_t const A_matrix_size_col{A_col_d}; // equal to B_matrix_size_row for matrix multiplication A*B
    uint32_t const B_matrix_size_row{A_matrix_size_col}; 
    uint32_t const B_matrix_size_col{B_col_d};
    uint32_t const C_matrix_size_row{A_matrix_size_row};
    uint32_t const C_matrix_size_col{B_matrix_size_col};
    uint32_t const number_matrix{nMatrix};

    std::cout << "Batched matrix Multiplication tensor core Latency" << std::endl;
    std::cout << "number_matrix: " << number_matrix << std::endl;
    std::cout << "A_matrix_size_row: " << A_matrix_size_row << std::endl;
    std::cout << "A_matrix_size_col: " << A_matrix_size_col << std::endl;
    std::cout << "B_matrix_size_row: " << B_matrix_size_row << std::endl;
    std::cout << "B_matrix_size_col: " << B_matrix_size_col << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
 
    std::vector<cuComplex> matrix_a_float2(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<cuComplex> matrix_b_float2(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<__half2>   matrix_a_half2(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<__half2>   matrix_b_half2(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<cuComplex> matrix_c_float2(number_matrix * C_matrix_size_row * C_matrix_size_col);
    std::vector<cuComplex> matrix_c_float2_ref(number_matrix * C_matrix_size_row * C_matrix_size_col);

    // partitioned real and image matrices
    std::vector<float>     matrix_a_float_real(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<float>     matrix_a_float_imag(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<float>     matrix_b_float_real(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<float>     matrix_b_float_imag(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<__half>    matrix_a_half_real(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<__half>    matrix_a_half_imag(number_matrix * A_matrix_size_row * A_matrix_size_col);
    std::vector<__half>    matrix_b_half_real(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<__half>    matrix_b_half_imag(number_matrix * B_matrix_size_row * B_matrix_size_col);
    std::vector<float>     matrix_c_float_real(number_matrix * C_matrix_size_row * C_matrix_size_col);
    std::vector<float>     matrix_c_float_imag(number_matrix * C_matrix_size_row * C_matrix_size_col);
    std::vector<float>     matrix_c_float_ref_real(number_matrix * C_matrix_size_row * C_matrix_size_col);
    std::vector<float>     matrix_c_float_ref_imag(number_matrix * C_matrix_size_row * C_matrix_size_col);


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

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half_real, matrix_a_half_real.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half_imag, matrix_a_half_imag.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half_real, matrix_b_half_real.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half_imag, matrix_b_half_imag.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float_real, matrix_c_float_real.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float_imag, matrix_c_float_imag.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half2, matrix_a_half2.size() * sizeof(__half2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half2, matrix_b_half2.size() * sizeof(__half2)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float2, matrix_c_float2.size() * sizeof(cuComplex)));
        
    // Copy data from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half_real, h_matrix_a_half_real, matrix_a_half_real.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half_imag, h_matrix_a_half_imag, matrix_a_half_imag.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half_real, h_matrix_b_half_real, matrix_b_half_real.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half_imag, h_matrix_b_half_imag, matrix_b_half_imag.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half2, h_matrix_a_half2, matrix_a_half2.size() * sizeof(__half2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half2, h_matrix_b_half2, matrix_b_half2.size() * sizeof(__half2), cudaMemcpyHostToDevice));

    launch_batched_mm_complex(h_matrix_a_float_real, h_matrix_a_float_imag, h_matrix_b_float_real, h_matrix_b_float_imag, h_matrix_c_float_ref_real, h_matrix_c_float_ref_imag, number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col);
    // Compute matrix multiplication reference output using CUDA WMMA.
    launch_batched_wmma_mm_complex(d_matrix_a_half_real, d_matrix_a_half_imag, d_matrix_b_half_real, d_matrix_b_half_imag, d_matrix_c_float_real, d_matrix_c_float_imag,
        number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, stream);
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
        launch_batched_wmma_mm_complex<__half, float>, d_matrix_a_half_real, d_matrix_a_half_imag, d_matrix_b_half_real, d_matrix_b_half_imag, d_matrix_c_float_real, d_matrix_c_float_imag, number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, std::placeholders::_1)};
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

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}