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
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "../../src/api.h"
#include "cuda_fp16.h"
#include <functional>
#include <cuda_runtime.h>
#include <mma.h>

#define halfPrecision

#define nMatrix   4096 // multiples of 16

#define matDim   64 // 16, 32, 64

#if matDim == 16
#define BLOCK_DIM 16
#else
#define BLOCK_DIM 32
#endif

// A and B must be square matrices of the same size
#define A_row_d    matDim
#define A_col_d    matDim
#define B_col_d    matDim 

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::vector<cuComplex> create_rand_vector(size_t n)
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> uniform_dist(-256, 256);

    std::vector<cuComplex> vec(n);
    for (size_t i{0}; i < n; ++i)
    {
        vec[i].x = uniform_dist(e);
        vec[i].y = uniform_dist(e);
    }

    return vec;
}

void mm(cuComplex const* mat_1, cuComplex const* mat_2, cuComplex* mat_3, size_t numMatrix, size_t AMatrixRow, size_t AMatrixCol, size_t BMatrixCol)
{
    for (size_t matIdx{0}; matIdx < numMatrix; ++matIdx) {
        uint32_t AMatStart = matIdx*AMatrixRow*AMatrixCol;
        uint32_t BMatStart = matIdx*AMatrixCol*BMatrixCol;
        uint32_t CMatStart = matIdx*AMatrixRow*BMatrixCol;
        for (size_t i{0}; i < AMatrixRow; ++i) {
            for (size_t j{0}; j < BMatrixCol; ++j) {
                cuComplex acc_sum;
                acc_sum.x = 0;
                acc_sum.y = 0;
                for (size_t k{0}; k < AMatrixCol; ++k) {
                    acc_sum.x += mat_1[AMatStart + i + k*AMatrixRow].x * mat_2[BMatStart + k + j*AMatrixCol].x - mat_1[AMatStart + i + k*AMatrixRow].y * mat_2[BMatStart + k + j*AMatrixCol].y;
                    acc_sum.y += mat_1[AMatStart + i + k*AMatrixRow].x * mat_2[BMatStart + k + j*AMatrixCol].y + mat_1[AMatStart + i + k*AMatrixRow].y * mat_2[BMatStart + k + j*AMatrixCol].x;
                }
                mat_3[CMatStart + i + j*AMatrixRow] = acc_sum;
            }
        }
    }
}

template <typename T>
__global__ void mm_kernel(T const* mat_1, T const* mat_2, cuComplex* mat_3, 
    size_t numMatrix, size_t AMatrixRow, size_t AMatrixCol, size_t BMatrixCol, int numBlocksPerMatDim)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    int matIdx = blockIdx.x/numBlocksPerMatDim;
    int matBlockX = blockIdx.x - matIdx*numBlocksPerMatDim;
    int matBlockY = blockIdx.y;

    size_t i{matBlockX*BLOCK_DIM + threadIdx.x};
    size_t j{matBlockY*BLOCK_DIM + threadIdx.y};

    uint32_t AMatStart = matIdx*AMatrixRow*AMatrixCol;
    uint32_t BMatStart = matIdx*AMatrixCol*BMatrixCol;
    uint32_t CMatStart = matIdx*AMatrixRow*BMatrixCol;

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((i >= AMatrixRow) || (j >= BMatrixCol))
    {
        return;
    }
#ifdef halfPrecision   
    T acc_sum(0, 0);
#else
    T acc_sum;
    acc_sum.x = 0;
    acc_sum.y = 0;
#endif
    for (size_t k{0}; k < AMatrixCol; ++k)
    {
#ifdef halfPrecision    
        acc_sum = __hcmadd(mat_1[AMatStart + i + k*AMatrixRow], mat_2[BMatStart + k + j*AMatrixCol], acc_sum);
#else
        acc_sum.x += mat_1[AMatStart + i + k*AMatrixRow].x * mat_2[BMatStart + k + j*AMatrixCol].x - mat_1[AMatStart + i + k*AMatrixRow].y * mat_2[BMatStart + k + j*AMatrixCol].y;
        acc_sum.y += mat_1[AMatStart + i + k*AMatrixRow].x * mat_2[BMatStart + k + j*AMatrixCol].y + mat_1[AMatStart + i + k*AMatrixRow].y * mat_2[BMatStart + k + j*AMatrixCol].x;
#endif
    }
#ifdef halfPrecision
    mat_3[CMatStart + i + j*AMatrixRow] = __half22float2(acc_sum);
#else
    mat_3[CMatStart + i + j*AMatrixRow] = acc_sum;
#endif
}

template <typename T>
__global__ void mm_kernel_optimized(T const* mat_1, T const* mat_2, cuComplex* mat_3,
    size_t numMatrix, size_t AMatrixRow, size_t AMatrixCol, size_t BMatrixCol, int numBlocksPerMatDim)
{
    __shared__ T mat_1_tile[BLOCK_DIM][BLOCK_DIM];
    __shared__ T mat_2_tile[BLOCK_DIM][BLOCK_DIM];

    int matIdx = blockIdx.x/numBlocksPerMatDim;
    int matBlockX = blockIdx.x - matIdx*numBlocksPerMatDim;
    int matBlockY = blockIdx.y;

    uint32_t AMatStart = matIdx*AMatrixRow*AMatrixCol;
    uint32_t BMatStart = matIdx*AMatrixCol*BMatrixCol;
    uint32_t CMatStart = matIdx*AMatrixRow*BMatrixCol;

#ifdef halfPrecision  
    T acc_sum(0, 0);
#else
    T acc_sum;
    acc_sum.x = 0;
    acc_sum.y = 0;
#endif
    for (size_t tile_idx{0};
         tile_idx < ceilf(static_cast<float>(AMatrixCol) / BLOCK_DIM); ++tile_idx)
    {
        size_t i{matBlockX*BLOCK_DIM + threadIdx.y};
        size_t j{tile_idx*BLOCK_DIM + threadIdx.x};
        if ((i < AMatrixRow) && (j < AMatrixCol))
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = mat_1[AMatStart + i + j*AMatrixRow];
        }
        else
        {
            mat_1_tile[threadIdx.y][threadIdx.x].x = 0;
            mat_1_tile[threadIdx.y][threadIdx.x].y = 0;
        }
        i = tile_idx * BLOCK_DIM + threadIdx.y;
        j = matBlockY * BLOCK_DIM + threadIdx.x;
        if ((i < AMatrixCol) && (j < BMatrixCol))
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = mat_2[BMatStart + i + j*AMatrixCol];
        }
        else
        {
            mat_2_tile[threadIdx.y][threadIdx.x].x = 0;
            mat_2_tile[threadIdx.y][threadIdx.x].y = 0;
        }
        __syncthreads();
        for (size_t k{0}; k < BLOCK_DIM; ++k)
        {
#ifdef halfPrecision   
            acc_sum = __hcmadd(mat_1_tile[threadIdx.y][k], mat_2_tile[k][threadIdx.x], acc_sum);
#else
            acc_sum.x += mat_1_tile[threadIdx.y][k].x * mat_2_tile[k][threadIdx.x].x - mat_1_tile[threadIdx.y][k].y * mat_2_tile[k][threadIdx.x].y;
            acc_sum.y += mat_1_tile[threadIdx.y][k].x * mat_2_tile[k][threadIdx.x].y + mat_1_tile[threadIdx.y][k].y * mat_2_tile[k][threadIdx.x].x;
#endif
        }
        __syncthreads();
    }

    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{matBlockX*BLOCK_DIM + threadIdx.y};
    size_t j{matBlockY*BLOCK_DIM + threadIdx.x};

    if ((i < AMatrixRow) && (j < BMatrixCol))
    {
#ifdef halfPrecision
        mat_3[CMatStart + i + j*AMatrixRow] = __half22float2(acc_sum);
#else
        mat_3[CMatStart + i + j*AMatrixRow] = acc_sum;
#endif
    }
}

template <typename T>
void mm_cuda(T const* mat_1, T const* mat_2, cuComplex* mat_3, 
             size_t numMatrix, size_t AMatrixRow, size_t AMatrixCol, size_t BMatrixCol,
             void (*f)(T const*, T const*, cuComplex*, size_t, size_t, size_t, size_t, int))
{
    assert(AMatrixRow == AMatrixCol && AMatrixCol == BMatrixCol);
    assert(AMatrixRow == 16 || AMatrixRow == 32 || AMatrixRow == 64);

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);

    int numBlocksPerMatDim = AMatrixRow/BLOCK_DIM;
    blocks_per_grid.x = numMatrix*numBlocksPerMatDim;
    blocks_per_grid.y = numBlocksPerMatDim;
    f<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, numMatrix, AMatrixRow, AMatrixCol, BMatrixCol, numBlocksPerMatDim);
}

template <typename T>
bool allclose(std::vector<T> const& vec_1, std::vector<T> const& vec_2,
              float const& abs_tol)
{
    if (vec_1.size() != vec_2.size())
    {
        return false;
    }

    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < vec_1.size(); ++i)
    {
        sum_abs_diff_ratio += std::abs(vec_1[i].x - vec_2[i].x)/std::abs(vec_1[i].x + vec_2[i].x);
        sum_abs_diff_ratio += std::abs(vec_1[i].y - vec_2[i].y)/std::abs(vec_1[i].y + vec_2[i].y);
    }
    sum_abs_diff_ratio = sum_abs_diff_ratio/vec_1.size()/2.0;
    if (sum_abs_diff_ratio > abs_tol) {
        printf("sum_abs_diff_ratio = %f\n", sum_abs_diff_ratio);
        return false;
    }
    return true;
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
bool random_test_mm_cuda(uint32_t numMatrix, uint32_t AMatrixRow,
                         uint32_t AMatrixCol, uint32_t BMatrixCol,
                         void (*f)(T const*, T const*, cuComplex*, size_t, size_t, size_t,
                                   size_t, int))
{
    std::vector<cuComplex> mat_1_vec{create_rand_vector(numMatrix*AMatrixRow*AMatrixCol)};
    std::vector<cuComplex> mat_2_vec{create_rand_vector(numMatrix*AMatrixCol*BMatrixCol)};
    std::vector<cuComplex> mat_3_vec(numMatrix*AMatrixRow*BMatrixCol);
    std::vector<cuComplex> mat_4_vec(numMatrix*AMatrixRow*BMatrixCol);

    cuComplex* mat_1_float2{mat_1_vec.data()};
    cuComplex* mat_2_float2{mat_2_vec.data()};

#ifdef halfPrecision
    std::vector<__half2> mat_1_half2(numMatrix*AMatrixRow*AMatrixCol);
    std::vector<__half2> mat_2_half2(numMatrix*AMatrixCol*BMatrixCol);
    T* mat_1{mat_1_half2.data()};
    T* mat_2{mat_2_half2.data()};
    float2ToHalf2(mat_1, mat_1_float2, mat_1_vec.size());
    float2ToHalf2(mat_2, mat_2_float2, mat_2_vec.size());
#else
    T* mat_1 = mat_1_float2;
    T* mat_2 = mat_2_float2;
#endif

    cuComplex* mat_3{mat_3_vec.data()};
    cuComplex* mat_4{mat_4_vec.data()};

    mm(mat_1_float2, mat_2_float2, mat_3, numMatrix, AMatrixRow, AMatrixCol, BMatrixCol);
    
    T* d_mat_1;
    T* d_mat_2;
    cuComplex* d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * mat_1_vec.size()));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * mat_2_vec.size()));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(cuComplex) * mat_4_vec.size()));

    // Copy data from host to device.
    checkCuda(cudaMemcpy(d_mat_1, mat_1, sizeof(T) * mat_1_vec.size(),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat_2, mat_2, sizeof(T) * mat_2_vec.size(),
                         cudaMemcpyHostToDevice));
                         
    // Run matrix multiplication on GPU.
    mm_cuda(d_mat_1, d_mat_2, d_mat_4, numMatrix, AMatrixRow, AMatrixCol, BMatrixCol, f);
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // Copy data from device to host.
    checkCuda(cudaMemcpy(mat_4, d_mat_4, sizeof(cuComplex) * mat_4_vec.size(),
                         cudaMemcpyDeviceToHost));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    return allclose<cuComplex>(mat_3_vec, mat_4_vec, 0.01);
}

template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests,
                                  void (*f)(T const*, T const*, cuComplex*, size_t,
                                            size_t, size_t, size_t, int))
{
    uint32_t const      A_matrix_size_row{A_row_d};
    uint32_t const      A_matrix_size_col{A_col_d}; // equal to B_matrix_size_row for matrix multiplication A*B
    uint32_t const      B_matrix_size_col{B_col_d};
    uint32_t const      number_matrix{nMatrix};
    bool success{false};
   
    for (size_t i{0}; i < num_tests; ++i)
    {
        success = random_test_mm_cuda<T>(number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, f);
        if (!success)
        {
            return false;
        }
    }

    return true;
}

template <typename T>
float measure_latency_mm_cuda(size_t numMatrix, size_t AMatrixRow, size_t AMatrixCol, size_t BMatrixCol, size_t num_tests,
                              size_t num_warmups,
                              void (*f)(T const*, T const*, cuComplex*, size_t, size_t,
                                        size_t, size_t, int))
{
    cudaEvent_t startEvent, stopEvent;
    float time{0.0f};

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    T *d_mat_1, *d_mat_2;
    cuComplex* d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * numMatrix * AMatrixRow * AMatrixCol));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * numMatrix * AMatrixCol * BMatrixCol));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(cuComplex) * numMatrix * AMatrixRow * BMatrixCol));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, numMatrix, AMatrixRow, AMatrixCol, BMatrixCol, f);
    }

    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i{0}; i < num_tests; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, numMatrix, AMatrixRow, AMatrixCol, BMatrixCol, f);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    float latency{time / num_tests};

    return latency;
}

int main()
{
    constexpr size_t    num_tests{10};
    constexpr size_t    num_measurement_tests{100};
    constexpr size_t    num_measurement_warmups{10};
    uint32_t const      A_matrix_size_row{A_row_d};
    uint32_t const      A_matrix_size_col{A_col_d}; // equal to B_matrix_size_row for matrix multiplication A*B
    uint32_t const      B_matrix_size_row{A_matrix_size_col}; 
    uint32_t const      B_matrix_size_col{B_col_d};
    uint32_t const      number_matrix{nMatrix};

#ifdef halfPrecision
    assert(random_multiple_test_mm_cuda<__half2>(num_tests, mm_kernel));
    assert(random_multiple_test_mm_cuda<__half2>(num_tests, mm_kernel_optimized));
    float mm_cuda_float_latency{measure_latency_mm_cuda<__half2>(
        number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, num_measurement_tests, num_measurement_warmups, mm_kernel)};
#else
    assert(random_multiple_test_mm_cuda<cuComplex>(num_tests, mm_kernel));
    assert(random_multiple_test_mm_cuda<cuComplex>(num_tests, mm_kernel_optimized));
    float mm_cuda_float_latency{measure_latency_mm_cuda<cuComplex>(
        number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, num_measurement_tests, num_measurement_warmups, mm_kernel)};
#endif

    std::cout << "Batched matrix Multiplication CUDA Latency" << std::endl;
    std::cout << "number_matrix: " << number_matrix << std::endl;
    std::cout << "A_matrix_size_row: " << A_matrix_size_row << std::endl;
    std::cout << "A_matrix_size_col: " << A_matrix_size_col << std::endl;
    std::cout << "B_matrix_size_row: " << B_matrix_size_row << std::endl;
    std::cout << "B_matrix_size_col: " << B_matrix_size_col << std::endl;

#ifdef halfPrecision
    std::cout << "__half2: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl; 

    mm_cuda_float_latency = measure_latency_mm_cuda<__half2>(
                number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, num_measurement_tests, num_measurement_warmups,
                mm_kernel_optimized);
#else
    std::cout << "cuComplex: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;

    mm_cuda_float_latency = measure_latency_mm_cuda<cuComplex>(
                number_matrix, A_matrix_size_row, A_matrix_size_col, B_matrix_size_col, num_measurement_tests, num_measurement_warmups,
                mm_kernel_optimized);
#endif
    
    std::cout << "Optimized Matrix Multiplication CUDA Latency" << std::endl;
    std::cout << "number_matrix: " << number_matrix << std::endl;
    std::cout << "A_matrix_size_row: " << A_matrix_size_row << std::endl;
    std::cout << "A_matrix_size_col: " << A_matrix_size_col << std::endl;
    std::cout << "B_matrix_size_row: " << B_matrix_size_row << std::endl;
    std::cout << "B_matrix_size_col: " << B_matrix_size_col << std::endl;
#ifdef halfPrecision
    std::cout << "__half2: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;
#else
    std::cout << "cuComplex: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;
#endif
}