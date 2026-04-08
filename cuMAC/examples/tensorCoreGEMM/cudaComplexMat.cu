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

#define BLOCK_DIM 32

#define row_d    4096
#define mid_d    4096
#define col_d    4096  

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

// mat_1: m x n
// mat_2: n x p
// mat_3: m x p
void mm(cuComplex const* mat_1, cuComplex const* mat_2, cuComplex* mat_3, size_t m, size_t n, size_t p)
{
    // Compute the cells in mat_3 sequentially.
    for (size_t i{0}; i < m; ++i)
    {
        for (size_t j{0}; j < p; ++j)
        {
            cuComplex acc_sum;
            acc_sum.x = 0;
            acc_sum.y = 0;
            for (size_t k{0}; k < n; ++k)
            {
                acc_sum.x += mat_1[i + k*m].x * mat_2[k + j*n].x - mat_1[i + k*m].y * mat_2[k + j*n].y;
                acc_sum.y += mat_1[i + k*m].x * mat_2[k + j*n].y + mat_1[i + k*m].y * mat_2[k + j*n].x;
            }
            mat_3[i + j*m] = acc_sum;
        }
    }
}

template <typename T>
__global__ void mm_kernel(T const* mat_1, T const* mat_2, cuComplex* mat_3, size_t m,
                          size_t n, size_t p)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((i >= m) || (j >= p))
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
    for (size_t k{0}; k < n; ++k)
    {
#ifdef halfPrecision    
        acc_sum = __hcmadd(mat_1[i + k*m], mat_2[k + j*n], acc_sum);
#else
        acc_sum.x += mat_1[i + k*m].x * mat_2[k + j*n].x - mat_1[i + k*m].y * mat_2[k + j*n].y;
        acc_sum.y += mat_1[i + k*m].x * mat_2[k + j*n].y + mat_1[i + k*m].y * mat_2[k + j*n].x;
#endif
    }
#ifdef halfPrecision
    mat_3[i + j*m] = __half22float2(acc_sum);
#else
    mat_3[i + j*m] = acc_sum;
#endif
}

template <typename T>
__global__ void mm_kernel_optimized(T const* mat_1, T const* mat_2, cuComplex* mat_3,
                                    size_t m, size_t n, size_t p)
{
    __shared__ T mat_1_tile[BLOCK_DIM][BLOCK_DIM];
    __shared__ T mat_2_tile[BLOCK_DIM][BLOCK_DIM];

#ifdef halfPrecision  
    T acc_sum(0, 0);
#else
    T acc_sum;
    acc_sum.x = 0;
    acc_sum.y = 0;
#endif
    for (size_t tile_idx{0};
         tile_idx < ceilf(static_cast<float>(n) / BLOCK_DIM); ++tile_idx)
    {
        size_t i{blockIdx.y * blockDim.y + threadIdx.y};
        size_t j{tile_idx * blockDim.x + threadIdx.x};
        if ((i < m) && (j < n))
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = mat_1[i + j*m];
        }
        else
        {
            mat_1_tile[threadIdx.y][threadIdx.x].x = 0;
            mat_1_tile[threadIdx.y][threadIdx.x].y = 0;
        }
        i = tile_idx * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if ((i < n) && (j < p))
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = mat_2[i + j*n];
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
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    if ((i < m) && (j < p))
    {
#ifdef halfPrecision
        mat_3[i + j*m] = __half22float2(acc_sum);
#else
        mat_3[i + j*m] = acc_sum;
#endif
    }
}

template <typename T>
void mm_cuda(T const* mat_1, T const* mat_2, cuComplex* mat_3, size_t m, size_t n,
             size_t p,
             void (*f)(T const*, T const*, cuComplex*, size_t, size_t, size_t))
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    f<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n, p);
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
bool random_test_mm_cuda(size_t m, size_t n, size_t p,
                         void (*f)(T const*, T const*, cuComplex*, size_t, size_t,
                                   size_t))
{
    std::vector<cuComplex> mat_1_vec{create_rand_vector(m * n)};
    std::vector<cuComplex> mat_2_vec{create_rand_vector(n * p)};
    std::vector<cuComplex> mat_3_vec(m * p);
    std::vector<cuComplex> mat_4_vec(m * p);

    cuComplex* mat_1_float2{mat_1_vec.data()};
    cuComplex* mat_2_float2{mat_2_vec.data()};

#ifdef halfPrecision
    std::vector<__half2> mat_1_half2(m * n);
    std::vector<__half2> mat_2_half2(n * p);
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

    mm(mat_1_float2, mat_2_float2, mat_3, m, n, p);
    
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
    mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p, f);
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
                                            size_t, size_t))
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(1, 256);

    size_t m{0}, n{0}, p{0};
    bool success{false};
   
    for (size_t i{0}; i < num_tests; ++i)
    {
        m = static_cast<size_t>(uniform_dist(e));
        n = static_cast<size_t>(uniform_dist(e));
        p = static_cast<size_t>(uniform_dist(e));
        success = random_test_mm_cuda<T>(m, n, p, f);
        if (!success)
        {
            return false;
        }
    }

    return true;
}

template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p, size_t num_tests,
                              size_t num_warmups,
                              void (*f)(T const*, T const*, cuComplex*, size_t, size_t,
                                        size_t))
{
    cudaEvent_t startEvent, stopEvent;
    float time{0.0f};

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    T *d_mat_1, *d_mat_2;
    cuComplex* d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * m * n));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * n * p));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(cuComplex) * m * p));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p, f);
    }

    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i{0}; i < num_tests; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p, f);
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
    constexpr size_t num_tests{10};
    constexpr size_t num_measurement_tests{100};
    constexpr size_t num_measurement_warmups{10};
    const size_t m{row_d}, n{mid_d}, p{col_d};

#ifdef halfPrecision
    assert(random_multiple_test_mm_cuda<__half2>(num_tests, mm_kernel));
    assert(random_multiple_test_mm_cuda<__half2>(num_tests, mm_kernel_optimized));
    float mm_cuda_float_latency{measure_latency_mm_cuda<__half2>(
        m, n, p, num_measurement_tests, num_measurement_warmups, mm_kernel)};
#else
    assert(random_multiple_test_mm_cuda<cuComplex>(num_tests, mm_kernel));
    assert(random_multiple_test_mm_cuda<cuComplex>(num_tests, mm_kernel_optimized));
    float mm_cuda_float_latency{measure_latency_mm_cuda<cuComplex>(
        m, n, p, num_measurement_tests, num_measurement_warmups, mm_kernel)};
#endif

    std::cout << "Matrix Multiplication CUDA Latency" << std::endl;
    std::cout << "m: " << m << " "
              << "n: " << n << " "
              << "p: " << p << std::endl;
#ifdef halfPrecision
    std::cout << "__half2: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl; 

    mm_cuda_float_latency = measure_latency_mm_cuda<__half2>(
                m, n, p, num_measurement_tests, num_measurement_warmups,
                mm_kernel_optimized);
#else
    std::cout << "cuComplex: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;

    mm_cuda_float_latency = measure_latency_mm_cuda<cuComplex>(
                m, n, p, num_measurement_tests, num_measurement_warmups,
                mm_kernel_optimized);
#endif
    
    std::cout << "Optimized Matrix Multiplication CUDA Latency" << std::endl;
    std::cout << "m: " << m << " "
              << "n: " << n << " "
              << "p: " << p << std::endl;
#ifdef halfPrecision
    std::cout << "__half2: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;
#else
    std::cout << "cuComplex: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;
#endif
}