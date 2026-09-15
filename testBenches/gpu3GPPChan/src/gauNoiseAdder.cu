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

#include "gauNoiseAdder.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <iostream>

// Constructor
template <typename Tcomplex>
GauNoiseAdder<Tcomplex>::GauNoiseAdder(uint32_t nThreads, int seed, cudaStream_t strm)
    : m_nThreads(nThreads), m_seed(seed), m_strm(strm) {
    
    // Set block and grid sizes
    m_threadsPerBlock = 256;
    m_blocksPerGrid = (m_nThreads + m_threadsPerBlock - 1) / m_threadsPerBlock;

    // Allocate memory for cuRAND states
    cudaMalloc(&m_d_states, m_nThreads * sizeof(curandState));

    // Initialize cuRAND states using cuLaunchKernel
    void* args[] = { &m_d_states, &m_seed, &m_nThreads };
    
    // Assuming the module and function are already loaded and available as functionPtr
    cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(init_curand_states));
    CUresult status = cuLaunchKernel(m_functionPtr,
                                     m_blocksPerGrid, 1, 1,
                                     m_threadsPerBlock, 1, 1,
                                     0, m_strm,
                                     args, nullptr);
    CHECK_CURESULT(status);

    // prepare for adding kernel
    cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(add_gaussian_noise<Tcomplex>));
}

// Destructor
template <typename Tcomplex>
GauNoiseAdder<Tcomplex>::~GauNoiseAdder() {
    // Free cuRAND states
    cudaFree(m_d_states);
}

// Method to add noise
template <typename Tcomplex>
void GauNoiseAdder<Tcomplex>::addNoise(Tcomplex* d_signal, uint32_t signalSize, float snr_db) {
    float snr_linear = powf(10.0f, snr_db / 10.0f);
    float noise_std = sqrtf(0.5f / snr_linear);

    // Launch kernel to add Gaussian noise using cuLaunchKernel
    void* args[] = { &d_signal, &signalSize, &noise_std, &m_d_states };

    // Assuming the module and function are already loaded and available as functionPtr
    CUresult status = cuLaunchKernel(m_functionPtr,
                                     m_blocksPerGrid, 1, 1,
                                     m_threadsPerBlock, 1, 1,
                                     0, m_strm,
                                     args, nullptr);
    CHECK_CURESULT(status);
}

// Kernel for initializing cuRAND states
__global__ void init_curand_states(curandState* states, unsigned long seed, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel for adding Gaussian noise
template <typename Tcomplex>
__global__ void add_gaussian_noise(Tcomplex* signal, uint32_t signalSize, float noise_std, curandState* states) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Iterate over the signal in chunks handled by each thread
    for (uint32_t i = idx; i < signalSize; i += blockDim.x * gridDim.x) {
        curandState localState = states[idx];
        float noise_real = noise_std * curand_normal(&localState);
        float noise_imag = noise_std * curand_normal(&localState);

        signal[i].x = float(signal[i].x) + noise_real;
        signal[i].y = float(signal[i].y) + noise_imag;

        states[idx] = localState;
    }
}