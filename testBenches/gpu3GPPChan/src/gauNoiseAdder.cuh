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

#ifndef GAU_NOISE_ADDER_CUH
#define GAU_NOISE_ADDER_CUH

#include <curand_kernel.h>
#include <cstdint>
#include <cuda.h>
#include "gpu3gppchanCommon.h"

template <typename Tcomplex>
class GauNoiseAdder {
public:
    GauNoiseAdder(uint32_t nThreads, int seed, cudaStream_t strm);
    ~GauNoiseAdder();

    void addNoise(Tcomplex* d_signal, uint32_t signalSize, float snr_db);

private:
    uint32_t m_nThreads;
    int m_seed;
    uint32_t m_threadsPerBlock;
    uint32_t m_blocksPerGrid;
    float m_noiseStd;
    curandState* m_d_states;
    cudaStream_t m_strm;
    cudaFunction_t m_functionPtr;
};

// Explicitly instantiate the template to resovle "undefined functions"
template class GauNoiseAdder<__half2>;
template class GauNoiseAdder<cuComplex>;

// Kernel for initializing cuRAND states
static __global__ void init_curand_states(curandState* states, unsigned long seed, uint32_t size);

// CUDA kernel for adding Gaussian noise
template <typename Tcomplex>
static __global__ void add_gaussian_noise(Tcomplex* signal, uint32_t signalSize, float noise_std, curandState* states);

#endif // GAU_NOISE_ADDER_CUH