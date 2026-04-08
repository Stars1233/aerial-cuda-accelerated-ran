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

#if !defined(OFDM_COMMON_FUNC_H_INCLUDED_)
#define OFDM_COMMON_FUNC_H_INCLUDED_

#define USE_MEMOERY_FFT_SHIFT_ // use fft shift in memeory or multiplication
#define OFDM_FFTs_PER_BLOCK_CONST_ 2 // FFTs per block in ofdm modulation and demodulation, changing this will affect run times, should be a divisor of OFDM_SYMBOLS_PER_SLOT (14)

#include <cuda.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cmath>
#include "../chanModelsCommon.h"

/**
 * @brief the carriers parameters with default values
 * 
 */
typedef struct cuphyCarrierPrms
{
    uint16_t N_sc = 3276; // 12 * num of RBs
    uint16_t N_FFT = 4096;  // also N_IFFT
    uint16_t N_bsLayer = 4; 
    uint16_t N_ueLayer = 4;
    uint16_t id_slot = 0;  // per sub frame
    uint16_t id_subFrame = 0; // per frame
    uint16_t mu = 1; // numerology
    uint16_t cpType = 0; // 0 for normal CP, 1 for extended CP
    uint32_t f_c = 480e3 * 4096; // delta_f_max * N_f based on 38.211
    float T_c = 5.0863e-10;
    uint32_t f_samp = 15e3 * 8192; // 15e3 * 2^mu * Nfft
    uint16_t N_symbol_slot = 14; // 14 OFDMs per slot per normal CP and 12 for extended CP
    uint16_t k_const = 64;  // kappa = 64 (2^6); constants defined in 38.211
    uint16_t kappa_bits = 6; // kappa in bits
    uint32_t ofdmWindowLen = 0; // ofdm windowing
    float rolloffFactor = 0.5; // rolloff factor for rcos in windowing
    uint32_t N_samp_slot = 61440;

    // PRACH parameters
    uint32_t N_u_mu = 65536;
    uint32_t startRaSym = 0;
    uint32_t delta_f_RA = 30000;
    uint32_t N_CP_RA = 29952;
    uint32_t K = 1;
    int32_t  k1 = -1638;
    uint32_t kBar = 2;
    uint32_t N_u = 786432;
    uint32_t L_RA = 139;
    uint32_t n_slot_RA_sel = 0;
    uint32_t N_rep = 12;
} cuphyCarrierPrms_t;

/**
 * @brief launch configuration structure, includes driver info and kernel input variables
 * 
 */
typedef struct {
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} ofdmLaunchCfg_t;


#endif // !defined(OFDM_COMMON_FUNC_H_INCLUDED_)