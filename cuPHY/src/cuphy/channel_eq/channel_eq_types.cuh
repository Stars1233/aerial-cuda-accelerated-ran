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

#pragma once

#include "channel_eq/channel_eq.hpp"
#include "cuphy_kernel_util.cuh"
#include "cuphy_complex_ops.cuh"

namespace channel_eq {

//=============================================================================
// Constants and Configuration
//=============================================================================

static constexpr uint32_t CUDA_MAX_N_THRDS_PER_BLK = 1024;

// FP16 - largest normal number
static constexpr float LLR_LOW_LIM  = -65504.0f; // std::numeric_limits<__half>::lowest();
static constexpr float LLR_HIGH_LIM =  65504.0f; // std::numeric_limits<__half>::max();

// Inverse of zero-forcing regularizer. Equivalent to diagonal MMSE with SNR = 10^(3.6)
static constexpr float INV_ZF_REGULARIZER = 3981.071705534973f;

// Import types from cuphy_cmplx (explicit declarations to avoid ambiguity)
using cuphy_cmplx::tensor_ref;
using cuphy_cmplx::block_1D;
using cuphy_cmplx::block_2D;
using cuphy_cmplx::block_3D;

// Import functions and operators from cuphy_cmplx
using namespace cuphy_cmplx;

//=============================================================================
// Debug Functions
//=============================================================================
//#define CUPHY_DEBUG 1

#if CUPHY_DEBUG
__device__ float debug_LLR_get_elem(const float4& Llr, int idx)
{
    switch(idx)
    {
    default:
    case 0: return Llr.x;
    case 1: return Llr.y;
    case 2: return Llr.z;
    case 3: return Llr.w;
    }
}

__device__ float debug_LLR_get_elem(const float2& Llr, int idx)
{
    cuphy_i::word_t w0, w1;
    w0.f32 = Llr.x;
    w1.f32 = Llr.y;
    switch(idx)
    {
    default:
    case 0: return __low2float(w0.f16x2);
    case 1: return __high2float(w0.f16x2);
    case 2: return __low2float(w1.f16x2);
    case 3: return __high2float(w1.f16x2);
    }
}
#endif

// clang-format on

//=============================================================================
// QAM Tag Mapping Templates
//=============================================================================

// Use tag dispatching to invoke different behaviours (different functions) for each QAM
template <QAM_t>
struct QAMEnumToTagMap;

// Only even QAMs supported by 3GPP
template <>
struct QAMEnumToTagMap<QAM_t::QAM_4>
{
    struct QAM4Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_16>
{
    struct QAM16Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_64>
{
    struct QAM64Tag {};
};

template <>
struct QAMEnumToTagMap<QAM_t::QAM_256>
{
    struct QAM256Tag {};
};

} // namespace channel_eq
