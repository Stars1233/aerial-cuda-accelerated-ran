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

#include "pucch_F0_receiver.hpp"
#include "descrambling.cuh"
#include <functional>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cuComplex.h"
#include "cuda_fp16.h"
#include "math_utils.cuh"
#include "cuphy.hpp"
#include <cassert>
#include <cmath>

namespace cg = cooperative_groups;

bool pucchF0Rx::isConstMemInited = false;
std::mutex pucchF0Rx::m_mutexConstMemInit;


namespace pucch_F0
{
/*VCAST_DONT_INSTRUMENT_START*/
template <typename TElem>
struct tensor_ref
{
    TElem*         addr;
    const int32_t* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
        addr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }
    CUDA_BOTH int offset(int i0) const
    {
        return (strides[0] * i0);
    }
    CUDA_BOTH int offset(int i0, int i1) const
    {
        return (strides[0] * i0) + (strides[1] * i1);
    }
    CUDA_BOTH int offset(int i0, int i1, int i2) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2);
    };
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3);
    };
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                               { return *(addr + offset(i0));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                       { return *(addr + offset(i0, i1));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)               { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)       { return *(addr + offset(i0, i1, i2, i3)); }

    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(addr + offset(i0));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(addr + offset(i0, i1));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }
    // clang-format on
};
/*VCAST_DONT_INSTRUMENT_END*/

// Lookup tables in constant memory. These are transposed from MATLAB to make accesses row-major
static __device__ __constant__ __half2 d_rBase[30][12];
static __half2                         rBase[30][12] = {{{-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}},
                                {{0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}},
                                {{0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}},
                                {{0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}},
                                {{0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {-0.707031, -0.707031}},
                                {{0.707031, 0.707031}, {0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {0.707031, 0.707031}},
                                {{-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}, {-0.707031, -0.707031}, {-0.707031, 0.707031}, {0.707031, -0.707031}, {0.707031, -0.707031}, {0.707031, 0.707031}, {-0.707031, 0.707031}, {-0.707031, -0.707031}}};

static __device__ __constant__ __half2 d_csPhaseRamp[12][12];
static __half2                         csPhaseRamp[12][12] = {{{1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}, {1.000000, 0.000000}},
                                      {{1.000000, 0.000000}, {0.866211, 0.500000}, {0.500000, 0.866211}, {0.000000, 1.000000}, {-0.500000, 0.866211}, {-0.866211, 0.500000}, {-1.000000, 0.000000}, {-0.866211, -0.500000}, {-0.500000, -0.866211}, {0.000000, -1.000000}, {0.500000, -0.866211}, {0.866211, -0.500000}},
                                      {{1.000000, 0.000000}, {0.500000, 0.866211}, {-0.500000, 0.866211}, {-1.000000, 0.000000}, {-0.500000, -0.866211}, {0.500000, -0.866211}, {1.000000, 0.000000}, {0.500000, 0.866211}, {-0.500000, 0.866211}, {-1.000000, 0.000000}, {-0.500000, -0.866211}, {0.500000, -0.866211}},
                                      {{1.000000, 0.000000}, {0.000000, 1.000000}, {-1.000000, 0.000000}, {0.000000, -1.000000}, {1.000000, 0.000000}, {0.000000, 1.000000}, {-1.000000, 0.000000}, {0.000000, -1.000000}, {1.000000, 0.000000}, {0.000000, 1.000000}, {-1.000000, 0.000000}, {0.000000, -1.000000}},
                                      {{1.000000, 0.000000}, {-0.500000, 0.866211}, {-0.500000, -0.866211}, {1.000000, 0.000000}, {-0.500000, 0.866211}, {-0.500000, -0.866211}, {1.000000, 0.000000}, {-0.500000, 0.866211}, {-0.500000, -0.866211}, {1.000000, 0.000000}, {-0.500000, 0.866211}, {-0.500000, -0.866211}},
                                      {{1.000000, 0.000000}, {-0.866211, 0.500000}, {0.500000, -0.866211}, {0.000000, 1.000000}, {-0.500000, -0.866211}, {0.866211, 0.500000}, {-1.000000, 0.000000}, {0.866211, -0.500000}, {-0.500000, 0.866211}, {0.000000, -1.000000}, {0.500000, 0.866211}, {-0.866211, -0.500000}},
                                      {{1.000000, 0.000000}, {-1.000000, 0.000000}, {1.000000, 0.000000}, {-1.000000, 0.000000}, {1.000000, 0.000000}, {-1.000000, 0.000000}, {1.000000, 0.000000}, {-1.000000, 0.000000}, {1.000000, 0.000000}, {-1.000000, 0.000000}, {1.000000, 0.000000}, {-1.000000, 0.000000}},
                                      {{1.000000, 0.000000}, {-0.866211, -0.500000}, {0.500000, 0.866211}, {0.000000, -1.000000}, {-0.500000, 0.866211}, {0.866211, -0.500000}, {-1.000000, 0.000000}, {0.866211, 0.500000}, {-0.500000, -0.866211}, {0.000000, 1.000000}, {0.500000, -0.866211}, {-0.866211, 0.500000}},
                                      {{1.000000, 0.000000}, {-0.500000, -0.866211}, {-0.500000, 0.866211}, {1.000000, 0.000000}, {-0.500000, -0.866211}, {-0.500000, 0.866211}, {1.000000, 0.000000}, {-0.500000, -0.866211}, {-0.500000, 0.866211}, {1.000000, 0.000000}, {-0.500000, -0.866211}, {-0.500000, 0.866211}},
                                      {{1.000000, 0.000000}, {0.000000, -1.000000}, {-1.000000, 0.000000}, {0.000000, 1.000000}, {1.000000, 0.000000}, {0.000000, -1.000000}, {-1.000000, 0.000000}, {0.000000, 1.000000}, {1.000000, 0.000000}, {0.000000, -1.000000}, {-1.000000, 0.000000}, {0.000000, 1.000000}},
                                      {{1.000000, 0.000000}, {0.500000, -0.866211}, {-0.500000, -0.866211}, {-1.000000, 0.000000}, {-0.500000, 0.866211}, {0.500000, 0.866211}, {1.000000, 0.000000}, {0.500000, -0.866211}, {-0.500000, -0.866211}, {-1.000000, 0.000000}, {-0.500000, 0.866211}, {0.500000, 0.866211}},
                                      {{1.000000, 0.000000}, {0.866211, -0.500000}, {0.500000, -0.866211}, {0.000000, -1.000000}, {-0.500000, -0.866211}, {-0.866211, -0.500000}, {-1.000000, 0.000000}, {-0.866211, 0.500000}, {-0.500000, 0.866211}, {0.000000, 1.000000}, {0.500000, 0.866211}, {0.866211, 0.500000}}};

/**
  * PUCCH Format 0 Receiver
  * 
  * Implements the PUCCH format 0 message per 3GPP TS 38.211. This is considered a 
  * "low priority" kernel, so it attempts to use very few resources on a device, and is
  * not heavily optimized.
  * 
  * @param pDynDescr
  *   Pointer to all input parameters for UCIs
  * 
  **/
static __global__ void
pucchF0RxKernel(pucchF0RxDynDescr_t* pDynDescr)
{
    auto tile = cg::tiled_partition<F0_CG_SIZE>(cg::this_thread_block());

    // Global UCI group across all CTAs
    uint16_t global_group = blockIdx.x * blockDim.y + threadIdx.y;

    if(global_group >= pDynDescr->numUciGrps) {
        return;
    }

    // UCI group local to this CTA
    auto local_group = tile.meta_group_rank();

    // UCI data for this group
    auto group_data = &pDynDescr->uciGrpPrms[global_group];

    // cell idx
    auto    cellIdx  = group_data->cellIdx;
    int16_t numRxAnt;
    if(pDynDescr->enableUlRxBf)
    {
       numRxAnt = group_data->nUplinkStreams;
    }
    else
    {
       numRxAnt = pDynDescr->pCellPrms[cellIdx].nRxAnt;
    } 

    __shared__ __half2 y_corr[F0_GROUPS_PER_BLOCK * MAX_SYMS_F0 * N_TONES_PER_PRB * MAX_RX_ANTENNA];
    __shared__ __half2 r1[F0_GROUPS_PER_BLOCK * N_TONES_PER_PRB];
    __shared__ __half2 r2[F0_GROUPS_PER_BLOCK * N_TONES_PER_PRB];
    __shared__ float   cor_array[F0_GROUPS_PER_BLOCK * MAX_SYMS_F0 * N_TONES_PER_PRB];

    // Per-group shared memory carveouts
    __half2* group_y  = &y_corr[local_group * MAX_SYMS_F0 * N_TONES_PER_PRB * MAX_RX_ANTENNA];
    __half2* group_r1 = &r1[local_group * N_TONES_PER_PRB];
    __half2* group_r2 = &r2[local_group * N_TONES_PER_PRB];

    float* group_cor_arr = &cor_array[local_group * N_TONES_PER_PRB * MAX_SYMS_F0];

    // Populate group sequences using gold codes
    {
        uint16_t pucchHoppingId = pDynDescr->pCellPrms[cellIdx].pucchHoppingId;
        uint16_t slotNum        = pDynDescr->pCellPrms[cellIdx].slotNum;

        // sequence number (38.211 6.3.2.2.1)
        if(group_data->groupHopFlag == 0)
        {
            group_data->u[0] = pucchHoppingId % 30;
            group_data->u[1] = pucchHoppingId % 30;
        }else
        {
            uint32_t g   = descrambling::gold32n(pucchHoppingId / 30, 16 * slotNum);
            uint8_t f_ss = pucchHoppingId % 30;

            // first hop
            uint8_t f_gh = (g & LOWER_BYTE_BMSK) % 30;
            group_data->u[0] = (f_ss + f_gh) % 30;

            // second hop
            if(group_data->freqHopFlag)
            {
                f_gh = ((g >> 8) & LOWER_BYTE_BMSK) % 30;
                group_data->u[1] = (f_ss + f_gh) % 30;

            }else
            {
                group_data->u[1] = group_data->u[0];
            }
        }

        // common cyclic shift (38.211 6.3.2.2.2)
        uint32_t g  = descrambling::gold32n(pucchHoppingId, 14*8*slotNum + 8*group_data->startSym);
        group_data->csCommon[0] = (g & LOWER_BYTE_BMSK);
        group_data->csCommon[1] = (g >> 8)  & LOWER_BYTE_BMSK;
    }

    // Lane without our group
    int lane = tile.thread_rank();

    tensor_ref<const __half2> tDataRx(pDynDescr->pCellPrms[cellIdx].tDataRx.pAddr, pDynDescr->pCellPrms[cellIdx].tDataRx.strides);

    // Pull in all values needed to construct Y_pucch from the MATLAB code. Because of the relatively low amount of data to fetch, and the
    // layout of this data, the access patterns are not ideal. Because of this, we switch how we access the data once we get to 8 antennas or more.
    // With fewer than 8 antennas we fetch by PRB first in an unrolled loop on the antenna number. With 8 and higher every thread participates
    // in accessing every PRB, but only a single antenna value each.
    // switch to constexpr when c++17 is supported potentially
    int startSc = (group_data->startPrb + group_data->bwpStart) * N_TONES_PER_PRB;
    switch(numRxAnt)
    {
    case 1: {
        if(lane < N_TONES_PER_PRB)
        {
            group_y[lane * 1] = tDataRx(startSc + lane, group_data->startSym, 0);
        }
        break;
    }
    case 2: {
        if(lane < N_TONES_PER_PRB)
        {
#pragma unroll
            for(int m = 0; m < 2; m++)
            {
                group_y[lane * 2 + m] = tDataRx(startSc + lane, group_data->startSym, m);
            }
        }
        break;
    }
    case 4: {
        if(lane < N_TONES_PER_PRB)
        {
#pragma unroll
            for(int m = 0; m < 4; m++)
            {
                group_y[lane * 4 + m] = tDataRx(startSc + lane, group_data->startSym, m);
            }
        }
        break;
    }
    case 8: {
        if(lane < 8)
        {
#pragma unroll
            for(int tone = 0; tone < N_TONES_PER_PRB; tone++)
            {
                group_y[N_TONES_PER_PRB * tone + lane] = tDataRx(startSc + tone, group_data->startSym, lane);
            }
        }
        break;
    }
    case 16: {
        if(lane < 16)
        {
#pragma unroll
            for(int tone = 0; tone < N_TONES_PER_PRB; tone++)
            {
                group_y[N_TONES_PER_PRB * tone + lane] = tDataRx(startSc + tone, group_data->startSym, lane);
            }
        }

        break;
    }
    default:
        return;
    }

    // Same loop above but on the second symbol (if applicable)
    if(group_data->freqHopFlag || group_data->nSym > 1)
    {
        int secondStartSc = group_data->freqHopFlag ? group_data->secondHopPrb : group_data->startPrb;
        secondStartSc += group_data->bwpStart;
        secondStartSc *= N_TONES_PER_PRB;
        switch(numRxAnt)
        {
        case 1: {
            if(lane < N_TONES_PER_PRB)
            {
                group_y[N_TONES_PER_PRB + lane] = tDataRx(secondStartSc + lane, group_data->startSym + 1, 0);
            }
            break;
        }
        case 2: {
            if(lane < N_TONES_PER_PRB)
            {
#pragma unroll
                for(int m = 0; m < 2; m++)
                {
                    group_y[N_TONES_PER_PRB * 2 + lane * 2 + m] = tDataRx(secondStartSc + lane, group_data->startSym + 1, m);
                }
            }
            break;
        }
        case 4: {
            if(lane < N_TONES_PER_PRB)
            {
#pragma unroll
                for(int m = 0; m < 4; m++)
                {
                    group_y[N_TONES_PER_PRB * 4 + lane * 4 + m] = tDataRx(secondStartSc + lane, group_data->startSym + 1, m);
                }
            }
            break;
        }
        case 8: {
            if(lane < 8)
            {
#pragma unroll
                for(int tone = 0; tone < N_TONES_PER_PRB; tone++)
                {
                    group_y[N_TONES_PER_PRB * 8 + tone * 8 + lane] = tDataRx(secondStartSc + tone, group_data->startSym + 1, lane);
                }
            }

            break;
        }
        case 16: {
#pragma unroll
            for(int tone = 0; tone < N_TONES_PER_PRB; tone++)
            {
                group_y[N_TONES_PER_PRB * 16 + tone * 8 + lane] = tDataRx(secondStartSc + tone, group_data->startSym + 1, lane);
            }

            break;
        }
        /*VCAST_DONT_INSTRUMENT_START*/
        default:
            return;
        }
        /*VCAST_DONT_INSTRUMENT_END*/
    }

    // Fetch rBase values for first and second symbol
    if(lane < N_TONES_PER_PRB)
    {
        group_r1[lane] = d_rBase[group_data->u[0]][lane];

        if(group_data->freqHopFlag && group_data->groupHopFlag)
        {
            group_r2[lane] = d_rBase[group_data->u[1]][lane];
        }
        else
        {
            group_r2[lane] = d_rBase[group_data->u[0]][lane];
        }
    }

    tile.sync();

    float corr1 = 0;
    float corr2 = 0;
    float2 tmpf;

    float rssi_linear_temp = 0;

    // Loop through computing the outer product for each correlation array entry. Each thread in a group accumulates a single
    // PRB value for all antennas. The results are stored in the corr1/2 registers for the first and second symbol, respectively.
    if(lane < N_TONES_PER_PRB)
    {
        for(int ant = 0; ant < numRxAnt; ant++)
        {
            tmpf = __half22float2(group_y[lane* numRxAnt + ant]);
            rssi_linear_temp += tmpf.x*tmpf.x + tmpf.y*tmpf.y;

            // Inner product per thread for each corArray element
            __half2 tmp = {0, 0};
#pragma unroll
            for(int i = 0; i < N_TONES_PER_PRB; i++)
            {
                tmp += complex_conjmul(complex_mul(group_r1[i], d_csPhaseRamp[lane][i]), group_y[i * numRxAnt + ant]);
            }

            tmpf = __half22float2(tmp); //Convert to float before squaring to prevent overflow to inf.
            corr1 += tmpf.x * tmpf.x + tmpf.y * tmpf.y;
        }


        group_cor_arr[lane] = corr1;

        // If we're doing a second symbol, do it here
        if(group_data->freqHopFlag || group_data->nSym > 1)
        {
            for(int ant = 0; ant < numRxAnt; ant++)
            {
                tmpf = __half22float2(group_y[N_TONES_PER_PRB * numRxAnt + lane* numRxAnt + ant]);
                rssi_linear_temp += tmpf.x*tmpf.x + tmpf.y*tmpf.y;

                __half2 tmp = {0, 0};

// Inner product per thread for each corArray element
#pragma unroll
                for(int i = 0; i < N_TONES_PER_PRB; i++)
                {
                    tmp += complex_conjmul(complex_mul(group_r2[i], d_csPhaseRamp[lane][i]), group_y[N_TONES_PER_PRB * numRxAnt + i * numRxAnt + ant]);
                }

                tmpf = __half22float2(tmp);
                corr2 += tmpf.x * tmpf.x + tmpf.y * tmpf.y;
            }

            group_cor_arr[N_TONES_PER_PRB + lane] = corr2;
        }
    }

    tile.sync();

    // Summ correlation values across all PRBs
    float sumcorr1 = cg::reduce(tile, corr1, cg::plus<float>());
    float sumcorr2 = cg::reduce(tile, corr2, cg::plus<float>());

    // Summ rssi values across all subcarriers
    float rssi_linear = cg::reduce(tile, rssi_linear_temp, cg::plus<float>()) / static_cast<float>(group_data->nSym);

    // Scaling parameter for DTXthreshold
    float beta = 7.7/sqrt(numRxAnt*group_data->nSym);

    // Process all UCIs
    uint32_t index;
    float    max_corr;

    // Loop through each UCI in the group. The thread pattern changes here to where each thread is now responsible for a single UCI in a
    // group, and all threads access the same correlation array computed in shared memory above

    __shared__ float noiseCorr;

    if (tile.thread_rank() == 0)
        noiseCorr = sumcorr1+sumcorr2;

    tile.sync();

    for(int uci = lane; uci < group_data->nUciInGrp; uci += F0_CG_SIZE)
    {
        // Find the max correlation score and the adjusted index for the correlation score. The adjusted index is computed similar to
        // this formula in the simulation: csArray(cs_i, i) = mod(csCommon(i) + cs0 + m_cs, 12);
        if(group_data->bitLenHarq[uci] == 0)
        {
            if(group_data->srFlag[uci])
            {
                UpdateCsArray<1>(group_data->csCommon, group_data->nSym, group_data->cs0[uci], group_cor_arr, &index, &max_corr);
            }
        }
        else if(group_data->bitLenHarq[uci] == 1)
        {
            if(group_data->srFlag[uci])
            {
                UpdateCsArray<4>(group_data->csCommon, group_data->nSym, group_data->cs0[uci], group_cor_arr, &index, &max_corr);
            }
            else
            {
                UpdateCsArray<2>(group_data->csCommon, group_data->nSym, group_data->cs0[uci], group_cor_arr, &index, &max_corr);
            }
        }
        else
        {
            if(group_data->srFlag[uci])
            {
                UpdateCsArray<8>(group_data->csCommon, group_data->nSym, group_data->cs0[uci], group_cor_arr, &index, &max_corr);
            }
            else
            {
                UpdateCsArray<4>(group_data->csCommon, group_data->nSym, group_data->cs0[uci], group_cor_arr, &index, &max_corr);
            }
        }

        atomicAdd(&noiseCorr, -max_corr);
    }

    tile.sync();

    for(int uci = lane; uci < group_data->nUciInGrp; uci += F0_CG_SIZE)
    {
        float dtx_thresh = noiseCorr * beta * __half2float(group_data->DTXthreshold[uci])/static_cast<float>(12-group_data->nUciInGrp);

        int      sr            = 0;
        int      pucch_payload = 0;
        uint32_t uciIdx        = group_data->uciOutputIdx[uci];

        float rsrpTemp = max_corr/static_cast<float>(group_data->nSym)/static_cast<float>(numRxAnt)/144.0; // 144.0 == N_TONES_PER_PRB^2
        float dB_rsrp = 10*log10(rsrpTemp); 

        float dB_rssi = 10*log10(rssi_linear);

        // RSSI report    
        pDynDescr->pF0UcisOut[uciIdx].RSSI = dB_rssi;

        // RSRP report
        pDynDescr->pF0UcisOut[uciIdx].RSRP = dB_rsrp;

        // Determine SR/HARQ confidence levels
        // According to SCF FAPI, Table 3-68, 0 stands for "Good" and 1 stands for "Bad"
        uint8_t SRconfidenceLevel   = 0;
        uint8_t HarqconfidenceLevel = 0;

        float gapPercDtx = (max_corr - dtx_thresh) / max_corr;

        if(gapPercDtx < confidenceThrF0)
        {
            SRconfidenceLevel   = 1;
            HarqconfidenceLevel = 1;
        }

        // Calculate the SR and payload values based on the index and HARQ bit len + SR flag
        if(max_corr > dtx_thresh)
        {
            if(group_data->bitLenHarq[uci] == 0)
            {
                sr = 1;
            }
            else if(group_data->bitLenHarq[uci] == 1)
            {
                if(group_data->srFlag[uci])
                {
                    pucch_payload = index & 1;
                    sr            = (__popc(index) == 1);
                }
                else
                {
                    pucch_payload = index;
                }
            }
            else if(group_data->bitLenHarq[uci] == 2)
            {
                if(group_data->srFlag[uci])
                {
                    pucch_payload = index >> 1;
                    sr            = index & 1;
                }
                else
                {
                    pucch_payload = index;
                }
            }

            // Write out payload on detection
            pDynDescr->pF0UcisOut[uciIdx].HarqValues[0] = pucch_payload & 1;
            pDynDescr->pF0UcisOut[uciIdx].HarqValues[1] = (pucch_payload & 2) >> 1;
        } else {
            if(group_data->bitLenHarq[uci] > 0)
            {
                // Nothing detected. Use 2 as placeholder
                pDynDescr->pF0UcisOut[uciIdx].HarqValues[0] = 2;
                pDynDescr->pF0UcisOut[uciIdx].HarqValues[1] = 2;
            }
        }
        pDynDescr->pF0UcisOut[uciIdx].SinrDB              = 0; // TODO add SINR report
        pDynDescr->pF0UcisOut[uciIdx].InterfDB            = 0; // TODO add interference + noise level report
        pDynDescr->pF0UcisOut[uciIdx].SRindication        = sr;
        pDynDescr->pF0UcisOut[uciIdx].NumHarq             = group_data->bitLenHarq[uci];
        pDynDescr->pF0UcisOut[uciIdx].SRconfidenceLevel   = SRconfidenceLevel;
        pDynDescr->pF0UcisOut[uciIdx].HarqconfidenceLevel = HarqconfidenceLevel;
        pDynDescr->pF0UcisOut[uciIdx].taEstMicroSec       = 0; // TODO add TA Estimation
    }
}

} // namespace pucch_F0


using namespace pucch_F0;
/**
 * Loads the rBase and csPhaseRamp tables into constant memory
 */
void pucchF0Rx::InitConstantMem(cudaStream_t strm)
{
   CUDA_CHECK(cudaMemcpyToSymbolAsync(d_rBase, rBase, sizeof(rBase), 0, cudaMemcpyHostToDevice, strm));
   CUDA_CHECK(cudaMemcpyToSymbolAsync(d_csPhaseRamp, csPhaseRamp, sizeof(csPhaseRamp), 0, cudaMemcpyHostToDevice, strm));
}

pucchF0Rx::pucchF0Rx(cudaStream_t strm)
{
  std::lock_guard lockGuardConstMemInit(m_mutexConstMemInit);
  if(!isConstMemInited)
  {
     isConstMemInited = true;
     InitConstantMem(strm);
  }
}



void  pucchF0Rx::kernelSelect(uint16_t                   nUciGrps,
                              cuphyPucchF0RxLaunchCfg_t* pLaunchCfg)
{
   // kernel (only one kernel option for now)
   void* kernelFunc = reinterpret_cast<void*>(pucchF0RxKernel);
   {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}

   // launch geometry (can change!)

   dim3 gridDim((nUciGrps+ F0_GROUPS_PER_BLOCK -1)/ F0_GROUPS_PER_BLOCK);
   dim3 blockDim(F0_CG_SIZE, F0_GROUPS_PER_BLOCK);

   // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void pucchF0Rx::setup(cuphyTensorPrm_t*          pDataRx,                     // input slot buffer
                      cuphyPucchF0F1UciOut_t*    pF0UcisOut,                  // pointer to output uci buffer
                      uint16_t                   nCells,                      // number of cells
                      uint16_t                   nF0Ucis,                     // number of F01 Ucis
                      uint8_t                    enableUlRxBf,
                      cuphyPucchUciPrm_t*        pF0UciPrms,                  // pointer to uci prm buffer
                      cuphyPucchCellPrm_t*       pCmnCellPrms,                // number of antennas, slot number, hopping idx and input slot buffer
                      bool                       enableCpuToGpuDescrAsyncCpy, // flag, indicates if descriptors copied to gpu at setup
                      pucchF0RxDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                      void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                      cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                      cudaStream_t strm) // stream to perform copy
{
   // Bin Ucis into groups based on start prb and symbol:
   uint16_t nUciGrps = 0;
   pCpuDynDesc->pCellPrms = pCmnCellPrms;

   for(int uciIdx = 0; uciIdx < nF0Ucis; ++uciIdx)
   {
       int      newGrpFlag  = 1;
       uint16_t uciStartCrb = pF0UciPrms[uciIdx].startPrb + pF0UciPrms[uciIdx].bwpStart;
       uint8_t  uciStartSym = pF0UciPrms[uciIdx].startSym;
       uint16_t uciCellIdx  = pF0UciPrms[uciIdx].cellPrmDynIdx;
       float extDTXthreshold =  pF0UciPrms[uciIdx].DTXthreshold;

       for(int grpIdx = 0; grpIdx < nUciGrps; grpIdx++)
      {
         uint16_t grpStartCrb = pCpuDynDesc->uciGrpPrms[grpIdx].startPrb + pCpuDynDesc->uciGrpPrms[grpIdx].bwpStart;
         uint8_t  grpStartSym = pCpuDynDesc->uciGrpPrms[grpIdx].startSym;
         uint16_t grpCellIdx  = pCpuDynDesc->uciGrpPrms[grpIdx].cellIdx;

         if((uciStartCrb == grpStartCrb) && (uciStartSym == grpStartSym) && (uciCellIdx == grpCellIdx))
         {
            newGrpFlag = 0;
            uint8_t nUciInGrp = pCpuDynDesc->uciGrpPrms[grpIdx].nUciInGrp;
            if(CUPHY_PUCCH_F0_MAX_UCI_PER_GRP <= nUciInGrp)
            {
                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Number of PF0 UCIs in group {} is more than max allocation ({}).  Dropping additional UCIs.",grpIdx,CUPHY_PUCCH_F0_MAX_UCI_PER_GRP);
                break;
            }
            pCpuDynDesc->uciGrpPrms[grpIdx].nUciInGrp                 = nUciInGrp + 1;
            pCpuDynDesc->uciGrpPrms[grpIdx].bitLenHarq[nUciInGrp]     = pF0UciPrms[uciIdx].bitLenHarq;
            pCpuDynDesc->uciGrpPrms[grpIdx].srFlag[nUciInGrp]         = pF0UciPrms[uciIdx].srFlag;
            pCpuDynDesc->uciGrpPrms[grpIdx].cs0[nUciInGrp]            = pF0UciPrms[uciIdx].initialCyclicShift;
            pCpuDynDesc->uciGrpPrms[grpIdx].uciOutputIdx[nUciInGrp]   = pF0UciPrms[uciIdx].uciOutputIdx;
            if(extDTXthreshold > CUPHY_DEFAULT_EXT_DTX_THRESHOLD)
            {
                pCpuDynDesc->uciGrpPrms[grpIdx].DTXthreshold[nUciInGrp] = __float2half(extDTXthreshold);
            } else {
                pCpuDynDesc->uciGrpPrms[grpIdx].DTXthreshold[nUciInGrp] = 1.0;
            }
            break;
         }
      }

      if(newGrpFlag == 1)
      {
        if(CUPHY_PUCCH_F0_MAX_GRPS <= nUciGrps)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Number of PF0 UCI groups is more than max allocation ({}).  Dropping additional groups.",CUPHY_PUCCH_F0_MAX_GRPS);
            continue;
        }
         // uci group prms:
         pCpuDynDesc->uciGrpPrms[nUciGrps].nUciInGrp    = 1;
         pCpuDynDesc->uciGrpPrms[nUciGrps].freqHopFlag  = pF0UciPrms[uciIdx].freqHopFlag;
         pCpuDynDesc->uciGrpPrms[nUciGrps].bwpStart     = pF0UciPrms[uciIdx].bwpStart;
         pCpuDynDesc->uciGrpPrms[nUciGrps].startPrb     = pF0UciPrms[uciIdx].startPrb;
         pCpuDynDesc->uciGrpPrms[nUciGrps].startSym     = pF0UciPrms[uciIdx].startSym;
         pCpuDynDesc->uciGrpPrms[nUciGrps].nSym         = pF0UciPrms[uciIdx].nSym;
         pCpuDynDesc->uciGrpPrms[nUciGrps].groupHopFlag = pF0UciPrms[uciIdx].groupHopFlag;
         pCpuDynDesc->uciGrpPrms[nUciGrps].secondHopPrb = pF0UciPrms[uciIdx].secondHopPrb;

         // cell index
         pCpuDynDesc->uciGrpPrms[nUciGrps].cellIdx          = uciCellIdx;

         // uci specfic prms:
         pCpuDynDesc->uciGrpPrms[nUciGrps].bitLenHarq[0]    = pF0UciPrms[uciIdx].bitLenHarq;
         pCpuDynDesc->uciGrpPrms[nUciGrps].cs0[0]           = pF0UciPrms[uciIdx].initialCyclicShift;
         pCpuDynDesc->uciGrpPrms[nUciGrps].uciOutputIdx[0]  = pF0UciPrms[uciIdx].uciOutputIdx;
         if(extDTXthreshold > CUPHY_DEFAULT_EXT_DTX_THRESHOLD)
         {
            pCpuDynDesc->uciGrpPrms[nUciGrps].DTXthreshold[0]  = __float2half(extDTXthreshold);
         } else {
            pCpuDynDesc->uciGrpPrms[nUciGrps].DTXthreshold[0]  = 1.0;
         }
         pCpuDynDesc->uciGrpPrms[nUciGrps].srFlag[0]        = pF0UciPrms[uciIdx].srFlag;
         
         pCpuDynDesc->uciGrpPrms[nUciGrps].nUplinkStreams   = pF0UciPrms[uciIdx].nUplinkStreams;
         
         // update number of uci groups:
         nUciGrps += 1;
      }
   }

   pCpuDynDesc->numUciGrps   = nUciGrps;
   pCpuDynDesc->enableUlRxBf = enableUlRxBf;

   // parameters common to all uci groups:
   for (uint16_t i = 0; i < nCells; i++)
   {
       copyTensorPrm2Info(pDataRx[i], pCpuDynDesc->pCellPrms[i].tDataRx);
   }
   pCpuDynDesc->pF0UcisOut = pF0UcisOut;

   pucchF0KernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<pucchF0RxDynDescr_t*>(pGpuDynDesc);

   // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
      CUDA_CHECK(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(pucchF0RxDynDescr_t), cudaMemcpyHostToDevice, strm));
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nUciGrps, pLaunchCfg);

   pLaunchCfg->kernelArgs[0] = &m_kernelArgs.pDynDescr; 
   pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);

}

 void pucchF0Rx::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
    dynDescrSizeBytes  = sizeof(pucchF0RxDynDescr_t);
    dynDescrAlignBytes = alignof(pucchF0RxDynDescr_t);
 }
