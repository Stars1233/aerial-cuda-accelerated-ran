/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * Test kernels for the warp-synchronous PDCCH polar encoder: launch wrappers so
 * test_polar_encoder.cpp (compiled with g++) can compare encodeRateMatchPdcchWarp
 * word-for-word against the block-wide reference (encode_pdcch_pbch_LUT + rateMatch).
 * Both implementations come from the same shared header, so this is a pure
 * equivalence check of the two encode/rate-match code paths.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "polar_encoder_pdcch.cuh"

namespace {

// Block-wide reference path, mirroring encodeRateMatchMultipleDCIsKernel's non-TM branch.
__global__ void refEncodeRateMatchKernel(uint32_t K, uint32_t N, uint32_t E,
                                         const uint8_t*  pInfoBits,
                                         const uint16_t* cidx2uidx,
                                         uint8_t*        pCodedBits,
                                         uint8_t*        pTxBits)
{
    __shared__ __align__(sizeof(uint32_t)) uint8_t smem[polar_encoder::N_RM_SCRATCH_BYTES];
    polar_encoder::encode_pdcch_pbch_LUT(K, N, E, pInfoBits, cidx2uidx, reinterpret_cast<uint32_t*>(smem), pCodedBits);
    polar_encoder::rateMatch(K, N, E, pCodedBits, reinterpret_cast<uint32_t*>(smem), pTxBits);
}

// Warp-synchronous path, mirroring fusedPdcchTxKernel's warp-0 phase A.
__global__ void warpEncodeRateMatchKernel(uint32_t K, uint32_t N, uint32_t E,
                                          const uint8_t*  pInfoBits,
                                          const uint16_t* cidx2uidx,
                                          uint32_t*       pTxWords)
{
    __shared__ uint32_t scratch[polar_encoder::N_MAX_CODED_BITS / 32];
    polar_encoder::encodeRateMatchPdcchWarp(K, N, E, pInfoBits, cidx2uidx, scratch, pTxWords);
}

// Resolve a kernel symbol to a CUfunction once (cudaGetFuncBySymbol is the CE.1-allowed
// runtime-API exception) and launch via the driver API — no triple-chevron (rule 1.6).
CUfunction resolveKernel(const void* symbol)
{
    CUfunction func = nullptr;
    if(cudaGetFuncBySymbol(&func, symbol) != cudaSuccess)
    {
        return nullptr; // cuLaunchKernel(nullptr, ...) fails; caught by the test's error checks
    }
    return func;
}

} // namespace

extern "C" {

CUresult testLaunchRefEncodeRateMatch(uint32_t K, uint32_t N, uint32_t E,
                                      const uint8_t* d_info, const uint16_t* d_lut,
                                      uint8_t* d_coded, uint8_t* d_tx, CUstream strm)
{
    static const CUfunction func   = resolveKernel(reinterpret_cast<const void*>(refEncodeRateMatchKernel));
    void*                   args[] = {&K, &N, &E, &d_info, &d_lut, &d_coded, &d_tx};
    return cuLaunchKernel(func, 1, 1, 1, 128, 1, 1, 0, strm, args, nullptr);
}

CUresult testLaunchWarpEncodeRateMatch(uint32_t K, uint32_t N, uint32_t E,
                                       const uint8_t* d_info, const uint16_t* d_lut,
                                       uint32_t* d_tx, CUstream strm)
{
    static const CUfunction func   = resolveKernel(reinterpret_cast<const void*>(warpEncodeRateMatchKernel));
    void*                   args[] = {&K, &N, &E, &d_info, &d_lut, &d_tx};
    return cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, strm, args, nullptr);
}

// Host-side access to the shared sizing/LUT-offset helpers for the test driver.
uint32_t testPdcchPolarNumCodedBits(uint32_t nInfoBits, uint32_t aggrLevel)
{
    return polar_encoder::pdcchPolarNumCodedBits(nInfoBits, aggrLevel);
}

int testPdcchLutElemOffset(int K, int log2AL)
{
    return polar_encoder::pdcch_polar_cidx2uidx_lut_elem_offset(K, log2AL);
}

} // extern "C"
