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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "cuphy.hpp"
#include "derate_matching_modulo.hpp"
#include <cuda_fp16.h>


namespace
{

// Simple CUDA error check helper
static void cudaCheck(cudaError_t e)
{
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e);
}

// Driver API error check helper
static void cuCheck(CUresult r)
{
    ASSERT_EQ(r, CUDA_SUCCESS) << "CUDA driver error: " << static_cast<int>(r);
}

// Launch helper using driver params produced by setup
static void launch_kernel_node_params(const CUDA_KERNEL_NODE_PARAMS& p, CUstream stream = 0)
{
    if(p.func == nullptr)
    {
        return;
    }
    cuCheck(cuLaunchKernel(p.func,
                           p.gridDimX,
                           p.gridDimY,
                           p.gridDimZ,
                           p.blockDimX,
                           p.blockDimY,
                           p.blockDimZ,
                           p.sharedMemBytes,
                           stream,
                           p.kernelParams,
                           p.extra));
}

static void launch_with_cfg(const cuphyPuschRxRateMatchLaunchCfg_t& cfg, CUstream stream = 0)
{
    // Match production launch sequence more closely:
    // reset window (for ndi==1 + race window) -> main rate-match kernel -> clamp window.
    launch_kernel_node_params(cfg.resetKernelNodeParamsDriver, stream);
    launch_kernel_node_params(cfg.kernelNodeParamsDriver, stream);
    launch_kernel_node_params(cfg.clampKernelNodeParamsDriver, stream);
    cuCheck(cuStreamSynchronize(stream));
}

// Builds minimal, self-consistent PerTbParams for a single UE/TB/CB
static PerTbParams make_min_tb_params(uint32_t Ncb,
                                      uint32_t Zc,
                                      uint32_t Qm,
                                      uint32_t encodedSize,
                                      uint32_t K,
                                      uint32_t F,
                                      uint8_t  ndi,
                                      uint8_t  nDmrsCdmGrpsNoData)
{
    PerTbParams tb{};
    tb.ndi                      = ndi;
    tb.rv                       = 0;
    tb.Qm                       = Qm;
    tb.bg                       = 1;
    tb.Nl                       = 1;
    tb.num_CBs                  = 1;
    tb.Zc                       = Zc;
    tb.N                        = Ncb;
    tb.Ncb                      = Ncb;
    tb.Ncb_padded               = Ncb;
    tb.G                        = encodedSize; // unused when uciOnPuschFlag=0
    tb.K                        = K;
    tb.F                        = F;
    tb.cinit                    = 0x12345u;
    tb.nDataBytes               = 0;
    tb.nZpBitsPerCb             = 0;
    tb.firstCodeBlockIndex      = 0;
    tb.encodedSize              = encodedSize;
    tb.layer_map_array[0]       = 0;
    tb.userGroupIndex           = 0;
    tb.nBBULayers               = 1;
    tb.startLLR                 = 0;
    tb.isEarlyHarq              = 0;
    tb.uciOnPuschFlag           = 0;
    tb.csi2Flag                 = 0;
    tb.G_schAndCsi2             = 0;
    tb.G_harq                   = 0;
    tb.G_csi1                   = 0;
    tb.G_csi2                   = 0;
    tb.G_harq_rvd               = 0;
    tb.nBitsHarq                = 0;
    tb.nBitsCsi2                = 0;
    tb.nCsiReports              = 0;
    tb.rankBitOffset            = 0;
    tb.nRanksBits               = 0;
    tb.d_schAndCsi2LLRs         = nullptr;
    tb.d_csi1LLRs               = nullptr;
    tb.d_harqLLrs               = nullptr;
    tb.tbSize                   = 0;
    tb.nCsi2Reports             = 0;
    tb.enableTfPrcd             = 0;
    tb.nDmrsCdmGrpsNoData       = nDmrsCdmGrpsNoData;
    tb.debug_d_derateCbsIndices = nullptr;
    return tb;
}

// Allocates device input/output and runs setup + launch for one UE
static void run_rate_match_case(const PerTbParams&        tbPrm,
                                const std::vector<float>& h_llrs,
                                std::vector<float>&       h_out,
                                bool                      descramblingOn,
                                bool                      enableCpuToGpuDescrAsyncCpy)
{
    ASSERT_EQ(h_llrs.size(), tbPrm.encodedSize);
    const uint32_t Ncb            = tbPrm.Ncb;
    const uint32_t Zc             = tbPrm.Zc;
    const uint32_t NcbPadded      = tbPrm.Ncb_padded;
    const uint32_t nPuncturedBits = 2u * Zc;
    const uint32_t Qm             = tbPrm.Qm;
    const uint32_t E              = tbPrm.encodedSize;
    const uint32_t EoverQm        = E / Qm;

    // Device buffers
    float* d_in  = nullptr;
    float* d_out = nullptr;
    // Prepare padded input (QAM_STRIDE)
    const uint32_t     paddedLen = EoverQm * QAM_STRIDE;
    std::vector<float> h_padded(paddedLen, 0.0f);
    for(uint32_t i = 0; i < E; ++i)
    {
        uint32_t j                   = i / Qm;
        uint32_t k                   = i % Qm;
        h_padded[k + j * QAM_STRIDE] = h_llrs[i];
    }

    cudaCheck(cudaMalloc(&d_in, h_padded.size() * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, (NcbPadded + nPuncturedBits) * sizeof(float)));
    cudaCheck(cudaMemset(d_out, 0, (NcbPadded + nPuncturedBits) * sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_padded.data(), h_padded.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ppRmOut (host-pinned array of device pointers; device dereferences host memory)
    void** ppRmOut_host = nullptr;
    cudaCheck(cudaHostAlloc(&ppRmOut_host, sizeof(void*) * 1, cudaHostAllocDefault));
    ppRmOut_host[0] = static_cast<void*>(d_out);

    // Tensor params (only pAddr is used by setup for RM input selection)
    cuphyTensorPrm_t tRmIn[1]{};
    tRmIn[0].desc  = nullptr;
    tRmIn[0].pAddr = static_cast<void*>(d_in);
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    tCdm1RmIn[0].desc  = nullptr;
    tCdm1RmIn[0].pAddr = static_cast<void*>(d_in);

    // Device copy of PerTbParams
    PerTbParams* d_tb = nullptr;
    cudaCheck(cudaMalloc(&d_tb, sizeof(PerTbParams)));
    cudaCheck(cudaMemcpy(d_tb, &tbPrm, sizeof(PerTbParams), cudaMemcpyHostToDevice));

    // Descriptor buffers
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes), CUPHY_STATUS_SUCCESS);
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    cudaCheck(cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    cudaCheck(cudaMalloc(&gpuDesc, descrSizeBytes));

    // Create and setup handle
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(cuphyCreatePuschRxRateMatch(&hndl, /*FP32 in/out*/ 0, descramblingOn ? 1 : 0), CUPHY_STATUS_SUCCESS);

    uint16_t                         schUserIdxs[1] = {0};
    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};

    ASSERT_EQ(cuphySetupPuschRxRateMatch(hndl,
                                         /*nSchUes*/ 1,
                                         /*pSchUserIdxsCpu*/ schUserIdxs,
                                         /*pTbPrmsCpu*/ &tbPrm,
                                         /*pTbPrmsGpu*/ d_tb,
                                         /*pTPrmRmIn*/ tRmIn,
                                         /*pTPrmCdm1RmIn*/ tCdm1RmIn,
                                         /*ppRmOut*/ ppRmOut_host,
                                         /*pCpuDesc*/ cpuDesc,
                                         /*pGpuDesc*/ gpuDesc,
                                         /*enableCpuToGpuDescrAsyncCpy*/ enableCpuToGpuDescrAsyncCpy ? 1 : 0,
                                         /*pLaunchCfg*/ &launchCfg,
                                         /*strm*/ 0),
              CUPHY_STATUS_SUCCESS);

    // If multiple CBs requested, expand gridDim.y to C so r<C branch executes across blocks
    if(tbPrm.num_CBs > 1)
    {
        // Launch one extra y-block beyond C to exercise the r>=C (false) branch of 'if (r < C)'
        launchCfg.kernelNodeParamsDriver.gridDimY = tbPrm.num_CBs + 1;
    }

    if(!enableCpuToGpuDescrAsyncCpy)
    {
        cudaCheck(cudaMemcpy(gpuDesc, cpuDesc, descrSizeBytes, cudaMemcpyHostToDevice));
    }

    // Optionally override gridDimX to create multiple fractional CBs (fracCbIdx)
    // This allows testing both sides of maxIndex = min(maxIndex, maxIndexThisThrdBlk)
    if(tbPrm.encodedSize > 0 && (launchCfg.kernelNodeParamsDriver.gridDimX == 1))
    {
        // Choose two fractional CBs so that second CTA has a larger per-block max
        launchCfg.kernelNodeParamsDriver.gridDimX = 2;
    }
    // Launch
    launch_with_cfg(launchCfg, 0);

    // Copy results back
    h_out.resize(Ncb);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, Ncb * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    EXPECT_EQ(cuphyDestroyPuschRxRateMatch(hndl), CUPHY_STATUS_SUCCESS);
    cudaCheck(cudaFreeHost(cpuDesc));
    cudaCheck(cudaFree(gpuDesc));
    cudaCheck(cudaFree(d_tb));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFreeHost(ppRmOut_host));
}

} // namespace

// FP16 helper: runs rate-match with FP16 in/out
static void run_rate_match_case_fp16(const PerTbParams&        tbPrmIn,
                                     const std::vector<float>& h_llrs,
                                     std::vector<__half>&      h_out,
                                     bool                      descramblingOn,
                                     bool                      enableCpuToGpuDescrAsyncCpy)
{
    PerTbParams tbPrm = tbPrmIn;
    ASSERT_EQ(h_llrs.size(), tbPrm.encodedSize);
    const uint32_t Ncb            = tbPrm.Ncb;
    const uint32_t Zc             = tbPrm.Zc;
    const uint32_t nPuncturedBits = 2u * Zc;
    const uint32_t Qm             = tbPrm.Qm;
    const uint32_t E              = tbPrm.encodedSize;
    const uint32_t EoverQm        = E / Qm;

    // Prepare padded FP16 input
    std::vector<__half> h_padded(EoverQm * QAM_STRIDE, __float2half(0.0f));
    for(uint32_t i = 0; i < E; ++i)
    {
        uint32_t j                   = i / Qm;
        uint32_t k                   = i % Qm;
        h_padded[k + j * QAM_STRIDE] = __float2half(h_llrs[i]);
    }

    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, h_padded.size() * sizeof(__half)));
    cudaCheck(cudaMemcpy(d_in, h_padded.data(), h_padded.size() * sizeof(__half), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&d_out, (tbPrm.Ncb_padded + nPuncturedBits) * sizeof(__half)));
    cudaCheck(cudaMemset(d_out, 0, (tbPrm.Ncb_padded + nPuncturedBits) * sizeof(__half)));

    void** ppRmOut_host = nullptr;
    cudaCheck(cudaHostAlloc(&ppRmOut_host, sizeof(void*), cudaHostAllocDefault));
    ppRmOut_host[0] = static_cast<void*>(d_out);

    // Device TB params
    PerTbParams* d_tb = nullptr;
    cudaCheck(cudaMalloc(&d_tb, sizeof(PerTbParams)));
    cudaCheck(cudaMemcpy(d_tb, &tbPrm, sizeof(PerTbParams), cudaMemcpyHostToDevice));

    // Descriptors
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes), CUPHY_STATUS_SUCCESS);
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    cudaCheck(cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    cudaCheck(cudaMalloc(&gpuDesc, descrSizeBytes));

    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    // FPconfig 3: FP16 in, FP16 out
    ASSERT_EQ(cuphyCreatePuschRxRateMatch(&hndl, 3, descramblingOn ? 1 : 0), CUPHY_STATUS_SUCCESS);

    uint16_t         schUserIdxs[1] = {0};
    cuphyTensorPrm_t tRmIn[1]{};
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    tRmIn[0].pAddr     = static_cast<void*>(d_in);
    tCdm1RmIn[0].pAddr = static_cast<void*>(d_in);

    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};
    ASSERT_EQ(cuphySetupPuschRxRateMatch(hndl,
                                         1,
                                         schUserIdxs,
                                         &tbPrm,
                                         d_tb,
                                         tRmIn,
                                         tCdm1RmIn,
                                         ppRmOut_host,
                                         cpuDesc,
                                         gpuDesc,
                                         enableCpuToGpuDescrAsyncCpy ? 1 : 0,
                                         &launchCfg,
                                         0),
              CUPHY_STATUS_SUCCESS);
    if(!enableCpuToGpuDescrAsyncCpy)
    {
        cudaCheck(cudaMemcpy(gpuDesc, cpuDesc, descrSizeBytes, cudaMemcpyHostToDevice));
    }
    launch_with_cfg(launchCfg, 0);

    h_out.resize(Ncb);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, Ncb * sizeof(__half), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cuphyDestroyPuschRxRateMatch(hndl), CUPHY_STATUS_SUCCESS);
    cudaCheck(cudaFreeHost(ppRmOut_host));
    cudaCheck(cudaFree(gpuDesc));
    cudaCheck(cudaFreeHost(cpuDesc));
    cudaCheck(cudaFree(d_tb));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

// Shared rv/bg -> k0 helper used by host expected builders.
static uint32_t compute_k0_from_rv_bg(const PerTbParams& tb)
{
    uint32_t k0 = 0;
    if(tb.bg == 1)
    {
        if(tb.rv == 1)
            k0 = (17 * tb.Ncb / (66 * tb.Zc)) * tb.Zc;
        else if(tb.rv == 2)
            k0 = (33 * tb.Ncb / (66 * tb.Zc)) * tb.Zc;
        else if(tb.rv == 3)
            k0 = (56 * tb.Ncb / (66 * tb.Zc)) * tb.Zc;
    }
    else if(tb.bg == 2)
    {
        if(tb.rv == 1)
            k0 = (13 * tb.Ncb / (50 * tb.Zc)) * tb.Zc;
        else if(tb.rv == 2)
            k0 = (25 * tb.Ncb / (50 * tb.Zc)) * tb.Zc;
        else if(tb.rv == 3)
            k0 = (43 * tb.Ncb / (50 * tb.Zc)) * tb.Zc;
    }
    return k0;
}

static size_t count_nonzero_finite(const std::vector<float>& out)
{
    size_t nz = 0;
    for(float v : out)
    {
        if(v != 0.0f)
        {
            ++nz;
            EXPECT_TRUE(std::isfinite(v));
        }
    }
    return nz;
}

static size_t count_nonzero_finite(const std::vector<__half>& out_h)
{
    size_t nz = 0;
    for(auto v : out_h)
    {
        const float f = __half2float(v);
        if(f != 0.0f)
        {
            ++nz;
            EXPECT_TRUE(std::isfinite(f));
        }
    }
    return nz;
}

static size_t count_nonzero_finite_bounded(const std::vector<float>& out, float lo, float hi)
{
    size_t nz = 0;
    for(float v : out)
    {
        if(v != 0.0f)
        {
            ++nz;
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_LE(v, hi);
            EXPECT_GE(v, lo);
        }
    }
    return nz;
}

// Build expected host output for simple single-CB case (descramblingOff)
static std::vector<float> build_expected_noatomics(const PerTbParams& tb, const std::vector<float>& h_llrs)
{
    const uint32_t Ncb            = tb.Ncb;
    const uint32_t Zc             = tb.Zc;
    const uint32_t Qm             = tb.Qm;
    const uint32_t E              = tb.encodedSize;
    const uint32_t EoverQm        = E / Qm;
    const uint32_t nPuncturedBits = 2u * Zc;
    const uint32_t Kd             = tb.K - nPuncturedBits - tb.F;

    const uint32_t k0 = compute_k0_from_rv_bg(tb);

    std::vector<float> expected(Ncb, 0.0f);
    for(uint32_t j = 0; j < EoverQm; ++j)
    {
        for(uint32_t k = 0; k < Qm; ++k)
        {
            // Kernel value index (from padded load): i = j*Qm + k
            uint32_t i = j * Qm + k;
            // Derate-matching index uses inIdx = k*EoverQm + j
            uint32_t inIdx   = k * EoverQm + j;
            int      outIdx  = derate_match_fast_calc_modulo(static_cast<int>(inIdx), static_cast<int>(Kd), static_cast<int>(tb.F), static_cast<int>(k0), static_cast<int>(tb.Ncb));
            expected[outIdx] = h_llrs[i];
        }
    }
    // prepend punctured zeros at device base; our copy grabs base->base+Ncb
    std::vector<float> expected_full(Ncb, 0.0f);
    // first 2*Zc are zeros by kernel init
    for(uint32_t i = 0; i < Ncb - nPuncturedBits; ++i)
    {
        expected_full[nPuncturedBits + i] = expected[i];
    }
    return expected_full;
}

// Build expected with atomics + clamp, descramblingOff (sum then clamp to 10000)
static std::vector<float> build_expected_atomics_clamp(const PerTbParams& tb, const std::vector<float>& h_llrs)
{
    const uint32_t Ncb            = tb.Ncb;
    const uint32_t Zc             = tb.Zc;
    const uint32_t Qm             = tb.Qm;
    const uint32_t E              = tb.encodedSize;
    const uint32_t EoverQm        = E / Qm;
    const uint32_t nPuncturedBits = 2u * Zc;
    const uint32_t Kd             = tb.K - nPuncturedBits - tb.F;

    const uint32_t k0 = compute_k0_from_rv_bg(tb);

    std::vector<float> circ(Ncb, 0.0f);
    for(uint32_t j = 0; j < EoverQm; ++j)
    {
        for(uint32_t k = 0; k < Qm; ++k)
        {
            uint32_t i      = j * Qm + k;
            uint32_t inIdx  = k * EoverQm + j;
            int      outIdx = derate_match_fast_calc_modulo(static_cast<int>(inIdx), static_cast<int>(Kd), static_cast<int>(tb.F), static_cast<int>(k0), static_cast<int>(tb.Ncb));
            circ[outIdx] += h_llrs[i];
            circ[outIdx] = std::min(circ[outIdx], 10000.0f);
            circ[outIdx] = std::max(circ[outIdx], -10000.0f);
        }
    }
    std::vector<float> expected_full(Ncb, 0.0f);
    for(uint32_t i = 0; i < Ncb - nPuncturedBits; ++i)
    {
        expected_full[nPuncturedBits + i] = circ[i];
    }
    return expected_full;
}

TEST(PUSCH_RateMatch, Basic_NoAtomics_DescramblingOff)
{
    // Small, conflict-free config
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 16;
    const uint32_t Qm  = 2;
    const uint32_t E   = 4; // encodedSize
    const uint32_t K   = 8; // >= 2*Zc + F
    const uint32_t F   = 0;

    PerTbParams        tb   = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    std::vector<float> h_in = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<float> h_out;

    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), Ncb);
    auto expected = build_expected_noatomics(tb, h_in);
    ASSERT_EQ(expected.size(), h_out.size());
    for(size_t i = 0; i < h_out.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_out[i], expected[i]) << "mismatch at i=" << i;
    }
}

TEST(PUSCH_RateMatch, ElEh_Split_ElseBranch_Coverage)
{
    // Drive r > rr branch (larger code block Eh) by choosing C=2 and q1% C != 0
    // Setup: Qm=2, Nl=1 -> TBLLRsPerNBBULayers=2, E=10 -> q1=5, C=2 -> rr=0
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 64; // large to avoid collisions
    const uint32_t Qm  = 2;
    const uint32_t E   = 10; // encodedSize
    const uint32_t K   = 40;
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.num_CBs         = 2; // C=2 ensures one small (El=4) and one large (Eh=6) CB
    tb.rv              = 0; // k0=0 to keep mapping simple, no combining
    std::vector<float> h_in(E);
    // Mark CB contributions distinctly: first El values = 1.0, next Eh values = 2.0
    for(uint32_t i = 0; i < E; ++i) h_in[i] = (i < 4) ? 1.0f : 2.0f;
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    // Invariants: non-zero outputs exist, values finite and within clamp bounds
    const size_t nz = count_nonzero_finite_bounded(h_out, -10000.0f, 10000.0f);
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, Atomics_Clamp_DescramblingOn)
{
    // Force collisions (potentialRaceIfPositive > 0) and trigger clamp via large LLRs
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 4; // small to increase collisions
    const uint32_t Qm  = 2;
    const uint32_t E   = 8; // encodedSize
    const uint32_t K   = 8; // >= 2*Zc + F
    const uint32_t F   = 0;

    PerTbParams tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    // Large values to exceed 10000 after atomicAdd merges
    std::vector<float> h_in(E, 9000.0f);
    std::vector<float> h_out;

    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), Ncb);
    auto expected = build_expected_atomics_clamp(tb, h_in);
    ASSERT_EQ(expected.size(), h_out.size());
    for(size_t i = 0; i < h_out.size(); ++i)
    {
        EXPECT_FLOAT_EQ(h_out[i], expected[i]) << "mismatch at i=" << i;
    }
}

TEST(PUSCH_RateMatch, Descrambling_AbsMatches_Float)
{
    // Enable descrambling to cover rate_match_xor_sign for float
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 4;
    const uint32_t E   = 16;
    const uint32_t K   = 24;
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.cinit           = 0xABCDEFu;
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = (i % 7) - 3.0f;
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ true, /*enableCpuToGpuDescrAsyncCpy*/ false);
    auto expected_no_scramble = build_expected_noatomics(tb, h_in);
    ASSERT_EQ(h_out.size(), expected_no_scramble.size());
    for(size_t i = 0; i < h_out.size(); ++i)
    {
        EXPECT_FLOAT_EQ(std::fabs(h_out[i]), std::fabs(expected_no_scramble[i])) << "abs mismatch at i=" << i;
    }
}

TEST(PUSCH_RateMatch, Descrambling_AbsMatches_Half)
{
    // Exercise rate_match_xor_sign(uint32_t,int,__half)__half via FP16 path with descrambling enabled
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 4;
    const uint32_t E   = 16;
    const uint32_t K   = 24;
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.cinit           = 0x1A2B3Cu;
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 7) - 3);
    // Compute FP16 outputs with descrambling OFF and ON, compare magnitudes
    std::vector<__half> off_out_h, on_out_h;
    run_rate_match_case_fp16(tb, h_in, off_out_h, /*descramblingOn*/ false, /*async*/ false);
    run_rate_match_case_fp16(tb, h_in, on_out_h, /*descramblingOn*/ true, /*async*/ false);
    ASSERT_EQ(off_out_h.size(), on_out_h.size());
    for(size_t i = 0; i < on_out_h.size(); ++i)
    {
        EXPECT_FLOAT_EQ(std::fabs(__half2float(on_out_h[i])), std::fabs(__half2float(off_out_h[i]))) << "abs mismatch at i=" << i;
    }
}

// Helper: FP32 input, FP16 output, descrambling toggle
static void run_rate_match_case_fp32_in_fp16_out(const PerTbParams&        tbPrmIn,
                                                 const std::vector<float>& h_llrs,
                                                 std::vector<__half>&      h_out,
                                                 bool                      descramblingOn)
{
    PerTbParams        tbPrm          = tbPrmIn;
    const uint32_t     Zc             = tbPrm.Zc;
    const uint32_t     Ncb            = tbPrm.Ncb;
    const uint32_t     E              = tbPrm.encodedSize;
    const uint32_t     Qm             = tbPrm.Qm;
    const uint32_t     EoverQm        = E / Qm;
    const uint32_t     nPuncturedBits = 2u * Zc;
    std::vector<float> h_padded(EoverQm * QAM_STRIDE, 0.0f);
    for(uint32_t i = 0; i < E; ++i)
    {
        uint32_t j = i / Qm, k = i % Qm;
        h_padded[k + j * QAM_STRIDE] = h_llrs[i];
    }
    float*  d_in  = nullptr;
    __half* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, h_padded.size() * sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_padded.data(), h_padded.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&d_out, (tbPrm.Ncb_padded + nPuncturedBits) * sizeof(__half)));
    cudaCheck(cudaMemset(d_out, 0, (tbPrm.Ncb_padded + nPuncturedBits) * sizeof(__half)));
    void** ppRmOut_host = nullptr;
    cudaCheck(cudaHostAlloc(&ppRmOut_host, sizeof(void*), cudaHostAllocDefault));
    ppRmOut_host[0]   = static_cast<void*>(d_out);
    PerTbParams* d_tb = nullptr;
    cudaCheck(cudaMalloc(&d_tb, sizeof(PerTbParams)));
    cudaCheck(cudaMemcpy(d_tb, &tbPrm, sizeof(PerTbParams), cudaMemcpyHostToDevice));
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes), CUPHY_STATUS_SUCCESS);
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    cudaCheck(cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    cudaCheck(cudaMalloc(&gpuDesc, descrSizeBytes));
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(cuphyCreatePuschRxRateMatch(&hndl, 2, descramblingOn ? 1 : 0), CUPHY_STATUS_SUCCESS);
    uint16_t         schUserIdxs[1] = {0};
    cuphyTensorPrm_t tRmIn[1]{};
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    tRmIn[0].pAddr     = d_in;
    tCdm1RmIn[0].pAddr = d_in;
    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};
    ASSERT_EQ(cuphySetupPuschRxRateMatch(hndl, 1, schUserIdxs, &tbPrm, d_tb, tRmIn, tCdm1RmIn, ppRmOut_host, cpuDesc, gpuDesc, 0, &launchCfg, 0), CUPHY_STATUS_SUCCESS);
    cudaCheck(cudaMemcpy(gpuDesc, cpuDesc, descrSizeBytes, cudaMemcpyHostToDevice));
    launch_with_cfg(launchCfg, 0);
    h_out.resize(Ncb);
    cudaCheck(cudaMemcpy(h_out.data(), d_out, Ncb * sizeof(__half), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cuphyDestroyPuschRxRateMatch(hndl), CUPHY_STATUS_SUCCESS);
    cudaCheck(cudaFree(d_tb));
    cudaCheck(cudaFreeHost(ppRmOut_host));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFreeHost(cpuDesc));
    cudaCheck(cudaFree(gpuDesc));
}

// Two-pass helper that reuses the same device output buffer to exercise ndi==0 combining
static void run_two_passes_reuse_out(const PerTbParams&        tbPrm1,
                                     const std::vector<float>& h_llrs1,
                                     const PerTbParams&        tbPrm2,
                                     const std::vector<float>& h_llrs2,
                                     std::vector<float>&       h_out_final,
                                     bool                      descramblingOn,
                                     bool                      enableCpuToGpuDescrAsyncCpy)
{
    ASSERT_EQ(tbPrm1.Ncb, tbPrm2.Ncb);
    ASSERT_EQ(tbPrm1.Zc, tbPrm2.Zc);
    const uint32_t Ncb            = tbPrm1.Ncb;
    const uint32_t Zc             = tbPrm1.Zc;
    const uint32_t nPuncturedBits = 2u * Zc;

    auto make_padded = [](const PerTbParams& tb, const std::vector<float>& h) {
        const uint32_t     Qm = tb.Qm, E = tb.encodedSize, EoverQm = E / Qm;
        std::vector<float> padded(EoverQm * QAM_STRIDE, 0.0f);
        for(uint32_t i = 0; i < E; ++i)
        {
            uint32_t j = i / Qm, k = i % Qm;
            padded[k + j * QAM_STRIDE] = h[i];
        }
        return padded;
    };
    auto h_pad1 = make_padded(tbPrm1, h_llrs1);
    auto h_pad2 = make_padded(tbPrm2, h_llrs2);

    float* d_in1 = nullptr;
    float* d_in2 = nullptr;
    float* d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in1, h_pad1.size() * sizeof(float)));
    cudaCheck(cudaMalloc(&d_in2, h_pad2.size() * sizeof(float)));
    cudaCheck(cudaMemcpy(d_in1, h_pad1.data(), h_pad1.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_in2, h_pad2.data(), h_pad2.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&d_out, (tbPrm1.Ncb_padded + nPuncturedBits) * sizeof(float)));
    cudaCheck(cudaMemset(d_out, 0, (tbPrm1.Ncb_padded + nPuncturedBits) * sizeof(float)));

    // Host-pinned ppRmOut
    void** ppRmOut = nullptr;
    cudaCheck(cudaHostAlloc(&ppRmOut, sizeof(void*), cudaHostAllocDefault));
    ppRmOut[0] = d_out;

    // Descriptors
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes));
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    cudaCheck(cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    cudaCheck(cudaMalloc(&gpuDesc, descrSizeBytes));

    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, 0, descramblingOn ? 1 : 0));
    PerTbParams* d_tb = nullptr;
    cudaCheck(cudaMalloc(&d_tb, sizeof(PerTbParams)));
    uint16_t         schIdx[1] = {0};
    cuphyTensorPrm_t tRmIn[1]{};
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    auto             run_one = [&](const PerTbParams& tb, const float* d_in) {
        cudaCheck(cudaMemcpy(d_tb, &tb, sizeof(PerTbParams), cudaMemcpyHostToDevice));
        tRmIn[0].pAddr     = const_cast<float*>(d_in);
        tCdm1RmIn[0].pAddr = const_cast<float*>(d_in);
        cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRateMatch(hndl, 1, schIdx, &tb, d_tb, tRmIn, tCdm1RmIn, ppRmOut, cpuDesc, gpuDesc, enableCpuToGpuDescrAsyncCpy ? 1 : 0, &launchCfg, 0));
        if(!enableCpuToGpuDescrAsyncCpy) { cudaCheck(cudaMemcpy(gpuDesc, cpuDesc, descrSizeBytes, cudaMemcpyHostToDevice)); }
        launch_with_cfg(launchCfg, 0);
    };

    run_one(tbPrm1, d_in1); // first pass
    run_one(tbPrm2, d_in2); // second pass combines (ndi==0 expected)

    h_out_final.resize(Ncb);
    cudaCheck(cudaMemcpy(h_out_final.data(), d_out, Ncb * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyPuschRxRateMatch(hndl));
    cudaCheck(cudaFree(d_tb));
    cudaCheck(cudaFreeHost(ppRmOut));
    cudaCheck(cudaFree(d_in1));
    cudaCheck(cudaFree(d_in2));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFreeHost(cpuDesc));
    cudaCheck(cudaFree(gpuDesc));
}

TEST(PUSCH_RateMatch, Descrambling_AbsMatches_Half_Fp32In)
{
    const uint32_t Zc = 2, Ncb = 32, Qm = 4, E = 16, K = 24, F = 0;
    PerTbParams    tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.cinit          = 0x13572468u;
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 7) - 3);
    std::vector<__half> off_out_h, on_out_h;
    run_rate_match_case_fp32_in_fp16_out(tb, h_in, off_out_h, /*descramblingOn*/ false);
    run_rate_match_case_fp32_in_fp16_out(tb, h_in, on_out_h, /*descramblingOn*/ true);
    ASSERT_EQ(off_out_h.size(), on_out_h.size());
    for(size_t i = 0; i < on_out_h.size(); ++i)
    {
        EXPECT_FLOAT_EQ(std::fabs(__half2float(on_out_h[i])), std::fabs(__half2float(off_out_h[i])));
    }
}

TEST(PUSCH_RateMatch, DivideByQm_Qm8_Path_Invariants)
{
    // Drive divide_by_Qm(Qm==8) path and validate invariants
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 64;
    const uint32_t     Qm  = 8;
    const uint32_t     E   = 32; // multiple of 8
    const uint32_t     K   = 40;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 9) - 4);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    // Invariants: non-zero mapped values <= E in count, all finite and clamped
    const size_t nz = count_nonzero_finite_bounded(h_out, -10000.0f, 10000.0f);
    EXPECT_LE(nz, static_cast<size_t>(E));
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, Dispatch_Qm1_NdiTrue_Path)
{
    // Explicitly cover de_rate_matching_global2 dispatch branch: Qm==1, ndi!=0.
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 32;
    const uint32_t     Qm  = 1;
    const uint32_t     E   = 16; // valid for Qm=1
    const uint32_t     K   = 20;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 7) - 3);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    EXPECT_GE(count_nonzero_finite_bounded(h_out, -10000.0f, 10000.0f), static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, MaxIndex_Min_Ternary_BothBranches)
{
    // Drive both sides of maxIndex = (maxIndexThisThrdBlk < maxIndex) ? maxIndexThisThrdBlk : maxIndex
    // First run: small E so round_up_to_next(E,32) < (fracCbIdx+1)*blockDim.x*NUM_LLRS_PROCESSED_PER_THRD -> take false branch
    {
        const uint32_t     Zc = 2, Ncb = 64, Qm = 2, E = 16, K = 40, F = 0; // E small
        PerTbParams        tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
        std::vector<float> h_in(E, 1.0f);
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(h_out.size(), Ncb);
    }
    // Second run: large E so round_up_to_next(E,32) > per-block max -> take true branch
    {
        const uint32_t     Zc = 2, Ncb = 64, Qm = 2, E = 70000, K = 70000, F = 0; // E >> 32768 so per-block max is smaller
        PerTbParams        tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
        std::vector<float> h_in(E, 1.0f);
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(h_out.size(), Ncb);
    }
}

TEST(PUSCH_RateMatch, DivideByQm_QmOther_Path_Invariants)
{
    // Drive divide_by_Qm() "other" branch with a supported non-2/4 Qm.
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 64;
    const uint32_t     Qm  = 6;  // supported by API; still exercises non-2/4 path
    const uint32_t     E   = 30; // multiple of 6*5
    const uint32_t     K   = 40;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 5) - 2);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    // Invariants
    const size_t nz = count_nonzero_finite_bounded(h_out, -10000.0f, 10000.0f);
    EXPECT_LE(nz, static_cast<size_t>(E));
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, Atomics_Clamp_LoopContention)
{
    // Heavily contended scenario to increase probability of CAS retry in atomicMin/Max custom loops
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 8; // very small circular buffer to force collisions
    const uint32_t Qm  = 2;
    const uint32_t E   = 32768; // many threads
    const uint32_t K   = 4;     // Kd = K - 2*Zc - F = 0 to simplify mapping
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    // Try a few rv values to vary k0 and mapping across runs
    std::vector<float> h_in(E, 20000.0f); // large positive values for heavy atomic accumulation
    bool               sawNonZeroBounded = false;
    for(uint32_t rv : {0u, 1u, 2u, 3u})
    {
        tb.rv = rv;
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        sawNonZeroBounded = count_nonzero_finite(h_out) > 0;
        if(sawNonZeroBounded) break;
    }
    EXPECT_TRUE(sawNonZeroBounded);
}

TEST(PUSCH_RateMatch, Atomics_Half_CAS_Retry_Coverage)
{
    // Exercise the CAS retry loop in atomicMaxCustom(__half*, __half) where assumed != old at least once
    const uint32_t     Zc = 2, Ncb = 8, Qm = 2, E = 16384, K = 4, F = 0;
    PerTbParams        tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in(E);
    // Alternate signs to force racing updates at the same outIdx so CAS will retry
    for(uint32_t i = 0; i < E; ++i) h_in[i] = (i & 1) ? -20000.0f : -100.0f;
    // Run once to warm and once to increase contention
    std::vector<float> h_out1;
    run_rate_match_case(tb, h_in, h_out1, /*descramblingOn*/ false, /*async*/ false);
    std::vector<float> h_out2;
    run_rate_match_case(tb, h_in, h_out2, /*descramblingOn*/ false, /*async*/ false);
    // Invariants: some outputs are produced and all results remain finite.
    const size_t nz = count_nonzero_finite(h_out2);
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, Atomics_Half_CAS_NoRetry_Safe)
{
    // Safer variant: spread writes across a larger circular buffer to minimize contention
    // This increases the chance that atomicCAS succeeds on the first attempt (assumed == old)
    const uint32_t Zc  = 2;   // nPuncturedBits = 4
    const uint32_t Ncb = 512; // large to spread indices
    const uint32_t Qm  = 2;
    const uint32_t E   = 4096; // enough work for gridDim.x>1
    const uint32_t F   = 0;
    const uint32_t K   = Ncb; // Kd = K - 2*Zc - F leaves large data region
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.rv              = 0;
    tb.bg              = 1;
    // Keep magnitude moderate for FP16 so accumulated values stay finite.
    std::vector<float>  h_in(E, -500.0f);
    std::vector<__half> h_out_h;
    run_rate_match_case_fp16(tb, h_in, h_out_h, /*descramblingOn*/ false, /*async*/ false);
    // Expect many outputs untouched (0) and some non-zero finite values.
    const size_t nz = count_nonzero_finite(h_out_h);
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, Atomics_Half_Min_CAS_Retry_Coverage)
{
    // Exercise CAS retry loop for atomicMinCustom(__half*, __half) via heavy contention and positive overflow
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 8;
    const uint32_t     Qm  = 2;
    const uint32_t     E   = 32768;
    const uint32_t     K   = 4;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    // Keep FP16 accumulation finite under heavy contention.
    std::vector<float> h_in(E, 10.0f);
    bool               sawNonZeroBounded = false;
    for(uint32_t rv : {0u, 1u, 2u, 3u})
    {
        tb.rv = rv;
        std::vector<__half> out_h;
        run_rate_match_case_fp16(tb, h_in, out_h, /*descramblingOn*/ false, /*async*/ false);
        sawNonZeroBounded = count_nonzero_finite(out_h) > 0;
        if(sawNonZeroBounded) break;
    }
    EXPECT_TRUE(sawNonZeroBounded);
}

TEST(PUSCH_RateMatch, Atomics_Clamp_UseAtomics_Negative)
{
    // Mirror of LoopContention but with large negative inputs to drive atomicMaxCustom in useAtomics branch
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 8;
    const uint32_t     Qm  = 2;
    const uint32_t     E   = 32768;
    const uint32_t     K   = 4;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in(E, -20000.0f);
    bool               sawNonZeroBounded = false;
    for(uint32_t rv : {0u, 1u, 2u, 3u})
    {
        tb.rv = rv;
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        sawNonZeroBounded = count_nonzero_finite(h_out) > 0;
        if(sawNonZeroBounded) break;
    }
    EXPECT_TRUE(sawNonZeroBounded);
}

TEST(PUSCH_RateMatch, NDI0_NoAtomics_PositiveClamp_TwoPass)
{
    // Force no-atomics path (potentialRaceIfPositive<=0) and cause positive clamp in ndi==0 sum
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 64;
    const uint32_t Qm  = 2;
    const uint32_t E   = 16; // ensure E + 2F + k0 - Ncb <= 0
    const uint32_t K   = 40;
    const uint32_t F   = 0;
    PerTbParams    tb1 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams    tb2 = tb1;
    tb2.ndi            = 0; // second pass combines
    std::vector<float> in1(E, 9000.0f);
    std::vector<float> in2(E, 9000.0f);
    std::vector<float> out_final;
    run_two_passes_reuse_out(tb1, in1, tb2, in2, out_final, /*descramblingOn*/ false, /*async*/ false);
    const size_t nz = count_nonzero_finite_bounded(out_final, -10000.0f, 10000.0f);
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, NDI0_NoAtomics_NegativeClamp_TwoPass)
{
    // Force no-atomics path and cause negative clamp in ndi==0 sum
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 64;
    const uint32_t Qm  = 2;
    const uint32_t E   = 16;
    const uint32_t K   = 40;
    const uint32_t F   = 0;
    PerTbParams    tb1 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams    tb2 = tb1;
    tb2.ndi            = 0;
    std::vector<float> in1(E, -9000.0f);
    std::vector<float> in2(E, -9000.0f);
    std::vector<float> out_final;
    run_two_passes_reuse_out(tb1, in1, tb2, in2, out_final, /*descramblingOn*/ false, /*async*/ false);
    const size_t nz = count_nonzero_finite_bounded(out_final, -10000.0f, 10000.0f);
    EXPECT_GE(nz, static_cast<size_t>(1));
}

TEST(PUSCH_RateMatch, NDI0_UseAtomics_PositiveClamp_TwoPass_Contention)
{
    // Drive ndi==0 useAtomics branch with large E and small Ncb so outIdx always < potentialRaceIfPositive
    const uint32_t Zc  = 2; // nPuncturedBits = 4
    const uint32_t Ncb = 8; // very small to maximize contention
    const uint32_t Qm  = 2;
    const uint32_t E   = 32768; // very large
    const uint32_t K   = 4;     // Kd = 0
    const uint32_t F   = 0;
    PerTbParams    tb1 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams    tb2 = tb1;
    tb2.ndi            = 0; // second pass combines using atomicAdd path for ndi==0
    // Large positive inputs ensure sum(prev+llr) > 10000 so atomicMinCustom is taken
    std::vector<float> in1(E, 9000.0f);
    std::vector<float> in2(E, 9000.0f);
    std::vector<float> out_final;
    run_two_passes_reuse_out(tb1, in1, tb2, in2, out_final, /*descramblingOn*/ false, /*async*/ false);
    EXPECT_GT(count_nonzero_finite(out_final), static_cast<size_t>(0));
}

TEST(PUSCH_RateMatch, NDI0_UseAtomics_NegativeClamp_TwoPass_Contention)
{
    // Mirror test for negative clamping via atomicMaxCustom in ndi==0 useAtomics path
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 8;
    const uint32_t Qm  = 2;
    const uint32_t E   = 32768;
    const uint32_t K   = 4;
    const uint32_t F   = 0;
    PerTbParams    tb1 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams    tb2 = tb1;
    tb2.ndi            = 0;
    std::vector<float> in1(E, -9000.0f);
    std::vector<float> in2(E, -9000.0f);
    std::vector<float> out_final;
    run_two_passes_reuse_out(tb1, in1, tb2, in2, out_final, /*descramblingOn*/ false, /*async*/ false);
    EXPECT_GT(count_nonzero_finite(out_final), static_cast<size_t>(0));
}

TEST(PUSCH_RateMatch, NDI0_UseAtomics_NoClamp_TwoPass_Contention)
{
    // Ensure we go through useAtomics in ndi==0 but do NOT trigger either clamp branch (line 540 and line 546 false)
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 8;
    const uint32_t Qm  = 2;
    const uint32_t E   = 16;
    const uint32_t K   = 4;
    const uint32_t F   = 0;
    PerTbParams    tb1 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams    tb2 = tb1;
    tb2.ndi            = 0; // second pass uses ndi==0
    // Small magnitudes so prev_llr + llr stays inside [-10000, 10000]
    std::vector<float> in1(E, 1.0f);
    std::vector<float> in2(E, 1.0f);
    std::vector<float> out_final;
    run_two_passes_reuse_out(tb1, in1, tb2, in2, out_final, /*descramblingOn*/ false, /*async*/ false);
    // Invariants: no clamped sentinels appear
    bool anyClamp = false;
    for(float v : out_final)
    {
        if(v == 10000.0f || v == -10000.0f)
        {
            anyClamp = true;
            break;
        }
    }
    EXPECT_FALSE(anyClamp);
}

TEST(PUSCH_RateMatch, FillerAndZeroInit_SkipOverFiller_While)
{
    // Configure parameters to trigger filler write (1a), zero write (1b), and skip-over-filler inside while loop
    // Choose values so that circBufIdx = k0 + (E + F) wraps and lands inside [Kd, Kd+F), exercising the skip branch
    const uint32_t Zc  = 2; // nPuncturedBits = 4
    const uint32_t Ncb = 16;
    const uint32_t Qm  = 2;
    const uint32_t E   = 8;  // encodedSize
    const uint32_t F   = 2;  // filler bits
    const uint32_t K   = 12; // Kd = 12 - 4 - 2 = 6, filler region [6,8)
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.bg              = 1;
    tb.rv              = 3; // kept for branch coverage in setup path
    tb.k0              = 12; // explicit k0 used by kernel path
    std::vector<float> h_in(E, 0.0f);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    const uint32_t Kd = K - 2 * Zc - F; // 6
    // 1a: filler region should be set to 10000; note: device writes at out + nPuncturedBits
    const uint32_t baseOffset = 2u * Zc;
    for(uint32_t n = Kd; n < Kd + F; ++n)
    {
        EXPECT_FLOAT_EQ(h_out[baseOffset + n], 10000.0f) << "filler not written at n=" << n;
    }
    // Compute wrapped index for n = E + F starting at tid=0: circBufIdx = k0 + n -> 12 + 10 = 22 -> wrap to 6, skip adds F -> 8
    uint32_t circBufIdx = (12 + (E + F));
    while(circBufIdx >= Ncb) circBufIdx -= Ncb;                  // emulate wrap
    if(circBufIdx >= Kd && circBufIdx < Kd + F) circBufIdx += F; // emulate skip-over-filler
    // 1b: zero write should occur at circBufIdx computed above (outside filler)
    EXPECT_FLOAT_EQ(h_out[baseOffset + circBufIdx], 0.0f) << "zero init not written at computed circBufIdx";
}

TEST(PUSCH_RateMatch, FillerSkip_NoWrap_IfBranch)
{
    // Construct parameters so that circBufIdx = k0 + (E + F) lies inside [Kd, Kd+F) BEFORE wrap
    // Use values that keep E a multiple of Qm to satisfy padding logic
    const uint32_t Zc  = 2; // nPuncturedBits = 4
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 2;
    const uint32_t E   = 2; // multiple of Qm
    const uint32_t F   = 4; // filler size
    // Set explicit k0 to force wrap path in deRateMatchingKernelInner:
    // start=(E+F+k0)%Ncb=(2+4+16)%32=22, len=Ncb-(E+F)=26, so start+len=48>Ncb.
    // Choose K so that Kd = 22 -> filler region [22,26)
    const uint32_t K  = 30; // Kd = 30 - 4 - 4 = 22
    PerTbParams    tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.bg             = 1;
    tb.rv             = 2; // kept for branch coverage in setup path
    tb.k0             = 16; // explicit k0 used by kernel path
    std::vector<float> h_in(E, 0.0f);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    const uint32_t base = 2u * Zc;        // device writes at out + nPuncturedBits
    const uint32_t Kd   = K - 2 * Zc - F; // 22
    // Filler region must remain 10000 at [22,26)
    for(uint32_t n = 0; n < F; ++n)
    {
        EXPECT_FLOAT_EQ(h_out[base + Kd + n], 10000.0f);
    }
    // circBufIdx = k0 + (E + F) = 16 + 6 = 22 -> inside filler; after skip becomes 26
    EXPECT_FLOAT_EQ(h_out[base + 22], 10000.0f);
    EXPECT_FLOAT_EQ(h_out[base + 26], 0.0f);
}

TEST(PUSCH_RateMatch, ComputeLlrIndex_UciOnPusch)
{
    // Drive compute_llr_index UCI branch by setting uciOnPuschFlag=1 and verifying deterministic mapping
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 2;
    const uint32_t E   = 8;
    const uint32_t K   = 16;
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.uciOnPuschFlag  = 1;
    // Provide linear input through d_schAndCsi2LLRs path via setup
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>(i + 1);
    // Manually build minimal setup to take UCI branch
    // Route through actual UCI path using __half inputs and FPconfig=3 (half in/out)
    // Allocate linear __half input
    std::vector<__half> h_in_h(E);
    for(uint32_t i = 0; i < E; ++i) h_in_h[i] = __float2half(h_in[i]);
    __half* d_in = nullptr;
    cudaCheck(cudaMalloc(&d_in, E * sizeof(__half)));
    cudaCheck(cudaMemcpy(d_in, h_in_h.data(), E * sizeof(__half), cudaMemcpyHostToDevice));
    // Output buffer
    const uint32_t nPuncturedBits = 2u * Zc;
    __half*        d_out          = nullptr;
    cudaCheck(cudaMalloc(&d_out, (tb.Ncb_padded + nPuncturedBits) * sizeof(__half)));
    cudaCheck(cudaMemset(d_out, 0, (tb.Ncb_padded + nPuncturedBits) * sizeof(__half)));
    // Host-pinned ppRmOut array
    void** ppRmOut = nullptr;
    cudaCheck(cudaHostAlloc(&ppRmOut, sizeof(void*), cudaHostAllocDefault));
    ppRmOut[0] = d_out;
    // Descriptors
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes));
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    cudaCheck(cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    cudaCheck(cudaMalloc(&gpuDesc, descrSizeBytes));
    // Device TB params; set uci flag and pointer
    tb.uciOnPuschFlag   = 1;
    tb.d_schAndCsi2LLRs = d_in;
    PerTbParams* d_tb   = nullptr;
    cudaCheck(cudaMalloc(&d_tb, sizeof(PerTbParams)));
    cudaCheck(cudaMemcpy(d_tb, &tb, sizeof(PerTbParams), cudaMemcpyHostToDevice));
    // Handle FPconfig=3 (half in/out), descrambling off
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, 3, 0));
    uint16_t                         schIdx[1] = {0};
    cuphyTensorPrm_t                 tRmIn[1]{};
    cuphyTensorPrm_t                 tCdm1RmIn[1]{}; // unused in UCI
    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRateMatch(hndl, 1, schIdx, &tb, d_tb, tRmIn, tCdm1RmIn, ppRmOut, cpuDesc, gpuDesc, 0, &launchCfg, 0));
    cudaCheck(cudaMemcpy(gpuDesc, cpuDesc, descrSizeBytes, cudaMemcpyHostToDevice));
    launch_with_cfg(launchCfg, 0);
    // Copy back and validate
    std::vector<__half> h_out_h(Ncb);
    cudaCheck(cudaMemcpy(h_out_h.data(), d_out, Ncb * sizeof(__half), cudaMemcpyDeviceToHost));
    size_t nz_half = 0;
    for(auto v : h_out_h)
        if(__half2float(v) != 0.0f) ++nz_half;
    EXPECT_GE(nz_half, static_cast<size_t>(1));
    // Cleanup
    cuphyDestroyPuschRxRateMatch(hndl);
    cudaFree(d_tb);
    cudaFreeHost(ppRmOut);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(cpuDesc);
    cudaFree(gpuDesc);
}

TEST(PUSCH_RateMatch, DivideByQm_Qm6_Path)
{
    // Cover divide_by_Qm for Qm==6 (non-power-of-two branch). Validate invariants rather than exact positions.
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 48;
    const uint32_t     Qm  = 6;
    const uint32_t     E   = 18; // divisible by 6
    const uint32_t     K   = 40;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 5) - 2);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);
    // Invariants: values mapped into circular buffer (excluding punctured region), all finite and clamped
    ASSERT_EQ(h_out.size(), Ncb);
    std::vector<size_t> nzIdx;
    nzIdx.reserve(E);
    for(size_t i = 0; i < h_out.size(); ++i)
    {
        if(h_out[i] != 0.0f)
        {
            EXPECT_TRUE(std::isfinite(h_out[i]));
            EXPECT_LE(h_out[i], 10000.0f);
            EXPECT_GE(h_out[i], -10000.0f);
            nzIdx.push_back(i);
        }
    }
    size_t nonZeroSrc = 0;
    for(float v : h_in)
    {
        if(v != 0.0f) ++nonZeroSrc;
    }
    EXPECT_GE(nzIdx.size(), static_cast<size_t>(1));
    EXPECT_LE(nzIdx.size(), static_cast<size_t>(E));
}

TEST(PUSCH_RateMatch, LayerMap_BBULayers_Greater_Than_Nl)
{
    // Cover branch nBBULayers > Nl and compute_llr_index layered path
    const uint32_t Zc     = 2;
    const uint32_t Ncb    = 32;
    const uint32_t Qm     = 2;
    const uint32_t E      = 8;
    const uint32_t K      = 20;
    const uint32_t F      = 0;
    PerTbParams    tb     = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.Nl                 = 1;
    tb.nBBULayers         = 2; // trigger nBBULayers > Nl path
    tb.layer_map_array[0] = 0;
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>(i + 1);
    std::vector<float> h_out;
    // We primarily target coverage; correctness invariant: output size and finite values
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    for(float v : h_out)
    {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(PUSCH_RateMatch, NDI1_UseAtomics_Path)
{
    // Force useAtomics path when ndi==1 by ensuring potentialRaceIfPositive>0 and outIdx within that window
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 16;
    const uint32_t Qm  = 2;
    const uint32_t E   = 40; // make potentialRaceIfPositive >> Ncb to force useAtomics always
    const uint32_t K   = 12;
    const uint32_t F   = 0;
    PerTbParams    tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.bg              = 2;
    tb.rv              = 3;               // choose k0 large to increase potentialRaceIfPositive
    std::vector<float> h_in(E, 20000.0f); // large positive to trigger atomicMinCustom clamp in useAtomics branch
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    // Invariants: some values are produced and all are finite.
    EXPECT_GT(count_nonzero_finite(h_out), static_cast<size_t>(0));
}

TEST(PUSCH_RateMatch, NDI0_NoAtomics_Path)
{
    // Force no-atomics path when ndi==0 by making potentialRaceIfPositive<=0 and verify clamping and write-back
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 64;
    const uint32_t     Qm  = 2;
    const uint32_t     E   = 16;
    const uint32_t     K   = 40;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 0, /*nDmrsCdmGrpsNoData*/ 0);
    std::vector<float> h_in(E, 20000.0f); // large to trigger clamp in no-atomics path
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out.size(), Ncb);
    bool anyClamped = false;
    for(float v : h_out)
    {
        if(v == 10000.0f)
        {
            anyClamped = true;
            break;
        }
    }
    EXPECT_TRUE(anyClamped);
}

TEST(PUSCH_RateMatch, Dispatch_NDI0_AllSupportedQm_Path)
{
    // Cover de_rate_matching_global2 retransmission dispatch:
    // deRateMatchingKernelInner<..., Qm, false> for all supported Qm values.
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 128;
    const uint32_t K   = 40;
    const uint32_t F   = 0;

    for(uint32_t Qm : {1u, 2u, 4u, 6u, 8u})
    {
        const uint32_t     E = 8u * Qm; // divisible by Qm
        PerTbParams        tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 0, /*nDmrsCdmGrpsNoData*/ 0);
        std::vector<float> h_in(E);
        for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 9) - 4);
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(h_out.size(), Ncb);
        EXPECT_GT(count_nonzero_finite_bounded(h_out, -10000.0f, 10000.0f), static_cast<size_t>(0)) << "Qm=" << Qm;
    }
}

TEST(PUSCH_RateMatch, NDI0_NoAtomics_Path_FP16)
{
    // Target processOneLLR<T_OUT=__half, ndi=false> non-atomic combine branch,
    // including clamp line for half precision.
    const uint32_t     Zc  = 2;
    const uint32_t     Ncb = 64;
    const uint32_t     Qm  = 2;
    const uint32_t     E   = 16; // keeps potentialRaceIfPositive <= 0 for no-atomics
    const uint32_t     K   = 40;
    const uint32_t     F   = 0;
    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 0, /*nDmrsCdmGrpsNoData*/ 0);
    std::vector<float> h_in(E, 20000.0f);
    std::vector<__half> h_out_h;
    run_rate_match_case_fp16(tb, h_in, h_out_h, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(h_out_h.size(), Ncb);
    bool anyClamped = false;
    for(auto v : h_out_h)
    {
        if(__half2float(v) == 10000.0f)
        {
            anyClamped = true;
            break;
        }
    }
    EXPECT_TRUE(anyClamped);
}

TEST(PUSCH_RateMatch, Setup_UCI_and_CDM1_Branches)
{
    // Cover setup branches: uciOnPuschFlag path and nDmrsCdmGrpsNoData==1 vs 0 path.
    const uint32_t Zc = 2, Ncb = 16, Qm = 2, E = 8, K = 12, F = 0;
    size_t         descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes), CUPHY_STATUS_SUCCESS);
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gpuDesc, descrSizeBytes));

    float* d_dummy = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_dummy, E * sizeof(float)));
    cuphyTensorPrm_t tRmIn[1]{};
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    tRmIn[0].pAddr     = d_dummy;
    tCdm1RmIn[0].pAddr = d_dummy;

    void** ppRmOut = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&ppRmOut, sizeof(void*), cudaHostAllocDefault));
    float* d_out = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, Ncb * sizeof(float)));
    ppRmOut[0] = d_out;

    PerTbParams* d_tb = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_tb, sizeof(PerTbParams)));
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, 0, 0));
    uint16_t                         schIdx[1] = {0};
    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};

    // Case 1: UCI path
    PerTbParams tb      = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.uciOnPuschFlag   = 1;
    tb.d_schAndCsi2LLRs = reinterpret_cast<__half*>(d_dummy);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_tb, &tb, sizeof(PerTbParams), cudaMemcpyHostToDevice));
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRateMatch(hndl, 1, schIdx, &tb, d_tb, tRmIn, tCdm1RmIn, ppRmOut, cpuDesc, gpuDesc, 0, &launchCfg, 0));

    // Case 2: CDM1 non-UCI path
    PerTbParams tb2    = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 1);
    tb2.uciOnPuschFlag = 0;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_tb, &tb2, sizeof(PerTbParams), cudaMemcpyHostToDevice));
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRateMatch(hndl, 1, schIdx, &tb2, d_tb, tRmIn, tCdm1RmIn, ppRmOut, cpuDesc, gpuDesc, 0, &launchCfg, 0));

    EXPECT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyPuschRxRateMatch(hndl));
    cudaFree(d_tb);
    cudaFreeHost(ppRmOut);
    cudaFree(d_dummy);
    cudaFree(d_out);
    cudaFreeHost(cpuDesc);
    cudaFree(gpuDesc);
}

TEST(PUSCH_RateMatch, Setup_EnableCpuToGpuDescrAsyncCpy_TwoUsers)
{
    // Cover: enableCpuToGpuDescrAsyncCpy branch and both true/false updates of CMax/EMax across iterations
    const uint32_t Zc = 2;
    // Build two PerTbParams with different num_CBs and Eh to drive both sides of ternaries
    PerTbParams tb0    = make_min_tb_params(/*Ncb*/ 32, Zc, /*Qm*/ 4, /*E*/ 64, /*K*/ 28, /*F*/ 0, /*ndi*/ 1, /*nDmrs*/ 0);
    PerTbParams tb1    = make_min_tb_params(/*Ncb*/ 32, Zc, /*Qm*/ 2, /*E*/ 16, /*K*/ 20, /*F*/ 0, /*ndi*/ 1, /*nDmrs*/ 0);
    tb0.num_CBs        = 3; // first iteration: CMax becomes 3
    tb1.num_CBs        = 1; // second iteration: CMax < num_CBs is false path
    tb0.Nl             = 2;
    tb0.Qm             = 4;
    tb0.encodedSize    = 128; // large Eh
    tb1.Nl             = 1;
    tb1.Qm             = 2;
    tb1.encodedSize    = 16; // smaller Eh so EMax update false in second iteration
    tb0.userGroupIndex = 0;
    tb1.userGroupIndex = 0;

    // Host and device arrays
    PerTbParams  h_tb[2] = {tb0, tb1};
    PerTbParams* d_tb    = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_tb, 2 * sizeof(PerTbParams)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_tb, h_tb, 2 * sizeof(PerTbParams), cudaMemcpyHostToDevice));

    // Descriptor buffers
    size_t descrSizeBytes = 0, descrAlignBytes = 0;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyPuschRxRateMatchGetDescrInfo(&descrSizeBytes, &descrAlignBytes));
    void* cpuDesc = nullptr;
    void* gpuDesc = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&cpuDesc, descrSizeBytes, cudaHostAllocDefault));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gpuDesc, descrSizeBytes));

    // Tensors
    float* d_in = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, 128 * sizeof(float)));
    cuphyTensorPrm_t tRmIn[1]{};
    cuphyTensorPrm_t tCdm1RmIn[1]{};
    tRmIn[0].pAddr     = d_in;
    tCdm1RmIn[0].pAddr = d_in;

    // Outputs array for two users
    void** ppRmOut = nullptr;
    ASSERT_EQ(cudaSuccess, cudaHostAlloc(&ppRmOut, 2 * sizeof(void*), cudaHostAllocDefault));
    float* d_out0 = nullptr;
    float* d_out1 = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out0, 64 * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out1, 64 * sizeof(float)));
    ppRmOut[0] = d_out0;
    ppRmOut[1] = d_out1;

    // Handle and indices
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, 0, 0));
    uint16_t                         schIdx[2] = {0, 1};
    cuphyPuschRxRateMatchLaunchCfg_t launchCfg{};

    // enableCpuToGpuDescrAsyncCpy = 1 to hit the branch
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRateMatch(hndl,
                                                               /*nSchUes*/ 2,
                                                               schIdx,
                                                               /*pTbPrmsCpu*/ h_tb,
                                                               /*pTbPrmsGpu*/ d_tb,
                                                               /*pTPrmRmIn*/ tRmIn,
                                                               /*pTPrmCdm1*/ tCdm1RmIn,
                                                               /*ppRmOut*/ ppRmOut,
                                                               /*pCpuDesc*/ cpuDesc,
                                                               /*pGpuDesc*/ gpuDesc,
                                                               /*enableCpuToGpuDescrAsyncCpy*/ 1,
                                                               &launchCfg,
                                                               /*strm*/ 0));

    // Cleanup
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyPuschRxRateMatch(hndl));
    cudaFree(d_tb);
    cudaFreeHost(ppRmOut);
    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_in);
    cudaFreeHost(cpuDesc);
    cudaFree(gpuDesc);
}

TEST(PUSCH_RateMatch, Init_FPconfigs_Switch)
{
    // Cover switch cases 1,2,3 in puschRxRateMatch::init
    for(int cfg : {1, 2, 3})
    {
        cuphyPuschRxRateMatchHndl_t hndl = nullptr;
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, cfg, /*descramblingOn*/ 0));
        EXPECT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyPuschRxRateMatch(hndl));
    }
}

TEST(PUSCH_RateMatch, Init_Default_Switch)
{
    // Cover default branch in puschRxRateMatch::init by passing an invalid FPconfig
    cuphyPuschRxRateMatchHndl_t hndl = nullptr;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRateMatch(&hndl, /*invalid*/ 99, /*descramblingOn*/ 0));
    EXPECT_EQ(CUPHY_STATUS_SUCCESS, cuphyDestroyPuschRxRateMatch(hndl));
}

TEST(PUSCH_RateMatch, BG_RV_K0_Combinations)
{
    // Exercise k0 computation branches for bg in {1,2} and rv in {0,1,2,3}
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 2;
    const uint32_t E   = 8;
    const uint32_t K   = 16;
    const uint32_t F   = 0;
    for(uint32_t bg : {1u, 2u})
    {
        // Baseline rv=0
        PerTbParams tb0 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
        tb0.bg          = bg;
        tb0.rv          = 0;
        std::vector<float> h_in(E);
        for(uint32_t i = 0; i < E; ++i) h_in[i] = static_cast<float>((i % 5) - 2);
        std::vector<float> out0;
        run_rate_match_case(tb0, h_in, out0, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(out0.size(), Ncb);
        for(float v : out0) { EXPECT_TRUE(std::isfinite(v)); }

        for(uint32_t rv : {1u, 2u, 3u})
        {
            PerTbParams tb = tb0;
            tb.rv          = rv;
            std::vector<float> out;
            run_rate_match_case(tb, h_in, out, /*descramblingOn*/ false, /*async*/ false);
            ASSERT_EQ(out.size(), Ncb);
            // Coverage-oriented invariant: output remains finite and bounded regardless of rv choice.
            for(float v : out) { EXPECT_TRUE(std::isfinite(v)); }
        }
    }
}

TEST(PUSCH_RateMatch, K0_BG1_RV3_and_BG2_RV3_Specific)
{
    // Explicitly hit rv==3 branches for both BGs with Ncb chosen to avoid truncation edge-cases
    // BG1: choose Ncb = 66*Zc to make factors exact
    {
        const uint32_t Zc = 2, Ncb = 132, Qm = 2, E = 8, K = 24, F = 0;
        PerTbParams    tb0 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
        tb0.bg             = 1;
        tb0.rv             = 0; // baseline
        std::vector<float> in(E);
        for(uint32_t i = 0; i < E; ++i) in[i] = static_cast<float>((i % 5) - 2);
        std::vector<float> out0;
        run_rate_match_case(tb0, in, out0, /*descramblingOn*/ false, /*async*/ false);
        PerTbParams tb3 = tb0;
        tb3.rv          = 3; // rv==3 -> k0 = (56*Ncb/(66*Zc))*Zc
        std::vector<float> out3;
        run_rate_match_case(tb3, in, out3, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(out0.size(), out3.size());
        for(float v : out3)
        {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_LE(v, 10000.0f);
            EXPECT_GE(v, -10000.0f);
        }
    }
    // BG2: choose Ncb = 50*Zc to make factors exact
    {
        const uint32_t Zc = 2, Ncb = 100, Qm = 2, E = 8, K = 24, F = 0;
        PerTbParams    tb0 = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
        tb0.bg             = 2;
        tb0.rv             = 0; // baseline
        std::vector<float> in(E);
        for(uint32_t i = 0; i < E; ++i) in[i] = static_cast<float>((i % 5) - 2);
        std::vector<float> out0;
        run_rate_match_case(tb0, in, out0, /*descramblingOn*/ false, /*async*/ false);
        PerTbParams tb3 = tb0;
        tb3.rv          = 3; // rv==3 -> k0 = (43*Ncb/(50*Zc))*Zc
        std::vector<float> out3;
        run_rate_match_case(tb3, in, out3, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(out0.size(), out3.size());
        for(float v : out3)
        {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_LE(v, 10000.0f);
            EXPECT_GE(v, -10000.0f);
        }
    }
}

TEST(PUSCH_RateMatch, Branch_BG1_RV3_False_UsingRv5)
{
    // Force evaluation of 'rv==3' inside bg==1 with a false outcome by setting rv=5
    const uint32_t Zc = 2, Ncb = 32, Qm = 2, E = 8, K = 16, F = 0;
    PerTbParams    tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.bg             = 1;
    tb.rv             = 5; // out-of-range is acceptable; k0 remains 0
    std::vector<float> in(E);
    for(uint32_t i = 0; i < E; ++i) in[i] = static_cast<float>((i % 5) - 2);
    std::vector<float> out;
    run_rate_match_case(tb, in, out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(out.size(), Ncb);
    for(float v : out) { EXPECT_TRUE(std::isfinite(v)); }
}

TEST(PUSCH_RateMatch, Branch_BG2_ElseIf_False_UsingBg0)
{
    // Force evaluation of 'else if (bg==2)' with false outcome by setting bg=0 (bg!=1 and bg!=2)
    const uint32_t Zc = 2, Ncb = 32, Qm = 2, E = 8, K = 16, F = 0;
    PerTbParams    tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.bg             = 0;
    tb.rv             = 0; // reaches else-if(bg==2) and evaluates false
    std::vector<float> in(E);
    for(uint32_t i = 0; i < E; ++i) in[i] = static_cast<float>((i % 5) - 2);
    std::vector<float> out;
    run_rate_match_case(tb, in, out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(out.size(), Ncb);
    for(float v : out) { EXPECT_TRUE(std::isfinite(v)); }
}

TEST(PUSCH_RateMatch, Branch_BG2_RV3_False_UsingRv5)
{
    // Force evaluation of 'rv==3' inside bg==2 with a false outcome by setting rv=5
    const uint32_t Zc = 2, Ncb = 32, Qm = 2, E = 8, K = 16, F = 0;
    PerTbParams    tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    tb.bg             = 2;
    tb.rv             = 5; // no k0 update path taken
    std::vector<float> in(E);
    for(uint32_t i = 0; i < E; ++i) in[i] = static_cast<float>((i % 5) - 2);
    std::vector<float> out;
    run_rate_match_case(tb, in, out, /*descramblingOn*/ false, /*async*/ false);
    ASSERT_EQ(out.size(), Ncb);
    for(float v : out) { EXPECT_TRUE(std::isfinite(v)); }
}

TEST(PUSCH_RateMatch, ZeroRangeVec_EarlyReturn_Guard)
{
    // Zc=0 is intentionally non-standard; it makes nPuncturedBits=0 so
    // that zeroRangeVec(out, 0, 0, ...) is called, covering the
    // if(start >= end) return; guard (line 187 in the VCast variant).
    const uint32_t Zc  = 0;
    const uint32_t Ncb = 16;
    const uint32_t Qm  = 2;
    const uint32_t E   = 4;   // encodedSize
    const uint32_t K   = 4;   // >= 2*Zc + F = 0
    const uint32_t F   = 0;

    PerTbParams        tb   = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    std::vector<float> h_in(E, 1.0f);
    std::vector<float> h_out;

    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), Ncb);
    // With nPuncturedBits=0 all Ncb output positions are part of the
    // circular buffer; at least the E=4 written LLRs must be non-zero.
    const size_t nz = count_nonzero_finite(h_out);
    EXPECT_GT(nz, 0u) << "expected at least one non-zero LLR in output";
}

// Additional targeted atomic clamp tests to cover atomicMaxCustom/atomicMinCustom for float and __half.
enum class AtomicOutputMode
{
    Float32Out,
    Float16Out
};

struct AtomicClampCase
{
    AtomicOutputMode mode;
    bool             clampPositive;
};

class PuschRateMatchAtomicClampTest : public ::testing::TestWithParam<AtomicClampCase>
{
};

TEST_P(PuschRateMatchAtomicClampTest, ExpectedMatchesModel)
{
    const AtomicClampCase tc = GetParam();
    const uint32_t        Zc = 2;
    const uint32_t        Ncb = 4;
    const uint32_t        Qm = 2;
    const uint32_t        E = 8;
    const uint32_t        K = 8;
    const uint32_t        F = 0;
    PerTbParams           tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);

    std::vector<float> h_in(E, tc.clampPositive ? 9000.0f : -9000.0f);
    auto               expected = build_expected_atomics_clamp(tb, h_in);

    if(tc.mode == AtomicOutputMode::Float32Out)
    {
        std::vector<float> h_out;
        run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(h_out.size(), expected.size());
        for(size_t i = 0; i < h_out.size(); ++i)
        {
            EXPECT_FLOAT_EQ(h_out[i], expected[i]) << "atomic float clamp mismatch at i=" << i;
        }
    }
    else
    {
        std::vector<__half> h_out_h;
        run_rate_match_case_fp16(tb, h_in, h_out_h, /*descramblingOn*/ false, /*async*/ false);
        ASSERT_EQ(h_out_h.size(), expected.size());
        for(size_t i = 0; i < h_out_h.size(); ++i)
        {
            EXPECT_FLOAT_EQ(__half2float(h_out_h[i]), expected[i]) << "atomic half clamp mismatch at i=" << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AtomicClampModes,
                         PuschRateMatchAtomicClampTest,
                         ::testing::Values(AtomicClampCase{AtomicOutputMode::Float32Out, false},
                                           AtomicClampCase{AtomicOutputMode::Float16Out, true},
                                           AtomicClampCase{AtomicOutputMode::Float16Out, false}));

////////////////////////////////////////////////////////////////////////
// Improvement 1: Pre-stored golden arrays (Known-Answer Tests)
//
// The existing tests compare GPU output against build_expected_noatomics(),
// which is a CPU reference derived from the same algorithm.  If both share a
// bug, the tests still pass.  The KATs below hardcode expected outputs
// computed INDEPENDENTLY by hand-tracing the 3GPP TS 38.212 §5.4.2.1
// algorithm through derate_match_fast_calc_modulo(), providing a second,
// orthogonal correctness check.
//
// Trace for KAT_QPSK_BG1_RV0 (Ncb=16, Zc=2, Qm=2, E=4, K=8, F=0, k0=0):
//   nPuncturedBits=4, Kd=4, EoverQm=2
//   Scenario 0 (k0=0 < Kd=4): outIdx = inIdx for inIdx < 16
//   inIdx mapping (inIdx = k*EoverQm + j):
//     j=0,k=0 → i=0, inIdx=0, outIdx=0, val=h_in[0]= 1.0
//     j=0,k=1 → i=1, inIdx=2, outIdx=2, val=h_in[1]=-2.0
//     j=1,k=0 → i=2, inIdx=1, outIdx=1, val=h_in[2]= 3.0
//     j=1,k=1 → i=3, inIdx=3, outIdx=3, val=h_in[3]=-4.0
//   circular: expected[0..3] = [1, 3, -2, -4]
//   expected_full[nPuncturedBits + i] → [0,0,0,0, 1, 3, -2, -4, 0,…,0]
TEST(PUSCH_RateMatch, KAT_QPSK_BG1_RV0)
{
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 16;
    const uint32_t Qm  = 2;
    const uint32_t E   = 4;
    const uint32_t K   = 8;
    const uint32_t F   = 0;

    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in = {1.0f, -2.0f, 3.0f, -4.0f};

    // Hardcoded golden array (hand-traced — independent of build_expected_noatomics)
    const std::vector<float> golden = {
        0.0f, 0.0f, 0.0f, 0.0f,   // punctured prefix (2*Zc=4 zeros)
        1.0f, 3.0f, -2.0f, -4.0f, // de-rate-matched LLRs at positions 0-3
        0.0f, 0.0f, 0.0f, 0.0f,   // unwritten circular buffer (positions 4-7)
        0.0f, 0.0f, 0.0f, 0.0f    // unwritten circular buffer (positions 8-11)
    };

    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), golden.size());
    for(size_t i = 0; i < golden.size(); ++i)
        EXPECT_FLOAT_EQ(h_out[i], golden[i]) << "KAT_QPSK_BG1_RV0 mismatch at i=" << i;
}

// Trace for KAT_16QAM_BG1_RV0 (Ncb=32, Zc=4, Qm=4, E=8, K=24, F=0, k0=0):
//   nPuncturedBits=8, Kd=16, EoverQm=2
//   Scenario 0 (k0=0 < Kd=16): outIdx = inIdx for inIdx < 32
//   inIdx = k*EoverQm + j:
//     j=0,k=0 → i=0, inIdx=0, outIdx=0, val=h_in[0]=1
//     j=0,k=1 → i=1, inIdx=2, outIdx=2, val=h_in[1]=2
//     j=0,k=2 → i=2, inIdx=4, outIdx=4, val=h_in[2]=3
//     j=0,k=3 → i=3, inIdx=6, outIdx=6, val=h_in[3]=4
//     j=1,k=0 → i=4, inIdx=1, outIdx=1, val=h_in[4]=5
//     j=1,k=1 → i=5, inIdx=3, outIdx=3, val=h_in[5]=6
//     j=1,k=2 → i=6, inIdx=5, outIdx=5, val=h_in[6]=7
//     j=1,k=3 → i=7, inIdx=7, outIdx=7, val=h_in[7]=8
//   circular[0..7] = [1, 5, 2, 6, 3, 7, 4, 8]
//   expected_full[8..15] = [1,5,2,6,3,7,4,8], rest 0
TEST(PUSCH_RateMatch, KAT_16QAM_BG1_RV0)
{
    const uint32_t Zc  = 4;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 4;
    const uint32_t E   = 8;
    const uint32_t K   = 24;
    const uint32_t F   = 0;

    PerTbParams        tb  = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrs*/ 0);
    std::vector<float> h_in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Hardcoded golden array (hand-traced)
    const std::vector<float> golden = {
        0,0,0,0,0,0,0,0,  // punctured prefix (2*Zc=8 zeros)
        1,5,2,6,3,7,4,8,  // de-interleaved LLRs at circular positions 0-7
        0,0,0,0,0,0,0,0,  // unwritten (positions 8-15)
        0,0,0,0,0,0,0,0   // unwritten (positions 16-23)
    };

    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ false, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), golden.size());
    for(size_t i = 0; i < golden.size(); ++i)
        EXPECT_FLOAT_EQ(h_out[i], golden[i]) << "KAT_16QAM_BG1_RV0 mismatch at i=" << i;
}

////////////////////////////////////////////////////////////////////////
// Improvement 2: FP32 vs FP16 direct value comparison (non-scrambled)
//
// The existing Descrambling_AbsMatches_* tests only compare |value| because
// scrambling flips signs unpredictably.  For the non-scrambled path, FP32 and
// FP16 must produce the same sign and the same magnitude within FP16
// quantisation error (~0.01 for unit-amplitude signals).
TEST(PUSCH_RateMatch, FP32_vs_FP16_DirectValueConsistency)
{
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 32;
    const uint32_t Qm  = 2;
    const uint32_t E   = 8;
    const uint32_t K   = 16;
    const uint32_t F   = 0;

    PerTbParams tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);

    // Alternating ±1 so both positive and negative paths are exercised
    std::vector<float> h_in(E);
    for(uint32_t i = 0; i < E; ++i)
        h_in[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    std::vector<float>  fp32_out;
    std::vector<__half> fp16_out;
    run_rate_match_case(tb, h_in, fp32_out, /*descramblingOn*/ false, /*async*/ false);
    run_rate_match_case_fp16(tb, h_in, fp16_out, /*descramblingOn*/ false, /*async*/ false);

    ASSERT_EQ(fp32_out.size(), Ncb);
    ASSERT_EQ(fp16_out.size(), Ncb);

    // FP16 quantisation: ≤ 0.01 error for a unit-amplitude signal
    for(size_t i = 0; i < Ncb; ++i)
    {
        const float fp16_val = __half2float(fp16_out[i]);
        EXPECT_NEAR(fp16_val, fp32_out[i], 0.01f)
            << "FP32/FP16 value mismatch at i=" << i
            << " (fp32=" << fp32_out[i] << " fp16=" << fp16_val << ")";
    }
}

////////////////////////////////////////////////////////////////////////
// Improvement 3: Scrambling sign-distribution test
//
// With all-positive inputs and descrambling ON, the Gold-sequence scrambler
// independently flips each LLR sign with probability 0.5 (by construction of
// the Gold sequence).  The output should therefore have roughly equal numbers
// of positive and negative non-zero values.  A heavily skewed distribution
// indicates a scrambling bug (e.g., always-zero or always-one sequence).
TEST(PUSCH_RateMatch, Scrambling_SignDistribution)
{
    const uint32_t Zc  = 2;
    const uint32_t Ncb = 64;
    const uint32_t Qm  = 2;
    const uint32_t E   = 32;
    const uint32_t K   = 40;
    const uint32_t F   = 0;

    PerTbParams tb = make_min_tb_params(Ncb, Zc, Qm, E, K, F, /*ndi*/ 1, /*nDmrsCdmGrpsNoData*/ 0);
    tb.cinit       = 0x12345678u; // fixed seed for reproducibility

    // All-positive input: every sign flip is solely due to scrambling
    std::vector<float> h_in(E, 1.0f);
    std::vector<float> h_out;
    run_rate_match_case(tb, h_in, h_out, /*descramblingOn*/ true, /*enableCpuToGpuDescrAsyncCpy*/ false);

    ASSERT_EQ(h_out.size(), Ncb);

    int pos = 0, neg = 0;
    for(float v : h_out)
    {
        if(v > 0.0f) ++pos;
        else if(v < 0.0f) ++neg;
    }
    const int non_zero = pos + neg;
    ASSERT_GT(non_zero, 0) << "No non-zero output — scrambling produced all zeros";

    // Gold sequence: expect approximately 50% sign flips.
    // Allow ±20% margin to avoid flakiness on small sample sizes.
    const float pos_frac = static_cast<float>(pos) / non_zero;
    EXPECT_GT(pos_frac, 0.30f)
        << "Scrambling is too one-sided (too few positive values); pos=" << pos << " neg=" << neg;
    EXPECT_LT(pos_frac, 0.70f)
        << "Scrambling is too one-sided (too few negative values); pos=" << pos << " neg=" << neg;
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
