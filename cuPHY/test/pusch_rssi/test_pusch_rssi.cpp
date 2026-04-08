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
#include "cuphy.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <cstring>

class PuschRssiGTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        cuInit(0);
        ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking));
        // Query descriptor sizes
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphyPuschRxRssiGetDescrInfo(&rssiDynDescrSizeBytes,
                                               &rssiDynDescrAlignBytes,
                                               &rsrpDynDescrSizeBytes,
                                               &rsrpDynDescrAlignBytes));

        // Allocate CPU/GPU buffers
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&rssiDynDescrCpu, rssiDynDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&rssiDynDescrGpu, rssiDynDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&rsrpDynDescrCpu, rsrpDynDescrSizeBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&rsrpDynDescrGpu, rsrpDynDescrSizeBytes));

        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphyCreatePuschRxRssi(&rssiHndl));
    }

    void TearDown() override
    {
        if(rssiHndl)
        {
            cuphyDestroyPuschRxRssi(rssiHndl);
            rssiHndl = nullptr;
        }
        if(rssiDynDescrCpu)
        {
            cudaFreeHost(rssiDynDescrCpu);
            rssiDynDescrCpu = nullptr;
        }
        if(rssiDynDescrGpu)
        {
            cudaFree(rssiDynDescrGpu);
            rssiDynDescrGpu = nullptr;
        }
        if(rsrpDynDescrCpu)
        {
            cudaFreeHost(rsrpDynDescrCpu);
            rsrpDynDescrCpu = nullptr;
        }
        if(rsrpDynDescrGpu)
        {
            cudaFree(rsrpDynDescrGpu);
            rsrpDynDescrGpu = nullptr;
        }
        if(drvdUeGrpPrmsCpu)
        {
            cudaFreeHost(drvdUeGrpPrmsCpu);
            drvdUeGrpPrmsCpu = nullptr;
        }
        if(drvdUeGrpPrmsGpu)
        {
            cudaFree(drvdUeGrpPrmsGpu);
            drvdUeGrpPrmsGpu = nullptr;
        }
        cudaStreamDestroy(cuStream);
        cudaDeviceSynchronize();
    }

    void allocUeGrp(uint16_t nUeGrps)
    {
        ASSERT_EQ(cudaSuccess, cudaMallocHost(&drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&drvdUeGrpPrmsGpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t)));
        std::memset(drvdUeGrpPrmsCpu, 0, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t));
    }

    // Minimal tensor helper: set base pointers and strides for 1D/3D tensors used by RSSI/RSRP
    static void setTensor1D(cuphyTensorInfo1_t& ti, void* p, cuphyDataType_t t, int32_t n0)
    {
        ti.pAddr      = p;
        ti.elemType   = t;
        ti.strides[0] = 1;
        (void)n0;
    }
    static void setTensor3D(cuphyTensorInfo3_t& ti, void* p, cuphyDataType_t t, int32_t s0, int32_t s1, int32_t s2)
    {
        ti.pAddr      = p;
        ti.elemType   = t;
        ti.strides[0] = s0;
        ti.strides[1] = s1;
        ti.strides[2] = s2;
    }
    static void setTensor4D(cuphyTensorInfo4_t& ti, void* p, cuphyDataType_t t, int32_t s0, int32_t s1, int32_t s2, int32_t s3)
    {
        ti.pAddr      = p;
        ti.elemType   = t;
        ti.strides[0] = s0;
        ti.strides[1] = s1;
        ti.strides[2] = s2;
        ti.strides[3] = s3;
    }

    void validateOutput(const std::vector<cuComplex>& dataRx,
                        const std::vector<float>&     rssiOut,
                        const std::vector<float>&     rssiEhq,
                        const std::vector<float>&     rsrp,
                        const std::vector<float>&     sinrPre,
                        const std::vector<float>&     sinrPost,
                        const std::vector<float>&     noisePre,
                        const std::vector<float>&     noisePost,
                        uint16_t                      nPrb,
                        uint8_t                       nRxAnt,
                        uint8_t                       nLayers,
                        uint8_t                       dmrsMaxLen,
                        uint16_t                      dmrsSymBmsk,
                        bool                          expectPostEqMetrics)
    {
        const uint32_t nSc = nPrb * CUPHY_N_TONES_PER_PRB;

        auto sumPowerDataRx = [&](uint8_t symIdxStart, uint8_t nSymsSel) -> double {
            double s = 0.0;
            for(uint8_t symOff = 0; symOff < nSymsSel; ++symOff)
            {
                uint8_t sym = drvdUeGrpPrmsCpu[0].dmrsSymLoc[symIdxStart + symOff];
                for(uint32_t sc = 0; sc < nSc; ++sc)
                {
                    for(uint8_t ant = 0; ant < nRxAnt; ++ant)
                    {
                        size_t idx = sc + sym * static_cast<uint32_t>(nSc) + ant * static_cast<uint32_t>(nSc) * OFDM_SYMBOLS_PER_SLOT;
                        float  xr  = cuCrealf(dataRx[idx]);
                        float  xi  = cuCimagf(dataRx[idx]);
                        s += static_cast<double>(xr) * xr + static_cast<double>(xi) * xi;
                    }
                }
            }
            return s;
        };

        const uint32_t totalDmrsSel = __builtin_popcount(dmrsSymBmsk);

        // Validate FIRST DMRS RSSI (Ehq) - relaxed
        if(dmrsMaxLen > 0)
        {
            double sumFirst    = sumPowerDataRx(0, dmrsMaxLen);
            double rssiDbFirst = 10.0 * std::log10(sumFirst / static_cast<double>(dmrsMaxLen));
            if(std::isfinite(rssiEhq[0]) && std::isfinite(rssiDbFirst))
            {
                EXPECT_NEAR(rssiEhq[0], static_cast<float>(rssiDbFirst), 0.5f);
            }
            else
            {
                SUCCEED();
            }
        }

        // Validate final RSSI
        double expectedRssiDb = 0.0;
        if(totalDmrsSel > dmrsMaxLen)
        {
            double sumNoFirst = sumPowerDataRx(dmrsMaxLen, static_cast<uint8_t>(totalDmrsSel - dmrsMaxLen));
            expectedRssiDb    = 10.0 * std::log10(sumNoFirst / static_cast<double>(totalDmrsSel));
        }
        else
        {
            double sumFull = sumPowerDataRx(0, static_cast<uint8_t>(totalDmrsSel));
            expectedRssiDb = 10.0 * std::log10(sumFull / static_cast<double>(totalDmrsSel));
        }
        if(std::isfinite(rssiOut[0]) && std::isfinite(expectedRssiDb))
        {
            EXPECT_NEAR(rssiOut[0], static_cast<float>(expectedRssiDb), 0.5f);
        }
        else
        {
            SUCCEED();
        }

        // Validate final RSRP (per-UE)
        const double   hPwr           = 0.5 * 0.5 + 0.25 * 0.25;           // |0.5 - 0.25j|^2 = 0.3125
        const uint32_t nTimeChEst     = static_cast<uint32_t>(dmrsMaxLen); // dmrsAddlnPos+1
        double         timeUsed       = (totalDmrsSel > dmrsMaxLen) ? std::max<int>(static_cast<int>(nTimeChEst) - 1, 1) : nTimeChEst;
        double         rsrpLin        = static_cast<double>(nLayers) * hPwr * (static_cast<double>(timeUsed) / static_cast<double>(nTimeChEst));
        double         expectedRsrpDb = 10.0 * std::log10(rsrpLin);
        if(std::isfinite(rsrp[0]) && std::isfinite(expectedRsrpDb))
        {
            EXPECT_NEAR(rsrp[0], static_cast<float>(expectedRsrpDb), 1.0f);
        }
        else
        {
            SUCCEED();
        }

        // Basic cross-metric consistency checks.
        if(std::isfinite(sinrPre[0]) && std::isfinite(rsrp[0]) && std::isfinite(noisePre[0]))
        {
            EXPECT_NEAR(sinrPre[0], rsrp[0] - noisePre[0], 0.1f);
        }

        // post-eq outputs are produced by rsrpMeasKernel_v1 path only.
        if(expectPostEqMetrics && std::isfinite(sinrPost[0]) && std::isfinite(noisePost[0]))
        {
            EXPECT_NEAR(sinrPost[0], -noisePost[0], 0.1f);
        }
    }

    // Initializes a minimal valid configuration and launches RSSI/RSRP kernels (host+device)
    void runRssiRsrp(uint16_t nUeGrps,
                     uint16_t nPrb,
                     uint8_t  nRxAnt,
                     uint8_t  nLayers,
                     uint8_t  dmrsMaxLen,
                     uint16_t dmrsSymBmsk,
                     bool     enableTdi = false,
                     bool     enableWeightedAverageCfo = false)
    {
        allocUeGrp(nUeGrps);

        // Host buffers for inputs/outputs
        const uint32_t         nSc = nPrb * CUPHY_N_TONES_PER_PRB;
        std::vector<cuComplex> dataRx(nSc * OFDM_SYMBOLS_PER_SLOT * nRxAnt);
        std::vector<float>     rssiFull(OFDM_SYMBOLS_PER_SLOT * nRxAnt * nUeGrps, 0.0f);
        std::vector<float>     rssiOut(nUeGrps, 0.0f);
        std::vector<float>     rssiEhq(nUeGrps, 0.0f);
        std::vector<uint32_t>  rssiInterCta(nUeGrps, 0);

        std::vector<cuComplex> hEst(nRxAnt * nLayers * nSc * dmrsMaxLen);
        const uint32_t         reeElemsPerTime = nSc * nLayers;
        std::vector<float>     reeDiagInv(reeElemsPerTime * std::max<uint32_t>(dmrsMaxLen, 1), 1.0f);
        std::vector<float>     rsrp(nUeGrps, 0.0f);
        std::vector<float>     rsrpEhq(nUeGrps, 0.0f);
        std::vector<float>     sinrPre(nUeGrps, 0.0f);
        std::vector<float>     sinrPost(nUeGrps, 0.0f);
        std::vector<float>     noisePre(nUeGrps, 0.0f);
        std::vector<float>     noisePost(nUeGrps, 0.0f);

        // Fill deterministic input pattern
        for(size_t i = 0; i < dataRx.size(); ++i) dataRx[i] = make_cuComplex(static_cast<float>((i % 17) - 8), static_cast<float>(((i / 3) % 11) - 5));
        for(size_t i = 0; i < hEst.size(); ++i) hEst[i] = make_cuComplex(0.5f, -0.25f);

        // Allocate device buffers
        cuComplex *dDataRx = nullptr, *dHEst = nullptr;
        float *    dRssiFull = nullptr, *dRssi = nullptr, *dRssiEhq = nullptr;
        float *    dRsrp = nullptr, *dRsrpEhq = nullptr, *dSinrPre = nullptr, *dSinrPost = nullptr, *dNoisePre = nullptr, *dNoisePost = nullptr;
        uint32_t * dRssiInterCta = nullptr, *dRsrpInterCta = nullptr;
        float*     dReeDiagInv = nullptr;

        // Ensure device buffers are freed even when ASSERT_* aborts this helper early.
        struct DeviceBuffersCleanup
        {
            cuComplex*& dDataRx;
            cuComplex*& dHEst;
            float*&     dRssiFull;
            float*&     dRssi;
            float*&     dRssiEhq;
            uint32_t*&  dRssiInterCta;
            float*&     dReeDiagInv;
            float*&     dRsrp;
            float*&     dRsrpEhq;
            float*&     dSinrPre;
            float*&     dSinrPost;
            float*&     dNoisePre;
            float*&     dNoisePost;
            uint32_t*&  dRsrpInterCta;

            ~DeviceBuffersCleanup()
            {
                if(dDataRx) cudaFree(dDataRx);
                if(dHEst) cudaFree(dHEst);
                if(dRssiFull) cudaFree(dRssiFull);
                if(dRssi) cudaFree(dRssi);
                if(dRssiEhq) cudaFree(dRssiEhq);
                if(dRssiInterCta) cudaFree(dRssiInterCta);
                if(dReeDiagInv) cudaFree(dReeDiagInv);
                if(dRsrp) cudaFree(dRsrp);
                if(dRsrpEhq) cudaFree(dRsrpEhq);
                if(dSinrPre) cudaFree(dSinrPre);
                if(dSinrPost) cudaFree(dSinrPost);
                if(dNoisePre) cudaFree(dNoisePre);
                if(dNoisePost) cudaFree(dNoisePost);
                if(dRsrpInterCta) cudaFree(dRsrpInterCta);
            }
        } deviceBuffersCleanup{dDataRx,
                               dHEst,
                               dRssiFull,
                               dRssi,
                               dRssiEhq,
                               dRssiInterCta,
                               dReeDiagInv,
                               dRsrp,
                               dRsrpEhq,
                               dSinrPre,
                               dSinrPost,
                               dNoisePre,
                               dNoisePost,
                               dRsrpInterCta};

        ASSERT_EQ(cudaSuccess, cudaMalloc(&dDataRx, dataRx.size() * sizeof(cuComplex)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRssiFull, rssiFull.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRssi, rssiOut.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRssiEhq, rssiEhq.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRssiInterCta, rssiInterCta.size() * sizeof(uint32_t)));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dDataRx, dataRx.data(), dataRx.size() * sizeof(cuComplex), cudaMemcpyHostToDevice, cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiFull, 0, rssiFull.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssi, 0, rssiOut.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiEhq, 0, rssiEhq.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiInterCta, 0, rssiInterCta.size() * sizeof(uint32_t), cuStream));

        ASSERT_EQ(cudaSuccess, cudaMalloc(&dHEst, hEst.size() * sizeof(cuComplex)));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dHEst, hEst.data(), hEst.size() * sizeof(cuComplex), cudaMemcpyHostToDevice, cuStream));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dReeDiagInv, reeDiagInv.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dReeDiagInv, reeDiagInv.data(), reeDiagInv.size() * sizeof(float), cudaMemcpyHostToDevice, cuStream));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRsrp, rsrp.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRsrpEhq, rsrpEhq.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dSinrPre, sinrPre.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dSinrPost, sinrPost.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dNoisePre, noisePre.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dNoisePost, noisePost.size() * sizeof(float)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dRsrpInterCta, nUeGrps * sizeof(uint32_t)));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrp, 0, rsrp.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpEhq, 0, rsrpEhq.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dSinrPre, 0, sinrPre.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dSinrPost, 0, sinrPost.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dNoisePre, 0, noisePre.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dNoisePost, 0, noisePost.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpInterCta, 0, nUeGrps * sizeof(uint32_t), cuStream));

        // Populate UE group params (CPU)
        auto& prm          = drvdUeGrpPrmsCpu[0];
        prm.nUes           = 1;
        prm.nLayers        = nLayers;
        prm.nPrb           = nPrb;
        prm.nRxAnt         = nRxAnt;
        prm.nDmrsSyms      = __builtin_popcount(dmrsSymBmsk);
        prm.dmrsMaxLen     = dmrsMaxLen;
        prm.dmrsAddlnPos   = (dmrsMaxLen > 0) ? (dmrsMaxLen - 1) : 0;
        prm.enablePuschTdi = enableTdi ? 1 : 0;
        prm.enableWeightedAverageCfo = enableWeightedAverageCfo ? 1 : 0;
        prm.rssiSymPosBmsk = dmrsSymBmsk;
        // dmrsSymLoc array: fill all DMRS symbol indices in ascending order
        {
            uint8_t cnt = 0;
            for(uint8_t i = 0; i < OFDM_SYMBOLS_PER_SLOT; ++i)
            {
                if((dmrsSymBmsk >> i) & 1)
                {
                    prm.dmrsSymLoc[cnt++] = i;
                }
            }
        }
        // Bind tensor infos
        setTensor3D(prm.tInfoDataRx, dDataRx, CUPHY_C_32F, 1, nSc, nSc * OFDM_SYMBOLS_PER_SLOT);
        setTensor3D(prm.tInfoRssiFull, dRssiFull, CUPHY_R_32F, 1, OFDM_SYMBOLS_PER_SLOT, OFDM_SYMBOLS_PER_SLOT * nRxAnt);
        setTensor1D(prm.tInfoRssi, dRssi, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoRssiEhq, dRssiEhq, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoRssiInterCtaSyncCnt, dRssiInterCta, CUPHY_R_32U, nUeGrps);

        // tHEst(N_BS_ANTS, N_LAYERS, NF, NH) with simple packed layout
        setTensor4D(prm.tInfoHEst, dHEst, CUPHY_C_32F, 1, nRxAnt, nRxAnt * nLayers, nRxAnt * nLayers * nSc);
        // tReeDiagInv as 4D: (N_SC, N_LAYERS, N_PRB dummy=1, NH)
        prm.tInfoReeDiagInv.pAddr      = dReeDiagInv;
        prm.tInfoReeDiagInv.elemType   = CUPHY_R_32F;
        prm.tInfoReeDiagInv.strides[0] = 1;
        prm.tInfoReeDiagInv.strides[1] = nSc;
        prm.tInfoReeDiagInv.strides[2] = nSc * nLayers;
        prm.tInfoReeDiagInv.strides[3] = reeElemsPerTime;
        // Minimal UE mapping for RSRP accumulation
        for(uint8_t l = 0; l < nLayers; ++l)
        {
            prm.ueGrpLayerToUeIdx[l] = 0; // map all layers to UE 0
        }
        prm.ueIdxs[0]    = 0;
        prm.nUeLayers[0] = nLayers;
        setTensor1D(prm.tInfoRsrp, dRsrp, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoRsrpEhq, dRsrpEhq, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoSinrPreEq, dSinrPre, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoSinrPostEq, dSinrPost, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoNoiseVarPreEq, dNoisePre, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoNoiseVarPostEq, dNoisePost, CUPHY_R_32F, nUeGrps);
        setTensor1D(prm.tInfoRsrpInterCtaSyncCnt, dRsrpInterCta, CUPHY_R_32U, nUeGrps);

        // Copy UE group params to GPU when not using async copy
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

        // Prepare launch configs
        cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
        rssiLaunch.nCfgs = 1;
        cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
        rsrpLaunch.nCfgs = 1;

        // Setup and launch FIRST_DMRS (early-HARQ) paths
        // Ensure outputs/counters are reset before this pass
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiFull, 0, rssiFull.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssi, 0, rssiOut.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiEhq, 0, rssiEhq.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiInterCta, 0, rssiInterCta.size() * sizeof(uint32_t), cuStream));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRssi(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSSI_EST_FIRST_DMRS, 1, rssiDynDescrCpu, rssiDynDescrGpu, &rssiLaunch, cuStream));
        // Reset RSRP-related outputs/counters before early-HARQ pass
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrp, 0, rsrp.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpEhq, 0, rsrpEhq.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dSinrPost, 0, sinrPost.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpInterCta, 0, nUeGrps * sizeof(uint32_t), cuStream));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRsrp(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSRP_EST_FIRST_DMRS, 1, rsrpDynDescrCpu, rsrpDynDescrGpu, &rsrpLaunch, cuStream));

        // Launch the kernels via driver API
        {
            const CUDA_KERNEL_NODE_PARAMS& k1 = rssiLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_NE(k1.func, nullptr) << "RSSI FIRST kernel func is null (likely datatype mismatch or symbol not linked)";
            ASSERT_GT(k1.gridDimX, 0u);
            ASSERT_GT(k1.blockDimX, 0u);
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(k1.func, k1.gridDimX, k1.gridDimY, k1.gridDimZ, k1.blockDimX, k1.blockDimY, k1.blockDimZ, k1.sharedMemBytes, static_cast<CUstream>(cuStream), k1.kernelParams, k1.extra));
            const CUDA_KERNEL_NODE_PARAMS& k2 = rsrpLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(k2.func, k2.gridDimX, k2.gridDimY, k2.gridDimZ, k2.blockDimX, k2.blockDimY, k2.blockDimZ, k2.sharedMemBytes, static_cast<CUstream>(cuStream), k2.kernelParams, k2.extra));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
        }

        // Setup and launch FULL_SLOT_DMRS and, if applicable, WITHOUT_FIRST_DMRS
        // Reset RSSI outputs/counters before full-slot pass
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiFull, 0, rssiFull.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssi, 0, rssiOut.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiInterCta, 0, rssiInterCta.size() * sizeof(uint32_t), cuStream));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRssi(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS, 1, rssiDynDescrCpu, rssiDynDescrGpu, &rssiLaunch, cuStream));
        {
            const CUDA_KERNEL_NODE_PARAMS& kFullRssi = rssiLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_NE(kFullRssi.func, nullptr);
            ASSERT_GT(kFullRssi.gridDimX, 0u);
            ASSERT_GT(kFullRssi.blockDimX, 0u);
            // Overprovision gridDimZ by +1 to exercise (blockIdx.z >= nSymb*nRxAnt) path
            unsigned int gridZ = kFullRssi.gridDimZ + 1;
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(kFullRssi.func, kFullRssi.gridDimX, kFullRssi.gridDimY, gridZ, kFullRssi.blockDimX, kFullRssi.blockDimY, kFullRssi.blockDimZ, kFullRssi.sharedMemBytes, static_cast<CUstream>(cuStream), kFullRssi.kernelParams, kFullRssi.extra));
        }

        const uint32_t totalDmrsSel_forNoFirst = __builtin_popcount(dmrsSymBmsk);
        if(totalDmrsSel_forNoFirst > dmrsMaxLen)
        {
            // Reset before without-first pass
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiFull, 0, rssiFull.size() * sizeof(float), cuStream));
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssi, 0, rssiOut.size() * sizeof(float), cuStream));
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRssiInterCta, 0, rssiInterCta.size() * sizeof(uint32_t), cuStream));
            ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRssi(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS, 1, rssiDynDescrCpu, rssiDynDescrGpu, &rssiLaunch, cuStream));
            const CUDA_KERNEL_NODE_PARAMS& kNoFirstRssi = rssiLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_NE(kNoFirstRssi.func, nullptr);
            ASSERT_GT(kNoFirstRssi.gridDimX, 0u);
            ASSERT_GT(kNoFirstRssi.blockDimX, 0u);
            unsigned int gridZ2 = kNoFirstRssi.gridDimZ + 1;
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(kNoFirstRssi.func, kNoFirstRssi.gridDimX, kNoFirstRssi.gridDimY, gridZ2, kNoFirstRssi.blockDimX, kNoFirstRssi.blockDimY, kNoFirstRssi.blockDimZ, kNoFirstRssi.sharedMemBytes, static_cast<CUstream>(cuStream), kNoFirstRssi.kernelParams, kNoFirstRssi.extra));
        }

        // Reset RSRP outputs/counters before full-slot pass
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrp, 0, rsrp.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dSinrPost, 0, sinrPost.size() * sizeof(float), cuStream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpInterCta, 0, nUeGrps * sizeof(uint32_t), cuStream));
        ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRsrp(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS, 1, rsrpDynDescrCpu, rsrpDynDescrGpu, &rsrpLaunch, cuStream));
        {
            const CUDA_KERNEL_NODE_PARAMS& kFullRsrp = rsrpLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_NE(kFullRsrp.func, nullptr);
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(kFullRsrp.func, kFullRsrp.gridDimX, kFullRsrp.gridDimY, kFullRsrp.gridDimZ, kFullRsrp.blockDimX, kFullRsrp.blockDimY, kFullRsrp.blockDimZ, kFullRsrp.sharedMemBytes, static_cast<CUstream>(cuStream), kFullRsrp.kernelParams, kFullRsrp.extra));
        }

        if(totalDmrsSel_forNoFirst > dmrsMaxLen)
        {
            // Reset before without-first RSRP pass
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrp, 0, rsrp.size() * sizeof(float), cuStream));
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dSinrPost, 0, sinrPost.size() * sizeof(float), cuStream));
            ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dRsrpInterCta, 0, nUeGrps * sizeof(uint32_t), cuStream));
            ASSERT_EQ(CUPHY_STATUS_SUCCESS, cuphySetupPuschRxRsrp(rssiHndl, drvdUeGrpPrmsCpu, drvdUeGrpPrmsGpu, nUeGrps, nPrb, CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS, 1, rsrpDynDescrCpu, rsrpDynDescrGpu, &rsrpLaunch, cuStream));
            const CUDA_KERNEL_NODE_PARAMS& kNoFirstRsrp = rsrpLaunch.cfgs[0].kernelNodeParamsDriver;
            ASSERT_NE(kNoFirstRsrp.func, nullptr);
            ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(kNoFirstRsrp.func, kNoFirstRsrp.gridDimX, kNoFirstRsrp.gridDimY, kNoFirstRsrp.gridDimZ, kNoFirstRsrp.blockDimX, kNoFirstRsrp.blockDimY, kNoFirstRsrp.blockDimZ, kNoFirstRsrp.sharedMemBytes, static_cast<CUstream>(cuStream), kNoFirstRsrp.kernelParams, kNoFirstRsrp.extra));
        }
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

        // Read back results
        ASSERT_EQ(cudaSuccess, cudaMemcpy(rssiOut.data(), dRssi, rssiOut.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(rssiEhq.data(), dRssiEhq, rssiEhq.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(rsrp.data(), dRsrp, rsrp.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(sinrPre.data(), dSinrPre, sinrPre.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(sinrPost.data(), dSinrPost, sinrPost.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(noisePre.data(), dNoisePre, noisePre.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(noisePost.data(), dNoisePost, noisePost.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Validate outputs against host-computed expectations
        validateOutput(dataRx,
                       rssiOut,
                       rssiEhq,
                       rsrp,
                       sinrPre,
                       sinrPost,
                       noisePre,
                       noisePost,
                       nPrb,
                       nRxAnt,
                       nLayers,
                       dmrsMaxLen,
                       dmrsSymBmsk,
                       !enableWeightedAverageCfo);

        (void)deviceBuffersCleanup;
    }

protected:
    cudaStream_t           cuStream = nullptr;
    cuphyPuschRxRssiHndl_t rssiHndl = nullptr;

    // Descriptor buffers and sizes
    size_t rssiDynDescrSizeBytes = 0, rssiDynDescrAlignBytes = 0;
    size_t rsrpDynDescrSizeBytes = 0, rsrpDynDescrAlignBytes = 0;
    void*  rssiDynDescrCpu = nullptr;
    void*  rssiDynDescrGpu = nullptr;
    void*  rsrpDynDescrCpu = nullptr;
    void*  rsrpDynDescrGpu = nullptr;

    // UE group params
    cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrmsCpu = nullptr;
    cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrmsGpu = nullptr;
};

// End-to-end RSSI/RSRP with float types over small geometry; exercises FIRST/FULL/WITHOUT_FIRST paths
TEST_F(PuschRssiGTest, EndToEnd_RssiRsrp_SmallGeometry)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 16;
    const uint8_t  nRxAnt      = 2;
    const uint8_t  nLayers     = 1;
    const uint8_t  dmrsMaxLen  = 1;
    const uint16_t dmrsSymBmsk = 0x0004; // symbol 2
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk);
}

// Use dmrsMaxLen=2 and multiple DMRS symbols to exercise without-first logic
TEST_F(PuschRssiGTest, RssiRsrp_MultiDmrs_TwoSymbols)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 8;
    const uint8_t  nRxAnt      = 1;
    const uint8_t  nLayers     = 1;
    const uint8_t  dmrsMaxLen  = 2;
    const uint16_t dmrsSymBmsk = (1u << 2) | (1u << 10); // two DMRS symbols
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk);
}

// Exercise WITHOUT_FIRST behavior by choosing dmrsMaxLen < totalDmrsSel
TEST_F(PuschRssiGTest, RssiRsrp_WithoutFirstDmrs)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 8;
    const uint8_t  nRxAnt      = 1;
    const uint8_t  nLayers     = 1;
    const uint8_t  dmrsMaxLen  = 1;                      // first DMRS length
    const uint16_t dmrsSymBmsk = (1u << 2) | (1u << 10); // totalDmrsSel=2 > dmrsMaxLen
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk);
}

// Drive grid-x stride reduction using large PRB count and multiple RX antennas
TEST_F(PuschRssiGTest, Rssi_GridStrideReduction_TdiEnabled)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 273; // large to ensure grid-x > 1
    const uint8_t  nRxAnt      = 4;   // multiple antennas -> grid.z > 1
    const uint8_t  nLayers     = 1;
    const uint8_t  dmrsMaxLen  = 1;
    const uint16_t dmrsSymBmsk = (1u << 2) | (1u << 10); // two DMRS
    // enableTdi=true to cover dmrsIdx selection path in Ree accumulation (attachment)
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk, true);
}

// Exercise per-layer reduction with multiple layers (iLayer == jLayer and iLayer != jLayer)
TEST_F(PuschRssiGTest, Rsrp_Reduction_MultiLayer)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 16;
    const uint8_t  nRxAnt      = 2;
    const uint8_t  nLayers     = 3; // multiple layers to hit both branches
    const uint8_t  dmrsMaxLen  = 1;
    const uint16_t dmrsSymBmsk = (1u << 2) | (1u << 10);
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk, true);
}

// Exercise weighted-average-CFO path to launch rsrpMeasWithoutPostEqMeasKernel_v1
TEST_F(PuschRssiGTest, Rsrp_WeightedAverageCfo_WithoutPostEqKernel)
{
    const uint16_t nUeGrps     = 1;
    const uint16_t nPrb        = 16;
    const uint8_t  nRxAnt      = 2;
    const uint8_t  nLayers     = 2;
    const uint8_t  dmrsMaxLen  = 1;
    const uint16_t dmrsSymBmsk = (1u << 2) | (1u << 10); // exercise full-slot and without-first
    runRssiRsrp(nUeGrps, nPrb, nRxAnt, nLayers, dmrsMaxLen, dmrsSymBmsk, false, true);
}

// Cover rssiMeasKernelSelect half-precision branch via setup selection (no kernel launch)
TEST_F(PuschRssiGTest, RssiSelector_C16_R32_SelectsKernel_NoLaunch)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);

    // Minimal UE group params
    auto& prm          = drvdUeGrpPrmsCpu[0];
    prm.nUes           = 1;
    prm.nLayers        = 1;
    prm.nPrb           = nPrb;
    prm.nRxAnt         = 1;
    prm.nDmrsSyms      = 1;
    prm.dmrsMaxLen     = 1;
    prm.dmrsAddlnPos   = 0;
    prm.rssiSymPosBmsk = (1u << 2);
    prm.dmrsSymLoc[0]  = 2;

    // Types to hit the C_16F selection branch; outputs remain R_32F
    prm.tInfoDataRx.elemType   = CUPHY_C_16F;
    prm.tInfoRssi.elemType     = CUPHY_R_32F;
    prm.tInfoRssiEhq.elemType  = CUPHY_R_32F;
    prm.tInfoRssiFull.elemType = CUPHY_R_32F;

    // Prepare launch cfg container
    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;

    // Copy UE params CPU->GPU (not required for selection but consistent with API)
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    // Call setup to exercise kernel select branch (no kernel launch)
    ASSERT_EQ(CUPHY_STATUS_SUCCESS,
              cuphySetupPuschRxRssi(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    CUPHY_PUSCH_RSSI_EST_FIRST_DMRS,
                                    1, // enable async copy of descr
                                    rssiDynDescrCpu,
                                    rssiDynDescrGpu,
                                    &rssiLaunch,
                                    cuStream));

    // Validate a function pointer got selected
    ASSERT_EQ(1u, rssiLaunch.nCfgs);
    ASSERT_NE(nullptr, rssiLaunch.cfgs[0].kernelNodeParamsDriver.func);
}

// Cover early invalid-argument returns and symbLocBmsk==0 skip path in setup
TEST_F(PuschRssiGTest, RssiSetup_InvalidArgs_And_NoSymbolSkip)
{
    // Invalid arguments: null all pointers should return invalid argument
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRssi(rssiHndl,
                                    nullptr,
                                    nullptr,
                                    0,
                                    0,
                                    CUPHY_PUSCH_RSSI_EST_FIRST_DMRS,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    cuStream));

    // Allocate two UE groups to exercise symbLocBmsk==0 continue and nMaxRxAnt update
    const uint16_t nUeGrps = 2;
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);

    // UE group 0: bmsk=0 triggers continue path
    drvdUeGrpPrmsCpu[0].nUes                   = 1;
    drvdUeGrpPrmsCpu[0].nLayers                = 1;
    drvdUeGrpPrmsCpu[0].nPrb                   = nPrb;
    drvdUeGrpPrmsCpu[0].nRxAnt                 = 1;
    drvdUeGrpPrmsCpu[0].dmrsMaxLen             = 1;
    drvdUeGrpPrmsCpu[0].rssiSymPosBmsk         = 0; // skip
    drvdUeGrpPrmsCpu[0].tInfoDataRx.elemType   = CUPHY_C_32F;
    drvdUeGrpPrmsCpu[0].tInfoRssi.elemType     = CUPHY_R_32F;
    drvdUeGrpPrmsCpu[0].tInfoRssiFull.elemType = CUPHY_R_32F;

    // UE group 1: valid path
    drvdUeGrpPrmsCpu[1]                = drvdUeGrpPrmsCpu[0];
    drvdUeGrpPrmsCpu[1].nRxAnt         = 3; // update nMaxRxAnt branch
    drvdUeGrpPrmsCpu[1].rssiSymPosBmsk = (1u << 2);
    drvdUeGrpPrmsCpu[1].dmrsSymLoc[0]  = 2;

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;
    EXPECT_EQ(CUPHY_STATUS_SUCCESS,
              cuphySetupPuschRxRssi(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    CUPHY_PUSCH_RSSI_EST_FIRST_DMRS,
                                    1,
                                    rssiDynDescrCpu,
                                    rssiDynDescrGpu,
                                    &rssiLaunch,
                                    cuStream));
}

// Cover invalid dmrsSymbolIdx branch
TEST_F(PuschRssiGTest, RssiSetup_InvalidDmrsSymbolIdx)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);
    drvdUeGrpPrmsCpu[0].nUes                   = 1;
    drvdUeGrpPrmsCpu[0].nLayers                = 1;
    drvdUeGrpPrmsCpu[0].nPrb                   = nPrb;
    drvdUeGrpPrmsCpu[0].nRxAnt                 = 1;
    drvdUeGrpPrmsCpu[0].dmrsMaxLen             = 1;
    drvdUeGrpPrmsCpu[0].rssiSymPosBmsk         = (1u << 2);
    drvdUeGrpPrmsCpu[0].dmrsSymLoc[0]          = 2;
    drvdUeGrpPrmsCpu[0].tInfoDataRx.elemType   = CUPHY_C_32F;
    drvdUeGrpPrmsCpu[0].tInfoRssi.elemType     = CUPHY_R_32F;
    drvdUeGrpPrmsCpu[0].tInfoRssiFull.elemType = CUPHY_R_32F;
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;
    // Use an invalid dmrsSymbolIdx (e.g., 255) to hit early return branch
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRssi(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    255,
                                    1,
                                    rssiDynDescrCpu,
                                    rssiDynDescrGpu,
                                    &rssiLaunch,
                                    cuStream));
}

// Hit setupRssiMeas early invalid-argument by providing null derived UE-group pointers
TEST_F(PuschRssiGTest, RssiSetup_NullDerivedUeGrpPtrs)
{
    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRssi(rssiHndl,
                                    nullptr, // pDrvdUeGrpPrmsCpu
                                    nullptr, // pDrvdUeGrpPrmsGpu
                                    0,
                                    0,
                                    CUPHY_PUSCH_RSSI_EST_FIRST_DMRS,
                                    1,
                                    rssiDynDescrCpu, // valid
                                    rssiDynDescrGpu, // valid
                                    &rssiLaunch,     // valid
                                    cuStream));
}

// ----------------------------- RSRP setup coverage -----------------------------

// Cover optional CPU->GPU descriptor copy disabled path and multiple heterogeneous configs
TEST_F(PuschRssiGTest, RsrpSetup_NoAsyncCopy_MultipleHetConfigs)
{
    const uint16_t nUeGrps = 2; // also drive nMaxRxAnt update across groups
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);
    // UE0
    drvdUeGrpPrmsCpu[0].nUes           = 1;
    drvdUeGrpPrmsCpu[0].nLayers        = 1;
    drvdUeGrpPrmsCpu[0].nPrb           = nPrb;
    drvdUeGrpPrmsCpu[0].nRxAnt         = 1;
    drvdUeGrpPrmsCpu[0].dmrsMaxLen     = 1;
    drvdUeGrpPrmsCpu[0].dmrsSymLoc[0]  = 2;
    drvdUeGrpPrmsCpu[0].rssiSymPosBmsk = (1u << 2);
    // UE1 higher RxAnt to trigger nMaxRxAnt update
    drvdUeGrpPrmsCpu[1]        = drvdUeGrpPrmsCpu[0];
    drvdUeGrpPrmsCpu[1].nRxAnt = 3;
    // Types
    for(int i = 0; i < nUeGrps; ++i)
    {
        drvdUeGrpPrmsCpu[i].tInfoHEst.elemType = CUPHY_C_32F;
        drvdUeGrpPrmsCpu[i].tInfoRsrp.elemType = CUPHY_R_32F;
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 2; // multi-het cfg
    EXPECT_EQ(CUPHY_STATUS_SUCCESS,
              cuphySetupPuschRxRsrp(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS,
                                    0, // disable async copy path
                                    rsrpDynDescrCpu,
                                    rsrpDynDescrGpu,
                                    &rsrpLaunch,
                                    cuStream));
    // Sanity
    EXPECT_NE(nullptr, rsrpLaunch.cfgs[0].kernelNodeParamsDriver.func);
}

// Exercise RSRP setup guard coverage using existing helpers
TEST_F(PuschRssiGTest, RsrpSetup_InvalidArgs)
{
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRsrp(rssiHndl,
                                    nullptr,
                                    nullptr,
                                    0,
                                    0,
                                    CUPHY_PUSCH_RSRP_EST_FIRST_DMRS,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    cuStream));
}

TEST_F(PuschRssiGTest, RsrpSetup_NullDerivedUeGrpPtrs)
{
    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRsrp(rssiHndl,
                                    nullptr, // pDrvdUeGrpPrmsCpu
                                    nullptr, // pDrvdUeGrpPrmsGpu
                                    0,
                                    0,
                                    CUPHY_PUSCH_RSRP_EST_FIRST_DMRS,
                                    1,
                                    rsrpDynDescrCpu, // valid dynamic descriptor pointers
                                    rsrpDynDescrGpu,
                                    &rsrpLaunch,
                                    cuStream));
}

TEST_F(PuschRssiGTest, RsrpSetup_InvalidDmrsSymbolIdx)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);
    // Minimal valid UE group
    drvdUeGrpPrmsCpu[0].nUes               = 1;
    drvdUeGrpPrmsCpu[0].nLayers            = 1;
    drvdUeGrpPrmsCpu[0].nPrb               = nPrb;
    drvdUeGrpPrmsCpu[0].nRxAnt             = 1;
    drvdUeGrpPrmsCpu[0].dmrsMaxLen         = 1;
    drvdUeGrpPrmsCpu[0].dmrsSymLoc[0]      = 2;
    drvdUeGrpPrmsCpu[0].rssiSymPosBmsk     = (1u << 2);
    drvdUeGrpPrmsCpu[0].tInfoHEst.elemType = CUPHY_C_32F;
    drvdUeGrpPrmsCpu[0].tInfoRsrp.elemType = CUPHY_R_32F;
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;
    EXPECT_EQ(CUPHY_STATUS_INVALID_ARGUMENT,
              cuphySetupPuschRxRsrp(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    255, // invalid dmrs idx
                                    1,
                                    rsrpDynDescrCpu,
                                    rsrpDynDescrGpu,
                                    &rsrpLaunch,
                                    cuStream));
}

TEST_F(PuschRssiGTest, RsrpSetup_ValidPaths)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 8;
    allocUeGrp(nUeGrps);
    drvdUeGrpPrmsCpu[0].nUes               = 1;
    drvdUeGrpPrmsCpu[0].nLayers            = 1;
    drvdUeGrpPrmsCpu[0].nPrb               = nPrb;
    drvdUeGrpPrmsCpu[0].nRxAnt             = 1;
    drvdUeGrpPrmsCpu[0].dmrsMaxLen         = 1;
    drvdUeGrpPrmsCpu[0].dmrsSymLoc[0]      = 2;
    drvdUeGrpPrmsCpu[0].rssiSymPosBmsk     = (1u << 2);
    drvdUeGrpPrmsCpu[0].tInfoHEst.elemType = CUPHY_C_32F;
    drvdUeGrpPrmsCpu[0].tInfoRsrp.elemType = CUPHY_R_32F;
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));
    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;

    auto callSetup = [&](uint8_t dmrsIdx) {
        EXPECT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphySetupPuschRxRsrp(rssiHndl,
                                        drvdUeGrpPrmsCpu,
                                        drvdUeGrpPrmsGpu,
                                        nUeGrps,
                                        nPrb,
                                        dmrsIdx,
                                        1,
                                        rsrpDynDescrCpu,
                                        rsrpDynDescrGpu,
                                        &rsrpLaunch,
                                        cuStream));
        EXPECT_NE(nullptr, rsrpLaunch.cfgs[0].kernelNodeParamsDriver.func);
    };

    callSetup(CUPHY_PUSCH_RSRP_EST_FIRST_DMRS);
    callSetup(CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS);
    callSetup(CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS);
}

// Exercise selector when HEst type is not CUPHY_C_32F: no kernel should be bound
TEST_F(PuschRssiGTest, RsrpSelector_UnsupportedHEstType_NoKernel)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 4;
    allocUeGrp(nUeGrps);

    auto& prm              = drvdUeGrpPrmsCpu[0];
    prm.nUes               = 1;
    prm.nLayers            = 1;
    prm.nPrb               = nPrb;
    prm.nRxAnt             = 1;
    prm.dmrsMaxLen         = 1;
    prm.dmrsSymLoc[0]      = 2;
    prm.rssiSymPosBmsk     = (1u << 2);
    prm.tInfoHEst.elemType = CUPHY_C_16F; // force condition at 1644 to be false
    prm.tInfoRsrp.elemType = CUPHY_R_32F;

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;
    (void)cuphySetupPuschRxRsrp(rssiHndl,
                                drvdUeGrpPrmsCpu,
                                drvdUeGrpPrmsGpu,
                                nUeGrps,
                                nPrb,
                                CUPHY_PUSCH_RSRP_EST_FIRST_DMRS,
                                1,
                                rsrpDynDescrCpu,
                                rsrpDynDescrGpu,
                                &rsrpLaunch,
                                cuStream);
    // With unsupported hEstType selector should not bind a kernel
    EXPECT_EQ(nullptr, rsrpLaunch.cfgs[0].kernelNodeParamsDriver.func);
}

// Exercise selector with CUPHY_C_32F HEst but non-R_32F output: no kernel should be bound
TEST_F(PuschRssiGTest, RsrpSelector_C32_NonR32_NoKernel)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 4;
    allocUeGrp(nUeGrps);

    auto& prm              = drvdUeGrpPrmsCpu[0];
    prm.nUes               = 1;
    prm.nLayers            = 1;
    prm.nPrb               = nPrb;
    prm.nRxAnt             = 1;
    prm.dmrsMaxLen         = 1;
    prm.dmrsSymLoc[0]      = 2;
    prm.rssiSymPosBmsk     = (1u << 2);
    prm.tInfoHEst.elemType = CUPHY_C_32F; // outer if true
    prm.tInfoRsrp.elemType = CUPHY_R_16F; // inner if false (non-R_32F)

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;
    (void)cuphySetupPuschRxRsrp(rssiHndl,
                                drvdUeGrpPrmsCpu,
                                drvdUeGrpPrmsGpu,
                                nUeGrps,
                                nPrb,
                                CUPHY_PUSCH_RSRP_EST_FIRST_DMRS,
                                1,
                                rsrpDynDescrCpu,
                                rsrpDynDescrGpu,
                                &rsrpLaunch,
                                cuStream);
    // No kernel should be selected when rsrpType != R_32F
    EXPECT_EQ(nullptr, rsrpLaunch.cfgs[0].kernelNodeParamsDriver.func);
}

// Cover rsrpMeasLaunchGeo branch where nThrdBlks is recalculated based on layers/ants
TEST_F(PuschRssiGTest, RsrpLaunchGeometry_RecalculateThreadBlocks)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 1; // small PRB to make initial nThrdBlks tiny
    allocUeGrp(nUeGrps);

    auto& prm              = drvdUeGrpPrmsCpu[0];
    prm.nUes               = 1;
    prm.nLayers            = 1;
    prm.nPrb               = nPrb;
    prm.nRxAnt             = 8; // large RxAnt to increase RHS
    prm.dmrsMaxLen         = 1;
    prm.dmrsSymLoc[0]      = 2;
    prm.rssiSymPosBmsk     = (1u << 2);
    prm.tInfoHEst.elemType = CUPHY_C_32F; // required selector types
    prm.tInfoRsrp.elemType = CUPHY_R_32F;

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunch{};
    rsrpLaunch.nCfgs = 1;
    ASSERT_EQ(CUPHY_STATUS_SUCCESS,
              cuphySetupPuschRxRsrp(rssiHndl,
                                    drvdUeGrpPrmsCpu,
                                    drvdUeGrpPrmsGpu,
                                    nUeGrps,
                                    nPrb,
                                    CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS,
                                    1,
                                    rsrpDynDescrCpu,
                                    rsrpDynDescrGpu,
                                    &rsrpLaunch,
                                    cuStream));

    // For nPrb=1: initial nThrdBlks=ceil((1*12)/48)=1; THRD_GRP_TILE_SIZE=32; maxLayers=min(8, nRxAnt)=8
    // Since 1*32 < 8*8, branch should recalc: nThrdBlks=ceil((8*8)/48)=2
    EXPECT_EQ(2u, rsrpLaunch.cfgs[0].kernelNodeParamsDriver.gridDimX);
}

// ----------------------------- RSSI selector coverage -----------------------------

// Cover rssiMeasKernelSelect C_32F path
TEST_F(PuschRssiGTest, RssiSelector_C32_R32)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 4;
    allocUeGrp(nUeGrps);

    auto& prm          = drvdUeGrpPrmsCpu[0];
    prm.nUes           = 1;
    prm.nLayers        = 1;
    prm.nPrb           = nPrb;
    prm.nRxAnt         = 1;
    prm.dmrsMaxLen     = 1;
    prm.dmrsSymLoc[0]  = 2;
    prm.rssiSymPosBmsk = (1u << 2);
    // Types to drive selector
    prm.tInfoDataRx.elemType   = CUPHY_C_32F;
    prm.tInfoRssi.elemType     = CUPHY_R_32F;
    prm.tInfoRssiFull.elemType = CUPHY_R_32F;

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;
    auto call        = [&](uint8_t dmrsIdx) {
        rssiLaunch.nCfgs = 1;
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphySetupPuschRxRssi(rssiHndl,
                                        drvdUeGrpPrmsCpu,
                                        drvdUeGrpPrmsGpu,
                                        nUeGrps,
                                        nPrb,
                                        dmrsIdx,
                                        1,
                                        rssiDynDescrCpu,
                                        rssiDynDescrGpu,
                                        &rssiLaunch,
                                        cuStream));
        EXPECT_NE(nullptr, rssiLaunch.cfgs[0].kernelNodeParamsDriver.func);
    };
    call(CUPHY_PUSCH_RSSI_EST_FIRST_DMRS);
    call(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS);
    call(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS);
}

// Cover rssiMeasKernelSelect C_16F path
TEST_F(PuschRssiGTest, RssiSelector_C16_R32)
{
    const uint16_t nUeGrps = 1;
    const uint16_t nPrb    = 4;
    allocUeGrp(nUeGrps);

    auto& prm          = drvdUeGrpPrmsCpu[0];
    prm.nUes           = 1;
    prm.nLayers        = 1;
    prm.nPrb           = nPrb;
    prm.nRxAnt         = 1;
    prm.dmrsMaxLen     = 1;
    prm.dmrsSymLoc[0]  = 2;
    prm.rssiSymPosBmsk = (1u << 2);
    // Types to drive half-precision selector
    prm.tInfoDataRx.elemType   = CUPHY_C_16F;
    prm.tInfoRssi.elemType     = CUPHY_R_32F;
    prm.tInfoRssiFull.elemType = CUPHY_R_32F;

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(drvdUeGrpPrmsGpu, drvdUeGrpPrmsCpu, nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(cuStream));

    cuphyPuschRxRssiLaunchCfgs_t rssiLaunch{};
    rssiLaunch.nCfgs = 1;
    auto call        = [&](uint8_t dmrsIdx) {
        rssiLaunch.nCfgs = 1;
        ASSERT_EQ(CUPHY_STATUS_SUCCESS,
                  cuphySetupPuschRxRssi(rssiHndl,
                                        drvdUeGrpPrmsCpu,
                                        drvdUeGrpPrmsGpu,
                                        nUeGrps,
                                        nPrb,
                                        dmrsIdx,
                                        1,
                                        rssiDynDescrCpu,
                                        rssiDynDescrGpu,
                                        &rssiLaunch,
                                        cuStream));
        EXPECT_NE(nullptr, rssiLaunch.cfgs[0].kernelNodeParamsDriver.func);
    };
    call(CUPHY_PUSCH_RSSI_EST_FIRST_DMRS);
    call(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS);
    call(CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS);
}

int main(int argc, char* argv[])
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    return result;
}
