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

#include "cuphy.h"
#include "cuphy.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <array>
#include <cmath>


//-----------------------------------------------------------------------------
// Test configuration
//-----------------------------------------------------------------------------

struct PucchF0Config
{
    uint16_t nCells;
    uint16_t nUcis;
    uint16_t nRxAnt;
    uint8_t  enableUlRxBf;
    uint16_t nUplinkStreams;
    uint8_t  nSym;
    uint8_t  freqHopFlag;
    uint8_t  groupHopFlag;
    uint8_t  bitLenHarq;
    uint8_t  srFlag;
    float    extDTXthreshold;
};

//-----------------------------------------------------------------------------
// Test fixture (mirrors structure used in other tests)
//-----------------------------------------------------------------------------

class PucchF0ReceiverTest : public ::testing::Test {
protected:
    cuphy::stream cuStrmMain;

    static cuphyTensorPrm_t makeTensorPrm(cuphy::tensor_device& t)
    {
        cuphyTensorPrm_t prm{};
        prm.desc  = t.desc().handle();
        prm.pAddr = t.addr();
        return prm;
    }

    static void fillTensorConstant(cuphy::tensor_device& t, __half2 value, cudaStream_t strm)
    {
        const size_t         num_bytes = t.desc().get_size_in_bytes();
        const size_t         num_elems = num_bytes / sizeof(__half2);
        std::vector<__half2> host(num_elems, value);
        cudaError_t          err = cudaMemcpyAsync(t.addr(), host.data(), num_bytes, cudaMemcpyHostToDevice, strm);
        EXPECT_EQ(err, cudaSuccess) << "cudaMemcpyAsync tDataRx failed: " << cudaGetErrorString(err);
    }

    static void buildUciPrmVector(std::vector<cuphyPucchUciPrm_t>& vec, const PucchF0Config& cfg)
    {
        vec.resize(cfg.nUcis);
        for(uint16_t i = 0; i < cfg.nUcis; ++i)
        {
            auto& u                = vec[i];
            u.cellPrmDynIdx        = 0;
            u.cellPrmStatIdx       = 0;
            u.uciOutputIdx         = i;
            u.formatType           = 0;
            u.rnti                 = 1u + i;
            u.bwpStart             = 0;
            u.multiSlotTxIndicator = 0;
            u.pi2Bpsk              = 0;
            u.startPrb             = 0;
            u.prbSize              = 1;
            u.startSym             = 0;
            u.nSym                 = cfg.nSym;
            u.freqHopFlag          = cfg.freqHopFlag;
            u.secondHopPrb         = 1;
            u.groupHopFlag         = cfg.groupHopFlag;
            u.sequenceHopFlag      = 0;
            u.initialCyclicShift   = 0;
            u.timeDomainOccIdx     = 0;
            u.srFlag               = cfg.srFlag;
            u.bitLenSr             = cfg.srFlag ? 1 : 0;
            u.bitLenHarq           = cfg.bitLenHarq;
            u.bitLenCsiPart1       = 0;
            u.AddDmrsFlag          = 0;
            u.dataScramblingId     = 0;
            u.DmrsScramblingId     = 0;
            u.maxCodeRate          = 0;
            u.nBitsCsi2            = 0;
            u.rankBitOffset        = 0;
            u.nRanksBits           = 0;
            u.DTXthreshold         = cfg.extDTXthreshold;
            u.uciP1P2Crpd_t        = {};
            u.nUplinkStreams       = cfg.nUplinkStreams;
        }
    }

    static void buildCellPrmVector(std::vector<cuphyPucchCellPrm_t>& cells,
                                   const PucchF0Config&              cfg,
                                   uint16_t                          pucchHoppingId,
                                   uint16_t                          slotNum)
    {
        cells.resize(cfg.nCells);
        for(uint16_t c = 0; c < cfg.nCells; ++c)
        {
            auto& cp          = cells[c];
            cp.nRxAnt         = cfg.nRxAnt;
            cp.slotNum        = slotNum;
            cp.pucchHoppingId = pucchHoppingId;
            cp.mu             = 0;
            cp.tDataRx        = {};
        }
    }

    // Aligns with the example app methodology: perform sanity/consistency checks
    // rather than golden-value comparisons (we do not load reference datasets here).
    static bool validateOutputs(const std::vector<cuphyPucchF0F1UciOut_t>& out, const PucchF0Config& cfg)
    {
        if(out.size() != cfg.nUcis) return false;
        bool ok = true;
        for(uint16_t i = 0; i < cfg.nUcis; ++i)
        {
            const auto& o = out[i];

            // NumHarq must be feasible for requested HARQ length
            if(cfg.bitLenHarq == 0)
            {
                ok &= (o.NumHarq == 0);
            }
            else
            {
                ok &= (o.NumHarq <= cfg.bitLenHarq);
            }

            // Confidence and SR flags are booleans in practice
            ok &= (o.SRconfidenceLevel == 0 || o.SRconfidenceLevel == 1);
            ok &= (o.HarqconfidenceLevel == 0 || o.HarqconfidenceLevel == 1);
            ok &= (o.SRindication == 0 || o.SRindication == 1);

            // Payload bounds sanity: values are small integers
            // Allow up to 3 to cover 2-bit payloads; tolerant for internal representations
            ok &= (o.HarqValues[0] <= 3 && o.HarqValues[1] <= 3);

            // Power metrics should be finite
            ok &= std::isfinite(o.RSSI);
            ok &= std::isfinite(o.RSRP);
        }
        return ok;
    }

    bool runScenario(const PucchF0Config& cfg)
    {
        try
        {
            const int nSc  = 12;
            const int nSym = std::max<int>(cfg.nSym, 2);
            const int nAnt = (cfg.enableUlRxBf ? cfg.nUplinkStreams : cfg.nRxAnt);

            cuphy::tensor_device tDataRx(CUPHY_C_16F, nSc, nSym, nAnt, cuphy::tensor_flags::align_tight);
            fillTensorConstant(tDataRx, __half2{__float2half(0.125f), __float2half(-0.0625f)}, cuStrmMain.handle());

            std::vector<cuphyTensorPrm_t> tPrmDataRxVec(cfg.nCells);
            tPrmDataRxVec[0] = makeTensorPrm(tDataRx);

            std::vector<cuphyPucchUciPrm_t> uciVec;
            buildUciPrmVector(uciVec, cfg);

            std::vector<cuphyPucchCellPrm_t> cellPrms;
            buildCellPrmVector(cellPrms, cfg, 0, 0);

            cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> outGpu(cfg.nUcis);

            size_t        dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
            cuphyStatus_t s = cuphyPucchF0RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
            EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
            if(CUPHY_STATUS_SUCCESS != s) return false;
            dynDescrSizeBytes += cfg.nCells * sizeof(cuphyPucchCellPrm_t);

            cuphy::buffer<uint8_t, cuphy::pinned_alloc> descrCpu(dynDescrSizeBytes);
            cuphy::buffer<uint8_t, cuphy::device_alloc> descrGpu(dynDescrSizeBytes);

            cuphyPucchF0RxHndl_t hndl{};
            s = cuphyCreatePucchF0Rx(&hndl, cuStrmMain.handle());
            EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
            if(CUPHY_STATUS_SUCCESS != s) return false;

            cuphyPucchF0RxLaunchCfg_t launchCfg{};
            const bool                enableCpuToGpuDescrAsyncCpy = false;
            s                                                     = cuphySetupPucchF0Rx(hndl,
                                    tPrmDataRxVec.data(),
                                    outGpu.addr(),
                                    cfg.nCells,
                                    cfg.nUcis,
                                    cfg.enableUlRxBf,
                                    uciVec.data(),
                                    cellPrms.data(),
                                    enableCpuToGpuDescrAsyncCpy,
                                    descrCpu.addr(),
                                    descrGpu.addr(),
                                    &launchCfg,
                                    cuStrmMain.handle());
            EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
            if(CUPHY_STATUS_SUCCESS != s)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            const size_t cellsBytes  = static_cast<size_t>(cfg.nCells) * sizeof(cuphyPucchCellPrm_t);
            const size_t cellsOffset = dynDescrSizeBytes - cellsBytes;
            uint8_t*     cpuBase     = descrCpu.addr();
            uint8_t*     gpuBase     = descrGpu.addr();
            std::memcpy(cpuBase + cellsOffset, cellPrms.data(), cellsBytes);
            void**       hdrPtrs     = reinterpret_cast<void**>(cpuBase);
            const size_t numPtrSlots = cellsOffset / sizeof(void*);
            void*        cpuCellPtr  = reinterpret_cast<void*>(cellPrms.data());
            void*        gpuCellPtr  = reinterpret_cast<void*>(gpuBase + cellsOffset);
            for(size_t i = 0; i < numPtrSlots; ++i)
            {
                if(hdrPtrs[i] == cpuCellPtr)
                {
                    hdrPtrs[i] = gpuCellPtr;
                    break;
                }
            }

            cudaError_t err = cudaMemcpyAsync(descrGpu.addr(), descrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            EXPECT_EQ(err, cudaSuccess) << "Descriptor H2D failed: " << cudaGetErrorString(err);
            if(err != cudaSuccess)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            err = cudaStreamSynchronize(cuStrmMain.handle());
            EXPECT_EQ(err, cudaSuccess);
            if(err != cudaSuccess)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
            CUresult                       e = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes, static_cast<CUstream>(cuStrmMain.handle()), k.kernelParams, k.extra);
            EXPECT_EQ(e, CUDA_SUCCESS);
            if(e != CUDA_SUCCESS)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            err = cudaStreamSynchronize(cuStrmMain.handle());
            EXPECT_EQ(err, cudaSuccess);
            if(err != cudaSuccess)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            std::vector<cuphyPucchF0F1UciOut_t> out(cfg.nUcis);
            err = cudaMemcpyAsync(out.data(), outGpu.addr(), sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cudaMemcpyDeviceToHost, cuStrmMain.handle());
            EXPECT_EQ(err, cudaSuccess);
            if(err != cudaSuccess)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }
            err = cudaStreamSynchronize(cuStrmMain.handle());
            EXPECT_EQ(err, cudaSuccess);
            if(err != cudaSuccess)
            {
                cuphyDestroyPucchF0Rx(hndl);
                return false;
            }

            (void)cuphyDestroyPucchF0Rx(hndl);

            return validateOutputs(out, cfg);
        }
        catch(const std::exception& e)
        {
            ADD_FAILURE() << "Exception: " << e.what();
            return false;
        }
    }
};

//-----------------------------------------------------------------------------
// Tests

TEST_F(PucchF0ReceiverTest, BasicSingleUci_NoHop_1Ant)
{
    PucchF0Config cfg{1, 1, 1, 0, 1, 1, 0, 0, 1, 0, CUPHY_DEFAULT_EXT_DTX_THRESHOLD};
    EXPECT_TRUE(runScenario(cfg));
}

TEST_F(PucchF0ReceiverTest, SROnly_NoHarq)
{
    PucchF0Config cfg{1, 2, 2, 0, 2, 1, 0, 0, 0, 1, CUPHY_DEFAULT_EXT_DTX_THRESHOLD};
    EXPECT_TRUE(runScenario(cfg));
}

// Exercise load/store branches across antenna counts and hop flags
TEST_F(PucchF0ReceiverTest, MultiAnt_LoadStoreBranches)
{
    // Iterate over antenna counts and flags to exercise kernel load/store branches
    const std::array<uint16_t, 5> ants{1, 2, 4, 8, 16};
    for(uint16_t nAnt : ants)
    {
        for(uint8_t hop = 0; hop < 2; ++hop)
        {
            PucchF0Config cfg{1, 3, nAnt, 0, (uint16_t)nAnt, (uint8_t)((nAnt == 16) ? 1 : 2), hop, hop, 2, 1, CUPHY_DEFAULT_EXT_DTX_THRESHOLD};
            EXPECT_TRUE(runScenario(cfg));
        }
    }
}

TEST_F(PucchF0ReceiverTest, EnableUlRxBf_Streams)
{
    PucchF0Config cfg{1, 2, 16, 1, 4, 2, 1, 0, 1, 1, CUPHY_DEFAULT_EXT_DTX_THRESHOLD};
    EXPECT_TRUE(runScenario(cfg));
}

// Force DTX path with a very high threshold; validates early DTX handling
TEST_F(PucchF0ReceiverTest, ForcedDTXPath_DTXEarlyExit)
{
    // Use large external DTX threshold to force DTX branch
    PucchF0Config cfg{1, 4, 4, 0, 4, 2, 1, 1, 2, 0, 1e6f};
    EXPECT_TRUE(runScenario(cfg));
}

// Cover bitLenHarq == 1 with SR present path
TEST_F(PucchF0ReceiverTest, OneBitHarq_WithSr)
{
    // Use low external DTX threshold to encourage detection
    PucchF0Config cfg{1, 2, 4, 0, 4, 2, 0, 0, 1, 1, 0.01f};
    EXPECT_TRUE(runScenario(cfg));
}

// Cover bitLenHarq == 2 with no SR path
TEST_F(PucchF0ReceiverTest, TwoBitHarq_NoSr)
{
    // Use low external DTX threshold to encourage detection
    PucchF0Config cfg{1, 2, 4, 0, 4, 2, 0, 0, 2, 0, 0.01f};
    EXPECT_TRUE(runScenario(cfg));
}

// Explicitly cover second-symbol load for numRxAnt==16 (kernel switch-case 16 in second-symbol path)
TEST_F(PucchF0ReceiverTest, SecondSymbolLoad_16Ant_CoversCase16)
{
    // Force:
    // - numRxAnt == 16 (enableUlRxBf=0 so kernel uses cellPrm nRxAnt)
    // - second symbol path enabled (nSym > 1, plus freqHopFlag=1 for completeness)
    PucchF0Config cfg{1, 1, 16, 0, 16, 2, 1, 0, 1, 0, 0.01f};
    EXPECT_TRUE(runScenario(cfg));
}

// Cover the host-side overflow guard when too many UCIs land in one group.
// This targets the `if(CUPHY_PUCCH_F0_MAX_UCI_PER_GRP <= nUciInGrp) { ... break; }` path in setup().
TEST_F(PucchF0ReceiverTest, TooManyUcisInOneGroup_DropsAdditionalUcis)
{
    // All UCIs are constructed with identical (cellIdx, startPrb+bwpStart, startSym),
    // so they bin into the same group and will exceed the per-group max.
    constexpr uint16_t kMaxPerGrp = CUPHY_PUCCH_F0_MAX_UCI_PER_GRP;
    PucchF0Config      cfg{1, static_cast<uint16_t>(kMaxPerGrp + 1), 2, 0, 2, 2, 0, 0, 1, 0, 0.01f};

    const int nSc  = 12;
    const int nSym = 2;
    const int nAnt = 2;

    cuphy::tensor_device tDataRx(CUPHY_C_16F, nSc, nSym, nAnt, cuphy::tensor_flags::align_tight);
    fillTensorConstant(tDataRx, __half2{__float2half(0.03125f), __float2half(0.0f)}, cuStrmMain.handle());

    std::vector<cuphyTensorPrm_t> tPrmDataRxVec(cfg.nCells);
    tPrmDataRxVec[0] = makeTensorPrm(tDataRx);

    std::vector<cuphyPucchUciPrm_t> uciVec;
    buildUciPrmVector(uciVec, cfg);

    std::vector<cuphyPucchCellPrm_t> cellPrms;
    buildCellPrmVector(cellPrms, cfg, 0, 0);

    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> outGpu(cfg.nUcis);
    cudaError_t err = cudaMemsetAsync(outGpu.addr(), 0, sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess) << "cudaMemsetAsync outGpu failed: " << cudaGetErrorString(err);

    size_t        dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    cuphyStatus_t s = cuphyPucchF0RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);
    dynDescrSizeBytes += cfg.nCells * sizeof(cuphyPucchCellPrm_t);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> descrCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> descrGpu(dynDescrSizeBytes);

    cuphyPucchF0RxHndl_t hndl{};
    s = cuphyCreatePucchF0Rx(&hndl, cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    cuphyPucchF0RxLaunchCfg_t launchCfg{};
    const bool                enableCpuToGpuDescrAsyncCpy = false;
    s                                                     = cuphySetupPucchF0Rx(hndl,
                                tPrmDataRxVec.data(),
                                outGpu.addr(),
                                cfg.nCells,
                                cfg.nUcis,
                                cfg.enableUlRxBf,
                                uciVec.data(),
                                cellPrms.data(),
                                enableCpuToGpuDescrAsyncCpy,
                                descrCpu.addr(),
                                descrGpu.addr(),
                                &launchCfg,
                                cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    // Patch cellPrms pointers inside descriptor, then H2D copy (mirrors runScenario()).
    const size_t cellsBytes  = static_cast<size_t>(cfg.nCells) * sizeof(cuphyPucchCellPrm_t);
    const size_t cellsOffset = dynDescrSizeBytes - cellsBytes;
    uint8_t*     cpuBase     = descrCpu.addr();
    uint8_t*     gpuBase     = descrGpu.addr();
    std::memcpy(cpuBase + cellsOffset, cellPrms.data(), cellsBytes);
    void**       hdrPtrs     = reinterpret_cast<void**>(cpuBase);
    const size_t numPtrSlots = cellsOffset / sizeof(void*);
    void*        cpuCellPtr  = reinterpret_cast<void*>(cellPrms.data());
    void*        gpuCellPtr  = reinterpret_cast<void*>(gpuBase + cellsOffset);
    for(size_t i = 0; i < numPtrSlots; ++i)
    {
        if(hdrPtrs[i] == cpuCellPtr)
        {
            hdrPtrs[i] = gpuCellPtr;
            break;
        }
    }

    err = cudaMemcpyAsync(descrGpu.addr(), descrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess) << "Descriptor H2D failed: " << cudaGetErrorString(err);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult                       e = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes, static_cast<CUstream>(cuStrmMain.handle()), k.kernelParams, k.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    std::vector<cuphyPucchF0F1UciOut_t> out(cfg.nUcis);
    err = cudaMemcpyAsync(out.data(), outGpu.addr(), sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cudaMemcpyDeviceToHost, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    (void)cuphyDestroyPucchF0Rx(hndl);

    // The overflow UCI (index kMaxPerGrp) should have been dropped by setup(), so its output remains zeroed.
    ASSERT_EQ(out.size(), static_cast<size_t>(cfg.nUcis));
    EXPECT_EQ(out[kMaxPerGrp].NumHarq, 0);
    EXPECT_EQ(out[kMaxPerGrp].SRconfidenceLevel, 0);
    EXPECT_EQ(out[kMaxPerGrp].HarqconfidenceLevel, 0);
    EXPECT_EQ(out[kMaxPerGrp].SRindication, 0);
    EXPECT_EQ(out[kMaxPerGrp].RSSI, 0.0f);
    EXPECT_EQ(out[kMaxPerGrp].RSRP, 0.0f);
}

// Cover the host-side overflow guard when too many distinct groups are created.
// This targets the `if(CUPHY_PUCCH_F0_MAX_GRPS <= nUciGrps) { ... continue; }` path in setup().
TEST_F(PucchF0ReceiverTest, TooManyUciGroups_DropsAdditionalGroups)
{
    constexpr uint16_t kMaxGrps = CUPHY_PUCCH_F0_MAX_GRPS;
    PucchF0Config      cfg{1, static_cast<uint16_t>(kMaxGrps + 1), 2, 0, 2, 1, 0, 0, 1, 0, 0.01f};

    const int nSc  = 12;
    const int nSym = 1;
    const int nAnt = 2;

    cuphy::tensor_device tDataRx(CUPHY_C_16F, nSc, nSym, nAnt, cuphy::tensor_flags::align_tight);
    fillTensorConstant(tDataRx, __half2{__float2half(0.03125f), __float2half(0.0f)}, cuStrmMain.handle());

    std::vector<cuphyTensorPrm_t> tPrmDataRxVec(cfg.nCells);
    tPrmDataRxVec[0] = makeTensorPrm(tDataRx);

    std::vector<cuphyPucchUciPrm_t> uciVec;
    buildUciPrmVector(uciVec, cfg);

    // Make each UCI fall into a distinct group by varying (startPrb, startSym).
    // Keep nSym==1 and hopping disabled so all startSym values [0..13] are legal.
    for(uint16_t i = 0; i < cfg.nUcis; ++i)
    {
        auto& u       = uciVec[i];
        u.bwpStart    = 0;
        u.startPrb    = static_cast<uint16_t>(i / 14);
        u.startSym    = static_cast<uint8_t>(i % 14);
        u.nSym        = 1;
        u.freqHopFlag = 0;
        u.groupHopFlag = 0;
        u.secondHopPrb = u.startPrb;
    }

    std::vector<cuphyPucchCellPrm_t> cellPrms;
    buildCellPrmVector(cellPrms, cfg, 0, 0);

    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> outGpu(cfg.nUcis);
    cudaError_t err = cudaMemsetAsync(outGpu.addr(), 0, sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess) << "cudaMemsetAsync outGpu failed: " << cudaGetErrorString(err);

    size_t        dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    cuphyStatus_t s = cuphyPucchF0RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);
    dynDescrSizeBytes += cfg.nCells * sizeof(cuphyPucchCellPrm_t);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> descrCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> descrGpu(dynDescrSizeBytes);

    cuphyPucchF0RxHndl_t hndl{};
    s = cuphyCreatePucchF0Rx(&hndl, cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    cuphyPucchF0RxLaunchCfg_t launchCfg{};
    const bool                enableCpuToGpuDescrAsyncCpy = false;
    s                                                     = cuphySetupPucchF0Rx(hndl,
                                tPrmDataRxVec.data(),
                                outGpu.addr(),
                                cfg.nCells,
                                cfg.nUcis,
                                cfg.enableUlRxBf,
                                uciVec.data(),
                                cellPrms.data(),
                                enableCpuToGpuDescrAsyncCpy,
                                descrCpu.addr(),
                                descrGpu.addr(),
                                &launchCfg,
                                cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    // Patch cellPrms pointers inside descriptor, then H2D copy (mirrors runScenario()).
    const size_t cellsBytes  = static_cast<size_t>(cfg.nCells) * sizeof(cuphyPucchCellPrm_t);
    const size_t cellsOffset = dynDescrSizeBytes - cellsBytes;
    uint8_t*     cpuBase     = descrCpu.addr();
    uint8_t*     gpuBase     = descrGpu.addr();
    std::memcpy(cpuBase + cellsOffset, cellPrms.data(), cellsBytes);
    void**       hdrPtrs     = reinterpret_cast<void**>(cpuBase);
    const size_t numPtrSlots = cellsOffset / sizeof(void*);
    void*        cpuCellPtr  = reinterpret_cast<void*>(cellPrms.data());
    void*        gpuCellPtr  = reinterpret_cast<void*>(gpuBase + cellsOffset);
    for(size_t i = 0; i < numPtrSlots; ++i)
    {
        if(hdrPtrs[i] == cpuCellPtr)
        {
            hdrPtrs[i] = gpuCellPtr;
            break;
        }
    }

    err = cudaMemcpyAsync(descrGpu.addr(), descrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess) << "Descriptor H2D failed: " << cudaGetErrorString(err);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult                       e = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes, static_cast<CUstream>(cuStrmMain.handle()), k.kernelParams, k.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    std::vector<cuphyPucchF0F1UciOut_t> out(cfg.nUcis);
    err = cudaMemcpyAsync(out.data(), outGpu.addr(), sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cudaMemcpyDeviceToHost, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    (void)cuphyDestroyPucchF0Rx(hndl);

    // The overflow group (UCI index kMaxGrps) should have been dropped by setup(), so its output remains zeroed.
    ASSERT_EQ(out.size(), static_cast<size_t>(cfg.nUcis));
    EXPECT_EQ(out[kMaxGrps].NumHarq, 0);
    EXPECT_EQ(out[kMaxGrps].SRconfidenceLevel, 0);
    EXPECT_EQ(out[kMaxGrps].HarqconfidenceLevel, 0);
    EXPECT_EQ(out[kMaxGrps].SRindication, 0);
    EXPECT_EQ(out[kMaxGrps].RSSI, 0.0f);
    EXPECT_EQ(out[kMaxGrps].RSRP, 0.0f);
}

// Cover bitLenHarq == 0 detection path (sets SR=1)
TEST_F(PucchF0ReceiverTest, ZeroBitHarq_SetsSr)
{
    // Encourage detection so that the branch executes and SR is set to 1
    // Note: for bitLenHarq==0, the kernel computes max_corr only when srFlag==1
    PucchF0Config cfg{1, 1, 2, 0, 2, 1, 0, 0, 0, 1, 0.01f};
    EXPECT_TRUE(runScenario(cfg));
}

// Cover bitLenHarq == 1 with no SR path (pucch_payload = index)
TEST_F(PucchF0ReceiverTest, OneBitHarq_NoSr)
{
    // Encourage detection to reach the payload assignment path
    PucchF0Config cfg{1, 2, 2, 0, 2, 2, 0, 0, 1, 0, 0.01f};
    EXPECT_TRUE(runScenario(cfg));
}

// Explicitly exercise the kernel bounds-check early-return by overlaunching an extra CTA
TEST_F(PucchF0ReceiverTest, BoundsCheck_ExtraCTA_ReturnPath)
{
    PucchF0Config cfg{1, 1, 1, 0, 1, 1, 0, 0, 1, 0, CUPHY_DEFAULT_EXT_DTX_THRESHOLD};

    const int            nSc  = 12;
    const int            nSym = 2;
    const int            nAnt = 1;
    cuphy::tensor_device tDataRx(CUPHY_C_16F, nSc, nSym, nAnt, cuphy::tensor_flags::align_tight);
    fillTensorConstant(tDataRx, __half2{__float2half(0.03125f), __float2half(0.0f)}, cuStrmMain.handle());

    std::vector<cuphyTensorPrm_t> tPrmDataRxVec(cfg.nCells);
    tPrmDataRxVec[0] = makeTensorPrm(tDataRx);

    std::vector<cuphyPucchUciPrm_t> uciVec;
    buildUciPrmVector(uciVec, cfg);

    std::vector<cuphyPucchCellPrm_t> cellPrms;
    buildCellPrmVector(cellPrms, cfg, 0, 0);

    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> outGpu(cfg.nUcis);

    size_t        dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    cuphyStatus_t s = cuphyPucchF0RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);
    dynDescrSizeBytes += cfg.nCells * sizeof(cuphyPucchCellPrm_t);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> descrCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> descrGpu(dynDescrSizeBytes);

    cuphyPucchF0RxHndl_t hndl{};
    s = cuphyCreatePucchF0Rx(&hndl, cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    cuphyPucchF0RxLaunchCfg_t launchCfg{};
    const bool                enableCpuToGpuDescrAsyncCpy = false;
    s                                                     = cuphySetupPucchF0Rx(hndl,
                            tPrmDataRxVec.data(),
                            outGpu.addr(),
                            cfg.nCells,
                            cfg.nUcis,
                            cfg.enableUlRxBf,
                            uciVec.data(),
                            cellPrms.data(),
                            enableCpuToGpuDescrAsyncCpy,
                            descrCpu.addr(),
                            descrGpu.addr(),
                            &launchCfg,
                            cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    const size_t cellsBytes  = static_cast<size_t>(cfg.nCells) * sizeof(cuphyPucchCellPrm_t);
    const size_t cellsOffset = dynDescrSizeBytes - cellsBytes;
    uint8_t*     cpuBase     = descrCpu.addr();
    uint8_t*     gpuBase     = descrGpu.addr();
    std::memcpy(cpuBase + cellsOffset, cellPrms.data(), cellsBytes);
    void**       hdrPtrs     = reinterpret_cast<void**>(cpuBase);
    const size_t numPtrSlots = cellsOffset / sizeof(void*);
    void*        cpuCellPtr  = reinterpret_cast<void*>(cellPrms.data());
    void*        gpuCellPtr  = reinterpret_cast<void*>(gpuBase + cellsOffset);
    for(size_t i = 0; i < numPtrSlots; ++i)
    {
        if(hdrPtrs[i] == cpuCellPtr)
        {
            hdrPtrs[i] = gpuCellPtr;
            break;
        }
    }

    cudaError_t err = cudaMemcpyAsync(descrGpu.addr(), descrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    // Overlaunch one extra CTA in X to trigger early return in that CTA
    launchCfg.kernelNodeParamsDriver.gridDimX += 1;

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult                       e = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes, static_cast<CUstream>(cuStrmMain.handle()), k.kernelParams, k.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    std::vector<cuphyPucchF0F1UciOut_t> out(cfg.nUcis);
    err = cudaMemcpyAsync(out.data(), outGpu.addr(), sizeof(cuphyPucchF0F1UciOut_t) * cfg.nUcis, cudaMemcpyDeviceToHost, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    (void)cuphyDestroyPucchF0Rx(hndl);

    // Per-UE validation because this test mixes bitLenHarq across UCIs
    ASSERT_EQ(out.size(), static_cast<size_t>(cfg.nUcis));
    const std::array<uint8_t, 3> expectedNumHarq{1, 2, 0};
    for(size_t i = 0; i < out.size(); ++i)
    {
        const auto& o = out[i];
        EXPECT_EQ(o.NumHarq, expectedNumHarq[i]);
        EXPECT_TRUE(o.SRconfidenceLevel == 0 || o.SRconfidenceLevel == 1);
        EXPECT_TRUE(o.HarqconfidenceLevel == 0 || o.HarqconfidenceLevel == 1);
        EXPECT_TRUE(o.SRindication == 0 || o.SRindication == 1);
        if(expectedNumHarq[i] > 0)
        {
            EXPECT_LE(o.HarqValues[0], 2);
            EXPECT_LE(o.HarqValues[1], 2);
        }
        EXPECT_TRUE(std::isfinite(o.RSSI));
        EXPECT_TRUE(std::isfinite(o.RSRP));
    }
}

// Group hopping only; single symbol; exercises u[1]=u[0] pathway where applicable
TEST_F(PucchF0ReceiverTest, GroupHop_Only_SingleSym)
{
    PucchF0Config cfg{1, 1, 2, 0, 2, 1, 0, 1, 1, 1, 0.02f};
    EXPECT_TRUE(runScenario(cfg));
}

// Drive first-symbol switch default via invalid numRxAnt; covers early return path.
// Exits before the second-symbol loop, so only the first default is exercised.
TEST_F(PucchF0ReceiverTest, InvalidNumRxAnt_DefaultReturn_FirstSymbol)
{
    // nRxAnt=3 triggers the default branch. Use two symbols to prove stability even if second path is gated.
    const uint16_t nRxAntInvalid = 3;
    PucchF0Config  cfg{1, 1, nRxAntInvalid, 0, nRxAntInvalid, 2, 0, 0, 1, 0, 0.02f};

    const int            nSc  = 12;
    const int            nSym = 2;
    const int            nAnt = nRxAntInvalid;
    cuphy::tensor_device tDataRx(CUPHY_C_16F, nSc, nSym, nAnt, cuphy::tensor_flags::align_tight);
    fillTensorConstant(tDataRx, __half2{__float2half(0.015625f), __float2half(0.0f)}, cuStrmMain.handle());

    std::vector<cuphyTensorPrm_t> tPrmDataRxVec(cfg.nCells);
    tPrmDataRxVec[0] = makeTensorPrm(tDataRx);

    std::vector<cuphyPucchUciPrm_t> uciVec;
    buildUciPrmVector(uciVec, cfg);
    std::vector<cuphyPucchCellPrm_t> cellPrms;
    buildCellPrmVector(cellPrms, cfg, 0, 0);

    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> outGpu(cfg.nUcis);

    size_t        dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    cuphyStatus_t s = cuphyPucchF0RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);
    dynDescrSizeBytes += cfg.nCells * sizeof(cuphyPucchCellPrm_t);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> descrCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> descrGpu(dynDescrSizeBytes);

    cuphyPucchF0RxHndl_t hndl{};
    s = cuphyCreatePucchF0Rx(&hndl, cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    cuphyPucchF0RxLaunchCfg_t launchCfg{};
    const bool                enableCpuToGpuDescrAsyncCpy = false;
    s                                                     = cuphySetupPucchF0Rx(hndl,
                            tPrmDataRxVec.data(),
                            outGpu.addr(),
                            cfg.nCells,
                            cfg.nUcis,
                            cfg.enableUlRxBf,
                            uciVec.data(),
                            cellPrms.data(),
                            enableCpuToGpuDescrAsyncCpy,
                            descrCpu.addr(),
                            descrGpu.addr(),
                            &launchCfg,
                            cuStrmMain.handle());
    ASSERT_EQ(s, CUPHY_STATUS_SUCCESS);

    const size_t cellsBytes  = static_cast<size_t>(cfg.nCells) * sizeof(cuphyPucchCellPrm_t);
    const size_t cellsOffset = dynDescrSizeBytes - cellsBytes;
    uint8_t*     cpuBase     = descrCpu.addr();
    uint8_t*     gpuBase     = descrGpu.addr();
    std::memcpy(cpuBase + cellsOffset, cellPrms.data(), cellsBytes);
    void**       hdrPtrs     = reinterpret_cast<void**>(cpuBase);
    const size_t numPtrSlots = cellsOffset / sizeof(void*);
    void*        cpuCellPtr  = reinterpret_cast<void*>(cellPrms.data());
    void*        gpuCellPtr  = reinterpret_cast<void*>(gpuBase + cellsOffset);
    for(size_t i = 0; i < numPtrSlots; ++i)
    {
        if(hdrPtrs[i] == cpuCellPtr)
        {
            hdrPtrs[i] = gpuCellPtr;
            break;
        }
    }

    cudaError_t err = cudaMemcpyAsync(descrGpu.addr(), descrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult                       e = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ, k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes, static_cast<CUstream>(cuStrmMain.handle()), k.kernelParams, k.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    err = cudaStreamSynchronize(cuStrmMain.handle());
    ASSERT_EQ(err, cudaSuccess);

    (void)cuphyDestroyPucchF0Rx(hndl);
}

//-----------------------------------------------------------------------------
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
