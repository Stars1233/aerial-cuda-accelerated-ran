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
#include <memory>
#include <cstring>
#include <cmath>

#include "cuda_fp16.h"
#include "cuphy.h"
#include "cuphy.hpp"
#include "pucch_F1_receiver/pucch_F1_receiver.hpp"

namespace
{

using cuphy::linear_alloc;
using cuphy::device_alloc;

struct F1TestHarness
{
    cuphy::stream                                    cuStrmMain;
    std::unique_ptr<linear_alloc<128, device_alloc>> pLinearAlloc;

    // Device buffers
    cuphyPucchF0F1UciOut_t* dF1Out{nullptr};
    cuphyPucchCellPrm_t*    dCellPrms{nullptr};

    // Tensor for input data (device)
    cuphy::tensor_device tDataRx;

    F1TestHarness()
    {
        pLinearAlloc = std::make_unique<linear_alloc<128, device_alloc>>(50 * 1024 * 1024);
    }

    void allocDataRxTensor(int nScTotal, int nOfdmSyms, int nRxAnt, bool fp16 = true)
    {
        tDataRx = cuphy::tensor_device(fp16 ? CUPHY_C_16F : CUPHY_C_32F,
                                       nScTotal,
                                       nOfdmSyms,
                                       nRxAnt,
                                       cuphy::tensor_flags::align_tight);
    }

    void fillDataRxTensorHost(int nScTotal, int nOfdmSyms, int nRxAnt, int startCrb, int secondHopCrb, int nSym, bool freqHop)
    {
        const int            nTonesPerPrb = CUPHY_N_TONES_PER_PRB;
        const size_t         elems        = static_cast<size_t>(nScTotal) * nOfdmSyms * nRxAnt;
        std::vector<__half2> h(elems);
        auto                 idx = [nOfdmSyms, nRxAnt](int sc, int sym, int ant) { return static_cast<size_t>(sc) * nOfdmSyms * nRxAnt + static_cast<size_t>(sym) * nRxAnt + ant; };
        // Deterministic pattern: DMRS (even offset from startSym) stronger than data
        for(int sc = 0; sc < nScTotal; ++sc)
        {
            for(int sym = 0; sym < nOfdmSyms; ++sym)
            {
                for(int ant = 0; ant < nRxAnt; ++ant)
                {
                    float a        = 0.01f;
                    bool  inFirst  = (sc >= startCrb * nTonesPerPrb) && (sc < (startCrb + 1) * nTonesPerPrb);
                    bool  inSecond = freqHop && (sc >= secondHopCrb * nTonesPerPrb) && (sc < (secondHopCrb + 1) * nTonesPerPrb);
                    if(inFirst || inSecond)
                    {
                        if(sym < nSym)
                        {
                            if(((sym) & 1) == 0)
                                a = 0.50f;
                            else
                                a = 0.25f;
                        }
                    }
                    __half2 v;
                    v.x                  = __float2half(a);
                    v.y                  = __float2half(0.0f);
                    h[idx(sc, sym, ant)] = v;
                }
            }
        }
        cudaError_t err = cudaMemcpyAsync(tDataRx.addr(), h.data(), h.size() * sizeof(__half2), cudaMemcpyHostToDevice, cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
    }

    void allocOutputs(int nUcis)
    {
        dF1Out = static_cast<cuphyPucchF0F1UciOut_t*>(pLinearAlloc->alloc(nUcis * sizeof(cuphyPucchF0F1UciOut_t)));
    }

    void allocDeviceCellPrms(int nCells)
    {
        dCellPrms = static_cast<cuphyPucchCellPrm_t*>(pLinearAlloc->alloc(nCells * sizeof(cuphyPucchCellPrm_t)));
    }

    static cuphyPucchUciPrm_t makeUciPrm(uint16_t cellIdx,
                                         uint16_t outIdx,
                                         uint16_t bwpStart,
                                         uint16_t startPrb,
                                         uint8_t  nSym,
                                         uint8_t  startSym,
                                         bool     freqHop,
                                         uint16_t secondHopPrb,
                                         bool     groupHop,
                                         uint8_t  timeOcc,
                                         uint8_t  srFlag,
                                         uint16_t bitLenHarq,
                                         float    extDtxThr)
    {
        cuphyPucchUciPrm_t u{};
        u.cellPrmDynIdx           = cellIdx;
        u.cellPrmStatIdx          = cellIdx;
        u.uciOutputIdx            = outIdx;
        u.formatType              = 1;
        u.rnti                    = 0x1234;
        u.bwpStart                = bwpStart;
        u.multiSlotTxIndicator    = 0;
        u.pi2Bpsk                 = 0;
        u.startPrb                = startPrb;
        u.prbSize                 = 1;
        u.startSym                = startSym;
        u.nSym                    = nSym;
        u.freqHopFlag             = freqHop ? 1 : 0;
        u.secondHopPrb            = secondHopPrb;
        u.groupHopFlag            = groupHop ? 1 : 0;
        u.sequenceHopFlag         = 0;
        u.initialCyclicShift      = 0;
        u.timeDomainOccIdx        = timeOcc;
        u.srFlag                  = srFlag;
        u.bitLenSr                = 0;
        u.bitLenHarq              = bitLenHarq;
        u.bitLenCsiPart1          = 0;
        u.AddDmrsFlag             = 0;
        u.dataScramblingId        = 0;
        u.DmrsScramblingId        = 0;
        u.maxCodeRate             = 0;
        u.nBitsCsi2               = 0;
        u.rankBitOffset           = 0;
        u.nRanksBits              = 0;
        u.DTXthreshold            = extDtxThr;
        u.uciP1P2Crpd_t.numPart2s = 0;
        u.nUplinkStreams          = 1;
        return u;
    }

    bool runF1Scenario(const std::vector<cuphyTensorPrm_t>&    dataRxPrm,
                       const std::vector<cuphyPucchUciPrm_t>&  uciPrms,
                       const std::vector<cuphyPucchCellPrm_t>& cellPrmsCpu,
                       cuphyPucchF0F1UciOut_t&                 out0,
                       uint8_t                                 enableUlRxBfFlag    = 0,
                       bool                                    oversubscribeBlocks = false)
    {
        cuphyPucchF1RxHndl_t f1Handle{};
        cuphyStatus_t        status = cuphyCreatePucchF1Rx(&f1Handle, cuStrmMain.handle());
        if(status != CUPHY_STATUS_SUCCESS) return false;

        size_t dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
        status = cuphyPucchF1RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
        if(status != CUPHY_STATUS_SUCCESS) return false;

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrGpu(dynDescrSizeBytes);

        cuphyPucchF1RxLaunchCfg_t launchCfg{};

        status = cuphySetupPucchF1Rx(
            f1Handle,
            const_cast<cuphyTensorPrm_t*>(dataRxPrm.data()),
            dF1Out,
            /*nCells*/ 1,
            /*nF1Ucis*/ static_cast<uint16_t>(uciPrms.size()),
            /*enableUlRxBf*/ enableUlRxBfFlag,
            const_cast<cuphyPucchUciPrm_t*>(uciPrms.data()),
            const_cast<cuphyPucchCellPrm_t*>(cellPrmsCpu.data()),
            /*enableCpuToGpuDescrAsyncCpy*/ 0,
            dynDescrCpu.addr(),
            dynDescrGpu.addr(),
            &launchCfg,
            cuStrmMain.handle());
        if(status != CUPHY_STATUS_SUCCESS) return false;

        // Make descriptor point to device-resident cell parameters
        allocDeviceCellPrms(1);
        cudaError_t err = cudaMemcpyAsync(dCellPrms, cellPrmsCpu.data(), sizeof(cuphyPucchCellPrm_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        auto* pDyn      = reinterpret_cast<pucchF1RxDynDescr_t*>(dynDescrCpu.addr());
        pDyn->pCellPrms = dCellPrms;

        err = cudaMemcpyAsync(dynDescrGpu.addr(), dynDescrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        if(oversubscribeBlocks)
        {
            launchCfg.kernelNodeParamsDriver.gridDimX += 1; // add one extra CTA to trigger early return path
        }
        const CUDA_KERNEL_NODE_PARAMS& k            = launchCfg.kernelNodeParamsDriver;
        CUresult                       launchResult = cuLaunchKernel(
            k.func,
            k.gridDimX,
            k.gridDimY,
            k.gridDimZ,
            k.blockDimX,
            k.blockDimY,
            k.blockDimZ,
            k.sharedMemBytes,
            static_cast<CUstream>(cuStrmMain.handle()),
            k.kernelParams,
            k.extra);
        EXPECT_EQ(launchResult, CUDA_SUCCESS);
        if(launchResult != CUDA_SUCCESS) return false;
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        err = cudaMemcpy(&out0, dF1Out + 0, sizeof(cuphyPucchF0F1UciOut_t), cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess);

        cuphyDestroyPucchF1Rx(f1Handle);
        return true;
    }

    template <typename PatchFn>
    bool runF1ScenarioPatchedDynDescr(const std::vector<cuphyTensorPrm_t>&    dataRxPrm,
                                      const std::vector<cuphyPucchUciPrm_t>&  uciPrms,
                                      const std::vector<cuphyPucchCellPrm_t>& cellPrmsCpu,
                                      PatchFn&&                               patchFn,
                                      cuphyPucchF0F1UciOut_t&                 out0,
                                      uint8_t                                 enableUlRxBfFlag = 0)
    {
        cuphyPucchF1RxHndl_t f1Handle{};
        cuphyStatus_t        status = cuphyCreatePucchF1Rx(&f1Handle, cuStrmMain.handle());
        if(status != CUPHY_STATUS_SUCCESS) return false;

        size_t dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
        status = cuphyPucchF1RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
        if(status != CUPHY_STATUS_SUCCESS) return false;

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrGpu(dynDescrSizeBytes);

        cuphyPucchF1RxLaunchCfg_t launchCfg{};
        status = cuphySetupPucchF1Rx(
            f1Handle,
            const_cast<cuphyTensorPrm_t*>(dataRxPrm.data()),
            dF1Out,
            /*nCells*/ 1,
            /*nF1Ucis*/ static_cast<uint16_t>(uciPrms.size()),
            /*enableUlRxBf*/ enableUlRxBfFlag,
            const_cast<cuphyPucchUciPrm_t*>(uciPrms.data()),
            const_cast<cuphyPucchCellPrm_t*>(cellPrmsCpu.data()),
            /*enableCpuToGpuDescrAsyncCpy*/ 0,
            dynDescrCpu.addr(),
            dynDescrGpu.addr(),
            &launchCfg,
            cuStrmMain.handle());
        if(status != CUPHY_STATUS_SUCCESS) return false;

        // Make descriptor point to device-resident cell parameters
        allocDeviceCellPrms(1);
        cudaError_t err = cudaMemcpyAsync(dCellPrms, cellPrmsCpu.data(), sizeof(cuphyPucchCellPrm_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        auto* pDyn      = reinterpret_cast<pucchF1RxDynDescr_t*>(dynDescrCpu.addr());
        pDyn->pCellPrms = dCellPrms;

        // Apply caller-provided patch to the CPU descriptor before copying to device.
        patchFn(*pDyn);

        err = cudaMemcpyAsync(dynDescrGpu.addr(), dynDescrCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        const CUDA_KERNEL_NODE_PARAMS& k            = launchCfg.kernelNodeParamsDriver;
        CUresult                       launchResult = cuLaunchKernel(
            k.func,
            k.gridDimX,
            k.gridDimY,
            k.gridDimZ,
            k.blockDimX,
            k.blockDimY,
            k.blockDimZ,
            k.sharedMemBytes,
            static_cast<CUstream>(cuStrmMain.handle()),
            k.kernelParams,
            k.extra);
        EXPECT_EQ(launchResult, CUDA_SUCCESS);
        if(launchResult != CUDA_SUCCESS) return false;
        err = cudaStreamSynchronize(cuStrmMain.handle());
        EXPECT_EQ(err, cudaSuccess);

        err = cudaMemcpy(&out0, dF1Out + 0, sizeof(cuphyPucchF0F1UciOut_t), cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess);

        cuphyDestroyPucchF1Rx(f1Handle);
        return true;
    }

    // Setup only: returns a pointer to the CPU dynamic descriptor for inspection
    bool setupOnly(const std::vector<cuphyTensorPrm_t>&         dataRxPrm,
                   const std::vector<cuphyPucchUciPrm_t>&       uciPrms,
                   const std::vector<cuphyPucchCellPrm_t>&      cellPrmsCpu,
                   cuphy::buffer<uint8_t, cuphy::pinned_alloc>& dynDescrCpu,
                   cuphy::buffer<uint8_t, cuphy::device_alloc>& dynDescrGpu,
                   cuphyPucchF1RxHndl_t&                        f1HandleOut,
                   uint8_t                                      enableUlRxBfFlag          = 0,
                   uint8_t                                      enableCpuToGpuDescrAsyncCpy = 0)
    {
        f1HandleOut          = nullptr;
        cuphyStatus_t status = cuphyCreatePucchF1Rx(&f1HandleOut, cuStrmMain.handle());
        if(status != CUPHY_STATUS_SUCCESS) return false;

        size_t dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
        status = cuphyPucchF1RxGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
        if(status != CUPHY_STATUS_SUCCESS) return false;

        dynDescrCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(dynDescrSizeBytes);
        dynDescrGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(dynDescrSizeBytes);

        cuphyPucchF1RxLaunchCfg_t launchCfg{};
        status = cuphySetupPucchF1Rx(
            f1HandleOut,
            const_cast<cuphyTensorPrm_t*>(dataRxPrm.data()),
            dF1Out,
            /*nCells*/ 1,
            /*nF1Ucis*/ static_cast<uint16_t>(uciPrms.size()),
            /*enableUlRxBf*/ enableUlRxBfFlag,
            const_cast<cuphyPucchUciPrm_t*>(uciPrms.data()),
            const_cast<cuphyPucchCellPrm_t*>(cellPrmsCpu.data()),
            /*enableCpuToGpuDescrAsyncCpy*/ enableCpuToGpuDescrAsyncCpy,
            dynDescrCpu.addr(),
            dynDescrGpu.addr(),
            &launchCfg,
            cuStrmMain.handle());
        return status == CUPHY_STATUS_SUCCESS;
    }
};

// Lightweight helper builders to reduce duplication in tests
static std::vector<cuphyPucchCellPrm_t> makeCellVec(int nRxAnt, int slotNum, int hopId, uint8_t mu = 0)
{
    cuphyPucchCellPrm_t cell{};
    cell.nRxAnt         = static_cast<uint16_t>(nRxAnt);
    cell.slotNum        = static_cast<uint16_t>(slotNum);
    cell.pucchHoppingId = static_cast<uint16_t>(hopId);
    cell.mu             = mu;
    return {cell};
}

// Build a single-entry tensor parameter vector for the harness tensor
static std::vector<cuphyTensorPrm_t> makeTensorPrmVec(F1TestHarness& h)
{
    cuphyTensorPrm_t tPrm{};
    tPrm.desc  = h.tDataRx.desc().handle();
    tPrm.pAddr = h.tDataRx.addr();
    return {tPrm};
}

// Fill a generic pattern: DMRS on even symbols with dmrsAmp, data on odd symbols with dataAmp
static void fillPattern(F1TestHarness& h,
                        int            nScTotal,
                        int            nOfdm,
                        int            nRxAnt,
                        int            startPrb,
                        int            secondHopPrb,
                        int            nSym,
                        bool           freqHop,
                        float          dmrsAmp,
                        float          dataAmp,
                        float          varEps = 0.0f)
{
    const size_t         elems = static_cast<size_t>(nScTotal) * nOfdm * nRxAnt;
    std::vector<__half2> hbuf(elems);
    auto                 idx              = [nOfdm, nRxAnt](int sc, int sym, int ant) { return static_cast<size_t>(sc) * nOfdm * nRxAnt + static_cast<size_t>(sym) * nRxAnt + ant; };
    int                  scFirstHopStart  = startPrb * CUPHY_N_TONES_PER_PRB;
    int                  scSecondHopStart = secondHopPrb * CUPHY_N_TONES_PER_PRB;
    for(int sc = 0; sc < nScTotal; ++sc)
    {
        for(int sym = 0; sym < nOfdm; ++sym)
        {
            for(int ant = 0; ant < nRxAnt; ++ant)
            {
                float a        = 0.0f;
                bool  inFirst  = (sc >= scFirstHopStart && sc < scFirstHopStart + CUPHY_N_TONES_PER_PRB);
                bool  inSecond = (sc >= scSecondHopStart && sc < scSecondHopStart + CUPHY_N_TONES_PER_PRB);
                if(sym >= 0 && sym < nSym)
                {
                    bool validSc = inFirst || (freqHop && inSecond);
                    if(validSc)
                    {
                        a = (((sym) & 1) == 0) ? dmrsAmp : dataAmp;
                    }
                }
                // Optional slight variation to avoid pathological cancellation in synthetic inputs
                if(varEps != 0.0f) a *= (1.0f + varEps * static_cast<float>(sc));
                __half2 v;
                v.x                     = __float2half(a);
                v.y                     = __float2half(0.0f);
                hbuf[idx(sc, sym, ant)] = v;
            }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(h.tDataRx.addr(), hbuf.data(), hbuf.size() * sizeof(__half2), cudaMemcpyHostToDevice, h.cuStrmMain.handle()));
    CUDA_CHECK(cudaStreamSynchronize(h.cuStrmMain.handle()));
}

static void fillTinyDmrsZeroData(F1TestHarness& h,
                                 int            nScTotal,
                                 int            nOfdm,
                                 int            nRxAnt,
                                 int            startPrb,
                                 int            secondHopPrb,
                                 int            nSym)
{
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 1e-7f, /*dataAmp*/ 0.0f, /*varEps*/ 1e-3f);
}

// Validator utilities for GPU outputs

// Combined validation: example-style mismatch count + optional per-field expectations
struct ExpectedOut
{
    int  idx;
    int  expectedNumHarq;
    bool expectErrorDefaults;
    int  expectedHarq0;
    int  expectedHarq1;
};

// validateOutputs:
// - Pulls GPU outputs and returns mismatch count (0 == success)
// - Optional expectations let tests assert specific fields
// - checkFinite can be disabled for extreme synthetic cases
static uint16_t validateOutputs(const cuphyPucchF0F1UciOut_t*   dOut,
                                int                             n,
                                cudaStream_t                    stream,
                                const std::vector<ExpectedOut>& expectations = {},
                                bool                            checkFinite  = true)
{
    std::vector<cuphyPucchF0F1UciOut_t> h(n);
    cudaError_t                         err = cudaMemcpyAsync(h.data(), dOut, n * sizeof(cuphyPucchF0F1UciOut_t), cudaMemcpyDeviceToHost, stream);
    EXPECT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(stream);
    EXPECT_EQ(err, cudaSuccess);

    uint16_t mismatches = 0;
    if(checkFinite)
    {
        for(int i = 0; i < n; ++i)
        {
            const auto& o  = h[i];
            bool        ok = std::isfinite(o.RSSI) && std::isfinite(o.RSRP) && std::isfinite(o.SinrDB) && std::isfinite(o.InterfDB);
            mismatches += ok ? 0 : 1;
        }
    }

    for(const auto& exp : expectations)
    {
        if(exp.idx < 0 || exp.idx >= n)
        {
            ++mismatches;
            continue;
        }
        const auto& out = h[exp.idx];
        if(exp.expectErrorDefaults)
        {
            EXPECT_EQ(out.NumHarq, 0);
            EXPECT_EQ(out.SRindication, 0);
            EXPECT_EQ(out.SRconfidenceLevel, 1);
            EXPECT_EQ(out.HarqconfidenceLevel, 1);
            EXPECT_EQ(out.SinrDB, -99);
            EXPECT_EQ(out.InterfDB, -99);
            EXPECT_FLOAT_EQ(out.taEstMicroSec, 0.0f);
        }
        else
        {
            EXPECT_EQ(out.NumHarq, exp.expectedNumHarq);
            if(exp.expectedHarq0 >= 0) EXPECT_EQ(out.HarqValues[0], exp.expectedHarq0);
            if(exp.expectedHarq1 >= 0) EXPECT_EQ(out.HarqValues[1], exp.expectedHarq1);
        }
    }
    return mismatches;
}

static cuphyPucchF0F1UciOut_t fetchOut0(const cuphyPucchF0F1UciOut_t* dOut, cudaStream_t stream)
{
    cuphyPucchF0F1UciOut_t h{};
    cudaError_t            err = cudaMemcpyAsync(&h, dOut, sizeof(h), cudaMemcpyDeviceToHost, stream);
    EXPECT_EQ(err, cudaSuccess);
    err = cudaStreamSynchronize(stream);
    EXPECT_EQ(err, cudaSuccess);
    return h;
}

TEST(PucchF1ReceiverTest, ErrorCondition_NoBits_NoSR)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 4;
    const int startSym     = 0;
    const int startPrb     = 0;
    const int secondHopPrb = 1;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    h.allocOutputs(1);

    auto vCell = makeCellVec(nRxAnt, 10, 7, 0);

    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, nSym, startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 0,
                                       /*extDtxThr*/ 1.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    std::vector<ExpectedOut> exp{{0, 0, true, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp), 0);
}

TEST(PucchF1ReceiverTest, DtxDetected_WithHarq_Hopping)
{
    F1TestHarness h;

    const int nRxAnt       = 2;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ true, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    h.allocOutputs(1);

    auto vCell = makeCellVec(nRxAnt, 20, 17, 0);

    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, nSym, startSym,
                                       /*freqHop*/ true,
                                       secondHopPrb,
                                       /*groupHop*/ true,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 1,
                                       /*bitLenHarq*/ 2,
                                       /*extDtxThr*/ 1000.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    // For this floor-threshold test, just ensure we produced a valid output
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), /*expectations*/ {}, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, GroupingAndDefaultDtxThresholdBranch)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 4;
    const int startSym     = 0;
    const int startPrb     = 3;
    const int secondHopPrb = 7;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // Two UCIs that should group together (same startPrb/startSym and cell)
    auto u0 = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, nSym, startSym,
                                        /*freqHop*/ false,
                                        secondHopPrb,
                                        /*groupHop*/ false,
                                        /*timeOcc*/ 1,
                                        /*srFlag*/ 0,
                                        /*bitLenHarq*/ 1,
                                        /*extDtx*/ 1000.0f);
    // Second UCI uses default/legacy DTX path (<= CUPHY_DEFAULT_EXT_DTX_THRESHOLD)
    auto                            u1 = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 1, /*bwpStart*/ 0, startPrb, nSym, startSym,
                                        /*freqHop*/ false,
                                        secondHopPrb,
                                        /*groupHop*/ false,
                                        /*timeOcc*/ 2,
                                        /*srFlag*/ 0,
                                        /*bitLenHarq*/ 1,
                                        /*extDtx*/ CUPHY_DEFAULT_EXT_DTX_THRESHOLD);
    std::vector<cuphyPucchUciPrm_t> vUci{u0, u1};

    // Output and cell params
    h.allocOutputs(2);
    auto vCell = makeCellVec(nRxAnt, 30, 11, 0);

    // Setup only to inspect CPU dynamic descriptor
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle));

    auto* pDyn = reinterpret_cast<pucchF1RxDynDescr_t*>(dynCpu.addr());
    ASSERT_EQ(pDyn->numUciGrps, 1);
    ASSERT_EQ(pDyn->uciGrpPrms[0].nUciInGrp, 2);
    // First UCI took external DTX
    float dtx0 = __half2float(pDyn->uciGrpPrms[0].DTXthreshold[0]);
    // Second UCI used default branch -> 1.0
    float dtx1 = __half2float(pDyn->uciGrpPrms[0].DTXthreshold[1]);
    EXPECT_GT(dtx0, 1.0f);
    EXPECT_FLOAT_EQ(dtx1, 1.0f);

    cuphyDestroyPucchF1Rx(handle);

    // Also run the kernel and validate both outputs via helper
    cuphyPucchF0F1UciOut_t out0{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out0));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}, {1, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 2, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, MaxUciPerGroupOverflowIsClamped)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 4;
    const int startSym     = 0;
    const int startPrb     = 4;
    const int secondHopPrb = 6;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    const int                       nWanted = CUPHY_PUCCH_F1_MAX_UCI_PER_GRP + 2; // exceed by 2
    std::vector<cuphyPucchUciPrm_t> vUci;
    vUci.reserve(nWanted);
    for(int i = 0; i < nWanted; ++i)
    {
        vUci.push_back(F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ static_cast<uint16_t>(i), /*bwpStart*/ 0, startPrb, nSym, startSym,
                                                 /*freqHop*/ false,
                                                 secondHopPrb,
                                                 /*groupHop*/ false,
                                                 /*timeOcc*/ static_cast<uint8_t>(i % 3),
                                                 /*srFlag*/ 0,
                                                 /*bitLenHarq*/ 1,
                                                 /*extDtx*/ 1000.0f));
    }

    h.allocOutputs(nWanted);
    auto vCell = makeCellVec(nRxAnt, 40, 13, 0);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle));

    auto* pDyn = reinterpret_cast<pucchF1RxDynDescr_t*>(dynCpu.addr());
    ASSERT_EQ(pDyn->numUciGrps, 1);
    EXPECT_EQ(pDyn->uciGrpPrms[0].nUciInGrp, CUPHY_PUCCH_F1_MAX_UCI_PER_GRP);

    cuphyDestroyPucchF1Rx(handle);
}

TEST(PucchF1ReceiverTest, MaxGroupsOverflow_DropsAdditionalGroups)
{
    F1TestHarness h;

    const int nRxAnt   = 1;
    const int nOfdm    = OFDM_SYMBOLS_PER_SLOT;
    const int nSym     = 4;
    const int startSym = 0;

    const int nWanted = static_cast<int>(CUPHY_PUCCH_F1_MAX_GRPS) + 1; // exceed by 1 to trigger `continue;` when adding a new group

    // Make the tensor large enough to cover the largest startPrb we use (even though this test only inspects setup).
    const int startPrbMax = nWanted - 1;
    const int nScTotal    = (startPrbMax + 2) * CUPHY_N_TONES_PER_PRB;
    const int secondHopPrb = 0;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, /*startPrb*/ 0, /*secondHopPrb*/ secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // Output and cell params
    h.allocOutputs(nWanted);
    auto vCell = makeCellVec(nRxAnt, 100, 77, 0);

    // Create > MAX_GRPS UCIs that *must* form distinct groups by varying startPrb (thus startCrb).
    std::vector<cuphyPucchUciPrm_t> vUci;
    vUci.reserve(nWanted);
    for(int i = 0; i < nWanted; ++i)
    {
        vUci.push_back(F1TestHarness::makeUciPrm(/*cellIdx*/ 0,
                                                 /*outIdx*/ static_cast<uint16_t>(i),
                                                 /*bwpStart*/ 0,
                                                 /*startPrb*/ static_cast<uint16_t>(i),
                                                 /*nSym*/ static_cast<uint8_t>(nSym),
                                                 /*startSym*/ static_cast<uint8_t>(startSym),
                                                 /*freqHop*/ false,
                                                 /*secondHopPrb*/ static_cast<uint16_t>(secondHopPrb),
                                                 /*groupHop*/ false,
                                                 /*timeOcc*/ 0,
                                                 /*srFlag*/ 0,
                                                 /*bitLenHarq*/ 1,
                                                 /*extDtx*/ 10.0f));
    }

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle));

    auto* pDyn = reinterpret_cast<pucchF1RxDynDescr_t*>(dynCpu.addr());

    // Setup must clamp groups to MAX_GRPS and drop the extra group via the `continue;` path.
    ASSERT_EQ(pDyn->numUciGrps, CUPHY_PUCCH_F1_MAX_GRPS);

    const uint16_t droppedOutIdx = static_cast<uint16_t>(nWanted - 1);
    for(uint16_t g = 0; g < pDyn->numUciGrps; ++g)
    {
        EXPECT_EQ(pDyn->uciGrpPrms[g].nUciInGrp, 1);
        EXPECT_NE(pDyn->uciGrpPrms[g].uciOutputIdx[0], droppedOutIdx);
    }

    cuphyDestroyPucchF1Rx(handle);
}

TEST(PucchF1ReceiverTest, NewGroup_FreqHop_Calculations_And_DefaultDTX)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 7; // odd to exercise floor divisions
    const int startSym     = 0;
    const int startPrb     = 8;
    const int secondHopPrb = 12;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ true, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // Create a single UCI that forces a new group with freqHop on and default DTX (<= default)
    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 2, startPrb, nSym, startSym,
                                       /*freqHop*/ true,
                                       secondHopPrb,
                                       /*groupHop*/ true,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ CUPHY_DEFAULT_EXT_DTX_THRESHOLD);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 50, 21, 0);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle));

    auto* pDyn = reinterpret_cast<pucchF1RxDynDescr_t*>(dynCpu.addr());
    ASSERT_EQ(pDyn->numUciGrps, 1);
    const auto& g = pDyn->uciGrpPrms[0];

    // New group created
    EXPECT_EQ(g.nUciInGrp, 1);
    EXPECT_EQ(g.freqHopFlag, 1);
    EXPECT_EQ(g.groupHopFlag, 1);
    // second hop crb = secondHopPrb + bwpStart
    EXPECT_EQ(g.secondHopCrb, static_cast<uint16_t>(secondHopPrb + u.bwpStart));

    // Derived symbol counts
    uint8_t nSym_data = static_cast<uint8_t>(std::floor(nSym / 2.0));
    uint8_t nSym_dmrs = static_cast<uint8_t>(nSym - nSym_data);
    uint8_t nData1    = static_cast<uint8_t>(std::floor(nSym_data / 2.0));
    uint8_t nFirstHop = nSym_data;
    uint8_t nDmrs1    = static_cast<uint8_t>(nFirstHop - nData1);
    uint8_t nData2    = static_cast<uint8_t>(nSym_data - nData1);
    uint8_t nDmrs2    = static_cast<uint8_t>(nSym_dmrs - nDmrs1);

    EXPECT_EQ(g.nSym_data, nSym_data);
    EXPECT_EQ(g.nSym_dmrs, nSym_dmrs);
    EXPECT_EQ(g.nSymDataFirstHop, nData1);
    EXPECT_EQ(g.nSymFirstHop, nFirstHop);
    EXPECT_EQ(g.nSymDMRSFirstHop, nDmrs1);
    EXPECT_EQ(g.nSymDataSecondHop, nData2);
    EXPECT_EQ(g.nSymDMRSSecondHop, nDmrs2);

    // Default DTX path in new group branch -> 1.0
    EXPECT_FLOAT_EQ(__half2float(g.DTXthreshold[0]), 1.0f);

    cuphyDestroyPucchF1Rx(handle);

    // Launch and validate with helper
    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    std::vector<ExpectedOut> exp2{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp2, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, FreqHop_WtTable_SwitchCoverage)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);

    auto vCell = makeCellVec(nRxAnt, 60, 31, 0);

    for(int nSym = 4; nSym <= 14; ++nSym)
    {
        fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ true, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

        auto vT = makeTensorPrmVec(h);

        auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                           /*freqHop*/ true,
                                           secondHopPrb,
                                           /*groupHop*/ true,
                                           /*timeOcc*/ 0,
                                           /*srFlag*/ 0,
                                           /*bitLenHarq*/ 1,
                                           /*extDtx*/ 10.0f);
        std::vector<cuphyPucchUciPrm_t> vUci{u};

        cuphyPucchF0F1UciOut_t out{};
        ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
        std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
        EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0) << "nSym=" << nSym;
    }
}

TEST(PucchF1ReceiverTest, NonHop_WtTable_SwitchCoverage)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int startSym     = 0;
    const int startPrb     = 1;
    const int secondHopPrb = 3;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);

    auto vCell = makeCellVec(nRxAnt, 70, 41, 0);

    for(int nSym = 4; nSym <= 14; ++nSym)
    {
        fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

        auto vT = makeTensorPrmVec(h);

        auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                           /*freqHop*/ false,
                                           secondHopPrb,
                                           /*groupHop*/ false,
                                           /*timeOcc*/ 0,
                                           /*srFlag*/ 0,
                                           /*bitLenHarq*/ 1,
                                           /*extDtx*/ 10.0f);
        std::vector<cuphyPucchUciPrm_t> vUci{u};

        cuphyPucchF0F1UciOut_t out{};
        ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
        std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
        EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0) << "nSym=" << nSym;
    }
}

TEST(PucchF1ReceiverTest, FreqHop_WtTable_DefaultCase_Coverage)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    // Start with a valid nSym so setup succeeds and shared-memory sizing is sane.
    const int nSymValid = 6;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSymValid, /*freqHop*/ true, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 75, 45, 0);

    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSymValid), startSym,
                                       /*freqHop*/ true,
                                       secondHopPrb,
                                       /*groupHop*/ true,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1ScenarioPatchedDynDescr(
        vT,
        vUci,
        vCell,
        [](pucchF1RxDynDescr_t& dyn) {
            // Force an invalid group nSym to execute the `default: break;` path in the freqHop Wt lookup switch.
            // Also set the derived symbol counts to avoid dereferencing the (uninitialized) Wt pointers after the switch.
            auto& g           = dyn.uciGrpPrms[0];
            g.nSym            = 3; // not in [4..14]
            g.nSym_data       = 0;
            g.nSym_dmrs       = 1;
            g.nSymDataFirstHop  = 0;
            g.nSymFirstHop      = 0;
            g.nSymDMRSFirstHop  = 1;
            g.nSymDataSecondHop = 0;
            g.nSymDMRSSecondHop = 0;
        },
        out));

    // Just ensure the kernel ran and outputs can be copied back (values may be non-finite due to synthetic invalid nSym).
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), {}, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, NonHop_WtTable_DefaultCase_Coverage)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int startSym     = 0;
    const int startPrb     = 1;
    const int secondHopPrb = 3;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    const int nSymValid = 6;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSymValid, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 76, 46, 0);

    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSymValid), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1ScenarioPatchedDynDescr(
        vT,
        vUci,
        vCell,
        [](pucchF1RxDynDescr_t& dyn) {
            // Force an invalid group nSym to execute the `default: break;` path in the nonHop Wt lookup switch.
            // Also set symbol counts to prevent dereferencing uninitialized Wt pointers.
            auto& g      = dyn.uciGrpPrms[0];
            g.nSym       = 3; // not in [4..14]
            g.nSym_data  = 0;
            g.nSym_dmrs  = 1;
            // Keep other hop-related fields benign (not used in non-hop path, but set defensively).
            g.nSymDataFirstHop   = 0;
            g.nSymFirstHop       = 0;
            g.nSymDMRSFirstHop   = 0;
            g.nSymDataSecondHop  = 0;
            g.nSymDMRSSecondHop  = 0;
        },
        out));

    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), {}, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, BitLenHarq_SwitchCases_NoDTX)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6; // any even number > 4
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    // Small external DTX threshold to avoid DTX branch
    const float smallDtx = 1e-6f;

    // Shared tensor & outputs
    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 80, 51, 0);

    // Case 0: bitLenHarq = 0, srFlag = 1
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);
    auto vT = makeTensorPrmVec(h);
    auto u0 = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                        /*freqHop*/ false,
                                        secondHopPrb,
                                        /*groupHop*/ false,
                                        /*timeOcc*/ 0,
                                        /*srFlag*/ 1,
                                        /*bitLenHarq*/ 0,
                                        /*extDtx*/ smallDtx);
    {
        std::vector<cuphyPucchUciPrm_t> vUci{u0};
        cuphyPucchF0F1UciOut_t          out{};
        ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
        std::vector<ExpectedOut> exp{{0, 0, false, -1, -1}}; // NumHarq 0; SR may vary with sign
        EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp), 0);
    }

    // Case 1: bitLenHarq = 1
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);
    auto u1       = u0;
    u1.bitLenHarq = 1;
    u1.srFlag     = 0;
    {
        std::vector<cuphyPucchUciPrm_t> vUci{u1};
        cuphyPucchF0F1UciOut_t          out{};
        ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
        std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}}; // ignore exact HARQ bit
        EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp), 0);
    }

    // Case 2: bitLenHarq = 2
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);
    auto u2       = u0;
    u2.bitLenHarq = 2;
    u2.srFlag     = 0;
    {
        std::vector<cuphyPucchUciPrm_t> vUci{u2};
        cuphyPucchF0F1UciOut_t          out{};
        ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
        std::vector<ExpectedOut> exp{{0, 2, false, -1, -1}}; // ignore exact HARQ bits
        EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp), 0);
    }
}

TEST(PucchF1ReceiverTest, BitLenHarq_Case0_SetsSr_When_QamX_Positive)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    // Use a standard, non-pathological pattern and then search (cs0,timeOcc) to guarantee qam_est_x > 0 once.
    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 86, 59, 0);

    // Keep input params valid for setup; we will patch the group descriptor for the exact case we want.
    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 1,
                                       /*bitLenHarq*/ 0,
                                       /*extDtx*/ 1e-6f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    bool found = false;
    for(uint8_t occ = 0; occ < 7 && !found; ++occ)
    {
        for(uint8_t cs0 = 0; cs0 < 12 && !found; ++cs0)
        {
            cuphyPucchF0F1UciOut_t out{};
            ASSERT_TRUE(h.runF1ScenarioPatchedDynDescr(
                vT,
                vUci,
                vCell,
                [occ, cs0](pucchF1RxDynDescr_t& dyn) {
                    auto& g                 = dyn.uciGrpPrms[0];
                    g.bitLenHarq[0]         = 0;
                    g.srFlag[0]             = 1;
                    g.cs0[0]                = cs0;
                    g.timeDomainOccIdx[0]   = occ;
                    // Keep the DTX threshold as small as possible (it is clamped to 1e-16 in-kernel).
                    g.DTXthreshold[0]       = __float2half(0.0f);
                },
                out));

            const auto hOut = fetchOut0(h.dF1Out, h.cuStrmMain.handle());
            if(hOut.SRindication == 1)
            {
                // SRindication==1 with bitLenHarq==0 implies the `if(qam_est_x > 0) { sr = 1; }` line executed.
                found = true;
            }
        }
    }
    ASSERT_TRUE(found);
}

TEST(PucchF1ReceiverTest, BitLenHarq_Case2_SetsSecondBit_When_QamY_NonPositive)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 87, 60, 0);

    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 2,
                                       /*extDtx*/ 1e-6f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    bool found = false;
    for(uint8_t occ = 0; occ < 7 && !found; ++occ)
    {
        for(uint8_t cs0 = 0; cs0 < 12 && !found; ++cs0)
        {
            cuphyPucchF0F1UciOut_t out{};
            ASSERT_TRUE(h.runF1ScenarioPatchedDynDescr(
                vT,
                vUci,
                vCell,
                [occ, cs0](pucchF1RxDynDescr_t& dyn) {
                    auto& g               = dyn.uciGrpPrms[0];
                    g.bitLenHarq[0]       = 2;
                    g.srFlag[0]           = 0;
                    g.cs0[0]              = cs0;
                    g.timeDomainOccIdx[0] = occ;
                    g.DTXthreshold[0]     = __float2half(0.0f);
                },
                out));

            const auto hOut = fetchOut0(h.dF1Out, h.cuStrmMain.handle());
            // Avoid the DTX path (it sets HARQ values to 2,2 for bitLenHarq>0).
            if(hOut.HarqValues[0] == 2 && hOut.HarqValues[1] == 2) continue;

            if(hOut.NumHarq == 2 && hOut.HarqValues[1] == 1)
            {
                // HarqValues[1]==1 implies b_est bit1 was set, which only happens at:
                // `if(qam_est_y <= 0) { b_est |= 0b10; }`
                found = true;
            }
        }
    }
    ASSERT_TRUE(found);
}

TEST(PucchF1ReceiverTest, BitLenHarq_DefaultCase_Coverage_PatchedDescriptor)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 5;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    h.allocOutputs(1);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 88, 61, 0);

    // Use a valid bitLenHarq for setup; patch to an out-of-range value at kernel time.
    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 1e-6f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1ScenarioPatchedDynDescr(
        vT,
        vUci,
        vCell,
        [](pucchF1RxDynDescr_t& dyn) {
            auto& g           = dyn.uciGrpPrms[0];
            g.bitLenHarq[0]   = 3; // triggers `default:` in switch(bitLenHarq)
            g.srFlag[0]       = 0;
            g.DTXthreshold[0] = __float2half(0.0f);
        },
        out));

    const auto hOut = fetchOut0(h.dF1Out, h.cuStrmMain.handle());
    EXPECT_EQ(hOut.NumHarq, 3);
    // In the default case, b_est stays 0 so both HARQ bits are 0 (and we must not be in the DTX branch).
    EXPECT_NE(hOut.HarqValues[0], 2);
    EXPECT_NE(hOut.HarqValues[1], 2);
    EXPECT_EQ(hOut.HarqValues[0], 0);
    EXPECT_EQ(hOut.HarqValues[1], 0);
}

TEST(PucchF1ReceiverTest, BitLenHarq_DefaultCase_NoDTX)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 3;
    const int secondHopPrb = 6;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    const float smallDtx = 1e-6f;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // Use an invalid/non-supported bitLenHarq to trigger default branch
    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 3,
                                       /*extDtx*/ smallDtx);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 85, 57, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    // Expect NumHarq to reflect requested bitLenHarq (3), but ignore bit values
    std::vector<ExpectedOut> exp{{0, 3, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, EnableUlRxBf_BranchCoverage)
{
    F1TestHarness h;

    const int nRxAnt       = 4; // enough to exercise branch but keep small
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 1;
    const int secondHopPrb = 2;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // UCI with nUplinkStreams set to 2 so that numRxAnt branch uses it when enableUlRxBf=1
    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    // adjust nUplinkStreams through descriptor path by grouping setup
    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 90, 61, 0);

    // First, setup to access dyn descriptor and set uci group nUplinkStreams
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};
    std::vector<cuphyPucchUciPrm_t>             vUci{u};
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle, /*enableUlRxBf*/ 1));
    auto* pDyn = reinterpret_cast<pucchF1RxDynDescr_t*>(dynCpu.addr());
    ASSERT_EQ(pDyn->numUciGrps, 1);
    pDyn->uciGrpPrms[0].nUplinkStreams = 2;
    cuphyDestroyPucchF1Rx(handle);

    // Now run with enableUlRxBf=1 to trigger branch (uses nUplinkStreams)
    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out, /*enableUlRxBf*/ 1));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, Setup_EnableCpuToGpuDescrAsyncCopy_CoversMemcpyAsync)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 4;
    const int startSym     = 0;
    const int startPrb     = 0;
    const int secondHopPrb = 1;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);
    h.allocOutputs(1);

    auto vT    = makeTensorPrmVec(h);
    auto vCell = makeCellVec(nRxAnt, 110, 79, 0);

    auto u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynGpu(1);
    cuphyPucchF1RxHndl_t                        handle{};

    // Enable the optional CPU->GPU descriptor async copy in setup (covers the cudaMemcpyAsync line in setup()).
    ASSERT_TRUE(h.setupOnly(vT, vUci, vCell, dynCpu, dynGpu, handle, /*enableUlRxBf*/ 0, /*enableCpuToGpuDescrAsyncCpy*/ 1));
    cudaError_t err = cudaStreamSynchronize(h.cuStrmMain.handle());
    EXPECT_EQ(err, cudaSuccess);

    cuphyDestroyPucchF1Rx(handle);
}

TEST(PucchF1ReceiverTest, EarlyExit_When_GlobalGroupExceedsNumUciGrps)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 4;
    const int startSym     = 0;
    const int startPrb     = 0;
    const int secondHopPrb = 1;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 95, 63, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out, /*enableUlRxBf*/ 0, /*oversubscribeBlocks*/ true));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, GroupHop_WithoutFreqHop_SecondHopElseBranch)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 4;
    const int secondHopPrb = 9;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    // groupHopFlag = 1, freqHopFlag = 0
    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ true,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 97, 23, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, AlphaBranch_NumRxAnt_3to5)
{
    F1TestHarness h;

    const int nRxAnt       = 4; // triggers alpha = ALPHA_R3R4R5
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 2;
    const int secondHopPrb = 4;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 101, 19, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp, /*checkFinite*/ false), 0);
}

TEST(PucchF1ReceiverTest, AlphaBranch_NumRxAnt_ge6)
{
    F1TestHarness h;

    const int nRxAnt       = 6; // triggers alpha = ALPHA_DEFAULT
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 3;
    const int secondHopPrb = 7;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    fillPattern(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym, /*freqHop*/ false, /*dmrsAmp*/ 0.5f, /*dataAmp*/ 0.25f);

    auto vT = makeTensorPrmVec(h);

    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 1,
                                       /*extDtx*/ 10.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 103, 29, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    std::vector<ExpectedOut> exp{{0, 1, false, -1, -1}};
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), exp), 0);
}

TEST(PucchF1ReceiverTest, DtxThresholdFlooredToMin)
{
    F1TestHarness h;

    const int nRxAnt       = 1;
    const int nOfdm        = OFDM_SYMBOLS_PER_SLOT;
    const int nSym         = 6;
    const int startSym     = 0;
    const int startPrb     = 0;
    const int secondHopPrb = 1;
    const int nScTotal     = (secondHopPrb + 2) * CUPHY_N_TONES_PER_PRB;

    h.allocDataRxTensor(nScTotal, nOfdm, nRxAnt, true);
    // Build tensor with tiny DMRS energy and zero data so RSSI is finite but qam_est ~ 0
    fillTinyDmrsZeroData(h, nScTotal, nOfdm, nRxAnt, startPrb, secondHopPrb, nSym);

    auto vT = makeTensorPrmVec(h);

    // Set bitLenHarq=2 so DTX path sets HarqValues to 2/2
    auto                            u = F1TestHarness::makeUciPrm(/*cellIdx*/ 0, /*outIdx*/ 0, /*bwpStart*/ 0, startPrb, static_cast<uint8_t>(nSym), startSym,
                                       /*freqHop*/ false,
                                       secondHopPrb,
                                       /*groupHop*/ false,
                                       /*timeOcc*/ 0,
                                       /*srFlag*/ 0,
                                       /*bitLenHarq*/ 2,
                                       /*extDtx*/ 1.0f);
    std::vector<cuphyPucchUciPrm_t> vUci{u};

    h.allocOutputs(1);
    auto vCell = makeCellVec(nRxAnt, 107, 33, 0);

    cuphyPucchF0F1UciOut_t out{};
    ASSERT_TRUE(h.runF1Scenario(vT, vUci, vCell, out));
    // Just ensure a valid output was produced; the goal is to hit the floor branch
    EXPECT_EQ(validateOutputs(h.dF1Out, 1, h.cuStrmMain.handle(), /*expectations*/ {}, /*checkFinite*/ false), 0);
}

} // namespace

int main(int argc, char* argv[])
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    return result;
}
