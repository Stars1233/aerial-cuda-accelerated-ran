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
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "cuphy.h"
#include "cuda_fp16.h"
#include "common_utils.hpp"
#include "pucch_F2_front_end/pucch_F2_front_end.hpp"
#include <cmath>


namespace
{

constexpr int SYM_PER_SLOT = 14;

struct DevAlloc
{
    template <typename T>
    static T* malloc_array(size_t count)
    {
        T* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        return p;
    }
};

struct PucchF2Synthetic
{
    void*                   d_tDataRx    = nullptr;
    size_t                  tDataRxElems = 0;
    cuphyTensorPrm_t        tPrmDataRx{};
    cuphyTensorDescriptor_t tDesc{};

    std::vector<__half*>  descramLLRaddrsVec;
    __half*               sharedDescramLLR = nullptr;
    std::vector<uint16_t> Eseg1Host;
    uint8_t*              d_pDTXflags = nullptr;
    float *               d_pSinr = nullptr, *d_pRssi = nullptr, *d_pRsrp = nullptr, *d_pInterf = nullptr, *d_pNoiseVar = nullptr, *d_pTaEst = nullptr;

    void* dynDescrCpu = nullptr;
    void* dynDescrGpu = nullptr;

    cuphyPucchF2RxHndl_t      rxHndl = nullptr;
    cuphyPucchF2RxLaunchCfg_t launchCfg{};

    cudaStream_t stream = nullptr;

    void initStream() { CUDA_CHECK(cudaStreamCreate(&stream)); }
    void destroyStream()
    {
        if(stream) cudaStreamDestroy(stream);
        stream = nullptr;
    }

    void createTensor3D_C16F(int nScTotal, int nSyms, int nRxAnt)
    {
        tDataRxElems = static_cast<size_t>(nScTotal) * nSyms * nRxAnt;
        CUDA_CHECK(cudaMalloc(&d_tDataRx, tDataRxElems * sizeof(__half2)));
        std::vector<__half2> host(tDataRxElems);
        for(size_t i = 0; i < tDataRxElems; ++i)
        {
            host[i].x = __float2half(0.01f);
            host[i].y = __float2half(0.02f);
        }
        CUDA_CHECK(cudaMemcpy(d_tDataRx, host.data(), host.size() * sizeof(__half2), cudaMemcpyHostToDevice));
        ASSERT_EQ(cuphyCreateTensorDescriptor(&tDesc), CUPHY_STATUS_SUCCESS);
        int32_t dims[3] = {nScTotal, nSyms, nRxAnt};
        ASSERT_EQ(cuphySetTensorDescriptor(tDesc, CUPHY_C_16F, 3, dims, nullptr, 0), CUPHY_STATUS_SUCCESS);
        tPrmDataRx.desc  = tDesc;
        tPrmDataRx.pAddr = d_tDataRx;
    }

    void allocOutputs(int nUcis)
    {
        descramLLRaddrsVec.resize(nUcis, nullptr);
        Eseg1Host.resize(nUcis, 0);
        d_pDTXflags = DevAlloc::malloc_array<uint8_t>(nUcis);
        d_pSinr     = DevAlloc::malloc_array<float>(nUcis);
        d_pRssi     = DevAlloc::malloc_array<float>(nUcis);
        d_pRsrp     = DevAlloc::malloc_array<float>(nUcis);
        d_pInterf   = DevAlloc::malloc_array<float>(nUcis);
        d_pNoiseVar = DevAlloc::malloc_array<float>(nUcis);
        d_pTaEst    = DevAlloc::malloc_array<float>(nUcis);
    }

    void freeOutputs()
    {
        if(sharedDescramLLR)
        {
            cudaFree(sharedDescramLLR);
            sharedDescramLLR = nullptr;
        }
        else
        {
            for(auto p : descramLLRaddrsVec)
            {
                if(p) cudaFree(p);
            }
        }
        if(d_pDTXflags) cudaFree(d_pDTXflags);
        if(d_pSinr) cudaFree(d_pSinr);
        if(d_pRssi) cudaFree(d_pRssi);
        if(d_pRsrp) cudaFree(d_pRsrp);
        if(d_pInterf) cudaFree(d_pInterf);
        if(d_pNoiseVar) cudaFree(d_pNoiseVar);
        if(d_pTaEst) cudaFree(d_pTaEst);
        descramLLRaddrsVec.clear();
        Eseg1Host.clear();
    }

    void destroyTensor()
    {
        if(d_tDataRx) cudaFree(d_tDataRx);
        d_tDataRx = nullptr;
        if(tDesc) cuphyDestroyTensorDescriptor(tDesc);
        tDesc = nullptr;
    }
};

} // namespace

// Common helper to run a set of UCI scenarios
namespace
{
struct RunResult
{
    std::vector<uint8_t> dtx;
    std::vector<float>   sinr, rssi, rsrp, interf;
};

static RunResult runScenario(const std::vector<cuphyPucchUciPrm_t>& ucis, int nRxAnt, int nScTotal)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = 1;
    const int nUcis  = static_cast<int>(ucis.size());

    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);

    // Copy UCIs and prepare output buffers
    std::vector<cuphyPucchUciPrm_t> prms = ucis;
    for(int i = 0; i < nUcis; ++i)
    {
        hw.Eseg1Host[i]          = static_cast<uint16_t>(prms[i].nSym * prms[i].prbSize * 16);
        hw.descramLLRaddrsVec[i] = DevAlloc::malloc_array<__half>(hw.Eseg1Host[i]);
    }

    // Unified memory for cell params so device can dereference
    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    pCell[0].nRxAnt         = nRxAnt;
    pCell[0].pucchHoppingId = 0;
    pCell[0].slotNum        = 0;

    // Receiver + descriptor
    RunResult     out; // prepare result holder
    cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    if(stCreate != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyCreatePucchF2Rx failed";
        return out;
    }
    size_t        dynDescrSize = 0, dynDescrAlign = 0;
    cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    if(stInfo != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyPucchF2RxGetDescrInfo failed";
        return out;
    }
    dynDescrSize += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));
    cuphyStatus_t stSetup = cuphySetupPucchF2Rx(hw.rxHndl,
                                                &hw.tPrmDataRx,
                                                hw.descramLLRaddrsVec.data(),
                                                hw.d_pDTXflags,
                                                hw.d_pSinr,
                                                hw.d_pRssi,
                                                hw.d_pRsrp,
                                                hw.d_pInterf,
                                                hw.d_pNoiseVar,
                                                hw.d_pTaEst,
                                                nCells,
                                                nUcis,
                                                0,
                                                prms.data(),
                                                pCell,
                                                false,
                                                hw.dynDescrCpu,
                                                hw.dynDescrGpu,
                                                &hw.launchCfg,
                                                hw.stream);
    if(stSetup != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphySetupPucchF2Rx failed";
        return out;
    }
    CUDA_CHECK(cudaMemcpyAsync(hw.dynDescrGpu, hw.dynDescrCpu, dynDescrSize, cudaMemcpyHostToDevice, hw.stream));
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));

    const CUDA_KERNEL_NODE_PARAMS& p = hw.launchCfg.kernelNodeParamsDriver;
    {
        CUresult lres = cuLaunchKernel(p.func, p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY, p.blockDimZ, p.sharedMemBytes, static_cast<CUstream>(hw.stream), p.kernelParams, p.extra);
        if(lres != CUDA_SUCCESS)
        {
            ADD_FAILURE() << "cuLaunchKernel failed";
            return out;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));

    out.dtx.resize(nUcis);
    out.sinr.resize(nUcis);
    out.rssi.resize(nUcis);
    out.rsrp.resize(nUcis);
    out.interf.resize(nUcis);
    CUDA_CHECK(cudaMemcpy(out.dtx.data(), hw.d_pDTXflags, nUcis * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.sinr.data(), hw.d_pSinr, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rssi.data(), hw.d_pRssi, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rsrp.data(), hw.d_pRsrp, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.interf.data(), hw.d_pInterf, nUcis * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();
    return out;
}

static uint16_t setupAndGetNumUcisAfterClamp(const std::vector<cuphyPucchUciPrm_t>& ucis, int nRxAnt, int nScTotal)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = 1;
    const int nUcis  = static_cast<int>(ucis.size());

    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);

    // Avoid N cudaMalloc calls: allocate a single shared LLR buffer.
    size_t maxEseg1 = 0;
    for(const auto& u : ucis)
    {
        maxEseg1 = std::max<size_t>(maxEseg1, static_cast<size_t>(u.nSym) * u.prbSize * 16);
    }
    if(maxEseg1 == 0) maxEseg1 = 16;
    hw.sharedDescramLLR = DevAlloc::malloc_array<__half>(maxEseg1);
    for(int i = 0; i < nUcis; ++i)
    {
        hw.Eseg1Host[i]          = static_cast<uint16_t>(maxEseg1);
        hw.descramLLRaddrsVec[i] = hw.sharedDescramLLR;
    }

    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    pCell[0].nRxAnt         = nRxAnt;
    pCell[0].pucchHoppingId = 0;
    pCell[0].slotNum        = 0;

    cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    EXPECT_EQ(stCreate, CUPHY_STATUS_SUCCESS);

    size_t dynDescrSize = 0, dynDescrAlign = 0;
    cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    EXPECT_EQ(stInfo, CUPHY_STATUS_SUCCESS);

    dynDescrSize += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    if(hw.dynDescrCpu == nullptr)
    {
        ADD_FAILURE() << "Failed to allocate dyn descriptor host memory";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return 0;
    }
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));

    // Call setup (this is what contains the clamp branch we want to cover).
    std::vector<cuphyPucchUciPrm_t> prms = ucis;
    cuphyStatus_t stSetup                = cuphySetupPucchF2Rx(hw.rxHndl,
                                               &hw.tPrmDataRx,
                                               hw.descramLLRaddrsVec.data(),
                                               hw.d_pDTXflags,
                                               hw.d_pSinr,
                                               hw.d_pRssi,
                                               hw.d_pRsrp,
                                               hw.d_pInterf,
                                               hw.d_pNoiseVar,
                                               hw.d_pTaEst,
                                               nCells,
                                               nUcis,
                                               0,
                                               prms.data(),
                                               pCell,
                                               false,
                                               hw.dynDescrCpu,
                                               hw.dynDescrGpu,
                                               &hw.launchCfg,
                                               hw.stream);
    EXPECT_EQ(stSetup, CUPHY_STATUS_SUCCESS);

    const auto* dyn = reinterpret_cast<const pucchF2RxDynDescr_t*>(hw.dynDescrCpu);
    const uint16_t numUcisAfterClamp = dyn ? dyn->numUcis : 0;

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();

    return numUcisAfterClamp;
}

static size_t setupAndGetSharedMemBytes(const std::vector<cuphyPucchUciPrm_t>& ucis, int nRxAnt, int nScTotal, uint8_t enableUlRxBf)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = 1;
    const int nUcis  = static_cast<int>(ucis.size());

    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);

    // Avoid N cudaMalloc calls: allocate a single shared LLR buffer.
    size_t maxEseg1 = 0;
    for(const auto& u : ucis)
    {
        maxEseg1 = std::max<size_t>(maxEseg1, static_cast<size_t>(u.nSym) * u.prbSize * 16);
    }
    if(maxEseg1 == 0) maxEseg1 = 16;
    hw.sharedDescramLLR = DevAlloc::malloc_array<__half>(maxEseg1);
    for(int i = 0; i < nUcis; ++i)
    {
        hw.Eseg1Host[i]          = static_cast<uint16_t>(maxEseg1);
        hw.descramLLRaddrsVec[i] = hw.sharedDescramLLR;
    }

    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    pCell[0].nRxAnt         = static_cast<uint8_t>(nRxAnt);
    pCell[0].pucchHoppingId = 0;
    pCell[0].slotNum        = 0;

    cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    EXPECT_EQ(stCreate, CUPHY_STATUS_SUCCESS);

    size_t dynDescrSize = 0, dynDescrAlign = 0;
    cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    EXPECT_EQ(stInfo, CUPHY_STATUS_SUCCESS);

    dynDescrSize += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    if(hw.dynDescrCpu == nullptr)
    {
        ADD_FAILURE() << "Failed to allocate dyn descriptor host memory";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return 0;
    }
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));

    // Call setup (this includes the enableUlRxBf branch we want to cover).
    std::vector<cuphyPucchUciPrm_t> prms = ucis;
    cuphyStatus_t stSetup                = cuphySetupPucchF2Rx(hw.rxHndl,
                                               &hw.tPrmDataRx,
                                               hw.descramLLRaddrsVec.data(),
                                               hw.d_pDTXflags,
                                               hw.d_pSinr,
                                               hw.d_pRssi,
                                               hw.d_pRsrp,
                                               hw.d_pInterf,
                                               hw.d_pNoiseVar,
                                               hw.d_pTaEst,
                                               nCells,
                                               nUcis,
                                               enableUlRxBf,
                                               prms.data(),
                                               pCell,
                                               false,
                                               hw.dynDescrCpu,
                                               hw.dynDescrGpu,
                                               &hw.launchCfg,
                                               hw.stream);
    EXPECT_EQ(stSetup, CUPHY_STATUS_SUCCESS);

    // Sanity: descriptor reflects enable flag.
    const auto* dyn = reinterpret_cast<const pucchF2RxDynDescr_t*>(hw.dynDescrCpu);
    EXPECT_EQ(dyn ? dyn->enableUlRxBf : 0, enableUlRxBf);

    const size_t sharedMemBytes = hw.launchCfg.kernelNodeParamsDriver.sharedMemBytes;

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();

    return sharedMemBytes;
}

static size_t setupAndGetSharedMemBytesMultiCell(const std::vector<cuphyPucchUciPrm_t>& ucis,
                                                 const std::vector<uint8_t>&          cellRxAnts,
                                                 int                                  nScTotal,
                                                 uint8_t                              enableUlRxBf)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = static_cast<int>(cellRxAnts.size());
    const int nUcis  = static_cast<int>(ucis.size());

    // Create a single slot buffer; all cells share it for this setup-only test.
    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, /*nRxAnt=*/1);
    hw.allocOutputs(nUcis);

    // Avoid N cudaMalloc calls: allocate a single shared LLR buffer.
    size_t maxEseg1 = 0;
    for(const auto& u : ucis)
    {
        maxEseg1 = std::max<size_t>(maxEseg1, static_cast<size_t>(u.nSym) * u.prbSize * 16);
    }
    if(maxEseg1 == 0) maxEseg1 = 16;
    hw.sharedDescramLLR = DevAlloc::malloc_array<__half>(maxEseg1);
    for(int i = 0; i < nUcis; ++i)
    {
        hw.Eseg1Host[i]          = static_cast<uint16_t>(maxEseg1);
        hw.descramLLRaddrsVec[i] = hw.sharedDescramLLR;
    }

    // Cell params (UM so device can dereference if needed)
    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    for(int i = 0; i < nCells; ++i)
    {
        pCell[i].nRxAnt         = cellRxAnts[i];
        pCell[i].pucchHoppingId = 0;
        pCell[i].slotNum        = 0;
    }

    // Provide an array of tensor params as expected by setup when nCells>1.
    std::vector<cuphyTensorPrm_t> dataRx(static_cast<size_t>(nCells));
    for(int i = 0; i < nCells; ++i)
    {
        dataRx[static_cast<size_t>(i)] = hw.tPrmDataRx;
    }

    cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    EXPECT_EQ(stCreate, CUPHY_STATUS_SUCCESS);

    size_t dynDescrSize = 0, dynDescrAlign = 0;
    cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    EXPECT_EQ(stInfo, CUPHY_STATUS_SUCCESS);

    dynDescrSize += static_cast<size_t>(nCells) * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    if(hw.dynDescrCpu == nullptr)
    {
        ADD_FAILURE() << "Failed to allocate dyn descriptor host memory";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return 0;
    }
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));

    std::vector<cuphyPucchUciPrm_t> prms = ucis;
    cuphyStatus_t stSetup                = cuphySetupPucchF2Rx(hw.rxHndl,
                                               dataRx.data(),
                                               hw.descramLLRaddrsVec.data(),
                                               hw.d_pDTXflags,
                                               hw.d_pSinr,
                                               hw.d_pRssi,
                                               hw.d_pRsrp,
                                               hw.d_pInterf,
                                               hw.d_pNoiseVar,
                                               hw.d_pTaEst,
                                               static_cast<uint16_t>(nCells),
                                               static_cast<uint16_t>(nUcis),
                                               enableUlRxBf,
                                               prms.data(),
                                               pCell,
                                               false,
                                               hw.dynDescrCpu,
                                               hw.dynDescrGpu,
                                               &hw.launchCfg,
                                               hw.stream);
    EXPECT_EQ(stSetup, CUPHY_STATUS_SUCCESS);

    const size_t sharedMemBytes = hw.launchCfg.kernelNodeParamsDriver.sharedMemBytes;

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();

    return sharedMemBytes;
}

static bool setupAndVerifyDescrAsyncCopy(const cuphyPucchUciPrm_t& prm, int nRxAnt, int nScTotal)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = 1;
    const int nUcis  = 1;

    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);

    // Minimal single-UCI output buffers
    const uint16_t Eseg1 = static_cast<uint16_t>(prm.nSym * prm.prbSize * 16);
    hw.sharedDescramLLR  = DevAlloc::malloc_array<__half>(std::max<uint16_t>(Eseg1, 16));
    hw.Eseg1Host[0]      = Eseg1;
    hw.descramLLRaddrsVec[0] = hw.sharedDescramLLR;

    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    pCell[0].nRxAnt         = static_cast<uint8_t>(nRxAnt);
    pCell[0].pucchHoppingId = 0;
    pCell[0].slotNum        = 0;

    const cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    if(stCreate != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyCreatePucchF2Rx failed";
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return false;
    }

    size_t dynDescrSize = 0, dynDescrAlign = 0;
    const cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    if(stInfo != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyPucchF2RxGetDescrInfo failed";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return false;
    }

    dynDescrSize += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    if(hw.dynDescrCpu == nullptr)
    {
        ADD_FAILURE() << "Failed to allocate dyn descriptor host memory";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return false;
    }
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));

    cuphyPucchUciPrm_t uci = prm;
    const cuphyStatus_t stSetup = cuphySetupPucchF2Rx(hw.rxHndl,
                                                      &hw.tPrmDataRx,
                                                      hw.descramLLRaddrsVec.data(),
                                                      hw.d_pDTXflags,
                                                      hw.d_pSinr,
                                                      hw.d_pRssi,
                                                      hw.d_pRsrp,
                                                      hw.d_pInterf,
                                                      hw.d_pNoiseVar,
                                                      hw.d_pTaEst,
                                                      nCells,
                                                      nUcis,
                                                      /*enableUlRxBf*/ 0,
                                                      &uci,
                                                      pCell,
                                                      /*enableCpuToGpuDescrAsyncCpy*/ true,
                                                      hw.dynDescrCpu,
                                                      hw.dynDescrGpu,
                                                      &hw.launchCfg,
                                                      hw.stream);
    if(stSetup != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphySetupPucchF2Rx failed";
        cuphyDestroyPucchF2Rx(hw.rxHndl);
        if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
        if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
        if(pCell) cudaFree(pCell);
        hw.freeOutputs();
        hw.destroyTensor();
        hw.destroyStream();
        return false;
    }

    // Ensure the async copy (inside setup) has completed.
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));

    const auto* hostDesc = reinterpret_cast<const pucchF2RxDynDescr_t*>(hw.dynDescrCpu);
    pucchF2RxDynDescr_t gpuDesc{};
    CUDA_CHECK(cudaMemcpy(&gpuDesc, hw.dynDescrGpu, sizeof(gpuDesc), cudaMemcpyDeviceToHost));

    const bool ok = (hostDesc != nullptr) && (gpuDesc.numUcis == hostDesc->numUcis) &&
                    (gpuDesc.enableUlRxBf == hostDesc->enableUlRxBf) &&
                    (gpuDesc.uciPrms[0].prbSize == hostDesc->uciPrms[0].prbSize) &&
                    (gpuDesc.uciPrms[0].nSym == hostDesc->uciPrms[0].nSym);

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();

    return ok;
}

static RunResult runScenarioWithUlRxBf(const std::vector<cuphyPucchUciPrm_t>& ucis, int nRxAnt, int nScTotal, uint8_t enableUlRxBf)
{
    PucchF2Synthetic hw;
    hw.initStream();

    const int nCells = 1;
    const int nUcis  = static_cast<int>(ucis.size());

    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);

    // Copy UCIs and prepare output buffers
    std::vector<cuphyPucchUciPrm_t> prms = ucis;
    for(int i = 0; i < nUcis; ++i)
    {
        hw.Eseg1Host[i]          = static_cast<uint16_t>(prms[i].nSym * prms[i].prbSize * 16);
        hw.descramLLRaddrsVec[i] = DevAlloc::malloc_array<__half>(hw.Eseg1Host[i]);
    }

    // Unified memory for cell params so device can dereference
    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, nCells * sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, nCells * sizeof(cuphyPucchCellPrm_t));
    pCell[0].nRxAnt         = static_cast<uint8_t>(nRxAnt);
    pCell[0].pucchHoppingId = 0;
    pCell[0].slotNum        = 0;

    // Receiver + descriptor
    RunResult     out; // prepare result holder
    cuphyStatus_t stCreate = cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    if(stCreate != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyCreatePucchF2Rx failed";
        return out;
    }
    size_t        dynDescrSize = 0, dynDescrAlign = 0;
    cuphyStatus_t stInfo = cuphyPucchF2RxGetDescrInfo(&dynDescrSize, &dynDescrAlign);
    if(stInfo != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphyPucchF2RxGetDescrInfo failed";
        return out;
    }
    dynDescrSize += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynDescrSize);
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynDescrSize));
    cuphyStatus_t stSetup = cuphySetupPucchF2Rx(hw.rxHndl,
                                                &hw.tPrmDataRx,
                                                hw.descramLLRaddrsVec.data(),
                                                hw.d_pDTXflags,
                                                hw.d_pSinr,
                                                hw.d_pRssi,
                                                hw.d_pRsrp,
                                                hw.d_pInterf,
                                                hw.d_pNoiseVar,
                                                hw.d_pTaEst,
                                                nCells,
                                                nUcis,
                                                enableUlRxBf,
                                                prms.data(),
                                                pCell,
                                                false,
                                                hw.dynDescrCpu,
                                                hw.dynDescrGpu,
                                                &hw.launchCfg,
                                                hw.stream);
    if(stSetup != CUPHY_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "cuphySetupPucchF2Rx failed";
        return out;
    }
    CUDA_CHECK(cudaMemcpyAsync(hw.dynDescrGpu, hw.dynDescrCpu, dynDescrSize, cudaMemcpyHostToDevice, hw.stream));
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));

    const CUDA_KERNEL_NODE_PARAMS& p = hw.launchCfg.kernelNodeParamsDriver;
    {
        CUresult lres = cuLaunchKernel(p.func, p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY, p.blockDimZ, p.sharedMemBytes, static_cast<CUstream>(hw.stream), p.kernelParams, p.extra);
        if(lres != CUDA_SUCCESS)
        {
            ADD_FAILURE() << "cuLaunchKernel failed";
            return out;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));

    out.dtx.resize(nUcis);
    out.sinr.resize(nUcis);
    out.rssi.resize(nUcis);
    out.rsrp.resize(nUcis);
    out.interf.resize(nUcis);
    CUDA_CHECK(cudaMemcpy(out.dtx.data(), hw.d_pDTXflags, nUcis * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.sinr.data(), hw.d_pSinr, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rssi.data(), hw.d_pRssi, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.rsrp.data(), hw.d_pRsrp, nUcis * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.interf.data(), hw.d_pInterf, nUcis * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();
    return out;
}
} // namespace

// Helper to build a UCI parameter struct with sensible defaults
namespace
{
static cuphyPucchUciPrm_t makeUciPrm(
    uint16_t rnti,
    int      startPrb,
    int      prbSize,
    int      startSym,
    int      nSym,
    int      freqHopFlag,
    int      secondHopPrb,
    int      dataScramblingId,
    int      dmrsScramblingId,
    float    dtxThreshold,
    int      uciOutputIdx   = 0,
    int      cellPrmDynIdx  = 0,
    int      cellPrmStatIdx = 0,
    int      bwpStart       = 0)
{
    cuphyPucchUciPrm_t prm{};
    prm.cellPrmDynIdx    = cellPrmDynIdx;
    prm.cellPrmStatIdx   = cellPrmStatIdx;
    prm.uciOutputIdx     = uciOutputIdx;
    prm.rnti             = rnti;
    prm.bwpStart         = bwpStart;
    prm.startPrb         = startPrb;
    prm.prbSize          = prbSize;
    prm.startSym         = startSym;
    prm.nSym             = nSym;
    prm.freqHopFlag      = freqHopFlag;
    prm.secondHopPrb     = secondHopPrb;
    prm.dataScramblingId = dataScramblingId;
    prm.DmrsScramblingId = dmrsScramblingId;
    prm.DTXthreshold     = dtxThreshold;
    return prm;
}
} // namespace

// Forward declaration for LLR energy validation helper used by tests below
namespace
{
static void expectLLREnergyPositive(const cuphyPucchUciPrm_t& prm, int nRxAnt, int nScTotal, double minEnergy = 1e-6);
}

// Tests refactored to use the helper
// W2 branch
TEST(PucchF2FrontEnd, Synthetic_prbSize_2_W2_Branch)
{
    const int          nRxAnt = 1, nScTotal = 12 * 8;
    cuphyPucchUciPrm_t prm = makeUciPrm(
        /*rnti*/ 0x1111,
        /*startPrb*/ 0,
        /*prbSize*/ 2,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 3,
        /*dmrsScramblingId*/ 4,
        /*dtxThreshold*/ -100.0f);
    RunResult res = runScenario({prm}, nRxAnt, nScTotal);
    ASSERT_EQ(res.dtx.size(), 1u);
    EXPECT_EQ(res.dtx[0], 0);
    expectLLREnergyPositive(prm, nRxAnt, nScTotal);
}

// W3 branch
TEST(PucchF2FrontEnd, Synthetic_prbSize_3_W3_Branch)
{
    const int          nRxAnt = 1, nScTotal = 12 * 12;
    cuphyPucchUciPrm_t prm = makeUciPrm(
        /*rnti*/ 0x2222,
        /*startPrb*/ 0,
        /*prbSize*/ 3,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 7,
        /*dmrsScramblingId*/ 8,
        /*dtxThreshold*/ -100.0f);
    RunResult res = runScenario({prm}, nRxAnt, nScTotal);
    ASSERT_EQ(res.dtx.size(), 1u);
    EXPECT_EQ(res.dtx[0], 0);
    expectLLREnergyPositive(prm, nRxAnt, nScTotal);
}

TEST(PucchF2FrontEnd, Setup_Clamps_NumUcis_Above_Max)
{
    const int          nRxAnt = 1, nScTotal = 12 * 8;
    cuphyPucchUciPrm_t base   = makeUciPrm(
        /*rnti*/ 0x7777,
        /*startPrb*/ 0,
        /*prbSize*/ 1,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f);

    const uint16_t requestedUcis = static_cast<uint16_t>(CUPHY_PUCCH_F2_MAX_UCI + 1);
    std::vector<cuphyPucchUciPrm_t> prms(requestedUcis, base);
    for(size_t i = 0; i < prms.size(); ++i)
    {
        prms[i].uciOutputIdx = static_cast<uint16_t>(i);
    }

    const uint16_t numUcisAfterClamp = setupAndGetNumUcisAfterClamp(prms, nRxAnt, nScTotal);
    EXPECT_EQ(numUcisAfterClamp, static_cast<uint16_t>(CUPHY_PUCCH_F2_MAX_UCI));
}

TEST(PucchF2FrontEnd, Setup_Uses_UplinkStreams_When_UlRxBf_Enabled)
{
    const int          nRxAnt = 1, nScTotal = 12 * 20;
    cuphyPucchUciPrm_t prm    = makeUciPrm(
        /*rnti*/ 0x8888,
        /*startPrb*/ 0,
        /*prbSize*/ 5,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f);
    prm.nUplinkStreams = 4; // > cell nRxAnt (1) so BF path should increase maxNumRxAnt

    const size_t smemNoBf = setupAndGetSharedMemBytes({prm}, nRxAnt, nScTotal, /*enableUlRxBf*/ 0);
    const size_t smemBf   = setupAndGetSharedMemBytes({prm}, nRxAnt, nScTotal, /*enableUlRxBf*/ 1);

    EXPECT_GT(smemNoBf, 0u);
    EXPECT_GT(smemBf, 0u);
    EXPECT_GT(smemBf, smemNoBf); // BF path uses nUplinkStreams when sizing shared memory
}

TEST(PucchF2FrontEnd, Setup_MaxPrbSize_And_MaxNumSyms_Ternaries_Both_Paths)
{
    const int nRxAnt = 1;
    // Need enough subcarriers for the larger PRB size.
    const int nScTotal = 12 * 16;

    cuphyPucchUciPrm_t big = makeUciPrm(
        /*rnti*/ 0x1357,
        /*startPrb*/ 0,
        /*prbSize*/ 4,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f,
        /*uciOutputIdx*/ 0);

    cuphyPucchUciPrm_t small = makeUciPrm(
        /*rnti*/ 0x2468,
        /*startPrb*/ 0,
        /*prbSize*/ 1,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f,
        /*uciOutputIdx*/ 1);

    // Order matters:
    // - First UCI forces maxPrbSize/maxNumSyms to grow (false branch of ternary).
    // - Second UCI is smaller so (max > current) becomes true, exercising the true branch.
    const size_t smem = setupAndGetSharedMemBytes({big, small}, nRxAnt, nScTotal, /*enableUlRxBf*/ 0);
    EXPECT_GT(smem, 0u);
}

TEST(PucchF2FrontEnd, Setup_MaxNumRxAnt_Ternary_Both_Paths_NoBf_MultiCell)
{
    // Exercise both paths of:
    // maxNumRxAnt = (maxNumRxAnt > pCmnCellPrms[cellIdx].nRxAnt) ? maxNumRxAnt : pCmnCellPrms[cellIdx].nRxAnt;
    // by using 2 cells with different nRxAnt and 2 UCIs pointing at different cells.
    const int nScTotal = 12 * 8;

    cuphyPucchUciPrm_t uciCell0 = makeUciPrm(
        /*rnti*/ 0xaaaa,
        /*startPrb*/ 0,
        /*prbSize*/ 2,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f,
        /*uciOutputIdx*/ 0,
        /*cellPrmDynIdx*/ 0);

    cuphyPucchUciPrm_t uciCell1 = makeUciPrm(
        /*rnti*/ 0xbbbb,
        /*startPrb*/ 0,
        /*prbSize*/ 1,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f,
        /*uciOutputIdx*/ 1,
        /*cellPrmDynIdx*/ 1);

    // Cell 0 has larger nRxAnt; Cell 1 is smaller so second update keeps the max (true branch).
    const size_t smem = setupAndGetSharedMemBytesMultiCell({uciCell0, uciCell1}, /*cellRxAnts*/ {4, 1}, nScTotal, /*enableUlRxBf*/ 0);
    EXPECT_GT(smem, 0u);
}

TEST(PucchF2FrontEnd, Setup_AsyncCpuToGpuDescrCopy_When_Enabled)
{
    const int          nRxAnt = 1, nScTotal = 12 * 8;
    cuphyPucchUciPrm_t prm    = makeUciPrm(
        /*rnti*/ 0x9999,
        /*startPrb*/ 0,
        /*prbSize*/ 2,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 3,
        /*dmrsScramblingId*/ 4,
        /*dtxThreshold*/ -100.0f);

    EXPECT_TRUE(setupAndVerifyDescrAsyncCopy(prm, nRxAnt, nScTotal));
}

TEST(PucchF2FrontEnd, Kernel_Uses_UplinkStreams_When_UlRxBf_Enabled)
{
    // Make nRxAnt match nUplinkStreams so the kernel doesn't index beyond the tensor's antenna dimension.
    const int          nRxAnt = 4, nScTotal = 12 * 8;
    cuphyPucchUciPrm_t prm    = makeUciPrm(
        /*rnti*/ 0xabcd,
        /*startPrb*/ 0,
        /*prbSize*/ 2,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 3,
        /*dmrsScramblingId*/ 4,
        /*dtxThreshold*/ -100.0f);
    prm.nUplinkStreams = static_cast<uint16_t>(nRxAnt);

    RunResult res = runScenarioWithUlRxBf({prm}, nRxAnt, nScTotal, /*enableUlRxBf*/ 1);
    ASSERT_EQ(res.dtx.size(), 1u);
    EXPECT_TRUE(std::isfinite(res.sinr[0]));
    EXPECT_TRUE(std::isfinite(res.rssi[0]));
    EXPECT_TRUE(std::isfinite(res.rsrp[0]));
    EXPECT_TRUE(std::isfinite(res.interf[0]));
}

TEST(PucchF2FrontEnd, Synthetic_NoHop_And_Validation)
{
    const int                       nRxAnt = 1, nScTotal = 12 * 16;
    std::vector<cuphyPucchUciPrm_t> prms(2);
    prms[0] = makeUciPrm(
        /*rnti*/ 0x1234,
        /*startPrb*/ 0,
        /*prbSize*/ 1,
        /*startSym*/ 0,
        /*nSym*/ 1,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ 100.0f,
        /*uciOutputIdx*/ 0);
    prms[1] = makeUciPrm(
        /*rnti*/ 0x1234,
        /*startPrb*/ 4,
        /*prbSize*/ 4,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 1,
        /*secondHopPrb*/ 6,
        /*dataScramblingId*/ 1,
        /*dmrsScramblingId*/ 2,
        /*dtxThreshold*/ -100.0f,
        /*uciOutputIdx*/ 1);
    RunResult res = runScenario(prms, nRxAnt, nScTotal);
    EXPECT_EQ(res.dtx[0], 0);
    EXPECT_EQ(res.dtx[1], 0);
    for(size_t i = 0; i < res.dtx.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(res.sinr[i]));
        EXPECT_TRUE(std::isfinite(res.rssi[i]));
        EXPECT_TRUE(std::isfinite(res.rsrp[i]));
        EXPECT_TRUE(std::isfinite(res.interf[i]));
    }
    // Validate LLRs for both UCIs
    expectLLREnergyPositive(prms[0], nRxAnt, nScTotal);
    expectLLREnergyPositive(prms[1], nRxAnt, nScTotal);
}

TEST(PucchF2FrontEnd, Synthetic_FreqHop_Branch_Coverage)
{
    const int          nRxAnt = 1, nScTotal = 12 * 8;
    cuphyPucchUciPrm_t prm = makeUciPrm(
        /*rnti*/ 0x55aa,
        /*startPrb*/ 2,
        /*prbSize*/ 4,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 1,
        /*secondHopPrb*/ 3,
        /*dataScramblingId*/ 5,
        /*dmrsScramblingId*/ 6,
        /*dtxThreshold*/ -100.0f);
    RunResult res = runScenario({prm}, nRxAnt, nScTotal);
    ASSERT_EQ(res.dtx.size(), 1u);
    EXPECT_EQ(res.dtx[0], 0);
    expectLLREnergyPositive(prm, nRxAnt, nScTotal);
}

// W4 partial-block branch: prbSize = 5 -> num_dmrs = 20, triggers the partial path
TEST(PucchF2FrontEnd, Synthetic_prbSize_5_W4_PartialBlock)
{
    const int          nRxAnt = 1, nScTotal = 12 * 20; // enough tones
    cuphyPucchUciPrm_t prm = makeUciPrm(
        /*rnti*/ 0x3333,
        /*startPrb*/ 0,
        /*prbSize*/ 5,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 9,
        /*dmrsScramblingId*/ 10,
        /*dtxThreshold*/ -100.0f);
    RunResult res = runScenario({prm}, nRxAnt, nScTotal);
    ASSERT_EQ(res.dtx.size(), 1u);
    EXPECT_EQ(res.dtx[0], 0);
    expectLLREnergyPositive(prm, nRxAnt, nScTotal);
}

// Helper to run a single-UE scenario and fetch LLRs for validation
namespace
{
static std::vector<__half> runAndFetchLLRs(const cuphyPucchUciPrm_t& prm, int nRxAnt, int nScTotal)
{
    PucchF2Synthetic hw;
    hw.initStream();
    const int nCells = 1, nUcis = 1;
    hw.createTensor3D_C16F(nScTotal, SYM_PER_SLOT, nRxAnt);
    hw.allocOutputs(nUcis);
    hw.Eseg1Host[0]            = static_cast<uint16_t>(prm.nSym * prm.prbSize * 16);
    hw.descramLLRaddrsVec[0]   = DevAlloc::malloc_array<__half>(hw.Eseg1Host[0]);
    cuphyPucchCellPrm_t* pCell = nullptr;
    CUDA_CHECK(cudaMallocManaged(&pCell, sizeof(cuphyPucchCellPrm_t)));
    std::memset(pCell, 0, sizeof(cuphyPucchCellPrm_t));
    pCell->nRxAnt         = nRxAnt;
    pCell->pucchHoppingId = 0;
    pCell->slotNum        = 0;
    cuphyCreatePucchF2Rx(&hw.rxHndl, hw.stream);
    size_t dynSz = 0, dynAl = 0;
    cuphyPucchF2RxGetDescrInfo(&dynSz, &dynAl);
    dynSz += nCells * sizeof(cuphyPucchCellPrm_t);
    hw.dynDescrCpu = std::malloc(dynSz);
    CUDA_CHECK(cudaMalloc(&hw.dynDescrGpu, dynSz));
    cuphySetupPucchF2Rx(hw.rxHndl, &hw.tPrmDataRx, hw.descramLLRaddrsVec.data(), hw.d_pDTXflags, hw.d_pSinr, hw.d_pRssi, hw.d_pRsrp, hw.d_pInterf, hw.d_pNoiseVar, hw.d_pTaEst, nCells, nUcis, 0, const_cast<cuphyPucchUciPrm_t*>(&prm), pCell, false, hw.dynDescrCpu, hw.dynDescrGpu, &hw.launchCfg, hw.stream);
    CUDA_CHECK(cudaMemcpyAsync(hw.dynDescrGpu, hw.dynDescrCpu, dynSz, cudaMemcpyHostToDevice, hw.stream));
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));
    const CUDA_KERNEL_NODE_PARAMS& p = hw.launchCfg.kernelNodeParamsDriver;
    cuLaunchKernel(p.func, p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY, p.blockDimZ, p.sharedMemBytes, static_cast<CUstream>(hw.stream), p.kernelParams, p.extra);
    CUDA_CHECK(cudaStreamSynchronize(hw.stream));
    std::vector<__half> llrs(hw.Eseg1Host[0]);
    CUDA_CHECK(cudaMemcpy(llrs.data(), hw.descramLLRaddrsVec[0], hw.Eseg1Host[0] * sizeof(__half), cudaMemcpyDeviceToHost));
    cuphyDestroyPucchF2Rx(hw.rxHndl);
    if(hw.dynDescrCpu) std::free(hw.dynDescrCpu);
    if(hw.dynDescrGpu) cudaFree(hw.dynDescrGpu);
    if(pCell) cudaFree(pCell);
    hw.freeOutputs();
    hw.destroyTensor();
    hw.destroyStream();
    return llrs;
}
} // namespace

// LLR output validation helper (single definition)
namespace
{
static void expectLLREnergyPositive(const cuphyPucchUciPrm_t& prm, int nRxAnt, int nScTotal, double minEnergy)
{
    std::vector<__half> llrs = runAndFetchLLRs(prm, nRxAnt, nScTotal);
    ASSERT_FALSE(llrs.empty());
    double energy = 0.0;
    for(size_t i = 0; i < llrs.size(); ++i)
    {
        float v = __half2float(llrs[i]);
        energy += static_cast<double>(v) * static_cast<double>(v);
    }
    EXPECT_GT(energy, minEnergy);
}
} // namespace

TEST(PucchF2FrontEnd, Synthetic_OutputLLR_Validation)
{
    const int          nRxAnt = 1, nScTotal = 12 * 16;
    cuphyPucchUciPrm_t prm = makeUciPrm(
        /*rnti*/ 0x4444,
        /*startPrb*/ 4,
        /*prbSize*/ 4,
        /*startSym*/ 0,
        /*nSym*/ 2,
        /*freqHopFlag*/ 0,
        /*secondHopPrb*/ 0,
        /*dataScramblingId*/ 11,
        /*dmrsScramblingId*/ 12,
        /*dtxThreshold*/ -100.0f);
    std::vector<__half> llrs = runAndFetchLLRs(prm, nRxAnt, nScTotal);
    ASSERT_FALSE(llrs.empty());
    // Compute simple energy metric and ensure > small epsilon
    double energy = 0.0;
    for(size_t i = 0; i < llrs.size(); ++i)
    {
        float v = __half2float(llrs[i]);
        energy += static_cast<double>(v) * static_cast<double>(v);
    }
    EXPECT_GT(energy, 1e-6);
}

TEST(PucchF2FrontEnd, Synthetic_OutputLLR_Validation_All)
{
    const int nRxAnt = 1;

    // W1
    {
        int                nScTotal = 12 * 8;
        cuphyPucchUciPrm_t prm      = makeUciPrm(
            /*rnti*/ 0x5001,
            /*startPrb*/ 0,
            /*prbSize*/ 1,
            /*startSym*/ 0,
            /*nSym*/ 1,
            /*freqHopFlag*/ 0,
            /*secondHopPrb*/ 0,
            /*dataScramblingId*/ 21,
            /*dmrsScramblingId*/ 22,
            /*dtxThreshold*/ -100.0f);
        expectLLREnergyPositive(prm, nRxAnt, nScTotal);
    }

    // W2
    {
        int                nScTotal = 12 * 8;
        cuphyPucchUciPrm_t prm      = makeUciPrm(
            /*rnti*/ 0x5002,
            /*startPrb*/ 0,
            /*prbSize*/ 2,
            /*startSym*/ 0,
            /*nSym*/ 1,
            /*freqHopFlag*/ 0,
            /*secondHopPrb*/ 0,
            /*dataScramblingId*/ 23,
            /*dmrsScramblingId*/ 24,
            /*dtxThreshold*/ -100.0f);
        expectLLREnergyPositive(prm, nRxAnt, nScTotal);
    }

    // W3
    {
        int                nScTotal = 12 * 12;
        cuphyPucchUciPrm_t prm      = makeUciPrm(
            /*rnti*/ 0x5003,
            /*startPrb*/ 0,
            /*prbSize*/ 3,
            /*startSym*/ 0,
            /*nSym*/ 2,
            /*freqHopFlag*/ 0,
            /*secondHopPrb*/ 0,
            /*dataScramblingId*/ 25,
            /*dmrsScramblingId*/ 26,
            /*dtxThreshold*/ -100.0f);
        expectLLREnergyPositive(prm, nRxAnt, nScTotal);
    }

    // W4 full
    {
        int                nScTotal = 12 * 16;
        cuphyPucchUciPrm_t prm      = makeUciPrm(
            /*rnti*/ 0x5004,
            /*startPrb*/ 4,
            /*prbSize*/ 4,
            /*startSym*/ 0,
            /*nSym*/ 2,
            /*freqHopFlag*/ 0,
            /*secondHopPrb*/ 0,
            /*dataScramblingId*/ 27,
            /*dmrsScramblingId*/ 28,
            /*dtxThreshold*/ -100.0f);
        expectLLREnergyPositive(prm, nRxAnt, nScTotal);
    }

    // W4 partial
    {
        int                nScTotal = 12 * 20;
        cuphyPucchUciPrm_t prm      = makeUciPrm(
            /*rnti*/ 0x5005,
            /*startPrb*/ 0,
            /*prbSize*/ 5,
            /*startSym*/ 0,
            /*nSym*/ 2,
            /*freqHopFlag*/ 0,
            /*secondHopPrb*/ 0,
            /*dataScramblingId*/ 29,
            /*dmrsScramblingId*/ 30,
            /*dtxThreshold*/ -100.0f);
        expectLLREnergyPositive(prm, nRxAnt, nScTotal);
    }
}

int main(int argc, char* argv[])
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    return result;
}
