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

#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "4T4R/multiCellScheduler.cuh"
#include "4T4R/multiCellSchedulerCpu.h"


using namespace cumac;

namespace {

// RAII helper that silences stdout for the duration of its scope.
// Uses dup/dup2 so it works without a controlling terminal.
class ScopedStdoutSilencer {
public:
    ScopedStdoutSilencer() {
        std::fflush(stdout);
        savedFd_ = ::dup(STDOUT_FILENO);
        const int devnullFd = ::open("/dev/null", O_WRONLY);
        if (savedFd_ >= 0 && devnullFd >= 0) {
            ::dup2(devnullFd, STDOUT_FILENO);
            active_ = true;
        }
        if (devnullFd >= 0) ::close(devnullFd);
    }
    ~ScopedStdoutSilencer() {
        if (active_) {
            std::fflush(stdout);
            ::dup2(savedFd_, STDOUT_FILENO);
            std::clearerr(stdout);
        }
        if (savedFd_ >= 0) ::close(savedFd_);
    }
    ScopedStdoutSilencer(const ScopedStdoutSilencer&)            = delete;
    ScopedStdoutSilencer& operator=(const ScopedStdoutSilencer&) = delete;
private:
    int  savedFd_ = -1;
    bool active_  = false;
};

// Test fixture providing minimal GPU buffers and descriptors for the
// scheduler. Success-path tests assert that the launch and stream
// synchronize succeed.
class MultiCellSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        int devCount = 0;
        if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override
    {
        FreeAll();
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Allocate every buffer the scheduler may touch. Values are
    // zero-filled except for the channel buffer (deterministic random).
    void BuildPrms(uint16_t nCell,
                   uint16_t nUe,
                   uint16_t nPrbGrp,
                   uint8_t  nBsAnt,
                   uint8_t  nUeAnt,
                   uint8_t  numUeSchdPerCellTTI,
                   bool     enableHarq,
                   bool     dlInd)
    {
        FreeAll();

        nCell_                = nCell;
        nUe_                  = nUe;
        nPrbGrp_              = nPrbGrp;
        nBsAnt_               = nBsAnt;
        nUeAnt_               = nUeAnt;
        numUeSchdPerCellTTI_  = numUeSchdPerCellTTI;
        totNumCell_           = nCell;

        const size_t nChan = static_cast<size_t>(nPrbGrp) * nUe * totNumCell_
                                                 * nBsAnt * nUeAnt;
        const size_t nAssoc      = static_cast<size_t>(nCell) * nUe;
        const size_t nAllocSol   = static_cast<size_t>(nCell) * nPrbGrp;
        const size_t nPostEqSinr = static_cast<size_t>(nUe) * nPrbGrp * nUeAnt;
        const size_t nSinVal     = nPostEqSinr;
        // Conservative upper bound so the buffer is valid across all
        // dispatch modes.
        const size_t nPfArr      = static_cast<size_t>(nCell) * nPrbGrp * nUe;
        const size_t nPrdMat     = static_cast<size_t>(nCell) * nPrbGrp * nBsAnt * nBsAnt;
        const size_t nDetMat     = nPrdMat;

        h_estH_fr_.resize(nChan);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& c : h_estH_fr_) { c.x = dist(rng); c.y = dist(rng); }

        // Round-robin: associate each UE to one cell.
        h_cellAssoc_.assign(nAssoc, 0);
        for (uint16_t u = 0; u < nUe; ++u) {
            const uint16_t c = static_cast<uint16_t>(u % nCell);
            h_cellAssoc_[c * nUe + u] = 1;
        }

        h_cellId_.resize(nCell);
        for (uint16_t c = 0; c < nCell; ++c) h_cellId_[c] = c;

        h_avgRates_.assign(nUe, 1.0f);
        h_setSchdUe_.assign(static_cast<size_t>(nCell) * numUeSchdPerCellTTI, 0xFFFF);
        for (uint16_t c = 0; c < nCell; ++c) {
            for (uint8_t i = 0; i < numUeSchdPerCellTTI; ++i) {
                const uint16_t uIdx = static_cast<uint16_t>(c * numUeSchdPerCellTTI + i);
                if (uIdx < nUe) {
                    h_setSchdUe_[c * numUeSchdPerCellTTI + i] = uIdx;
                }
            }
        }

        ASSERT_EQ(cudaMalloc(&d_estH_fr_,      nChan      * sizeof(cuComplex)),       cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_estH_fr_half_, nChan      * sizeof(__nv_bfloat162)),  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_cellAssoc_,    nAssoc     * sizeof(uint8_t)),         cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_cellId_,       nCell      * sizeof(uint16_t)),        cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_avgRates_,     nUe        * sizeof(float)),           cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_setSchdUe_,    h_setSchdUe_.size() * sizeof(uint16_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_postEqSinr_,   nPostEqSinr * sizeof(float)),          cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_sinVal_,       nSinVal    * sizeof(float)),           cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_allocSol_,     std::max<size_t>(nAllocSol, 2u * nUe) * sizeof(int16_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_pfMetricArr_,  nPfArr     * sizeof(float)),           cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_pfIdArr_,      nPfArr     * sizeof(uint16_t)),        cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_prdMat_,       nPrdMat    * sizeof(cuComplex)),       cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_detMat_,       nDetMat    * sizeof(cuComplex)),       cudaSuccess);

        ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_estH_fr_.data(),
                                  nChan * sizeof(cuComplex),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_estH_fr_half_, 0, nChan * sizeof(__nv_bfloat162), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, h_cellAssoc_.data(),
                                  nAssoc * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_cellId_, h_cellId_.data(),
                                  nCell * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, h_avgRates_.data(),
                                  nUe * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_setSchdUe_, h_setSchdUe_.data(),
                                  h_setSchdUe_.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_postEqSinr_, 0, nPostEqSinr * sizeof(float), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_sinVal_,     0, nSinVal     * sizeof(float), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_allocSol_,   0xFF, std::max<size_t>(nAllocSol, 2u * nUe) * sizeof(int16_t), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_pfMetricArr_, 0, nPfArr * sizeof(float), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_pfIdArr_,     0, nPfArr * sizeof(uint16_t), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_prdMat_, 0, nPrdMat * sizeof(cuComplex), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_detMat_, 0, nDetMat * sizeof(cuComplex), stream_), cudaSuccess);

        // GPU descriptor fields consumed by setup().
        cellGrpPrmsGpu_              = cumacCellGrpPrms{};
        cellGrpPrmsGpu_.dlSchInd     = dlInd ? 1 : 0;
        cellGrpPrmsGpu_.harqEnabledInd = enableHarq ? 1 : 0;
        cellGrpPrmsGpu_.nCell        = nCell;
        cellGrpPrmsGpu_.nUe          = nUe;
        cellGrpPrmsGpu_.nPrbGrp      = nPrbGrp;
        cellGrpPrmsGpu_.nBsAnt       = nBsAnt;
        cellGrpPrmsGpu_.nUeAnt       = nUeAnt;
        cellGrpPrmsGpu_.numUeSchdPerCellTTI = numUeSchdPerCellTTI;
        cellGrpPrmsGpu_.W            = 360e3f;
        cellGrpPrmsGpu_.sigmaSqrd    = 1e-6f;
        cellGrpPrmsGpu_.betaCoeff    = 1.0f;
        cellGrpPrmsGpu_.allocType    = 0;
        cellGrpPrmsGpu_.precodingScheme = 0;
        cellGrpPrmsGpu_.cellId       = d_cellId_;
        cellGrpPrmsGpu_.cellAssoc    = d_cellAssoc_;
        cellGrpPrmsGpu_.estH_fr      = d_estH_fr_;
        cellGrpPrmsGpu_.estH_fr_half = d_estH_fr_half_;
        cellGrpPrmsGpu_.prdMat       = d_prdMat_;
        cellGrpPrmsGpu_.detMat       = d_detMat_;
        cellGrpPrmsGpu_.sinVal       = d_sinVal_;
        cellGrpPrmsGpu_.postEqSinr   = d_postEqSinr_;

        schdSolGpu_                  = cumacSchdSol{};
        schdSolGpu_.allocSol         = d_allocSol_;
        schdSolGpu_.pfMetricArr      = d_pfMetricArr_;
        schdSolGpu_.pfIdArr          = d_pfIdArr_;
        schdSolGpu_.setSchdUePerCellTTI = d_setSchdUe_;

        ueStatusGpu_                 = cumacCellGrpUeStatus{};
        ueStatusGpu_.avgRates        = d_avgRates_;

        simParam_                    = cumacSimParam{};
        simParam_.totNumCell         = totNumCell_;
    }

    // HARQ-only device buffers.
    void BuildHarqBuffers()
    {
        const size_t nNew    = nUe_;
        const size_t nLastTx = static_cast<size_t>(nCell_) * nPrbGrp_;
        // PRG mask is byte-per-PRG; default to "available".
        const size_t nMaskBytes = static_cast<size_t>(nPrbGrp_) * sizeof(uint8_t);

        ASSERT_EQ(cudaMalloc(&d_newDataActUe_,   nNew * sizeof(int8_t)),    cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_allocSolLastTx_, nLastTx * sizeof(int16_t)), cudaSuccess);
        d_prgMskRows_.resize(nCell_);
        for (uint16_t c = 0; c < nCell_; ++c) {
            uint8_t* row = nullptr;
            ASSERT_EQ(cudaMalloc(&row, nMaskBytes), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(row, 0x01, nMaskBytes, stream_), cudaSuccess);
            d_prgMskRows_[c] = row;
        }
        ASSERT_EQ(cudaMalloc(&d_prgMsk_, nCell_ * sizeof(uint8_t*)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                                  nCell_ * sizeof(uint8_t*),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_newDataActUe_,   0, nNew * sizeof(int8_t)),   cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_allocSolLastTx_, 0xFF, nLastTx * sizeof(int16_t)), cudaSuccess);

        ueStatusGpu_.newDataActUe   = d_newDataActUe_;
        ueStatusGpu_.allocSolLastTx = d_allocSolLastTx_;
        cellGrpPrmsGpu_.prgMsk      = d_prgMsk_;
    }

    // Aerial-Sim-only buffers.
    void BuildAsimBuffers()
    {
        const size_t nChan    = static_cast<size_t>(nUe_) * nPrbGrp_ * nBsAnt_ * nUeAnt_;
        const size_t nPrdMat  = static_cast<size_t>(nUe_) * nPrbGrp_ * nBsAnt_ * nBsAnt_;
        const size_t nDetMat  = nPrdMat;
        const size_t nSinVal  = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
        const size_t nWbSinr  = static_cast<size_t>(nUe_) * nUeAnt_;

        d_srsEstChanRows_.resize(nCell_, nullptr);
        for (uint16_t c = 0; c < nCell_; ++c) {
            cuComplex* row = nullptr;
            ASSERT_EQ(cudaMalloc(&row, nChan * sizeof(cuComplex)), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(row, 0, nChan * sizeof(cuComplex), stream_), cudaSuccess);
            d_srsEstChanRows_[c] = row;
        }
        ASSERT_EQ(cudaMalloc(&d_srsEstChan_, nCell_ * sizeof(cuComplex*)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_srsEstChan_, d_srsEstChanRows_.data(),
                                  nCell_ * sizeof(cuComplex*),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);

        ASSERT_EQ(cudaMalloc(&d_prdMat_asim_, nPrdMat * sizeof(cuComplex)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_detMat_asim_, nDetMat * sizeof(cuComplex)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_sinVal_asim_, nSinVal * sizeof(float)),     cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_wbSinr_,      nWbSinr * sizeof(float)),     cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_prdMat_asim_, 0, nPrdMat * sizeof(cuComplex), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_detMat_asim_, 0, nDetMat * sizeof(cuComplex), stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_sinVal_asim_, 0, nSinVal * sizeof(float),     stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_wbSinr_,      0, nWbSinr * sizeof(float),     stream_), cudaSuccess);

        cellGrpPrmsGpu_.srsEstChan  = d_srsEstChan_;
        cellGrpPrmsGpu_.prdMat_asim = d_prdMat_asim_;
        cellGrpPrmsGpu_.detMat_asim = d_detMat_asim_;
        cellGrpPrmsGpu_.sinVal_asim = d_sinVal_asim_;
        cellGrpPrmsGpu_.wbSinr      = d_wbSinr_;
    }

    void FreeAll()
    {
        auto safeFree = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
        safeFree(reinterpret_cast<void*&>(d_estH_fr_));
        safeFree(reinterpret_cast<void*&>(d_estH_fr_half_));
        safeFree(reinterpret_cast<void*&>(d_cellAssoc_));
        safeFree(reinterpret_cast<void*&>(d_cellId_));
        safeFree(reinterpret_cast<void*&>(d_avgRates_));
        safeFree(reinterpret_cast<void*&>(d_setSchdUe_));
        safeFree(reinterpret_cast<void*&>(d_postEqSinr_));
        safeFree(reinterpret_cast<void*&>(d_sinVal_));
        safeFree(reinterpret_cast<void*&>(d_allocSol_));
        safeFree(reinterpret_cast<void*&>(d_pfMetricArr_));
        safeFree(reinterpret_cast<void*&>(d_pfIdArr_));
        safeFree(reinterpret_cast<void*&>(d_prdMat_));
        safeFree(reinterpret_cast<void*&>(d_detMat_));
        safeFree(reinterpret_cast<void*&>(d_newDataActUe_));
        safeFree(reinterpret_cast<void*&>(d_allocSolLastTx_));
        safeFree(reinterpret_cast<void*&>(d_prgMsk_));
        for (auto*& row : d_prgMskRows_) safeFree(reinterpret_cast<void*&>(row));
        d_prgMskRows_.clear();
        safeFree(reinterpret_cast<void*&>(d_srsEstChan_));
        for (auto*& row : d_srsEstChanRows_) safeFree(reinterpret_cast<void*&>(row));
        d_srsEstChanRows_.clear();
        safeFree(reinterpret_cast<void*&>(d_prdMat_asim_));
        safeFree(reinterpret_cast<void*&>(d_detMat_asim_));
        safeFree(reinterpret_cast<void*&>(d_sinVal_asim_));
        safeFree(reinterpret_cast<void*&>(d_wbSinr_));
    }

    // Members ---------------------------------------------------------------
    cudaStream_t stream_ = nullptr;

    // host-side scratch
    std::vector<cuComplex> h_estH_fr_;
    std::vector<uint8_t>   h_cellAssoc_;
    std::vector<uint16_t>  h_cellId_;
    std::vector<float>     h_avgRates_;
    std::vector<uint16_t>  h_setSchdUe_;

    // device-side scratch
    cuComplex*       d_estH_fr_      = nullptr;
    __nv_bfloat162*  d_estH_fr_half_ = nullptr;
    uint8_t*         d_cellAssoc_    = nullptr;
    uint16_t*        d_cellId_       = nullptr;
    float*           d_avgRates_     = nullptr;
    uint16_t*        d_setSchdUe_    = nullptr;
    float*           d_postEqSinr_   = nullptr;
    float*           d_sinVal_       = nullptr;
    int16_t*         d_allocSol_     = nullptr;
    float*           d_pfMetricArr_  = nullptr;
    uint16_t*        d_pfIdArr_      = nullptr;
    cuComplex*       d_prdMat_       = nullptr;
    cuComplex*       d_detMat_       = nullptr;

    // HARQ
    int8_t*          d_newDataActUe_   = nullptr;
    int16_t*         d_allocSolLastTx_ = nullptr;
    uint8_t**        d_prgMsk_         = nullptr;
    std::vector<uint8_t*> d_prgMskRows_;

    // Aerial Sim
    cuComplex**      d_srsEstChan_   = nullptr;
    std::vector<cuComplex*> d_srsEstChanRows_;
    cuComplex*       d_prdMat_asim_  = nullptr;
    cuComplex*       d_detMat_asim_  = nullptr;
    float*           d_sinVal_asim_  = nullptr;
    float*           d_wbSinr_       = nullptr;

    // descriptor structs
    cumacCellGrpPrms     cellGrpPrmsGpu_{};
    cumacSchdSol         schdSolGpu_{};
    cumacCellGrpUeStatus ueStatusGpu_{};
    cumacSimParam        simParam_{};

    // CPU reference mirrors used for cross-validation.
    std::vector<int16_t>   h_allocSolGpu_;
    std::vector<int16_t>   h_allocSolCpu_;
    std::vector<float>     h_postEqSinrCpu_;
    std::vector<cuComplex> h_prdMatHost_;
    std::vector<cuComplex> h_estHHost_;
    std::vector<float>     h_sinValHost_;
    cumacCellGrpPrms       cellGrpPrmsCpu_{};
    cumacSchdSol           schdSolCpu_{};
    cumacCellGrpUeStatus   ueStatusCpu_{};

    // dim cache
    uint16_t nCell_   = 0;
    uint16_t nUe_     = 0;
    uint16_t nPrbGrp_ = 0;
    uint8_t  nBsAnt_  = 0;
    uint8_t  nUeAnt_  = 0;
    uint8_t  numUeSchdPerCellTTI_ = 0;
    uint16_t totNumCell_ = 0;

    // Comparison strictness for the CPU/GPU cross-check.
    //  - Strict        : per-entry equality between CPU and GPU. Use when
    //                    the matrix algebra is well-conditioned in fp32.
    //  - InvariantOnly : skip per-entry equality and only check structural
    //                    invariants (chosen UE is cell-associated; type-1
    //                    ranges are in-bounds and non-overlapping). Use
    //                    when fp drift between CPU and GPU can swing
    //                    argmax to a different UE.
    enum class CmpMode { Strict, InvariantOnly };

    // Runs the CPU reference scheduler with host mirrors of the current
    // device inputs and validates the GPU result per `mode`. Only call
    // from variants that have a matching CPU implementation.
    void ValidateAgainstCpuRef(uint8_t columnMajor,
                               CmpMode mode = CmpMode::Strict)
    {
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        const size_t nAllocSolBuf = std::max<size_t>(
            static_cast<size_t>(nCell_) * nPrbGrp_, 2u * nUe_);
        const size_t nChan        = static_cast<size_t>(nPrbGrp_) * nUe_
                                       * totNumCell_ * nBsAnt_ * nUeAnt_;
        const size_t nPostEqSinr  = static_cast<size_t>(nUe_) * nPrbGrp_
                                       * nUeAnt_;
        const size_t nPrdMat      = static_cast<size_t>(nUe_) * nPrbGrp_
                                       * nBsAnt_ * nBsAnt_;
        const size_t nSinVal      = nPostEqSinr;

        const bool isUL = (cellGrpPrmsGpu_.dlSchInd == 0);

        // Pull current device inputs back so the CPU sees the same bytes
        // the GPU just consumed (tests may overwrite inputs after setup).
        h_estHHost_.assign(nChan, cuComplex{0.f, 0.f});
        if (!isUL) {
            ASSERT_EQ(cudaMemcpy(h_estHHost_.data(), d_estH_fr_,
                                 nChan * sizeof(cuComplex),
                                 cudaMemcpyDeviceToHost), cudaSuccess);
        }
        h_sinValHost_.assign(nSinVal, 0.0f);
        if (isUL) {
            ASSERT_EQ(cudaMemcpy(h_sinValHost_.data(), d_sinVal_,
                                 nSinVal * sizeof(float),
                                 cudaMemcpyDeviceToHost), cudaSuccess);
        }
        ASSERT_EQ(cudaMemcpy(h_avgRates_.data(), d_avgRates_,
                             nUe_ * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(h_cellAssoc_.data(), d_cellAssoc_,
                             static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        h_allocSolGpu_.assign(nAllocSolBuf, -1);
        ASSERT_EQ(cudaMemcpy(h_allocSolGpu_.data(), d_allocSol_,
                             nAllocSolBuf * sizeof(int16_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        // CPU-side scratch.
        h_allocSolCpu_.assign(nAllocSolBuf, -1);
        h_postEqSinrCpu_.assign(nPostEqSinr, 0.0f);
        h_prdMatHost_.assign(nPrdMat, cuComplex{0.f, 0.f});

        cellGrpPrmsCpu_ = cumacCellGrpPrms{};
        cellGrpPrmsCpu_.dlSchInd            = cellGrpPrmsGpu_.dlSchInd;
        cellGrpPrmsCpu_.harqEnabledInd      = cellGrpPrmsGpu_.harqEnabledInd;
        cellGrpPrmsCpu_.nCell               = nCell_;
        cellGrpPrmsCpu_.nUe                 = nUe_;
        cellGrpPrmsCpu_.nPrbGrp             = nPrbGrp_;
        cellGrpPrmsCpu_.nBsAnt              = nBsAnt_;
        cellGrpPrmsCpu_.nUeAnt              = nUeAnt_;
        cellGrpPrmsCpu_.numUeSchdPerCellTTI = numUeSchdPerCellTTI_;
        cellGrpPrmsCpu_.W                   = cellGrpPrmsGpu_.W;
        cellGrpPrmsCpu_.sigmaSqrd           = cellGrpPrmsGpu_.sigmaSqrd;
        cellGrpPrmsCpu_.betaCoeff           = cellGrpPrmsGpu_.betaCoeff;
        cellGrpPrmsCpu_.allocType           = cellGrpPrmsGpu_.allocType;
        cellGrpPrmsCpu_.precodingScheme     = cellGrpPrmsGpu_.precodingScheme;
        cellGrpPrmsCpu_.cellId              = h_cellId_.data();
        cellGrpPrmsCpu_.cellAssoc           = h_cellAssoc_.data();
        cellGrpPrmsCpu_.estH_fr             = isUL ? nullptr : h_estHHost_.data();
        cellGrpPrmsCpu_.prdMat              = isUL ? nullptr : h_prdMatHost_.data();
        cellGrpPrmsCpu_.sinVal              = isUL ? h_sinValHost_.data() : nullptr;
        cellGrpPrmsCpu_.postEqSinr          = h_postEqSinrCpu_.data();

        schdSolCpu_ = cumacSchdSol{};
        schdSolCpu_.allocSol                = h_allocSolCpu_.data();
        schdSolCpu_.setSchdUePerCellTTI     = h_setSchdUe_.data();

        ueStatusCpu_ = cumacCellGrpUeStatus{};
        ueStatusCpu_.avgRates               = h_avgRates_.data();

        cumacSimParam simParamCpu{};
        simParamCpu.totNumCell              = totNumCell_;

        multiCellSchedulerCpu mcSchCpu(&cellGrpPrmsCpu_);
        mcSchCpu.setup(&ueStatusCpu_, &schdSolCpu_, &cellGrpPrmsCpu_,
                       &simParamCpu, columnMajor);
        mcSchCpu.run();

        if (cellGrpPrmsGpu_.allocType == 0) {
            // Type-0: selected UE per (rbg, cell), -1 if none.
            for (uint16_t rbg = 0; rbg < nPrbGrp_; ++rbg) {
                for (uint16_t c = 0; c < totNumCell_; ++c) {
                    const size_t idx = static_cast<size_t>(rbg) * totNumCell_ + c;
                    const int16_t gpuPick = h_allocSolGpu_[idx];

                    // Invariant: chosen UE must be cell-associated.
                    if (gpuPick >= 0) {
                        ASSERT_LT(gpuPick, static_cast<int16_t>(nUe_))
                            << "Type-0 GPU UE index out of range at rbg=" << rbg
                            << " cell=" << c;
                        EXPECT_EQ(h_cellAssoc_[static_cast<size_t>(c) * nUe_ + gpuPick], 1)
                            << "Type-0 GPU picked non-associated UE " << gpuPick
                            << " for cell " << c << " at rbg=" << rbg;
                    }

                    if (mode == CmpMode::Strict) {
                        EXPECT_EQ(gpuPick, h_allocSolCpu_[idx])
                            << "Type-0 allocSol mismatch at rbg=" << rbg
                            << " cell=" << c;
                    }
                }
            }
        } else {
            // Type-1: per-UE (start, end) PRG range, -1 if unallocated.

            // Per-cell occupancy bitmap for non-overlap invariant.
            std::vector<std::vector<uint8_t>> cellPrgBusy(
                nCell_, std::vector<uint8_t>(nPrbGrp_, 0));

            for (uint16_t u = 0; u < nUe_; ++u) {
                const int16_t gpuStart = h_allocSolGpu_[2 * u];
                const int16_t gpuEnd   = h_allocSolGpu_[2 * u + 1];

                // Invariant: either unallocated, or in-range, associated,
                // and non-overlapping with other UEs in the same cell.
                if (gpuStart == -1 && gpuEnd == -1) {
                    // Unallocated is always valid.
                } else {
                    EXPECT_GE(gpuStart, 0) << "Type-1 GPU start<0 at UE=" << u;
                    EXPECT_GT(gpuEnd, gpuStart)
                        << "Type-1 GPU end<=start at UE=" << u;
                    EXPECT_LE(gpuEnd, static_cast<int16_t>(nPrbGrp_))
                        << "Type-1 GPU end>nPrbGrp at UE=" << u;

                    // Find UE's cell.
                    int16_t uCell = -1;
                    for (uint16_t c = 0; c < nCell_; ++c) {
                        if (h_cellAssoc_[static_cast<size_t>(c) * nUe_ + u]) {
                            uCell = static_cast<int16_t>(c);
                            break;
                        }
                    }
                    EXPECT_GE(uCell, 0)
                        << "Type-1 GPU allocated non-associated UE=" << u;

                    if (uCell >= 0 && gpuStart >= 0 && gpuEnd <= static_cast<int16_t>(nPrbGrp_)) {
                        for (int16_t p = gpuStart; p < gpuEnd; ++p) {
                            EXPECT_EQ(cellPrgBusy[uCell][p], 0)
                                << "Type-1 GPU PRG " << p
                                << " double-allocated in cell " << uCell
                                << " (collides at UE=" << u << ")";
                            cellPrgBusy[uCell][p] = 1;
                        }
                    }
                }

                if (mode == CmpMode::Strict) {
                    EXPECT_EQ(gpuStart, h_allocSolCpu_[2 * u])
                        << "Type-1 allocSol start mismatch at UE=" << u;
                    EXPECT_EQ(gpuEnd, h_allocSolCpu_[2 * u + 1])
                        << "Type-1 allocSol end mismatch at UE=" << u;
                }
            }
        }
    }

    // Structural validator for variants without a CPU reference.
    // Verifies entries are in range, refer to associated UEs, and
    // do not double-allocate within a cell.
    void ValidateSvdInvariants()
    {
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        const size_t nAllocSolBuf = std::max<size_t>(
            static_cast<size_t>(nCell_) * nPrbGrp_, 2u * nUe_);

        h_allocSolGpu_.assign(nAllocSolBuf, -1);
        ASSERT_EQ(cudaMemcpy(h_allocSolGpu_.data(), d_allocSol_,
                             nAllocSolBuf * sizeof(int16_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(h_cellAssoc_.data(), d_cellAssoc_,
                             static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);

        if (cellGrpPrmsGpu_.allocType == 0) {
            for (uint16_t rbg = 0; rbg < nPrbGrp_; ++rbg) {
                for (uint16_t c = 0; c < totNumCell_; ++c) {
                    const size_t idx = static_cast<size_t>(rbg) * totNumCell_ + c;
                    const int16_t gpuPick = h_allocSolGpu_[idx];
                    if (gpuPick >= 0) {
                        ASSERT_LT(gpuPick, static_cast<int16_t>(nUe_))
                            << "Type-0 SVD UE index out of range at rbg=" << rbg
                            << " cell=" << c;
                        EXPECT_EQ(h_cellAssoc_[static_cast<size_t>(c) * nUe_ + gpuPick], 1)
                            << "Type-0 SVD picked non-associated UE " << gpuPick
                            << " for cell " << c << " at rbg=" << rbg;
                    }
                }
            }
        } else {
            std::vector<std::vector<uint8_t>> cellPrgBusy(
                nCell_, std::vector<uint8_t>(nPrbGrp_, 0));

            for (uint16_t u = 0; u < nUe_; ++u) {
                const int16_t gpuStart = h_allocSolGpu_[2 * u];
                const int16_t gpuEnd   = h_allocSolGpu_[2 * u + 1];

                if (gpuStart == -1 && gpuEnd == -1) {
                    continue;
                }

                EXPECT_GE(gpuStart, 0) << "Type-1 SVD start<0 at UE=" << u;
                EXPECT_GT(gpuEnd, gpuStart)
                    << "Type-1 SVD end<=start at UE=" << u;
                EXPECT_LE(gpuEnd, static_cast<int16_t>(nPrbGrp_))
                    << "Type-1 SVD end>nPrbGrp at UE=" << u;

                int16_t uCell = -1;
                for (uint16_t c = 0; c < nCell_; ++c) {
                    if (h_cellAssoc_[static_cast<size_t>(c) * nUe_ + u]) {
                        uCell = static_cast<int16_t>(c);
                        break;
                    }
                }
                EXPECT_GE(uCell, 0)
                    << "Type-1 SVD allocated non-associated UE=" << u;

                if (uCell >= 0 && gpuStart >= 0
                    && gpuEnd <= static_cast<int16_t>(nPrbGrp_)) {
                    for (int16_t p = gpuStart; p < gpuEnd; ++p) {
                        EXPECT_EQ(cellPrgBusy[uCell][p], 0)
                            << "Type-1 SVD PRG " << p
                            << " double-allocated in cell " << uCell
                            << " (collides at UE=" << u << ")";
                        cellPrgBusy[uCell][p] = 1;
                    }
                }
            }
        }
    }

    // Count/sum helpers; call only after the validator has populated the host mirror.

    int CountAllocatedType1Ues() const
    {
        int count = 0;
        for (uint16_t u = 0; u < nUe_; ++u) {
            const int16_t s = h_allocSolGpu_[2 * u];
            const int16_t e = h_allocSolGpu_[2 * u + 1];
            if (s != -1 || e != -1) count++;
        }
        return count;
    }

    int TotalAllocatedType1Prgs() const
    {
        int total = 0;
        for (uint16_t u = 0; u < nUe_; ++u) {
            const int16_t s = h_allocSolGpu_[2 * u];
            const int16_t e = h_allocSolGpu_[2 * u + 1];
            if (s >= 0 && e > s) total += (e - s);
        }
        return total;
    }

    int CountAllocatedType0Picks() const
    {
        int count = 0;
        for (uint16_t rbg = 0; rbg < nPrbGrp_; ++rbg) {
            for (uint16_t c = 0; c < totNumCell_; ++c) {
                if (h_allocSolGpu_[static_cast<size_t>(rbg) * totNumCell_ + c] >= 0) {
                    count++;
                }
            }
        }
        return count;
    }
};

// ----------------------------------------------------------------------
// Success paths (DL, non-Asim).

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingDefaultSelectsNoPrdMmseIrc)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1);
}

// Exercises multi-round UE scheduling by using larger antenna counts so
// the first round fills capacity and forces a second round. Larger
// matrices introduce CPU/GPU fp drift, so use InvariantOnly.
TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1, CmpMode::InvariantOnly);
}

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingHalfSelectsHalfKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/1, /*lightWeight=*/0,
                1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the half-precision UE-selection update with a non-zero
// bfloat162 channel pattern that yields positive per-UE PF metrics.
TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingHalfNonZeroChannelExercisesUeSelect)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChanHalf = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<__nv_bfloat162> h_chanHalf(nChanHalf);
    for (size_t i = 0; i < nChanHalf; ++i) {
        const float re = 0.25f + 0.0625f * static_cast<float>(i % 7);
        const float im = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
        h_chanHalf[i] = __floats2bfloat162_rn(re, im);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_half_, h_chanHalf.data(),
                              nChanHalf * sizeof(__nv_bfloat162),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/1, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises multi-round UE scheduling on the half-precision path with
// larger antenna counts so the first round fills capacity.
TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingHalfLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChanHalf = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<__nv_bfloat162> h_chanHalf(nChanHalf);
    for (size_t i = 0; i < nChanHalf; ++i) {
        const float re = 0.25f + 0.0625f * static_cast<float>(i % 7);
        const float im = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
        h_chanHalf[i] = __floats2bfloat162_rn(re, im);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_half_, h_chanHalf.data(),
                              nChanHalf * sizeof(__nv_bfloat162),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/1, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingLightWeightComputeSelectsLwKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/1,
                1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises multi-round UE scheduling on the lightweight SINR-compute
// path with larger antenna counts.
TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingLightWeightComputeLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/1,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingLightWeightLoadSelectsLwKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/2,
                1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the parallel reduction and tie-break paths on the lightweight
// SINR-load kernel by providing strictly-increasing post-eq SINR values.
TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingLightWeightLoadNonUniformSinrExercisesReduction)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    // Make SINR strictly increasing by UE index so per-UE PF metrics
    // are also strictly increasing.
    const size_t nPostEqSinr = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> h_postEqSinr(nPostEqSinr);
    for (uint16_t u = 0; u < nUe_; ++u) {
        const float sinr = 0.5f * static_cast<float>(u + 1);
        for (uint16_t r = 0; r < nPrbGrp_; ++r) {
            for (uint8_t l = 0; l < nUeAnt_; ++l) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                 + static_cast<size_t>(r) * nUeAnt_
                                 + l;
                h_postEqSinr[idx] = sinr;
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_postEqSinr_, h_postEqSinr.data(),
                              nPostEqSinr * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/2,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingColumnMajorSelectsType1Cm)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1);
}

// Exercises multi-round UE scheduling on the type-1 column-major path
// with larger antenna counts; uses InvariantOnly due to fp drift.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingColumnMajorLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1, CmpMode::InvariantOnly);
}

// Exercises the empty-cell early-return path by zeroing cellAssoc so
// no UE is associated to any cell.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingColumnMajorNoAssociatedUeReturnsEarly)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1);
}

// Exercises the all-easy-peaks path by configuring a single UE per cell.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingColumnMajorSingleUePerCellExercisesAllEasyPeaks)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/2, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/1, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1);
}

// Exercises gap-fill bisection and difficult-peaks paths with a wider
// PRG count, non-uniform avgRates, and a biased channel.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingColumnMajorNonUniformExercisesGapFillAndDifficultPeaks)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/32,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    // Non-uniform avgRates so PF metrics differ between UEs.
    std::vector<float> nonUniformRates(nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        nonUniformRates[u] = 0.1f + 0.05f * static_cast<float>(u % 7);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, nonUniformRates.data(),
                              nUe_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Biased channel so different UEs dominate different bands.
    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.2f + 0.05f * static_cast<float>(i % 11);
        h_chan[i].y = 0.1f + 0.05f * static_cast<float>((i + 5) % 9);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/1);
}

TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorSelectsType1Rm)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/0);
}

// Exercises multi-round UE scheduling on the type-1 row-major path with
// larger antenna counts.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/0, CmpMode::InvariantOnly);
}

// Exercises the riding-peaks exhaustion break by zeroing the channel so
// all PF metrics are zero and the loop drains without allocating.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorZeroChannelExercisesAllocExhaustion)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    ASSERT_EQ(cudaMemsetAsync(d_estH_fr_, 0,
                              nChan * sizeof(cuComplex), stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the gap-fill tail on the type-1 row-major path with a wide
// PRG count and contiguous-per-cell cell association.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorNonUniformExercisesGapFill)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/32,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    // Contiguous-per-cell association layout.
    std::vector<uint8_t> contiguousAssoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t c = 0; c < nCell_; ++c) {
        for (uint8_t i = 0; i < numUeSchdPerCellTTI_; ++i) {
            const uint16_t u = static_cast<uint16_t>(c * numUeSchdPerCellTTI_ + i);
            if (u < nUe_) {
                contiguousAssoc[c * nUe_ + u] = 1;
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, contiguousAssoc.data(),
                              contiguousAssoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Non-uniform avgRates so PF metrics differ between UEs.
    std::vector<float> nonUniformRates(nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        nonUniformRates[u] = 0.1f + 0.05f * static_cast<float>(u % 7);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, nonUniformRates.data(),
                              nUe_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/0);
}

// Exercises the empty-cell early-return path by zeroing cellAssoc.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorNoAssociatedUeReturnsEarly)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/0);
}

// Exercises the bitonic-sort padding loop using a non-power-of-two
// scheduled-UE count with contiguous-per-cell association.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingRowMajorNonPow2ScheduledUesExercisesPadding)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/10, /*nPrbGrp=*/8,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/5, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    std::vector<uint8_t> contiguousAssoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t c = 0; c < nCell_; ++c) {
        for (uint8_t i = 0; i < numUeSchdPerCellTTI_; ++i) {
            const uint16_t u = static_cast<uint16_t>(c * numUeSchdPerCellTTI_ + i);
            if (u < nUe_) {
                contiguousAssoc[c * nUe_ + u] = 1;
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, contiguousAssoc.data(),
                              contiguousAssoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateAgainstCpuRef(/*columnMajor=*/0);
}

TEST_F(MultiCellSchedulerTest, DLType0SvdPrecodingSelectsSvdKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Exercises the SVD UE-selection update path with a strictly-positive
// channel pattern that yields positive per-UE PF metrics.
TEST_F(MultiCellSchedulerTest, DLType0SvdPrecodingNonZeroChannelExercisesUeSelect)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Exercises multi-round UE scheduling on the SVD path with larger
// antenna counts.
TEST_F(MultiCellSchedulerTest, DLType0SvdPrecodingLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Exercises the natural-exit path of the SVD UE-selection loop by
// using a single cell with every UE associated to it.
TEST_F(MultiCellSchedulerTest, DLType0SvdPrecodingSingleCellAllUeExercisesLoopNaturalExit)
{
    BuildPrms(/*nCell=*/1, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/8, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingSelectsType1SvdKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Exercises multi-round UE scheduling on the type-1 SVD column-major
// path with larger antenna counts.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Exercises the empty-cell early-return path by zeroing cellAssoc.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorNoAssociatedUeReturnsEarly)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    // No association → no UE allocated.
    for (uint16_t u = 0; u < nUe_; ++u) {
        EXPECT_EQ(h_allocSolGpu_[2 * u],     -1) << "UE " << u << " should be unallocated";
        EXPECT_EQ(h_allocSolGpu_[2 * u + 1], -1) << "UE " << u << " should be unallocated";
    }
}

// Multi-UE gap-fill with non-uniform avgRates and biased channel.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorNonUniformExercisesGapFillAndDifficultPeaks)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/32,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    std::vector<float> nonUniformRates(nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        nonUniformRates[u] = 0.1f + 0.05f * static_cast<float>(u % 7);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, nonUniformRates.data(),
                              nUe_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.2f + 0.05f * static_cast<float>(i % 11);
        h_chan[i].y = 0.1f + 0.05f * static_cast<float>((i + 5) % 9);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Non-power-of-two pfSize to exercise the bitonic-sort padding tail.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorNonPow2ScheduledUesExercisesPadding)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/10, /*nPrbGrp=*/8,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/5, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    std::vector<uint8_t> contiguousAssoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t c = 0; c < nCell_; ++c) {
        for (uint8_t i = 0; i < numUeSchdPerCellTTI_; ++i) {
            const uint16_t u = static_cast<uint16_t>(c * numUeSchdPerCellTTI_ + i);
            if (u < nUe_) {
                contiguousAssoc[c * nUe_ + u] = 1;
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, contiguousAssoc.data(),
                              contiguousAssoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Zero channel to drive the riding-peaks drain path.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorZeroChannelExercisesAllocExhaustion)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    ASSERT_EQ(cudaMemsetAsync(d_estH_fr_, 0,
                              nChan * sizeof(cuComplex), stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
}

// Single UE per cell to exercise the all-easy-peaks early-exit path.
TEST_F(MultiCellSchedulerTest, DLType1SvdPrecodingColumnMajorSingleUePerCellExercisesAllEasyPeaks)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/2, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/1, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    // Small random inputs may legitimately produce no allocation;
    // rely on structural checks only.
}

TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingSelectsUlKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Kernel produced no allocations";
}

// Exercises the riding-peaks exhaustion break on the UL SVD path by
// setting avgRates to +INF so all PF metrics collapse to zero.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingInfAvgRatesExercisesAllocExhaustion)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    std::vector<float> infRates(nUe_, std::numeric_limits<float>::infinity());
    ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, infRates.data(),
                              nUe_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    // Infinite average rates collapse every PF metric to zero → no UE allocated.
    for (uint16_t u = 0; u < nUe_; ++u) {
        EXPECT_EQ(h_allocSolGpu_[2 * u],     -1) << "UE " << u << " should be unallocated";
        EXPECT_EQ(h_allocSolGpu_[2 * u + 1], -1) << "UE " << u << " should be unallocated";
    }
}

// Exercises multi-round UE scheduling on the UL type-1 SVD path with
// larger antenna counts.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nChan = static_cast<size_t>(nPrbGrp_) * nUe_ * totNumCell_
                                                  * nBsAnt_ * nUeAnt_;
    std::vector<cuComplex> h_chan(nChan);
    for (size_t i = 0; i < nChan; ++i) {
        h_chan[i].x = 0.25f + 0.0625f * static_cast<float>(i % 7);
        h_chan[i].y = 0.125f + 0.0625f * static_cast<float>((i + 3) % 5);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_chan.data(),
                              nChan * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Kernel produced no allocations";
}

// Exercises the empty-cell early-return path by associating all UEs to
// a single cell so the other cells have no associated UEs.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingEmptyCellExercisesEarlyReturn)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/4, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    // Place all UEs in cell 0; cell 1 has none.
    std::vector<uint8_t> h_assoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t u = 0; u < nUe_; ++u) {
        h_assoc[0 * nUe_ + u] = 1;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, h_assoc.data(),
                              h_assoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Cell 0 should produce at least one allocation";
}

// Strictly-positive sinVal to exercise the non-zero PF-metric rescale.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingNonZeroSinValExercisesPfRescale)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> h_sinVal(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        h_sinVal[i] = 0.5f + 0.25f * static_cast<float>(i % 13);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_, h_sinVal.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Kernel produced no allocations";
}

// Non-power-of-two pfSize to exercise the bitonic-sort padding tail.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingNonPow2ScheduledUesExercisesPadding)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/10, /*nPrbGrp=*/8,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/5, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    std::vector<uint8_t> contiguousAssoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t c = 0; c < nCell_; ++c) {
        for (uint8_t i = 0; i < numUeSchdPerCellTTI_; ++i) {
            const uint16_t u = static_cast<uint16_t>(c * numUeSchdPerCellTTI_ + i);
            if (u < nUe_) {
                contiguousAssoc[c * nUe_ + u] = 1;
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, contiguousAssoc.data(),
                              contiguousAssoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> h_sinVal(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        h_sinVal[i] = 0.5f + 0.25f * static_cast<float>(i % 13);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_, h_sinVal.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Kernel produced no allocations";
}

// Wide PRG count + non-uniform avgRates + biased sinVal to exercise
// peak-collision and band-extend arms.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingNonUniformExercisesPeakCollisionAndExtend)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/32,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    std::vector<float> nonUniformRates(nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        nonUniformRates[u] = 0.1f + 0.05f * static_cast<float>(u % 7);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, nonUniformRates.data(),
                              nUe_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> h_sinVal(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        h_sinVal[i] = 0.2f + 0.05f * static_cast<float>(i % 11);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_, h_sinVal.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_GT(CountAllocatedType1Ues(), 0) << "Kernel produced no allocations";
}

// Single UE per cell to exercise the all-easy-peaks early-exit path.
TEST_F(MultiCellSchedulerTest, ULType1SvdPrecodingSingleUePerCellExercisesAllEasyPeaks)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/2, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/1, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> h_sinVal(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        h_sinVal[i] = 0.5f + 0.25f * static_cast<float>(i % 13);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_, h_sinVal.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/0,
                /*percSmNumThrdBlk=*/1.0f, stream_);
    mcSch.run(stream_);
    ValidateSvdInvariants();
    EXPECT_EQ(CountAllocatedType1Ues(), 2) << "Both single UEs should be allocated";
}

// ----------------------------------------------------------------------
// Throw paths (DL).

TEST_F(MultiCellSchedulerTest, DLNonAsimHarqEnabledThrowsRuntimeError)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    1, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingRowMajorThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    /*columnMajor=*/0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, DLType0NoPrecodingHalfWithLightWeightThrows)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    1, /*halfPrecision=*/1, /*lightWeight=*/1, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, DLType1SvdRowMajorThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    /*columnMajor=*/0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, DLType0SvdRowMajorThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    /*columnMajor=*/0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, DLSvdLightWeightThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    1, 0, /*lightWeight=*/1, 1.0f, stream_),
        std::runtime_error);
}

// Exercises the throw path for type-1 allocation combined with
// lightweight scheduling.
TEST_F(MultiCellSchedulerTest, DLType1NoPrecodingLightWeightThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    /*columnMajor=*/1, /*halfPrecision=*/0,
                    /*lightWeight=*/1, 1.0f, stream_),
        std::runtime_error);
}

// ----------------------------------------------------------------------
// Throw paths (UL).

TEST_F(MultiCellSchedulerTest, ULType0AllocThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, ULHalfPrecisionThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    0, /*halfPrecision=*/1, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, ULLightWeightThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    0, 0, /*lightWeight=*/1, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, ULNonAsimNoPrecodingThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, ULNonAsimHarqThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    0, 0, 0, 1.0f, stream_),
        std::runtime_error);
}

// Zero SM-occupancy ratio triggers an early throw in setup.
TEST_F(MultiCellSchedulerTest, SetupZeroPercSmNumThrdBlkThrows)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    1, 0, 0, /*percSmNumThrdBlk=*/0.0f, stream_),
        std::runtime_error);
}

// ----------------------------------------------------------------------
// Aerial Sim setup() overload - success and throw paths.

// Helper that pre-initializes the lightWeight member to 0 before the
// Aerial Sim setup is called, since the Asim setup overload does not
// assign it.
static void InitLightWeightZeroOnAsim(multiCellScheduler& mcSch,
                                      cumacCellGrpUeStatus* ueSt,
                                      cumacSchdSol* sol,
                                      cumacCellGrpPrms* prms,
                                      cumacSimParam* sim,
                                      cudaStream_t strm)
{
    try {
        // Deliberately throws after lightWeight has been set.
        mcSch.setup(ueSt, sol, prms, sim,
                    /*columnMajor=*/1, /*halfPrecision=*/0, /*lightWeight=*/0,
                    1.0f, strm);
    } catch (const std::runtime_error&) {
        // Expected.
    }
}

TEST_F(MultiCellSchedulerTest, AsimDLType1NoPrecodingSelectsWbSinrKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the riding-peaks extend-down, extend-up and skip arms by
// supplying a non-uniform bumpy SINR pattern across PRGs.
TEST_F(MultiCellSchedulerTest, AsimDLType1NoPrecodingWbSinrCoversExtendDownAndSkip)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/2, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/2, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();

    // Bumpy sinVal_asim per UE so the non-zero-mean SINR path runs.
    const float bumpyPattern[4] = {0.5f, 1.0f, 0.7f, 0.8f};
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t u = 0; u < nUe_; ++u) {
        for (uint16_t p = 0; p < nPrbGrp_; ++p) {
            for (uint8_t a = 0; a < nUeAnt_; ++a) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                       + p * nUeAnt_ + a;
                sinValPattern[idx] = bumpyPattern[p];
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises multi-round UE scheduling on the Asim no-HARQ wbSinr path
// with a large nPrbGrp so the first round fills capacity.
TEST_F(MultiCellSchedulerTest, AsimDLType1NoPrecodingWbSinrLargeNPrbGrpExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/16, /*nPrbGrp=*/128,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/8, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the empty-cell early-return on the Asim no-HARQ wbSinr path
// by zeroing cellAssoc.
TEST_F(MultiCellSchedulerTest, AsimDLType1NoPrecodingWbSinrNoAssociatedUeReturnsEarly)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the riding-peaks retry/exit path on the Asim no-HARQ wbSinr
// kernel using a sparse SINR pattern with only one positive PRG per UE.
TEST_F(MultiCellSchedulerTest, AsimDLType1NoPrecodingWbSinrSparsePfExercisesRetry)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/2, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/2, /*enableHarq=*/false, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();

    // Sparse SINR: each UE has positive value at only one PRG.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    const uint16_t pfPrgForUe[2] = {0, 2};
    for (uint16_t u = 0; u < nUe_; ++u) {
        for (uint8_t a = 0; a < nUeAnt_; ++a) {
            const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                   + static_cast<size_t>(pfPrgForUe[u]) * nUeAnt_
                                   + a;
            sinValPattern[idx] = 1.0f;
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, AsimDLType1HarqSelectsWbSinrHarqKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the available/unavailable PRG and new-TX/re-TX paths on the
// Asim HARQ wbSinr kernel with a mixed prgMsk and mixed newDataActUe.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqWbSinrHarqCoversAvailableAndNewTx)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // Mark one PRG unavailable, others available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    if (nPrbGrp_ > 1) prgPattern[1] = 0;
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Mix of new-TX and re-TX UEs.
    std::vector<int8_t> newDataPattern(nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        newDataPattern[u] = (u < (nUe_ / 2)) ? 1 : 0;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // UE 0 SINR stays zero (zero-mean fallback). Others non-uniform positive.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t u = 1; u < nUe_; ++u) {
        for (uint16_t p = 0; p < nPrbGrp_; ++p) {
            for (uint8_t a = 0; a < nUeAnt_; ++a) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                       + p * nUeAnt_ + a;
                sinValPattern[idx] = 0.5f + 0.125f
                                   * static_cast<float>((u + p + a) % 11);
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the riding-peaks extend-high and non-adjacent-skip arms with
// a bumpy SINR pattern that peaks in the middle of the PRG range.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqWbSinrHarqCoversHighExtendAndSkip)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // All PRGs available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Only UE 0 is new-TX so its pick order dominates the global sort.
    std::vector<int8_t> newDataPattern(nUe_, 0);
    newDataPattern[0] = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Only UE 0 has the bumpy SINR pattern; others stay zero.
    const float bumpyPattern[4] = {0.5f, 1.0f, 0.7f, 0.8f};
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t p = 0; p < nPrbGrp_; ++p) {
        for (uint8_t a = 0; a < nUeAnt_; ++a) {
            const size_t idx = static_cast<size_t>(0) * nPrbGrp_ * nUeAnt_
                                   + p * nUeAnt_ + a;
            sinValPattern[idx] = bumpyPattern[p];
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the new-TX unavailable PRG path by masking PRG 0 with all
// UEs marked as new-TX.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqWbSinrHarqCoversPrgMaskedNewTx)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // PRG 0 unavailable, others available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    prgPattern[0] = 0;
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // All UEs are new-TX.
    std::vector<int8_t> newDataPattern(nUe_, 1);
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Non-zero SINR pattern so the available-PRG arms run normally.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t u = 0; u < nUe_; ++u) {
        for (uint16_t p = 0; p < nPrbGrp_; ++p) {
            for (uint8_t a = 0; a < nUeAnt_; ++a) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                       + p * nUeAnt_ + a;
                sinValPattern[idx] = 0.5f + 0.125f
                                   * static_cast<float>((u + p + a) % 11);
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// PRG-masked + re-TX: allocSol is copied from allocSolLastTx.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqWbSinrHarqCoversPrgMaskedReTx)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // PRG 0 unavailable, others available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    prgPattern[0] = 0;
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // All UEs on re-TX.
    std::vector<int8_t> newDataPattern(nUe_, 0);
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Resize allocSolLastTx so the kernel reads valid memory when
    // the UE count exceeds the fixture's default sizing.
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaFree(d_allocSolLastTx_), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_allocSolLastTx_,
                         static_cast<size_t>(2u) * nUe_ * sizeof(int16_t)),
              cudaSuccess);
    ueStatusGpu_.allocSolLastTx = d_allocSolLastTx_;

    // Distinct (start, end) per UE for the copy-back to verify.
    std::vector<int16_t> lastTxPattern(static_cast<size_t>(2) * nUe_);
    for (uint16_t u = 0; u < nUe_; ++u) {
        const int16_t s = static_cast<int16_t>(u % nPrbGrp_);
        lastTxPattern[2 * u]     = s;
        lastTxPattern[2 * u + 1] = static_cast<int16_t>(s + 1);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_allocSolLastTx_, lastTxPattern.data(),
                              lastTxPattern.size() * sizeof(int16_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Non-zero SINR pattern.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t u = 0; u < nUe_; ++u) {
        for (uint16_t p = 0; p < nPrbGrp_; ++p) {
            for (uint8_t a = 0; a < nUeAnt_; ++a) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                       + p * nUeAnt_ + a;
                sinValPattern[idx] = 0.5f + 0.125f
                                   * static_cast<float>((u + p + a) % 11);
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // Each re-TX UE should retain its previous allocation.
    const size_t nAllocSolBuf = std::max<size_t>(
        static_cast<size_t>(nCell_) * nPrbGrp_, 2u * nUe_);
    std::vector<int16_t> hostAllocSol(nAllocSolBuf, -1);
    ASSERT_EQ(cudaMemcpy(hostAllocSol.data(), d_allocSol_,
                         nAllocSolBuf * sizeof(int16_t),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (uint16_t u = 0; u < nUe_; ++u) {
        EXPECT_EQ(hostAllocSol[2 * u],     lastTxPattern[2 * u])
            << "Re-TX UE " << u << " allocSol[start] mismatch";
        EXPECT_EQ(hostAllocSol[2 * u + 1], lastTxPattern[2 * u + 1])
            << "Re-TX UE " << u << " allocSol[end] mismatch";
    }
}

// Exercises multi-round UE scheduling on the Asim HARQ wbSinr path with
// a large nPrbGrp so the first round fills capacity.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqWbSinrHarqLargeNPrbGrpExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/16, /*nPrbGrp=*/128,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/8, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // All PRGs available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    std::vector<int8_t> newDataPattern(nUe_, 1);
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the empty-cell early-return path by zeroing cellAssoc.
TEST_F(MultiCellSchedulerTest, AsimDLType1HarqNoAssociatedUeReturnsEarly)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;
    BuildAsimBuffers();
    BuildHarqBuffers();

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, AsimDLColumnMajorThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                    /*columnMajor=*/1, 0, stream_),
        std::runtime_error);
}

TEST_F(MultiCellSchedulerTest, AsimDLHalfPrecisionThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                    /*columnMajor=*/0, /*halfPrecision=*/1, stream_),
        std::runtime_error);
}

// Exercises the lightweight-not-supported throw on the Asim path.
TEST_F(MultiCellSchedulerTest, AsimDLLightWeightThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                    /*columnMajor=*/0, /*halfPrecision=*/0, /*lightWeight=*/1,
                    1.0f, stream_),
        std::runtime_error);
}

// ----------------------------------------------------------------------
// Aerial Sim UL paths.

TEST_F(MultiCellSchedulerTest, AsimULType1SvdSelectsUlKernel)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/false, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

TEST_F(MultiCellSchedulerTest, AsimULType1HarqSelectsHarqUlKernel)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the available/unavailable PRG, new-TX/re-TX, and bitonic
// padding paths on the Asim UL HARQ kernel.
TEST_F(MultiCellSchedulerTest, AsimULType1HarqCoversAvailableAndNewTx)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/6, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/3, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // One PRG unavailable, others available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    if (nPrbGrp_ > 1) prgPattern[1] = 0;
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Mix of new-TX and re-TX UEs; one new-TX UE has zero SINR.
    std::vector<int8_t> newDataPattern(nUe_, 0);
    if (nUe_ > 0) newDataPattern[0] = 1;
    if (nUe_ > 2) newDataPattern[2] = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // UE 0 SINR stays zero (fallback). Others non-uniform positive.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t u = 1; u < nUe_; ++u) {
        for (uint16_t p = 0; p < nPrbGrp_; ++p) {
            for (uint8_t a = 0; a < nUeAnt_; ++a) {
                const size_t idx = static_cast<size_t>(u) * nPrbGrp_ * nUeAnt_
                                       + p * nUeAnt_ + a;
                sinValPattern[idx] = 0.25f + 0.0625f
                                   * static_cast<float>((u + p + a) % 11);
            }
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the riding-peaks extend-high and skip arms with a bumpy
// SINR pattern on a single new-TX UE.
TEST_F(MultiCellSchedulerTest, AsimULType1HarqCoversHighExtendAndSkip)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // All PRGs available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Only UE 0 is new-TX.
    std::vector<int8_t> newDataPattern(nUe_, 0);
    newDataPattern[0] = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Only UE 0 has the bumpy SINR pattern.
    const float bumpyPattern[4] = {0.5f, 1.0f, 0.7f, 0.8f};
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal, 0.0f);
    for (uint16_t p = 0; p < nPrbGrp_; ++p) {
        for (uint8_t a = 0; a < nUeAnt_; ++a) {
            const size_t idx = static_cast<size_t>(0) * nPrbGrp_ * nUeAnt_
                                   + p * nUeAnt_ + a;
            sinValPattern[idx] = bumpyPattern[p];
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the new-TX unavailable PRG arm on the UL HARQ path by
// masking PRG 0 with new-TX UEs present.
TEST_F(MultiCellSchedulerTest, AsimULType1HarqCoversPrgMaskedNewTxRbg0)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/6, /*nPrbGrp=*/4,
              /*nBsAnt=*/4, /*nUeAnt=*/4,
              /*numUeSchdPerCellTTI=*/3, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // PRG 0 unavailable, others available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    prgPattern[0] = 0;
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Some UEs are marked new-TX.
    std::vector<int8_t> newDataPattern(nUe_, 0);
    if (nUe_ > 0) newDataPattern[0] = 1;
    if (nUe_ > 2) newDataPattern[2] = 1;
    ASSERT_EQ(cudaMemcpyAsync(d_newDataActUe_, newDataPattern.data(),
                              nUe_ * sizeof(int8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Non-zero SINR so the allocator runs normally.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        sinValPattern[i] = 0.25f + 0.0625f * static_cast<float>(i % 11);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises multi-round UE scheduling on the Asim UL HARQ path with
// larger antenna counts.
TEST_F(MultiCellSchedulerTest, AsimULType1HarqLargeBsAntExercisesMultiRoundSched)
{
    BuildPrms(/*nCell=*/2, /*nUe=*/8, /*nPrbGrp=*/4,
              /*nBsAnt=*/16, /*nUeAnt=*/16,
              /*numUeSchdPerCellTTI=*/4, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    // All PRGs available.
    std::vector<uint8_t> prgPattern(nPrbGrp_, 1);
    for (uint16_t c = 0; c < nCell_; ++c) {
        ASSERT_EQ(cudaFree(d_prgMskRows_[c]), cudaSuccess);
        uint8_t* row = nullptr;
        ASSERT_EQ(cudaMalloc(&row, nPrbGrp_ * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(row, prgPattern.data(),
                                  nPrbGrp_ * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);
        d_prgMskRows_[c] = row;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_prgMsk_, d_prgMskRows_.data(),
                              nCell_ * sizeof(uint8_t*),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);

    // Non-uniform SINR.
    const size_t nSinVal = static_cast<size_t>(nUe_) * nPrbGrp_ * nUeAnt_;
    std::vector<float> sinValPattern(nSinVal);
    for (size_t i = 0; i < nSinVal; ++i) {
        sinValPattern[i] = 0.25f + 0.0625f * static_cast<float>(i % 11);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_sinVal_asim_, sinValPattern.data(),
                              nSinVal * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the empty-cell early-return path by zeroing cellAssoc.
TEST_F(MultiCellSchedulerTest, AsimULType1HarqNoAssociatedUeReturnsEarly)
{
    BuildPrms(2, 8, 4, 4, 4, 4, /*enableHarq=*/true, /*dlInd=*/false);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 1;
    BuildAsimBuffers();
    BuildHarqBuffers();

    ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0,
                              static_cast<size_t>(nCell_) * nUe_ * sizeof(uint8_t),
                              stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    InitLightWeightZeroOnAsim(mcSch, &ueStatusGpu_, &schdSolGpu_,
                              &cellGrpPrmsGpu_, &simParam_, stream_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                /*columnMajor=*/0, /*halfPrecision=*/0, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// Exercises the type-0 allocation throw on the Asim path.
TEST_F(MultiCellSchedulerTest, AsimDLType0AllocThrowsUnsupported)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_, /*in_Asim=*/1);
    EXPECT_THROW(
        mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_,
                    /*columnMajor=*/0, /*halfPrecision=*/0, stream_),
        std::runtime_error);
}

// ----------------------------------------------------------------------
// Exercises the default switch case when precodingScheme is unknown.
TEST_F(MultiCellSchedulerTest, DLUnknownPrecodingFallsThroughToNoPrdMmseIrc)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 2;  // unknown value

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, 0, 0, 1.0f, stream_);
    mcSch.run(stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// ----------------------------------------------------------------------
// debugLog() tests. stdout is redirected to /dev/null to keep output clean.

TEST_F(MultiCellSchedulerTest, DebugLogRunsWithoutCrash)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                1, 0, 0, 1.0f, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    {
        ScopedStdoutSilencer silencer;
        mcSch.debugLog();
    }
    SUCCEED();
}

// Exercises the row-major print path in debugLog(). Requires type-1
// allocation since type-0 row-major is unsupported.
TEST_F(MultiCellSchedulerTest, DebugLogRowMajorExercisesRowMajorPrint)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 1;
    cellGrpPrmsGpu_.precodingScheme = 0;

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/0, 0, 0, 1.0f, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    {
        ScopedStdoutSilencer silencer;
        mcSch.debugLog();
    }
    SUCCEED();
}

// Exercises debugLog()'s per-cell UE scan loop by leaving one cell
// with no associated UE.
TEST_F(MultiCellSchedulerTest, DebugLogCellWithNoAssocUeExercisesLoopCompletion)
{
    BuildPrms(2, 8, 4, 4, 4, 4, false, true);
    cellGrpPrmsGpu_.allocType       = 0;
    cellGrpPrmsGpu_.precodingScheme = 0;

    // Leave one cell empty while keeping the other populated.
    std::vector<uint8_t> partialAssoc(static_cast<size_t>(nCell_) * nUe_, 0);
    for (uint16_t u = 0; u < nUe_; ++u) {
        if ((u % nCell_) == 1) {
            partialAssoc[1 * nUe_ + u] = 1;
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, partialAssoc.data(),
                              partialAssoc.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    multiCellScheduler mcSch(&cellGrpPrmsGpu_);
    mcSch.setup(&ueStatusGpu_, &schdSolGpu_, &cellGrpPrmsGpu_, &simParam_,
                /*columnMajor=*/1, 0, 0, 1.0f, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    {
        ScopedStdoutSilencer silencer;
        mcSch.debugLog();
    }
    SUCCEED();
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // CUDA is not fork-safe; "threadsafe" forces fork+exec for death tests.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    int rc = RUN_ALL_TESTS();

    return rc;
}
