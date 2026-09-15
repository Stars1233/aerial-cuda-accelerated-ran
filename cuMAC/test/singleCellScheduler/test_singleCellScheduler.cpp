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

#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "4T4R/singleCellScheduler.cuh"
#include "4T4R/singleCellSchedulerCpu.h"


using namespace cumac;

namespace {

/**
 * @brief Smallest power of two greater than or equal to @p v.
 *
 * @param[in] v Value to round up to a power of two.
 * @return Next power of two >= @p v; 1 when @p v is 0; 0 when the result would
 *         overflow uint32_t (no representable next power of two).
 */
uint32_t nextPow2(uint32_t v)
{
    if(v == 0) {
        return 1;
    }
    uint32_t p = 1;
    while(p < v)
    {
        // Next shift would overflow uint32_t; no representable power of two.
        if(p > (std::numeric_limits<uint32_t>::max() >> 1)) {
            return 0;
        }
        p <<= 1;
    }
    return p;
}

/**
 * @brief Drives the GPU single-cell scheduler against its bundled CPU reference.
 *
 * Runs both implementations on identical synthetic inputs and asserts they
 * produce the same scheduling solution. Matching the project's own reference
 * implementation is a functional-correctness check, not just an execution check.
 *
 * A single coordinated cell is modelled; the same host channel data backs both
 * the CPU run and the device copy, so any divergence is the GPU's.
 */
class SingleCellSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override
    {
        for(auto* p : d_perUe_) {
            if(p) {
                cudaFree(p);
            }
        }
        for(auto* p : h_perUe_) {
            delete[] p;
        }
        if(d_estHptr_)     { cudaFree(d_estHptr_); }
        if(d_cellAssoc_)   { cudaFree(d_cellAssoc_); }
        if(d_avgRates_)    { cudaFree(d_avgRates_); }
        if(d_prdMat_)      { cudaFree(d_prdMat_); }
        if(d_allocSol_)    { cudaFree(d_allocSol_); }
        if(d_pfMetricArr_) { cudaFree(d_pfMetricArr_); }
        if(d_pfIdArr_)     { cudaFree(d_pfIdArr_); }
        if(stream_)        { cudaStreamDestroy(stream_); }
    }

    /**
     * @brief Build all host/device buffers and populate the GPU and CPU parameter structs.
     *
     * @param[in] nUe Number of UEs.
     * @param[in] nPrbGrp Number of PRB groups.
     * @param[in] nBsAnt Number of base-station antennas.
     * @param[in] nUeAnt Number of UE antennas.
     * @param[in] allocType Allocation type (1 selects the PF-metric path).
     * @param[in] precodingScheme Precoding scheme (1 allocates precoder matrices).
     * @param[in] sigmaSqrd Noise variance.
     * @param[in] W Bandwidth/weight scalar passed to the scheduler.
     * @param[in] assocMask Optional per-UE association mask (size must equal nUe);
     *            empty means all UEs are associated.
     */
    void Build(uint16_t nUe, uint16_t nPrbGrp, uint8_t nBsAnt, uint8_t nUeAnt,
               uint8_t allocType, uint8_t precodingScheme,
               float sigmaSqrd, float W,
               const std::vector<uint8_t>& assocMask = {})
    {
        nUe_ = nUe; nPrbGrp_ = nPrbGrp; nBsAnt_ = nBsAnt; nUeAnt_ = nUeAnt;

        const size_t nChanPerUe = static_cast<size_t>(nPrbGrp) * nBsAnt * nUeAnt;

        // Reproducible synthetic channel: fixed seed, unit-variance complex
        // Gaussian. The specific seed is arbitrary.
        std::mt19937 rng(0x5C5C5C5Cu);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        const float kInvSqrt2 = 1.0f / std::sqrt(2.0f);

        h_perUe_.assign(nUe, nullptr);
        d_perUe_.assign(nUe, nullptr);
        std::vector<cuComplex*> h_ptrDev(nUe, nullptr);
        h_ptrCpu_.assign(nUe, nullptr);
        for(uint16_t u = 0; u < nUe; ++u)
        {
            auto* host = new cuComplex[nChanPerUe];
            for(size_t i = 0; i < nChanPerUe; ++i)
            {
                host[i].x = dist(rng) * kInvSqrt2;
                host[i].y = dist(rng) * kInvSqrt2;
            }
            h_perUe_[u]  = host;
            h_ptrCpu_[u] = host;
            ASSERT_EQ(cudaMalloc(&d_perUe_[u], nChanPerUe * sizeof(cuComplex)), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(d_perUe_[u], host, nChanPerUe * sizeof(cuComplex),
                                      cudaMemcpyHostToDevice, stream_), cudaSuccess);
            h_ptrDev[u] = d_perUe_[u];
        }
        ASSERT_EQ(cudaMalloc(&d_estHptr_, nUe * sizeof(cuComplex*)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_estHptr_, h_ptrDev.data(), nUe * sizeof(cuComplex*),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);

        // Cell association (single cell -> index [uIdx]).
        h_cellAssoc_.assign(nUe, 1);
        if(!assocMask.empty())
        {
            ASSERT_EQ(assocMask.size(), static_cast<size_t>(nUe));
            h_cellAssoc_ = assocMask;
        }
        ASSERT_EQ(cudaMalloc(&d_cellAssoc_, nUe * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, h_cellAssoc_.data(), nUe * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);

        // Long-term average rates (unit -> PF metric == raw data rate).
        h_avgRates_.assign(nUe, 1.0f);
        ASSERT_EQ(cudaMalloc(&d_avgRates_, nUe * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, h_avgRates_.data(), nUe * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_), cudaSuccess);

        // Precoder matrices (only consumed by the SVD kernel). Sized to match
        // the scheduler's own indexing: nPrbGrp * nUe * nCell * nBsAnt^2.
        if(precodingScheme == 1)
        {
            const size_t nPrd = static_cast<size_t>(nPrbGrp) * nUe * nCell_ *
                                nBsAnt * nBsAnt;
            h_prdMat_.resize(nPrd);
            for(auto& c : h_prdMat_) { c.x = dist(rng) * kInvSqrt2; c.y = dist(rng) * kInvSqrt2; }
            ASSERT_EQ(cudaMalloc(&d_prdMat_, nPrd * sizeof(cuComplex)), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(d_prdMat_, h_prdMat_.data(), nPrd * sizeof(cuComplex),
                                      cudaMemcpyHostToDevice, stream_), cudaSuccess);
        }

        // Scheduling-solution buffer: allocate enough for either allocation type
        // and zero-init both sides identically, so any entry the kernel leaves
        // untouched still compares equal.
        const size_t allocLen = std::max<size_t>(
            static_cast<size_t>(nPrbGrp) * nCell_, static_cast<size_t>(2) * nUe);
        h_allocSolCpu_.assign(allocLen, 0);
        h_allocSolGpu_.assign(allocLen, 0);
        ASSERT_EQ(cudaMalloc(&d_allocSol_, allocLen * sizeof(int16_t)), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_allocSol_, 0, allocLen * sizeof(int16_t), stream_), cudaSuccess);

        if(allocType == 1)
        {
            const uint32_t pow2N = nextPow2(static_cast<uint32_t>(nUe) * nPrbGrp);
            ASSERT_EQ(cudaMalloc(&d_pfMetricArr_, pow2N * sizeof(float)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_pfIdArr_, pow2N * sizeof(uint16_t)), cudaSuccess);
        }

        // GPU-side parameter structs point at device buffers.
        prmsGpu_ = cumacCellGrpPrms{};
        prmsGpu_.cellAssoc            = d_cellAssoc_;
        prmsGpu_.estH_fr_perUeBuffer  = d_estHptr_;
        prmsGpu_.prdMat               = d_prdMat_;
        prmsGpu_.nUe                  = nUe;
        prmsGpu_.nPrbGrp             = nPrbGrp;
        prmsGpu_.nBsAnt              = nBsAnt;
        prmsGpu_.nUeAnt             = nUeAnt;
        prmsGpu_.W                   = W;
        prmsGpu_.sigmaSqrd          = sigmaSqrd;
        prmsGpu_.allocType          = allocType;
        prmsGpu_.precodingScheme    = precodingScheme;

        prmsCpu_ = cumacCellGrpPrms{};
        prmsCpu_.cellAssoc            = h_cellAssoc_.data();
        prmsCpu_.estH_fr_perUeBuffer  = h_ptrCpu_.data();
        prmsCpu_.prdMat               = h_prdMat_.empty() ? nullptr : h_prdMat_.data();
        prmsCpu_.nUe                  = nUe;
        prmsCpu_.nPrbGrp             = nPrbGrp;
        prmsCpu_.nBsAnt              = nBsAnt;
        prmsCpu_.nUeAnt             = nUeAnt;
        prmsCpu_.W                   = W;
        prmsCpu_.sigmaSqrd          = sigmaSqrd;
        prmsCpu_.allocType          = allocType;
        prmsCpu_.precodingScheme    = precodingScheme;

        ueStatGpu_ = cumacCellGrpUeStatus{};
        ueStatGpu_.avgRates = d_avgRates_;
        ueStatCpu_ = cumacCellGrpUeStatus{};
        ueStatCpu_.avgRates = h_avgRates_.data();

        schdGpu_ = cumacSchdSol{};
        schdGpu_.allocSol    = d_allocSol_;
        schdGpu_.pfMetricArr = d_pfMetricArr_;
        schdGpu_.pfIdArr     = d_pfIdArr_;
        schdCpu_ = cumacSchdSol{};
        schdCpu_.allocSol    = h_allocSolCpu_.data();

        sim_ = cumacSimParam{};
        sim_.totNumCell = nCell_;

        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }

    /**
     * @brief Run the GPU and CPU schedulers and copy the GPU solution back to host.
     */
    void RunBoth()
    {
        singleCellScheduler    gpu;
        singleCellSchedulerCpu cpu;

        gpu.setup(cellId_, &ueStatGpu_, &schdGpu_, &prmsGpu_, &sim_, stream_);
        cpu.setup(cellId_, &ueStatCpu_, &schdCpu_, &prmsCpu_, &sim_);

        gpu.run(stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        cpu.run();

        ASSERT_EQ(cudaMemcpyAsync(h_allocSolGpu_.data(), d_allocSol_,
                                  h_allocSolGpu_.size() * sizeof(int16_t),
                                  cudaMemcpyDeviceToHost, stream_), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }

    /**
     * @brief Re-upload one UE's (possibly mutated) host channel buffer to the device.
     *
     * @param[in] ue Index of the UE whose channel buffer is re-uploaded.
     */
    void ReuploadUe(uint16_t ue)
    {
        const size_t n = static_cast<size_t>(nPrbGrp_) * nBsAnt_ * nUeAnt_;
        ASSERT_EQ(cudaMemcpy(d_perUe_[ue], h_perUe_[ue], n * sizeof(cuComplex),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }

    /**
     * @brief Give every PRG of a UE the same channel so all its metrics tie.
     *
     * The sort then orders PRGs by index, forcing the consecutive allocator to
     * seed at the lowest PRG and grow its range upward.
     *
     * @param[in] ue Index of the UE whose PRGs are made identical.
     */
    void MakeUePrgsIdentical(uint16_t ue)
    {
        const size_t blk = static_cast<size_t>(nBsAnt_) * nUeAnt_;
        for(uint16_t p = 1; p < nPrbGrp_; ++p)
            std::memcpy(h_perUe_[ue] + p * blk, h_perUe_[ue], blk * sizeof(cuComplex));
        ReuploadUe(ue);
    }

    /**
     * @brief Compare the entire scheduling-solution buffer between GPU and CPU.
     *
     * Both sides are zero-init and the kernels write the same entries, so
     * untouched slots compare equal.
     */
    void ExpectAllocSolFullMatch()
    {
        ASSERT_EQ(h_allocSolGpu_.size(), h_allocSolCpu_.size());
        for(size_t i = 0; i < h_allocSolGpu_.size(); ++i)
        {
            ASSERT_EQ(h_allocSolGpu_[i], h_allocSolCpu_[i])
                << "allocSol mismatch at idx=" << i
                << " gpu=" << h_allocSolGpu_[i] << " cpu=" << h_allocSolCpu_[i];
        }
    }

    /**
     * @brief Compare the type-0 solution (one UE index per PRG) across the nPrbGrp selected UEs.
     */
    void ExpectType0Match()
    {
        int nScheduled = 0;
        for(uint16_t prg = 0; prg < nPrbGrp_; ++prg)
        {
            const size_t idx = static_cast<size_t>(prg) * nCell_ + cellId_;
            ASSERT_EQ(h_allocSolGpu_[idx], h_allocSolCpu_[idx])
                << "allocSol mismatch at prg=" << prg
                << " gpu=" << h_allocSolGpu_[idx] << " cpu=" << h_allocSolCpu_[idx];
            // Selected UE must be a real, associated UE or -1 (none).
            const int16_t sel = h_allocSolGpu_[idx];
            ASSERT_GE(sel, -1);
            ASSERT_LT(sel, static_cast<int16_t>(nUe_));
            if(sel >= 0)
            {
                EXPECT_EQ(h_cellAssoc_[sel], 1) << "scheduled UE " << sel << " not associated";
                ++nScheduled;
            }
        }
        // A strong channel and low noise guarantee at least one PRG is scheduled,
        // exercising the winner-assignment path.
        EXPECT_GT(nScheduled, 0);
    }

    cudaStream_t stream_ = nullptr;

    std::vector<cuComplex*> d_perUe_;
    std::vector<cuComplex*> h_perUe_;
    std::vector<cuComplex*> h_ptrCpu_;
    cuComplex** d_estHptr_ = nullptr;
    uint8_t*  d_cellAssoc_ = nullptr;
    std::vector<uint8_t> h_cellAssoc_;
    float* d_avgRates_ = nullptr;
    std::vector<float> h_avgRates_;
    cuComplex* d_prdMat_ = nullptr;
    std::vector<cuComplex> h_prdMat_;
    int16_t* d_allocSol_ = nullptr;
    std::vector<int16_t> h_allocSolCpu_;
    std::vector<int16_t> h_allocSolGpu_;
    float* d_pfMetricArr_ = nullptr;
    uint16_t* d_pfIdArr_ = nullptr;

    cumacCellGrpPrms     prmsGpu_{};
    cumacCellGrpPrms     prmsCpu_{};
    cumacCellGrpUeStatus ueStatGpu_{};
    cumacCellGrpUeStatus ueStatCpu_{};
    cumacSchdSol         schdGpu_{};
    cumacSchdSol         schdCpu_{};
    cumacSimParam        sim_{};

    uint16_t nUe_ = 0, nPrbGrp_ = 0;
    uint16_t nCell_ = 1, cellId_ = 0;
    uint8_t  nBsAnt_ = 0, nUeAnt_ = 0;
};

// No precoding, type-0 allocation: each PRG independently selects its best-metric
// UE. GPU must match the CPU reference.
TEST_F(SingleCellSchedulerTest, NoPrecodingType0MatchesCpu)
{
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectType0Match();
}

// SVD precoding, type-0 allocation: per-PRG selection over the effective
// (precoded) channel.
TEST_F(SingleCellSchedulerTest, SvdPrecodingType0MatchesCpu)
{
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/1, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectType0Match();
}

// Type-1 consecutive allocation: sorts all UE/PRG metrics, then assigns each UE a
// contiguous PRG range. Compare the per-UE ranges against the CPU reference.
TEST_F(SingleCellSchedulerTest, Type1ConsecutiveMatchesCpu)
{
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    int nAllocated = 0;
    for(uint16_t u = 0; u < nUe_; ++u)
    {
        if(!h_cellAssoc_[u]) {
            continue;
        }
        const int16_t s = h_allocSolGpu_[2 * u];
        const int16_t e = h_allocSolGpu_[2 * u + 1];
        EXPECT_EQ(s, h_allocSolCpu_[2 * u])     << "type1 start mismatch UE " << u;
        EXPECT_EQ(e, h_allocSolCpu_[2 * u + 1]) << "type1 end mismatch UE " << u;
        if(s >= 0)
        {
            EXPECT_GE(e, s);
            nAllocated += (e - s);
        }
    }
    // Consecutive allocation tiles every PRG exactly once across all UEs.
    EXPECT_EQ(nAllocated, static_cast<int>(nPrbGrp_));
}

// Partial association: some UEs are not associated to the cell, exercising the
// unassociated-UE skip path in selection.
TEST_F(SingleCellSchedulerTest, PartialAssociationType0MatchesCpu)
{
    const std::vector<uint8_t> mask = {1, 0, 1, 1, 0, 1};
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f,
          mask);
    RunBoth();
    ExpectType0Match();
}

// An unrecognized precoding scheme falls back to the no-precoding path; the
// result must still match the CPU reference.
TEST_F(SingleCellSchedulerTest, DefaultPrecodingRoutesToNoPrd)
{
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/2, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectType0Match();
}

// A large associated-UE count makes the kernel discover UEs over multiple
// internal rounds; this drives that multi-round path (no precoding).
TEST_F(SingleCellSchedulerTest, NoPrecodingType0MultiRound)
{
    Build(/*nUe=*/64, /*nPrbGrp=*/2, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectAllocSolFullMatch();
}

// Multi-round discovery for the SVD-precoding kernel.
TEST_F(SingleCellSchedulerTest, SvdPrecodingType0MultiRound)
{
    Build(/*nUe=*/64, /*nPrbGrp=*/2, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/1, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectAllocSolFullMatch();
}

// Multi-round discovery for the type-1 consecutive kernel.
TEST_F(SingleCellSchedulerTest, Type1MultiRound)
{
    Build(/*nUe=*/64, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    RunBoth();
    ExpectAllocSolFullMatch();
}

// Partial association on the SVD path: exercises the unassociated-UE skip path
// with precoding enabled.
TEST_F(SingleCellSchedulerTest, SvdPartialAssociationMatchesCpu)
{
    const std::vector<uint8_t> mask = {1, 0, 1, 1, 0, 1};
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/0, /*precodingScheme=*/1, /*sigmaSqrd=*/0.05f, /*W=*/1.0f,
          mask);
    RunBoth();
    ExpectType0Match();
}

// Partial association with type-1 allocation: skips unassociated UEs while still
// tiling every PRG.
TEST_F(SingleCellSchedulerTest, Type1PartialAssociationMatchesCpu)
{
    const std::vector<uint8_t> mask = {1, 0, 1, 1, 0, 1};
    Build(/*nUe=*/6, /*nPrbGrp=*/8, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f,
          mask);
    RunBoth();
    ExpectAllocSolFullMatch();
}

// No UE associated to the cell: the type-1 kernel finds no candidates and writes
// no allocation.
TEST_F(SingleCellSchedulerTest, Type1NoAssociationReturnsEarly)
{
    const std::vector<uint8_t> mask = {0, 0, 0, 0};
    Build(/*nUe=*/4, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f,
          mask);
    RunBoth();
    ExpectAllocSolFullMatch();
    for(const auto v : h_allocSolGpu_) {
        EXPECT_EQ(v, 0) << "no UE should be allocated";
    }
}

// SVD precoding combined with type-1 allocation is an unsupported combination:
// setup() reports an error and aborts. Sandboxed in a death test.
TEST_F(SingleCellSchedulerTest, SvdType1CombinationIsUnavailableAndAborts)
{
    Build(/*nUe=*/4, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/1, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);

    EXPECT_EXIT(
        {
            dup2(STDERR_FILENO, STDOUT_FILENO);
            singleCellScheduler gpu;
            gpu.setup(cellId_, &ueStatGpu_, &schdGpu_, &prmsGpu_, &sim_, stream_);
            // unreachable: setup() aborts before returning.
        },
        ::testing::ExitedWithCode(EXIT_FAILURE),
        "Error: Kernel function not available");
}

// Type-1 upward extension: with one UE whose PRGs share the same channel, the
// allocator seeds at the lowest PRG and extends its range upward to cover them all.
TEST_F(SingleCellSchedulerTest, Type1UpwardExtensionMatchesCpu)
{
    Build(/*nUe=*/1, /*nPrbGrp=*/4, /*nBsAnt=*/4, /*nUeAnt=*/4,
          /*allocType=*/1, /*precodingScheme=*/0, /*sigmaSqrd=*/0.05f, /*W=*/1.0f);
    MakeUePrgsIdentical(0);
    RunBoth();
    ExpectAllocSolFullMatch();
    EXPECT_EQ(h_allocSolGpu_[0], 0);                              // range start
    EXPECT_EQ(h_allocSolGpu_[1], static_cast<int16_t>(nPrbGrp_)); // range end (exclusive)
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    const int rc = RUN_ALL_TESTS();

    return rc;
}
