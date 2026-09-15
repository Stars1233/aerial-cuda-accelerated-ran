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
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "4T4R/cellAssociation.cuh"
#include "4T4R/cellAssociationCpu.h"


using namespace cumac;

namespace {

// Fixture allocates a stream and a synthetic channel matrix on the GPU.
// BuildPrms() sizes the buffers for each scenario under test.
class CellAssociationTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override
    {
        if(d_estH_fr_ != nullptr) cudaFree(d_estH_fr_);
        if(d_cellAssoc_ != nullptr) cudaFree(d_cellAssoc_);
        if(stream_) cudaStreamDestroy(stream_);
    }

    /**
     * @brief Allocate device buffers and populate cell group/simulation parameters.
     *
     * The estH_fr buffer is filled with random gains so the kernel has a
     * nontrivial argmax to find.
     *
     * @param totNumCell Total number of cells to include in the association.
     * @param nUe Number of UEs per cell.
     * @param nPrbGrp Number of PRB groups in the channel buffer.
     * @param nBsAnt Number of base-station antennas.
     * @param nUeAnt Number of UE antennas.
     */
    void BuildPrms(const uint16_t totNumCell,
                   const uint16_t nUe,
                   const uint16_t nPrbGrp,
                   const uint8_t  nBsAnt,
                   const uint8_t  nUeAnt)
    {
        totNumCell_ = totNumCell;
        nUe_        = nUe;
        nPrbGrp_    = nPrbGrp;
        nBsAnt_     = nBsAnt;
        nUeAnt_     = nUeAnt;

        const size_t nChan = static_cast<size_t>(nPrbGrp) * nUe * totNumCell * nBsAnt * nUeAnt;
        const size_t nAssoc = static_cast<size_t>(totNumCell) * nUe;

        h_estH_fr_.resize(nChan);
        // Fixed seed so CPU/GPU comparisons are reproducible across runs;
        // the specific value is arbitrary.
        constexpr uint32_t kRngSeed = 0xC0FFEEu;
        std::mt19937 rng(kRngSeed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        const float kInvSqrt2 = 1.0f / std::sqrt(2.0f);
        for(auto& c : h_estH_fr_) { c.x = dist(rng) * kInvSqrt2; c.y = dist(rng) * kInvSqrt2; }
        h_cellAssoc_cpu_.assign(nAssoc, 0);
        h_cellAssoc_gpu_.assign(nAssoc, 0);

        ASSERT_EQ(cudaMalloc(&d_estH_fr_, nChan * sizeof(cuComplex)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_cellAssoc_, nAssoc * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_estH_fr_.data(),
                                  nChan * sizeof(cuComplex),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_cellAssoc_, 0, nAssoc * sizeof(uint8_t), stream_),
                  cudaSuccess);

        // GPU-side parameters point at device buffers.
        cellGrpPrmsGpu_.estH_fr   = d_estH_fr_;
        cellGrpPrmsGpu_.cellAssoc = d_cellAssoc_;
        cellGrpPrmsGpu_.nUe       = nUe;
        cellGrpPrmsGpu_.nPrbGrp   = nPrbGrp;
        cellGrpPrmsGpu_.nBsAnt    = nBsAnt;
        cellGrpPrmsGpu_.nUeAnt    = nUeAnt;
        cellGrpPrmsGpu_.nCell     = totNumCell;

        // CPU-side parameters share the same host channel buffer.
        cellGrpPrmsCpu_.estH_fr   = h_estH_fr_.data();
        cellGrpPrmsCpu_.cellAssoc = h_cellAssoc_cpu_.data();
        cellGrpPrmsCpu_.nUe       = nUe;
        cellGrpPrmsCpu_.nPrbGrp   = nPrbGrp;
        cellGrpPrmsCpu_.nBsAnt    = nBsAnt;
        cellGrpPrmsCpu_.nUeAnt    = nUeAnt;
        cellGrpPrmsCpu_.nCell     = totNumCell;

        simParam_.totNumCell = totNumCell;
    }

    // Copy GPU result back and verify it matches the CPU reference.
    // 'cellAssocCpu' should already have run.
    void CompareCpuGpu()
    {
        const size_t nAssoc = static_cast<size_t>(totNumCell_) * nUe_;
        ASSERT_EQ(cudaMemcpyAsync(h_cellAssoc_gpu_.data(), d_cellAssoc_,
                                  nAssoc * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        // Full element-wise compare so any stray/garbage byte fails the test
        // — the per-UE "one winner" check below only inspects entries equal
        // to 1 and would otherwise miss other discrepancies.
        for(size_t i = 0; i < nAssoc; ++i)
        {
            ASSERT_EQ(h_cellAssoc_cpu_[i], h_cellAssoc_gpu_[i])
                << "cellAssoc mismatch at idx=" << i
                << " cpu=" << static_cast<int>(h_cellAssoc_cpu_[i])
                << " gpu=" << static_cast<int>(h_cellAssoc_gpu_[i]);
        }

        // Each UE should have exactly one cell selected on both sides.
        // The selected cell must be the same.
        for(uint16_t ue = 0; ue < nUe_; ++ue)
        {
            int cpuChosen = -1, gpuChosen = -1;
            int cpuOnes = 0, gpuOnes = 0;
            for(uint16_t cell = 0; cell < totNumCell_; ++cell)
            {
                const size_t idx = static_cast<size_t>(cell) * nUe_ + ue;
                if(h_cellAssoc_cpu_[idx] == 1) { cpuChosen = cell; ++cpuOnes; }
                if(h_cellAssoc_gpu_[idx] == 1) { gpuChosen = cell; ++gpuOnes; }
            }
            ASSERT_EQ(cpuOnes, 1) << "ue=" << ue << " CPU did not pick exactly one cell";
            ASSERT_EQ(gpuOnes, 1) << "ue=" << ue << " GPU did not pick exactly one cell";
            ASSERT_EQ(cpuChosen, gpuChosen) << "ue=" << ue << " CPU/GPU chose different cells";
        }
    }

    cudaStream_t        stream_         = nullptr;
    cuComplex*          d_estH_fr_      = nullptr;
    uint8_t*            d_cellAssoc_    = nullptr;
    std::vector<cuComplex> h_estH_fr_;
    std::vector<uint8_t>   h_cellAssoc_cpu_;
    std::vector<uint8_t>   h_cellAssoc_gpu_;
    cumacCellGrpPrms    cellGrpPrmsGpu_{};
    cumacCellGrpPrms    cellGrpPrmsCpu_{};
    cumacSimParam       simParam_{};
    uint16_t totNumCell_ = 0;
    uint16_t nUe_        = 0;
    uint16_t nPrbGrp_    = 0;
    uint8_t  nBsAnt_     = 0;
    uint8_t  nUeAnt_     = 0;
};

// Small PRBG/cell combinations use the per-PRBG association kernel.
// With dims 4 cells * 8 PRBGs = 32 <= 1024.
TEST_F(CellAssociationTest, SetupPicksParaPrbgCellKernelOnSmallDims)
{
    BuildPrms(/*totNumCell=*/4, /*nUe=*/2, /*nPrbGrp=*/8,
              /*nBsAnt=*/4,    /*nUeAnt=*/4);

    cellAssociation<cuComplex> gpu;
    cellAssociationCpu         cpu;

    gpu.setup(&cellGrpPrmsGpu_, &simParam_, stream_);
    cpu.setup(&cellGrpPrmsCpu_, &simParam_);

    gpu.run(stream_);
    cpu.run();
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    CompareCpuGpu();
}

// Builds deterministic metrics where the strongest cell is carried by the
// second reduction slot, which complements the random-channel cases.
// The two UEs arrange their strongest cells in different pair positions.
//
// Per-cell metrics (from PRBG 0 only, all other PRBGs zeroed) are:
//   UE 0: {m0,m1,m2,m3} = {5,10,2,1}  -> winner cell 1.
//   UE 1: {m0,m1,m2,m3} = {1, 2,5,10} -> winner cell 3.
TEST_F(CellAssociationTest, ParaPrbgCellKernelFinalCompareSelectsCellAssocIdx1)
{
    constexpr uint16_t kTotNumCell = 4;
    constexpr uint16_t kNUe        = 2;
    constexpr uint16_t kNPrbGrp    = 8;
    constexpr uint8_t  kNBsAnt     = 4;
    constexpr uint8_t  kNUeAnt     = 4;
    BuildPrms(kTotNumCell, kNUe, kNPrbGrp, kNBsAnt, kNUeAnt);

    // Zero the whole channel buffer, then deposit the per-cell metric into
    // PRBG 0, antenna (0,0). Layout matches the kernel's chanOffsetAnt:
    // [prbg][ue][cell][txAnt][rxAnt].
    std::fill(h_estH_fr_.begin(), h_estH_fr_.end(), make_cuComplex(0.0f, 0.0f));
    const float kMetrics[kNUe][kTotNumCell] = {
        {5.0f, 10.0f, 2.0f, 1.0f},
        {1.0f,  2.0f, 5.0f, 10.0f},
    };
    const size_t kCellStride = static_cast<size_t>(kNBsAnt) * kNUeAnt;
    const size_t kUeStride   = static_cast<size_t>(kTotNumCell) * kCellStride;
    for(uint16_t u = 0; u < kNUe; ++u)
    {
        for(uint16_t c = 0; c < kTotNumCell; ++c)
        {
            const size_t idx = u * kUeStride + c * kCellStride;  // prbg=0, ant=(0,0)
            h_estH_fr_[idx].x = std::sqrt(kMetrics[u][c]);
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(d_estH_fr_, h_estH_fr_.data(),
                              h_estH_fr_.size() * sizeof(cuComplex),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    cellAssociation<cuComplex> gpu;
    cellAssociationCpu         cpu;
    gpu.setup(&cellGrpPrmsGpu_, &simParam_, stream_);
    cpu.setup(&cellGrpPrmsCpu_, &simParam_);

    gpu.run(stream_);
    cpu.run();
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    CompareCpuGpu();

    // cellAssoc layout: cell-major, i.e. cellAssoc[cell * nUe + ue].
    EXPECT_EQ(h_cellAssoc_gpu_[1u * kNUe + 0u], 1) << "UE 0 should pick cell 1";
    EXPECT_EQ(h_cellAssoc_gpu_[3u * kNUe + 1u], 1) << "UE 1 should pick cell 3";
}

// Larger PRBG/cell combinations use the per-cell association kernel.
// With dims 20 cells * 68 PRBGs = 1360 > 1024.
// Mirrors the production parameters.h default (numCellConst=20, nPrbGrpsConst=68).
// Also verifies getCellAssociaResGpu() returns the descriptor pointer.
TEST_F(CellAssociationTest, SetupPicksParaCellKernelOnDefaultDims)
{
    BuildPrms(/*totNumCell=*/20, /*nUe=*/8, /*nPrbGrp=*/68,
              /*nBsAnt=*/4,     /*nUeAnt=*/4);

    cellAssociation<cuComplex> gpu;
    cellAssociationCpu         cpu;

    gpu.setup(&cellGrpPrmsGpu_, &simParam_, stream_);
    cpu.setup(&cellGrpPrmsCpu_, &simParam_);

    gpu.run(stream_);
    cpu.run();
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    CompareCpuGpu();

    // getCellAssociaResGpu() should hand back the same GPU pointer that
    // BuildPrms() put into the descriptor.
    EXPECT_EQ(gpu.getCellAssociaResGpu(), d_cellAssoc_);
}

// setup() exits when totNumCell exceeds the supported limit.
// Use a Google Test death test so the exit is sandboxed in a forked child
// and RUN_ALL_TESTS() can continue.
TEST_F(CellAssociationTest, SetupOverflowsAndExitsWhenTotNumCellExceedsLimit)
{
    // Smallest valid descriptor — the overflow check fires before any kernel
    // launch, so the channel tensor's shape doesn't matter, only that the
    // cudaMemcpyAsync of the descriptor itself succeeds.
    BuildPrms(/*totNumCell=*/2, /*nUe=*/1, /*nPrbGrp=*/2,
              /*nBsAnt=*/1,    /*nUeAnt=*/1);

    EXPECT_EXIT(
        {
            dup2(STDERR_FILENO, STDOUT_FILENO);
            cellAssociation<cuComplex> gpu;
            cumacSimParam              simParamOverflow = simParam_;
            simParamOverflow.totNumCell = 1025;  // > 1024 exceeds supported limit.
            gpu.setup(&cellGrpPrmsGpu_, &simParamOverflow, stream_);
            // unreachable
        },
        ::testing::ExitedWithCode(1),
        "Max supported totalNumCell is 1024");
}

// Verifies that the cellAssociation<__half2> instantiation is wired correctly.
TEST_F(CellAssociationTest, HalfPrecisionInstantiationConstructsAndDestructs)
{
    BuildPrms(/*totNumCell=*/4, /*nUe=*/2, /*nPrbGrp=*/8,
              /*nBsAnt=*/4,    /*nUeAnt=*/4);

    // Construct, set up, run, and destruct the __half2 specialization.
    {
        cellAssociation<__half2> gpu;
        gpu.setup(&cellGrpPrmsGpu_, &simParam_, stream_);
        gpu.run(stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    SUCCEED();
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // CUDA is not fork-safe; the default "fast" death-test style forks without
    // exec, leaving the child with an invalid CUDA context. "threadsafe" does
    // fork+exec so the child re-initializes CUDA from scratch.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    const int rc = RUN_ALL_TESTS();

    return rc;
}
