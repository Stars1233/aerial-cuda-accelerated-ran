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

// Unit tests for cumac::multiCellMuUeSort (cuMAC/src/64T64R/multiCellMuUeSort.cu).
// The module has no CPU reference, so each test re-implements the kernel's per-UE
// weight + descending bitonic-sort ordering on the host and compares it against the
// GPU's sortedUeList / muMimoInd outputs.
//
// The kernel fully overwrites its outputs each launch, so the tests assert exact
// output values regardless of how many times the kernel is relaunched.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "64T64R/multiCellMuUeSort.cuh"


using namespace cumac;

namespace {

constexpr uint16_t kInvalidUe = 0xFFFF;

// Fixture: holds the synthetic per-UE inputs and the device buffers, builds the
// cumac descriptor structs, runs the module, and compares the GPU output against
// a host re-implementation of the kernel's ordering.
class MuUeSortTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override
    {
        FreeDevice();
        if (stream_) cudaStreamDestroy(stream_);
    }

    void FreeDevice()
    {
        if (d_cellAssoc_)  { cudaFree(d_cellAssoc_);  d_cellAssoc_  = nullptr; }
        if (d_wbSinr_)     { cudaFree(d_wbSinr_);     d_wbSinr_     = nullptr; }
        if (d_srsWbSnr_)   { cudaFree(d_srsWbSnr_);   d_srsWbSnr_   = nullptr; }
        if (d_avgRates_)   { cudaFree(d_avgRates_);   d_avgRates_   = nullptr; }
        if (d_newData_)    { cudaFree(d_newData_);    d_newData_    = nullptr; }
        if (d_bufferSize_) { cudaFree(d_bufferSize_); d_bufferSize_ = nullptr; }
        if (d_muMimoInd_)  { cudaFree(d_muMimoInd_);  d_muMimoInd_  = nullptr; }
        for (auto p : d_cellLists_) { if (p) cudaFree(p); }
        d_cellLists_.clear();
        if (d_sortedUeList_) { cudaFree(d_sortedUeList_); d_sortedUeList_ = nullptr; }
    }

    // Configure one scenario. Host input vectors must already be populated:
    //   h_cellAssoc_ (nCell*nActiveUe), h_wbSinr_ (nActiveUe*nUeAnt),
    //   h_srsWbSnr_/h_avgRates_ (nActiveUe), h_newData_ (nActiveUe, harq only),
    //   h_bufferSize_ (nActiveUe, only if useBuffer).
    void BuildAndAllocate()
    {
        const size_t nAssocBuf = static_cast<size_t>(nCell_) * nActiveUe_;
        const size_t nSinrBuf  = static_cast<size_t>(nActiveUe_) * nUeAnt_;

        ASSERT_EQ(cudaMalloc(&d_cellAssoc_, nAssocBuf * sizeof(uint8_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_wbSinr_, nSinrBuf * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_srsWbSnr_, nActiveUe_ * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_avgRates_, nActiveUe_ * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_muMimoInd_, nActiveUe_ * sizeof(uint8_t)), cudaSuccess);

        ASSERT_EQ(cudaMemcpyAsync(d_cellAssoc_, h_cellAssoc_.data(),
                                  nAssocBuf * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_wbSinr_, h_wbSinr_.data(),
                                  nSinrBuf * sizeof(float), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_srsWbSnr_, h_srsWbSnr_.data(),
                                  nActiveUe_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_avgRates_, h_avgRates_.data(),
                                  nActiveUe_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        // Sentinel so unwritten muMimoInd entries are obviously distinct from 0/1.
        ASSERT_EQ(cudaMemsetAsync(d_muMimoInd_, 0xAB, nActiveUe_ * sizeof(uint8_t), stream_),
                  cudaSuccess);

        if (harq_) {
            ASSERT_EQ(cudaMalloc(&d_newData_, nActiveUe_ * sizeof(int8_t)), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(d_newData_, h_newData_.data(),
                                      nActiveUe_ * sizeof(int8_t), cudaMemcpyHostToDevice, stream_),
                      cudaSuccess);
        }
        if (useBuffer_) {
            ASSERT_EQ(cudaMalloc(&d_bufferSize_, nActiveUe_ * sizeof(uint32_t)), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(d_bufferSize_, h_bufferSize_.data(),
                                      nActiveUe_ * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_),
                      cudaSuccess);
        }

        // sortedUeList is uint16_t**: a device array of nCell device pointers, each
        // pointing at an nMaxActUePerCell buffer. Init each to kInvalidUe (0xFFFF).
        std::vector<uint16_t*> hostPtrs(nCell_, nullptr);
        for (uint16_t c = 0; c < nCell_; ++c) {
            uint16_t* p = nullptr;
            ASSERT_EQ(cudaMalloc(&p, nMaxActUePerCell_ * sizeof(uint16_t)), cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(p, 0xFF, nMaxActUePerCell_ * sizeof(uint16_t), stream_),
                      cudaSuccess);
            d_cellLists_.push_back(p);
            hostPtrs[c] = p;
        }
        ASSERT_EQ(cudaMalloc(&d_sortedUeList_, nCell_ * sizeof(uint16_t*)), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_sortedUeList_, hostPtrs.data(),
                                  nCell_ * sizeof(uint16_t*), cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        // Populate the cumac descriptor structs the module consumes.
        prms_              = cumacCellGrpPrms{};
        prms_.harqEnabledInd = harq_ ? 1 : 0;
        prms_.nCell          = nCell_;
        prms_.nActiveUe      = nActiveUe_;
        prms_.nMaxActUePerCell = nMaxActUePerCell_;
        prms_.nPrbGrp        = 1;
        prms_.nBsAnt         = 64;
        prms_.nUeAnt         = nUeAnt_;
        prms_.W              = W_;
        prms_.betaCoeff      = betaCoeff_;
        prms_.muCoeff        = muCoeff_;
        prms_.srsSnrThr      = srsSnrThr_;
        prms_.cellAssocActUe = d_cellAssoc_;
        prms_.wbSinr         = d_wbSinr_;
        prms_.srsWbSnr       = d_srsWbSnr_;
        prms_.srsUeMap       = nullptr;  // not read by the sort kernel

        ueStatus_                = cumacCellGrpUeStatus{};
        ueStatus_.avgRatesActUe  = d_avgRates_;
        ueStatus_.newDataActUe   = harq_ ? d_newData_ : nullptr;
        ueStatus_.bufferSize     = useBuffer_ ? d_bufferSize_ : nullptr;

        schdSol_              = cumacSchdSol{};
        schdSol_.muMimoInd    = d_muMimoInd_;
        schdSol_.sortedUeList = d_sortedUeList_;
    }

    // Construct the module, run it once, and copy the outputs back to the host.
    void RunModule()
    {
        multiCellMuUeSort mod(&prms_);
        mod.setup(&ueStatus_, &schdSol_, &prms_, stream_);
        mod.run(stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        h_muMimo_.assign(nActiveUe_, 0);
        ASSERT_EQ(cudaMemcpy(h_muMimo_.data(), d_muMimoInd_,
                             nActiveUe_ * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                  cudaSuccess);

        h_sorted_.assign(static_cast<size_t>(nCell_) * nMaxActUePerCell_, 0);
        for (uint16_t c = 0; c < nCell_; ++c) {
            ASSERT_EQ(cudaMemcpy(&h_sorted_[static_cast<size_t>(c) * nMaxActUePerCell_],
                                 d_cellLists_[c], nMaxActUePerCell_ * sizeof(uint16_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
    }

    // Host replica of the kernel's per-UE weight (normal new-data/SU/MU path),
    // matching the device arithmetic: pow(dataRate,beta) * (mu?muCoeff:1) / avgRate.
    float HostWeight(uint16_t u) const
    {
        float dataRate = 0.0f;
        for (uint8_t j = 0; j < nUeAnt_; ++j) {
            dataRate += W_ * std::log2(1.0f + h_wbSinr_[u * nUeAnt_ + j]);
        }
        double base = std::pow(static_cast<double>(dataRate), static_cast<double>(betaCoeff_));
        bool mu = (h_srsWbSnr_[u] >= srsSnrThr_);
        double w = mu ? base * muCoeff_ : base;
        return static_cast<float>(w / h_avgRates_[u]);
    }

    // Verify the GPU output for one cell against the host model.
    void VerifyCell(uint16_t c)
    {
        struct Entry { uint16_t id; float weight; uint8_t mu; };
        std::vector<Entry> assoc;
        for (uint16_t u = 0; u < nActiveUe_; ++u) {
            if (h_cellAssoc_[static_cast<size_t>(c) * nActiveUe_ + u] != 1) continue;
            Entry e{};
            e.id = u;
            if (harq_ && h_newData_[u] == 0) {           // HARQ re-tx -> always SU, top priority
                e.weight = std::numeric_limits<float>::max();
                e.mu     = 0;
            } else {
                e.mu = (h_srsWbSnr_[u] >= srsSnrThr_) ? 1 : 0;
                if (useBuffer_ && h_bufferSize_[u] == 0) {
                    e.weight = -1.0f;                    // bufferSize==0 -> skipped, weight stays -1
                } else {
                    e.weight = HostWeight(u);
                }
            }
            assoc.push_back(e);
        }

        // Kernel sorts descending by weight, ties broken by ascending UE id.
        std::sort(assoc.begin(), assoc.end(), [](const Entry& a, const Entry& b) {
            if (a.weight != b.weight) return a.weight > b.weight;
            return a.id < b.id;
        });

        // assoc should never exceed the per-cell list capacity; surface it if it does
        // and bound the row accesses so we never read past this cell's row.
        ASSERT_LE(assoc.size(), static_cast<size_t>(nMaxActUePerCell_))
            << "cell=" << c << " has more associated UEs than nMaxActUePerCell_";
        const size_t checkCount = std::min(assoc.size(), static_cast<size_t>(nMaxActUePerCell_));

        const uint16_t* row = &h_sorted_[static_cast<size_t>(c) * nMaxActUePerCell_];
        for (size_t i = 0; i < checkCount; ++i) {
            EXPECT_EQ(row[i], assoc[i].id)
                << "cell=" << c << " rank=" << i << " expected ue=" << assoc[i].id
                << " (weight=" << assoc[i].weight << ")";
        }
        for (size_t i = checkCount; i < nMaxActUePerCell_; ++i) {
            EXPECT_EQ(row[i], kInvalidUe)
                << "cell=" << c << " rank=" << i << " should be invalid";
        }

        // muMimoInd is written per associated UE.
        for (const auto& e : assoc) {
            EXPECT_EQ(h_muMimo_[e.id], e.mu)
                << "cell=" << c << " ue=" << e.id << " muMimoInd mismatch";
        }
    }

    void VerifyAllCells()
    {
        for (uint16_t c = 0; c < nCell_; ++c) VerifyCell(c);
    }

    // ---- scenario configuration ----
    uint16_t nCell_           = 1;
    uint16_t nActiveUe_       = 0;
    uint16_t nMaxActUePerCell_ = 8;
    uint8_t  nUeAnt_          = 2;
    float    W_               = 1.0f;
    float    betaCoeff_       = 1.0f;
    float    muCoeff_         = 1.5f;
    float    srsSnrThr_       = 5.0f;
    bool     harq_            = false;
    bool     useBuffer_       = false;

    std::vector<uint8_t>  h_cellAssoc_;
    std::vector<float>    h_wbSinr_;
    std::vector<float>    h_srsWbSnr_;
    std::vector<float>    h_avgRates_;
    std::vector<int8_t>   h_newData_;
    std::vector<uint32_t> h_bufferSize_;

    std::vector<uint8_t>  h_muMimo_;
    std::vector<uint16_t> h_sorted_;

    // ---- device buffers ----
    cudaStream_t stream_       = nullptr;
    uint8_t*     d_cellAssoc_  = nullptr;
    float*       d_wbSinr_     = nullptr;
    float*       d_srsWbSnr_   = nullptr;
    float*       d_avgRates_   = nullptr;
    int8_t*      d_newData_    = nullptr;
    uint32_t*    d_bufferSize_ = nullptr;
    uint8_t*     d_muMimoInd_  = nullptr;
    std::vector<uint16_t*> d_cellLists_;
    uint16_t**   d_sortedUeList_ = nullptr;

    cumacCellGrpPrms     prms_{};
    cumacCellGrpUeStatus ueStatus_{};
    cumacSchdSol         schdSol_{};

    // Helper: set both layers of a UE to the same SINR.
    void SetSinr(uint16_t u, float s) { for (uint8_t j = 0; j < nUeAnt_; ++j) h_wbSinr_[u * nUeAnt_ + j] = s; }
};

// Non-HARQ: 5 associated UEs in one cell with distinct, well-separated weights and a
// mix of MU/SU, driving the full bitonic-sort network. Checks the non-HARQ ordering,
// both weight branches, muMimoInd, and the bufferSize==nullptr case.
TEST_F(MuUeSortTest, NonHarqSortsFiveUesByDescendingWeight)
{
    nCell_ = 1; nActiveUe_ = 5; nMaxActUePerCell_ = 8; harq_ = false; useBuffer_ = false;
    h_cellAssoc_.assign(nActiveUe_, 1);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);

    SetSinr(0, 1.0f);   h_srsWbSnr_[0] = 10.0f;  // MU
    SetSinr(1, 3.0f);   h_srsWbSnr_[1] = 0.0f;   // SU
    SetSinr(2, 7.0f);   h_srsWbSnr_[2] = 10.0f;  // MU
    SetSinr(3, 15.0f);  h_srsWbSnr_[3] = 0.0f;   // SU
    SetSinr(4, 0.5f);   h_srsWbSnr_[4] = 10.0f;  // MU

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

// Non-HARQ: a single associated UE — the minimal sort case.
TEST_F(MuUeSortTest, NonHarqSingleUeHitsSmallestPow2Base)
{
    nCell_ = 1; nActiveUe_ = 1; nMaxActUePerCell_ = 8; harq_ = false; useBuffer_ = false;
    h_cellAssoc_.assign(nActiveUe_, 1);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);
    SetSinr(0, 4.0f); h_srsWbSnr_[0] = 10.0f;  // MU

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

// Non-HARQ with bufferSize provided: UEs with bufferSize==0 are skipped (weight
// stays -1, sorted to the bottom but still listed).
TEST_F(MuUeSortTest, NonHarqBufferSizeZeroIsDeprioritized)
{
    nCell_ = 1; nActiveUe_ = 3; nMaxActUePerCell_ = 8; harq_ = false; useBuffer_ = true;
    h_cellAssoc_.assign(nActiveUe_, 1);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);
    h_bufferSize_.assign(nActiveUe_, 0);

    SetSinr(0, 7.0f);  h_srsWbSnr_[0] = 10.0f; h_bufferSize_[0] = 0;   // skipped
    SetSinr(1, 7.0f);  h_srsWbSnr_[1] = 0.0f;  h_bufferSize_[1] = 100; // normal SU
    SetSinr(2, 7.0f);  h_srsWbSnr_[2] = 10.0f; h_bufferSize_[2] = 0;   // skipped

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

// Non-HARQ, two cells where the second cell has no associated UE, so its list stays
// all-invalid (no sort).
TEST_F(MuUeSortTest, NonHarqEmptySecondCellProducesAllInvalid)
{
    nCell_ = 2; nActiveUe_ = 4; nMaxActUePerCell_ = 8; harq_ = false; useBuffer_ = false;
    h_cellAssoc_.assign(static_cast<size_t>(nCell_) * nActiveUe_, 0);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);

    // Cell 0 associates UEs 0,1,2; cell 1 associates none.
    h_cellAssoc_[0 * nActiveUe_ + 0] = 1;
    h_cellAssoc_[0 * nActiveUe_ + 1] = 1;
    h_cellAssoc_[0 * nActiveUe_ + 2] = 1;

    SetSinr(0, 1.0f);  h_srsWbSnr_[0] = 10.0f;
    SetSinr(1, 7.0f);  h_srsWbSnr_[1] = 0.0f;
    SetSinr(2, 3.0f);  h_srsWbSnr_[2] = 10.0f;

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

// HARQ: mix of new-data and re-tx UEs. Re-tx UEs (newDataActUe==0) get FLT_MAX weight
// and sort to the top; new-data UEs use the normal weight. Checks HARQ ordering and
// muMimoInd.
TEST_F(MuUeSortTest, HarqRetxUesSortToTop)
{
    nCell_ = 1; nActiveUe_ = 4; nMaxActUePerCell_ = 8; harq_ = true; useBuffer_ = false;
    h_cellAssoc_.assign(nActiveUe_, 1);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);
    h_newData_.assign(nActiveUe_, 1);

    SetSinr(0, 3.0f);  h_srsWbSnr_[0] = 10.0f; h_newData_[0] = 1;  // new data, MU
    SetSinr(1, 0.0f);  h_srsWbSnr_[1] = 0.0f;  h_newData_[1] = 0;  // re-tx
    SetSinr(2, 15.0f); h_srsWbSnr_[2] = 0.0f;  h_newData_[2] = 1;  // new data, SU
    SetSinr(3, 0.0f);  h_srsWbSnr_[3] = 0.0f;  h_newData_[3] = 0;  // re-tx

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

// HARQ with bufferSize: a new-data UE with bufferSize==0 is skipped.
TEST_F(MuUeSortTest, HarqNewDataBufferSizeZeroIsDeprioritized)
{
    nCell_ = 1; nActiveUe_ = 2; nMaxActUePerCell_ = 8; harq_ = true; useBuffer_ = true;
    h_cellAssoc_.assign(nActiveUe_, 1);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);
    h_newData_.assign(nActiveUe_, 1);
    h_bufferSize_.assign(nActiveUe_, 0);

    SetSinr(0, 7.0f);  h_srsWbSnr_[0] = 10.0f; h_newData_[0] = 1; h_bufferSize_[0] = 0;   // skipped
    SetSinr(1, 7.0f);  h_srsWbSnr_[1] = 0.0f;  h_newData_[1] = 1; h_bufferSize_[1] = 100; // normal

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();

    // debugLog() is a no-op but part of the public surface; exercise it once.
    multiCellMuUeSort mod(&prms_);
    mod.debugLog();
}

// HARQ, two cells: cell 0 associates only a subset of the UEs and cell 1 associates
// none. Exercises an unassociated UE and an empty cell on the HARQ path.
TEST_F(MuUeSortTest, HarqUnassociatedUeAndEmptyCell)
{
    nCell_ = 2; nActiveUe_ = 4; nMaxActUePerCell_ = 8; harq_ = true; useBuffer_ = false;
    h_cellAssoc_.assign(static_cast<size_t>(nCell_) * nActiveUe_, 0);
    h_wbSinr_.assign(static_cast<size_t>(nActiveUe_) * nUeAnt_, 0.0f);
    h_srsWbSnr_.assign(nActiveUe_, 0.0f);
    h_avgRates_.assign(nActiveUe_, 1.0f);
    h_newData_.assign(nActiveUe_, 1);

    // Cell 0 associates UE0 (new data) and UE2 (re-tx); UE1, UE3 unassociated.
    // Cell 1 associates nobody.
    h_cellAssoc_[0 * nActiveUe_ + 0] = 1;
    h_cellAssoc_[0 * nActiveUe_ + 2] = 1;

    SetSinr(0, 7.0f);  h_srsWbSnr_[0] = 10.0f; h_newData_[0] = 1;  // new data, MU
    SetSinr(2, 0.0f);  h_srsWbSnr_[2] = 0.0f;  h_newData_[2] = 0;  // re-tx

    BuildAndAllocate();
    RunModule();
    VerifyAllCells();
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();

    return rc;
}
