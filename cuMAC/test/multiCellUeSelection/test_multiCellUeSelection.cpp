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
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "cumac.h"

// Optional VectorCAST coverage flush. Resolved as a weak symbol so the same
// test binary links both against the instrumented cumac (where this is
// defined by VectorCAST) and the regular build (where it stays a no-op).

namespace
{

// ---------------------------------------------------------------------
// Device-memory helpers (RAII)
struct DeviceFree
{
    void operator()(void* p) const
    {
        if (p) {
            cudaFree(p);
        }
    }
};
using device_ptr = std::unique_ptr<void, DeviceFree>;

template <typename T>
device_ptr alloc_and_copy(const std::vector<T>& host)
{
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
              cudaSuccess);
    return device_ptr(d);
}

template <typename T>
device_ptr alloc_device(size_t count)
{
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, count * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemset(d, 0, count * sizeof(T)), cudaSuccess);
    return device_ptr(d);
}

// ---------------------------------------------------------------------
// Scenario holder. Owns all device buffers + the C-API parameter structs,
// so each test can mutate fields independently before constructing the
// multiCellUeSelection object.
struct Scenario
{
    // Layout
    uint16_t nCell                = 1;
    uint16_t nActiveUe            = 4;
    uint8_t  numUeSchdPerCellTTI  = 2;
    uint8_t  nUeAnt               = 1;
    uint8_t  harqEnabledInd       = 0;
    float    W                    = 1.0f;
    float    betaCoeff            = 1.0f;

    // Host-side raw arrays used to populate device buffers
    std::vector<uint16_t> cellId;
    std::vector<uint8_t>  cellAssocActUe;        // size nCell*nActiveUe
    std::vector<float>    wbSinr;                // size nActiveUe*nUeAnt
    std::vector<float>    avgRatesActUe;         // size nActiveUe
    std::vector<uint8_t>  numUeSchdPerCellTTIArrHost; // optional, size nCell
    std::vector<uint32_t> bufferSizeHost;        // optional, size nActiveUe
    std::vector<int8_t>   newDataActUeHost;      // optional, size nActiveUe

    // Device buffers
    device_ptr d_cellId, d_cellAssoc, d_wbSinr, d_avgRates;
    device_ptr d_numUeSchdPerCellTTIArr;     // null unless set
    device_ptr d_bufferSize;                 // null unless set
    device_ptr d_newDataActUe;               // null unless set
    device_ptr d_setSchdUePerCellTTI;        // output

    cumac::cumacCellGrpPrms    prms{};
    cumac::cumacCellGrpUeStatus ueStatus{};
    cumac::cumacSchdSol         schdSol{};

    void initDefaultAssocAndSinr()
    {
        cellId.assign(nCell, 0);
        for (uint16_t c = 0; c < nCell; ++c) {
            cellId[c] = c;
        }

        cellAssocActUe.assign(static_cast<size_t>(nCell) * nActiveUe, 1);

        wbSinr.assign(static_cast<size_t>(nActiveUe) * nUeAnt, 0.0f);
        for (uint16_t u = 0; u < nActiveUe; ++u) {
            for (uint8_t a = 0; a < nUeAnt; ++a) {
                // Monotonically decreasing SINR so UE0 > UE1 > ...
                wbSinr[u * nUeAnt + a] = static_cast<float>(nActiveUe - u);
            }
        }
        avgRatesActUe.assign(nActiveUe, 1.0f);
    }

    void uploadCommon()
    {
        d_cellId    = alloc_and_copy(cellId);
        d_cellAssoc = alloc_and_copy(cellAssocActUe);
        d_wbSinr    = alloc_and_copy(wbSinr);
        d_avgRates  = alloc_and_copy(avgRatesActUe);
        d_setSchdUePerCellTTI = alloc_device<uint16_t>(
            static_cast<size_t>(nCell) * numUeSchdPerCellTTI);

        prms.nCell                = nCell;
        prms.nActiveUe            = nActiveUe;
        prms.numUeSchdPerCellTTI  = numUeSchdPerCellTTI;
        prms.nUeAnt               = nUeAnt;
        prms.W                    = W;
        prms.betaCoeff            = betaCoeff;
        prms.harqEnabledInd       = harqEnabledInd;
        prms.cellId               = static_cast<uint16_t*>(d_cellId.get());
        prms.cellAssocActUe       = static_cast<uint8_t*>(d_cellAssoc.get());
        prms.wbSinr               = static_cast<float*>(d_wbSinr.get());
        prms.numUeSchdPerCellTTIArr = nullptr;

        ueStatus.avgRatesActUe = static_cast<float*>(d_avgRates.get());
        ueStatus.bufferSize    = nullptr;
        ueStatus.newDataActUe  = nullptr;

        schdSol.setSchdUePerCellTTI =
            static_cast<uint16_t*>(d_setSchdUePerCellTTI.get());
    }

    void enableHeterogeneous(const std::vector<uint8_t>& perCellCap)
    {
        ASSERT_EQ(perCellCap.size(), static_cast<size_t>(nCell));
        numUeSchdPerCellTTIArrHost = perCellCap;
        d_numUeSchdPerCellTTIArr = alloc_and_copy(numUeSchdPerCellTTIArrHost);
        prms.numUeSchdPerCellTTIArr =
            static_cast<uint8_t*>(d_numUeSchdPerCellTTIArr.get());
    }

    void enableBufferSize(const std::vector<uint32_t>& bs)
    {
        ASSERT_EQ(bs.size(), static_cast<size_t>(nActiveUe));
        bufferSizeHost = bs;
        d_bufferSize = alloc_and_copy(bufferSizeHost);
        ueStatus.bufferSize = static_cast<uint32_t*>(d_bufferSize.get());
    }

    void enableHarq(const std::vector<int8_t>& newData)
    {
        ASSERT_EQ(newData.size(), static_cast<size_t>(nActiveUe));
        harqEnabledInd = 1;
        prms.harqEnabledInd = 1;
        newDataActUeHost = newData;
        d_newDataActUe = alloc_and_copy(newDataActUeHost);
        ueStatus.newDataActUe = static_cast<int8_t*>(d_newDataActUe.get());
    }

    std::vector<uint16_t> readSchdUes() const
    {
        std::vector<uint16_t> h(static_cast<size_t>(nCell) * numUeSchdPerCellTTI);
        EXPECT_EQ(cudaMemcpy(h.data(), d_setSchdUePerCellTTI.get(),
                             h.size() * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return h;
    }
};

// ---------------------------------------------------------------------
class MultiCellUeSelectionTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};

    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    }
    void TearDown() override
    {
        cudaStreamDestroy(stream);
        // Flush VectorCAST coverage data when running under the instrumented
        // build; no-op otherwise (weak symbol).
    }

    void runSelection(Scenario& s)
    {
        // Constructor reads harqEnabledInd; ensure prms is populated first.
        s.initDefaultAssocAndSinr();
        s.uploadCommon();
    }

    void execute(Scenario& s, cumac::multiCellUeSelection& sel)
    {
        sel.setup(&s.ueStatus, &s.schdSol, &s.prms, stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        sel.run(stream);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
};

// ---------------------------------------------------------------------
// Baseline: HARQ off, bufferSize null, homogeneous selection.
// Validates default behavior.
TEST_F(MultiCellUeSelectionTest, HomogeneousBasic_SelectsHighestPriorityUes)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    // SINR is monotonically decreasing in UE index; top-2 are UE 0 and UE 1.
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0u);
    EXPECT_EQ(out[1], 1u);
}

// ---------------------------------------------------------------------
// Heterogeneous scheduling when per-cell caps are provided.
TEST_F(MultiCellUeSelectionTest, Heterogeneous_NumUeSchdPerCellTTIArr_CapsPerCell)
{
    Scenario s;
    s.nCell = 2; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 3; s.nUeAnt = 1;
    runSelection(s);

    // Cell 0 should pick 1 UE, cell 1 should pick 2 UEs.
    s.enableHeterogeneous({1, 2});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 6u);

    // Both cells see the same UE pool (all UEs associated by default).
    // Top SINR is UE 0; cell 0 keeps only slot 0, slots 1-2 become sentinel.
    EXPECT_EQ(out[0], 0u);
    EXPECT_EQ(out[1], 0xFFFFu);
    EXPECT_EQ(out[2], 0xFFFFu);
    // Cell 1 keeps top 2, slot 2 is sentinel.
    EXPECT_EQ(out[3], 0u);
    EXPECT_EQ(out[4], 1u);
    EXPECT_EQ(out[5], 0xFFFFu);
}

// ---------------------------------------------------------------------
// HARQ-enabled retransmission prioritization.
TEST_F(MultiCellUeSelectionTest, HarqEnabled_RetransmissionForcesMaxPriority)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    // UE 3 has lowest SINR by default. Mark it as a retransmission so it
    // gets std::numeric_limits<float>::max() priority and is selected first.
    s.enableHarq({1, 1, 1, /*retx*/ 0});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 3u);  // retx wins
    EXPECT_EQ(out[1], 0u);  // next-highest by SINR
}

// ---------------------------------------------------------------------
// Heterogeneous scheduling with HARQ retransmission.
TEST_F(MultiCellUeSelectionTest, Heterogeneous_HarqRetx_CoversHeteroRetxBranch)
{
    Scenario s;
    s.nCell = 2; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 3; s.nUeAnt = 1;
    runSelection(s);

    // Cell 0 keeps 1 UE, cell 1 keeps 2 UEs.
    s.enableHeterogeneous({1, 2});
    // UE 3 has the lowest SINR; mark it as a retransmission so the hetero
    // kernel takes the else-branch and assigns float::max priority.
    s.enableHarq({1, 1, 1, /*retx*/ 0});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 6u);

    // Cell 0: only slot 0 is kept; the retx UE (3) wins on max priority.
    EXPECT_EQ(out[0], 3u);
    EXPECT_EQ(out[1], 0xFFFFu);
    EXPECT_EQ(out[2], 0xFFFFu);
    // Cell 1: top 2 are the retx UE (3) and the highest-SINR UE (0).
    EXPECT_EQ(out[3], 3u);
    EXPECT_EQ(out[4], 0u);
    EXPECT_EQ(out[5], 0xFFFFu);
}

// ---------------------------------------------------------------------
// Non-null buffer filtering behavior.
TEST_F(MultiCellUeSelectionTest, BufferSizeNonNull_FiltersZeroBufferUe)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    // UE 0 has the highest SINR but an empty buffer; it must be skipped.
    s.enableBufferSize({/*UE0*/ 0u, 100u, 100u, 100u});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 2u);
    // UE 0 must never appear.
    EXPECT_NE(out[0], 0u);
    EXPECT_NE(out[1], 0u);
}

// ---------------------------------------------------------------------
// Single associated UE case.
TEST_F(MultiCellUeSelectionTest, Homogeneous_SingleCandidate_K1Pow2Path)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    // Only UE 2 is associated.
    std::fill(s.cellAssocActUe.begin(), s.cellAssocActUe.end(), 0);
    s.cellAssocActUe[2] = 1;
    ASSERT_EQ(cudaMemcpy(s.d_cellAssoc.get(), s.cellAssocActUe.data(),
                         s.cellAssocActUe.size(), cudaMemcpyHostToDevice),
              cudaSuccess);

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 2u);
    EXPECT_EQ(out[1], 0xFFFFu);
}

// ---------------------------------------------------------------------
// No UE associated with the cell; output should be sentinels.
TEST_F(MultiCellUeSelectionTest, Homogeneous_NoCandidates_ProducesSentinel)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    std::fill(s.cellAssocActUe.begin(), s.cellAssocActUe.end(), 0);
    ASSERT_EQ(cudaMemcpy(s.d_cellAssoc.get(), s.cellAssocActUe.data(),
                         s.cellAssocActUe.size(), cudaMemcpyHostToDevice),
              cudaSuccess);

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0xFFFFu);
    EXPECT_EQ(out[1], 0xFFFFu);
}

// ---------------------------------------------------------------------
// Proportional-fair metric: priority = dataRate / avgRatesActUe. A UE with
// the LOWEST SINR must outrank the highest-SINR UE once its average rate is
// small enough. All previous tests hold avgRatesActUe == 1, so this is the
// only case that exercises the denominator of the PF metric.
TEST_F(MultiCellUeSelectionTest, PfMetric_LowAvgRateOutranksHighSinr)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    s.initDefaultAssocAndSinr();
    // Default SINR is monotonically decreasing (UE0 highest, UE3 lowest).
    // Give UE3 a tiny average rate so dataRate/avgRate dominates: its metric
    // becomes ~1.0/0.1 = 10, far above UE0's ~log2(5)/1 = 2.32.
    s.avgRatesActUe = {1.0f, 1.0f, 1.0f, 0.1f};
    s.uploadCommon();

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 3u);  // lowest SINR, but smallest avg rate -> highest PF metric
    EXPECT_EQ(out[1], 0u);  // next is the highest-SINR UE
}

// ---------------------------------------------------------------------
// Multi-antenna SINR aggregation: dataRate sums log2(1+SINR) across nUeAnt
// antennas. The per-antenna profile is chosen so the ranking by the antenna
// SUM differs from the ranking by antenna 0 alone, proving the summation
// loop is exercised (nUeAnt == 1 everywhere else).
TEST_F(MultiCellUeSelectionTest, MultiAntenna_SumsRatePerAntenna)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 2;
    s.initDefaultAssocAndSinr();
    // Layout is wbSinr[ue * nUeAnt + ant].
    //   UE0 [1 , 1 ] -> log2(2)+log2(2)          = 2.000
    //   UE1 [0.5, 15] -> log2(1.5)+log2(16)       = 4.585
    //   UE2 [3 , 3 ] -> log2(4)+log2(4)           = 4.000
    //   UE3 [0 , 0 ] -> log2(1)+log2(1)           = 0.000
    // By antenna-0 alone UE2 would win; by the per-antenna SUM, UE1 > UE2.
    s.wbSinr = {1.0f, 1.0f, 0.5f, 15.0f, 3.0f, 3.0f, 0.0f, 0.0f};
    s.uploadCommon();

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 1u);  // wins only because antenna 1 is summed in
    EXPECT_EQ(out[1], 2u);
}

// ---------------------------------------------------------------------
// HARQ retransmissions bypass the zero-buffer filter. A retx UE is forced to
// max priority before the bufferSize check, so it is selected even with an
// empty buffer, while a *new-transmission* UE with an empty buffer is still
// filtered out.
TEST_F(MultiCellUeSelectionTest, HarqRetx_BypassesZeroBufferFilter)
{
    Scenario s;
    s.nCell = 1; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    // newDataActUe: 1 == new transmission, 0 == retransmission -> UE3 is retx.
    s.enableHarq({1, 1, 1, /*retx*/ 0});
    // UE0 (highest SINR) has an empty buffer and is a new tx -> must be dropped.
    // UE3 has an empty buffer but is a retx -> must still be selected.
    s.enableBufferSize({/*UE0*/ 0u, 100u, 100u, /*UE3*/ 0u});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 3u);  // retx wins despite empty buffer
    EXPECT_EQ(out[1], 1u);  // best new tx with a non-empty buffer
    EXPECT_NE(out[0], 0u);  // UE0 filtered: empty buffer + new tx
    EXPECT_NE(out[1], 0u);
}

// ---------------------------------------------------------------------
// betaCoeff is the exponent applied to dataRate: metric = dataRate^beta /
// avgRate. With dataRate/avgRate chosen so the two UEs cross over, changing
// only betaCoeff flips the winner. Every other test uses betaCoeff == 1.
TEST_F(MultiCellUeSelectionTest, BetaCoeff_ExponentAltersPriorityOrder)
{
    // SINR 3 -> dataRate log2(4) = 2; SINR 15 -> dataRate log2(16) = 4.
    // UE0: dataRate 2, avgRate 1 ;  UE1: dataRate 4, avgRate 3.
    //   beta = 1 : 2/1 = 2.00  vs  4/3 = 1.33  -> UE0 wins
    //   beta = 2 : 4/1 = 4.00  vs 16/3 = 5.33  -> UE1 wins
    auto run_with_beta = [&](float beta) {
        Scenario s;
        s.nCell = 1; s.nActiveUe = 2; s.numUeSchdPerCellTTI = 1; s.nUeAnt = 1;
        s.initDefaultAssocAndSinr();
        s.wbSinr        = {3.0f, 15.0f};
        s.avgRatesActUe = {1.0f, 3.0f};
        s.betaCoeff     = beta;
        s.uploadCommon();

        cumac::multiCellUeSelection sel(&s.prms);
        execute(s, sel);
        return s.readSchdUes();
    };

    auto outBeta1 = run_with_beta(1.0f);
    ASSERT_EQ(outBeta1.size(), 1u);
    EXPECT_EQ(outBeta1[0], 0u);  // linear metric favors the low-avgRate UE

    auto outBeta2 = run_with_beta(2.0f);
    ASSERT_EQ(outBeta2.size(), 1u);
    EXPECT_EQ(outBeta2[0], 1u);  // squaring dataRate flips the winner
}

// ---------------------------------------------------------------------
// Heterogeneous selection with a per-cell cap of zero: that cell must emit
// only sentinels while the other cell still schedules normally.
TEST_F(MultiCellUeSelectionTest, Heterogeneous_CapZero_EmitsAllSentinelForCell)
{
    Scenario s;
    s.nCell = 2; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    runSelection(s);

    // Cell 0 schedules nothing; cell 1 schedules its top 2 UEs.
    s.enableHeterogeneous({0, 2});

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 4u);
    // Cell 0: cap 0 -> both slots sentinel.
    EXPECT_EQ(out[0], 0xFFFFu);
    EXPECT_EQ(out[1], 0xFFFFu);
    // Cell 1: cap 2 -> top two by SINR.
    EXPECT_EQ(out[2], 0u);
    EXPECT_EQ(out[3], 1u);
}

// ---------------------------------------------------------------------
// Per-cell association is honored: each cell only considers UEs associated
// with it (cellAssocActUe[cell * nActiveUe + ue]), so the globally highest
// SINR UE is NOT picked by a cell it is not associated with.
TEST_F(MultiCellUeSelectionTest, MultiCell_DistinctAssociation_RespectsPerCellPool)
{
    Scenario s;
    s.nCell = 2; s.nActiveUe = 4; s.numUeSchdPerCellTTI = 2; s.nUeAnt = 1;
    s.initDefaultAssocAndSinr();
    // Cell 0 owns only UE2/UE3; cell 1 owns only UE0/UE1.
    s.cellAssocActUe = {/*cell0*/ 0, 0, 1, 1,
                        /*cell1*/ 1, 1, 0, 0};
    s.uploadCommon();

    cumac::multiCellUeSelection sel(&s.prms);
    execute(s, sel);

    auto out = s.readSchdUes();
    ASSERT_EQ(out.size(), 4u);
    // Cell 0: among UE2/UE3, UE2 has higher SINR.
    EXPECT_EQ(out[0], 2u);
    EXPECT_EQ(out[1], 3u);
    // Cell 1: among UE0/UE1, UE0 has higher SINR.
    EXPECT_EQ(out[2], 0u);
    EXPECT_EQ(out[3], 1u);
}

} // namespace

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
