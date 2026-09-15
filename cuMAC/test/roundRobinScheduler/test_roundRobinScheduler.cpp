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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "4T4R/roundRobinScheduler.cuh"


using namespace cumac;

namespace {

#define CU_OK(expr) ASSERT_EQ((expr), cudaSuccess)

// Drives the GPU round-robin scheduler (multiCellRRScheduler) against small,
// fully-specified scenarios. The kernels discover associated UEs with a racing
// atomicAdd, so the order in which UEs are packed into the allocation is
// nondeterministic. Assertions therefore check order-independent allocation
// invariants (contiguous non-overlapping tiling of the PRG axis, exact PRG
// totals, and per-UE priority-weight outcomes) instead of per-UE PRG offsets.
class RRSchedTest : public ::testing::Test {
protected:
    /**
     * Create the CUDA stream used by every scenario.
     */
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }

    /**
     * Release all device buffers and destroy the CUDA stream.
     */
    void TearDown() override
    {
        FreeDev();
        if(stream_)
        {
            cudaStreamDestroy(stream_);
        }
    }

    /**
     * Free every device buffer allocated by Build() and reset the pointers.
     *
     * Safe to call more than once; already-freed pointers are skipped.
     */
    void FreeDev()
    {
        for(void* p : {(void*)d_cellId_, (void*)d_cellAssoc_, (void*)d_setSchd_,
                       (void*)d_prio_, (void*)d_allocSol_, (void*)d_newData_,
                       (void*)d_allocLast_})
        {
            if(p)
            {
                cudaFree(p);
            }
        }
        d_cellId_ = nullptr;   d_cellAssoc_ = nullptr; d_setSchd_ = nullptr;
        d_prio_   = nullptr;   d_allocSol_  = nullptr; d_newData_ = nullptr;
        d_allocLast_ = nullptr;
    }

    /**
     * Allocate device buffers and populate the scheduler parameters for one scenario.
     *
     * cellAssoc has size nCell*nUe and uses the cellAssoc[cell*nUe + ue] layout
     * the kernels expect. setSchd maps local UE index -> active-UE id (identity
     * here), so prioWeightActUe is sized by nUe. For HARQ, newData and allocLast
     * must be supplied.
     *
     * @param[in] nCell Number of cells.
     * @param[in] nUe Number of UEs.
     * @param[in] nPrbGrp Number of PRB groups in the PRG axis.
     * @param[in] allocType Allocation type selector (1 is supported by the GPU RR scheduler).
     * @param[in] harq HARQ-enabled indicator (1 enables the HARQ kernel path).
     * @param[in] cellAssoc Cell-UE association matrix, size nCell*nUe, cellAssoc[cell*nUe + ue] layout.
     * @param[in] prioInit Initial priority weight assigned to every UE.
     * @param[in] prioStep Priority-weight bump applied to dropped UEs (cumacCellGrpPrms::prioWeightStep).
     * @param[in] newData HARQ only: per-UE new-data flag, size nUe (1 = new Tx, 0 = re-Tx).
     * @param[in] allocLast HARQ only: per-UE last-Tx [start,end) ranges, size 2*nUe.
     */
    void Build(uint16_t nCell, uint16_t nUe, uint16_t nPrbGrp,
               uint8_t allocType, uint8_t harq,
               const std::vector<uint8_t>& cellAssoc,
               uint16_t prioInit  = 7,
               uint16_t prioStep  = 100,
               const std::vector<int8_t>*  newData   = nullptr,
               const std::vector<int16_t>* allocLast = nullptr)
    {
        nCell_ = nCell; nUe_ = nUe; nPrbGrp_ = nPrbGrp;
        prioInit_ = prioInit; prioStep_ = prioStep;

        ASSERT_EQ(cellAssoc.size(), static_cast<size_t>(nCell) * nUe);

        std::vector<uint16_t> cellId(nCell);
        std::iota(cellId.begin(), cellId.end(), uint16_t{0});
        std::vector<uint16_t> setSchd(nUe);
        std::iota(setSchd.begin(), setSchd.end(), uint16_t{0});
        std::vector<uint16_t> prio(nUe, prioInit);
        std::vector<int16_t>  allocSol(static_cast<size_t>(2) * nUe, int16_t{-1});

        CU_OK(cudaMalloc(&d_cellId_,    nCell * sizeof(uint16_t)));
        CU_OK(cudaMalloc(&d_cellAssoc_, static_cast<size_t>(nCell) * nUe * sizeof(uint8_t)));
        CU_OK(cudaMalloc(&d_setSchd_,   nUe * sizeof(uint16_t)));
        CU_OK(cudaMalloc(&d_prio_,      nUe * sizeof(uint16_t)));
        CU_OK(cudaMalloc(&d_allocSol_,  static_cast<size_t>(2) * nUe * sizeof(int16_t)));

        CU_OK(cudaMemcpyAsync(d_cellId_, cellId.data(), nCell * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, stream_));
        CU_OK(cudaMemcpyAsync(d_cellAssoc_, cellAssoc.data(),
                              static_cast<size_t>(nCell) * nUe * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_));
        CU_OK(cudaMemcpyAsync(d_setSchd_, setSchd.data(), nUe * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, stream_));
        CU_OK(cudaMemcpyAsync(d_prio_, prio.data(), nUe * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, stream_));
        CU_OK(cudaMemcpyAsync(d_allocSol_, allocSol.data(),
                              static_cast<size_t>(2) * nUe * sizeof(int16_t),
                              cudaMemcpyHostToDevice, stream_));

        prms_ = cumacCellGrpPrms{};
        prms_.cellId         = d_cellId_;
        prms_.cellAssoc      = d_cellAssoc_;
        prms_.nUe            = nUe;
        prms_.nCell          = nCell;
        prms_.nPrbGrp        = nPrbGrp;
        prms_.prioWeightStep = prioStep;
        prms_.allocType      = allocType;
        prms_.harqEnabledInd = harq;

        status_ = cumacCellGrpUeStatus{};
        status_.prioWeightActUe = d_prio_;

        sol_ = cumacSchdSol{};
        sol_.allocSol             = d_allocSol_;
        sol_.setSchdUePerCellTTI  = d_setSchd_;

        if(harq == 1)
        {
            ASSERT_NE(newData,   nullptr);
            ASSERT_NE(allocLast, nullptr);
            ASSERT_EQ(newData->size(),   static_cast<size_t>(nUe));
            ASSERT_EQ(allocLast->size(), static_cast<size_t>(2) * nUe);
            CU_OK(cudaMalloc(&d_newData_,   nUe * sizeof(int8_t)));
            CU_OK(cudaMalloc(&d_allocLast_, static_cast<size_t>(2) * nUe * sizeof(int16_t)));
            CU_OK(cudaMemcpyAsync(d_newData_, newData->data(), nUe * sizeof(int8_t),
                                  cudaMemcpyHostToDevice, stream_));
            CU_OK(cudaMemcpyAsync(d_allocLast_, allocLast->data(),
                                  static_cast<size_t>(2) * nUe * sizeof(int16_t),
                                  cudaMemcpyHostToDevice, stream_));
            status_.newDataActUe   = d_newData_;
            status_.allocSolLastTx = d_allocLast_;
        }

        CU_OK(cudaStreamSynchronize(stream_));
    }

    /**
     * Copy allocSol and prioWeight back from the device into host vectors.
     *
     * @param[out] allocOut Receives the per-UE [start,end) ranges, size 2*nUe.
     * @param[out] prioOut  Receives the per-UE priority weights, size nUe.
     */
    void Fetch(std::vector<int16_t>& allocOut, std::vector<uint16_t>& prioOut)
    {
        allocOut.assign(static_cast<size_t>(2) * nUe_, int16_t{0});
        prioOut.assign(nUe_, uint16_t{0});
        CU_OK(cudaMemcpy(allocOut.data(), d_allocSol_,
                         static_cast<size_t>(2) * nUe_ * sizeof(int16_t),
                         cudaMemcpyDeviceToHost));
        CU_OK(cudaMemcpy(prioOut.data(), d_prio_, nUe_ * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost));
    }

    /**
     * Construct, set up, run the scheduler and copy results back to the host.
     *
     * @param[out] allocOut Receives the per-UE [start,end) ranges, size 2*nUe.
     * @param[out] prioOut  Receives the per-UE priority weights, size nUe.
     */
    void RunAndFetch(std::vector<int16_t>& allocOut, std::vector<uint16_t>& prioOut)
    {
        multiCellRRScheduler sched(&prms_);
        sched.setup(&status_, &sol_, &prms_, stream_);
        sched.run(stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        Fetch(allocOut, prioOut);
    }

    /**
     * Construct and set up only (no run), then copy results back to the host.
     *
     * Used for the unsupported-alloc-type path: the scheduler rejects the type
     * during setup and never initializes a launch config, so calling run()
     * afterwards would launch an invalid kernel.
     *
     * @param[out] allocOut Receives the per-UE [start,end) ranges, size 2*nUe.
     * @param[out] prioOut  Receives the per-UE priority weights, size nUe.
     */
    void SetupOnlyAndFetch(std::vector<int16_t>& allocOut, std::vector<uint16_t>& prioOut)
    {
        multiCellRRScheduler sched(&prms_);
        sched.setup(&status_, &sol_, &prms_, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        Fetch(allocOut, prioOut);
    }

    /**
     * Assert the allocated ranges of `ues` tile [0, expectTiledPrg) exactly.
     *
     * Verifies each range is non-inverted, then sorts the ranges and asserts
     * they form a gap-free, overlap-free chain covering [0, expectTiledPrg).
     *
     * @param[in] alloc          Host copy of allocSol (2 entries per UE).
     * @param[in] ues            UE indices whose ranges participate in the tiling.
     * @param[in] expectTiledPrg Expected total number of tiled PRGs.
     * @return Per-UE allocated PRG counts in `ues` order (0 if unallocated).
     */
    std::vector<int> CheckTiling(const std::vector<int16_t>& alloc,
                                 const std::vector<int>&     ues,
                                 int                         expectTiledPrg)
    {
        std::vector<int> sizes;
        std::vector<std::pair<int, int>> ranges;
        for(int ue : ues)
        {
            const int s = alloc[2 * ue];
            const int e = alloc[2 * ue + 1];
            if(s < 0)
            {
                sizes.push_back(0);
                continue;
            }
            EXPECT_GT(e, s) << "ue=" << ue << " has empty or inverted range";
            sizes.push_back(e - s);
            ranges.emplace_back(s, e);
        }
        std::sort(ranges.begin(), ranges.end());
        int cursor = 0;
        for(const auto& r : ranges)
        {
            EXPECT_EQ(r.first, cursor) << "gap/overlap before [" << r.first << "," << r.second << ")";
            cursor = r.second;
        }
        EXPECT_EQ(cursor, expectTiledPrg) << "allocated PRGs do not tile [0," << expectTiledPrg << ")";
        return sizes;
    }

    cudaStream_t stream_ = nullptr;
    uint16_t*    d_cellId_     = nullptr;
    uint8_t*     d_cellAssoc_  = nullptr;
    uint16_t*    d_setSchd_    = nullptr;
    uint16_t*    d_prio_       = nullptr;
    int16_t*     d_allocSol_   = nullptr;
    int8_t*      d_newData_    = nullptr;
    int16_t*     d_allocLast_  = nullptr;

    cumacCellGrpPrms     prms_{};
    cumacCellGrpUeStatus status_{};
    cumacSchdSol         sol_{};

    uint16_t nCell_ = 0, nUe_ = 0, nPrbGrp_ = 0;
    uint16_t prioInit_ = 0, prioStep_ = 0;
};

// type-1, no HARQ, PRGs divide evenly: every associated UE gets exactly
// nPrbGrp/N PRGs and the allocation tiles the whole PRG axis.
TEST_F(RRSchedTest, Type1EvenSplitTilesAllPrgs)
{
    const uint16_t nCell = 1, nUe = 4, nPrbGrp = 8;  // 8 / 4 = 2 each, no remainder
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0,
          /*cellAssoc=*/{1, 1, 1, 1});

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);
    for(int sz : sizes)
    {
        EXPECT_EQ(sz, 2) << "even split should give 2 PRGs/UE";
    }
    for(int ue : ues)
    {
        EXPECT_EQ(prio[ue], 0) << "scheduled UE priority must reset to 0";
    }
}

// type-1, no HARQ, PRGs do not divide evenly: exactly `remainder` UEs get one
// extra PRG (floor+1) and the rest get floor.
TEST_F(RRSchedTest, Type1UnevenSplitDistributesRemainder)
{
    const uint16_t nCell = 1, nUe = 3, nPrbGrp = 7;  // 7 / 3 = 2 rem 1
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0,
          /*cellAssoc=*/{1, 1, 1});

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);

    const int nBig   = std::count(sizes.begin(), sizes.end(), 3);  // floor+1
    const int nSmall = std::count(sizes.begin(), sizes.end(), 2);  // floor
    EXPECT_EQ(nBig, 1)   << "exactly one UE should get the remainder PRG";
    EXPECT_EQ(nSmall, 2) << "the rest should get floor(nPrbGrp/N) PRGs";
    for(int ue : ues)
    {
        EXPECT_EQ(prio[ue], 0);
    }
}

// type-1, no HARQ, more UEs than PRGs (floor == 0): exactly nPrbGrp UEs get one
// PRG and the surplus UEs are left unallocated (-1) with their priority weight
// bumped by prioWeightStep.
TEST_F(RRSchedTest, Type1MoreUesThanPrgsLeavesSurplusUnallocated)
{
    const uint16_t nCell = 1, nUe = 6, nPrbGrp = 4;  // floor = 0, 4 winners, 2 losers
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0,
          /*cellAssoc=*/{1, 1, 1, 1, 1, 1}, /*prioInit=*/7, /*prioStep=*/100);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3, 4, 5};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);

    const int nAlloc   = std::count(sizes.begin(), sizes.end(), 1);
    const int nUnalloc = std::count(sizes.begin(), sizes.end(), 0);
    EXPECT_EQ(nAlloc, nPrbGrp);
    EXPECT_EQ(nUnalloc, nUe - nPrbGrp);

    // Allocated UEs -> priority reset to 0; unallocated -> bumped by step.
    // NOTE: a kernel-timing-measurement build re-launches the kernel multiple
    // times against the same priority buffer. The reset and allocation are
    // idempotent, but the unallocated bump accumulates (clamped at 0xFFFF), so we
    // only assert the launch-count-independent invariant: a dropped UE's priority
    // is bumped strictly above its initial value.
    for(int ue : ues)
    {
        if(alloc[2 * ue] < 0)
        {
            EXPECT_GT(prio[ue], prioInit_);
        }
        else
        {
            EXPECT_EQ(prio[ue], 0);
        }
    }
}

// A cell with no associated UEs makes its thread block hit the
// "nAssocUeFound == 0" early return; UEs associated to no cell are never touched.
TEST_F(RRSchedTest, Type1EmptyCellEarlyReturnLeavesUntouchedUes)
{
    const uint16_t nCell = 2, nUe = 4, nPrbGrp = 4;
    // cell 0 owns UEs 0,1; cell 1 owns nobody; UEs 2,3 belong to no cell.
    std::vector<uint8_t> cellAssoc(static_cast<size_t>(nCell) * nUe, 0);
    cellAssoc[0 * nUe + 0] = 1;
    cellAssoc[0 * nUe + 1] = 1;
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0, cellAssoc,
          /*prioInit=*/7, /*prioStep=*/100);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    // Cell 0's two UEs split the 4 PRGs evenly (2 each) and tile [0,4).
    const std::vector<int> ues0 = {0, 1};
    const std::vector<int> sizes = CheckTiling(alloc, ues0, nPrbGrp);
    for(int sz : sizes)
    {
        EXPECT_EQ(sz, 2);
    }
    EXPECT_EQ(prio[0], 0);
    EXPECT_EQ(prio[1], 0);

    // UEs associated to no cell keep the unallocated sentinel and untouched prio.
    for(int ue : {2, 3})
    {
        EXPECT_EQ(alloc[2 * ue], -1);
        EXPECT_EQ(alloc[2 * ue + 1], -1);
        EXPECT_EQ(prio[ue], prioInit_);
    }
}

// Two cells, each owning several UEs, are scheduled independently. Each cell
// must tile its OWN PRG axis [0, nPrbGrp): cell 0 (3 UEs) splits 6 PRGs evenly
// (2 each); cell 1 (4 UEs) splits 6 PRGs as 1 floor + 2 remainder (two UEs get
// 2, two get 1). Verifies cross-cell allocation independence.
TEST_F(RRSchedTest, Type1MultiCellEachCellTilesOwnPrgAxis)
{
    const uint16_t nCell = 2, nUe = 7, nPrbGrp = 6;
    // cell 0 owns UEs 0,1,2; cell 1 owns UEs 3,4,5,6 (each UE in exactly one cell).
    std::vector<uint8_t> cellAssoc(static_cast<size_t>(nCell) * nUe, 0);
    cellAssoc[0 * nUe + 0] = 1;
    cellAssoc[0 * nUe + 1] = 1;
    cellAssoc[0 * nUe + 2] = 1;
    cellAssoc[1 * nUe + 3] = 1;
    cellAssoc[1 * nUe + 4] = 1;
    cellAssoc[1 * nUe + 5] = 1;
    cellAssoc[1 * nUe + 6] = 1;
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0, cellAssoc,
          /*prioInit=*/7, /*prioStep=*/100);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    // Cell 0: 6 / 3 = 2 each, no remainder; tiles [0,6).
    const std::vector<int> ues0  = {0, 1, 2};
    const std::vector<int> sizes0 = CheckTiling(alloc, ues0, nPrbGrp);
    for(int sz : sizes0)
    {
        EXPECT_EQ(sz, 2) << "cell 0 even split should give 2 PRGs/UE";
    }

    // Cell 1: 6 / 4 = 1 floor, remainder 2 -> two UEs get 2, two get 1; tiles [0,6).
    const std::vector<int> ues1  = {3, 4, 5, 6};
    const std::vector<int> sizes1 = CheckTiling(alloc, ues1, nPrbGrp);
    EXPECT_EQ(std::count(sizes1.begin(), sizes1.end(), 2), 2) << "two cell-1 UEs get floor+1";
    EXPECT_EQ(std::count(sizes1.begin(), sizes1.end(), 1), 2) << "two cell-1 UEs get floor";

    // Every scheduled UE in both cells has its priority reset to 0.
    for(int ue = 0; ue < nUe; ++ue)
    {
        EXPECT_EQ(prio[ue], 0);
    }
}

// allocType == 0 is unsupported by the GPU RR scheduler: setup must reject it
// without selecting a kernel, leaving the allocation untouched. run() is skipped
// because it would launch an uninitialized kernel handle.
TEST_F(RRSchedTest, AllocTypeZeroIsRejectedWithoutLaunch)
{
    const uint16_t nCell = 1, nUe = 4, nPrbGrp = 8;
    Build(nCell, nUe, nPrbGrp, /*allocType=*/0, /*harq=*/0,
          /*cellAssoc=*/{1, 1, 1, 1});

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    SetupOnlyAndFetch(alloc, prio);

    // No kernel ran: every entry stays at the -1 sentinel and priorities unchanged.
    for(int ue = 0; ue < nUe; ++ue)
    {
        EXPECT_EQ(alloc[2 * ue], -1);
        EXPECT_EQ(alloc[2 * ue + 1], -1);
        EXPECT_EQ(prio[ue], prioInit_);
    }
}

// HARQ kernel: re-Tx UEs first reclaim their reserved PRG counts, then new-Tx
// UEs split the remainder.
TEST_F(RRSchedTest, Type1HarqReTxReservesThenNewTxSplitsRemainder)
{
    const uint16_t nCell = 1, nUe = 4, nPrbGrp = 8;
    // UEs 0,1 = re-Tx reserving 2 PRGs each (last-Tx ranges below); 2,3 = new-Tx.
    const std::vector<int8_t>  newData  = {0, 0, 1, 1};
    const std::vector<int16_t> allocLast = {0, 2,   2, 4,   0, 0,   0, 0};
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1,
          /*cellAssoc=*/{1, 1, 1, 1}, /*prioInit=*/7, /*prioStep=*/100,
          &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);

    // re-Tx UEs reclaim exactly 2 PRGs; remaining 4 PRGs split evenly to the
    // two new-Tx UEs (2 each).
    EXPECT_EQ(sizes[0], 2);
    EXPECT_EQ(sizes[1], 2);
    EXPECT_EQ(sizes[2], 2);
    EXPECT_EQ(sizes[3], 2);
    for(int ue : ues)
    {
        EXPECT_EQ(prio[ue], 0);
    }
}

// HARQ kernel: a re-Tx UE whose reserved PRGs exceed what remains is dropped
// (-1) and its priority weight pinned to the 0xFFFF ceiling.
TEST_F(RRSchedTest, Type1HarqReTxOverflowDropsAndPinsPriority)
{
    const uint16_t nCell = 1, nUe = 2, nPrbGrp = 4;
    // Both UEs re-Tx, each reserving 3 PRGs. Only one can fit (3 <= 4); the other
    // needs 3 > 1 remaining and is dropped. Which one fits is race-dependent.
    const std::vector<int8_t>  newData  = {0, 0};
    const std::vector<int16_t> allocLast = {0, 3,   0, 3};
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1,
          /*cellAssoc=*/{1, 1}, /*prioInit=*/7, /*prioStep=*/100,
          &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    int nFit = 0, nDropped = 0;
    for(int ue : {0, 1})
    {
        if(alloc[2 * ue] < 0)
        {
            ++nDropped;
            EXPECT_EQ(alloc[2 * ue + 1], -1);
            EXPECT_EQ(prio[ue], 0xFFFF) << "dropped re-Tx UE priority must pin to 0xFFFF";
        }
        else
        {
            ++nFit;
            EXPECT_EQ(alloc[2 * ue + 1] - alloc[2 * ue], 3) << "fitting re-Tx UE keeps its 3 reserved PRGs";
            EXPECT_EQ(prio[ue], 0);
        }
    }
    EXPECT_EQ(nFit, 1);
    EXPECT_EQ(nDropped, 1);
}

// HARQ kernel, all new-Tx, PRGs do not divide evenly: exactly `remainder`
// new-Tx UEs get floor+1 PRGs.
TEST_F(RRSchedTest, Type1HarqNewTxRemainderGivesExtraPrg)
{
    const uint16_t nCell = 1, nUe = 3, nPrbGrp = 7;  // no re-Tx: 7 / 3 = 2 rem 1
    const std::vector<int8_t>  newData  = {1, 1, 1};
    const std::vector<int16_t> allocLast(static_cast<size_t>(2) * nUe, 0);
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1,
          /*cellAssoc=*/{1, 1, 1}, /*prioInit=*/7, /*prioStep=*/100,
          &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);
    EXPECT_EQ(std::count(sizes.begin(), sizes.end(), 3), 1);
    EXPECT_EQ(std::count(sizes.begin(), sizes.end(), 2), 2);
    for(int ue : ues)
    {
        EXPECT_EQ(prio[ue], 0);
    }
}

// HARQ kernel, all new-Tx, more UEs than PRGs (floor == 0): nPrbGrp UEs get one
// PRG, the surplus new-Tx UEs are dropped (-1) with priority bumped by the step.
TEST_F(RRSchedTest, Type1HarqNewTxMoreUesThanPrgsDropsSurplus)
{
    const uint16_t nCell = 1, nUe = 6, nPrbGrp = 4;  // floor = 0: 4 winners, 2 losers
    const std::vector<int8_t>  newData  = {1, 1, 1, 1, 1, 1};
    const std::vector<int16_t> allocLast(static_cast<size_t>(2) * nUe, 0);
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1,
          /*cellAssoc=*/{1, 1, 1, 1, 1, 1}, /*prioInit=*/7, /*prioStep=*/100,
          &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3, 4, 5};
    const std::vector<int> sizes = CheckTiling(alloc, ues, nPrbGrp);
    EXPECT_EQ(std::count(sizes.begin(), sizes.end(), 1), nPrbGrp);
    EXPECT_EQ(std::count(sizes.begin(), sizes.end(), 0), nUe - nPrbGrp);
    // A kernel-timing-measurement build re-launches the kernel multiple times,
    // so the dropped-UE priority bump accumulates (clamped at 0xFFFF). Assert
    // only that a dropped UE's priority is bumped strictly above its initial
    // value.
    for(int ue : ues)
    {
        if(alloc[2 * ue] < 0)
        {
            EXPECT_GT(prio[ue], prioInit_);
        }
        else
        {
            EXPECT_EQ(prio[ue], 0);
        }
    }
}

// HARQ kernel with a cell that owns no UEs: that block takes the empty-cell
// early return and the unassociated UEs are left untouched.
TEST_F(RRSchedTest, Type1HarqEmptyCellEarlyReturn)
{
    const uint16_t nCell = 2, nUe = 4, nPrbGrp = 4;
    std::vector<uint8_t> cellAssoc(static_cast<size_t>(nCell) * nUe, 0);
    cellAssoc[0 * nUe + 0] = 1;  // cell 0 owns UEs 0,1; cell 1 owns nobody
    cellAssoc[0 * nUe + 1] = 1;
    const std::vector<int8_t>  newData  = {1, 1, 1, 1};
    const std::vector<int16_t> allocLast(static_cast<size_t>(2) * nUe, 0);
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1, cellAssoc,
          /*prioInit=*/7, /*prioStep=*/100, &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues0 = {0, 1};
    const std::vector<int> sizes = CheckTiling(alloc, ues0, nPrbGrp);
    for(int sz : sizes)
    {
        EXPECT_EQ(sz, 2);
    }
    for(int ue : {2, 3})
    {
        EXPECT_EQ(alloc[2 * ue], -1);
        EXPECT_EQ(prio[ue], prioInit_);
    }
}

// type-1, no HARQ, surplus UE whose bumped priority would exceed the 16-bit
// range: it must saturate at 0xFFFF.
TEST_F(RRSchedTest, Type1UnallocatedPrioritySaturatesAtCeiling)
{
    const uint16_t nCell = 1, nUe = 6, nPrbGrp = 4;
    // prioInit + step = 65500 + 100 = 65600 > 0xFFFF, so dropped UEs clamp.
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/0,
          /*cellAssoc=*/{1, 1, 1, 1, 1, 1}, /*prioInit=*/65500, /*prioStep=*/100);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3, 4, 5};
    CheckTiling(alloc, ues, nPrbGrp);
    for(int ue : ues)
    {
        if(alloc[2 * ue] < 0)
        {
            EXPECT_EQ(prio[ue], 0xFFFF) << "bumped priority must saturate at 0xFFFF";
        }
        else
        {
            EXPECT_EQ(prio[ue], 0);
        }
    }
}

// HARQ kernel, all new-Tx, more UEs than PRGs, with a bumped priority that would
// exceed the 16-bit range: dropped new-Tx UEs must saturate at 0xFFFF.
TEST_F(RRSchedTest, Type1HarqNewTxUnallocatedPrioritySaturatesAtCeiling)
{
    const uint16_t nCell = 1, nUe = 6, nPrbGrp = 4;  // floor = 0: 2 UEs dropped
    const std::vector<int8_t>  newData  = {1, 1, 1, 1, 1, 1};
    const std::vector<int16_t> allocLast(static_cast<size_t>(2) * nUe, 0);
    // prioInit + step = 65500 + 100 > 0xFFFF, so dropped UEs clamp at 0xFFFF.
    Build(nCell, nUe, nPrbGrp, /*allocType=*/1, /*harq=*/1,
          /*cellAssoc=*/{1, 1, 1, 1, 1, 1}, /*prioInit=*/65500, /*prioStep=*/100,
          &newData, &allocLast);

    std::vector<int16_t> alloc;
    std::vector<uint16_t> prio;
    RunAndFetch(alloc, prio);

    const std::vector<int> ues = {0, 1, 2, 3, 4, 5};
    CheckTiling(alloc, ues, nPrbGrp);
    for(int ue : ues)
    {
        if(alloc[2 * ue] < 0)
        {
            EXPECT_EQ(prio[ue], 0xFFFF);
        }
        else
        {
            EXPECT_EQ(prio[ue], 0);
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // CUDA is not fork-safe; "threadsafe" death tests fork+exec so any child
    // re-initializes CUDA cleanly (matches the cellAssociation test convention).
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    const int rc = RUN_ALL_TESTS();

    return rc;
}
