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

// Dedicated GoogleTest unit test for cumac::multiCellSrsScheduler.
//
// The shipped example (cuMAC/examples/multiCellSrsScheduler) is disabled in the
// coverage unit-test driver, so this class was reported at 0% coverage. This
// test exercises the GPU scheduler (both kernel versions) against the in-class
// CPU reference (cpuScheduler_v0 / cpuScheduler_v1) and asserts they agree,
// mirroring the example's compare logic but as deterministic gtest cases.
//
// Kernel selection (see multiCellSrsScheduler::setup):
//   - kernel_v0 / cpuScheduler_v0  : the round-robin/age-based path. Selected
//     whenever NOT (nBsAnt==64 && srsSchedulingSel==1).
//   - kernel_v1 / cpuScheduler_v1  : the 64TR MU/SU-MIMO + age path. Selected
//     when nBsAnt==64 && srsSchedulingSel==1; needs schdSol->muMimoInd and
//     schdSol->sortedUeList.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "multiCellSrsScheduler/multiCellSrsScheduler.cuh"


using namespace cumac;

namespace {

// Small RAII helper for a device buffer.
template <typename T>
struct DevBuf {
    T* p = nullptr;
    // Zero-initialize on allocation: cudaMalloc returns recycled, uninitialized
    // device memory, and a fresh allocation after a larger test reuses regions
    // still holding the previous test's data. Any field the kernel reads before
    // writing must therefore start from a known state, not whatever was left
    // behind, otherwise results become test-order dependent.
    void alloc(size_t n) {
        ASSERT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
        ASSERT_EQ(cudaMemset(p, 0, n * sizeof(T)), cudaSuccess);
    }
    ~DevBuf()
    {
        if (p) {
            cudaFree(p);
        }
    }
};

class MultiCellSrsSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
    void TearDown() override { if (stream_) cudaStreamDestroy(stream_); }

    // Build one randomized scenario, run GPU setup()+run(), then run the CPU
    // reference on identical inputs and assert the scheduling solutions match.
    //
    // injectInvalidCfg : give a couple of UEs srsConfigIndex>63 / srsBwIndex>3
    //                    so the W_SRS_LAST table-lookup guard takes its false
    //                    leg (default bandwidth) on both CPU and GPU.
    // emptyCell0       : leave cell 0 with no associated UEs so the
    //                    no-associated-UE early-return path is exercised.
    // craftTpc : replace the random power-control inputs with deterministic
    //            per-UE values that walk the TPC (transmit-power-control) routine
    //            through every accumulation/threshold branch. With alpha = 0 the
    //            power-control "gap" reduces to txPwrMax - pwr0 - 10log10(2*M_SRS)
    //            (M_SRS = 816 for config 63 / bw 0 / comb 4 -> ~32.127 dB), so the
    //            gap and the SNR-gap can be dialed in directly per UE.
    void RunAndCompare(uint32_t   seed,
                       uint16_t   nCell,
                       uint16_t   nMaxActUePerCell,
                       uint8_t    nBsAnt,
                       uint8_t    srsSchedulingSel,
                       bool       injectInvalidCfg = false,
                       bool       emptyCell0       = false,
                       bool       craftTpc         = false,
                       uint16_t   nAssocPerCell    = 0)  // 0 -> fully populated
    {
        const uint16_t nActiveUe     = static_cast<uint16_t>(nMaxActUePerCell * nCell);
        const uint16_t nSymbsPerSlot = 14;
        const bool     useV1         = (nBsAnt == 64) && (srsSchedulingSel == 1);

        std::mt19937 rng(seed);

        multiCellSrsScheduler sched;

        // ----------------------------- CPU-side inputs -----------------------------
        cumacSrsCellGrpPrms     prmCpu{};
        cumacSrsCellGrpUeStatus stCpu{};
        cumacSchdSol            schdCpu{};
        cumacSrsSchdSol         solCpu{};      // receives GPU results
        cumacSrsSchdSol         solRef{};      // receives CPU reference results

        prmCpu.nActiveUe        = nActiveUe;
        prmCpu.nMaxActUePerCell = nMaxActUePerCell;
        prmCpu.nCell            = nCell;
        prmCpu.nSymbsPerSlot    = nSymbsPerSlot;
        prmCpu.nBsAnt           = nBsAnt;
        prmCpu.srsSchedulingSel = srsSchedulingSel;

        std::vector<uint16_t> cellId(nCell);
        for (uint16_t c = 0; c < nCell; ++c) cellId[c] = c;
        prmCpu.cellId = cellId.data();

        // Block association: UE uIdx belongs to cell floor(uIdx/nMaxActUePerCell).
        std::vector<uint8_t> cellAssoc(static_cast<size_t>(nCell) * nActiveUe, 0);
        for (uint16_t c = 0; c < nCell; ++c) {
            if (emptyCell0 && c == 0) continue;  // leave cell 0 empty
            for (uint16_t u = 0; u < nActiveUe; ++u) {
                if (static_cast<uint16_t>(u / nMaxActUePerCell) == c)
                    cellAssoc[static_cast<size_t>(c) * nActiveUe + u] = 1;
            }
        }
        prmCpu.cellAssocActUe = cellAssoc.data();

        std::vector<int8_t>   newDataActUe(nActiveUe, 1);
        std::vector<uint32_t> srsLastTxCounter0(nActiveUe);
        std::vector<uint8_t>  srsNumAntPorts(nActiveUe, 4);
        std::vector<uint8_t>  srsResourceType(nActiveUe, 0);
        std::vector<float>    srsWbSnr(nActiveUe);
        std::vector<float>    srsWbSnrThreshold(nActiveUe, 5.0f);
        std::vector<float>    srsWidebandSignalEnergy(nActiveUe);
        std::vector<float>    srsTxPwrMax(nActiveUe, 23.0f);
        std::vector<float>    srsPwr0(nActiveUe, -10.0f);
        std::vector<float>    srsPwrAlpha(nActiveUe);
        std::vector<uint8_t>  srsTpcAccumulationFlag(nActiveUe);
        std::vector<float>    srsPcAdjState0(nActiveUe);
        std::vector<uint8_t>  srsPowerHeadroomReport(nActiveUe, 0);

        std::uniform_real_distribution<float>   dSnr(-5.0f, 15.0f);
        std::uniform_real_distribution<float>   dEnergy(0.0f, 10.0f);
        std::uniform_int_distribution<uint32_t> dCounter(0, 19);
        std::uniform_int_distribution<uint32_t> dAlphaIdx(0, 4);
        std::uniform_int_distribution<uint32_t> dBit(0, 1);
        std::uniform_int_distribution<uint32_t> dPcAdjStep(0, 19);
        for (uint16_t u = 0; u < nActiveUe; ++u) {
            srsLastTxCounter0[u]       = dCounter(rng);
            srsWbSnr[u]                = dSnr(rng);
            srsWidebandSignalEnergy[u] = dEnergy(rng);
            srsPwrAlpha[u]             = dAlphaIdx(rng) * 0.2f;
            srsTpcAccumulationFlag[u]  = static_cast<uint8_t>(dBit(rng));
            srsPcAdjState0[u]          = dPcAdjStep(rng) * 1.0f - 10.0f;
        }

        std::vector<uint32_t> srsLastTxCounter = srsLastTxCounter0;  // mutated by CPU ref
        std::vector<float>    srsPcAdjState     = srsPcAdjState0;     // mutated by CPU ref

        stCpu.newDataActUe                   = newDataActUe.data();
        stCpu.srsLastTxCounter               = srsLastTxCounter.data();
        stCpu.srsNumAntPorts                 = srsNumAntPorts.data();
        stCpu.srsResourceType                = srsResourceType.data();
        stCpu.srsWbSnr                       = srsWbSnr.data();
        stCpu.srsWbSnrThreshold              = srsWbSnrThreshold.data();
        stCpu.srsWidebandSignalEnergy        = srsWidebandSignalEnergy.data();
        stCpu.srsTxPwrMax                    = srsTxPwrMax.data();
        stCpu.srsPwr0                        = srsPwr0.data();
        stCpu.srsPwrAlpha                    = srsPwrAlpha.data();
        stCpu.srsTpcAccumulationFlag         = srsTpcAccumulationFlag.data();
        stCpu.srsPowerControlAdjustmentState = srsPcAdjState.data();
        stCpu.srsPowerHeadroomReport         = srsPowerHeadroomReport.data();

        // schdSol: muMimoInd + sortedUeList (pinned host, device-accessible).
        std::vector<uint8_t> muMimoInd(nActiveUe);
        for (uint16_t u = 0; u < nActiveUe; ++u)
            muMimoInd[u] = static_cast<uint8_t>(dBit(rng));
        schdCpu.muMimoInd = muMimoInd.data();

        ASSERT_EQ(cudaMallocHost(&schdCpu.sortedUeList, sizeof(uint16_t*) * nCell), cudaSuccess);
        for (uint16_t c = 0; c < nCell; ++c)
            ASSERT_EQ(cudaMallocHost(&schdCpu.sortedUeList[c], sizeof(uint16_t) * nMaxActUePerCell),
                      cudaSuccess);
        for (uint16_t c = 0; c < nCell; ++c) {
            std::vector<uint16_t> order(nMaxActUePerCell);
            for (uint16_t u = 0; u < nMaxActUePerCell; ++u) order[u] = u;
            std::shuffle(order.begin(), order.end(), rng);
            for (uint16_t u = 0; u < nMaxActUePerCell; ++u)
                schdCpu.sortedUeList[c][u] =
                    static_cast<uint16_t>(order[u] + c * nMaxActUePerCell);
        }

        // srsSchdSol input fields + output storage (CPU result copy).
        std::vector<uint8_t>  cfgIdx0(nActiveUe, 63);
        std::vector<uint8_t>  bwIdx0(nActiveUe, 0);
        std::vector<float>    txPwr0(nActiveUe);
        std::uniform_real_distribution<float> dTxPwrDist(10.0f, 23.0f);
        for (uint16_t u = 0; u < nActiveUe; ++u) txPwr0[u] = dTxPwrDist(rng);
        if (injectInvalidCfg && nActiveUe >= 2) {
            cfgIdx0[0] = 64;            // first operand of guard false
            cfgIdx0[1] = 63; bwIdx0[1] = 4;  // second operand of guard false
        }

        // For an empty cell 0, also zero its UEs' SRS aging counters. kernel_v1
        // has no no-associated-UE guard before the MU/SU phases: it would
        // otherwise schedule cell-0 UEs that still carry a nonzero initial
        // counter, while cpuScheduler_v1 short-circuits the whole cell. Zeroing
        // the counters makes both sides schedule nothing for the empty cell, so
        // they agree and the no-associated-UE early-return paths are exercised
        // consistently.
        if (emptyCell0) {
            for (uint16_t u = 0; u < nMaxActUePerCell; ++u) srsLastTxCounter0[u] = 0;
        }

        // Partial population: keep only the first nAssocPerCell UEs of each cell
        // associated and zero the counters of the rest so they are never SRS
        // eligible. This lets a cell carry fewer candidates than the per-symbol
        // SRS capacity while staying in the safe large-nMax buffer regime, so the
        // per-cell scheduling caps and the MU/SU loop-completion exits get
        // exercised.
        if (nAssocPerCell > 0 && nAssocPerCell < nMaxActUePerCell) {
            for (uint16_t c = 0; c < nCell; ++c) {
                if (emptyCell0 && c == 0) continue;
                for (uint16_t lu = nAssocPerCell; lu < nMaxActUePerCell; ++lu) {
                    const uint16_t u = static_cast<uint16_t>(c * nMaxActUePerCell + lu);
                    cellAssoc[static_cast<size_t>(c) * nActiveUe + u] = 0;
                    srsLastTxCounter0[u] = 0;
                }
            }
        }

        if (craftTpc) {
            // {flag, txPwrMax, adjState, snr, threshold}. gap0 = txPwrMax - 32.127,
            // gap1 = txPwrMax - 32.127 - adjState, snrGap = threshold - (snr-adjState).
            struct Combo { uint8_t flag; float txPwrMax; float adj; float snr; float thr; };
            static const Combo kCombos[] = {
                {1, 26.0f, 0.0f, 0.0f,  0.0f},   // accum:    gap <= 0
                {1, 42.0f, 0.0f, 0.0f,  0.0f},   // accum:    gap >  0
                {0, 31.6f, 0.0f, 0.0f,  0.0f},   // no-accum: gap in [-1, 0]
                {0, 29.6f, 0.0f, 0.0f,  0.0f},   // no-accum: gap in [-4, -1)
                {0, 26.0f, 0.0f, 0.0f,  0.0f},   // no-accum: gap <  -4
                {0, 42.0f, 0.0f, 0.0f, 10.0f},   // no-accum: gap>0, delta >= 4
                {0, 42.0f, 0.0f, 0.0f,  1.5f},   // no-accum: gap>0, delta in [1, 4)
                {0, 42.0f, 0.0f, 0.0f, -0.5f},   // no-accum: gap>0, delta in [-1, 1)
                {0, 42.0f, 0.0f, 0.0f, -3.0f},   // no-accum: gap>0, delta <  -1
                {2, 23.0f, 0.0f, 0.0f,  0.0f},   // flag != 0 and != 1: no TPC update
            };
            constexpr size_t kNCombo = sizeof(kCombos) / sizeof(kCombos[0]);
            for (uint16_t u = 0; u < nActiveUe; ++u) {
                const Combo& cb = kCombos[u % kNCombo];
                srsPwrAlpha[u]             = 0.0f;
                srsPwr0[u]                 = 0.0f;
                srsWidebandSignalEnergy[u] = 1.0f;
                txPwr0[u]                  = 10.0f;
                srsPcAdjState0[u]          = cb.adj;
                srsTxPwrMax[u]             = cb.txPwrMax;
                srsTpcAccumulationFlag[u]  = cb.flag;
                srsWbSnr[u]                = cb.snr;
                srsWbSnrThreshold[u]       = cb.thr;
            }
        }

        auto allocSol = [&](cumacSrsSchdSol& s,
                            std::vector<uint16_t>& nSch, std::vector<uint16_t>& txUe,
                            std::vector<uint16_t>& rxCell, std::vector<uint8_t>& numSymb,
                            std::vector<uint8_t>& timeStart, std::vector<uint8_t>& numRep,
                            std::vector<uint8_t>& cfg, std::vector<uint8_t>& bw,
                            std::vector<uint8_t>& comb, std::vector<uint8_t>& combOff,
                            std::vector<uint8_t>& freqStart, std::vector<uint16_t>& freqShift,
                            std::vector<uint8_t>& freqHop, std::vector<uint16_t>& seqId,
                            std::vector<uint8_t>& gosh, std::vector<uint8_t>& cyc,
                            std::vector<float>& txPwr) {
            nSch.assign(nCell, 0);
            txUe.assign(static_cast<size_t>(nMaxActUePerCell) * nCell, 0);
            rxCell.assign(nActiveUe, 0);
            numSymb.assign(nActiveUe, 0); timeStart.assign(nActiveUe, 0);
            numRep.assign(nActiveUe, 0);  cfg.assign(nActiveUe, 0);
            bw.assign(nActiveUe, 0);      comb.assign(nActiveUe, 0);
            combOff.assign(nActiveUe, 0); freqStart.assign(nActiveUe, 0);
            freqShift.assign(nActiveUe, 0); freqHop.assign(nActiveUe, 0);
            seqId.assign(nActiveUe, 0);   gosh.assign(nActiveUe, 0);
            cyc.assign(nActiveUe, 0);     txPwr.assign(nActiveUe, 0.0f);
            s.nSrsScheduledUePerCell = nSch.data();
            s.srsTxUe = txUe.data();      s.srsRxCell = rxCell.data();
            s.srsNumSymb = numSymb.data(); s.srsTimeStart = timeStart.data();
            s.srsNumRep = numRep.data();  s.srsConfigIndex = cfg.data();
            s.srsBwIndex = bw.data();     s.srsCombSize = comb.data();
            s.srsCombOffset = combOff.data(); s.srsFreqStart = freqStart.data();
            s.srsFreqShift = freqShift.data(); s.srsFreqHopping = freqHop.data();
            s.srsSequenceId = seqId.data(); s.srsGroupOrSequenceHopping = gosh.data();
            s.srsCyclicShift = cyc.data(); s.srsTxPwr = txPwr.data();
        };

        std::vector<uint16_t> cNSch, cTxUe, cRxCell, cFreqShift, cSeqId;
        std::vector<uint8_t>  cNumSymb, cTimeStart, cNumRep, cCfg, cBw, cComb, cCombOff,
                              cFreqStart, cFreqHop, cGosh, cCyc;
        std::vector<float>    cTxPwr;
        allocSol(solCpu, cNSch, cTxUe, cRxCell, cNumSymb, cTimeStart, cNumRep, cCfg, cBw,
                 cComb, cCombOff, cFreqStart, cFreqShift, cFreqHop, cSeqId, cGosh, cCyc, cTxPwr);
        // CPU-ref scheduling solution gets its own storage; seed the input fields.
        std::vector<uint16_t> rNSch, rTxUe, rRxCell, rFreqShift, rSeqId;
        std::vector<uint8_t>  rNumSymb, rTimeStart, rNumRep, rCfg, rBw, rComb, rCombOff,
                              rFreqStart, rFreqHop, rGosh, rCyc;
        std::vector<float>    rTxPwr;
        allocSol(solRef, rNSch, rTxUe, rRxCell, rNumSymb, rTimeStart, rNumRep, rCfg, rBw,
                 rComb, rCombOff, rFreqStart, rFreqShift, rFreqHop, rSeqId, rGosh, rCyc, rTxPwr);
        for (uint16_t u = 0; u < nActiveUe; ++u) {
            rCfg[u] = cfgIdx0[u]; rBw[u] = bwIdx0[u]; rTxPwr[u] = txPwr0[u];
        }

        // ----------------------------- GPU-side inputs -----------------------------
        cumacSrsCellGrpPrms     prmGpu = prmCpu;   // scalar fields copy
        cumacSrsCellGrpUeStatus stGpu{};
        cumacSchdSol            schdGpu{};
        cumacSrsSchdSol         solGpu{};

        DevBuf<uint16_t> dCellId;            dCellId.alloc(nCell);
        DevBuf<uint8_t>  dCellAssoc;         dCellAssoc.alloc(static_cast<size_t>(nCell) * nActiveUe);
        DevBuf<int8_t>   dNewData;           dNewData.alloc(nActiveUe);
        DevBuf<uint32_t> dLastTx;            dLastTx.alloc(nActiveUe);
        DevBuf<uint8_t>  dNumAntPorts;       dNumAntPorts.alloc(nActiveUe);
        DevBuf<uint8_t>  dResType;           dResType.alloc(nActiveUe);
        DevBuf<float>    dWbSnr;             dWbSnr.alloc(nActiveUe);
        DevBuf<float>    dWbSnrThr;          dWbSnrThr.alloc(nActiveUe);
        DevBuf<float>    dWbEnergy;          dWbEnergy.alloc(nActiveUe);
        DevBuf<float>    dTxPwrMax;          dTxPwrMax.alloc(nActiveUe);
        DevBuf<float>    dPwr0;              dPwr0.alloc(nActiveUe);
        DevBuf<float>    dPwrAlpha;          dPwrAlpha.alloc(nActiveUe);
        DevBuf<uint8_t>  dTpcAccum;          dTpcAccum.alloc(nActiveUe);
        DevBuf<float>    dPcAdj;             dPcAdj.alloc(nActiveUe);
        DevBuf<uint8_t>  dPhr;               dPhr.alloc(nActiveUe);
        DevBuf<uint8_t>  dMuMimo;            dMuMimo.alloc(nActiveUe);
        DevBuf<uint16_t*> dSortedUeList;     dSortedUeList.alloc(nCell);

        DevBuf<uint16_t> dNSch;              dNSch.alloc(nActiveUe);
        DevBuf<uint16_t> dTxUe;              dTxUe.alloc(static_cast<size_t>(nMaxActUePerCell) * nCell);
        DevBuf<uint16_t> dRxCell;            dRxCell.alloc(nActiveUe);
        DevBuf<uint8_t>  dNumSymb;           dNumSymb.alloc(nActiveUe);
        DevBuf<uint8_t>  dTimeStart;         dTimeStart.alloc(nActiveUe);
        DevBuf<uint8_t>  dNumRep;            dNumRep.alloc(nActiveUe);
        DevBuf<uint8_t>  dCfg;               dCfg.alloc(nActiveUe);
        DevBuf<uint8_t>  dBw;                dBw.alloc(nActiveUe);
        DevBuf<uint8_t>  dComb;              dComb.alloc(nActiveUe);
        DevBuf<uint8_t>  dCombOff;           dCombOff.alloc(nActiveUe);
        DevBuf<uint8_t>  dFreqStart;         dFreqStart.alloc(nActiveUe);
        DevBuf<uint16_t> dFreqShift;         dFreqShift.alloc(nActiveUe);
        DevBuf<uint8_t>  dFreqHop;           dFreqHop.alloc(nActiveUe);
        DevBuf<uint16_t> dSeqId;             dSeqId.alloc(nActiveUe);
        DevBuf<uint8_t>  dGosh;              dGosh.alloc(nActiveUe);
        DevBuf<uint8_t>  dCyc;               dCyc.alloc(nActiveUe);
        DevBuf<float>    dTxPwr;             dTxPwr.alloc(nActiveUe);

        auto H2D = [&](void* dst, const void* src, size_t bytes) {
            ASSERT_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_), cudaSuccess);
        };
        H2D(dCellId.p, cellId.data(), sizeof(uint16_t) * nCell);
        H2D(dCellAssoc.p, cellAssoc.data(), static_cast<size_t>(nCell) * nActiveUe);
        H2D(dNewData.p, newDataActUe.data(), nActiveUe);
        H2D(dLastTx.p, srsLastTxCounter0.data(), sizeof(uint32_t) * nActiveUe);
        H2D(dNumAntPorts.p, srsNumAntPorts.data(), nActiveUe);
        H2D(dResType.p, srsResourceType.data(), nActiveUe);
        H2D(dWbSnr.p, srsWbSnr.data(), sizeof(float) * nActiveUe);
        H2D(dWbSnrThr.p, srsWbSnrThreshold.data(), sizeof(float) * nActiveUe);
        H2D(dWbEnergy.p, srsWidebandSignalEnergy.data(), sizeof(float) * nActiveUe);
        H2D(dTxPwrMax.p, srsTxPwrMax.data(), sizeof(float) * nActiveUe);
        H2D(dPwr0.p, srsPwr0.data(), sizeof(float) * nActiveUe);
        H2D(dPwrAlpha.p, srsPwrAlpha.data(), sizeof(float) * nActiveUe);
        H2D(dTpcAccum.p, srsTpcAccumulationFlag.data(), nActiveUe);
        H2D(dPcAdj.p, srsPcAdjState0.data(), sizeof(float) * nActiveUe);
        H2D(dPhr.p, srsPowerHeadroomReport.data(), nActiveUe);
        H2D(dMuMimo.p, muMimoInd.data(), nActiveUe);
        H2D(dSortedUeList.p, schdCpu.sortedUeList, sizeof(uint16_t*) * nCell);
        H2D(dCfg.p, cfgIdx0.data(), nActiveUe);
        H2D(dBw.p, bwIdx0.data(), nActiveUe);
        H2D(dTxPwr.p, txPwr0.data(), sizeof(float) * nActiveUe);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        prmGpu.cellId = dCellId.p;
        prmGpu.cellAssocActUe = dCellAssoc.p;

        stGpu.newDataActUe = dNewData.p;
        stGpu.srsLastTxCounter = dLastTx.p;
        stGpu.srsNumAntPorts = dNumAntPorts.p;
        stGpu.srsResourceType = dResType.p;
        stGpu.srsWbSnr = dWbSnr.p;
        stGpu.srsWbSnrThreshold = dWbSnrThr.p;
        stGpu.srsWidebandSignalEnergy = dWbEnergy.p;
        stGpu.srsTxPwrMax = dTxPwrMax.p;
        stGpu.srsPwr0 = dPwr0.p;
        stGpu.srsPwrAlpha = dPwrAlpha.p;
        stGpu.srsTpcAccumulationFlag = dTpcAccum.p;
        stGpu.srsPowerControlAdjustmentState = dPcAdj.p;
        stGpu.srsPowerHeadroomReport = dPhr.p;

        schdGpu.muMimoInd = dMuMimo.p;
        schdGpu.sortedUeList = dSortedUeList.p;

        solGpu.nSrsScheduledUePerCell = dNSch.p;
        solGpu.srsTxUe = dTxUe.p;            solGpu.srsRxCell = dRxCell.p;
        solGpu.srsNumSymb = dNumSymb.p;      solGpu.srsTimeStart = dTimeStart.p;
        solGpu.srsNumRep = dNumRep.p;        solGpu.srsConfigIndex = dCfg.p;
        solGpu.srsBwIndex = dBw.p;           solGpu.srsCombSize = dComb.p;
        solGpu.srsCombOffset = dCombOff.p;   solGpu.srsFreqStart = dFreqStart.p;
        solGpu.srsFreqShift = dFreqShift.p;  solGpu.srsFreqHopping = dFreqHop.p;
        solGpu.srsSequenceId = dSeqId.p;     solGpu.srsGroupOrSequenceHopping = dGosh.p;
        solGpu.srsCyclicShift = dCyc.p;      solGpu.srsTxPwr = dTxPwr.p;

        // ----------------------------- GPU run -----------------------------
        sched.setup(&stGpu, &prmGpu, &schdGpu, &solGpu, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        sched.run(stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        auto D2H = [&](void* dst, const void* src, size_t bytes) {
            ASSERT_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_), cudaSuccess);
        };
        D2H(cNSch.data(), dNSch.p, sizeof(uint16_t) * nCell);
        D2H(cTxUe.data(), dTxUe.p, sizeof(uint16_t) * nMaxActUePerCell * nCell);
        D2H(cTimeStart.data(), dTimeStart.p, nActiveUe);
        D2H(cCombOff.data(), dCombOff.p, nActiveUe);
        D2H(cTxPwr.data(), dTxPwr.p, sizeof(float) * nActiveUe);
        // srsLastTxCounter was mutated in place on the device.
        std::vector<uint32_t> lastTxGpu(nActiveUe);
        D2H(lastTxGpu.data(), dLastTx.p, sizeof(uint32_t) * nActiveUe);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        // ----------------------------- CPU reference -----------------------------
        // srsLastTxCounter / srsPowerControlAdjustmentState start from the
        // originals (already the case: stCpu points at the mutable copies).
        srsLastTxCounter = srsLastTxCounter0;
        srsPcAdjState     = srsPcAdjState0;
        stCpu.srsLastTxCounter               = srsLastTxCounter.data();
        stCpu.srsPowerControlAdjustmentState = srsPcAdjState.data();

        if (useV1)
            sched.cpuScheduler_v1(&stCpu, &prmCpu, &schdCpu, &solRef);
        else
            sched.cpuScheduler_v0(&stCpu, &prmCpu, &solRef);

        // ----------------------------- Compare -----------------------------
        for (uint16_t c = 0; c < nCell; ++c) {
            ASSERT_EQ(cNSch[c], rNSch[c])
                << "nSrsScheduledUePerCell mismatch cell=" << c << " seed=" << seed
                << " v1=" << useV1;
            for (uint16_t i = 0; i < rNSch[c]; ++i) {
                const size_t k = static_cast<size_t>(c) * nMaxActUePerCell + i;
                ASSERT_EQ(cTxUe[k], rTxUe[k])
                    << "srsTxUe mismatch cell=" << c << " i=" << i << " seed=" << seed;
            }
            for (uint16_t i = 0; i < rNSch[c]; ++i) {
                const uint16_t ue = cTxUe[static_cast<size_t>(c) * nMaxActUePerCell + i];
                ASSERT_EQ(cTimeStart[ue], rTimeStart[ue])
                    << "srsTimeStart mismatch ue=" << ue << " seed=" << seed;
                ASSERT_EQ(cCombOff[ue], rCombOff[ue])
                    << "srsCombOffset mismatch ue=" << ue << " seed=" << seed;
                ASSERT_NEAR(cTxPwr[ue], rTxPwr[ue], 0.01f)
                    << "srsTxPwr mismatch ue=" << ue << " seed=" << seed;
            }
        }
        for (uint16_t u = 0; u < nActiveUe; ++u) {
            ASSERT_EQ(lastTxGpu[u], srsLastTxCounter[u])
                << "srsLastTxCounter mismatch ue=" << u << " seed=" << seed << " v1=" << useV1;
        }

        for (uint16_t c = 0; c < nCell; ++c) cudaFreeHost(schdCpu.sortedUeList[c]);
        cudaFreeHost(schdCpu.sortedUeList);
    }

    cudaStream_t stream_ = nullptr;
};

// ---- kernel_v0 / cpuScheduler_v0 (round-robin / age based) ------------------

// Small per-cell population: nMaxActUePerCell < nSymbsPerSlot*SRS_COMB_SIZE (56)
// so the "nSrsScheduledUePerCell > nAssocUeFound" cap takes its true leg.
TEST_F(MultiCellSrsSchedulerTest, V0_SmallPopulationRandomSeeds)
{
    for (uint32_t seed = 0; seed < 60; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/16,
                      /*nBsAnt=*/4, /*srsSchedulingSel=*/0);
}

// Large per-cell population (64 > 56): the cap's false leg + a full sorting
// network with pow2N=64.
TEST_F(MultiCellSrsSchedulerTest, V0_LargePopulationRandomSeeds)
{
    for (uint32_t seed = 100; seed < 130; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/4, /*srsSchedulingSel=*/0);
}

// Invalid SRS config/bw indices exercise the W_SRS_LAST table-lookup guard's
// false leg.
TEST_F(MultiCellSrsSchedulerTest, V0_InvalidConfigGuard)
{
    for (uint32_t seed = 200; seed < 210; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/16,
                      /*nBsAnt=*/4, /*srsSchedulingSel=*/0,
                      /*injectInvalidCfg=*/true);
}

// Cell 0 has no associated UEs: the no-associated-UE early-return path.
TEST_F(MultiCellSrsSchedulerTest, V0_EmptyCellEarlyReturn)
{
    for (uint32_t seed = 300; seed < 305; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/16,
                      /*nBsAnt=*/4, /*srsSchedulingSel=*/0,
                      /*injectInvalidCfg=*/false, /*emptyCell0=*/true);
}

// Deterministic per-UE power-control inputs that drive the transmit-power-control
// routine (cpuSrsSchedulerTpc on the host, multiCellSrsSchedulerTpcKernel on the
// device) through every accumulation/threshold branch. One cell with all UEs
// associated, so every UE is scheduled and its TPC branch executes on both sides.
TEST_F(MultiCellSrsSchedulerTest, V0_TpcBranchSweep)
{
    for (uint32_t seed = 400; seed < 405; ++seed)
        RunAndCompare(seed, /*nCell=*/1, /*nMaxActUePerCell=*/16,
                      /*nBsAnt=*/4, /*srsSchedulingSel=*/0,
                      /*injectInvalidCfg=*/false, /*emptyCell0=*/false,
                      /*craftTpc=*/true);
}

// ---- kernel_v1 / cpuScheduler_v1 (64TR MU/SU-MIMO + age) --------------------
//
// All v1 cases use a large per-cell population (nMaxActUePerCell=64): kernel_v1
// does an out-of-bounds device read at small nMaxActUePerCell, so keeping the
// population large keeps every invocation in the safe regime.

TEST_F(MultiCellSrsSchedulerTest, V1_RandomSeeds)
{
    for (uint32_t seed = 1000; seed < 1080; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1);
}

TEST_F(MultiCellSrsSchedulerTest, V1_ThreeCells)
{
    for (uint32_t seed = 1100; seed < 1120; ++seed)
        RunAndCompare(seed, /*nCell=*/3, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1);
}

TEST_F(MultiCellSrsSchedulerTest, V1_InvalidConfigGuard)
{
    for (uint32_t seed = 1200; seed < 1215; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1,
                      /*injectInvalidCfg=*/true);
}

// Cell 0 empty (no associated UEs, counters zeroed): exercises the age-phase
// no-associated-UE early return and the matching CPU per-cell skip, with both
// sides scheduling nothing for cell 0.
TEST_F(MultiCellSrsSchedulerTest, V1_EmptyCellEarlyReturn)
{
    for (uint32_t seed = 1300; seed < 1305; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1,
                      /*injectInvalidCfg=*/false, /*emptyCell0=*/true);
}

// Lightly populated cells: the MU/SU scheduling loops run to completion without
// hitting their per-symbol caps, exercising the loop-exit branches.
TEST_F(MultiCellSrsSchedulerTest, V1_PartialPopulationLoopCompletion)
{
    for (uint32_t seed = 1400; seed < 1420; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1,
                      /*injectInvalidCfg=*/false, /*emptyCell0=*/false,
                      /*craftTpc=*/false, /*nAssocPerCell=*/20);
}

// Moderately populated cells: MU/SU fill up to the per-symbol caps and the age
// phase schedules the remainder, so the age-phase per-cell cap takes its
// "shrink to candidate count" branch.
TEST_F(MultiCellSrsSchedulerTest, V1_PartialPopulationAgeCap)
{
    for (uint32_t seed = 1500; seed < 1520; ++seed)
        RunAndCompare(seed, /*nCell=*/2, /*nMaxActUePerCell=*/64,
                      /*nBsAnt=*/64, /*srsSchedulingSel=*/1,
                      /*injectInvalidCfg=*/false, /*emptyCell0=*/false,
                      /*craftTpc=*/false, /*nAssocPerCell=*/45);
}

// Host-only exercise of cpuScheduler_v1's "cell filled to capacity" early-exit
// branches in the MU/SU loops. These require a small nMaxActUePerCell so the
// scheduled count reaches capacity. The matching GPU kernel_v1 path has a latent
// out-of-bounds read at small nMaxActUePerCell (see the V1 comment block), so
// this scenario runs against the pure-host reference only, with no GPU launch or
// GPU/CPU comparison.
TEST_F(MultiCellSrsSchedulerTest, CpuV1SmallPopulationFillsCell)
{
    // muAll == true  : every UE is MU-MIMO, so the MU loop alone fills the cell
    //                  and hits its capacity early-exit.
    // muAll == false : half MU / half SU, so the SU loop finishes the fill and
    //                  hits its own capacity early-exit.
    for (bool muAll : {true, false}) {
        constexpr uint16_t nCell = 1;
        constexpr uint16_t nMax  = 8;
        constexpr uint16_t nUe   = nCell * nMax;
        const uint16_t nSymbsPerSlot = 14;

        multiCellSrsScheduler sched;

        cumacSrsCellGrpPrms     prm{};
        cumacSrsCellGrpUeStatus st{};
        cumacSchdSol            schd{};
        cumacSrsSchdSol         sol{};

        prm.nActiveUe = nUe; prm.nMaxActUePerCell = nMax; prm.nCell = nCell;
        prm.nSymbsPerSlot = nSymbsPerSlot; prm.nBsAnt = 64; prm.srsSchedulingSel = 1;

        std::vector<uint16_t> cellId(nCell); for (uint16_t c=0;c<nCell;++c) cellId[c]=c;
        prm.cellId = cellId.data();
        std::vector<uint8_t> assoc(static_cast<size_t>(nCell)*nUe, 0);
        for (uint16_t u=0; u<nUe; ++u) assoc[(u/nMax)*nUe + u] = 1;
        prm.cellAssocActUe = assoc.data();

        std::vector<int8_t>   newData(nUe, 1);
        std::vector<uint32_t> lastTx(nUe, 0);          // -> 1 after association bump
        std::vector<float>    wbSnr(nUe, 0.0f), wbSnrThr(nUe, 5.0f), wbEnergy(nUe, 1.0f);
        std::vector<float>    txPwrMax(nUe, 23.0f), pwr0(nUe, -10.0f), pwrAlpha(nUe, 0.5f);
        std::vector<uint8_t>  tpcAccum(nUe, 1);
        std::vector<float>    pcAdj(nUe, 0.0f);
        std::vector<uint8_t>  phr(nUe, 0), muInd(nUe);
        for (uint16_t u=0; u<nUe; ++u) muInd[u] = muAll ? 1 : (u < nMax/2 ? 1 : 0);

        st.newDataActUe=newData.data(); st.srsLastTxCounter=lastTx.data();
        st.srsWbSnr=wbSnr.data(); st.srsWbSnrThreshold=wbSnrThr.data();
        st.srsWidebandSignalEnergy=wbEnergy.data(); st.srsTxPwrMax=txPwrMax.data();
        st.srsPwr0=pwr0.data(); st.srsPwrAlpha=pwrAlpha.data();
        st.srsTpcAccumulationFlag=tpcAccum.data();
        st.srsPowerControlAdjustmentState=pcAdj.data(); st.srsPowerHeadroomReport=phr.data();

        schd.muMimoInd = muInd.data();
        std::vector<std::vector<uint16_t>> sortedStore(nCell, std::vector<uint16_t>(nMax));
        std::vector<uint16_t*> sortedPtr(nCell);
        for (uint16_t c=0;c<nCell;++c){ for (uint16_t l=0;l<nMax;++l) sortedStore[c][l]=c*nMax+l;
                                       sortedPtr[c]=sortedStore[c].data(); }
        schd.sortedUeList = sortedPtr.data();

        std::vector<uint16_t> nSch(nCell,0), txUe(static_cast<size_t>(nMax)*nCell,0), rxCell(nUe,0), freqShift(nUe,0), seqId(nUe,0);
        std::vector<uint8_t>  numSymb(nUe,0), timeStart(nUe,0), numRep(nUe,0), cfg(nUe,63), bw(nUe,0),
                              comb(nUe,0), combOff(nUe,0), freqStart(nUe,0), freqHop(nUe,0), gosh(nUe,0), cyc(nUe,0);
        std::vector<float>    txPwr(nUe, 15.0f);
        sol.nSrsScheduledUePerCell=nSch.data(); sol.srsTxUe=txUe.data(); sol.srsRxCell=rxCell.data();
        sol.srsNumSymb=numSymb.data(); sol.srsTimeStart=timeStart.data(); sol.srsNumRep=numRep.data();
        sol.srsConfigIndex=cfg.data(); sol.srsBwIndex=bw.data(); sol.srsCombSize=comb.data();
        sol.srsCombOffset=combOff.data(); sol.srsFreqStart=freqStart.data(); sol.srsFreqShift=freqShift.data();
        sol.srsFreqHopping=freqHop.data(); sol.srsSequenceId=seqId.data();
        sol.srsGroupOrSequenceHopping=gosh.data(); sol.srsCyclicShift=cyc.data(); sol.srsTxPwr=txPwr.data();

        sched.cpuScheduler_v1(&st, &prm, &schd, &sol);

        // The cell is fully packed: every one of its UEs is scheduled.
        EXPECT_EQ(nSch[0], nMax);
    }
}

// debugLog() is a no-op kept for parity with the production API surface.
TEST_F(MultiCellSrsSchedulerTest, DebugLogIsCallable)
{
    multiCellSrsScheduler sched;
    sched.debugLog();
    SUCCEED();
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    
    return rc;
}
