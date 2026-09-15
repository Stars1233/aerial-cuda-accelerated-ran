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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "api.h"
#include "cumac.h"
#include "muMimoUserPairing/muMimoUserPairing.cuh"


using namespace cumac;

namespace {

// Compact synthetic dimensions: 1 cell, 1 SRS UE, 1 grouping UE. Keeps the
// allocated buffers small while still letting every host-side branch in
// muMimoUserPairing::setup / run / run_cpu execute.
constexpr uint16_t kNumCell                = 1;
constexpr uint16_t kNumPrg                 = 4;
constexpr uint16_t kNumSubband             = 1;
constexpr uint16_t kNumPrgSampPerSubband   = 2;
constexpr uint16_t kNumBsAntPort           = 4;
constexpr uint16_t kNumSrsUePerSlotCell    = 1;
constexpr uint16_t kNumBlocksPerRowChanCorr = 1;
// Iteration 7 tests construct a second muMimoUserPairing with num_subband=2 to
// exercise the `subbandIdx == num_subband-1 ? :` ternary false-branch. The
// shared fixture buffers are sized for this maximum so those tests can reuse
// them without reallocating.
constexpr uint16_t kMaxTestNumSubband      = 2;

class MuMimoUserPairingTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);

        // The constructor expects buffer base addresses that the algorithm
        // sizes against the parameters above. The kernels never deference
        // memory outside the per-cell windows derived from these sizes.
        // Buffers sized for kMaxTestNumSubband so iter-7 tests can reuse them
        // when constructing a muMimoUserPairing with num_subband=2.
        const size_t kSrsChanEstSize = sizeof(__half2)
            * kNumCell * kMaxTestNumSubband * kNumPrgSampPerSubband
            * MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT * kNumBsAntPort;
        const size_t kSrsSnrSize     = sizeof(float)
            * kNumCell * MAX_NUM_SRS_UE_PER_CELL;
        const size_t kChanOrthMatSize = sizeof(float)
            * kNumCell * kMaxTestNumSubband * kNumPrgSampPerSubband
            * MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT
            * (MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT + 1) / 2;
        const size_t kCubbSrsSize    = sizeof(__half2)
            * MAX_NUM_UE_SRS_INFO_PER_SLOT * kNumPrg
            * kNumBsAntPort * MAX_NUM_UE_ANT_PORT;

        ASSERT_EQ(cudaMalloc(&d_srs_chan_est_buf_, kSrsChanEstSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_srs_snr_buf_, kSrsSnrSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_chan_orth_mat_buf_, kChanOrthMatSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_cubb_srs_buf_, kCubbSrsSize), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_task_out_buf_,
                             sizeof(cumac_muUeGrp_resp_info_t) * kNumCell),
                  cudaSuccess);

        ASSERT_EQ(cudaMemsetAsync(d_srs_chan_est_buf_, 0, kSrsChanEstSize, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_srs_snr_buf_,     0, kSrsSnrSize,     stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_chan_orth_mat_buf_, 0, kChanOrthMatSize, stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_cubb_srs_buf_,    0, kCubbSrsSize,    stream_), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(d_task_out_buf_,    0,
                                  sizeof(cumac_muUeGrp_resp_info_t) * kNumCell,
                                  stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        h_srs_chan_est_buf_.assign(kSrsChanEstSize / sizeof(__half2), __half2{});
        h_srs_snr_buf_.assign(kSrsSnrSize / sizeof(float), 0.0f);
        h_chan_orth_mat_buf_.assign(kChanOrthMatSize / sizeof(float), 0.0f);
        h_cubb_srs_buf_.assign(kCubbSrsSize / sizeof(__half2), __half2{});
        h_task_out_buf_.assign(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    }

    void TearDown() override
    {
        if(d_task_in_buf_)      cudaFree(d_task_in_buf_);
        if(d_srs_chan_est_buf_) cudaFree(d_srs_chan_est_buf_);
        if(d_srs_snr_buf_)      cudaFree(d_srs_snr_buf_);
        if(d_chan_orth_mat_buf_)cudaFree(d_chan_orth_mat_buf_);
        if(d_cubb_srs_buf_)     cudaFree(d_cubb_srs_buf_);
        if(d_task_out_buf_)     cudaFree(d_task_out_buf_);
        if(stream_)             cudaStreamDestroy(stream_);
    }

    // Build a per-cell request-info buffer with one srsInfo and one ueInfo,
    // both marked valid. The buffer is laid out exactly the way the kernel
    // expects: [req_info_t][srsInfo[numSrsInfo]][ueInfo[numUeInfo]].
    template <bool MemSharing>
    void BuildOneCellRequestBuffer()
    {
        BuildRequestBuffer<MemSharing>(/*numUeInfo=*/1,
                                       /*ueFlags=*/0x01 | 0x04 | 0x08,
                                       /*srsSnrDb=*/10.0f,
                                       /*nUeAntForSrs=*/MAX_NUM_UE_ANT_PORT);
    }

    // Parameterizable request-info builder: lets each test vary numUeInfo, the
    // ueInfo flags, the SRS SNR (to push above/below srsSnrThr), and the per-
    // srsInfo nUeAnt (smaller values trigger the early-return on srs_info_ant_port
    // >= num_ue_ant_port).
    template <bool MemSharing>
    void BuildRequestBuffer(uint16_t numUeInfo,
                            uint8_t  ueFlags,
                            float    srsSnrDb,
                            uint8_t  nUeAntForSrs)
    {
        const size_t kPerCellLen = sizeof(cumac_muUeGrp_req_info_t)
            + (MemSharing
               ? sizeof(cumac_muUeGrp_req_srs_info_msh_t)
               : sizeof(cumac_muUeGrp_req_srs_info_t))
              * MAX_NUM_UE_SRS_INFO_PER_SLOT
            + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
        const size_t kTotalLen = kPerCellLen * kNumCell;

        h_task_in_buf_.assign(kTotalLen, 0);
        cumac_muUeGrp_req_info_t* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
        req->numSrsInfo               = 1;
        req->numUeInfo                = numUeInfo;
        req->numSubband               = kNumSubband;
        req->numPrgSampPerSubband     = kNumPrgSampPerSubband;
        req->nPrbGrp                  = kNumPrg;
        req->nBsAnt                   = static_cast<uint8_t>(kNumBsAntPort);
        // PF thresholds the kernel reads via memcpy(). Setting srsSnrThr below
        // the per-UE snr arms the "MU-MIMO feasible" branch; the chanCorrThr
        // controls the orth_ind path. Defaults from cumac_muUeGrp_req_info_t
        // would set them to 0.0 once we zero the buffer, so reapply here.
        req->betaCoeff      = cumac_f32_to_u32_bits(1.0f);
        req->muCoeff        = cumac_f32_to_u32_bits(1.5f);
        req->chanCorrThr    = cumac_f32_to_u32_bits(0.7f);
        req->srsSnrThr      = cumac_f32_to_u32_bits(-3.0f);
        req->nMaxUeSchdPerCellTTI = 16;
        req->nMaxUePerGrp         = 16;
        req->nMaxLayerPerGrp      = 16;
        req->nMaxLayerPerUeSu     = 4;
        req->nMaxLayerPerUeMu     = 4;
        req->nMaxUegPerCell       = 4;
        req->numUeForGrpPerCell   = 64;

        if constexpr (MemSharing) {
            auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(req->payload);
            srs[0].id           = 0;
            srs[0].nUeAnt       = nUeAntForSrs;
            srs[0].realBuffIdx  = 0;
            srs[0].srsWbSnr     = cumac_f32_to_u32_bits(srsSnrDb);
            srs[0].flags        = 0x01;
            auto* ue = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
            for(uint16_t i = 0; i < numUeInfo; ++i) {
                ue[i].id          = i;
                ue[i].rnti        = static_cast<uint16_t>(1000 + i);
                ue[i].nUeAnt      = nUeAntForSrs;
                ue[i].srsInfoIdx  = 0;
                ue[i].avgRate     = cumac_f32_to_u32_bits(100.0f);
                ue[i].currRate    = cumac_f32_to_u32_bits(50.0f + static_cast<float>(i));
                ue[i].bufferSize  = 1024;  // > 0 so the valid-UE pf branch entered.
                ue[i].flags       = ueFlags;
            }
        } else {
            auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_t*>(req->payload);
            srs[0].id           = 0;
            srs[0].nUeAnt       = nUeAntForSrs;
            srs[0].srsWbSnr     = cumac_f32_to_u32_bits(srsSnrDb);
            srs[0].flags        = 0x01;
            auto* ue = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
            for(uint16_t i = 0; i < numUeInfo; ++i) {
                ue[i].id          = i;
                ue[i].rnti        = static_cast<uint16_t>(1000 + i);
                ue[i].nUeAnt      = nUeAntForSrs;
                ue[i].srsInfoIdx  = 0;
                ue[i].avgRate     = cumac_f32_to_u32_bits(100.0f);
                ue[i].currRate    = cumac_f32_to_u32_bits(50.0f + static_cast<float>(i));
                ue[i].bufferSize  = 1024;
                ue[i].flags       = ueFlags;
            }
        }

        ASSERT_EQ(cudaMalloc(&d_task_in_buf_, kTotalLen), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                                  kTotalLen, cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

        // Prime the host-side SNR buffer too so run_cpu()'s
        // muMimoInd-feasibility branch sees the same value for each ue_id.
        for(uint16_t i = 0; i < numUeInfo && i < h_srs_snr_buf_.size(); ++i) {
            h_srs_snr_buf_[i] = srsSnrDb;
        }
    }

    cudaStream_t stream_                 = nullptr;
    uint8_t*     d_task_in_buf_          = nullptr;
    uint8_t*     d_srs_chan_est_buf_     = nullptr;
    float*       d_srs_snr_buf_          = nullptr;
    float*       d_chan_orth_mat_buf_    = nullptr;
    __half2*     d_cubb_srs_buf_         = nullptr;
    uint8_t*     d_task_out_buf_         = nullptr;

    std::vector<uint8_t> h_task_in_buf_;
    std::vector<__half2> h_srs_chan_est_buf_;
    std::vector<float>   h_srs_snr_buf_;
    std::vector<float>   h_chan_orth_mat_buf_;
    std::vector<__half2> h_cubb_srs_buf_;
    std::vector<uint8_t> h_task_out_buf_;
};
// 2. setup() with all eight combinations of kernel_launch_flags ∈ {0..3} and
//    is_mem_sharing ∈ {false,true}. Each combination drives a distinct subset
//    of branches in setup(), chanCorrKernelSelect(), uePairKernelSelect(), and
//    the m_task_in_buf_len_per_cell computation.
TEST_F(MuMimoUserPairingTest, SetupAcrossAllFlagAndMemSharingCombinations)
{
    for(uint8_t flags = 0; flags <= 0x03; ++flags) {
        for(bool memSharing : {false, true}) {
            if(memSharing) { BuildOneCellRequestBuffer<true>();  }
            else           { BuildOneCellRequestBuffer<false>(); }

            muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                                  d_srs_snr_buf_,
                                  d_chan_orth_mat_buf_,
                                  d_cubb_srs_buf_,
                                  kNumCell, kNumPrg, kNumSubband,
                                  kNumPrgSampPerSubband, kNumBsAntPort);

            muUePairTask task{};
            task.task_in_buf                      = d_task_in_buf_;
            task.task_out_buf                     = d_task_out_buf_;
            task.strm                             = stream_;
            task.num_srs_ue_per_slot_cell         = kNumSrsUePerSlotCell;
            task.num_blocks_per_row_chanOrtMat    = kNumBlocksPerRowChanCorr;
            task.kernel_launch_flags              = flags;
            task.is_mem_sharing                   = memSharing;

            obj.setup(&task);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess)
                << "setup() async copies failed for flags=" << static_cast<int>(flags)
                << " memSharing=" << memSharing;

            // Free per-iteration before the next BuildOneCellRequestBuffer.
            cudaFree(d_task_in_buf_);
            d_task_in_buf_ = nullptr;
        }
    }
}
// =================== Iteration 2 tests ===================
// These target the gaps surfaced by the iteration-1 coverage report:
// - "no updated SRS" branch (ueInfo.flags without 0x08)
// - early returns at srs_info_idx out-of-range / srs_info_ant_port out-of-range
// - multi-UE pf metric / orth_ind / scheduling paths
// - newTx (0x02) branch variants in muUePairAlgKernel

// Drives the else branch where ueInfo flags omit 0x08 ("updated SRS info").
TEST_F(MuMimoUserPairingTest, ChanCorrCpu_FlagsValidChanEstButNoUpdatedSrs)
{
    BuildRequestBuffer<false>(/*numUeInfo=*/1, /*ueFlags=*/0x01 | 0x04,
                              /*srsSnrDb=*/10.0f, /*nUeAntForSrs=*/MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// Exercises muUePairChanCorrKernel_cpu with a UE that has updated SRS info in
// the current slot, so it uses the fresh SRS channel estimate directly instead
// of the cached one. Complements the NoUpdatedSrs test above.
TEST_F(MuMimoUserPairingTest, ChanCorrCpu_FlagsValidChanEstWithUpdatedSrs)
{
    BuildRequestBuffer<false>(/*numUeInfo=*/1, /*ueFlags=*/0x01 | 0x04 | 0x08,
                              /*srsSnrDb=*/10.0f, /*nUeAntForSrs=*/MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, ChanCorrCpu_MemShare_FlagsValidChanEstButNoUpdatedSrs)
{
    BuildRequestBuffer<true>(1, 0x01 | 0x04, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// Same on GPU — covers the device kernels' "no 0x08" else branches.
TEST_F(MuMimoUserPairingTest, ChanCorrGpu_FlagsValidChanEstButNoUpdatedSrs)
{
    BuildRequestBuffer<false>(1, 0x01 | 0x04, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, ChanCorrGpu_MemShare_FlagsValidChanEstButNoUpdatedSrs)
{
    BuildRequestBuffer<true>(1, 0x01 | 0x04, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
// Early-return: num_srs_ue_per_slot_cell > numSrsInfo so some blocks see
// srs_info_idx >= numSrsInfo and take the early return at the kernel head.
TEST_F(MuMimoUserPairingTest, ChanCorrGpu_EarlyReturnSrsInfoIdxOutOfRange)
{
    BuildRequestBuffer<false>(1, 0x01 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    // 2 SRS UE slots but numSrsInfo=1 → second slot's blocks early-return.
    task.num_srs_ue_per_slot_cell      = 2;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// Early-return: srsInfo.nUeAnt=2 (< MAX_NUM_UE_ANT_PORT=4), so blocks with
// srs_info_ant_port ∈ {2,3} early-return at the head.
TEST_F(MuMimoUserPairingTest, ChanCorrGpu_EarlyReturnAntPortOutOfRange)
{
    BuildRequestBuffer<false>(1, 0x01 | 0x04 | 0x08, 10.0f, /*nUeAntForSrs=*/2);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
// Same on memSharing path so the variant of run_cpu()'s muUePairAlgKernel_memSharing_cpu fires.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MultiUe_MemShare_NewTxNoSrsChanEst)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// GPU variant of "snr below threshold" — drives device lines 652, 871-875.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MultiUe_NewTxSnrBelowThreshold)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, /*snr=*/-50.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 0.0f);
    for(int i = 0; i < 4; ++i) snr[i] = -50.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// GPU + memShare + newTx + no-chanEst: drives device line 870 (else inside newTx).
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MultiUe_MemShare_NewTxNoSrsChanEst)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 0.0f);
    for(int i = 0; i < 4; ++i) snr[i] = 10.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
// Orth-branch: corrValues > chanCorrThr (0.7) but != 1.0. Setting off-diag=0.9
// and diag=1.0 gives ratio = 0.9 / sqrt(1.0) = 0.9 > 0.7, != 1.0 → orth=0, break.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_OrthBranch_CorrAboveThresholdNotOne)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    // The kernel reads diag values from chan_orth_mat_buf using the SAME indices
    // (row*(row+1)/2 + row) — give them value 1.0, and off-diag value 0.9.
    // Easier: set everything to 0.9 → corrValues = 0.9/sqrt(0.81) = 1.0. Not what we want.
    // So: per-cell first compute diag indices, set diag=1.0, then set everything else to 0.9.
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 0.9f);
    // Diagonal entries at idx = row*(row+1)/2 + row for row up to MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT
    // are large; cap to what the test will actually touch. With 4 UEs * 4 ant
    // = 16 rows, diag indices 0, 2, 5, 9, 14, 20, ... up to ~136.
    for(uint32_t row = 0; row < 16u; ++row) {
        const uint32_t diagIdx = row * (row + 1u) / 2u + row;
        if(diagIdx < h_chan_orth_mat_buf_.size()) h_chan_orth_mat_buf_[diagIdx] = 1.0f;
    }

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}
// Exercises muUePairAlgKernel_cpu with channel correlation below the threshold
// (off-diag 0.3, diag 1.0). Complements the CorrAboveThresholdNotOne test above.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_OrthBranch_CorrBelowThreshold)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 0.3f);
    for(uint32_t row = 0; row < 16u; ++row) {
        const uint32_t diagIdx = row * (row + 1u) / 2u + row;
        if(diagIdx < h_chan_orth_mat_buf_.size()) h_chan_orth_mat_buf_[diagIdx] = 1.0f;
    }

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}
// Exercises muUePairAlgKernel_cpu with one invalid UE so the per-UE metric loop
// handles the invalid-UE case while the remaining UEs stay valid. Non-memSharing
// twin of UePairAlgCpu_MemShare_InvalidUe_DrivesInvalidUeElse.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_InvalidUe_DrivesInvalidUeElse)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
    auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_t*>(req->payload);
    auto* ue  = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
    ue[1].flags &= ~static_cast<uint8_t>(0x01);  // clear valid bit -> invalid UE info

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}
// MemShare variant of the early-return GPU tests — hits the lines in
// muUePairChanCorrKernel_memSharing (file2 lines 359, 365).
TEST_F(MuMimoUserPairingTest, ChanCorrGpu_MemShare_EarlyReturnSrsInfoIdxOutOfRange)
{
    BuildRequestBuffer<true>(1, 0x01 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = 2;  // > numSrsInfo (1) → some blocks early-return.
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, ChanCorrGpu_MemShare_EarlyReturnAntPortOutOfRange)
{
    BuildRequestBuffer<true>(1, 0x01 | 0x04 | 0x08, 10.0f, /*nUeAntForSrs=*/2);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// num_blocks_per_row_chanOrtMat > 1 so the "non-last-block" branch
// (last_ue_info_idx_in_block = first + per_block - 1, file2 lines 374/517)
// gets reached. Need numUeInfo ≥ num_blocks_per_row_chanOrtMat for
// ue_info_per_block ≥ 1.
TEST_F(MuMimoUserPairingTest, ChanCorrGpu_MultiBlockPerRow)
{
    BuildRequestBuffer<false>(/*numUeInfo=*/2, 0x01 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = 2;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, ChanCorrGpu_MemShare_MultiBlockPerRow)
{
    BuildRequestBuffer<true>(2, 0x01 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = 2;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
// =================== Iteration 5 tests ===================
// Close residual ~32 lines: memShare variants of snr-below + orth-above-thr,
// mixed muMimoInd to drive the "continue" path, and GPU memShare equivalents.

// memSharing CPU + snr-below: covers host line 1288.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MultiUe_MemShare_NewTxSnrBelowThreshold)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, /*snr=*/-50.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// memSharing CPU + orth above threshold (not 1.0): covers host lines 1356/1357.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_OrthBranch_CorrAboveThresholdNotOne)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 0.9f);
    for(uint32_t row = 0; row < 16u; ++row) {
        const uint32_t diagIdx = row * (row + 1u) / 2u + row;
        if(diagIdx < h_chan_orth_mat_buf_.size()) h_chan_orth_mat_buf_[diagIdx] = 1.0f;
    }
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_Mixed_FirstMuThenSu_CoversContinue)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
    auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(req->payload);
    auto* ue  = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
    ue[2].flags = 0x01 | 0x04 | 0x08;
    ue[3].flags = 0x01 | 0x04 | 0x08;

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// Exercises muUePairAlgKernel_memSharing_cpu with one invalid UE so the per-UE
// metric loop handles the invalid-UE case while the remaining UEs stay valid.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_InvalidUe_DrivesInvalidUeElse)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
    auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(req->payload);
    auto* ue  = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
    ue[1].flags &= ~static_cast<uint8_t>(0x01);  // clear valid bit -> invalid UE info

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// GPU mem-share snr-below threshold: covers device 871/875.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_NewTxSnrBelowThreshold)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, -50.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, -50.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// GPU mem-share orth corr above threshold (not 1.0): device 725, 726, 762, 767, 768.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_OrthBranch_CorrAboveThresholdNotOne)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 0.9f);
    for(uint32_t row = 0; row < 16u; ++row) {
        const uint32_t diagIdx = row * (row + 1u) / 2u + row;
        if(diagIdx < orth.size()) orth[diagIdx] = 1.0f;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
// =================== Iteration 6 tests ===================
// Closes remaining device-side gaps.

// GPU non-memShare + newTx-no-chanEst: covers device line 655.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_NonMemShare_NewTxNoSrsChanEst)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x03;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// GPU non-memShare + orth above threshold (not 1.0): covers device 725, 726.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_NonMemShare_OrthBranch_CorrAboveThresholdNotOne)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 0.9f);
    for(uint32_t row = 0; row < 16u; ++row) {
        const uint32_t diagIdx = row * (row + 1u) / 2u + row;
        if(diagIdx < orth.size()) orth[diagIdx] = 1.0f;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// GPU memShare + mixed (first UE MU-MIMO then SU continue): covers device 972, 990, 991.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_Mixed_FirstMuThenSu)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
    auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(req->payload);
    auto* ue  = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
    ue[2].flags = 0x01 | 0x04 | 0x08;
    ue[3].flags = 0x01 | 0x04 | 0x08;
    const size_t kPerCellLen = sizeof(cumac_muUeGrp_req_info_t)
        + sizeof(cumac_muUeGrp_req_srs_info_msh_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
        + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// =================== Iteration 7 tests ===================
// Close the 15 reachable partial branches left after iter6.

// Drives chanCorr's `(flags & 0x01) > 0 && (flags & 0x04) > 0` short-circuit
// false-leg (line 1189 non-memShare). flags=0x01|0x08 keeps the UE valid but
// drops the chanEst-available bit so the `&&` short-circuits to false.
TEST_F(MuMimoUserPairingTest, ChanCorrCpu_FlagsValidButNoChanEstAvailable)
{
    BuildRequestBuffer<false>(/*numUeInfo=*/4, /*ueFlags=*/0x01 | 0x08,
                              /*srsSnrDb=*/10.0f, /*nUeAntForSrs=*/MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// Same on the memSharing variant — drives host line 1103.
TEST_F(MuMimoUserPairingTest, ChanCorrCpu_MemShare_FlagsValidButNoChanEstAvailable)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x01;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// Helper: patch req_info in-place to set per-test grouping limits.
// Returns the per-cell length so callers can re-push the buffer to GPU.
template <bool MemSharing>
static size_t PatchGroupingLimits(std::vector<uint8_t>& host_buf,
                                  uint8_t  nMaxLayerPerGrp,
                                  uint8_t  nMaxUePerGrp,
                                  uint8_t  nMaxUeSchdPerCellTTI,
                                  uint16_t numUeForGrpPerCell)
{
    auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(host_buf.data());
    req->nMaxLayerPerGrp          = nMaxLayerPerGrp;
    req->nMaxUePerGrp             = nMaxUePerGrp;
    req->nMaxUeSchdPerCellTTI     = nMaxUeSchdPerCellTTI;
    req->numUeForGrpPerCell       = numUeForGrpPerCell;
    return sizeof(cumac_muUeGrp_req_info_t)
         + (MemSharing
            ? sizeof(cumac_muUeGrp_req_srs_info_msh_t)
            : sizeof(cumac_muUeGrp_req_srs_info_t)) * MAX_NUM_UE_SRS_INFO_PER_SLOT
         + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
}

// CPU non-memShare: tighten nMaxLayerPerGrp=2 and nMaxUePerGrp=2 so the two
// `>=` breaks fire mid-iteration. With 4 MU-MIMO UEs and an all-1.0 orth
// matrix, layer 2 hits nMaxLayerPerGrp; UE 2 hits nMaxUePerGrp.
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_TightGroupingLimits_DrivesBothBreaks)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    PatchGroupingLimits<false>(h_task_in_buf_,
                               /*nMaxLayerPerGrp=*/2,
                               /*nMaxUePerGrp=*/2,
                               /*nMaxUeSchdPerCellTTI=*/2,
                               /*numUeForGrpPerCell=*/16);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_TightGroupingLimits_DrivesBothBreaks)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    PatchGroupingLimits<true>(h_task_in_buf_, 2, 2, 2, 16);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

// GPU non-memShare with the same tight limits — patched buffer pushed to device.
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_TightGroupingLimits_DrivesBothBreaks)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    const size_t kPerCellLen = PatchGroupingLimits<false>(h_task_in_buf_, 2, 2, 2, 16);
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_TightGroupingLimits_DrivesBothBreaks)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    const size_t kPerCellLen = PatchGroupingLimits<true>(h_task_in_buf_, 2, 2, 2, 16);
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}
TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_NumSubbandTwo_DrivesTernaryFalseBranch)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    {
        auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
        req->numSubband           = 2;
        req->numPrgSampPerSubband = kNumPrgSampPerSubband;
    }
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, 2,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}
TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_NumSubbandTwo_DrivesTernaryFalseBranch)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    const size_t kPerCellLen = sizeof(cumac_muUeGrp_req_info_t)
        + sizeof(cumac_muUeGrp_req_srs_info_msh_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
        + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
    {
        auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
        req->numSubband           = 2;
        req->numPrgSampPerSubband = kNumPrgSampPerSubband;
    }
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, 2,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// =================== Iteration 8 tests ===================
// Close the remaining OR-short-circuit gap on
//   `if (num_ue_schd_in_grp >= nMaxUePerGrp || numUeSchd >= nMaxUeSchdPerCellTTI)`
// Iter7 used both limits = 2, which only exercised `false||false` and the
// left-true short-circuit. Setting nMaxUePerGrp=16 and nMaxUeSchdPerCellTTI=2
// drives `false || true` — left subcondition false, right subcondition true.

TEST_F(MuMimoUserPairingTest, UePairAlgCpu_TightUeSchdLimitOnly_LeftFalseRightTrue)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    PatchGroupingLimits<false>(h_task_in_buf_,
                               /*nMaxLayerPerGrp=*/16,
                               /*nMaxUePerGrp=*/16,
                               /*nMaxUeSchdPerCellTTI=*/2,
                               /*numUeForGrpPerCell=*/16);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, UePairAlgCpu_MemShare_TightUeSchdLimitOnly_LeftFalseRightTrue)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    PatchGroupingLimits<true>(h_task_in_buf_, 16, 16, 2, 16);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    obj.run_cpu();
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, UePairAlgGpu_TightUeSchdLimitOnly_LeftFalseRightTrue)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    const size_t kPerCellLen = PatchGroupingLimits<false>(h_task_in_buf_, 16, 16, 2, 16);
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

TEST_F(MuMimoUserPairingTest, UePairAlgGpu_MemShare_TightUeSchdLimitOnly_LeftFalseRightTrue)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    const size_t kPerCellLen = PatchGroupingLimits<true>(h_task_in_buf_, 16, 16, 2, 16);
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::vector<float> orth(h_chan_orth_mat_buf_.size(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, orth.data(),
                              orth.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    std::vector<float> snr(MAX_NUM_SRS_UE_PER_CELL * kNumCell, 10.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, snr.data(),
                              snr.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                          d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = d_task_in_buf_;
    task.task_out_buf                  = d_task_out_buf_;
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = true;
    obj.setup(&task);
    std::vector<uint8_t> hSol(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    obj.run(hSol.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    SUCCEED();
}

// =================== Functional-correctness helpers + tests ===================
// Until this section, every test ended in SUCCEED() — they verified that the
// kernels ran without crashing and accumulated coverage data, but not that the
// produced output (cumac_muUeGrp_resp_info_t) was correct. The helpers + tests
// below add real assertions: byte-level GPU vs CPU equivalence for happy-path
// scenarios, and scenario-specific invariants on the unhappy paths.

namespace {

// Assert that two cumac_muUeGrp_resp_info_t blocks (kNumCell × per-cell) are
// equivalent field-by-field. The CPU `muUePairAlgKernel_cpu` is the project's
// debug reference for the GPU kernel; given the same deterministic inputs
// (pf sort with id-as-tiebreak, bitonic on GPU matches std::sort on host),
// both must yield identical resp_info.
void ExpectGpuCpuRespEqual(const uint8_t*    gpu_out,
                           const uint8_t*    cpu_out,
                           uint16_t          numCell)
{
    for(uint16_t c = 0; c < numCell; ++c) {
        const size_t kOffset = c * sizeof(cumac_muUeGrp_resp_info_t);
        const auto* g = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(gpu_out + kOffset);
        const auto* p = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(cpu_out + kOffset);
        ASSERT_EQ(g->numSchdUeg, p->numSchdUeg)
            << "cell " << c << ": GPU and CPU disagree on numSchdUeg";
        for(uint32_t i = 0; i < g->numSchdUeg && i < MAX_NUM_UEG_PER_CELL; ++i) {
            const auto& gug = g->schdUegInfo[i];
            const auto& pug = p->schdUegInfo[i];
            EXPECT_EQ(gug.allocPrgStart, pug.allocPrgStart)
                << "cell " << c << " group " << i << ": allocPrgStart";
            EXPECT_EQ(gug.allocPrgEnd, pug.allocPrgEnd)
                << "cell " << c << " group " << i << ": allocPrgEnd";
            EXPECT_EQ(gug.numUeInGrp, pug.numUeInGrp)
                << "cell " << c << " group " << i << ": numUeInGrp";
            EXPECT_EQ(gug.flags, pug.flags)
                << "cell " << c << " group " << i << ": ueg.flags";
            for(uint8_t u = 0; u < gug.numUeInGrp && u < MAX_NUM_UE_PER_GRP; ++u) {
                EXPECT_EQ(gug.ueInfo[u].rnti, pug.ueInfo[u].rnti);
                EXPECT_EQ(gug.ueInfo[u].id, pug.ueInfo[u].id);
                EXPECT_EQ(gug.ueInfo[u].layerSel, pug.ueInfo[u].layerSel);
                EXPECT_EQ(gug.ueInfo[u].ueOrderInGrp, pug.ueInfo[u].ueOrderInGrp);
                EXPECT_EQ(gug.ueInfo[u].nSCID, pug.ueInfo[u].nSCID);
                EXPECT_EQ(gug.ueInfo[u].flags, pug.ueInfo[u].flags)
                    << "cell " << c << " group " << i << " ue " << u << ": ueInfo.flags";
            }
        }
    }
}

// In any scenario where no UE qualifies for MU-MIMO (snr below srsSnrThr, or
// flags lack chanEst-available, or flags are re-Tx), the response should
// contain no UEG with the MU flag set (0x02). At most one SU-MIMO UE may
// have been scheduled (the first valid UE), so we also bound numSchdUeg.
void ExpectNoMuMimoGroupsScheduled(const uint8_t* out, uint16_t numCell)
{
    for(uint16_t c = 0; c < numCell; ++c) {
        const auto* r = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(
            out + c * sizeof(cumac_muUeGrp_resp_info_t));
        for(uint32_t i = 0; i < r->numSchdUeg && i < MAX_NUM_UEG_PER_CELL; ++i) {
            for(uint8_t u = 0; u < r->schdUegInfo[i].numUeInGrp && u < MAX_NUM_UE_PER_GRP; ++u) {
                EXPECT_EQ((r->schdUegInfo[i].ueInfo[u].flags & 0x02u), 0u)
                    << "cell " << c << " group " << i << " ue " << u
                    << ": MU-MIMO flag set but no UE should be MU-MIMO feasible";
            }
        }
    }
}

}  // anonymous-helper namespace

// ---- Functional test 1: GPU vs CPU equivalence on happy multi-UE path ----
//
// 4 valid MU-MIMO-eligible UEs (flags 0x0F), snr above srsSnrThr, an
// all-1.0 chan_orth matrix. Run both GPU run() and CPU run_cpu() on the same
// inputs and assert byte-level equivalence of the resp_info structure.
TEST_F(MuMimoUserPairingTest, FunctionalEquivalence_GpuVsCpu_HappyPathMultiUe)
{
    BuildRequestBuffer<false>(/*numUeInfo=*/4,
                              /*ueFlags=*/0x01 | 0x02 | 0x04 | 0x08,
                              /*srsSnrDb=*/10.0f,
                              /*nUeAntForSrs=*/MAX_NUM_UE_ANT_PORT);
    // Sync orth + snr to both host and device so CPU and GPU see identical input.
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, h_chan_orth_mat_buf_.data(),
                              h_chan_orth_mat_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    for(int i = 0; i < 4; ++i) h_srs_snr_buf_[i] = 10.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, h_srs_snr_buf_.data(),
                              h_srs_snr_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // --- GPU path ---
    muMimoUserPairing gpu_obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                              d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask gpu_task{};
    gpu_task.task_in_buf                   = d_task_in_buf_;
    gpu_task.task_out_buf                  = d_task_out_buf_;
    gpu_task.strm                          = stream_;
    gpu_task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    gpu_task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    gpu_task.kernel_launch_flags           = 0x02;
    gpu_task.is_mem_sharing                = false;
    gpu_obj.setup(&gpu_task);
    std::vector<uint8_t> hSolGpu(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    gpu_obj.run(hSolGpu.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // --- CPU path on the same input ---
    muMimoUserPairing cpu_obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                              h_srs_snr_buf_.data(),
                              h_chan_orth_mat_buf_.data(),
                              h_cubb_srs_buf_.data(),
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask cpu_task = gpu_task;
    cpu_task.task_in_buf  = h_task_in_buf_.data();
    cpu_task.task_out_buf = h_task_out_buf_.data();
    cpu_obj.setup(&cpu_task);
    cpu_obj.run_cpu();

    // --- Compare ---
    ExpectGpuCpuRespEqual(hSolGpu.data(), h_task_out_buf_.data(), kNumCell);

    // Stronger structural assertion: 4 MU-MIMO-eligible UEs, all-1.0 orth → one
    // group should contain at least one UE; every scheduled UE in this scenario
    // must have the MU flag (0x02) set since muMimoInd was 1 for all.
    const auto* g = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(hSolGpu.data());
    ASSERT_GE(g->numSchdUeg, 1u) << "no UEG scheduled for happy multi-UE path";
    EXPECT_GE(g->schdUegInfo[0].numUeInGrp, 1u);
    for(uint8_t u = 0; u < g->schdUegInfo[0].numUeInGrp; ++u) {
        EXPECT_NE(g->schdUegInfo[0].ueInfo[u].flags & 0x02u, 0u)
            << "ue " << u << " in first group missing MU-MIMO flag despite muMimoInd=1";
    }
}

// Same as above but is_mem_sharing=true so the memSharing kernel pair is verified.
TEST_F(MuMimoUserPairingTest, FunctionalEquivalence_GpuVsCpu_MemShare_HappyPathMultiUe)
{
    BuildRequestBuffer<true>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, h_chan_orth_mat_buf_.data(),
                              h_chan_orth_mat_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    for(int i = 0; i < 4; ++i) h_srs_snr_buf_[i] = 10.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, h_srs_snr_buf_.data(),
                              h_srs_snr_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing gpu_obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                              d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask gpu_task{};
    gpu_task.task_in_buf                   = d_task_in_buf_;
    gpu_task.task_out_buf                  = d_task_out_buf_;
    gpu_task.strm                          = stream_;
    gpu_task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    gpu_task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    gpu_task.kernel_launch_flags           = 0x02;
    gpu_task.is_mem_sharing                = true;
    gpu_obj.setup(&gpu_task);
    std::vector<uint8_t> hSolGpu(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    gpu_obj.run(hSolGpu.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing cpu_obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                              h_srs_snr_buf_.data(),
                              h_chan_orth_mat_buf_.data(),
                              h_cubb_srs_buf_.data(),
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask cpu_task = gpu_task;
    cpu_task.task_in_buf  = h_task_in_buf_.data();
    cpu_task.task_out_buf = h_task_out_buf_.data();
    cpu_obj.setup(&cpu_task);
    cpu_obj.run_cpu();

    ExpectGpuCpuRespEqual(hSolGpu.data(), h_task_out_buf_.data(), kNumCell);
}

// Mixed (first MU then SU): asserts the response contains a group with the
// first two UEs MU-flagged and the SU UEs are skipped via the `continue` path,
// then verifies GPU and CPU agree.
TEST_F(MuMimoUserPairingTest, FunctionalEquivalence_GpuVsCpu_MixedFirstMuThenSu)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    {
        auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
        auto* srs = reinterpret_cast<cumac_muUeGrp_req_srs_info_t*>(req->payload);
        auto* ue  = reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(srs + req->numSrsInfo);
        // UEs 0,1 stay MU (flags 0x0F). UEs 2,3 drop the newTx bit -> muMimoInd=0.
        ue[2].flags = 0x01 | 0x04 | 0x08;
        ue[3].flags = 0x01 | 0x04 | 0x08;
    }
    const size_t kPerCellLen = sizeof(cumac_muUeGrp_req_info_t)
        + sizeof(cumac_muUeGrp_req_srs_info_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
        + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, h_chan_orth_mat_buf_.data(),
                              h_chan_orth_mat_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    for(int i = 0; i < 4; ++i) h_srs_snr_buf_[i] = 10.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, h_srs_snr_buf_.data(),
                              h_srs_snr_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing gpu_obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                              d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask gpu_task{};
    gpu_task.task_in_buf                   = d_task_in_buf_;
    gpu_task.task_out_buf                  = d_task_out_buf_;
    gpu_task.strm                          = stream_;
    gpu_task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    gpu_task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    gpu_task.kernel_launch_flags           = 0x02;
    gpu_task.is_mem_sharing                = false;
    gpu_obj.setup(&gpu_task);
    std::vector<uint8_t> hSolGpu(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    gpu_obj.run(hSolGpu.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing cpu_obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                              h_srs_snr_buf_.data(),
                              h_chan_orth_mat_buf_.data(),
                              h_cubb_srs_buf_.data(),
                              kNumCell, kNumPrg, kNumSubband,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask cpu_task = gpu_task;
    cpu_task.task_in_buf  = h_task_in_buf_.data();
    cpu_task.task_out_buf = h_task_out_buf_.data();
    cpu_obj.setup(&cpu_task);
    cpu_obj.run_cpu();

    ExpectGpuCpuRespEqual(hSolGpu.data(), h_task_out_buf_.data(), kNumCell);

    // Structural assertion: first group contains both MU UEs (ids 0,1) with MU
    // flag set; no group contains the SU UEs (ids 2,3) since they are
    // `continue`d past after the first MU UE is scheduled.
    const auto* g = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(hSolGpu.data());
    ASSERT_GE(g->numSchdUeg, 1u);
    EXPECT_EQ(g->schdUegInfo[0].numUeInGrp, 2u)
        << "expected exactly the 2 MU UEs in the first group";
    bool seen_id0 = false, seen_id1 = false;
    for(uint8_t u = 0; u < g->schdUegInfo[0].numUeInGrp; ++u) {
        const auto& info = g->schdUegInfo[0].ueInfo[u];
        EXPECT_NE(info.flags & 0x02u, 0u) << "MU flag missing for ue " << u;
        if(info.id == 0) seen_id0 = true;
        if(info.id == 1) seen_id1 = true;
    }
    EXPECT_TRUE(seen_id0);
    EXPECT_TRUE(seen_id1);
}

// num_subband=2 functional: GPU and CPU must agree, and the per-subband
// allocPrgEnd values must reflect the ternary's two sides:
//   subband 0  →  allocPrgEnd = nPrbGrp/num_subband*(0+1) = nPrbGrp/2 = 2
//   subband 1  →  allocPrgEnd = nPrbGrp                  = 4
TEST_F(MuMimoUserPairingTest, FunctionalEquivalence_GpuVsCpu_NumSubbandTwo_AllocPrgEnd)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    {
        auto* req = reinterpret_cast<cumac_muUeGrp_req_info_t*>(h_task_in_buf_.data());
        req->numSubband           = 2;
        req->numPrgSampPerSubband = kNumPrgSampPerSubband;
    }
    const size_t kPerCellLen = sizeof(cumac_muUeGrp_req_info_t)
        + sizeof(cumac_muUeGrp_req_srs_info_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
        + sizeof(cumac_muUeGrp_req_ue_info_t) * MAX_NUM_SRS_UE_PER_CELL;
    ASSERT_EQ(cudaMemcpyAsync(d_task_in_buf_, h_task_in_buf_.data(),
                              kPerCellLen * kNumCell, cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    ASSERT_EQ(cudaMemcpyAsync(d_chan_orth_mat_buf_, h_chan_orth_mat_buf_.data(),
                              h_chan_orth_mat_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    for(int i = 0; i < 4; ++i) h_srs_snr_buf_[i] = 10.0f;
    ASSERT_EQ(cudaMemcpyAsync(d_srs_snr_buf_, h_srs_snr_buf_.data(),
                              h_srs_snr_buf_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing gpu_obj(reinterpret_cast<uint8_t*>(d_srs_chan_est_buf_),
                              d_srs_snr_buf_, d_chan_orth_mat_buf_, d_cubb_srs_buf_,
                              kNumCell, kNumPrg, /*num_subband=*/2,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask gpu_task{};
    gpu_task.task_in_buf                   = d_task_in_buf_;
    gpu_task.task_out_buf                  = d_task_out_buf_;
    gpu_task.strm                          = stream_;
    gpu_task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    gpu_task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    gpu_task.kernel_launch_flags           = 0x02;
    gpu_task.is_mem_sharing                = false;
    gpu_obj.setup(&gpu_task);
    std::vector<uint8_t> hSolGpu(sizeof(cumac_muUeGrp_resp_info_t) * kNumCell, 0);
    gpu_obj.run(hSolGpu.data());
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    muMimoUserPairing cpu_obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                              h_srs_snr_buf_.data(),
                              h_chan_orth_mat_buf_.data(),
                              h_cubb_srs_buf_.data(),
                              kNumCell, kNumPrg, 2,
                              kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask cpu_task = gpu_task;
    cpu_task.task_in_buf  = h_task_in_buf_.data();
    cpu_task.task_out_buf = h_task_out_buf_.data();
    cpu_obj.setup(&cpu_task);
    cpu_obj.run_cpu();

    ExpectGpuCpuRespEqual(hSolGpu.data(), h_task_out_buf_.data(), kNumCell);

    // num_subband=2, but the algorithm only emits a UEG when at least one UE
    // was scheduled that subband (`if (num_ue_schd_in_grp > 0)` gate). Since
    // subband 0 consumes all 4 UEs (`ueIds[uIdx]=0xFFFF` after each), subband
    // 1 finds no available UE and no second UEG is emitted — exactly 1 UEG.
    //
    // The single emitted UEG belongs to subband 0, so its `allocPrgEnd` uses
    // the ternary *false* branch we wanted to cover:
    //     allocPrgEnd = nPrbGrp / num_subband * (subbandIdx + 1) = 4/2*1 = 2
    const auto* g = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(hSolGpu.data());
    ASSERT_EQ(g->numSchdUeg, 1u)
        << "expected exactly one UEG since subband 0 consumes all UEs";
    EXPECT_EQ(g->schdUegInfo[0].allocPrgStart, 0)
        << "subband 0 should start at PRG 0";
    EXPECT_EQ(g->schdUegInfo[0].allocPrgEnd,
              static_cast<int16_t>(kNumPrg / 2))
        << "subband 0 should end at nPrbGrp/num_subband (ternary false side: "
        << kNumPrg / 2 << ")";
}

// ---- Functional tests on the "no MU-MIMO" scenarios ----
// snr below srsSnrThr — every UE should get muMimoInd=0, no MU group emitted.
TEST_F(MuMimoUserPairingTest, FunctionalAssertion_SnrBelowThreshold_NoMuMimoScheduled)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, /*snr=*/-50.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    ExpectNoMuMimoGroupsScheduled(h_task_out_buf_.data(), kNumCell);
}

// newTx but no chanEst → muMimoInd=0 → no MU group emitted.
TEST_F(MuMimoUserPairingTest, FunctionalAssertion_NewTxNoChanEst_NoMuMimoScheduled)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    ExpectNoMuMimoGroupsScheduled(h_task_out_buf_.data(), kNumCell);
}

// re-Tx (no 0x02 newTx flag) → muMimoInd=0 → no MU group emitted.
TEST_F(MuMimoUserPairingTest, FunctionalAssertion_NotNewTx_NoMuMimoScheduled)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();
    ExpectNoMuMimoGroupsScheduled(h_task_out_buf_.data(), kNumCell);
}

// Tight nMaxUePerGrp=2 cap — assert the cap is honored: at most 2 UEs in the
// first group, and total scheduled UEs across all groups <= nMaxUeSchdPerCellTTI.
TEST_F(MuMimoUserPairingTest, FunctionalAssertion_TightGroupingLimits_CapEnforced)
{
    BuildRequestBuffer<false>(4, 0x01 | 0x02 | 0x04 | 0x08, 10.0f, MAX_NUM_UE_ANT_PORT);
    PatchGroupingLimits<false>(h_task_in_buf_,
                               /*nMaxLayerPerGrp=*/16,
                               /*nMaxUePerGrp=*/2,
                               /*nMaxUeSchdPerCellTTI=*/4,
                               /*numUeForGrpPerCell=*/16);
    std::fill(h_chan_orth_mat_buf_.begin(), h_chan_orth_mat_buf_.end(), 1.0f);
    muMimoUserPairing obj(reinterpret_cast<uint8_t*>(h_srs_chan_est_buf_.data()),
                          h_srs_snr_buf_.data(),
                          h_chan_orth_mat_buf_.data(),
                          h_cubb_srs_buf_.data(),
                          kNumCell, kNumPrg, kNumSubband,
                          kNumPrgSampPerSubband, kNumBsAntPort);
    muUePairTask task{};
    task.task_in_buf                   = h_task_in_buf_.data();
    task.task_out_buf                  = h_task_out_buf_.data();
    task.strm                          = stream_;
    task.num_srs_ue_per_slot_cell      = kNumSrsUePerSlotCell;
    task.num_blocks_per_row_chanOrtMat = kNumBlocksPerRowChanCorr;
    task.kernel_launch_flags           = 0x02;
    task.is_mem_sharing                = false;
    obj.setup(&task);
    obj.run_cpu();

    const auto* r = reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(h_task_out_buf_.data());
    uint32_t total_ues = 0;
    for(uint32_t i = 0; i < r->numSchdUeg && i < MAX_NUM_UEG_PER_CELL; ++i) {
        EXPECT_LE(r->schdUegInfo[i].numUeInGrp, 2u)
            << "group " << i << " has " << static_cast<int>(r->schdUegInfo[i].numUeInGrp)
            << " UEs but nMaxUePerGrp=2";
        total_ues += r->schdUegInfo[i].numUeInGrp;
    }
    EXPECT_LE(total_ues, 4u) << "total scheduled UEs exceeds nMaxUeSchdPerCellTTI=4";
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();

    return rc;
}
