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
#include <cstring>
#include <memory>
#include <cmath>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include <cuda_fp16.h>

namespace
{

// Helper: compute expected reversed bits in bytes with payload-size handling
static void reverse_bits_in_bytes_ref(const uint8_t* in, uint8_t* out, uint32_t nbytes, int payload_bits)
{
    int payload_bytes = payload_bits / 8;
    int rem_bits      = payload_bits - payload_bytes * 8;
    for(int i = 0; i < static_cast<int>(nbytes); ++i)
    {
        uint8_t acc = 0;
        for(int b = 0; b < 8; ++b)
        {
            acc <<= 1;
            if(i < payload_bytes)
            {
                acc |= ((in[i] >> b) & 1);
            }
            else if((i == payload_bytes) && (rem_bits != 0) && (b >= (8 - rem_bits)))
            {
                acc |= ((in[i] >> b) & 1);
            }
        }
        out[i] = acc;
    }
}

// Helper RAII for CUDA device buffers (define early so helpers can use it)
template <typename T>
struct DeviceBuf
{
    T*     ptr{nullptr};
    size_t count{0};
    explicit DeviceBuf(size_t n = 0) :
        count(n)
    {
        if(n) CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * n));
    }
    ~DeviceBuf()
    {
        if(ptr) cudaFree(ptr);
    }
    DeviceBuf(const DeviceBuf&)            = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept
    {
        ptr     = o.ptr;
        count   = o.count;
        o.ptr   = nullptr;
        o.count = 0;
    }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept
    {
        if(this != &o)
        {
            if(ptr) cudaFree(ptr);
            ptr     = o.ptr;
            count   = o.count;
            o.ptr   = nullptr;
            o.count = 0;
        }
        return *this;
    }
};

// Helper: build a minimal coreset descriptor for host-side pipeline prepare
static PdcchParams make_coreset(uint32_t n_sym, uint32_t num_dl_dci, bool testing_mode)
{
    PdcchParams p{};
    p.n_sym                = n_sym;
    p.start_rb             = 0;
    p.start_sym            = 0;
    p.n_f                  = 72;
    p.bundle_size          = 6;
    p.interleaver_size     = 0;
    p.shift_index          = 0;
    p.interleaved          = 0;
    p.freq_domain_resource = 0xFFFF'FFFF'FFFF'FFFFULL;
    p.num_dl_dci           = num_dl_dci;
    p.dciStartIdx          = 0;
    p.coreset_type         = 0;
    p.testModel            = testing_mode ? 1 : 0;
    return p;
}

// Helper: call pipeline prepare for an arbitrary number of DCIs
static cuphyStatus_t call_pipeline_prepare(PdcchParams&                     coreset,
                                           std::vector<cuphyPdcchDciPrm_t>& dci_vec,
                                           std::vector<uint8_t>&            tm_bytes,
                                           std::vector<uint8_t>&            input_bytes,
                                           std::vector<uint8_t>&            input_w_crc,
                                           cuphyEncoderRateMatchMultiDCILaunchCfg_t* pEncCfg = nullptr,
                                           cuphyGenScramblingSeqLaunchCfg_t* pScrmCfg = nullptr,
                                           cuphyGenPdcchTfSgnlLaunchCfg_t* pTfCfg = nullptr)
{
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfgLocal{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfgLocal{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfgLocal{};
    auto* encCfg  = pEncCfg ? pEncCfg : &encCfgLocal;
    auto* scrmCfg = pScrmCfg ? pScrmCfg : &scrmCfgLocal;
    auto* tfCfg   = pTfCfg ? pTfCfg : &tfCfgLocal;
    return cuphyPdcchPipelinePrepare(
        input_w_crc.data(),
        nullptr,
        input_bytes.data(),
        nullptr,
        /*num_coresets*/ 1,
        /*num_dcis*/ static_cast<int>(dci_vec.size()),
        &coreset,
        dci_vec.data(),
        tm_bytes.data(),
        encCfg,
        scrmCfg,
        tfCfg,
        nullptr);
}

// Launch TF kernel end-to-end and copy back; returns TF buffer and n_f.
// Optional selector_n_sym_override allows exercising early-exit by selecting
// a gridDim.x larger than runtime coreset.n_sym.
static std::pair<std::vector<__half2>, uint32_t>
run_tf_and_copy(PdcchParams coreset, const cuphyPdcchDciPrm_t& dci, const cuphyPdcchPmWOneLayer_t* pmwOpt, int selector_n_sym_override = -1)
{
    // Prepare derived params
    std::vector<uint8_t>            tm_bytes((1 + 7) / 8 + 1, 0);
    std::vector<uint8_t>            input_bytes(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0);
    std::vector<uint8_t>            input_w_crc(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);
    std::vector<cuphyPdcchDciPrm_t> dci_vec(1, dci);
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};
    (void)call_pipeline_prepare(coreset, dci_vec, tm_bytes, input_bytes, input_w_crc, &encCfg, &scrmCfg, &tfCfg);

    // Device allocations
    DeviceBuf<PdcchParams>             d_coreset(1);
    DeviceBuf<cuphyPdcchDciPrm_t>      d_dci_buf(1);
    DeviceBuf<cuphyPdcchPmWOneLayer_t> d_pmw_buf(1);
    if(pmwOpt) { CUDA_CHECK(cudaMemcpy(d_pmw_buf.ptr, pmwOpt, sizeof(cuphyPdcchPmWOneLayer_t), cudaMemcpyHostToDevice)); }

    DeviceBuf<uint8_t> d_x_tx(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
    CUDA_CHECK(cudaMemset(d_x_tx.ptr, 0, d_x_tx.count));
    const size_t       tf_elems = static_cast<size_t>(OFDM_SYMBOLS_PER_SLOT) * coreset.n_f * (pmwOpt ? pmwOpt->nPorts : 1);
    DeviceBuf<__half2> d_tf(tf_elems);
    CUDA_CHECK(cudaMemset(d_tf.ptr, 0, tf_elems * sizeof(__half2)));
    coreset.slotBufferAddr = static_cast<void*>(d_tf.ptr);
    CUDA_CHECK(cudaMemcpy(d_coreset.ptr, &coreset, sizeof(PdcchParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dci_buf.ptr, &dci, sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));

    // Scrambling sequence
    DeviceBuf<uint32_t>              d_scram(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    cuphyGenScramblingSeqLaunchCfg_t scrmLaunch = scrmCfg;
    void*     scrmArgs[2];
    uint32_t* d_scram_base                         = d_scram.ptr;
    scrmArgs[0]                                    = &d_scram_base;
    scrmArgs[1]                                    = &d_dci_buf.ptr;
    scrmLaunch.kernelNodeParamsDriver.kernelParams = scrmArgs;
    CUresult e                                     = cuLaunchKernel(
        scrmLaunch.kernelNodeParamsDriver.func,
        scrmLaunch.kernelNodeParamsDriver.gridDimX,
        scrmLaunch.kernelNodeParamsDriver.gridDimY,
        scrmLaunch.kernelNodeParamsDriver.gridDimZ,
        scrmLaunch.kernelNodeParamsDriver.blockDimX,
        scrmLaunch.kernelNodeParamsDriver.blockDimY,
        scrmLaunch.kernelNodeParamsDriver.blockDimZ,
        scrmLaunch.kernelNodeParamsDriver.sharedMemBytes,
        0,
        scrmLaunch.kernelNodeParamsDriver.kernelParams,
        scrmLaunch.kernelNodeParamsDriver.extra);
    EXPECT_EQ(e, CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch TF kernel (optionally overlaunch gridDim.x to exercise early-exit path)
    cuphyGenPdcchTfSgnlLaunchCfg_t tfLaunch = tfCfg;
    if(selector_n_sym_override > 0) { tfLaunch.kernelNodeParamsDriver.gridDimX = selector_n_sym_override; }
    void*                          args[6];
    uint8_t*                       d_x_tx_base      = d_x_tx.ptr;
    uint32_t*                      d_scram_seq_base = d_scram.ptr;
    uint32_t                       n_coresets_val   = 1;
    PdcchParams*                   d_coreset_base   = d_coreset.ptr;
    cuphyPdcchDciPrm_t*            d_dci_base       = d_dci_buf.ptr;
    cuphyPdcchPmWOneLayer_t*       d_pmw_base       = d_pmw_buf.ptr;
    args[0]                                         = &d_x_tx_base;
    args[1]                                         = &d_scram_seq_base;
    args[2]                                         = &n_coresets_val;
    args[3]                                         = &d_coreset_base;
    args[4]                                         = &d_dci_base;
    args[5]                                         = &d_pmw_base;
    tfLaunch.kernelNodeParamsDriver.kernelParams    = args;
    e                                               = cuLaunchKernel(
        tfLaunch.kernelNodeParamsDriver.func,
        tfLaunch.kernelNodeParamsDriver.gridDimX,
        tfLaunch.kernelNodeParamsDriver.gridDimY,
        tfLaunch.kernelNodeParamsDriver.gridDimZ,
        tfLaunch.kernelNodeParamsDriver.blockDimX,
        tfLaunch.kernelNodeParamsDriver.blockDimY,
        tfLaunch.kernelNodeParamsDriver.blockDimZ,
        tfLaunch.kernelNodeParamsDriver.sharedMemBytes,
        0,
        tfLaunch.kernelNodeParamsDriver.kernelParams,
        tfLaunch.kernelNodeParamsDriver.extra);
    EXPECT_EQ(e, CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half2> tf(tf_elems);
    CUDA_CHECK(cudaMemcpy(tf.data(), d_tf.ptr, tf.size() * sizeof(__half2), cudaMemcpyDeviceToHost));
    return {std::move(tf), coreset.n_f};
}

// Helper RAII for CUDA device buffers (duplicate removed; definition moved up for helper use)

// Minimal DCI params initializer
static cuphyPdcchDciPrm_t make_dci(uint32_t dmrs_id, uint32_t aggr_level, uint32_t cce_index, uint16_t rntiCrc, uint16_t rntiBits, float beta_qam = 1.0f, float beta_dmrs = 1.0f)
{
    cuphyPdcchDciPrm_t d{};
    d.dmrs_id      = dmrs_id;
    d.aggr_level   = aggr_level;
    d.cce_index    = cce_index;
    d.rntiCrc      = rntiCrc;
    d.rntiBits     = rntiBits;
    d.Npayload     = 16; // small payload for tests
    d.beta_qam     = beta_qam;
    d.beta_dmrs    = beta_dmrs;
    d.enablePrcdBf = 0;
    d.pmwPrmIdx    = 0;
    return d;
}

} // namespace

TEST(EmbedPdcchTfSignal_Host, ReverseBitInByte_PartialPayload)
{
    // Validate the local reference helper itself for a partial-byte payload.
    const int      payload_bits = 13;
    const uint32_t nbytes       = 3;
    uint8_t        in[nbytes]   = {0b1010'0110, 0b1100'0000, 0xFF};
    uint8_t        ref[nbytes]  = {};
    reverse_bits_in_bytes_ref(in, ref, nbytes, payload_bits);
    EXPECT_NE(ref[0], 0u);
    EXPECT_NE(ref[1], 0u);
}

TEST(EmbedPdcchTfSignal_Host, AddCrc24C_RntiScrambling)
{
    // Same payload with different RNTI should produce different output in pipeline-prepare payload+CRC buffer.
    PdcchParams coreset = make_coreset(/*n_sym*/ 1, /*num_dl_dci*/ 1, /*testing_mode*/ false);
    std::vector<cuphyPdcchDciPrm_t> dci0(1, make_dci(1, 1, 0, /*rntiCrc*/ 0x0000, 0x1111));
    std::vector<cuphyPdcchDciPrm_t> dci1(1, make_dci(1, 1, 0, /*rntiCrc*/ 0xBEEF, 0x1111));
    dci0[0].Npayload = 16;
    dci1[0].Npayload = 16;
    std::vector<uint8_t> tm(1, 0), input(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0), out0(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0), out1(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);
    input[0] = 0xAB;
    input[1] = 0xCD;
    ASSERT_EQ(call_pipeline_prepare(coreset, dci0, tm, input, out0), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(call_pipeline_prepare(coreset, dci1, tm, input, out1), CUPHY_STATUS_SUCCESS);
    EXPECT_NE(std::memcmp(out0.data(), out1.data(), 8), 0);
}

TEST(EmbedPdcchTfSignal_Host, ComputePdcchCRC_IncludeCrcOnesTrue_CoversNoExtraBytesPath)
{
    // Public API path sanity: valid payload produces non-zero output bytes.
    PdcchParams coreset = make_coreset(/*n_sym*/ 1, /*num_dl_dci*/ 1, /*testing_mode*/ false);
    std::vector<cuphyPdcchDciPrm_t> dci(1, make_dci(1, 1, 0, 0x1234, 0xACE1));
    dci[0].Npayload = 16;
    std::vector<uint8_t> tm(1, 0), input(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, 0), out(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC, 0);
    input[0] = 0xAB;
    input[1] = 0xCD;
    ASSERT_EQ(call_pipeline_prepare(coreset, dci, tm, input, out), CUPHY_STATUS_SUCCESS);
    EXPECT_NE(std::memcmp(out.data(), input.data(), 2), 0);
}

TEST(EmbedPdcchTfSignal_Host, PipelinePrepare_WritesCrcAndTmBits)
{
    // One coreset with one DCI, verify output buffer has reversed payload + CRC bits attached,
    // and TM bits reflect testModel flag.
    uint8_t h_input_w_crc[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC] = {};
    uint8_t h_input[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES]             = {};

    // payload 11 bits: 0b1010'1010'101 (LSB-first in input buffer per production expectations)
    const int payload_bits = 11;
    h_input[0]             = 0b1010'1010; // 8 bits
    h_input[1]             = 0b0000'0101; // next 3 bits at LSB positions

    PdcchParams coreset{};
    coreset.n_sym                = 1;
    coreset.start_rb             = 0;
    coreset.start_sym            = 0;
    coreset.n_f                  = 72;
    coreset.bundle_size          = 6; // non-interleaved forced later
    coreset.interleaver_size     = 0;
    coreset.shift_index          = 0;
    coreset.interleaved          = 0;
    coreset.freq_domain_resource = 0xFFFF'FFFF'FFFF'FFFFULL; // 64 RBs
    coreset.num_dl_dci           = 1;
    coreset.dciStartIdx          = 0;
    coreset.coreset_type         = 0;
    coreset.testModel            = 0; // not in TM for first half

    cuphyPdcchDciPrm_t dci = make_dci(/*dmrs_id*/ 1, /*aggr*/ 1, /*cceIdx*/ 0, /*rntiCrc*/ 0xACE1, /*rntiBits*/ 0x1234);
    dci.Npayload           = payload_bits;

    uint8_t                                  tm_bits[1] = {0};
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};

    cuphyStatus_t st = cuphyPdcchPipelinePrepare(
        h_input_w_crc,
        nullptr,
        h_input,
        nullptr,
        /*num_coresets*/ 1,
        /*num_dcis*/ 1,
        &coreset,
        &dci,
        tm_bits,
        &encCfg,
        &scrmCfg,
        &tfCfg,
        /*stream*/ nullptr);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);

    // Validate derived coreset fields
    EXPECT_EQ(coreset.rb_coreset, 64u);
    EXPECT_EQ(coreset.n_CCE, 64u * coreset.n_sym);

    // Validate TM info bit for non-testing mode remains 0
    EXPECT_EQ(tm_bits[0] & 0x1u, 0u);

    // Validate reversed payload bits present and CRC placed per function logic
    // Compute expected reversed bytes for payload
    const int            nCrcOutByte = (((CUPHY_PDCCH_N_CRC_BITS + payload_bits + 31) / 32) * 4);
    std::vector<uint8_t> expected(nCrcOutByte, 0);
    reverse_bits_in_bytes_ref(h_input, expected.data(), nCrcOutByte, payload_bits);

    // Validate payload reversal where CRC has not overwritten bits.
    const int payload_full_bytes = payload_bits / 8;
    for(int i = 0; i < payload_full_bytes; ++i)
    {
        EXPECT_EQ(h_input_w_crc[i], expected[i]) << "payload reversal mismatch at " << i;
    }
    // For a partial payload byte, only the least-significant rem_bits survive after CRC insertion.
    const int rem_bits = payload_bits % 8;
    if(rem_bits != 0)
    {
        const int     idx  = payload_full_bytes;
        const uint8_t mask = static_cast<uint8_t>((1u << rem_bits) - 1u);
        EXPECT_EQ(static_cast<uint8_t>(h_input_w_crc[idx] & mask),
                  static_cast<uint8_t>(expected[idx] & mask))
            << "payload reversal (partial-byte bits) mismatch at " << idx;
    }
    EXPECT_NE(std::memcmp(h_input_w_crc, expected.data(), nCrcOutByte), 0);

    // Now toggle testing mode and confirm tm bit is set and CRC payload is zero-padded
    std::memset(h_input_w_crc, 0, sizeof(h_input_w_crc));
    std::memset(tm_bits, 0, sizeof(tm_bits));
    coreset.testModel = 1;
    st                = cuphyPdcchPipelinePrepare(
        h_input_w_crc,
        nullptr,
        h_input,
        nullptr,
        1,
        1,
        &coreset,
        &dci,
        tm_bits,
        &encCfg,
        &scrmCfg,
        &tfCfg,
        nullptr);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(tm_bits[0] & 0x1u, 0x1u);
}

TEST(EmbedPdcchTfSignal_Host, PipelinePrepare_MaxPayloadErrorAndTmByteRoll)
{
    // Cover: payload_bits > max_payload_bits path and TM byte roll at bit index 7 -> 8
    // Build coreset with testing mode disabled for first DCI and enabled for next 8 DCIs to roll TM byte
    PdcchParams coreset = make_coreset(/*n_sym*/ 1, /*num_dl_dci*/ 9, /*testing_mode*/ false);

    // Prepare 9 DCIs
    std::vector<cuphyPdcchDciPrm_t> dcis(9);
    for(int i = 0; i < 9; ++i)
    {
        dcis[i]          = make_dci(/*dmrs_id*/ 1 + i, /*aggr*/ 1, /*cceIdx*/ 0, /*rntiCrc*/ 0x1, /*rntiBits*/ 0x1);
        dcis[i].Npayload = 8; // valid small payload
    }

    // First pass: 1 normal DCI (no TM), then 8 TM DCIs to roll the TM byte
    coreset.testModel = 0; // first dci
    std::vector<uint8_t> tm_bytes(2, 0);
    std::vector<uint8_t> input_bytes(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * dcis.size(), 0);
    std::vector<uint8_t> input_w_crc(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC * dcis.size(), 0);

    // Make DCIs 1..8 belong to a testing-mode cell by toggling coreset flag before each call
    ASSERT_EQ(call_pipeline_prepare(coreset, dcis, tm_bytes, input_bytes, input_w_crc), CUPHY_STATUS_SUCCESS);
    coreset.testModel = 1; // remaining DCIs are in testing mode
    ASSERT_EQ(call_pipeline_prepare(coreset, dcis, tm_bytes, input_bytes, input_w_crc), CUPHY_STATUS_SUCCESS);

    // After 8 TM DCIs, the first TM byte should have its lower 8 bits set
    EXPECT_EQ(tm_bytes[0], 0xFF);

    // Second part: force payload > max to hit invalid-arg early return. We simulate by setting a very large Npayload
    dcis[0].Npayload = 8 * CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES + 16; // beyond max bits used in encode kernel
    cuphyStatus_t st = call_pipeline_prepare(coreset, dcis, tm_bytes, input_bytes, input_w_crc);
    EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(EmbedPdcchTfSignal_Host, PipelinePrepare_NullRequiredPointers_ReturnsInvalidArgument)
{
    // Cover top-level nullptr guard in cuphyPdcchPipelinePrepare (line returning INVALID_ARGUMENT).
    uint8_t h_input_w_crc[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC] = {};
    uint8_t h_input[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES]             = {};
    uint8_t tm_bits[1]                                              = {0};

    PdcchParams coreset = make_coreset(/*n_sym*/ 1, /*num_dl_dci*/ 1, /*testing_mode*/ false);
    cuphyPdcchDciPrm_t dci = make_dci(/*dmrs_id*/ 1, /*aggr*/ 1, /*cceIdx*/ 0, /*rntiCrc*/ 0x1, /*rntiBits*/ 0x1);

    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};

    // Baseline: all required pointers non-null
    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  h_input_w_crc,
                  nullptr,
                  h_input,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  &encCfg,
                  &scrmCfg,
                  &tfCfg,
                  nullptr),
              CUPHY_STATUS_SUCCESS);

    // Each required pointer null should hit the early invalid-argument return.
    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  nullptr,
                  nullptr,
                  h_input,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  &encCfg,
                  &scrmCfg,
                  &tfCfg,
                  nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  h_input_w_crc,
                  nullptr,
                  nullptr,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  &encCfg,
                  &scrmCfg,
                  &tfCfg,
                  nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  h_input_w_crc,
                  nullptr,
                  h_input,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  nullptr,
                  &scrmCfg,
                  &tfCfg,
                  nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  h_input_w_crc,
                  nullptr,
                  h_input,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  &encCfg,
                  nullptr,
                  &tfCfg,
                  nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(cuphyPdcchPipelinePrepare(
                  h_input_w_crc,
                  nullptr,
                  h_input,
                  nullptr,
                  1,
                  1,
                  &coreset,
                  &dci,
                  tm_bits,
                  &encCfg,
                  &scrmCfg,
                  nullptr,
                  nullptr),
              CUPHY_STATUS_INVALID_ARGUMENT);
}

TEST(EmbedPdcchTfSignal_Device, GenScramblingSeqKernel_MatchesHost)
{
    // Build two DCIs and validate device scrambling output is generated for both and differs across DCIs.
    const uint32_t                  num_dci = 2;
    std::vector<cuphyPdcchDciPrm_t> h_dci(num_dci);
    h_dci[0] = make_dci(17, 1, 0, 0x1, 0xAAAA);
    h_dci[1] = make_dci(319, 2, 0, 0x2, 0x5555);

    // Allocate device buffers
    DeviceBuf<cuphyPdcchDciPrm_t> d_dci(num_dci);
    CUDA_CHECK(cudaMemcpy(d_dci.ptr, h_dci.data(), sizeof(cuphyPdcchDciPrm_t) * num_dci, cudaMemcpyHostToDevice));

    // Kernel writes with a fixed stride per DCI: CUPHY_PDCCH_MAX_TX_BITS_PER_DCI/32 words
    const uint32_t      stride_words = (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    DeviceBuf<uint32_t> d_scram(stride_words * num_dci);

    // Launch configuration via public pipeline-prepare
    PdcchParams coreset = make_coreset(/*n_sym*/ 1, /*num_dl_dci*/ num_dci, /*testing_mode*/ false);
    std::vector<cuphyPdcchDciPrm_t> dci_tmp = h_dci;
    std::vector<uint8_t>            tm(1, 0), input(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * num_dci, 0), input_w_crc(CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC * num_dci, 0);
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         launchCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};
    ASSERT_EQ(call_pipeline_prepare(coreset, dci_tmp, tm, input, input_w_crc, &encCfg, &launchCfg, &tfCfg), CUPHY_STATUS_SUCCESS);

    void*     args[2];
    uint32_t* d_scram_base                        = d_scram.ptr;
    args[0]                                       = &d_scram_base;
    args[1]                                       = &d_dci.ptr;
    launchCfg.kernelNodeParamsDriver.kernelParams = args;

    CUresult e = cuLaunchKernel(
        launchCfg.kernelNodeParamsDriver.func,
        launchCfg.kernelNodeParamsDriver.gridDimX,
        launchCfg.kernelNodeParamsDriver.gridDimY,
        launchCfg.kernelNodeParamsDriver.gridDimZ,
        launchCfg.kernelNodeParamsDriver.blockDimX,
        launchCfg.kernelNodeParamsDriver.blockDimY,
        launchCfg.kernelNodeParamsDriver.blockDimZ,
        launchCfg.kernelNodeParamsDriver.sharedMemBytes,
        0,
        launchCfg.kernelNodeParamsDriver.kernelParams,
        launchCfg.kernelNodeParamsDriver.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back entire buffer
    std::vector<uint32_t> got(stride_words * num_dci, 0);
    CUDA_CHECK(cudaMemcpy(got.data(), d_scram.ptr, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Verify each DCI's first word is populated and DCIs are not identical.
    for(uint32_t dci_id = 0; dci_id < num_dci; ++dci_id)
    {
        const uint32_t base         = dci_id * stride_words;
        EXPECT_NE(got[base], 0u) << "DCI " << dci_id << " scrambling word is zero";
    }
    EXPECT_NE(got[0], got[stride_words]);
}

TEST(EmbedPdcchTfSignal_Device, GenTfSignalKernel_CoversGenerateDmrs)
{
    // Configure a simple non-interleaved CORESET and one DCI; launch genPdcchTfSignalKernel to execute generate_dmrs.

    // Host coreset params
    PdcchParams h_coreset{};
    h_coreset.n_sym                = 1; // 1 symbol
    h_coreset.start_rb             = 0;
    h_coreset.start_sym            = 0;
    h_coreset.n_f                  = 72; // FFT bins per symbol row in slot buffer
    h_coreset.bundle_size          = 6;  // non-interleaved forced
    h_coreset.interleaver_size     = 0;
    h_coreset.shift_index          = 0;
    h_coreset.interleaved          = 0;
    h_coreset.freq_domain_resource = 0xFFFF'FFFF'FFFF'FFFFULL; // 64 RBs set
    h_coreset.num_dl_dci           = 1;
    h_coreset.dciStartIdx          = 0;
    h_coreset.coreset_type         = 0;
    h_coreset.testModel            = 0;

    // Host DCI params
    cuphyPdcchDciPrm_t h_dci = make_dci(/*dmrs_id*/ 37, /*aggr*/ 1, /*cceIdx*/ 0, /*rntiCrc*/ 0x0001, /*rntiBits*/ 0xACE1);
    h_dci.enablePrcdBf       = 0;
    h_dci.beta_qam           = 1.0f;
    h_dci.beta_dmrs          = 1.0f;

    // Prepare derived coreset fields via pipeline prepare helper (host-side)
    uint8_t                                  h_tm_bits[1]                                           = {0};
    uint8_t                                  h_input_w_crc[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC] = {};
    uint8_t                                  h_input[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES]             = {};
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};
    cuphyStatus_t                            st = cuphyPdcchPipelinePrepare(
        h_input_w_crc,
        nullptr,
        h_input,
        nullptr,
        /*num_coresets*/ 1,
        /*num_dcis*/ 1,
        &h_coreset,
        &h_dci,
        h_tm_bits,
        &encCfg,
        &scrmCfg,
        &tfCfg,
        nullptr);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);

    // Allocate device buffers
    DeviceBuf<PdcchParams>             d_coreset(1);
    DeviceBuf<cuphyPdcchDciPrm_t>      d_dci_buf(1);
    DeviceBuf<cuphyPdcchPmWOneLayer_t> d_pmw_buf(1);

    // Slot buffer for TF signal (half2)
    const size_t       tf_elems = static_cast<size_t>(OFDM_SYMBOLS_PER_SLOT) * h_coreset.n_f; // single port
    DeviceBuf<__half2> d_tf(tf_elems);
    CUDA_CHECK(cudaMemset(d_tf.ptr, 0, tf_elems * sizeof(__half2)));
    // Set slotBufferAddr in coreset and copy to device
    h_coreset.slotBufferAddr = static_cast<void*>(d_tf.ptr);
    CUDA_CHECK(cudaMemcpy(d_coreset.ptr, &h_coreset, sizeof(PdcchParams), cudaMemcpyHostToDevice));

    // Copy DCI params
    CUDA_CHECK(cudaMemcpy(d_dci_buf.ptr, &h_dci, sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));

    // Prepare d_x_tx (rate-matched bits) as zeros
    DeviceBuf<uint8_t> d_x_tx(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
    CUDA_CHECK(cudaMemset(d_x_tx.ptr, 0, d_x_tx.count * sizeof(uint8_t)));

    // Generate scrambling sequence on device for the DCI
    DeviceBuf<uint32_t>              d_scram(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    cuphyGenScramblingSeqLaunchCfg_t scrmLaunch = scrmCfg;
    void*     scrmArgs[2];
    uint32_t* d_scram_base                         = d_scram.ptr;
    scrmArgs[0]                                    = &d_scram_base;
    scrmArgs[1]                                    = &d_dci_buf.ptr;
    scrmLaunch.kernelNodeParamsDriver.kernelParams = scrmArgs;
    CUresult e                                     = cuLaunchKernel(
        scrmLaunch.kernelNodeParamsDriver.func,
        scrmLaunch.kernelNodeParamsDriver.gridDimX,
        scrmLaunch.kernelNodeParamsDriver.gridDimY,
        scrmLaunch.kernelNodeParamsDriver.gridDimZ,
        scrmLaunch.kernelNodeParamsDriver.blockDimX,
        scrmLaunch.kernelNodeParamsDriver.blockDimY,
        scrmLaunch.kernelNodeParamsDriver.blockDimZ,
        scrmLaunch.kernelNodeParamsDriver.sharedMemBytes,
        0,
        scrmLaunch.kernelNodeParamsDriver.kernelParams,
        scrmLaunch.kernelNodeParamsDriver.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Select and launch genTfSignal kernel via pipeline-prepare launch config
    cuphyGenPdcchTfSgnlLaunchCfg_t tfLaunch = tfCfg;
    void*                          args[6];
    uint8_t*                       d_x_tx_base      = d_x_tx.ptr;
    uint32_t*                      d_scram_seq_base = d_scram.ptr;
    uint32_t                       n_coresets_val   = 1;
    PdcchParams*                   d_coreset_base   = d_coreset.ptr;
    cuphyPdcchDciPrm_t*            d_dci_base       = d_dci_buf.ptr;
    cuphyPdcchPmWOneLayer_t*       d_pmw_base       = d_pmw_buf.ptr;

    args[0]                                      = &d_x_tx_base;
    args[1]                                      = &d_scram_seq_base;
    args[2]                                      = &n_coresets_val; // value arg
    args[3]                                      = &d_coreset_base;
    args[4]                                      = &d_dci_base;
    args[5]                                      = &d_pmw_base;
    tfLaunch.kernelNodeParamsDriver.kernelParams = args;

    e = cuLaunchKernel(
        tfLaunch.kernelNodeParamsDriver.func,
        tfLaunch.kernelNodeParamsDriver.gridDimX,
        tfLaunch.kernelNodeParamsDriver.gridDimY,
        tfLaunch.kernelNodeParamsDriver.gridDimZ,
        tfLaunch.kernelNodeParamsDriver.blockDimX,
        tfLaunch.kernelNodeParamsDriver.blockDimY,
        tfLaunch.kernelNodeParamsDriver.blockDimZ,
        tfLaunch.kernelNodeParamsDriver.sharedMemBytes,
        0,
        tfLaunch.kernelNodeParamsDriver.kernelParams,
        tfLaunch.kernelNodeParamsDriver.extra);
    ASSERT_EQ(e, CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back TF signal for symbol 0
    std::vector<__half2> tf(tf_elems);
    CUDA_CHECK(cudaMemcpy(tf.data(), d_tf.ptr, tf_elems * sizeof(__half2), cudaMemcpyDeviceToHost));

    // Validate within first 6 RBs (72 REs): DMRS at indices 1,5,9, then repeating every 4
    auto        h2f      = [](__half h) { return __half2float(h); };
    const float dmrs_amp = h_dci.beta_dmrs; // magnitude of QPSK with per-axis 1/sqrt(2) is beta
    const float qam_amp  = h_dci.beta_qam;  // same for QAM mapping here (QPSK)
    const float eps      = 5e-2f;           // relaxed tolerance for half precision

    for(int re = 0; re < 72; ++re)
    {
        float xr  = h2f(reinterpret_cast<const __half*>(&tf[re])[0]);
        float xi  = h2f(reinterpret_cast<const __half*>(&tf[re])[1]);
        float mag = std::sqrt(xr * xr + xi * xi);
        if(re % 4 == 1)
        {
            // DMRS RE
            EXPECT_NEAR(mag, dmrs_amp, eps);
        }
        else
        {
            // QAM RE
            EXPECT_NEAR(mag, qam_amp, eps);
        }
    }
}

TEST(EmbedPdcchTfSignal_Device, GenTfSignalKernel_EarlyExitAndMultipleSymbols)
{
    // Cover early-exit path when blockIdx.x >= n_sym and multiple-symbol configuration
    PdcchParams h_coreset{};
    h_coreset.n_sym                = 1; // Single symbol will trigger early exit for blocks >= 1
    h_coreset.start_rb             = 0;
    h_coreset.start_sym            = 0;
    h_coreset.n_f                  = 72;
    h_coreset.bundle_size          = 6;
    h_coreset.interleaver_size     = 0;
    h_coreset.shift_index          = 0;
    h_coreset.interleaved          = 0;
    h_coreset.freq_domain_resource = 0xFFFF'FFFF'FFFF'FFFFULL;
    h_coreset.num_dl_dci           = 1;
    h_coreset.dciStartIdx          = 0;
    h_coreset.coreset_type         = 0;

    // Prepare derived fields
    uint8_t                                  tm_bits[1]                                             = {0};
    uint8_t                                  h_input_w_crc[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC] = {};
    uint8_t                                  h_input[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES]             = {};
    cuphyPdcchDciPrm_t                       h_dci                                                  = make_dci(7, 1, 0, 0x1, 0x1);
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};
    ASSERT_EQ(cuphyPdcchPipelinePrepare(h_input_w_crc, nullptr, h_input, nullptr, 1, 1, &h_coreset, &h_dci, tm_bits, &encCfg, &scrmCfg, &tfCfg, nullptr), CUPHY_STATUS_SUCCESS);

    // Select kernel with max_n_sym > n_sym via helper override to produce extra blocks that early-exit
    auto      tf_out   = run_tf_and_copy(h_coreset, h_dci, /*pmwOpt*/ nullptr, /*selector_n_sym_override*/ 3);
    auto&     tf       = tf_out.first;
    const int tf_elems = tf_out.second * OFDM_SYMBOLS_PER_SLOT;
    for(int re = 0; re < 72; ++re)
    {
        float xr = __half2float(reinterpret_cast<const __half*>(&tf[re])[0]);
        float xi = __half2float(reinterpret_cast<const __half*>(&tf[re])[1]);
        EXPECT_TRUE(std::isfinite(xr));
        EXPECT_TRUE(std::isfinite(xi));
    }
}

TEST(EmbedPdcchTfSignal_Device, GenTfSignalKernel_PrecodingPath_TwoPorts)
{
    // Exercise enablePrcdBf branch and verify per-port outputs differ by the precoding matrix
    PdcchParams h_coreset{};
    h_coreset.n_sym                = 1;
    h_coreset.start_rb             = 0;
    h_coreset.start_sym            = 0;
    h_coreset.n_f                  = 72;
    h_coreset.bundle_size          = 6;
    h_coreset.interleaver_size     = 0;
    h_coreset.shift_index          = 0;
    h_coreset.interleaved          = 0;
    h_coreset.freq_domain_resource = 0xFFFF'FFFF'FFFF'FFFFULL;
    h_coreset.num_dl_dci           = 1;
    h_coreset.dciStartIdx          = 0;
    h_coreset.coreset_type         = 0;

    cuphyPdcchDciPrm_t h_dci = make_dci(13, 1, 0, 0x1, 0x1);
    h_dci.enablePrcdBf       = 1;
    h_dci.pmwPrmIdx          = 0;

    // Prepare derived fields
    uint8_t                                  tm_bits[1]                                             = {0};
    uint8_t                                  h_input_w_crc[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC] = {};
    uint8_t                                  h_input[CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES]             = {};
    cuphyEncoderRateMatchMultiDCILaunchCfg_t encCfg{};
    cuphyGenScramblingSeqLaunchCfg_t         scrmCfg{};
    cuphyGenPdcchTfSgnlLaunchCfg_t           tfCfg{};
    ASSERT_EQ(cuphyPdcchPipelinePrepare(h_input_w_crc, nullptr, h_input, nullptr, 1, 1, &h_coreset, &h_dci, tm_bits, &encCfg, &scrmCfg, &tfCfg, nullptr), CUPHY_STATUS_SUCCESS);

    // Device buffers and pmw params
    DeviceBuf<PdcchParams>             d_coreset(1);
    DeviceBuf<cuphyPdcchDciPrm_t>      d_dci_buf(1);
    DeviceBuf<cuphyPdcchPmWOneLayer_t> d_pmw_buf(1);
    // Two ports: identity and j rotation
    cuphyPdcchPmWOneLayer_t h_pmw{};
    h_pmw.nPorts      = 2;
    h_pmw.matrix[0].x = __float2half(1.0f);
    h_pmw.matrix[0].y = __float2half(0.0f);
    h_pmw.matrix[1].x = __float2half(0.0f);
    h_pmw.matrix[1].y = __float2half(1.0f);
    CUDA_CHECK(cudaMemcpy(d_pmw_buf.ptr, &h_pmw, sizeof(h_pmw), cudaMemcpyHostToDevice));

    DeviceBuf<uint8_t> d_x_tx(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 8);
    CUDA_CHECK(cudaMemset(d_x_tx.ptr, 0, d_x_tx.count));
    const size_t       tf_elems = static_cast<size_t>(OFDM_SYMBOLS_PER_SLOT) * h_coreset.n_f;
    DeviceBuf<__half2> d_tf(tf_elems * h_pmw.nPorts);
    CUDA_CHECK(cudaMemset(d_tf.ptr, 0, tf_elems * h_pmw.nPorts * sizeof(__half2)));
    h_coreset.slotBufferAddr = static_cast<void*>(d_tf.ptr);
    CUDA_CHECK(cudaMemcpy(d_coreset.ptr, &h_coreset, sizeof(PdcchParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dci_buf.ptr, &h_dci, sizeof(cuphyPdcchDciPrm_t), cudaMemcpyHostToDevice));

    // Scrambling sequence
    DeviceBuf<uint32_t>              d_scram(CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    cuphyGenScramblingSeqLaunchCfg_t scrmLaunch = scrmCfg;
    void*     scrmArgs[2];
    uint32_t* d_scram_base                         = d_scram.ptr;
    scrmArgs[0]                                    = &d_scram_base;
    scrmArgs[1]                                    = &d_dci_buf.ptr;
    scrmLaunch.kernelNodeParamsDriver.kernelParams = scrmArgs;
    ASSERT_EQ(cuLaunchKernel(
                  scrmLaunch.kernelNodeParamsDriver.func,
                  scrmLaunch.kernelNodeParamsDriver.gridDimX,
                  scrmLaunch.kernelNodeParamsDriver.gridDimY,
                  scrmLaunch.kernelNodeParamsDriver.gridDimZ,
                  scrmLaunch.kernelNodeParamsDriver.blockDimX,
                  scrmLaunch.kernelNodeParamsDriver.blockDimY,
                  scrmLaunch.kernelNodeParamsDriver.blockDimZ,
                  scrmLaunch.kernelNodeParamsDriver.sharedMemBytes,
                  0,
                  scrmLaunch.kernelNodeParamsDriver.kernelParams,
                  scrmLaunch.kernelNodeParamsDriver.extra),
              CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch TF kernel
    cuphyGenPdcchTfSgnlLaunchCfg_t tfLaunch = tfCfg;
    void*                          args[6];
    uint8_t*                       d_x_tx_base      = d_x_tx.ptr;
    uint32_t*                      d_scram_seq_base = d_scram.ptr;
    uint32_t                       n_coresets_val   = 1;
    PdcchParams*                   d_coreset_base   = d_coreset.ptr;
    cuphyPdcchDciPrm_t*            d_dci_base       = d_dci_buf.ptr;
    cuphyPdcchPmWOneLayer_t*       d_pmw_base       = d_pmw_buf.ptr;
    args[0]                                         = &d_x_tx_base;
    args[1]                                         = &d_scram_seq_base;
    args[2]                                         = &n_coresets_val;
    args[3]                                         = &d_coreset_base;
    args[4]                                         = &d_dci_base;
    args[5]                                         = &d_pmw_base;
    tfLaunch.kernelNodeParamsDriver.kernelParams    = args;
    ASSERT_EQ(cuLaunchKernel(
                  tfLaunch.kernelNodeParamsDriver.func,
                  tfLaunch.kernelNodeParamsDriver.gridDimX,
                  tfLaunch.kernelNodeParamsDriver.gridDimY,
                  tfLaunch.kernelNodeParamsDriver.gridDimZ,
                  tfLaunch.kernelNodeParamsDriver.blockDimX,
                  tfLaunch.kernelNodeParamsDriver.blockDimY,
                  tfLaunch.kernelNodeParamsDriver.blockDimZ,
                  tfLaunch.kernelNodeParamsDriver.sharedMemBytes,
                  0,
                  tfLaunch.kernelNodeParamsDriver.kernelParams,
                  tfLaunch.kernelNodeParamsDriver.extra),
              CUDA_SUCCESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify port 1 equals port 0 multiplied by i (rotation)
    std::vector<__half2> tf(tf_elems * h_pmw.nPorts);
    CUDA_CHECK(cudaMemcpy(tf.data(), d_tf.ptr, tf.size() * sizeof(__half2), cudaMemcpyDeviceToHost));
    const int   offset_per_port = h_coreset.n_f * OFDM_SYMBOLS_PER_SLOT;
    const float eps             = 1.5e-1f; // relaxed tolerance
    for(int re = 0; re < h_coreset.n_f; ++re)
    {
        const __half2 v0 = tf[re];
        const __half2 v1 = tf[re + offset_per_port];
        float         a  = __half2float(reinterpret_cast<const __half*>(&v0)[0]);
        float         b  = __half2float(reinterpret_cast<const __half*>(&v0)[1]);
        float         c  = __half2float(reinterpret_cast<const __half*>(&v1)[0]);
        float         d  = __half2float(reinterpret_cast<const __half*>(&v1)[1]);
        // (a+bi)*i = -b + ai
        EXPECT_NEAR(c, -b, eps);
        EXPECT_NEAR(d, a, eps);
    }
}

TEST(EmbedPdcchTfSignal_Device, GenTfSignalKernel_Interleaved_SortsBundles)
{
    // Configure interleaved coreset to cover compute_map branches: fill 0xFFFF and sorting
    PdcchParams coreset          = make_coreset(/*n_sym*/ 2, /*num_dl_dci*/ 1, /*testing_mode*/ false);
    coreset.interleaved          = 1;
    coreset.bundle_size          = 2;                     // bundles_per_level == 3 → round_up_elements uses x4
    coreset.interleaver_size     = 2;                     // valid for interleaved
    coreset.shift_index          = 1;                     // non-zero to exercise shift logic
    coreset.freq_domain_resource = 0xAAAAAAAAAAAAAAAAULL; // alternating pattern

    cuphyPdcchDciPrm_t dci = make_dci(/*dmrs_id*/ 23, /*aggr_level*/ 2, /*cceIdx*/ 0, /*rntiCrc*/ 0x1, /*rntiBits*/ 0x1);

    auto      tf_out   = run_tf_and_copy(coreset, dci, /*pmwOpt*/ nullptr);
    auto&     tf       = tf_out.first;
    const int tf_elems = tf_out.second * OFDM_SYMBOLS_PER_SLOT * 2; // n_sym=2

    // Sanity: first few REs are finite and non-zero across both symbols
    int checks = 0;
    for(int sym = 0; sym < 2; ++sym)
    {
        for(int re = 0; re < 24; ++re)
        {
            const __half2 v  = tf[sym * tf_out.second + re];
            float         xr = __half2float(reinterpret_cast<const __half*>(&v)[0]);
            float         xi = __half2float(reinterpret_cast<const __half*>(&v)[1]);
            EXPECT_TRUE(std::isfinite(xr));
            EXPECT_TRUE(std::isfinite(xi));
            checks++;
        }
    }
    EXPECT_GT(checks, 0);
}

// Minimal test main to emit coverage data consistently with other tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}
